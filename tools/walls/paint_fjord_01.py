#!/usr/bin/env python3
"""
Fjord 01 ("Blue-Hour Fjord Boathouse") — Pure-Python 3D-Grounded Scene Pipeline (2048 x 1152)

Five-stage deterministic hand-painted scene (zero image models, zero external assets):
  Stage 1: Metric right-handed Camera3D (off-axis vanishing point ~62% width),
           exact 3x3 UV-to-screen plane homographies masked by valid UV bounds and
           ray-plane depth z_cam > 0.05 with exponential arctic-mist distance fog,
           near-plane 3D line clipping, and Voronoi (d2 - d1) weathered Nordic
           basalt quay masonry crevices.
  Stage 2: Single normalized 3D blue-hour twilight vector SUN_DIR (z < 0) traced
           backward across 6 height-refinement steps onto the pitched boathouse roof
           and timber bollards for exact cast shadows, combined with 3D water-plane
           mirrored architectural quads and Fresnel capillary-rippled fjord reflections.
  Stage 3: 4 depth-interleaved 2x supersampled PyCairo layers (Lanczos-downsampled)
           featuring ridged-multifractal Norwegian basalt cliffs with organic alpine
           snowfields, weathered dark-timber boathouse trim & jetty pilings, a moored
           hand-crafted Nordic clinker wooden skiff with correct solid outer hull &
           catenary mooring rope, foreground bollards & subtle blue-hour snowflakes,
           textured with isotropic fbm_gouache mottling, 80-100 flat-brush dabs,
           cold-press paper tooth, and shadow bounce light.
  Stage 4: Foreground upper-right Nordic pine branch silhouette convolved in
           premultiplied RGBA with a circular aperture disc bokeh kernel (r = 7.8)
           plus 1 px opposite R/B lateral shift, finished with 35mm multi-scale
           halation, radial CA, optical vignette, shadow black lift + C1 tanh
           highlight shoulder, luminance-weighted silver-halide grain, and faint
           analog YIQ transfer patina.
  Stage 5: Deterministic CPU render (<60s) exporting full 2048x1152 WebP assets
           plus full PNG and three 1:1 (600x450) macro audit crops.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import time
from typing import Iterable

import cairo
import numpy as np
from PIL import Image

W, H = 2048, 1152
Z_NEAR = 0.05
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Stage 2: Single normalized 3D light travel vector from open right fjord twilight (z < 0)
_raw_sun = np.array([-0.68, -0.34, -0.648], dtype=np.float64)
SUN_DIR = _raw_sun / np.linalg.norm(_raw_sun)

# World metric elevations (meters)
Z_WATER = 0.0
Z_DECK = 0.88
Z_QUAY = 1.02

# Boathouse footprint & pitched roof dimensions (meters)
BH_X_LEFT = -5.65
BH_X_RIGHT = -1.92
BH_X_RIDGE = -3.78
BH_Y0 = 6.95
BH_Y1 = 14.15
BH_Z_EAVE = 3.45
BH_Z_RIDGE = 5.48
BH_EAVE_OVERHANG = 0.42

# Warm boathouse window location on east (X = BH_X_RIGHT) timber wall
WIN_Y0, WIN_Y1 = 8.65, 10.15
WIN_Z0, WIN_Z1 = 1.72, 2.94


# =============================================================================
# Core Vectorized Primitives (NumPy + Pillow + PyCairo)
# =============================================================================

def reflect101(idx: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return np.zeros_like(idx)
    period = 2 * (size - 1)
    m = np.mod(idx, period)
    return np.where(m >= size, period - m, m)


def gaussian_blur_1d(arr: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    if sigma <= 0.05:
        return arr.copy()
    radius = max(1, int(math.ceil(sigma * 3.0)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    n = arr.shape[axis]
    pad_idx = reflect101(np.arange(-radius, n + radius), n)
    if axis == 1:
        padded = arr[:, pad_idx]
        out = np.zeros_like(arr, dtype=np.float32)
        for i, w_val in enumerate(k):
            out += w_val * padded[:, i : i + n]
        return out
    padded = arr[pad_idx, :]
    out = np.zeros_like(arr, dtype=np.float32)
    for i, w_val in enumerate(k):
        out += w_val * padded[i : i + n, :]
    return out


def gaussian_blur_2d(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.1:
        return arr.copy()
    if sigma <= 9.0:
        return gaussian_blur_1d(gaussian_blur_1d(arr, sigma, axis=1), sigma, axis=0)
    h, w = arr.shape[:2]
    factor = 4 if sigma < 40 else 8
    sw, sh = max(8, w // factor), max(8, h // factor)
    scaled_sigma = sigma / factor
    if arr.ndim == 2:
        pil = Image.fromarray(arr.astype(np.float32), mode="F").resize((sw, sh), Image.Resampling.BILINEAR)
        small = np.asarray(pil, dtype=np.float32)
        small = gaussian_blur_1d(gaussian_blur_1d(small, scaled_sigma, axis=1), scaled_sigma, axis=0)
        return np.asarray(Image.fromarray(small, mode="F").resize((w, h), Image.Resampling.BICUBIC), dtype=np.float32)
    out = np.empty_like(arr, dtype=np.float32)
    for c in range(arr.shape[2]):
        pil = Image.fromarray(arr[:, :, c].astype(np.float32), mode="F").resize((sw, sh), Image.Resampling.BILINEAR)
        small = np.asarray(pil, dtype=np.float32)
        small = gaussian_blur_1d(gaussian_blur_1d(small, scaled_sigma, axis=1), scaled_sigma, axis=0)
        out[:, :, c] = np.asarray(Image.fromarray(small, mode="F").resize((w, h), Image.Resampling.BICUBIC), dtype=np.float32)
    return out


def resize_bicubic(arr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    if arr.ndim == 2:
        return np.asarray(Image.fromarray(arr.astype(np.float32), mode="F").resize((out_w, out_h), Image.Resampling.BICUBIC), dtype=np.float32)
    out = np.empty((out_h, out_w, arr.shape[2]), dtype=np.float32)
    for c in range(arr.shape[2]):
        out[:, :, c] = np.asarray(Image.fromarray(arr[:, :, c].astype(np.float32), mode="F").resize((out_w, out_h), Image.Resampling.BICUBIC), dtype=np.float32)
    return out


def remap_bilinear(src: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    sh, sw = src.shape[:2]
    x0 = np.floor(map_x).astype(np.int32)
    y0 = np.floor(map_y).astype(np.int32)
    fx = (map_x - x0).astype(np.float32)
    fy = (map_y - y0).astype(np.float32)
    x1 = reflect101(x0 + 1, sw)
    y1 = reflect101(y0 + 1, sh)
    x0 = reflect101(x0, sw)
    y0 = reflect101(y0, sh)
    if src.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
    return (src[y0, x0] * (1.0 - fx) + src[y0, x1] * fx) * (1.0 - fy) + (src[y1, x0] * (1.0 - fx) + src[y1, x1] * fx) * fy


def fbm_gouache(h: int, w: int, rng: np.random.Generator, scales=((72, 68), (26, 29), (10.5, 9.5), (4.5, 4.8))) -> np.ndarray:
    acc = np.zeros((h, w), dtype=np.float32)
    weights = [0.46, 0.28, 0.17, 0.09]
    for (sy, sx), wt in zip(scales, weights):
        gh = max(4, int(round(h / sy)))
        gw = max(4, int(round(w / sx)))
        acc += wt * resize_bicubic(rng.standard_normal((gh, gw)).astype(np.float32), w, h)
    acc -= acc.mean()
    return acc / (float(acc.std()) + 1e-6)


def paper_tooth(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    coarse = resize_bicubic(rng.standard_normal((max(4, h // 3), max(4, w // 3))).astype(np.float32), w, h)
    fine = rng.standard_normal((h, w)).astype(np.float32)
    t = 0.55 * coarse + 0.45 * fine
    t -= t.mean()
    return t / (float(t.std()) + 1e-6)


def flat_brush_dabs(h: int, w: int, rng: np.random.Generator, count: int = 92) -> np.ndarray:
    qh, qw = max(16, h // 4), max(16, w // 4)
    arr_sum = np.zeros((qh, qw), dtype=np.float32)
    s_one = cairo.ImageSurface(cairo.FORMAT_A8, qw, qh)
    c1 = cairo.Context(s_one)
    stride = s_one.get_stride()
    for _ in range(count):
        c1.save()
        c1.set_operator(cairo.OPERATOR_CLEAR)
        c1.paint()
        c1.restore()
        cx = float(rng.uniform(0, qw))
        cy = float(rng.uniform(0, qh))
        ang = float(rng.uniform(0, math.pi))
        rx = float(rng.uniform(qw * 0.035, qw * 0.105))
        ry = float(rng.uniform(rx * 0.28, rx * 0.62))
        n_pts = int(rng.integers(5, 8))
        strength = float(rng.uniform(0.18, 0.44)) * (1.0 if rng.random() > 0.5 else -1.0)
        c1.save()
        c1.translate(cx, cy)
        c1.rotate(ang)
        pts = []
        for k in range(n_pts):
            a = (2.0 * math.pi * k) / n_pts + float(rng.uniform(-0.22, 0.22))
            rj = float(rng.uniform(0.78, 1.16))
            pts.append((math.cos(a) * rx * rj, math.sin(a) * ry * rj))
        c1.move_to(*pts[0])
        for p in pts[1:]:
            c1.line_to(*p)
        c1.close_path()
        c1.set_source_rgba(1, 1, 1, 1)
        c1.fill()
        c1.restore()
        s_one.flush()
        buf = np.frombuffer(s_one.get_data(), dtype=np.uint8).reshape(qh, stride)[:, :qw].astype(np.float32) / 255.0
        arr_sum += strength * buf
    arr_sum = gaussian_blur_2d(arr_sum, 3.0)
    arr_sum -= arr_sum.mean()
    arr_sum /= float(arr_sum.std()) + 1e-6
    return resize_bicubic(arr_sum, w, h)


def over_premul(dst_rgb: np.ndarray, src_premul_rgba: np.ndarray) -> np.ndarray:
    a = np.clip(src_premul_rgba[:, :, 3:4], 0.0, 1.0)
    return src_premul_rgba[:, :, :3] + dst_rgb * (1.0 - a)


def apply_shadow_bounce(rgb: np.ndarray, strength: float = 1.0) -> np.ndarray:
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    w_sh = np.clip((0.28 - lum) / 0.28, 0.0, 1.0)[..., None]
    return rgb + w_sh * (np.array([0.014, 0.012, 0.008], dtype=np.float32) * strength)


def ridged_profile_1d(n: int, rng: np.random.Generator, octaves=((5, 0.52), (11, 0.26), (23, 0.13), (49, 0.06), (105, 0.03))) -> np.ndarray:
    """Crisp jagged glaciated alpine ridge profile in [0, 1] using multi-octave ridged folding."""
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    acc = np.zeros(n, dtype=np.float64)
    for freq, amp in octaves:
        grid = rng.uniform(-1.0, 1.0, size=freq + 2)
        gx = np.linspace(0.0, 1.0, freq + 2)
        val = np.interp(x, gx, grid)
        ridge = 1.0 - np.abs(val)
        acc += amp * np.power(ridge, 1.55)
    acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-8)
    return acc.astype(np.float32)


def composite_painterly_mountain_layer(
    canvas: np.ndarray,
    ridge_y: np.ndarray,
    hy: float,
    rock_dark: tuple[float, float, float],
    rock_lit: tuple[float, float, float],
    snow_col: tuple[float, float, float],
    snow_bias: float,
    mist_col: tuple[float, float, float],
    mist_strength: float,
    mist_height: float,
    rng: np.random.Generator,
    mottle: np.ndarray,
    dabs: np.ndarray,
    tooth: np.ndarray,
) -> np.ndarray:
    """Paint craggy gouache-shaded Norwegian basalt fjord cliffs with analytical normals & organic snowfields."""
    y_min = max(0, int(math.floor(float(ridge_y.min()))) - 4)
    y_max = min(H, int(math.ceil(hy)) + 6)
    if y_min >= y_max:
        return canvas
    bh = y_max - y_min
    yy = np.arange(y_min, y_max, dtype=np.float32)[:, None]
    ry = ridge_y[None, :]
    depth = yy - ry

    mask = np.clip((depth + 0.8) / 1.6, 0.0, 1.0)
    mask *= np.clip((hy + 3.0 - yy) / 2.0, 0.0, 1.0)

    r_coarse = fbm_gouache(bh, W, rng, scales=((76, 50), (30, 22), (13, 10), (5.8, 4.8)))
    r_crag = 1.0 - np.abs(fbm_gouache(bh, W, rng, scales=((52, 36), (22, 16), (9.5, 7.5), (4.5, 4.0))))
    xx_shift = (np.arange(W, dtype=np.float32)[None, :] + depth * 0.44) % W
    yy_loc = np.broadcast_to(np.arange(bh, dtype=np.float32)[:, None], (bh, W))
    strata = remap_bilinear(r_crag, xx_shift, yy_loc)
    relief = 0.56 * strata + 0.44 * r_coarse

    gx = np.zeros_like(relief)
    gy = np.zeros_like(relief)
    gx[:, 1:-1] = (relief[:, 2:] - relief[:, :-2]) * 3.8
    gy[1:-1, :] = (relief[2:, :] - relief[:-2, :]) * 2.6
    dr_dx = np.gradient(ridge_y).astype(np.float32)[None, :] * 0.25
    lit = np.clip(0.46 + (gx + dr_dx) * 0.52 - gy * 0.22, 0.0, 1.0)

    rd = np.array(rock_dark, dtype=np.float32)[None, None, :]
    rl = np.array(rock_lit, dtype=np.float32)[None, None, :]
    sc = np.array(snow_col, dtype=np.float32)[None, None, :]
    mc = np.array(mist_col, dtype=np.float32)[None, None, :]

    m_sl = mottle[y_min:y_max]
    d_sl = dabs[y_min:y_max]
    t_sl = tooth[y_min:y_max]

    rock = rd * (1.0 - lit[..., None]) + rl * lit[..., None]
    rock += (m_sl * 0.018 + d_sl * 0.013 + t_sl * 0.005)[..., None]

    elev_f = np.exp(-np.maximum(0.0, depth) / 92.0)
    couloir = np.clip((strata - 0.37) * 2.2, 0.0, 1.0)
    snow_raw = (elev_f * 0.60 + couloir * 0.50 + lit * 0.36 + m_sl * 0.22 + d_sl * 0.14) - (1.0 - snow_bias)
    snow_m = np.clip(snow_raw * 2.1, 0.0, 0.90)
    snow_shaded = sc * (0.76 + 0.28 * lit[..., None]) + (m_sl * 0.014)[..., None]
    col = rock * (1.0 - snow_m[..., None]) + snow_shaded * snow_m[..., None]

    rim = np.exp(-np.maximum(0.0, depth) / 3.0) * np.clip(0.55 + dr_dx * 0.6, 0.15, 1.0) * 0.34
    col += rim[..., None] * sc

    h_above_water = np.maximum(0.0, hy - yy)
    mist_f = np.exp(-h_above_water / mist_height) * mist_strength * np.clip(0.75 + 0.32 * m_sl, 0.35, 1.15)
    mist_f = np.clip(mist_f, 0.0, 0.92)
    col = col * (1.0 - mist_f[..., None]) + mc * mist_f[..., None]

    out = canvas.copy()
    out[y_min:y_max] = out[y_min:y_max] * (1.0 - mask[..., None]) + col * mask[..., None]
    return out


# =============================================================================
# Stage 1 — Metric 3D Pinhole Camera & Exact 3x3 Plane Homography
# =============================================================================

class Camera3D:
    def __init__(self, w: int, h: int, eye: Iterable[float], target: Iterable[float], fov_x_deg: float = 56.0):
        self.w = w
        self.h = h
        self.eye = np.asarray(eye, dtype=np.float64)
        self.target = np.asarray(target, dtype=np.float64)
        fwd = self.target - self.eye
        fwd /= np.linalg.norm(fwd)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(fwd, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        self.R = np.vstack([right, -up, fwd])
        fx = (w * 0.5) / math.tan(math.radians(fov_x_deg * 0.5))
        self.K = np.array([[fx, 0.0, w * 0.5], [0.0, fx, h * 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)

        vp = self.K @ (self.R @ np.array([0.0, 1.0, 0.0]))
        self.vp_x = float(vp[0] / vp[2])
        self.vp_y = float(vp[1] / vp[2])
        self.horizon_y = self.vp_y

    def world_to_cam(self, pts: np.ndarray) -> np.ndarray:
        return (pts - self.eye) @ self.R.T

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cam = self.world_to_cam(pts)
        z = cam[..., 2]
        safe_z = np.where(np.abs(z) < 1e-6, 1e-6, z)
        u = self.K[0, 0] * (cam[..., 0] / safe_z) + self.K[0, 2]
        v = self.K[1, 1] * (cam[..., 1] / safe_z) + self.K[1, 2]
        return u, v, z

    def clip_segment_3d(self, p0: np.ndarray, p1: np.ndarray, z_near: float = Z_NEAR) -> tuple[np.ndarray, np.ndarray] | None:
        c0 = self.R @ (p0 - self.eye)
        c1 = self.R @ (p1 - self.eye)
        if c0[2] < z_near and c1[2] < z_near:
            return None
        a, b = p0.copy(), p1.copy()
        if c0[2] < z_near:
            t = (z_near - c0[2]) / (c1[2] - c0[2])
            a = p0 + t * (p1 - p0)
        elif c1[2] < z_near:
            t = (z_near - c1[2]) / (c0[2] - c1[2])
            b = p1 + t * (p0 - p1)
        return a, b

    def project_polyline(self, pts: np.ndarray) -> list[tuple[float, float, float]]:
        out: list[tuple[float, float, float]] = []
        for i in range(len(pts) - 1):
            cl = self.clip_segment_3d(pts[i], pts[i + 1])
            if cl is None:
                continue
            a, b = cl
            ua, va, za = self.project(a)
            ub, vb, zb = self.project(b)
            if not out:
                out.append((float(ua), float(va), float(za)))
            out.append((float(ub), float(vb), float(zb)))
        return out

    def plane_homography_uv_to_screen(self, p0: np.ndarray, e1: np.ndarray, e2: np.ndarray, uv_w: int, uv_h: int) -> np.ndarray:
        M = np.column_stack([self.R @ e1, self.R @ e2, self.R @ (p0 - self.eye)])
        H_uv = np.diag([1.0 / uv_w, 1.0 / uv_h, 1.0])
        return self.K @ M @ H_uv

    def warp_plane_over(
        self,
        dst_rgb: np.ndarray,
        tex_premul_rgba: np.ndarray,
        p0: np.ndarray,
        e1: np.ndarray,
        e2: np.ndarray,
        fog_rgb: np.ndarray,
        fog_start: float = 5.0,
        fog_density: float = 0.030,
        alpha_scale: float = 1.0,
        y_min_clip: int = 0,
    ) -> np.ndarray:
        th, tw = tex_premul_rgba.shape[:2]
        H_mat = self.plane_homography_uv_to_screen(p0, e1, e2, tw, th)
        try:
            H_inv = np.linalg.inv(H_mat)
        except np.linalg.LinAlgError:
            return dst_rgb

        corners = np.array([p0, p0 + e1, p0 + e1 + e2, p0 + e2], dtype=np.float64)
        xs, ys = [], []
        for i, j in ((0, 1), (1, 2), (2, 3), (3, 0)):
            cl = self.clip_segment_3d(corners[i], corners[j])
            if cl is None:
                continue
            for pt in cl:
                sx, sy, _ = self.project(pt)
                xs.append(float(sx))
                ys.append(float(sy))
        if not xs:
            return dst_rgb

        x0 = max(0, int(math.floor(min(xs))) - 1)
        x1 = min(self.w, int(math.ceil(max(xs))) + 2)
        y0 = max(y_min_clip, int(math.floor(min(ys))) - 1)
        y1 = min(self.h, int(math.ceil(max(ys))) + 2)
        if x0 >= x1 or y0 >= y1:
            return dst_rgb

        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        denom = H_inv[2, 0] * xx + H_inv[2, 1] * yy + H_inv[2, 2]
        safe_d = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        u_px = (H_inv[0, 0] * xx + H_inv[0, 1] * yy + H_inv[0, 2]) / safe_d
        v_px = (H_inv[1, 0] * xx + H_inv[1, 1] * yy + H_inv[1, 2]) / safe_d

        u_n = u_px / tw
        v_n = v_px / th
        valid = (u_n >= 0.0) & (u_n <= 1.0) & (v_n >= 0.0) & (v_n <= 1.0)
        if not np.any(valid):
            return dst_rgb

        r_e1 = self.R @ e1
        r_e2 = self.R @ e2
        r_p0 = self.R @ (p0 - self.eye)
        cam_x = r_p0[0] + u_n * r_e1[0] + v_n * r_e2[0]
        cam_y = r_p0[1] + u_n * r_e1[1] + v_n * r_e2[1]
        cam_z = r_p0[2] + u_n * r_e1[2] + v_n * r_e2[2]
        valid &= cam_z > Z_NEAR
        if not np.any(valid):
            return dst_rgb

        dist = np.sqrt(cam_x * cam_x + cam_y * cam_y + cam_z * cam_z).astype(np.float32)
        fog_t = 1.0 - np.exp(-np.maximum(0.0, dist - fog_start) * fog_density)

        u_cl = np.clip(u_px, 0.0, tw - 1.001)
        v_cl = np.clip(v_px, 0.0, th - 1.001)
        sampled = remap_bilinear(tex_premul_rgba, u_cl, v_cl)

        v_f = valid[..., None].astype(np.float32) * float(alpha_scale)
        alpha = sampled[:, :, 3:4] * v_f
        rgb_valid = sampled[:, :, :3] * v_f

        fg_rgb = rgb_valid * (1.0 - fog_t[..., None]) + (fog_rgb[None, None, :] * alpha) * fog_t[..., None]
        patch = np.concatenate([fg_rgb, alpha], axis=-1)
        dst_rgb[y0:y1, x0:x1] = over_premul(dst_rgb[y0:y1, x0:x1], patch)
        return dst_rgb


def mirror_plane_z0(p0: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact 3D reflection across horizontal water plane z = Z_WATER (n = [0, 0, 1])."""
    n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    p_ref = np.array([0.0, 0.0, Z_WATER], dtype=np.float64)

    def refl_pt(pt: np.ndarray) -> np.ndarray:
        return pt - 2.0 * np.dot(pt - p_ref, n) * n

    p0_m = refl_pt(p0)
    e1_m = refl_pt(p0 + e1) - p0_m
    e2_m = refl_pt(p0 + e2) - p0_m
    return p0_m, e1_m, e2_m


# =============================================================================
# Stage 2 — Unified 3D Light Solver (6-Step Pitched Roof & Bollard Shadow Trace)
# =============================================================================

def boathouse_roof_z(x: np.ndarray) -> np.ndarray:
    half_span = (BH_X_RIGHT + BH_EAVE_OVERHANG) - BH_X_RIDGE
    slope = (BH_Z_RIDGE - (BH_Z_EAVE - 0.20)) / half_span
    return BH_Z_RIDGE - np.abs(x - BH_X_RIDGE) * slope


def trace_boathouse_shadow(wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> np.ndarray:
    inv_sz = 1.0 / (-SUN_DIR[2])
    z_est = np.full_like(wx, BH_Z_EAVE + 0.8, dtype=np.float64)
    for _ in range(6):
        t = np.maximum(0.0, (z_est - wz) * inv_sz)
        hx = wx - SUN_DIR[0] * t
        hy = wy - SUN_DIR[1] * t
        z_est = np.clip(boathouse_roof_z(hx), BH_Z_EAVE - 0.25, BH_Z_RIDGE)
    t = np.maximum(0.0, (z_est - wz) * inv_sz)
    hx = wx - SUN_DIR[0] * t
    hy = wy - SUN_DIR[1] * t

    in_roof = (
        (hx >= BH_X_LEFT - BH_EAVE_OVERHANG)
        & (hx <= BH_X_RIGHT + BH_EAVE_OVERHANG)
        & (hy >= BH_Y0 - 0.35)
        & (hy <= BH_Y1 + 0.35)
        & (wz < z_est)
    )

    t_wall = np.where(np.abs(SUN_DIR[0]) > 1e-6, (BH_X_RIGHT - wx) / (-SUN_DIR[0]), -1.0)
    wy_hit = wy - SUN_DIR[1] * t_wall
    wz_hit = wz - SUN_DIR[2] * t_wall
    in_wall = (t_wall > 0.0) & (wy_hit >= BH_Y0) & (wy_hit <= BH_Y1) & (wz_hit >= Z_DECK - 0.05) & (wz_hit <= BH_Z_EAVE)

    dx_eave = np.maximum(0.0, hx - (BH_X_RIGHT + BH_EAVE_OVERHANG))
    dy_eave = np.maximum(0.0, np.maximum((BH_Y0 - 0.35) - hy, hy - (BH_Y1 + 0.35)))
    penumbra = np.exp(-(dx_eave * dx_eave + dy_eave * dy_eave) / (0.38 ** 2))
    in_bh = np.maximum(np.where(in_roof | in_wall, 1.0, 0.0), penumbra * 0.55 * ((hx > BH_X_LEFT - 1.0) & (wz < BH_Z_EAVE)))

    bollard_sh = np.zeros_like(wx, dtype=np.float64)
    for by in (3.7, 7.6, 11.8, 15.8):
        bx = 0.06
        for bz in (Z_DECK + 0.15, Z_DECK + 0.38, Z_DECK + 0.56):
            tb = np.maximum(0.0, (bz - wz) * inv_sz)
            pbx = wx - SUN_DIR[0] * tb
            pby = wy - SUN_DIR[1] * tb
            d2 = (pbx - bx) ** 2 + (pby - by) ** 2
            bollard_sh = np.maximum(bollard_sh, np.exp(-d2 / (0.14 ** 2)) * 0.55)

    occ = np.clip(np.maximum(in_bh * 0.70, bollard_sh), 0.0, 0.76)
    return (1.0 - occ).astype(np.float32)


# =============================================================================
# Stage 3 Helpers — 2x Supersampled PyCairo & Tapered Catmull-Rom Splines
# =============================================================================

def new_cairo_2x() -> tuple[cairo.ImageSurface, cairo.Context]:
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W * 2, H * 2)
    ctx = cairo.Context(surf)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.scale(2.0, 2.0)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    return surf, ctx


def finish_cairo_2x(surf: cairo.ImageSurface) -> np.ndarray:
    surf.flush()
    buf = np.frombuffer(surf.get_data(), dtype=np.uint8).reshape(H * 2, W * 2, 4)
    rgba_2x = np.stack([buf[:, :, 2], buf[:, :, 1], buf[:, :, 0], buf[:, :, 3]], axis=-1)
    a2 = rgba_2x[:, :, 3:4].astype(np.float32) / 255.0
    rgb2 = np.where(a2 > 1e-4, rgba_2x[:, :, :3].astype(np.float32) / np.maximum(a2, 1e-4), 0.0)
    out = np.empty((H, W, 4), dtype=np.float32)
    for c in range(3):
        im = Image.fromarray(np.clip(rgb2[:, :, c], 0, 255).astype(np.uint8), mode="L").resize((W, H), Image.Resampling.LANCZOS)
        out[:, :, c] = np.asarray(im, dtype=np.float32) / 255.0
    a1 = np.asarray(Image.fromarray(rgba_2x[:, :, 3], mode="L").resize((W, H), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    out[:, :, 3] = a1
    out[:, :, :3] *= a1[..., None]
    return out


def stroke_catmull_rom(
    ctx: cairo.Context,
    pts: list[tuple[float, float]],
    base_w: float,
    z_depth: float,
    rgba: tuple[float, float, float, float],
    rng: np.random.Generator,
    fade_dist: float = 24.0,
    taper: bool = True,
    tremor: float = 0.24,
) -> None:
    if len(pts) < 2:
        return
    decay = float(np.clip(1.0 / (1.0 + z_depth / (fade_dist * 0.45)), 0.25, 1.0))
    w_eff = base_w * decay
    ext = [pts[0]] + pts + [pts[-1]]
    dense: list[tuple[float, float]] = []
    steps = 6
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for s in range(steps):
            t = s / steps
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            if tremor > 0:
                x += float(rng.normal(0.0, tremor * decay))
                y += float(rng.normal(0.0, tremor * decay))
            dense.append((x, y))
    dense.append(pts[-1])

    ctx.set_source_rgba(*rgba)
    n = len(dense) - 1
    for i in range(n):
        frac = (i + 0.5) / max(1, n)
        wt = (0.35 + 0.65 * math.sin(frac * math.pi)) if taper else 1.0
        ctx.set_line_width(max(0.45, w_eff * wt))
        ctx.move_to(*dense[i])
        ctx.line_to(*dense[i + 1])
        ctx.stroke()


# =============================================================================
# Stage 1 & 3 — Blue-Hour Fjord Sky & Jagged Glaciated Basalt Cliffs
# =============================================================================

def paint_sky_and_mountains(cam: Camera3D, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Paint Nordic blue-hour twilight sky & 4 depth-layered craggy glaciated basalt fjord walls."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    hy = float(cam.horizon_y)
    u = xx / W
    v_sky = np.clip(yy / max(1.0, hy), 0.0, 1.15)

    mottle = fbm_gouache(H, W, rng)
    dabs = flat_brush_dabs(H, W, rng, count=94)
    tooth = paper_tooth(H, W, rng)

    # Nordic blue-hour twilight palette: #080f18 zenith -> arctic steel blue -> luminous glacier-blue horizon
    c_zen = np.array([0.031, 0.062, 0.105], dtype=np.float32)
    c_mid = np.array([0.098, 0.172, 0.262], dtype=np.float32)
    c_low = np.array([0.335, 0.515, 0.665], dtype=np.float32)
    c_hor = np.array([0.640, 0.805, 0.915], dtype=np.float32)

    t1 = np.clip(v_sky / 0.52, 0.0, 1.0)
    t2 = np.clip((v_sky - 0.44) / 0.42, 0.0, 1.0)
    t3 = np.clip((v_sky - 0.78) / 0.24, 0.0, 1.0)
    sky = (
        c_zen[None, None, :] * (1.0 - t1[..., None])
        + c_mid[None, None, :] * (t1 * (1.0 - t2))[..., None]
        + c_low[None, None, :] * (t2 * (1.0 - t3))[..., None]
        + c_hor[None, None, :] * (t2 * t3)[..., None]
    )

    du = (u - 0.62) * 1.90
    dv = (v_sky - 0.94) * 2.50
    glow = np.exp(-(du * du + dv * dv)).astype(np.float32)
    sky += glow[..., None] * np.array([0.21, 0.24, 0.28], dtype=np.float32)

    cloud_f = fbm_gouache(H, W, rng, scales=((110, 155), (42, 62), (16, 22), (6.5, 8.0)))
    band_env = np.exp(-((v_sky - 0.56) / 0.24) ** 2) * (0.45 + 0.55 * np.clip((u - 0.15) / 0.75, 0.0, 1.0))
    cloud_m = np.clip((cloud_f * 0.65 + 0.18) * band_env, 0.0, 1.0)
    c_cl_dark = np.array([0.085, 0.145, 0.225], dtype=np.float32)
    c_cl_lit = np.array([0.56, 0.74, 0.86], dtype=np.float32)
    lit_f = np.clip((v_sky - 0.35) / 0.55, 0.0, 1.0) * (0.55 + 0.45 * glow)
    cl_col = c_cl_dark[None, None, :] * (1.0 - lit_f[..., None]) + c_cl_lit[None, None, :] * lit_f[..., None]
    sky = sky * (1.0 - cloud_m[..., None] * 0.48) + cl_col * (cloud_m[..., None] * 0.48)
    sky += (mottle * 0.015 + dabs * 0.011 + tooth * 0.005)[..., None]

    ux = np.linspace(0.0, 1.0, W, dtype=np.float32)

    # Layer 0: Far snow-draped alpine fjord headwall across the misty pass
    p0 = ridged_profile_1d(W, rng, octaves=((6, 0.48), (13, 0.26), (29, 0.14), (63, 0.08), (135, 0.04)))
    env0 = 162.0 * np.exp(-((ux - 0.58) / 0.32) ** 2) + 42.0
    ridge0 = hy - env0 * (0.45 + 0.65 * p0)
    sky = composite_painterly_mountain_layer(
        sky, ridge0, hy,
        rock_dark=(0.20, 0.33, 0.47), rock_lit=(0.30, 0.46, 0.61), snow_col=(0.76, 0.88, 0.96),
        snow_bias=0.60, mist_col=(0.56, 0.74, 0.87), mist_strength=0.72, mist_height=72.0,
        rng=rng, mottle=mottle, dabs=dabs, tooth=tooth,
    )

    # Layer 1: Left-mid distant fjord mountain shoulder behind the boathouse
    p1 = ridged_profile_1d(W, rng, octaves=((5, 0.50), (11, 0.26), (25, 0.14), (55, 0.07), (120, 0.03)))
    env1 = np.where(ux < 0.60, 210.0 * np.power(np.clip((0.60 - ux) / 0.60, 0.0, 1.0), 0.78), 0.0)
    ridge1 = np.where(env1 > 1.0, hy - env1 * (0.48 + 0.58 * p1) - 6.0, hy + 24.0)
    sky = composite_painterly_mountain_layer(
        sky, ridge1, hy,
        rock_dark=(0.12, 0.21, 0.31), rock_lit=(0.20, 0.33, 0.46), snow_col=(0.72, 0.86, 0.95),
        snow_bias=0.52, mist_col=(0.52, 0.70, 0.84), mist_strength=0.62, mist_height=62.0,
        rng=rng, mottle=mottle, dabs=dabs, tooth=tooth,
    )

    # Layer 2: Towering dramatic Norwegian basalt fjord massif rising steeply on the right!
    p2 = ridged_profile_1d(W, rng, octaves=((5, 0.50), (11, 0.26), (23, 0.14), (51, 0.07), (115, 0.03)))
    env2 = np.where(ux > 0.55, 325.0 * np.power(np.clip((ux - 0.55) / 0.45, 0.0, 1.0), 0.72), 0.0)
    ridge2 = np.where(env2 > 1.0, hy - env2 * (0.50 + 0.58 * p2) - 8.0, hy + 24.0)
    sky = composite_painterly_mountain_layer(
        sky, ridge2, hy,
        rock_dark=(0.068, 0.122, 0.195), rock_lit=(0.145, 0.242, 0.348), snow_col=(0.74, 0.87, 0.96),
        snow_bias=0.56, mist_col=(0.48, 0.66, 0.80), mist_strength=0.52, mist_height=58.0,
        rng=rng, mottle=mottle, dabs=dabs, tooth=tooth,
    )

    # Layer 3: Closer dark basalt promontory jutting into the far right edge
    p3 = ridged_profile_1d(W, rng, octaves=((4, 0.52), (9, 0.27), (21, 0.14), (47, 0.07)))
    env3 = np.where(ux > 0.78, 225.0 * np.power(np.clip((ux - 0.78) / 0.22, 0.0, 1.0), 0.80), 0.0)
    ridge3 = np.where(env3 > 1.0, hy - env3 * (0.54 + 0.50 * p3) - 4.0, hy + 26.0)
    sky = composite_painterly_mountain_layer(
        sky, ridge3, hy,
        rock_dark=(0.038, 0.072, 0.122), rock_lit=(0.088, 0.152, 0.232), snow_col=(0.62, 0.78, 0.90),
        snow_bias=0.40, mist_col=(0.44, 0.62, 0.76), mist_strength=0.36, mist_height=42.0,
        rng=rng, mottle=mottle, dabs=dabs, tooth=tooth,
    )

    # Distant fjord shoreline cabin pinpricks across the bay
    s_lit, c_lit = new_cairo_2x()
    for lx, ly, lr in [(1362, hy - 5.2, 2.2), (1418, hy - 3.8, 1.8), (1545, hy - 6.0, 2.0)]:
        lg = cairo.RadialGradient(lx, ly, 0.2, lx, ly, lr * 5.2)
        lg.add_color_stop_rgba(0.0, 1.0, 0.88, 0.64, 0.92)
        lg.add_color_stop_rgba(0.45, 0.92, 0.72, 0.42, 0.30)
        lg.add_color_stop_rgba(1.0, 0.90, 0.66, 0.35, 0.0)
        c_lit.set_source(lg)
        c_lit.arc(lx, ly, lr * 5.2, 0, math.tau)
        c_lit.fill()
    sky = over_premul(sky, finish_cairo_2x(s_lit))

    # Horizontal arctic mist veil floating along the lower fjord walls
    mist_n = fbm_gouache(H, W, rng, scales=((85, 130), (34, 48), (14, 18), (6, 7)))
    mist_band = np.exp(-((yy - (hy - 28.0)) / 34.0) ** 2) * np.clip((u - 0.32) / 0.45, 0.0, 1.0)
    mist_a = np.clip((0.52 + 0.48 * mist_n) * mist_band * 0.44, 0.0, 0.56)
    sky = sky * (1.0 - mist_a[..., None]) + np.array([0.54, 0.72, 0.85], dtype=np.float32)[None, None, :] * mist_a[..., None]

    fog_rgb = np.array([0.36, 0.53, 0.67], dtype=np.float32)
    return np.clip(sky, 0.0, 1.25), fog_rgb, mottle, dabs, tooth


# =============================================================================
# Stage 1 & 2 — Planar UV Surfaces (Voronoi Basalt Quay, Wet Plank Jetty, Boathouse)
# =============================================================================

def paint_voronoi_basalt_uv(
    rng: np.random.Generator,
    u_span: tuple[float, float],
    v_span: tuple[float, float],
    is_top: bool,
    tw: int = 768,
    th: int = 640,
) -> np.ndarray:
    """Weathered Nordic basalt stone quay wall & flagstone apron with irregular Voronoi (d2 - d1) mortar."""
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    a_m = u_span[0] + u * (u_span[1] - u_span[0])
    b_m = v_span[0] + v * (v_span[1] - v_span[0])

    nx = resize_bicubic(rng.standard_normal((max(4, th // 24), max(4, tw // 24))).astype(np.float32), tw, th) * 0.048
    ny = resize_bicubic(rng.standard_normal((max(4, th // 24), max(4, tw // 24))).astype(np.float32), tw, th) * 0.042
    px, py = a_m + nx, b_m + ny

    cell_a, cell_b = (0.64, 0.58) if is_top else (0.76, 0.40)
    ix = np.floor(px / cell_a).astype(np.int32)
    iy = np.floor(py / cell_b).astype(np.int32)

    jx_tab = rng.uniform(0.18, 0.82, size=256)
    jy_tab = rng.uniform(0.18, 0.82, size=256)
    tone_tab = rng.uniform(-0.036, 0.042, size=256)
    hue_tab = rng.uniform(-0.014, 0.016, size=256)

    d1 = np.full((th, tw), 1e9, dtype=np.float64)
    d2 = np.full((th, tw), 1e9, dtype=np.float64)
    best_id = np.zeros((th, tw), dtype=np.int32)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cx, cy = ix + dx, iy + dy
            h_id = ((cx * 73856093) ^ (cy * 19349663)) & 255
            sx = (cx + jx_tab[h_id]) * cell_a
            sy = (cy + jy_tab[h_id]) * cell_b
            dist = np.hypot(px - sx, py - sy)
            closer = dist < d1
            d2 = np.where(closer, d1, np.minimum(d2, dist))
            d1 = np.where(closer, dist, d1)
            best_id = np.where(closer, h_id, best_id)

    edge_d = (d2 - d1).astype(np.float32)
    stone = np.clip((edge_d - 0.015) / 0.028, 0.0, 1.0)
    bevel = np.clip(1.0 - np.abs(edge_d - 0.046) / 0.022, 0.0, 1.0)

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=82)
    tooth = paper_tooth(th, tw, rng)
    sv = tone_tab[best_id].astype(np.float32)
    hv = hue_tab[best_id].astype(np.float32)

    base = np.zeros((th, tw, 3), dtype=np.float32)
    base[:, :, 0] = 0.082 + sv * 0.58 + hv * 0.35 + mottle * 0.014 + dabs * 0.011
    base[:, :, 1] = 0.120 + sv * 0.68 + mottle * 0.018 + dabs * 0.014
    base[:, :, 2] = 0.168 + sv * 0.78 - hv * 0.30 + mottle * 0.022 + dabs * 0.016

    mortar = np.array([0.016, 0.028, 0.046], dtype=np.float32)
    rgb = stone[..., None] * base + (1.0 - stone[..., None]) * mortar

    if is_top:
        sh = trace_boathouse_shadow(a_m, b_m, np.full_like(a_m, Z_QUAY))
        rgb *= (0.48 + 0.68 * sh)[..., None]
        # Subtle frost rime on stone bevels + wet sky sheen + glacier lantern spill near door
        rgb += (bevel * np.clip(0.5 + 0.5 * mottle, 0.0, 1.0) * 0.22)[..., None] * np.array([0.38, 0.54, 0.68], dtype=np.float32)
        puddle = np.clip((mottle * 0.58 + dabs * 0.38 - 0.08) * 1.4, 0.0, 1.0) * stone
        rgb += (puddle * sh)[..., None] * np.array([0.14, 0.23, 0.34], dtype=np.float32)
        d_lan = np.hypot((a_m - (-2.38)) / 1.85, (b_m - 6.45) / 2.20).astype(np.float32)
        rgb += (np.exp(-(d_lan * d_lan)) * (0.45 + 0.55 * puddle))[..., None] * np.array([0.18, 0.34, 0.46], dtype=np.float32)
    else:
        # Vertical raised stone quay curb face dropping cleanly onto the jetty deck
        sh = trace_boathouse_shadow(np.full_like(a_m, -1.92), a_m, b_m)
        rgb *= (0.58 + 0.52 * sh)[..., None]
        coping = np.clip((b_m.astype(np.float32) - (Z_QUAY - 0.06)) / 0.05, 0.0, 1.0)
        rgb += coping[..., None] * np.array([0.16, 0.25, 0.34], dtype=np.float32)

    rgb = apply_shadow_bounce(rgb, strength=0.85)
    rgb += tooth[..., None] * 0.006
    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.15)
    return rgba


def paint_wet_jetty_deck_uv(
    rng: np.random.Generator,
    x_span: tuple[float, float],
    y_span: tuple[float, float],
    tw: int = 768,
    th: int = 1152,
) -> np.ndarray:
    """Weathered Nordic timber plank jetty deck with aligned SUN_DIR shadows & warm window reflection."""
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wx = x_span[0] + u * (x_span[1] - x_span[0])
    wy = y_span[0] + v * (y_span[1] - y_span[0])
    wz = np.full_like(wx, Z_DECK)

    plank_pitch = 0.235
    py_jit = wy + resize_bicubic(rng.standard_normal((th // 28, tw // 28)).astype(np.float32), tw, th) * 0.007
    p_idx = np.floor(py_jit / plank_pitch).astype(np.int32)
    p_frac = (py_jit / plank_pitch) - p_idx
    seam = np.clip(np.minimum(p_frac, 1.0 - p_frac) / 0.085, 0.0, 1.0).astype(np.float32)

    tone_tab = rng.uniform(-0.032, 0.036, size=256)
    h_id = (p_idx * 73856093) & 255
    ptone = tone_tab[h_id].astype(np.float32)

    mottle = fbm_gouache(th, tw, rng, scales=((70, 64), (24, 28), (10, 11), (4.5, 5.0)))
    dabs = flat_brush_dabs(th, tw, rng, count=88)
    tooth = paper_tooth(th, tw, rng)
    wood_g = fbm_gouache(th, tw, rng, scales=((18, 42), (7, 16), (3.5, 7.5), (2.2, 4.0)))

    base = np.zeros((th, tw, 3), dtype=np.float32)
    base[:, :, 0] = 0.068 + ptone * 0.65 + wood_g * 0.011 + mottle * 0.012 + dabs * 0.010
    base[:, :, 1] = 0.095 + ptone * 0.72 + wood_g * 0.014 + mottle * 0.016 + dabs * 0.013
    base[:, :, 2] = 0.132 + ptone * 0.80 + wood_g * 0.018 + mottle * 0.020 + dabs * 0.016

    gap_col = np.array([0.014, 0.024, 0.040], dtype=np.float32)
    rgb = seam[..., None] * base + (1.0 - seam[..., None]) * gap_col

    sh = trace_boathouse_shadow(wx, wy, wz)
    rgb *= (0.44 + 0.72 * sh)[..., None]

    wet_pool = np.clip(0.52 + 0.42 * mottle + 0.25 * dabs, 0.0, 1.0) * seam
    edge_sky = np.clip((wx.astype(np.float32) - x_span[0]) / (x_span[1] - x_span[0]), 0.0, 1.0)
    sky_sheen = wet_pool * sh * (0.28 + 0.72 * edge_sky)
    rgb += sky_sheen[..., None] * np.array([0.13, 0.23, 0.34], dtype=np.float32)

    outer_frost = np.clip((wx.astype(np.float32) - (x_span[1] - 0.28)) / 0.26, 0.0, 1.0) * (0.55 + 0.45 * mottle)
    bevel_frost = np.clip(1.0 - np.abs(p_frac.astype(np.float32) - 0.11) / 0.06, 0.0, 1.0) * 0.35 * sh
    rgb += (outer_frost * 0.52 + bevel_frost * 0.38)[..., None] * np.array([0.36, 0.50, 0.64], dtype=np.float32)

    # Warm amber lantern/window reflection pool spilling across wet jetty planks!
    win_yc = 0.5 * (WIN_Y0 + WIN_Y1)
    dx_w = np.maximum(0.0, wx - BH_X_RIGHT)
    dy_w = wy - win_yc
    win_pool = np.exp(-((dy_w / 1.20) ** 2 + (dx_w / 1.95) ** 2)).astype(np.float32) * (0.48 + 0.52 * wet_pool)
    mull_y = 0.84 + 0.16 * np.cos((wy - win_yc) * math.pi / 0.55).astype(np.float32)
    rgb += (win_pool * mull_y * seam)[..., None] * np.array([0.50, 0.33, 0.15], dtype=np.float32)

    d_lan = np.hypot((wx - (-1.95)) / 1.35, (wy - 6.55) / 1.65).astype(np.float32)
    lan_glow = np.exp(-(d_lan * d_lan)) * (0.45 + 0.55 * wet_pool)
    rgb += (lan_glow * seam)[..., None] * np.array([0.16, 0.28, 0.38], dtype=np.float32)

    nail_x = (np.abs(wx - (x_span[1] - 0.14)) < 0.022) | (np.abs(wx - (x_span[0] + 0.22)) < 0.022)
    nail_y = np.abs(p_frac - 0.5) < 0.08
    rgb = np.where((nail_x & nail_y)[..., None], rgb * 0.52, rgb)

    rgb = apply_shadow_bounce(rgb, strength=0.95)
    rgb += tooth[..., None] * 0.006
    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.25)
    return rgba


def paint_boathouse_east_wall_uv(rng: np.random.Generator, tw: int = 768, th: int = 448) -> np.ndarray:
    """Dark tarred Norwegian timber siding wall facing the fjord with a warmly glowing 6-pane window."""
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wy = BH_Y0 + u * (BH_Y1 - BH_Y0)
    wz = Z_DECK + v * (BH_Z_EAVE - Z_DECK)

    pitch = 0.215
    b_idx = np.floor(wy / pitch).astype(np.int32)
    b_frac = (wy / pitch) - b_idx
    batten = np.clip(1.0 - np.abs(b_frac - 0.5) / 0.14, 0.0, 1.0).astype(np.float32)
    groove = np.clip(np.minimum(b_frac, 1.0 - b_frac) / 0.07, 0.0, 1.0).astype(np.float32)

    btone = rng.uniform(-0.024, 0.028, size=256)[(b_idx * 73856093) & 255].astype(np.float32)
    mottle = fbm_gouache(th, tw, rng)
    tooth = paper_tooth(th, tw, rng)

    rgb = np.zeros((th, tw, 3), dtype=np.float32)
    rgb[:, :, 0] = 0.054 + btone * 0.55 + mottle * 0.012 + batten * 0.012
    rgb[:, :, 1] = 0.074 + btone * 0.62 + mottle * 0.016 + batten * 0.016
    rgb[:, :, 2] = 0.104 + btone * 0.72 + mottle * 0.020 + batten * 0.022
    rgb *= (0.48 + 0.52 * groove)[..., None]

    eave_sh = np.clip((BH_Z_EAVE - wz.astype(np.float32)) / 0.58, 0.0, 1.0)
    rgb *= (0.46 + 0.54 * eave_sh)[..., None]
    rgb += (batten * eave_sh)[..., None] * np.array([0.028, 0.052, 0.076], dtype=np.float32)

    sill_frost = np.clip((Z_DECK + 0.24 - wz.astype(np.float32)) / 0.24, 0.0, 1.0) * np.clip(0.5 + 0.5 * mottle, 0.0, 1.0)
    rgb += sill_frost[..., None] * np.array([0.12, 0.18, 0.25], dtype=np.float32)

    frame_pad = 0.11
    in_frame = (wy >= WIN_Y0 - frame_pad) & (wy <= WIN_Y1 + frame_pad) & (wz >= WIN_Z0 - frame_pad) & (wz <= WIN_Z1 + frame_pad)
    in_glass = (wy >= WIN_Y0) & (wy <= WIN_Y1) & (wz >= WIN_Z0) & (wz <= WIN_Z1)

    wy_c, wz_c = 0.5 * (WIN_Y0 + WIN_Y1), 0.5 * (WIN_Z0 + WIN_Z1)
    d_win = np.hypot((wy - wy_c) / 1.38, (wz - wz_c) / 1.08).astype(np.float32)
    rgb += (np.exp(-(d_win * d_win * 1.42)) * 0.58)[..., None] * np.array([0.44, 0.28, 0.12], dtype=np.float32)

    frame_col = np.array([0.24, 0.30, 0.36], dtype=np.float32) + mottle[..., None] * 0.015 + np.array([0.24, 0.15, 0.06], dtype=np.float32)
    rgb = np.where(in_frame[..., None], frame_col, rgb)

    gu = np.clip((wy - WIN_Y0) / (WIN_Y1 - WIN_Y0), 0.0, 1.0).astype(np.float32)
    gv = np.clip((wz - WIN_Z0) / (WIN_Z1 - WIN_Z0), 0.0, 1.0).astype(np.float32)
    lamp_core = np.exp(-(((gu - 0.52) / 0.42) ** 2 + ((gv - 0.38) / 0.46) ** 2))
    glass_col = (
        np.array([0.46, 0.34, 0.24], dtype=np.float32)[None, None, :] * (1.0 - lamp_core[..., None])
        + np.array([1.10, 0.85, 0.52], dtype=np.float32)[None, None, :] * lamp_core[..., None]
    )
    glass_col += (np.clip(gv - 0.55, 0.0, 1.0) * 0.24)[..., None] * np.array([0.20, 0.36, 0.50], dtype=np.float32)
    glass_col += (mottle * 0.032)[..., None]

    muntin_v = (np.abs(gu - 0.333) < 0.028) | (np.abs(gu - 0.667) < 0.028)
    muntin_h = np.abs(gv - 0.50) < 0.034
    is_muntin = in_glass & (muntin_v | muntin_h)
    rgb = np.where(in_glass[..., None], glass_col, rgb)
    rgb = np.where(is_muntin[..., None], np.array([0.08, 0.09, 0.11], dtype=np.float32), rgb)

    in_w2 = (wy >= 11.92) & (wy <= 12.68) & (wz >= 1.92) & (wz <= 2.66)
    in_f2 = (wy >= 11.83) & (wy <= 12.77) & (wz >= 1.83) & (wz <= 2.75)
    rgb = np.where(in_f2[..., None], np.array([0.16, 0.22, 0.29], dtype=np.float32), rgb)
    w2_col = np.array([0.32, 0.46, 0.58], dtype=np.float32) * (0.75 + 0.25 * mottle[..., None])
    rgb = np.where(in_w2[..., None], w2_col, rgb)
    w2_bar = in_w2 & ((np.abs(wy - 12.30) < 0.024) | (np.abs(wz - 2.29) < 0.024))
    rgb = np.where(w2_bar[..., None], np.array([0.06, 0.09, 0.13], dtype=np.float32), rgb)

    rgb = apply_shadow_bounce(rgb, strength=0.85)
    rgb += tooth[..., None] * 0.005
    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.35)
    return rgba


def paint_boathouse_gable_front_uv(rng: np.random.Generator, tw: int = 640, th: int = 640) -> np.ndarray:
    """Front (camera-facing) dark-timber gable wall with richly planked double boathouse doors & iron hinges."""
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wx = BH_X_LEFT + u * (BH_X_RIGHT - BH_X_LEFT)
    wz = Z_DECK + v * (BH_Z_RIDGE - Z_DECK)

    z_roof = boathouse_roof_z(wx)
    alpha = np.clip((z_roof - wz) / 0.035, 0.0, 1.0).astype(np.float32)

    pitch = 0.21
    b_idx = np.floor(wx / pitch).astype(np.int32)
    b_frac = (wx / pitch) - b_idx
    groove = np.clip(np.minimum(b_frac, 1.0 - b_frac) / 0.08, 0.0, 1.0).astype(np.float32)
    btone = rng.uniform(-0.022, 0.026, size=256)[(b_idx * 73856093) & 255].astype(np.float32)

    mottle = fbm_gouache(th, tw, rng)
    tooth = paper_tooth(th, tw, rng)

    rgb = np.zeros((th, tw, 3), dtype=np.float32)
    rgb[:, :, 0] = 0.038 + btone * 0.48 + mottle * 0.011
    rgb[:, :, 1] = 0.056 + btone * 0.58 + mottle * 0.014
    rgb[:, :, 2] = 0.084 + btone * 0.70 + mottle * 0.018
    rgb *= (0.50 + 0.50 * groove)[..., None]

    roof_dist = (z_roof - wz).astype(np.float32)
    r_sh = np.clip(roof_dist / 0.48, 0.0, 1.0)
    rgb *= (0.50 + 0.50 * r_sh)[..., None]

    # Heavy planked double timber boat-doors (preserving individual vertical timber planks!)
    in_door = (wx >= -4.68) & (wx <= -2.88) & (wz >= Z_DECK + 0.04) & (wz <= 3.16)
    in_dframe = (wx >= -4.78) & (wx <= -2.78) & (wz >= Z_DECK) & (wz <= 3.26) & (~in_door)
    rgb = np.where(in_dframe[..., None], np.array([0.09, 0.13, 0.18], dtype=np.float32) + mottle[..., None] * 0.012, rgb)

    # Door planks (slightly narrower vertical boards + horizontal timber ledges + forged iron strap hinges!)
    dp_idx = np.floor(wx / 0.16).astype(np.int32)
    dp_frac = (wx / 0.16) - dp_idx
    dp_grv = np.clip(np.minimum(dp_frac, 1.0 - dp_frac) / 0.09, 0.0, 1.0).astype(np.float32)
    dp_tone = rng.uniform(-0.018, 0.022, size=256)[(dp_idx * 73856093) & 255].astype(np.float32)
    door_col = np.zeros_like(rgb)
    door_col[:, :, 0] = (0.044 + dp_tone * 0.5 + mottle * 0.011) * (0.48 + 0.52 * dp_grv)
    door_col[:, :, 1] = (0.064 + dp_tone * 0.6 + mottle * 0.014) * (0.48 + 0.52 * dp_grv)
    door_col[:, :, 2] = (0.094 + dp_tone * 0.7 + mottle * 0.018) * (0.48 + 0.52 * dp_grv)
    # Center split seam + horizontal cross-braces + dark iron strap hinges
    d_seam = (np.abs(wx - (-3.78)) < 0.024) | (np.abs(wz - 1.38) < 0.065) | (np.abs(wz - 2.75) < 0.065)
    hinges = ((np.abs(wz - 1.38) < 0.032) | (np.abs(wz - 2.75) < 0.032)) & ((wx < -4.18) | (wx > -3.38))
    door_col = np.where(d_seam[..., None], door_col * 0.65, door_col)
    door_col = np.where(hinges[..., None], np.array([0.020, 0.030, 0.045], dtype=np.float32), door_col)
    rgb = np.where(in_door[..., None], door_col, rgb)

    # Attic gable window
    in_att_f = (wx >= -4.18) & (wx <= -3.38) & (wz >= 3.68) & (wz <= 4.48)
    in_att_g = (wx >= -4.10) & (wx <= -3.46) & (wz >= 3.76) & (wz <= 4.40)
    rgb = np.where(in_att_f[..., None], np.array([0.16, 0.23, 0.30], dtype=np.float32), rgb)
    att_glow = np.array([0.58, 0.68, 0.76], dtype=np.float32) * (0.85 + 0.15 * mottle[..., None])
    rgb = np.where(in_att_g[..., None], att_glow, rgb)
    att_bar = in_att_g & ((np.abs(wx - (-3.78)) < 0.022) | (np.abs(wz - 4.08) < 0.022))
    rgb = np.where(att_bar[..., None], np.array([0.06, 0.09, 0.13], dtype=np.float32), rgb)

    # Glacier-blue porch lantern halo beside right side of front doors
    d_l = np.hypot((wx - (-2.38)) / 0.92, (wz - 2.66) / 0.98).astype(np.float32)
    rgb += np.exp(-(d_l * d_l * 1.65))[..., None] * np.array([0.24, 0.44, 0.58], dtype=np.float32)

    rgb = apply_shadow_bounce(rgb, strength=0.90)
    rgb += tooth[..., None] * 0.005
    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.25) * alpha[..., None]
    rgba[:, :, 3] = alpha
    return rgba


def paint_boathouse_roof_uv(rng: np.random.Generator, tw: int = 768, th: int = 448) -> np.ndarray:
    """Weathered Nordic dark-slate pitched roof with soft continuous wind-swept arctic snow mantle."""
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)   # along Y (front to back)
    v = yy / (th - 1)   # down slope from ridge (0) to eave (1)

    seam_u = np.sin(u * 28.0 * math.pi).astype(np.float32)
    course_v = np.mod(v * 16.0, 1.0).astype(np.float32)
    shingle_edge = np.clip(course_v / 0.12, 0.0, 1.0)

    mottle = fbm_gouache(th, tw, rng, scales=((64, 60), (24, 26), (10, 10), (4.5, 4.5)))
    tooth = paper_tooth(th, tw, rng)

    # Dark wet basalt-slate roof catching blue-hour sky light
    rgb = np.zeros((th, tw, 3), dtype=np.float32)
    rgb[:, :, 0] = 0.072 + mottle * 0.014 + seam_u * 0.009
    rgb[:, :, 1] = 0.114 + mottle * 0.018 + seam_u * 0.013
    rgb[:, :, 2] = 0.170 + mottle * 0.024 + seam_u * 0.018
    rgb *= (0.76 + 0.24 * shingle_edge)[..., None]

    # Soft, coherent wind-swept frost & thin snow dusting (NO high-contrast cow spots!)
    snow_drift = fbm_gouache(th, tw, rng, scales=((96, 82), (36, 32), (14, 14), (6, 6)))
    # Natural continuous thin frost/snow mantle thickest near the lower eave & ridge wind-shadow
    eave_mantle = np.clip((v.astype(np.float32) - 0.42) / 0.58, 0.0, 1.0) * 0.62
    ridge_frost = np.exp(-((v.astype(np.float32)) / 0.22) ** 2) * 0.34
    snow_m = np.clip(eave_mantle + ridge_frost + snow_drift * 0.18 + 0.06, 0.0, 0.72)
    # Soft feathered snow tone keeping subtle slate rib texture underneath!
    snow_col = np.array([0.50, 0.66, 0.80], dtype=np.float32) + (mottle * 0.022 + seam_u * 0.010)[..., None]
    rgb = rgb * (1.0 - snow_m[..., None]) + snow_col * snow_m[..., None]

    rgb = apply_shadow_bounce(rgb, strength=0.75)
    rgb += tooth[..., None] * 0.005
    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.2)
    return rgba


# =============================================================================
# Stage 2 — Fresnel Capillary-Rippled Fjord Water + 3D Plane-Mirrored Quads
# =============================================================================

def paint_fjord_water(
    cam: Camera3D,
    sky_mountain_rgb: np.ndarray,
    bh_east_uv: np.ndarray,
    bh_front_uv: np.ndarray,
    bh_roof_uv: np.ndarray,
    fog_rgb: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Combine capillary-rippled horizon reflections (y_src = 2 * horizon_y - y)
    with 3D plane-mirrored boathouse & jetty reflections across z = Z_WATER,
    weighted by Fresnel: fresnel_min + (fresnel_max - fresnel_min) * (1 - dist_t) ** 1.4.
    """
    hy = float(cam.horizon_y)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    y_mir = np.clip(2.0 * hy - yy, 0.0, hy - 1.0)
    refl = remap_bilinear(sky_mountain_rgb, xx, y_mir)

    p_east, e1_east, e2_east = mirror_plane_z0(
        np.array([BH_X_RIGHT, BH_Y0, Z_DECK]),
        np.array([0.0, BH_Y1 - BH_Y0, 0.0]),
        np.array([0.0, 0.0, BH_Z_EAVE - Z_DECK]),
    )
    p_front, e1_front, e2_front = mirror_plane_z0(
        np.array([BH_X_LEFT, BH_Y0, Z_DECK]),
        np.array([BH_X_RIGHT - BH_X_LEFT, 0.0, 0.0]),
        np.array([0.0, 0.0, BH_Z_RIDGE - Z_DECK]),
    )
    p_roof, e1_roof, e2_roof = mirror_plane_z0(
        np.array([BH_X_RIDGE, BH_Y0 - 0.35, BH_Z_RIDGE]),
        np.array([0.0, (BH_Y1 - BH_Y0) + 0.70, 0.0]),
        np.array([(BH_X_RIGHT + BH_EAVE_OVERHANG) - BH_X_RIDGE, 0.0, (BH_Z_EAVE - 0.20) - BH_Z_RIDGE]),
    )

    y_clip = max(0, int(math.floor(hy)) - 2)
    refl = cam.warp_plane_over(refl, bh_front_uv, p_front, e1_front, e2_front, fog_rgb, fog_start=5.5, fog_density=0.036, alpha_scale=0.92, y_min_clip=y_clip)
    refl = cam.warp_plane_over(refl, bh_east_uv, p_east, e1_east, e2_east, fog_rgb, fog_start=5.5, fog_density=0.036, alpha_scale=0.94, y_min_clip=y_clip)
    refl = cam.warp_plane_over(refl, bh_roof_uv, p_roof, e1_roof, e2_roof, fog_rgb, fog_start=5.5, fog_density=0.036, alpha_scale=0.88, y_min_clip=y_clip)

    # Mirror dark timber jetty pilings smoothly underneath z = 0 into the reflection buffer
    s_pil, c_pil = new_cairo_2x()
    for py in (18.4, 16.0, 13.6, 11.2, 8.8, 6.4, 4.2, 2.2):
        seg = cam.project_polyline(np.array([[0.16, py, 0.0], [0.16, py, -Z_DECK * 1.25]]))
        if len(seg) >= 2:
            dist = float(np.linalg.norm(np.array([0.16, py, 0.0]) - cam.eye))
            stroke_catmull_rom(c_pil, [(p[0], p[1]) for p in seg], base_w=17.0,
                               z_depth=dist, rgba=(0.020, 0.036, 0.060, 0.78), rng=rng, taper=False, tremor=0.22)
    refl = over_premul(refl, finish_cairo_2x(s_pil))

    dist_t = np.clip((yy - hy) / max(1.0, H - hy), 0.0, 1.0)
    rip_far = resize_bicubic(rng.standard_normal((H // 3, W // 44)).astype(np.float32), W, H)
    rip_mid = resize_bicubic(rng.standard_normal((H // 6, W // 22)).astype(np.float32), W, H)
    rip_fine = resize_bicubic(rng.standard_normal((H // 2, W // 14)).astype(np.float32), W, H)
    wave = 0.52 * rip_far + 0.34 * rip_mid + 0.14 * rip_fine

    amp_x = 1.8 + 11.0 * np.power(dist_t, 0.85)
    amp_y = 2.4 + 20.0 * np.power(dist_t, 1.15)
    map_x = xx + wave * amp_x
    map_y = np.maximum(hy + 1.0, yy + wave * amp_y)
    refl_rippled = remap_bilinear(refl, map_x, map_y)

    # Shimmering warm amber reflection column from the boathouse window dancing across the dark fjord water!
    wu_c, wv_c, _ = cam.project(np.array([0.55, 0.5 * (WIN_Y0 + WIN_Y1), 0.0]))
    dx_wr = (xx - float(wu_c)) / (78.0 + 135.0 * dist_t)
    dy_wr = np.clip((yy - float(wv_c) + 15.0) / 310.0, 0.0, 2.2)
    win_shimmer = np.exp(-(dx_wr * dx_wr)) * np.exp(-dy_wr * 1.12) * (yy > float(wv_c) - 18.0)
    rip_glint = np.clip(0.45 + 0.90 * wave, 0.0, 1.4)
    refl_rippled += (win_shimmer * rip_glint)[..., None] * np.array([0.44, 0.28, 0.12], dtype=np.float32)

    c_deep = np.array([0.028, 0.058, 0.095], dtype=np.float32)
    c_far = np.array([0.115, 0.210, 0.305], dtype=np.float32)
    water_body = c_far[None, None, :] * (1.0 - dist_t[..., None]) + c_deep[None, None, :] * dist_t[..., None]
    water_body += (wave * 0.012)[..., None]

    fresnel = 0.26 + (0.90 - 0.26) * np.power(1.0 - dist_t, 1.4)
    crest = np.clip(wave * 0.28 + 0.05, -0.18, 0.32) * (0.35 + 0.65 * dist_t)
    water_col = water_body * (1.0 - fresnel[..., None]) + refl_rippled * fresnel[..., None]
    water_col += crest[..., None] * np.array([0.08, 0.14, 0.20], dtype=np.float32)

    hor_mist = np.exp(-((yy - hy) / 14.0) ** 2) * 0.32
    water_col = water_col * (1.0 - hor_mist[..., None]) + np.array([0.52, 0.70, 0.84], dtype=np.float32)[None, None, :] * hor_mist[..., None]

    out = sky_mountain_rgb.copy()
    alpha_w = np.clip((yy - (hy - 1.5)) / 3.0, 0.0, 1.0)
    out = out * (1.0 - alpha_w[..., None]) + water_col * alpha_w[..., None]
    return np.clip(out, 0.0, 1.35)


# =============================================================================
# Stage 3 — Depth-Interleaved 2x PyCairo Pilings, Boathouse Trim, Skiff & Snow
# =============================================================================

def draw_underdeck_pilings(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
) -> None:
    """Band 1a: Timber support pilings plunging cleanly into the dark fjord waterline (no harsh wire ellipse!)."""
    for py in (18.4, 16.0, 13.6, 11.2, 8.8, 6.4, 4.2, 2.2):
        dist = float(np.linalg.norm(np.array([0.16, py, 0.3]) - cam.eye))
        ft = 1.0 - math.exp(-max(0.0, dist - 5.0) * 0.030)
        c_sh = tuple((np.array([0.025, 0.042, 0.068]) * (1.0 - ft) + fog_rgb * ft).tolist())
        c_hi = tuple((np.array([0.092, 0.145, 0.212]) * (1.0 - ft) + fog_rgb * ft).tolist())
        r_p = 0.115
        quad = [
            np.array([0.16 - r_p, py, -0.02]),
            np.array([0.16 + r_p, py, -0.02]),
            np.array([0.16 + r_p, py, Z_DECK + 0.02]),
            np.array([0.16 - r_p, py, Z_DECK + 0.02]),
        ]
        proj = [cam.project(q) for q in quad]
        if all(z >= Z_NEAR for _, _, z in proj):
            g = cairo.LinearGradient(float(proj[0][0]), float(proj[0][1]), float(proj[1][0]), float(proj[1][1]))
            g.add_color_stop_rgba(0.0, *c_sh, 1.0)
            g.add_color_stop_rgba(0.75, *c_hi, 1.0)
            g.add_color_stop_rgba(1.0, *c_sh, 1.0)
            ctx.set_source(g)
            ctx.move_to(float(proj[0][0]), float(proj[0][1]))
            for pu, pv, _ in proj[1:]:
                ctx.line_to(float(pu), float(pv))
            ctx.close_path()
            ctx.fill()

            # Smooth vertical wet tide-line gradient (seamless transition into weathered piling!)
            wet_quad = [
                np.array([0.16 - r_p, py, -0.02]),
                np.array([0.16 + r_p, py, -0.02]),
                np.array([0.16 + r_p, py, 0.30]),
                np.array([0.16 - r_p, py, 0.30]),
            ]
            wproj = [cam.project(q) for q in wet_quad]
            if all(z >= Z_NEAR for _, _, z in wproj):
                wg = cairo.LinearGradient(0, float(wproj[2][1]), 0, float(wproj[0][1]))
                wg.add_color_stop_rgba(0.0, 0.012, 0.024, 0.042, 0.0)
                wg.add_color_stop_rgba(0.65, 0.012, 0.024, 0.042, 0.42)
                wg.add_color_stop_rgba(1.0, 0.008, 0.018, 0.034, 0.72)
                ctx.set_source(wg)
                ctx.move_to(float(wproj[0][0]), float(wproj[0][1]))
                for pu, pv, _ in wproj[1:]:
                    ctx.line_to(float(pu), float(pv))
                ctx.close_path()
                ctx.fill()

            ru, rv, rz = cam.project(np.array([0.16, py, 0.0]))
            if rz >= Z_NEAR:
                rx = max(6.0, 230.0 / rz)
                for dy_w, sc_w, al_w in ((0.0, 0.95, 0.24), (3.0, 1.25, 0.14)):
                    ctx.set_source_rgba(0.56, 0.74, 0.88, al_w)
                    ctx.set_line_width(max(0.8, 1.6 / (1.0 + dist * 0.1)))
                    ctx.move_to(float(ru) - rx * sc_w, float(rv) + dy_w)
                    ctx.line_to(float(ru) + rx * sc_w, float(rv) + dy_w)
                    ctx.stroke()


def draw_boathouse_trim_and_lantern(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
) -> None:
    """Band 1b: Jetty outer fender beams, boathouse bargeboards, chimney, porch lantern & window bloom."""
    def fog_c(c3, dist: float):
        ft = 1.0 - math.exp(-max(0.0, dist - 5.0) * 0.030)
        c = np.asarray(c3, dtype=np.float64) * (1.0 - ft) + fog_rgb * ft
        return (float(c[0]), float(c[1]), float(c[2]))

    for z_beam, col_b in ((Z_DECK - 0.06, [0.068, 0.105, 0.155]), (Z_DECK - 0.20, [0.036, 0.060, 0.095])):
        seg = cam.project_polyline(np.array([[0.26, 1.2, z_beam], [0.26, 19.2, z_beam]]))
        if len(seg) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in seg], base_w=8.5, z_depth=7.0,
                               rgba=(*col_b, 0.96), rng=rng, taper=False, tremor=0.20)

    dist_bh = float(np.linalg.norm(np.array([BH_X_RIGHT, BH_Y0, BH_Z_EAVE]) - cam.eye))
    trim_col = fog_c([0.18, 0.25, 0.34], dist_bh)
    frost_col = fog_c([0.64, 0.80, 0.92], dist_bh)

    p_eave_l = np.array([BH_X_LEFT - BH_EAVE_OVERHANG, BH_Y0 - 0.36, BH_Z_EAVE - 0.20])
    p_ridge_f = np.array([BH_X_RIDGE, BH_Y0 - 0.36, BH_Z_RIDGE])
    p_eave_r = np.array([BH_X_RIGHT + BH_EAVE_OVERHANG, BH_Y0 - 0.36, BH_Z_EAVE - 0.20])
    p_eave_rb = np.array([BH_X_RIGHT + BH_EAVE_OVERHANG, BH_Y1 + 0.36, BH_Z_EAVE - 0.20])
    p_ridge_b = np.array([BH_X_RIDGE, BH_Y1 + 0.36, BH_Z_RIDGE])

    for p_a, p_b, bw in (
        (p_eave_l, p_ridge_f, 7.5),
        (p_ridge_f, p_eave_r, 8.0),
        (p_eave_r, p_eave_rb, 6.2),
        (p_ridge_f, p_ridge_b, 5.8),
    ):
        seg = cam.project_polyline(np.array([p_a, p_b]))
        if len(seg) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in seg], base_w=bw, z_depth=dist_bh,
                               rgba=(*trim_col, 0.98), rng=rng, taper=False, tremor=0.16)
            seg_hi = cam.project_polyline(np.array([p_a + [0, 0, 0.04], p_b + [0, 0, 0.04]]))
            if len(seg_hi) >= 2:
                stroke_catmull_rom(ctx, [(p[0], p[1]) for p in seg_hi], base_w=bw * 0.36, z_depth=dist_bh,
                                   rgba=(*frost_col, 0.72), rng=rng, taper=False, tremor=0.18)

    ch_base = np.array([BH_X_RIDGE + 0.35, 11.6, 5.15])
    ch_top = np.array([BH_X_RIDGE + 0.35, 11.6, 6.12])
    cseg = cam.project_polyline(np.array([ch_base, ch_top]))
    if len(cseg) >= 2:
        stroke_catmull_rom(ctx, [(p[0], p[1]) for p in cseg], base_w=7.5, z_depth=dist_bh + 4.0,
                           rgba=(*fog_c([0.05, 0.08, 0.12], dist_bh + 4.0), 0.96), rng=rng, taper=False, tremor=0.10)
        u_t, v_t, _ = cam.project(ch_top)
        ctx.set_source_rgba(*fog_c([0.14, 0.21, 0.28], dist_bh + 4.0), 0.95)
        ctx.set_line_width(3.0)
        ctx.move_to(float(u_t) - 8, float(v_t) - 3)
        ctx.line_to(float(u_t) + 8, float(v_t) - 3)
        ctx.stroke()

    # Porch lantern bracket & glowing pale glacier-blue fixture beside boathouse front doors
    p_lan = np.array([-2.36, BH_Y0 - 0.16, 2.66])
    lu, lv, lz = cam.project(p_lan)
    if lz >= Z_NEAR:
        lg = cairo.RadialGradient(float(lu), float(lv), 2.0, float(lu), float(lv), 56.0)
        lg.add_color_stop_rgba(0.0, 0.88, 0.97, 1.0, 0.94)
        lg.add_color_stop_rgba(0.28, 0.56, 0.80, 0.96, 0.44)
        lg.add_color_stop_rgba(1.0, 0.38, 0.64, 0.86, 0.0)
        ctx.set_source(lg)
        ctx.arc(float(lu), float(lv), 56.0, 0, math.tau)
        ctx.fill()
        ctx.set_source_rgba(0.04, 0.06, 0.10, 0.98)
        ctx.move_to(float(lu) - 8, float(lv) - 6)
        ctx.line_to(float(lu) + 8, float(lv) - 6)
        ctx.line_to(float(lu) + 4, float(lv) - 12)
        ctx.line_to(float(lu) - 4, float(lv) - 12)
        ctx.close_path()
        ctx.fill()
        ctx.set_source_rgba(0.92, 0.98, 1.0, 0.98)
        ctx.arc(float(lu), float(lv), 4.8, 0, math.tau)
        ctx.fill()

    wu, wv, wz = cam.project(np.array([BH_X_RIGHT + 0.04, 0.5 * (WIN_Y0 + WIN_Y1), 0.5 * (WIN_Z0 + WIN_Z1)]))
    if wz >= Z_NEAR:
        wg = cairo.RadialGradient(float(wu), float(wv), 10.0, float(wu), float(wv), 155.0)
        wg.add_color_stop_rgba(0.0, 1.0, 0.84, 0.54, 0.25)
        wg.add_color_stop_rgba(0.5, 0.96, 0.70, 0.36, 0.09)
        wg.add_color_stop_rgba(1.0, 0.88, 0.56, 0.24, 0.0)
        ctx.set_source(wg)
        ctx.arc(float(wu), float(wv), 155.0, 0, math.tau)
        ctx.fill()


def draw_nordic_wooden_skiff(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
) -> None:
    """
    Band 2 Hero Prop: Hand-crafted Norwegian clinker wooden skiff (faering) moored
    gently beside the jetty with mathematically verified near/far hull occlusion,
    solid clinker-planked outer shell, interior thwarts & resting oars, mirrored water
    reflection, soft broken waterline ripples, and a clean catenary mooring rope
    running directly from the bow ring to the nearest jetty bollard!
    """
    cx, cy = 1.82, 8.55
    # Yaw = +26 deg: bow (s = +1) points forward-left toward the jetty bollard at (0.06, 11.8)
    # Wait: let's check bow position!
    # lx = 0, ly = +2.15 -> wx = cx - ly*sin(yaw), wy = cy + ly*cos(yaw)
    # For yaw = +26 deg: wx = 1.82 - 2.15*0.438 = 0.88, wy = 8.55 + 2.15*0.899 = 10.48 -> RIGHT NEXT TO BOLLARD (0.06, 11.8)!
    yaw = math.radians(26.0)
    length = 4.30
    beam = 1.32

    def boat_world(s: float, side: float, z: float, flare: float = 1.0, add_sheer: bool = True, len_scale: float = 1.0) -> np.ndarray:
        """s in [-1 (stern), +1 (bow)], side in [-1, +1], z above waterline."""
        half_b = (beam * 0.5) * math.pow(max(0.0, 1.0 - s * s), 0.72) * flare
        lx = side * half_b
        ly = s * (length * 0.5 * len_scale)
        sheer = (0.22 * math.pow(abs(s), 2.2)) if add_sheer else 0.0
        wx = cx + lx * math.cos(yaw) - ly * math.sin(yaw)
        wy = cy + lx * math.sin(yaw) + ly * math.cos(yaw)
        return np.array([wx, wy, z + sheer], dtype=np.float64)

    # Determine mathematically which side (+1 vs -1) faces the camera so outer hull ALWAYS occludes interior!
    d_pos = float(np.linalg.norm(boat_world(0.0, 1.0, 0.25) - cam.eye))
    d_neg = float(np.linalg.norm(boat_world(0.0, -1.0, 0.25) - cam.eye))
    near_side = 1.0 if d_pos < d_neg else -1.0
    far_side = -near_side

    dist = float(np.linalg.norm(np.array([cx, cy, 0.2]) - cam.eye))
    ft = 1.0 - math.exp(-max(0.0, dist - 5.0) * 0.030)

    def fog_c(c3):
        c = np.asarray(c3, dtype=np.float64) * (1.0 - ft) + fog_rgb * ft
        return (float(c[0]), float(c[1]), float(c[2]))

    def proj_curve(pts3: list[np.ndarray]) -> list[tuple[float, float]]:
        out = []
        for p in pts3:
            u, v, z = cam.project(p)
            if z >= Z_NEAR:
                out.append((float(u), float(v)))
        return out

    ss = np.linspace(-1.0, 1.0, 31)
    gun_near = [boat_world(float(s), near_side, 0.44, flare=1.0, add_sheer=True, len_scale=1.0) for s in ss]
    gun_far = [boat_world(float(s), far_side, 0.44, flare=1.0, add_sheer=True, len_scale=1.0) for s in ss]
    # True curved Norwegian waterline at z = 0.0 (curved inward at stem & sternpost so stern tapers gracefully!)
    water_near = [boat_world(float(s), near_side, 0.0, flare=0.60, add_sheer=False, len_scale=0.80) for s in ss]

    # 1. Mirrored dark hull reflection underneath z = 0 in the fjord water
    refl_bot = []
    for p in gun_near:
        pm = p.copy()
        pm[2] = -p[2] * 0.88
        refl_bot.append(pm)
    refl_poly = proj_curve(water_near + list(reversed(refl_bot)))
    if len(refl_poly) > 4:
        ys_r = [p[1] for p in refl_poly]
        gr = cairo.LinearGradient(0, min(ys_r), 0, max(ys_r))
        gr.add_color_stop_rgba(0.0, 0.018, 0.036, 0.062, 0.66)
        gr.add_color_stop_rgba(1.0, 0.022, 0.044, 0.074, 0.12)
        ctx.set_source(gr)
        ctx.move_to(*refl_poly[0])
        for p in refl_poly[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.fill()

    # Subtle continuous waterline highlight hugging the hull entry into the glassy fjord
    w_proj = proj_curve(water_near)
    if len(w_proj) >= 2:
        stroke_catmull_rom(ctx, w_proj, base_w=1.8, z_depth=dist, rgba=(0.56, 0.74, 0.88, 0.30), rng=rng, taper=True, tremor=0.14)
        w_rip2 = [(px + float(rng.uniform(-4, 4)), py + 4.5) for px, py in w_proj[2:-2]]
        if len(w_rip2) >= 2:
            stroke_catmull_rom(ctx, w_rip2, base_w=1.1, z_depth=dist, rgba=(0.52, 0.70, 0.84, 0.16), rng=rng, taper=True, tremor=0.22)

    # 2. Interior hull cavity (far inner strakes visible inside the open wooden boat!)
    inner_poly = proj_curve(gun_far + list(reversed(gun_near)))
    if len(inner_poly) > 4:
        ys_i = [p[1] for p in inner_poly]
        c_in_top = fog_c([0.115, 0.130, 0.155])
        c_in_bot = fog_c([0.036, 0.052, 0.075])
        gi = cairo.LinearGradient(0, min(ys_i), 0, max(ys_i))
        gi.add_color_stop_rgba(0.0, *c_in_top, 1.0)
        gi.add_color_stop_rgba(1.0, *c_in_bot, 1.0)
        ctx.set_source(gi)
        ctx.move_to(*inner_poly[0])
        for p in inner_poly[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.fill()

    # Curved wooden interior ribs along the visible far inner hull
    for s_rib in np.linspace(-0.68, 0.68, 7):
        rib3 = [
            boat_world(float(s_rib), far_side, 0.42, flare=0.96),
            boat_world(float(s_rib), far_side * 0.65, 0.16, flare=0.80, add_sheer=False, len_scale=0.88),
            boat_world(float(s_rib), 0.0, 0.06, flare=0.65, add_sheer=False, len_scale=0.82),
        ]
        rp = proj_curve(rib3)
        if len(rp) >= 2:
            stroke_catmull_rom(ctx, rp, base_w=2.0, z_depth=dist, rgba=(*fog_c([0.05, 0.07, 0.10]), 0.68), rng=rng, taper=False, tremor=0.10)

    # 3. Wooden thwarts (benches) resting solidly across the interior from far_side to near_side!
    for s_th in (-0.44, 0.02, 0.46):
        th_pts = [
            boat_world(s_th - 0.08, far_side * 0.94, 0.31, flare=0.95),
            boat_world(s_th - 0.08, near_side * 0.94, 0.31, flare=0.95),
            boat_world(s_th + 0.08, near_side * 0.94, 0.31, flare=0.95),
            boat_world(s_th + 0.08, far_side * 0.94, 0.31, flare=0.95),
        ]
        tp = proj_curve(th_pts)
        if len(tp) == 4:
            ctx.set_source_rgba(*fog_c([0.165, 0.188, 0.220]), 0.98)
            ctx.move_to(*tp[0])
            for p in tp[1:]:
                ctx.line_to(*p)
            ctx.close_path()
            ctx.fill()
            ctx.set_source_rgba(*fog_c([0.42, 0.56, 0.68]), 0.58)
            ctx.set_line_width(1.3)
            ctx.move_to(*tp[0])
            ctx.line_to(*tp[1])
            ctx.stroke()

    # Resting wooden oars nestled inside across the benches!
    for oar_a, oar_b in (
        (boat_world(-0.62, far_side * 0.25, 0.36), boat_world(0.52, near_side * 0.18, 0.38)),
        (boat_world(-0.54, far_side * 0.48, 0.35), boat_world(0.46, far_side * 0.10, 0.37)),
    ):
        oseg = cam.project_polyline(np.array([oar_a, oar_b]))
        if len(oseg) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in oseg], base_w=4.0, z_depth=dist,
                               rgba=(*fog_c([0.24, 0.26, 0.29]), 0.95), rng=rng, taper=False, tremor=0.10)

    # 4. SOLID CAMERA-FACING OUTER CLINKER HULL SHELL (tapered gracefully from gunwale down to curved waterline!)
    outer_poly = proj_curve(gun_near + list(reversed(water_near)))
    if len(outer_poly) > 4:
        ys_h = [p[1] for p in outer_poly]
        c_top = fog_c([0.155, 0.178, 0.215])
        c_mid = fog_c([0.088, 0.108, 0.142])
        c_bot = fog_c([0.028, 0.048, 0.078])
        gh = cairo.LinearGradient(0, min(ys_h), 0, max(ys_h))
        gh.add_color_stop_rgba(0.0, *c_top, 1.0)
        gh.add_color_stop_rgba(0.48, *c_mid, 1.0)
        gh.add_color_stop_rgba(1.0, *c_bot, 1.0)
        ctx.set_source(gh)
        ctx.move_to(*outer_poly[0])
        for p in outer_poly[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.fill()

        # Overlapping clinker strake laps interpolated cleanly between waterline & gunwale!
        for frac_z in (0.26, 0.52, 0.76):
            strake = [water_near[k] * (1.0 - frac_z) + gun_near[k] * frac_z for k in range(len(ss))]
            sp = proj_curve(strake)
            if len(sp) >= 2:
                stroke_catmull_rom(ctx, sp, base_w=1.7, z_depth=dist,
                                   rgba=(*fog_c([0.018, 0.032, 0.056]), 0.82), rng=rng, taper=True, tremor=0.14)
                sp_hi = [(px, py - 1.2) for px, py in sp]
                stroke_catmull_rom(ctx, sp_hi, base_w=0.95, z_depth=dist,
                                   rgba=(*fog_c([0.28, 0.40, 0.52]), 0.42), rng=rng, taper=True, tremor=0.12)

        # Warm amber reflection glaze from the boathouse window touching the wet upper hull planks
        glint_pts = proj_curve([water_near[k] * 0.22 + gun_near[k] * 0.78 for k in range(10, 26)])
        if len(glint_pts) >= 2:
            stroke_catmull_rom(ctx, glint_pts, base_w=2.8, z_depth=dist, rgba=(0.86, 0.62, 0.34, 0.54), rng=rng, taper=True, tremor=0.14)

    # 5. Crisp frost-kissed gunwale timbers along BOTH far & near edges + curved bow/stern stemposts
    for side_c, w_g, col_g in (
        (gun_far, 3.0, fog_c([0.48, 0.64, 0.78])),
        (gun_near, 3.8, fog_c([0.38, 0.52, 0.66])),
    ):
        gp = proj_curve(side_c)
        if len(gp) >= 2:
            stroke_catmull_rom(ctx, gp, base_w=w_g, z_depth=dist, rgba=(*col_g, 0.94), rng=rng, taper=False, tremor=0.12)

    # 6. Clean sagging 3D catenary hemp mooring rope tied DEAD-ON to the bow stem ring (s = +0.99)!
    p_bow = gun_near[-1].copy()
    p_bol = np.array([0.06, 11.80, Z_DECK + 0.30], dtype=np.float64)
    rope_pts = []
    for k in range(19):
        t = k / 18.0
        p = p_bow * (1.0 - t) + p_bol * t
        p[2] -= 0.28 * math.sin(t * math.pi)
        rope_pts.append(p)
    rseg = cam.project_polyline(np.array(rope_pts))
    if len(rseg) >= 2:
        stroke_catmull_rom(ctx, [(p[0], p[1]) for p in rseg], base_w=2.6, z_depth=dist,
                           rgba=(*fog_c([0.28, 0.32, 0.36]), 0.92), rng=rng, taper=False, tremor=0.12)


def draw_foreground_bollards_and_snow(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
) -> None:
    """Band 3: Weathered wooden mooring bollards with frost caps, coiled rope, edge highlights & drifting snow."""
    def fog_c(c3, dist: float):
        ft = 1.0 - math.exp(-max(0.0, dist - 5.0) * 0.030)
        c = np.asarray(c3, dtype=np.float64) * (1.0 - ft) + fog_rgb * ft
        return (float(c[0]), float(c[1]), float(c[2]))

    for by in (15.8, 11.8, 7.6, 3.7):
        bx = 0.06
        dist = float(np.linalg.norm(np.array([bx, by, Z_DECK + 0.3]) - cam.eye))
        bw = 0.115
        bh = 0.56 if by > 4.0 else 0.62
        p_fl = np.array([bx - bw, by - bw, Z_DECK])
        p_fr = np.array([bx + bw, by - bw, Z_DECK])
        p_br = np.array([bx + bw, by + bw, Z_DECK])
        p_bl = np.array([bx - bw, by + bw, Z_DECK])
        t_fl, t_fr, t_br, t_bl = [p + np.array([0, 0, bh]) for p in (p_fl, p_fr, p_br, p_bl)]

        faces = [
            ([p_fl, p_fr, t_fr, t_fl], fog_c([0.052, 0.078, 0.112], dist)),
            ([p_fr, p_br, t_br, t_fr], fog_c([0.115, 0.165, 0.228], dist)),
            ([t_fl, t_fr, t_br, t_bl], fog_c([0.52, 0.68, 0.80], dist)),
        ]
        for pts3, col in faces:
            proj = [cam.project(p) for p in pts3]
            if any(z < Z_NEAR for _, _, z in proj):
                continue
            ctx.set_source_rgba(*col, 0.98)
            ctx.move_to(float(proj[0][0]), float(proj[0][1]))
            for pu, pv, _ in proj[1:]:
                ctx.line_to(float(pu), float(pv))
            ctx.close_path()
            ctx.fill()

        for k_r in range(3):
            zr = Z_DECK + bh * (0.48 + k_r * 0.10)
            seg = cam.project_polyline(np.array([[bx - bw * 1.08, by - bw * 1.05, zr], [bx + bw * 1.08, by - bw * 1.05, zr], [bx + bw * 1.08, by + bw * 1.05, zr]]))
            if len(seg) >= 2:
                stroke_catmull_rom(ctx, [(p[0], p[1]) for p in seg], base_w=4.2, z_depth=dist,
                                   rgba=(*fog_c([0.28, 0.30, 0.32], dist), 0.95), rng=rng, taper=False, tremor=0.14)

    coil_c = np.array([-0.55, 3.85, Z_DECK + 0.02])
    dist_c = float(np.linalg.norm(coil_c - cam.eye))
    for loop_i in range(4):
        rad = 0.14 + loop_i * 0.055
        pts_l = []
        for k in range(25):
            a = k * math.tau / 24.0
            p = coil_c + np.array([math.cos(a) * rad, math.sin(a) * rad * 1.12, 0.015 * math.sin(a * 2.0)])
            u, v, z = cam.project(p)
            if z >= Z_NEAR:
                pts_l.append((float(u), float(v)))
        if len(pts_l) > 3:
            stroke_catmull_rom(ctx, pts_l, base_w=5.0, z_depth=dist_c,
                               rgba=(*fog_c([0.22, 0.24, 0.28], dist_c), 0.90), rng=rng, taper=False, tremor=0.18)

    for p0, p1, w_l in (
        (np.array([0.25, 1.3, Z_DECK]), np.array([0.25, 19.1, Z_DECK]), 2.2),
        (np.array([BH_X_RIGHT, BH_Y0, Z_DECK]), np.array([BH_X_RIGHT, BH_Y0, BH_Z_EAVE]), 2.6),
    ):
        seg = cam.project_polyline(np.array([p0, p1]))
        if len(seg) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in seg], base_w=w_l, z_depth=6.5,
                               rgba=(0.48, 0.66, 0.80, 0.55), rng=rng, taper=False, tremor=0.16)

    for _ in range(135):
        sx = float(rng.uniform(0, W))
        sy = float(rng.uniform(0, H * 0.92))
        rad = float(rng.uniform(0.9, 2.7))
        alpha = float(rng.uniform(0.14, 0.50))
        warm = math.exp(-(((sx - 740.0) / 260.0) ** 2 + ((sy - 510.0) / 190.0) ** 2))
        r_c = 0.78 + 0.22 * warm
        g_c = 0.88 + 0.04 * warm
        b_c = 0.98 - 0.28 * warm
        sg = cairo.RadialGradient(sx, sy, 0.0, sx, sy, rad * 1.9)
        sg.add_color_stop_rgba(0.0, r_c, g_c, b_c, alpha)
        sg.add_color_stop_rgba(1.0, r_c, g_c, b_c, 0.0)
        ctx.set_source(sg)
        ctx.arc(sx, sy, rad * 1.9, 0, math.tau)
        ctx.fill()


# =============================================================================
# Stage 4 — Foreground Circular Disc Bokeh Pine Branch & 35mm Analog Finish
# =============================================================================

def make_disc_kernel(radius: float = 7.8) -> np.ndarray:
    rad_i = int(math.ceil(radius + 1.5))
    yy, xx = np.mgrid[-rad_i : rad_i + 1, -rad_i : rad_i + 1].astype(np.float32)
    r = np.hypot(xx, yy)
    disc = np.clip((radius - r + 0.65) / 1.15, 0.0, 1.0)
    rim = np.exp(-((r - radius * 0.86) / (radius * 0.16)) ** 2) * 0.24 * disc
    k = disc + rim
    k /= float(k.sum())
    return k.astype(np.float32)


def convolve_fft_2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    ih, iw = img.shape[:2]
    kh, kw = kernel.shape
    fh = 1 << int(math.ceil(math.log2(ih + kh - 1)))
    fw = 1 << int(math.ceil(math.log2(iw + kw - 1)))
    k_pad = np.zeros((fh, fw), dtype=np.float32)
    k_pad[:kh, :kw] = kernel
    k_pad = np.roll(np.roll(k_pad, -(kh // 2), axis=0), -(kw // 2), axis=1)
    K = np.fft.rfft2(k_pad)
    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        p = np.zeros((fh, fw), dtype=np.float32)
        p[:ih, :iw] = img[:, :, c]
        out[:, :, c] = np.fft.irfft2(np.fft.rfft2(p) * K, s=(fh, fw))[:ih, :iw]
    return out


def paint_foreground_bokeh_pine(rng: np.random.Generator) -> np.ndarray:
    """Near-camera Nordic pine branch silhouette framing the upper-right corner with circular disc bokeh."""
    surf, ctx = new_cairo_2x()
    branches = [
        ([(W + 30, 32), (W - 140, 86), (W - 315, 134), (W - 465, 198)], 11.5),
        ([(W - 125, 82), (W - 215, 178), (W - 310, 262)], 7.2),
        ([(W - 235, 114), (W - 365, 96), (W - 495, 118)], 5.8),
        ([(W + 20, 175), (W - 92, 224), (W - 212, 295)], 8.4),
        ([(W - 75, 220), (W - 142, 322), (W - 198, 396)], 5.5),
    ]
    for bpts, bw in branches:
        stroke_catmull_rom(ctx, bpts, base_w=bw, z_depth=1.0, rgba=(0.015, 0.030, 0.052, 0.98), rng=rng, taper=True, tremor=0.55)
        for i in range(len(bpts) - 1):
            x0, y0 = bpts[i]
            x1, y1 = bpts[i + 1]
            ang_b = math.atan2(y1 - y0, x1 - x0)
            n_clusters = 12
            for k in range(n_clusters):
                t = (k + float(rng.uniform(0.0, 0.8))) / n_clusters
                cx = x0 + (x1 - x0) * t
                cy = y0 + (y1 - y0) * t
                for _ in range(int(rng.integers(6, 10))):
                    side = 1.0 if rng.random() > 0.45 else -1.0
                    na = ang_b + side * float(rng.uniform(0.32, 1.48))
                    nlen = float(rng.uniform(26.0, 68.0))
                    droop = float(rng.uniform(4.0, 18.0))
                    ex = cx + math.cos(na) * nlen
                    ey = cy + math.sin(na) * nlen + droop
                    mx = cx + math.cos(na) * (nlen * 0.55)
                    my = cy + math.sin(na) * (nlen * 0.55) + droop * 0.35
                    ctx.set_source_rgba(0.016, 0.038, 0.062, float(rng.uniform(0.86, 0.98)))
                    ctx.set_line_width(float(rng.uniform(2.2, 4.0)))
                    ctx.move_to(cx, cy)
                    ctx.curve_to(mx, my, (mx + ex) * 0.5, (my + ey) * 0.5, ex, ey)
                    ctx.stroke()
                    if ey < cy + 6.0 and rng.random() < 0.34:
                        ctx.set_source_rgba(0.46, 0.66, 0.82, 0.54)
                        ctx.set_line_width(1.6)
                        ctx.move_to(cx, cy - 1.0)
                        ctx.curve_to(mx, my - 1.5, (mx + ex) * 0.5, (my + ey) * 0.5 - 1.5, ex, ey - 1.0)
                        ctx.stroke()

    fg_1x = finish_cairo_2x(surf)
    blurred = convolve_fft_2d(fg_1x, make_disc_kernel(radius=7.8))
    out = blurred.copy()
    out[:, 1:, 0] = blurred[:, :-1, 0]
    out[:, :-1, 2] = blurred[:, 1:, 2]
    return np.clip(out, 0.0, 1.0)


def apply_35mm_finish(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = rgb.astype(np.float32).copy()

    lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    hl_mask = np.clip((lum - 0.72) / 0.28, 0.0, 1.6)
    hl_src = img * hl_mask[..., None]
    tint = np.array([1.02, 1.00, 1.04], dtype=np.float32)
    hal = (
        0.24 * gaussian_blur_2d(hl_src, 6.5)
        + 0.16 * gaussian_blur_2d(hl_src, 24.0)
        + 0.10 * gaussian_blur_2d(hl_src, 72.0)
    ) * tint[None, None, :]
    img = 1.0 - (1.0 - np.clip(img, 0.0, 1.0)) * (1.0 - np.clip(hal, 0.0, 0.85))

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (xx - W * 0.5) / (W * 0.5)
    ny = (yy - H * 0.5) / (H * 0.5)
    r2 = nx * nx + ny * ny
    ca_scale = 0.00095
    img[:, :, 0] = remap_bilinear(img[:, :, 0], xx + nx * r2 * (W * ca_scale), yy + ny * r2 * (H * ca_scale))
    img[:, :, 2] = remap_bilinear(img[:, :, 2], xx - nx * r2 * (W * ca_scale), yy - ny * r2 * (H * ca_scale))
    img *= np.clip(1.0 - 0.19 * (r2 ** 1.15), 0.74, 1.0)[..., None]

    img = 0.012 + img * 0.988
    knee, ceiling = 0.80, 0.965
    span = ceiling - knee
    img = np.where(img > knee, knee + span * np.tanh((img - knee) / span), img)

    lum2 = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    env = np.clip(1.12 - 0.95 * np.power(np.clip(lum2, 0.0, 1.0), 1.55), 0.12, 1.12)
    grain = rng.normal(0.0, 0.0095, size=(H, W, 3)).astype(np.float32)
    mono_g = rng.normal(0.0, 0.0082, size=(H, W, 1)).astype(np.float32)
    img = np.clip(img + (0.68 * mono_g + 0.32 * grain) * env[..., None], 0.0, 1.0)

    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    I = 0.596 * R - 0.274 * G - 0.322 * B
    Q = 0.211 * R - 0.523 * G + 0.312 * B

    I_soft = gaussian_blur_1d(I, 1.0, axis=1)
    Q_soft = gaussian_blur_1d(Q, 1.0, axis=1)
    Y_ring = Y + 0.036 * (Y - gaussian_blur_1d(Y, 1.1, axis=1))

    out = np.empty_like(img)
    out[:, :, 0] = Y_ring + 0.956 * I_soft + 0.621 * Q_soft
    out[:, :, 1] = Y_ring - 0.272 * I_soft - 0.647 * Q_soft
    out[:, :, 2] = Y_ring - 1.106 * I_soft + 1.703 * Q_soft

    line_shift = gaussian_blur_1d(rng.normal(0.0, 0.16, size=(H, 1)).astype(np.float32), 1.2, axis=0)
    jx = np.broadcast_to(np.arange(W, dtype=np.float32)[None, :], (H, W)) + line_shift
    jy = np.broadcast_to(np.arange(H, dtype=np.float32)[:, None], (H, W))
    out = remap_bilinear(out, jx, jy)
    out *= (1.0 - 0.0065 * (0.5 - 0.5 * np.cos(np.arange(H, dtype=np.float32) * math.pi)))[:, None, None]
    return np.clip(out, 0.0, 1.0)


# =============================================================================
# Full Scene Assembly & Deterministic Export
# =============================================================================

def render_scene(
    seed: int = 202601,
    out_dir: pathlib.Path | None = None,
    webp_targets: list[pathlib.Path] | None = None,
) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    cam = Camera3D(W, H, eye=(1.15, -0.65, 2.38), target=(-1.05, 16.0, 1.82), fov_x_deg=56.0)
    print(f"[fjord-01] Camera VP: ({cam.vp_x:.1f}, {cam.vp_y:.1f}) -> {cam.vp_x / W * 100:.1f}% width, {cam.vp_y / H * 100:.1f}% height")

    # Stage 1 & 3 Band 0: Blue-hour Nordic sky + craggy glaciated basalt fjord cliffs
    sky_mtn, fog_rgb, mottle_full, _, _ = paint_sky_and_mountains(cam, rng)

    # Planar architectural UV textures (premultiplied RGBA)
    quay_curb_uv = paint_voronoi_basalt_uv(rng, (1.0, 7.15), (Z_DECK - 0.02, Z_QUAY), is_top=False, tw=640, th=192)
    quay_top_uv = paint_voronoi_basalt_uv(rng, (-5.65, -1.92), (1.0, 7.15), is_top=True, tw=768, th=640)
    jetty_uv = paint_wet_jetty_deck_uv(rng, (-1.92, 0.26), (1.15, 19.2), tw=768, th=1152)
    bh_east_uv = paint_boathouse_east_wall_uv(rng)
    bh_front_uv = paint_boathouse_gable_front_uv(rng)
    bh_roof_uv = paint_boathouse_roof_uv(rng)

    # Stage 2: Fresnel capillary-rippled fjord water + exact 3D plane-mirrored boathouse & pilings
    canvas = paint_fjord_water(cam, sky_mtn, bh_east_uv, bh_front_uv, bh_roof_uv, fog_rgb, rng)

    # Stage 3 Band 1a: Below-deck timber pilings rendered underneath the jetty deck
    s_pil, c_pil = new_cairo_2x()
    draw_underdeck_pilings(c_pil, cam, rng, fog_rgb)
    canvas = over_premul(canvas, finish_cairo_2x(s_pil))

    # Stage 1 Exact 3x3 Plane Homographies (back-to-front painter's order!)
    canvas = cam.warp_plane_over(
        canvas, jetty_uv,
        p0=np.array([-1.92, 1.15, Z_DECK]), e1=np.array([2.18, 0.0, 0.0]), e2=np.array([0.0, 18.05, 0.0]),
        fog_rgb=fog_rgb, fog_start=5.0, fog_density=0.030,
    )
    # Raised weathered Voronoi basalt stone quay apron & stone curb receding along the left foreground!
    canvas = cam.warp_plane_over(
        canvas, quay_curb_uv,
        p0=np.array([-1.92, 1.0, Z_DECK - 0.02]), e1=np.array([0.0, 6.15, 0.0]), e2=np.array([0.0, 0.0, Z_QUAY - (Z_DECK - 0.02)]),
        fog_rgb=fog_rgb, fog_start=5.0, fog_density=0.030,
    )
    canvas = cam.warp_plane_over(
        canvas, quay_top_uv,
        p0=np.array([-5.65, 1.0, Z_QUAY]), e1=np.array([3.73, 0.0, 0.0]), e2=np.array([0.0, 6.15, 0.0]),
        fog_rgb=fog_rgb, fog_start=5.0, fog_density=0.030,
    )
    # Norwegian dark-timber boathouse east wall, front gable wall & pitched slate roof
    canvas = cam.warp_plane_over(
        canvas, bh_east_uv,
        p0=np.array([BH_X_RIGHT, BH_Y0, Z_DECK]), e1=np.array([0.0, BH_Y1 - BH_Y0, 0.0]), e2=np.array([0.0, 0.0, BH_Z_EAVE - Z_DECK]),
        fog_rgb=fog_rgb, fog_start=5.0, fog_density=0.030,
    )
    canvas = cam.warp_plane_over(
        canvas, bh_front_uv,
        p0=np.array([BH_X_LEFT, BH_Y0, Z_DECK]), e1=np.array([BH_X_RIGHT - BH_X_LEFT, 0.0, 0.0]), e2=np.array([0.0, 0.0, BH_Z_RIDGE - Z_DECK]),
        fog_rgb=fog_rgb, fog_start=5.0, fog_density=0.030,
    )
    canvas = cam.warp_plane_over(
        canvas, bh_roof_uv,
        p0=np.array([BH_X_RIDGE, BH_Y0 - 0.35, BH_Z_RIDGE]),
        e1=np.array([0.0, (BH_Y1 - BH_Y0) + 0.70, 0.0]),
        e2=np.array([(BH_X_RIGHT + BH_EAVE_OVERHANG) - BH_X_RIDGE, 0.0, (BH_Z_EAVE - 0.20) - BH_Z_RIDGE]),
        fog_rgb=fog_rgb, fog_start=5.0, fog_density=0.030,
    )

    # Stage 3 Band 2: Boathouse architectural trim, porch lantern & hero moored wooden clinker skiff
    s_mid, c_mid = new_cairo_2x()
    draw_boathouse_trim_and_lantern(c_mid, cam, rng, fog_rgb)
    draw_nordic_wooden_skiff(c_mid, cam, rng, fog_rgb)
    mid_1x = finish_cairo_2x(s_mid)
    ma = mid_1x[:, :, 3:4]
    mid_1x[:, :, :3] = np.clip(mid_1x[:, :, :3] + ma * (mottle_full * 0.013)[..., None], 0.0, 1.25)
    canvas = over_premul(canvas, mid_1x)

    # Stage 3 Band 3: Foreground mooring bollards, coiled rope, edge highlights & drifting blue-hour snow motes
    s_near, c_near = new_cairo_2x()
    draw_foreground_bollards_and_snow(c_near, cam, rng, fog_rgb)
    near_1x = finish_cairo_2x(s_near)
    na = near_1x[:, :, 3:4]
    near_1x[:, :, :3] = np.clip(near_1x[:, :, :3] + na * (mottle_full * 0.012)[..., None], 0.0, 1.25)
    canvas = over_premul(canvas, near_1x)

    # Stage 4: Upper-right foreground circular-bokeh Nordic pine branch silhouette + 35mm analog optics
    fg_pine = paint_foreground_bokeh_pine(rng)
    canvas = over_premul(canvas, fg_pine)
    final_rgb = apply_35mm_finish(canvas, rng)

    elapsed = time.perf_counter() - t0
    print(f"[fjord-01] Rendered {W}x{H} in {elapsed:.2f}s")

    u8 = np.round(np.clip(final_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    full_im = Image.fromarray(u8, mode="RGB")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        full_im.save(out_dir / "w01_full.png", optimize=True)
        crops = {
            "w01_crop_aperture.png": (int(W * 0.20), int(H * 0.28)),     # Boathouse window glow + gable + roof + jetty
            "w01_crop_midground.png": (int(W * 0.48), int(H * 0.46)),    # Hero wooden clinker skiff + mooring rope + Fresnel fjord reflections
            "w01_crop_foreground.png": (W - 600, 0),                     # Upper-right circular disc bokeh pine branch + craggy basalt cliffs
        }
        for cname, (cx0, cy0) in crops.items():
            x0 = max(0, min(W - 600, cx0))
            y0 = max(0, min(H - 450, cy0))
            full_im.crop((x0, y0, x0 + 600, y0 + 450)).save(out_dir / cname)

    if webp_targets:
        for wp in webp_targets:
            wp.parent.mkdir(parents=True, exist_ok=True)
            full_im.save(wp, format="WEBP", quality=90, method=6)
            print(f"[fjord-01] Exported 2048x1152 WebP: {wp} ({wp.stat().st_size / 1024:.1f} KiB)")

    return final_rgb, {"elapsed": elapsed, "vp_ratio": cam.vp_x / W}


def main() -> None:
    ap = argparse.ArgumentParser(description="Render Fjord 01 — Blue-Hour Fjord Boathouse (2048x1152)")
    ap.add_argument("--seed", type=int, default=202601)
    ap.add_argument("--out-dir", type=pathlib.Path, default=None)
    ap.add_argument("--no-export", action="store_true", help="Skip exporting repository WebP assets")
    args = ap.parse_args()

    targets = None if args.no_export else [
        REPO_ROOT / "session/themes/fjord/walls/wall-01.webp",
        REPO_ROOT / "docs/assets/walls/fjord-01.webp",
    ]
    render_scene(seed=args.seed, out_dir=args.out_dir, webp_targets=targets)


if __name__ == "__main__":
    main()

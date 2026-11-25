#!/usr/bin/env python3
"""
Fjord 02 ("Glacier Station Overpass") — Pure-Python 3D-Grounded Scene Pipeline (2048 x 1152)

Five-stage deterministic hand-painted scene (zero image models, zero external assets):
  Stage 1: Metric right-handed Camera3D (off-axis vanishing point ~62% width),
           exact 3x3 UV-to-screen plane homographies masked by valid UV bounds and
           ray-plane depth z_cam > 0.05 with exponential fjord-mist distance fog,
           near-plane 3D line clipping, and Voronoi (d2 - d1) rain-and-frost-slicked
           irregular Nordic basalt paving stones.
  Stage 2: Single normalized 3D twilight vector SUN_DIR (z < 0) traced backward
           across 6 height-refinement steps onto the pitched cantilevered dark-steel
           station canopy & columns for geometrically aligned cast shadows, combined
           with Fresnel wet-stone glacier-blue lamp reflections, 3D mirrored column
           reflections, and capillary-rippled fjord bay water reflections.
  Stage 3: 4 depth-interleaved 2x supersampled PyCairo layers (Lanczos-downsampled)
           featuring ridged-multifractal snowy Norwegian peaks across the misty bay,
           gleaming continuous steel rails, slender H-beam catenary poles with sagging
           overhead power cables, recessed-web steel canopy H-columns, grounded frosted
           glass timetable shelter, Nordic timber bench & floating frost motes,
           textured with isotropic fbm_gouache mottling, 80-100 flat-brush dabs,
           cold-press paper tooth, and shadow bounce light.
  Stage 4: Foreground lower-right frosted steel girder & overpass railing silhouette
           convolved in premultiplied RGBA with a circular aperture disc bokeh kernel
           (r = 7.8) plus 1 px opposite R/B lateral shift, finished with 35mm
           multi-scale halation, radial CA, optical vignette, shadow black lift +
           C1 tanh highlight shoulder, luminance-weighted silver-halide grain, and
           faint analog YIQ transfer patina.
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

# Stage 2: Single normalized 3D blue-hour twilight light travel vector from open fjord right (z < 0)
_raw_sun = np.array([-0.64, 0.22, -0.736], dtype=np.float64)
SUN_DIR = _raw_sun / np.linalg.norm(_raw_sun)

# Metric platform, track, and cantilevered steel canopy geometry (meters)
Z_PLAT = 0.0
Z_TRACK = -0.96
PLAT_X0, PLAT_X1 = -6.10, 0.56
PLAT_Y0, PLAT_Y1 = 1.0, 52.0

CANOPY_X0, CANOPY_X1 = -6.15, 0.72
CANOPY_Y0, CANOPY_Y1 = 1.2, 42.0
CANOPY_Z0 = 3.78
CANOPY_Z1 = 4.46  # gentle upward architectural pitch toward the open fjord!

LAMP_X = -1.68
LAMP_YS = (5.8, 11.8, 17.8, 23.8, 29.8, 35.8)
COL_X = -3.32
COL_YS = (6.6, 14.6, 22.6, 30.6, 38.6)


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
    return rgb + w_sh * (np.array([0.012, 0.012, 0.009], dtype=np.float32) * strength)


def ridged_profile_1d(n: int, rng: np.random.Generator, octaves=((5, 0.52), (11, 0.26), (23, 0.13), (49, 0.06), (105, 0.03))) -> np.ndarray:
    """Crisp jagged glaciated alpine ridge profile in [0, 1] using multi-octave ridged folding."""
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    acc = np.zeros(n, dtype=np.float64)
    for freq, amp in octaves:
        grid = rng.uniform(-1.0, 1.0, size=freq + 2)
        gx = np.linspace(0.0, 1.0, freq + 2)
        val = np.interp(x, gx, grid)
        # Ridged multifractal fold -> sharp rocky alpine peaks & steep glacial valleys!
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
    """
    Paint a true gouache-shaded craggy Norwegian mountain range directly into float32
    using analytical cliff heightfield normals dotted with -SUN_DIR and organic alpine
    snowfields resting along ridges, shelves, and irregular eroded couloirs.
    """
    y_min = max(0, int(math.floor(float(ridge_y.min()))) - 4)
    y_max = min(H, int(math.ceil(hy)) + 6)
    if y_min >= y_max:
        return canvas
    bh = y_max - y_min
    yy = np.arange(y_min, y_max, dtype=np.float32)[:, None]
    ry = ridge_y[None, :]
    depth = yy - ry

    # Smooth antialiased silhouette mask (1.6 px soft gouache edge)
    mask = np.clip((depth + 0.8) / 1.6, 0.0, 1.0)
    mask *= np.clip((hy + 3.0 - yy) / 2.0, 0.0, 1.0)

    # Multi-scale glaciated rock relief field with slight geological strata warp
    r_coarse = fbm_gouache(bh, W, rng, scales=((78, 52), (32, 24), (14, 11), (6, 5)))
    r_crag = 1.0 - np.abs(fbm_gouache(bh, W, rng, scales=((54, 38), (22, 16), (9.5, 7.5), (4.5, 4.0))))
    # Diagonal tectonic shear matching glaciated fjord walls
    xx_shift = (np.arange(W, dtype=np.float32)[None, :] + depth * 0.42) % W
    yy_loc = np.broadcast_to(np.arange(bh, dtype=np.float32)[:, None], (bh, W))
    strata = remap_bilinear(r_crag, xx_shift, yy_loc)
    relief = 0.56 * strata + 0.44 * r_coarse

    # Finite-difference cliff normals lit by SUN_DIR (sun coming from upper-right -> +dx faces are lit!)
    gx = np.zeros_like(relief)
    gy = np.zeros_like(relief)
    gx[:, 1:-1] = (relief[:, 2:] - relief[:, :-2]) * 3.8
    gy[1:-1, :] = (relief[2:, :] - relief[:-2, :]) * 2.6
    # Combine with macro ridge slope so right-facing mountain flanks catch twilight!
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

    # Organic alpine snow accumulation: clinging to upper peak ridges, wind-packed couloirs & sunlit shelves!
    elev_f = np.exp(-np.maximum(0.0, depth) / 95.0)
    couloir = np.clip((strata - 0.36) * 2.2, 0.0, 1.0)
    snow_raw = (elev_f * 0.62 + couloir * 0.52 + lit * 0.36 + m_sl * 0.22 + d_sl * 0.14) - (1.0 - snow_bias)
    snow_m = np.clip(snow_raw * 2.15, 0.0, 0.92)
    # Soft blue snow shadow modulation on left-facing snowfaces
    snow_shaded = sc * (0.76 + 0.28 * lit[..., None]) + (m_sl * 0.015)[..., None]
    col = rock * (1.0 - snow_m[..., None]) + snow_shaded * snow_m[..., None]

    # Delicate pale twilight rim grazing the very top jagged mountain ridge
    rim = np.exp(-np.maximum(0.0, depth) / 3.2) * np.clip(0.55 + dr_dx * 0.6, 0.15, 1.0) * 0.36
    col += rim[..., None] * sc

    # Valley fjord mist climbing organically up from the shoreline base of this range!
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
    def __init__(self, w: int, h: int, eye: Iterable[float], target: Iterable[float], fov_x_deg: float = 58.0):
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
        fog_start: float = 6.0,
        fog_density: float = 0.025,
        alpha_scale: float = 1.0,
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
        y0 = max(0, int(math.floor(min(ys))) - 1)
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


# =============================================================================
# Stage 2 — Unified 3D Light Solver (6-Step Pitched Steel Canopy Shadow Trace)
# =============================================================================

def canopy_z_at(x: np.ndarray) -> np.ndarray:
    t = np.clip((x - CANOPY_X0) / (CANOPY_X1 - CANOPY_X0), 0.0, 1.0)
    return CANOPY_Z0 + t * (CANOPY_Z1 - CANOPY_Z0)


def trace_canopy_shadow(wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> np.ndarray:
    """
    Trace platform/track coordinates backward along -SUN_DIR across 6 fixed height
    refinement steps onto the pitched cantilevered steel canopy roof + support H-columns.
    """
    inv_sz = 1.0 / (-SUN_DIR[2])
    z_est = np.full_like(wx, 0.5 * (CANOPY_Z0 + CANOPY_Z1), dtype=np.float64)
    for _ in range(6):
        t = np.maximum(0.0, (z_est - wz) * inv_sz)
        hx = wx - SUN_DIR[0] * t
        hy = wy - SUN_DIR[1] * t
        z_est = canopy_z_at(hx)
    t = np.maximum(0.0, (z_est - wz) * inv_sz)
    hx = wx - SUN_DIR[0] * t
    hy = wy - SUN_DIR[1] * t

    in_y = (hy >= CANOPY_Y0) & (hy <= CANOPY_Y1)
    edge_soft = np.clip(((CANOPY_X1 + 0.16) - hx) / 0.34, 0.0, 1.0) * in_y * (hx >= CANOPY_X0 - 0.5)
    rib_band = 0.07 * np.exp(-((np.mod(hy - 2.6, 4.0) - 2.0) / 0.25) ** 2) * edge_soft

    prop_occ = np.zeros_like(wx, dtype=np.float64)
    for cy in COL_YS:
        for cz in (0.4, 1.1, 1.9, 2.8):
            tc = np.maximum(0.0, (cz - wz) * inv_sz)
            px = wx - SUN_DIR[0] * tc
            py = wy - SUN_DIR[1] * tc
            d2 = ((px - COL_X) / 0.24) ** 2 + ((py - cy) / 0.24) ** 2
            prop_occ = np.maximum(prop_occ, np.exp(-d2) * 0.48)

    tb = np.maximum(0.0, (0.55 - wz) * inv_sz)
    bx = wx - SUN_DIR[0] * tb
    by = wy - SUN_DIR[1] * tb
    in_bench = (bx >= -2.85) & (bx <= -2.10) & (by >= 8.3) & (by <= 10.9) & (wz < 0.52)
    prop_occ = np.maximum(prop_occ, np.where(in_bench, 0.52, 0.0))

    occ = np.clip(edge_soft * 0.64 + rib_band + prop_occ * 0.45, 0.0, 0.74)
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
    fade_dist: float = 34.0,
    taper: bool = True,
    tremor: float = 0.22,
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
# Stage 1 & 2 — Blue-Hour Sky, Craggy Glaciated Alpine Peaks & Rippled Fjord Bay
# =============================================================================

def paint_sky_mountains_and_bay(cam: Camera3D, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    hy = float(cam.horizon_y)
    u = xx / W
    v_sky = np.clip(yy / max(1.0, hy), 0.0, 1.15)

    mottle = fbm_gouache(H, W, rng)
    dabs = flat_brush_dabs(H, W, rng, count=94)
    tooth = paper_tooth(H, W, rng)

    # Fjord twilight sky: #080f18 zenith -> steel blue -> crisp glacier-blue #95c2e2 & silver dusk horizon
    c_zen = np.array([0.031, 0.059, 0.100], dtype=np.float32)
    c_mid = np.array([0.102, 0.180, 0.275], dtype=np.float32)
    c_low = np.array([0.335, 0.518, 0.672], dtype=np.float32)
    c_hor = np.array([0.655, 0.818, 0.925], dtype=np.float32)

    t1 = np.clip(v_sky / 0.50, 0.0, 1.0)
    t2 = np.clip((v_sky - 0.42) / 0.42, 0.0, 1.0)
    t3 = np.clip((v_sky - 0.78) / 0.24, 0.0, 1.0)
    sky = (
        c_zen[None, None, :] * (1.0 - t1[..., None])
        + c_mid[None, None, :] * (t1 * (1.0 - t2))[..., None]
        + c_low[None, None, :] * (t2 * (1.0 - t3))[..., None]
        + c_hor[None, None, :] * (t2 * t3)[..., None]
    )

    # Luminous blue-hour opening down the fjord bay pass (~64% width)
    du = (u - 0.64) * 1.80
    dv = (v_sky - 0.94) * 2.40
    glow = np.exp(-(du * du + dv * dv)).astype(np.float32)
    sky += glow[..., None] * np.array([0.20, 0.25, 0.29], dtype=np.float32)

    # Layered arctic stratus twilight clouds
    cf = fbm_gouache(H, W, rng, scales=((105, 145), (38, 56), (15, 20), (6.5, 8.0)))
    c_env = np.exp(-((v_sky - 0.54) / 0.25) ** 2) * np.clip((u - 0.18) / 0.55, 0.0, 1.0)
    cm = np.clip((cf * 0.62 + 0.16) * c_env, 0.0, 1.0)
    c_cl_d = np.array([0.090, 0.155, 0.238], dtype=np.float32)
    c_cl_l = np.array([0.58, 0.76, 0.89], dtype=np.float32)
    lf = np.clip((v_sky - 0.30) / 0.60, 0.0, 1.0) * (0.55 + 0.45 * glow)
    cl_c = c_cl_d[None, None, :] * (1.0 - lf[..., None]) + c_cl_l[None, None, :] * lf[..., None]
    sky = sky * (1.0 - cm[..., None] * 0.44) + cl_c * (cm[..., None] * 0.44)
    sky += (mottle * 0.015 + dabs * 0.011 + tooth * 0.005)[..., None]

    # 3 depth layers of jagged glaciated Norwegian alpine mountains (zero sinusoidal humps, zero zebra stripes!)
    ux = np.linspace(0.0, 1.0, W, dtype=np.float32)

    # Layer 0: Distant snow-draped alpine range across the far fjord opening
    p0 = ridged_profile_1d(W, rng, octaves=((6, 0.48), (13, 0.27), (29, 0.14), (63, 0.07), (135, 0.04)))
    env0 = 135.0 * np.exp(-((ux - 0.65) / 0.36) ** 2) + 88.0 * np.exp(-((ux - 0.18) / 0.28) ** 2) + 34.0
    ridge0 = hy - (env0 * (0.42 + 0.68 * p0))
    sky = composite_painterly_mountain_layer(
        sky, ridge0, hy,
        rock_dark=(0.18, 0.30, 0.43), rock_lit=(0.27, 0.43, 0.57), snow_col=(0.68, 0.81, 0.91),
        snow_bias=0.52, mist_col=(0.48, 0.66, 0.80), mist_strength=0.76, mist_height=74.0,
        rng=rng, mottle=mottle, dabs=dabs, tooth=tooth,
    )

    # Layer 1: Dramatic snowy fjord mountain wall sweeping up across the right bay
    p1 = ridged_profile_1d(W, rng, octaves=((5, 0.50), (11, 0.26), (25, 0.14), (55, 0.07), (120, 0.03)))
    env1 = np.where(ux > 0.46, 265.0 * np.power(np.clip((ux - 0.46) / 0.54, 0.0, 1.0), 0.78), 0.0)
    env1 += np.where(ux < 0.38, 125.0 * np.power(np.clip((0.38 - ux) / 0.38, 0.0, 1.0), 0.85), 0.0)
    ridge1 = np.where(env1 > 1.0, hy - env1 * (0.48 + 0.62 * p1) - 6.0, hy + 24.0)
    sky = composite_painterly_mountain_layer(
        sky, ridge1, hy,
        rock_dark=(0.088, 0.155, 0.242), rock_lit=(0.168, 0.282, 0.402), snow_col=(0.68, 0.82, 0.92),
        snow_bias=0.50, mist_col=(0.46, 0.64, 0.78), mist_strength=0.60, mist_height=60.0,
        rng=rng, mottle=mottle, dabs=dabs, tooth=tooth,
    )

    # Layer 2: Steep closer basalt headland jutting in from the right bay edge
    p2 = ridged_profile_1d(W, rng, octaves=((4, 0.50), (9, 0.28), (21, 0.14), (47, 0.08)))
    env2 = np.where(ux > 0.74, 215.0 * np.power(np.clip((ux - 0.74) / 0.26, 0.0, 1.0), 0.82), 0.0)
    ridge2 = np.where(env2 > 1.0, hy - env2 * (0.55 + 0.50 * p2) - 4.0, hy + 28.0)
    sky = composite_painterly_mountain_layer(
        sky, ridge2, hy,
        rock_dark=(0.048, 0.086, 0.142), rock_lit=(0.105, 0.178, 0.265), snow_col=(0.64, 0.80, 0.91),
        snow_bias=0.42, mist_col=(0.46, 0.64, 0.78), mist_strength=0.38, mist_height=42.0,
        rng=rng, mottle=mottle, dabs=dabs, tooth=tooth,
    )

    # Tiny distant fjord shoreline coastal village pinpricks along the misty bay base
    s_lit, c_lit = new_cairo_2x()
    for lx, ly, lr in [(1338, hy - 4.2, 1.9), (1446, hy - 4.8, 2.2), (1488, hy - 3.6, 1.6), (1672, hy - 4.8, 2.0), (1725, hy - 4.0, 1.7)]:
        lg = cairo.RadialGradient(lx, ly, 0.2, lx, ly, lr * 5.2)
        lg.add_color_stop_rgba(0.0, 1.0, 0.88, 0.66, 0.88)
        lg.add_color_stop_rgba(0.42, 0.92, 0.72, 0.44, 0.28)
        lg.add_color_stop_rgba(1.0, 0.88, 0.64, 0.36, 0.0)
        c_lit.set_source(lg)
        c_lit.arc(lx, ly, lr * 5.2, 0, math.tau)
        c_lit.fill()
    sky = over_premul(sky, finish_cairo_2x(s_lit))

    # Low sea-mist ribbon drifting gracefully across the fjord water base
    mist_n = fbm_gouache(H, W, rng, scales=((82, 125), (32, 46), (13, 18), (6, 7)))
    mist_b = np.exp(-((yy - (hy - 18.0)) / 30.0) ** 2) * np.clip((u - 0.35) / 0.45, 0.0, 1.0)
    mist_a = np.clip((0.50 + 0.50 * mist_n) * mist_b * 0.44, 0.0, 0.56)
    sky = sky * (1.0 - mist_a[..., None]) + np.array([0.55, 0.73, 0.86], dtype=np.float32)[None, None, :] * mist_a[..., None]

    # Stage 2: Capillary-rippled Fresnel fjord water below horizon
    dist_t = np.clip((yy - hy) / max(1.0, H - hy), 0.0, 1.0)
    y_mir = np.clip(2.0 * hy - yy, 0.0, hy - 1.0)
    rip1 = resize_bicubic(rng.standard_normal((H // 3, W // 42)).astype(np.float32), W, H)
    rip2 = resize_bicubic(rng.standard_normal((H // 6, W // 20)).astype(np.float32), W, H)
    wave = 0.60 * rip1 + 0.40 * rip2

    map_x = xx + wave * (1.6 + 10.5 * np.power(dist_t, 0.85))
    map_y = np.clip(y_mir + wave * (2.0 + 16.0 * np.power(dist_t, 1.1)), 0.0, hy - 1.0)
    refl_w = remap_bilinear(sky, map_x, map_y)

    c_deep = np.array([0.028, 0.056, 0.094], dtype=np.float32)
    c_far = np.array([0.125, 0.220, 0.315], dtype=np.float32)
    w_body = c_far[None, None, :] * (1.0 - dist_t[..., None]) + c_deep[None, None, :] * dist_t[..., None]
    fresnel = 0.26 + (0.88 - 0.26) * np.power(1.0 - dist_t, 1.4)
    w_col = w_body * (1.0 - fresnel[..., None]) + refl_w * fresnel[..., None]
    w_col += (np.clip(wave * 0.24, -0.12, 0.24) * (0.35 + 0.65 * dist_t))[..., None] * np.array([0.08, 0.15, 0.22], dtype=np.float32)

    alpha_w = np.clip((yy - (hy - 1.5)) / 3.0, 0.0, 1.0)
    out = sky * (1.0 - alpha_w[..., None]) + w_col * alpha_w[..., None]
    fog_rgb = np.array([0.38, 0.55, 0.69], dtype=np.float32)
    return np.clip(out, 0.0, 1.25), fog_rgb


# =============================================================================
# Stage 1 & 2 — Planar UV Surfaces (Voronoi Basalt Platform, Canopy, Track Bed)
# =============================================================================

def paint_voronoi_platform_uv(rng: np.random.Generator, tw: int = 1024, th: int = 1536) -> np.ndarray:
    """
    Rain-and-frost-slicked irregular Voronoi basalt paving stones beneath the
    cantilevered steel station canopy, with dark recessed mortar crevices,
    selective frost rime, exact 6-step SUN_DIR cast shadows, organic rain puddles,
    and Fresnel pale glacier-blue canopy lamp reflections!
    """
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wx = PLAT_X0 + u * (PLAT_X1 - PLAT_X0)
    wy = PLAT_Y0 + v * (PLAT_Y1 - PLAT_Y0)
    wz = np.full_like(wx, Z_PLAT)

    nx = resize_bicubic(rng.standard_normal((th // 32, tw // 28)).astype(np.float32), tw, th) * 0.050
    ny = resize_bicubic(rng.standard_normal((th // 32, tw // 28)).astype(np.float32), tw, th) * 0.056
    px, py = wx + nx, wy + ny

    cell_a, cell_b = 0.64, 0.70
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
    # Crisp dark recessed mortar groove (smooth antialiased transition so zero stair-stepping!)
    stone = np.clip((edge_d - 0.014) / 0.030, 0.0, 1.0)
    bevel = np.clip(1.0 - np.abs(edge_d - 0.046) / 0.022, 0.0, 1.0)

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=90)
    tooth = paper_tooth(th, tw, rng)
    sv = tone_tab[best_id].astype(np.float32)
    hv = hue_tab[best_id].astype(np.float32)

    # Dark basalt slate paving tones
    base = np.zeros((th, tw, 3), dtype=np.float32)
    base[:, :, 0] = 0.072 + sv * 0.58 + hv * 0.35 + mottle * 0.014 + dabs * 0.011
    base[:, :, 1] = 0.104 + sv * 0.68 + mottle * 0.018 + dabs * 0.014
    base[:, :, 2] = 0.146 + sv * 0.78 - hv * 0.30 + mottle * 0.022 + dabs * 0.016

    crevice = np.array([0.014, 0.024, 0.040], dtype=np.float32)
    rgb = stone[..., None] * base + (1.0 - stone[..., None]) * crevice

    # Subtle frosty glint along stone bevel shoulders (leaving recessed mortar genuinely dark!)
    frost_patch = np.clip((mottle * 0.55 + dabs * 0.35 + 0.12), 0.0, 1.0)
    rgb += (bevel * frost_patch * 0.18)[..., None] * np.array([0.38, 0.54, 0.70], dtype=np.float32)

    # Pale-silver Nordic granite platform coping slab & tactile safety strip near X = PLAT_X1
    in_coping = wx >= (PLAT_X1 - 0.32)
    coping_col = np.array([0.17, 0.24, 0.32], dtype=np.float32) + mottle[..., None] * 0.016
    coping_joint = np.clip(np.abs(np.mod(wy, 0.92) - 0.46) / 0.44, 0.0, 1.0).astype(np.float32)
    coping_col *= (0.76 + 0.24 * (1.0 - np.clip((coping_joint - 0.94) / 0.06, 0.0, 1.0)))[..., None]
    rgb = np.where(in_coping[..., None], coping_col, rgb)

    in_tactile = (wx >= -0.14) & (wx <= 0.10)
    ribs = 0.85 + 0.15 * np.sin(wx * 95.0).astype(np.float32)
    tact_col = np.array([0.34, 0.46, 0.56], dtype=np.float32) * ribs[..., None] + mottle[..., None] * 0.015
    rgb = np.where(in_tactile[..., None], tact_col, rgb)

    # Exact Stage 2 6-step SUN_DIR canopy & column cast shadows!
    sh = trace_canopy_shadow(wx, wy, wz)
    rgb *= (0.42 + 0.74 * sh)[..., None]

    # Organic rain-and-frost puddle wetness mask across the uneven basalt paving stones!
    puddle_f = fbm_gouache(th, tw, rng, scales=((64, 58), (24, 26), (10, 10), (4.5, 4.5)))
    wetness = np.clip(0.54 + 0.48 * puddle_f + 0.24 * dabs, 0.08, 1.0)
    puddle_pool = np.clip((puddle_f + 0.15) * 1.45, 0.0, 1.0)

    dist_t = np.clip((wy.astype(np.float32) - PLAT_Y0) / 32.0, 0.0, 1.0)
    fresnel_p = 0.24 + (0.84 - 0.24) * np.power(dist_t, 0.85)

    # Open fjord sky reflection on the exposed right side of the rain-slicked platform
    sky_refl = np.array([0.22, 0.37, 0.52], dtype=np.float32)
    rgb += (wetness * sh * fresnel_p * 0.62)[..., None] * sky_refl[None, None, :]

    # Pale glacier-blue canopy lamp pools & elongated wet-stone Fresnel reflections!
    lamp_acc = np.zeros((th, tw), dtype=np.float32)
    spec_acc = np.zeros((th, tw), dtype=np.float32)
    for ly in LAMP_YS:
        dx_l = (wx - LAMP_X).astype(np.float32)
        dy_l = (wy - ly).astype(np.float32)
        lamp_acc += np.exp(-((dx_l / 1.85) ** 2 + (dy_l / 2.35) ** 2)) * 0.42
        spec_acc += np.exp(-((dx_l / 0.42) ** 2 + ((dy_l + 0.65) / 1.45) ** 2)) * (0.45 + 0.75 * puddle_pool) * (0.45 + 0.55 * stone)

    lamp_col = np.array([0.46, 0.72, 0.90], dtype=np.float32)
    spec_col = np.array([0.74, 0.92, 1.04], dtype=np.float32)
    rgb += lamp_acc[..., None] * lamp_col[None, None, :] * (0.65 + 0.35 * stone)[..., None]
    rgb += (spec_acc * wetness * fresnel_p)[..., None] * spec_col[None, None, :]

    # Soft timetable kiosk reflection spill on wet stone around X = -4.1, Y = 11.5..15.5
    d_kiosk = np.hypot((wx - (-3.95)) / 1.7, (wy - 13.5) / 2.4).astype(np.float32)
    rgb += (np.exp(-(d_kiosk * d_kiosk)) * (0.42 + 0.58 * wetness))[..., None] * np.array([0.26, 0.44, 0.58], dtype=np.float32)

    # Dark 3D mirrored column shadows/reflections on wet stone right below each H-column!
    for cy in COL_YS:
        dc = np.hypot((wx - COL_X) / 0.28, (wy - (cy - 0.55)) / 0.95).astype(np.float32)
        col_refl = np.exp(-(dc * dc * 1.6)) * puddle_pool * 0.48
        rgb *= (1.0 - col_refl)[..., None]

    # Contact shadow grounding the bench and glass shelter curb
    d_bench = ((wx >= -2.82) & (wx <= -2.14) & (wy >= 8.38) & (wy <= 10.78)).astype(np.float32)
    rgb *= (1.0 - d_bench * 0.34)[..., None]

    rgb = apply_shadow_bounce(rgb, strength=0.88)
    rgb += tooth[..., None] * 0.006
    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.32)
    return rgba


def paint_platform_edge_wall_uv(rng: np.random.Generator, tw: int = 896, th: int = 192) -> np.ndarray:
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    v = yy / (th - 1)
    mottle = fbm_gouache(th, tw, rng)
    tooth = paper_tooth(th, tw, rng)

    rgb = np.zeros((th, tw, 3), dtype=np.float32)
    rgb[:, :, 0] = 0.038 + mottle * 0.010
    rgb[:, :, 1] = 0.058 + mottle * 0.014
    rgb[:, :, 2] = 0.088 + mottle * 0.018
    lip_sh = np.clip((0.92 - v.astype(np.float32)) / 0.25, 0.25, 1.0)
    rgb *= lip_sh[..., None]
    rgb += tooth[..., None] * 0.005
    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.0)
    return rgba


def paint_track_ballast_uv(rng: np.random.Generator, tw: int = 640, th: int = 1408) -> np.ndarray:
    """Fine crushed dark basalt railway ballast & frost-dusted sleepers (clean, natural snow drift!)."""
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wx = PLAT_X1 + u * (4.10 - PLAT_X1)
    wy = 1.0 + v * 58.0

    mottle = fbm_gouache(th, tw, rng, scales=((48, 44), (18, 18), (7.5, 7.0), (3.2, 3.2)))
    gravel = rng.normal(0.0, 0.018, size=(th, tw)).astype(np.float32)
    tooth = paper_tooth(th, tw, rng)

    # Dark wet crushed basalt gravel bed
    rgb = np.zeros((th, tw, 3), dtype=np.float32)
    rgb[:, :, 0] = 0.042 + mottle * 0.010 + gravel * 0.65
    rgb[:, :, 1] = 0.064 + mottle * 0.014 + gravel * 0.78
    rgb[:, :, 2] = 0.095 + mottle * 0.018 + gravel * 0.92

    # Concrete viaduct curb edge on far right (X > 3.58)
    in_curb = wx >= 3.58
    curb_c = np.array([0.11, 0.16, 0.23], dtype=np.float32) + mottle[..., None] * 0.014
    rgb = np.where(in_curb[..., None], curb_c, rgb)

    # Dark timber/concrete sleepers spaced every 0.72 m along Y with crisp shadow trough
    sleeper_pitch = 0.72
    sf = np.mod(wy, sleeper_pitch) / sleeper_pitch
    in_sleeper = (sf >= 0.34) & (sf <= 0.66) & (wx >= 1.15) & (wx <= 3.38)
    sl_col = np.array([0.070, 0.098, 0.136], dtype=np.float32) + mottle[..., None] * 0.012
    # Subtle silver-frost edge along sleeper shoulders (no blotchy cow spots!)
    frost_sl = np.clip((0.38 - np.abs(sf - 0.50)) * 4.5, 0.0, 1.0).astype(np.float32) * 0.12
    sl_col += frost_sl[..., None] * np.array([0.38, 0.56, 0.72], dtype=np.float32)
    rgb = np.where(in_sleeper[..., None], sl_col, rgb)

    # Gentle linear snow/frost drift hugging the sheltered right concrete curb & rail ballast edges
    edge_drift = np.exp(-((wx.astype(np.float32) - 3.48) / 0.28) ** 2) * np.clip(0.55 + 0.45 * mottle, 0.0, 1.0) * 0.36
    center_dust = np.clip(mottle * 0.30 + 0.10, 0.0, 0.22) * (1.0 - in_sleeper.astype(np.float32) * 0.5)
    rgb += (edge_drift + center_dust * 0.38)[..., None] * np.array([0.38, 0.54, 0.68], dtype=np.float32)

    # Soft platform shadow along left edge of ballast bed (X < 1.18)
    plat_sh = np.clip((wx.astype(np.float32) - 0.62) / 0.85, 0.36, 1.0)
    rgb *= plat_sh[..., None]

    rgb = apply_shadow_bounce(rgb, strength=0.8)
    rgb += tooth[..., None] * 0.005
    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.15)
    return rgba


def paint_steel_canopy_underside_uv(rng: np.random.Generator, tw: int = 768, th: int = 1152) -> np.ndarray:
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wx = CANOPY_X0 + u * (CANOPY_X1 - CANOPY_X0)
    wy = CANOPY_Y0 + v * (CANOPY_Y1 - CANOPY_Y0)

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=86)
    tooth = paper_tooth(th, tw, rng)

    slat = 0.5 + 0.5 * np.sin(wx * 22.0).astype(np.float32)
    rgb = np.zeros((th, tw, 3), dtype=np.float32)
    rgb[:, :, 0] = 0.036 + slat * 0.012 + mottle * 0.010 + dabs * 0.008
    rgb[:, :, 1] = 0.056 + slat * 0.018 + mottle * 0.014 + dabs * 0.010
    rgb[:, :, 2] = 0.084 + slat * 0.025 + mottle * 0.018 + dabs * 0.013

    outer_b = np.clip((wx.astype(np.float32) - (-1.4)) / 2.2, 0.0, 1.0)
    rgb += outer_b[..., None] * np.array([0.055, 0.095, 0.140], dtype=np.float32)

    rafter_dist = np.abs(np.mod(wy - 2.6, 4.0) - 2.0)
    in_rafter = rafter_dist < 0.14
    rafter_col = np.array([0.020, 0.034, 0.054], dtype=np.float32)
    rafter_rim = (rafter_dist >= 0.11) & (rafter_dist < 0.16)
    rgb = np.where(in_rafter[..., None], rafter_col, rgb)
    rgb = np.where(rafter_rim[..., None], rgb + np.array([0.04, 0.075, 0.11], dtype=np.float32), rgb)

    in_spine = (np.abs(wx - COL_X) < 0.18) | (wx >= CANOPY_X1 - 0.16)
    rgb = np.where(in_spine[..., None], np.array([0.018, 0.032, 0.052], dtype=np.float32), rgb)

    for ly in LAMP_YS:
        dx = (wx - LAMP_X).astype(np.float32)
        dy = (wy - ly).astype(np.float32)
        halo = np.exp(-((dx / 1.35) ** 2 + (dy / 1.55) ** 2))
        rgb += halo[..., None] * np.array([0.22, 0.39, 0.54], dtype=np.float32)
        in_fix = (np.abs(dx) < 0.17) & (np.abs(dy) < 0.68)
        in_core = (np.abs(dx) < 0.11) & (np.abs(dy) < 0.58)
        rgb = np.where(in_fix[..., None], np.array([0.52, 0.78, 0.94], dtype=np.float32), rgb)
        rgb = np.where(in_core[..., None], np.array([1.06, 1.14, 1.24], dtype=np.float32), rgb)

    rgb = apply_shadow_bounce(rgb, strength=0.75)
    rgb += tooth[..., None] * 0.005
    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.35)
    return rgba


# =============================================================================
# Stage 3 — Depth-Interleaved 2x PyCairo Rails, Catenary, Canopy Props & Frost
# =============================================================================

def draw_left_parapet_wall(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
) -> None:
    """Grounded dark-steel & concrete coastal parapet railing along the platform's left edge (X = PLAT_X0)."""
    def fog_c(c3, dist: float):
        ft = 1.0 - math.exp(-max(0.0, dist - 6.0) * 0.026)
        c = np.asarray(c3, dtype=np.float64) * (1.0 - ft) + fog_rgb * ft
        return (float(c[0]), float(c[1]), float(c[2]))

    # Low concrete parapet curb wall along left edge (Z = 0.0 .. 0.42) so platform never floats over water!
    curb_pts = [
        np.array([PLAT_X0, PLAT_Y0, Z_PLAT]),
        np.array([PLAT_X0, PLAT_Y1, Z_PLAT]),
        np.array([PLAT_X0, PLAT_Y1, Z_PLAT + 0.44]),
        np.array([PLAT_X0, PLAT_Y0, Z_PLAT + 0.44]),
    ]
    c_proj = [cam.project(p) for p in curb_pts]
    if all(z >= Z_NEAR for _, _, z in c_proj):
        ctx.set_source_rgba(0.045, 0.072, 0.110, 0.96)
        ctx.move_to(float(c_proj[0][0]), float(c_proj[0][1]))
        for pu, pv, _ in c_proj[1:]:
            ctx.line_to(float(pu), float(pv))
        ctx.close_path()
        ctx.fill()

    # Horizontal dark-steel safety rails & vertical balusters above the left parapet curb
    for zr, bw, col in ((0.46, 3.6, [0.16, 0.25, 0.36]), (0.78, 2.4, [0.06, 0.10, 0.15]), (1.12, 4.2, [0.22, 0.34, 0.46])):
        seg = cam.project_polyline(np.array([[PLAT_X0, PLAT_Y0, zr], [PLAT_X0, PLAT_Y1, zr]]))
        if len(seg) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in seg], base_w=bw, z_depth=12.0,
                               rgba=(*col, 0.94), rng=rng, taper=False, tremor=0.10)
    for py in np.arange(3.0, 48.0, 3.2):
        dist = float(np.linalg.norm(np.array([PLAT_X0, py, 0.6]) - cam.eye))
        pseg = cam.project_polyline(np.array([[PLAT_X0, py, Z_PLAT], [PLAT_X0, py, 1.14]]))
        if len(pseg) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in pseg], base_w=4.4, z_depth=dist,
                               rgba=(*fog_c([0.045, 0.075, 0.118], dist), 0.94), rng=rng, taper=False, tremor=0.08)


def draw_continuous_rails_and_viaduct_railing(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
) -> None:
    """Single continuous perspective pass with smooth per-segment depth width decay (zero thickness jumps!)."""
    def fog_c(c3, dist: float):
        ft = 1.0 - math.exp(-max(0.0, dist - 6.0) * 0.025)
        c = np.asarray(c3, dtype=np.float64) * (1.0 - ft) + fog_rgb * ft
        return (float(c[0]), float(c[1]), float(c[2]))

    def stroke_perspective_3d_line(x_val: float, z_val: float, y_a: float, y_b: float, base_w: float, col3: list[float], alpha: float) -> None:
        ys = np.geomspace(max(1.0, y_a), y_b, 48)
        pts3 = np.column_stack([np.full(len(ys), x_val), ys, np.full(len(ys), z_val)])
        proj = cam.project_polyline(pts3)
        for i in range(len(proj) - 1):
            u0, v0, z0 = proj[i]
            u1, v1, z1 = proj[i + 1]
            zm = 0.5 * (z0 + z1)
            decay = float(np.clip(1.0 / (1.0 + zm / 12.0), 0.16, 1.0))
            ctx.set_source_rgba(*fog_c(col3, zm), alpha)
            ctx.set_line_width(max(0.45, base_w * decay))
            ctx.move_to(u0, v0)
            ctx.line_to(u1, v1)
            ctx.stroke()

    # Smooth continuously tapering dark rail web + gleaming polished steel rail head!
    for rx in (1.56, 3.00):
        stroke_perspective_3d_line(rx, Z_TRACK + 0.05, 1.0, 59.0, 8.2, [0.036, 0.060, 0.095], 0.98)
        stroke_perspective_3d_line(rx, Z_TRACK + 0.115, 1.0, 59.0, 3.0, [0.68, 0.85, 0.97], 0.92)

    # Viaduct right parapet safety railing along X = 3.92
    for z_r, bw_r, col_r in (
        (Z_TRACK + 1.08, 4.6, [0.18, 0.28, 0.40]),
        (Z_TRACK + 0.62, 2.8, [0.06, 0.10, 0.15]),
        (Z_TRACK + 0.22, 2.6, [0.05, 0.08, 0.13]),
    ):
        stroke_perspective_3d_line(3.92, z_r, 1.0, 59.0, bw_r, col_r, 0.92)

    for py in np.arange(3.0, 58.0, 3.0):
        dist = float(np.linalg.norm(np.array([3.92, py, Z_TRACK + 0.5]) - cam.eye))
        seg_p = cam.project_polyline(np.array([[3.92, py, Z_TRACK], [3.92, py, Z_TRACK + 1.10]]))
        if len(seg_p) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in seg_p], base_w=3.2, z_depth=dist,
                               rgba=(*fog_c([0.05, 0.085, 0.135], dist), 0.90), rng=rng, taper=False, tremor=0.08)


def draw_catenary_masts_and_cables(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    y_min: float,
    y_max: float,
) -> None:
    """Slender Nordic steel H-beam catenary masts, truss arms, ceramic insulators, power cables & signal mast."""
    def fog_c(c3, dist: float):
        ft = 1.0 - math.exp(-max(0.0, dist - 6.0) * 0.025)
        c = np.asarray(c3, dtype=np.float64) * (1.0 - ft) + fog_rgb * ft
        return (float(c[0]), float(c[1]), float(c[2]))

    mast_ys = [y for y in (9.5, 19.5, 29.5, 39.5, 49.5, 59.5) if y_min - 0.5 <= y <= y_max + 0.5]
    for my in mast_ys:
        dist = float(np.linalg.norm(np.array([3.74, my, 2.5]) - cam.eye))
        c_steel = fog_c([0.042, 0.072, 0.114], dist)
        c_rim = fog_c([0.40, 0.58, 0.74], dist)
        p_bot = np.array([3.74, my, Z_TRACK])
        p_top = np.array([3.74, my, 6.05])
        mseg = cam.project_polyline(np.array([p_bot, p_top]))
        if len(mseg) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in mseg], base_w=9.2, z_depth=dist,
                               rgba=(*c_steel, 0.98), rng=rng, taper=False, tremor=0.08)
            mrim = cam.project_polyline(np.array([p_bot + [0.05, 0, 0], p_top + [0.05, 0, 0]]))
            if len(mrim) >= 2:
                stroke_catmull_rom(ctx, [(p[0], p[1]) for p in mrim], base_w=2.4, z_depth=dist,
                                   rgba=(*c_rim, 0.68), rng=rng, taper=False, tremor=0.06)

        # Cantilevered tubular steel truss arm over track center (X = 2.28)
        for pa, pb, w_a in (
            (np.array([3.74, my, 5.65]), np.array([2.28, my, 5.25]), 4.0),
            (np.array([3.74, my, 4.35]), np.array([2.28, my, 4.25]), 3.4),
            (np.array([3.74, my, 4.35]), np.array([2.28, my, 5.25]), 3.2),
        ):
            aseg = cam.project_polyline(np.array([pa, pb]))
            if len(aseg) >= 2:
                stroke_catmull_rom(ctx, [(p[0], p[1]) for p in aseg], base_w=w_a, z_depth=dist,
                                   rgba=(*c_steel, 0.96), rng=rng, taper=False, tremor=0.06)

        for iz in (5.58, 4.34):
            iu, iv, iz_c = cam.project(np.array([3.48, my, iz]))
            if iz_c >= Z_NEAR:
                rad = max(1.6, 54.0 / iz_c)
                ctx.set_source_rgba(*fog_c([0.56, 0.74, 0.88], dist), 0.92)
                ctx.arc(float(iu), float(iv), rad, 0, math.tau)
                ctx.fill()

    # Overhead sagging catenary messenger wire, contact wire, feeder wire & vertical droppers
    span_ys = [-0.5, 9.5, 19.5, 29.5, 39.5, 49.5, 59.5]
    for i in range(len(span_ys) - 1):
        ya, yb = span_ys[i], span_ys[i + 1]
        if yb < y_min or ya > y_max:
            continue
        dist_m = float(np.linalg.norm(np.array([2.28, 0.5 * (ya + yb), 4.8]) - cam.eye))
        c_wire = fog_c([0.075, 0.125, 0.188], dist_m)
        c_wire_hi = fog_c([0.50, 0.68, 0.84], dist_m)

        n_seg = 16
        ts = np.linspace(0.0, 1.0, n_seg)
        ys_w = ya + ts * (yb - ya)
        pts_contact = np.column_stack([np.full(n_seg, 2.28), ys_w, np.full(n_seg, 4.25)])
        pts_mess = np.column_stack([np.full(n_seg, 2.28), ys_w, 5.25 - 0.42 * np.sin(ts * math.pi)])
        pts_feed = np.column_stack([np.full(n_seg, 3.92), ys_w, 5.88 - 0.35 * np.sin(ts * math.pi)])

        for wpts, bw_w, w_col in ((pts_contact, 1.9, c_wire), (pts_mess, 1.8, c_wire), (pts_feed, 1.6, c_wire_hi)):
            wseg = cam.project_polyline(wpts)
            if len(wseg) >= 2:
                stroke_catmull_rom(ctx, [(p[0], p[1]) for p in wseg], base_w=bw_w, z_depth=dist_m,
                                   rgba=(*w_col, 0.88), rng=rng, taper=False, tremor=0.05)

        for t_d in (0.20, 0.40, 0.60, 0.80):
            yd = ya + t_d * (yb - ya)
            zm = 5.25 - 0.42 * np.sin(t_d * math.pi)
            dseg = cam.project_polyline(np.array([[2.28, yd, 4.25], [2.28, yd, zm]]))
            if len(dseg) >= 2:
                stroke_catmull_rom(ctx, [(p[0], p[1]) for p in dseg], base_w=1.1, z_depth=dist_m,
                                   rgba=(*c_wire, 0.72), rng=rng, taper=False, tremor=0.04)

    # Crisp Nordic railway signal mast at Y = 23.5
    if y_min <= 23.5 <= y_max:
        sig_p = np.array([3.52, 23.5, 2.15])
        su, sv, sz = cam.project(sig_p)
        if sz >= Z_NEAR:
            sp_seg = cam.project_polyline(np.array([[3.52, 23.5, Z_TRACK], [3.52, 23.5, 2.55]]))
            if len(sp_seg) >= 2:
                stroke_catmull_rom(ctx, [(p[0], p[1]) for p in sp_seg], base_w=5.0, z_depth=23.5,
                                   rgba=(0.04, 0.07, 0.11, 0.96), rng=rng, taper=False, tremor=0.08)
            ctx.set_source_rgba(0.03, 0.05, 0.08, 0.98)
            ctx.rectangle(float(su) - 6.5, float(sv) - 14.0, 13.0, 28.0)
            ctx.fill()
            sg = cairo.RadialGradient(float(su), float(sv) - 5.0, 1.0, float(su), float(sv) - 5.0, 24.0)
            sg.add_color_stop_rgba(0.0, 0.82, 0.99, 0.98, 0.96)
            sg.add_color_stop_rgba(0.35, 0.44, 0.86, 0.88, 0.45)
            sg.add_color_stop_rgba(1.0, 0.35, 0.72, 0.82, 0.0)
            ctx.set_source(sg)
            ctx.arc(float(su), float(sv) - 5.0, 24.0, 0, math.tau)
            ctx.fill()


def draw_station_columns_shelter_and_bench(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
) -> None:
    """Band 2: Architectural H-beam steel columns (with full 3D plinths & knee braces), grounded glass shelter & solid Nordic bench."""
    def fog_c(c3, dist: float):
        ft = 1.0 - math.exp(-max(0.0, dist - 6.0) * 0.026)
        c = np.asarray(c3, dtype=np.float64) * (1.0 - ft) + fog_rgb * ft
        return (float(c[0]), float(c[1]), float(c[2]))

    def fill_quad3(pts3: list[np.ndarray], rgba: tuple[float, float, float, float]) -> None:
        proj = [cam.project(p) for p in pts3]
        if any(z < Z_NEAR for _, _, z in proj):
            return
        ctx.set_source_rgba(*rgba)
        ctx.move_to(float(proj[0][0]), float(proj[0][1]))
        for pu, pv, _ in proj[1:]:
            ctx.line_to(float(pu), float(pv))
        ctx.close_path()
        ctx.fill()

    # 1. Grounded coastal glass wind shelter & glowing Nordic route/timetable kiosk (X = -4.20, Y = 11.2..16.2)
    kx = -4.20
    dist_k = float(np.linalg.norm(np.array([kx, 13.5, 1.4]) - cam.eye))
    # Dark steel floor anchor curb beam connecting shelter solidly to basalt platform (Z = 0.0..0.16)
    fill_quad3(
        [np.array([kx - 0.04, 11.15, Z_PLAT]), np.array([kx - 0.04, 16.25, Z_PLAT]),
         np.array([kx - 0.04, 16.25, 0.16]), np.array([kx - 0.04, 11.15, 0.16])],
        (*fog_c([0.045, 0.075, 0.115], dist_k), 0.98),
    )
    # Top steel shelter transom header & glass panes
    for ya, yb in ((11.2, 12.82), (12.88, 14.50), (14.56, 16.18)):
        g_pts = [
            np.array([kx, ya, 0.16]),
            np.array([kx, yb, 0.16]),
            np.array([kx, yb, 2.58]),
            np.array([kx, ya, 2.58]),
        ]
        gp = [cam.project(p) for p in g_pts]
        if all(z >= Z_NEAR for _, _, z in gp):
            gg = cairo.LinearGradient(float(gp[3][0]), float(gp[3][1]), float(gp[0][0]), float(gp[0][1]))
            gg.add_color_stop_rgba(0.0, 0.34, 0.54, 0.70, 0.44)
            gg.add_color_stop_rgba(0.58, 0.16, 0.28, 0.40, 0.36)
            gg.add_color_stop_rgba(1.0, 0.42, 0.60, 0.76, 0.48)  # subtle frost condensation near base!
            ctx.set_source(gg)
            ctx.move_to(float(gp[0][0]), float(gp[0][1]))
            for pu, pv, _ in gp[1:]:
                ctx.line_to(float(pu), float(pv))
            ctx.close_path()
            ctx.fill()
    # Grounded steel shelter stanchion posts going all the way to Z = 0.0!
    for py_s in (11.18, 12.85, 14.53, 16.20):
        sseg = cam.project_polyline(np.array([[kx, py_s, Z_PLAT], [kx, py_s, 2.64]]))
        if len(sseg) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in sseg], base_w=5.5, z_depth=dist_k,
                               rgba=(*fog_c([0.04, 0.068, 0.108], dist_k), 0.98), rng=rng, taper=False, tremor=0.06)
    hseg = cam.project_polyline(np.array([[kx, 11.16, 2.62], [kx, 16.22, 2.62]]))
    if len(hseg) >= 2:
        stroke_catmull_rom(ctx, [(p[0], p[1]) for p in hseg], base_w=6.2, z_depth=dist_k,
                           rgba=(*fog_c([0.05, 0.085, 0.132], dist_k), 0.98), rng=rng, taper=False, tremor=0.06)

    # Illuminated Scandinavian coastal railway timetable lightbox inside center shelter bay
    lb_pts = [
        np.array([kx + 0.08, 13.05, 0.96]),
        np.array([kx + 0.08, 14.32, 0.96]),
        np.array([kx + 0.08, 14.32, 2.24]),
        np.array([kx + 0.08, 13.05, 2.24]),
    ]
    lbp = [cam.project(p) for p in lb_pts]
    if all(z >= Z_NEAR for _, _, z in lbp):
        cx_l = sum(float(p[0]) for p in lbp) / 4.0
        cy_l = sum(float(p[1]) for p in lbp) / 4.0
        hg = cairo.RadialGradient(cx_l, cy_l, 8.0, cx_l, cy_l, 130.0)
        hg.add_color_stop_rgba(0.0, 0.72, 0.90, 1.0, 0.30)
        hg.add_color_stop_rgba(1.0, 0.42, 0.68, 0.88, 0.0)
        ctx.set_source(hg)
        ctx.arc(cx_l, cy_l, 130.0, 0, math.tau)
        ctx.fill()

        ctx.set_source_rgba(0.76, 0.90, 0.98, 0.94)
        ctx.move_to(float(lbp[0][0]), float(lbp[0][1]))
        for pu, pv, _ in lbp[1:]:
            ctx.line_to(float(pu), float(pv))
        ctx.close_path()
        ctx.fill_preserve()
        ctx.set_source_rgba(0.04, 0.08, 0.13, 0.98)
        ctx.set_line_width(3.6)
        ctx.stroke()

        # Header band + coastal route line diagram + schedule rows inside timetable lightbox!
        hdr = [
            lb_pts[3] * 0.86 + lb_pts[0] * 0.14,
            lb_pts[2] * 0.86 + lb_pts[1] * 0.14,
            lb_pts[2] * 0.96 + lb_pts[1] * 0.04,
            lb_pts[3] * 0.96 + lb_pts[0] * 0.04,
        ]
        fill_quad3(hdr, (0.12, 0.24, 0.36, 0.86))
        for row_f in (0.28, 0.40, 0.52, 0.64, 0.76, 0.86):
            pa = lb_pts[3] * (1.0 - row_f) + lb_pts[0] * row_f
            pb = lb_pts[2] * (1.0 - row_f) + lb_pts[1] * row_f
            pa = pa + (pb - pa) * 0.10
            pb = pa + (pb - pa) * 0.80
            rseg = cam.project_polyline(np.array([pa, pb]))
            if len(rseg) >= 2:
                stroke_catmull_rom(ctx, [(p[0], p[1]) for p in rseg], base_w=2.0,
                                   z_depth=dist_k, rgba=(0.14, 0.28, 0.42, 0.68), rng=rng, taper=False, tremor=0.04)

    # 2. Architectural cantilevered dark-steel H-beam canopy columns (back-to-front!)
    z_col_top = float(canopy_z_at(np.array(COL_X)))
    for cy in reversed(COL_YS):
        dist = float(np.linalg.norm(np.array([COL_X, cy, 1.8]) - cam.eye))
        hw = 0.145
        pw = hw * 1.42
        ph = 0.18
        # Draw 3D bolted steel plinth FIRST so the vertical column rises solidly OUT OF the plinth top!
        pf0 = np.array([COL_X - pw, cy - pw, Z_PLAT])
        pf1 = np.array([COL_X + pw, cy - pw, Z_PLAT])
        pf2 = np.array([COL_X + pw, cy + pw, Z_PLAT])
        pt0 = np.array([COL_X - pw, cy - pw, ph])
        pt1 = np.array([COL_X + pw, cy - pw, ph])
        pt2 = np.array([COL_X + pw, cy + pw, ph])
        pt3 = np.array([COL_X - pw, cy + pw, ph])
        fill_quad3([pf0, pf1, pt1, pt0], (*fog_c([0.055, 0.088, 0.132], dist), 0.98))
        fill_quad3([pf1, pf2, pt2, pt1], (*fog_c([0.108, 0.165, 0.235], dist), 0.98))
        fill_quad3([pt0, pt1, pt2, pt3], (*fog_c([0.22, 0.34, 0.46], dist), 0.95))

        p_fl = np.array([COL_X - hw, cy - hw, ph])
        p_fr = np.array([COL_X + hw, cy - hw, ph])
        p_br = np.array([COL_X + hw, cy + hw, ph])
        t_fl = np.array([COL_X - hw, cy - hw, z_col_top + 0.05])
        t_fr = np.array([COL_X + hw, cy - hw, z_col_top + 0.05])
        t_br = np.array([COL_X + hw, cy + hw, z_col_top + 0.05])

        # Front face (-Y) and sunlit right face (+X facing fjord!)
        faces = [
            ([p_fl, p_fr, t_fr, t_fl], fog_c([0.036, 0.058, 0.092], dist), fog_c([0.022, 0.038, 0.062], dist)),
            ([p_fr, p_br, t_br, t_fr], fog_c([0.118, 0.185, 0.262], dist), fog_c([0.052, 0.088, 0.135], dist)),
        ]
        for pts3, c_hi, c_lo in faces:
            proj = [cam.project(p) for p in pts3]
            if any(z < Z_NEAR for _, _, z in proj):
                continue
            g = cairo.LinearGradient(float(proj[2][0]), float(proj[2][1]), float(proj[0][0]), float(proj[0][1]))
            g.add_color_stop_rgba(0.0, *c_hi, 1.0)
            g.add_color_stop_rgba(1.0, *c_lo, 1.0)
            ctx.set_source(g)
            ctx.move_to(float(proj[0][0]), float(proj[0][1]))
            for pu, pv, _ in proj[1:]:
                ctx.line_to(float(pu), float(pv))
            ctx.close_path()
            ctx.fill()

        # Recessed H-beam inner web shadow band on right face (+X)
        w_fr = np.array([COL_X + hw, cy - hw * 0.55, ph + 0.04])
        w_br = np.array([COL_X + hw, cy + hw * 0.55, ph + 0.04])
        wt_fr = np.array([COL_X + hw, cy - hw * 0.55, z_col_top - 0.10])
        wt_br = np.array([COL_X + hw, cy + hw * 0.55, z_col_top - 0.10])
        fill_quad3([w_fr, w_br, wt_br, wt_fr], (*fog_c([0.032, 0.054, 0.088], dist), 0.85))

        eseg = cam.project_polyline(np.array([p_fr, t_fr]))
        if len(eseg) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in eseg], base_w=2.4, z_depth=dist,
                               rgba=(*fog_c([0.52, 0.72, 0.86], dist), 0.78), rng=rng, taper=False, tremor=0.08)

        # Solid architectural steel knee-brace I-beam linking column top out toward cantilevered canopy girder
        b_a0 = np.array([COL_X + hw, cy - 0.06, z_col_top - 0.92])
        b_a1 = np.array([COL_X + hw, cy - 0.06, z_col_top - 0.78])
        z_br_t = float(canopy_z_at(np.array(COL_X + 1.48))) + 0.02
        b_b0 = np.array([COL_X + 1.48, cy - 0.06, z_br_t - 0.10])
        b_b1 = np.array([COL_X + 1.48, cy - 0.06, z_br_t + 0.02])
        fill_quad3([b_a0, b_b0, b_b1, b_a1], (*fog_c([0.045, 0.075, 0.115], dist), 0.98))
        br_rim = cam.project_polyline(np.array([b_a0, b_b0]))
        if len(br_rim) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in br_rim], base_w=1.8, z_depth=dist,
                               rgba=(*fog_c([0.34, 0.52, 0.68], dist), 0.62), rng=rng, taper=False, tremor=0.06)

    # 3. Cohesive architectural Nordic timber-slat & solid dark-steel station bench (X = -2.42, Y = 8.4..10.75)
    dist_b = float(np.linalg.norm(np.array([-2.42, 9.6, 0.5]) - cam.eye))
    c_bsteel = fog_c([0.038, 0.064, 0.102], dist_b)
    c_bsteel_hi = fog_c([0.28, 0.42, 0.56], dist_b)
    for fy in (10.52, 8.62):
        frame_segs = [
            (np.array([-2.16, fy, Z_PLAT]), np.array([-2.16, fy, 0.48]), 8.0),
            (np.array([-2.68, fy, Z_PLAT]), np.array([-2.68, fy, 0.48]), 8.0),
            (np.array([-2.14, fy, 0.46]), np.array([-2.70, fy, 0.46]), 7.5),
            (np.array([-2.68, fy, 0.46]), np.array([-2.79, fy, 0.96]), 7.5),
            (np.array([-2.16, fy, 0.46]), np.array([-2.16, fy, 0.65]), 6.2),
            (np.array([-2.16, fy, 0.65]), np.array([-2.72, fy, 0.65]), 6.2),
        ]
        for pa, pb, bw_f in frame_segs:
            fseg = cam.project_polyline(np.array([pa, pb]))
            if len(fseg) >= 2:
                stroke_catmull_rom(ctx, [(p[0], p[1]) for p in fseg], base_w=bw_f, z_depth=dist_b,
                                   rgba=(*c_bsteel, 0.98), rng=rng, taper=False, tremor=0.06)
    tseg = cam.project_polyline(np.array([[-2.42, 8.62, 0.34], [-2.42, 10.52, 0.34]]))
    if len(tseg) >= 2:
        stroke_catmull_rom(ctx, [(p[0], p[1]) for p in tseg], base_w=6.5, z_depth=dist_b,
                           rgba=(*c_bsteel, 0.96), rng=rng, taper=False, tremor=0.06)

    # Horizontal weathered Scandinavian timber seat slats (top + front thickness)
    for sx, sz, dx_w, dz_h, col_top, col_side in (
        (-2.60, 0.50, 0.072, 0.036, [0.165, 0.195, 0.235], [0.075, 0.095, 0.125]),
        (-2.42, 0.50, 0.072, 0.036, [0.195, 0.235, 0.285], [0.085, 0.108, 0.142]),
        (-2.24, 0.50, 0.072, 0.036, [0.240, 0.295, 0.365], [0.095, 0.122, 0.160]),
    ):
        fill_quad3(
            [np.array([sx - dx_w, 8.40, sz]), np.array([sx + dx_w, 8.40, sz]),
             np.array([sx + dx_w, 10.74, sz]), np.array([sx - dx_w, 10.74, sz])],
            (*fog_c(col_top, dist_b), 0.98),
        )
        fill_quad3(
            [np.array([sx + dx_w, 8.40, sz - dz_h]), np.array([sx + dx_w, 10.74, sz - dz_h]),
             np.array([sx + dx_w, 10.74, sz]), np.array([sx + dx_w, 8.40, sz])],
            (*fog_c(col_side, dist_b), 0.98),
        )
        rim_s = cam.project_polyline(np.array([[sx + dx_w, 8.40, sz], [sx + dx_w, 10.74, sz]]))
        if len(rim_s) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in rim_s], base_w=1.5, z_depth=dist_b,
                               rgba=(*c_bsteel_hi, 0.60), rng=rng, taper=False, tremor=0.06)

    # Angled UPRIGHT timber backrest boards mounted flush onto the angled steel backrest posts!
    for xb, zb, xt, zt, col_face, col_top in (
        (-2.71, 0.65, -2.74, 0.77, [0.135, 0.172, 0.220], [0.24, 0.32, 0.42]),
        (-2.75, 0.81, -2.78, 0.93, [0.165, 0.210, 0.268], [0.30, 0.42, 0.54]),
    ):
        # Upright front backrest face facing the platform (+X)
        fill_quad3(
            [np.array([xb + 0.02, 8.40, zb]), np.array([xb + 0.02, 10.74, zb]),
             np.array([xt + 0.02, 10.74, zt]), np.array([xt + 0.02, 8.40, zt])],
            (*fog_c(col_face, dist_b), 0.98),
        )
        # Thin frost-rimmed top board edge
        rim_b = cam.project_polyline(np.array([[xt + 0.02, 8.40, zt], [xt + 0.02, 10.74, zt]]))
        if len(rim_b) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in rim_b], base_w=2.0, z_depth=dist_b,
                               rgba=(*fog_c(col_top, dist_b), 0.78), rng=rng, taper=False, tremor=0.05)

    # 4. Crisp steel canopy outer fascia girder beam & glowing pale glacier-blue volumetric lamp cones!
    p_f0 = np.array([CANOPY_X1, CANOPY_Y0, CANOPY_Z1])
    p_f1 = np.array([CANOPY_X1, CANOPY_Y1, CANOPY_Z1])
    p_ft0 = np.array([CANOPY_X1, CANOPY_Y0, CANOPY_Z1 + 0.24])
    p_ft1 = np.array([CANOPY_X1, CANOPY_Y1, CANOPY_Z1 + 0.24])
    fill_quad3([p_f0, p_f1, p_ft1, p_ft0], (0.036, 0.060, 0.095, 0.98))

    lip_seg = cam.project_polyline(np.array([p_ft0, p_ft1]))
    if len(lip_seg) >= 2:
        stroke_catmull_rom(ctx, [(p[0], p[1]) for p in lip_seg], base_w=2.8, z_depth=8.0,
                           rgba=(0.64, 0.82, 0.94, 0.82), rng=rng, taper=False, tremor=0.08)

    z_lamp = float(canopy_z_at(np.array(LAMP_X))) - 0.06
    for ly in LAMP_YS:
        lu, lv, lz = cam.project(np.array([LAMP_X, ly, z_lamp]))
        if lz >= Z_NEAR and -80 < lu < W + 80:
            rad = max(24.0, 720.0 / (lz + 2.2))
            lg = cairo.RadialGradient(float(lu), float(lv), rad * 0.06, float(lu), float(lv), rad)
            lg.add_color_stop_rgba(0.0, 0.84, 0.95, 1.0, 0.34)
            lg.add_color_stop_rgba(0.38, 0.56, 0.80, 0.96, 0.13)
            lg.add_color_stop_rgba(1.0, 0.42, 0.68, 0.88, 0.0)
            ctx.set_source(lg)
            ctx.arc(float(lu), float(lv), rad, 0, math.tau)
            ctx.fill()


def draw_near_details_and_frost_motes(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
) -> None:
    """Band 3: Subtle crisp platform edge frost line, hanging canopy icicles & drifting arctic frost motes."""
    p_seg = cam.project_polyline(np.array([[PLAT_X1, 1.1, Z_PLAT + 0.005], [PLAT_X1, 46.0, Z_PLAT + 0.005]]))
    if len(p_seg) >= 2:
        stroke_catmull_rom(ctx, [(p[0], p[1]) for p in p_seg], base_w=2.5, z_depth=6.0,
                           rgba=(0.56, 0.75, 0.90, 0.64), rng=rng, taper=False, tremor=0.12)

    for iy in np.linspace(3.5, 22.0, 26):
        ic_len = float(rng.uniform(0.05, 0.17))
        if rng.random() > 0.58:
            continue
        seg = cam.project_polyline(np.array([[CANOPY_X1, iy, CANOPY_Z1], [CANOPY_X1, iy, CANOPY_Z1 - ic_len]]))
        if len(seg) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in seg], base_w=2.0, z_depth=float(iy),
                               rgba=(0.72, 0.88, 0.98, 0.68), rng=rng, taper=True, tremor=0.05)

    for _ in range(125):
        sx = float(rng.uniform(0, W))
        sy = float(rng.uniform(0, H * 0.94))
        rad = float(rng.uniform(0.85, 2.5))
        alpha = float(rng.uniform(0.14, 0.48))
        sg = cairo.RadialGradient(sx, sy, 0.0, sx, sy, rad * 1.9)
        sg.add_color_stop_rgba(0.0, 0.82, 0.93, 1.0, alpha)
        sg.add_color_stop_rgba(1.0, 0.72, 0.88, 0.98, 0.0)
        ctx.set_source(sg)
        ctx.arc(sx, sy, rad * 1.9, 0, math.tau)
        ctx.fill()


# =============================================================================
# Stage 4 — Foreground Circular Disc Bokeh Steel Girder/Railing & 35mm Optics
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


def paint_foreground_bokeh_girder(rng: np.random.Generator) -> np.ndarray:
    """Near-camera frosted dark-steel station overpass girder & railing silhouette framing the lower-right corner."""
    surf, ctx = new_cairo_2x()

    # Beveled steel corner stanchion post rising along far-right lower frame with angled architectural top cap
    ctx.set_source_rgba(0.014, 0.028, 0.048, 0.98)
    ctx.move_to(W - 88, H + 30)
    ctx.line_to(W - 122, H - 355)
    ctx.line_to(W + 30, H - 418)
    ctx.line_to(W + 30, H + 30)
    ctx.close_path()
    ctx.fill()

    rails = [
        ([(W + 30, H - 395), (W - 225, H - 285), (W - 510, H - 165)], 26.0),
        ([(W + 30, H - 245), (W - 260, H - 115), (W - 490, H + 15)], 18.0),
        ([(W - 118, H - 320), (W - 335, H + 28)], 22.0),
        ([(W - 355, H - 222), (W - 385, H + 30)], 18.0),
    ]
    for rpts, rw in rails:
        stroke_catmull_rom(ctx, rpts, base_w=rw, z_depth=1.0, rgba=(0.014, 0.028, 0.050, 0.98), rng=rng, taper=False, tremor=0.22)

    # Crisp frost-kissed glacier-blue specular rim along top beveled flange
    stroke_catmull_rom(
        ctx,
        [(W + 25, H - 408), (W - 228, H - 298), (W - 512, H - 177)],
        base_w=6.5, z_depth=1.0, rgba=(0.48, 0.70, 0.86, 0.76), rng=rng, taper=False, tremor=0.24,
    )
    stroke_catmull_rom(
        ctx,
        [(W - 120, H - 352), (W - 88, H + 20)],
        base_w=4.8, z_depth=1.0, rgba=(0.34, 0.54, 0.72, 0.58), rng=rng, taper=False, tremor=0.18,
    )

    for rx, ry in [(W - 106, H - 320), (W - 102, H - 280), (W - 98, H - 240), (W - 165, H - 298), (W - 205, H - 280)]:
        ctx.set_source_rgba(0.30, 0.48, 0.64, 0.65)
        ctx.arc(rx, ry, 5.0, 0, math.tau)
        ctx.fill()

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
    tint = np.array([0.96, 1.02, 1.08], dtype=np.float32)
    hal = (
        0.24 * gaussian_blur_2d(hl_src, 6.5)
        + 0.17 * gaussian_blur_2d(hl_src, 24.0)
        + 0.11 * gaussian_blur_2d(hl_src, 72.0)
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
    seed: int = 202602,
    out_dir: pathlib.Path | None = None,
    webp_targets: list[pathlib.Path] | None = None,
) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    cam = Camera3D(W, H, eye=(-0.55, 0.0, 1.88), target=(-2.68, 16.0, 1.66), fov_x_deg=58.0)
    print(f"[fjord-02] Camera VP: ({cam.vp_x:.1f}, {cam.vp_y:.1f}) -> {cam.vp_x / W * 100:.1f}% width, {cam.vp_y / H * 100:.1f}% height")

    canvas, fog_rgb = paint_sky_mountains_and_bay(cam, rng)

    ballast_uv = paint_track_ballast_uv(rng)
    edge_wall_uv = paint_platform_edge_wall_uv(rng)
    platform_uv = paint_voronoi_platform_uv(rng)
    canopy_uv = paint_steel_canopy_underside_uv(rng)

    canvas = cam.warp_plane_over(
        canvas, ballast_uv,
        p0=np.array([PLAT_X1, 1.0, Z_TRACK]), e1=np.array([4.10 - PLAT_X1, 0.0, 0.0]), e2=np.array([0.0, 58.0, 0.0]),
        fog_rgb=fog_rgb, fog_start=6.0, fog_density=0.025,
    )
    canvas = cam.warp_plane_over(
        canvas, edge_wall_uv,
        p0=np.array([PLAT_X1, PLAT_Y0, Z_TRACK]), e1=np.array([0.0, PLAT_Y1 - PLAT_Y0, 0.0]), e2=np.array([0.0, 0.0, Z_PLAT - Z_TRACK]),
        fog_rgb=fog_rgb, fog_start=6.0, fog_density=0.026,
    )
    canvas = cam.warp_plane_over(
        canvas, platform_uv,
        p0=np.array([PLAT_X0, PLAT_Y0, Z_PLAT]), e1=np.array([PLAT_X1 - PLAT_X0, 0.0, 0.0]), e2=np.array([0.0, PLAT_Y1 - PLAT_Y0, 0.0]),
        fog_rgb=fog_rgb, fog_start=6.0, fog_density=0.025,
    )
    canvas = cam.warp_plane_over(
        canvas, canopy_uv,
        p0=np.array([CANOPY_X0, CANOPY_Y0, CANOPY_Z0]),
        e1=np.array([CANOPY_X1 - CANOPY_X0, 0.0, CANOPY_Z1 - CANOPY_Z0]),
        e2=np.array([0.0, CANOPY_Y1 - CANOPY_Y0, 0.0]),
        fog_rgb=fog_rgb, fog_start=6.0, fog_density=0.025,
    )

    # Stage 3 Band 1 (Left parapet wall, continuous gleaming steel rails, viaduct railing & far catenary poles/cables)
    s_far, c_far = new_cairo_2x()
    draw_left_parapet_wall(c_far, cam, rng, fog_rgb)
    draw_continuous_rails_and_viaduct_railing(c_far, cam, rng, fog_rgb)
    draw_catenary_masts_and_cables(c_far, cam, rng, fog_rgb, y_min=19.0, y_max=60.0)
    canvas = over_premul(canvas, finish_cairo_2x(s_far))

    # Stage 3 Band 2 (Midground steel canopy H-columns, grounded glass timetable shelter, cohesive Nordic bench & lamp cones)
    s_mid, c_mid = new_cairo_2x()
    draw_station_columns_shelter_and_bench(c_mid, cam, rng, fog_rgb)
    mid_1x = finish_cairo_2x(s_mid)
    m_mot = fbm_gouache(H, W, rng, scales=((64, 60), (24, 26), (10, 9), (4.5, 4.5)))
    ma = mid_1x[:, :, 3:4]
    mid_1x[:, :, :3] = np.clip(mid_1x[:, :, :3] + ma * (m_mot * 0.014)[..., None], 0.0, 1.2)
    canvas = over_premul(canvas, mid_1x)

    # Stage 3 Band 3 (Near catenary mast & cables, platform frost rim, icicles & floating frost motes)
    s_near, c_near = new_cairo_2x()
    draw_catenary_masts_and_cables(c_near, cam, rng, fog_rgb, y_min=1.0, y_max=19.6)
    draw_near_details_and_frost_motes(c_near, cam, rng)
    canvas = over_premul(canvas, finish_cairo_2x(s_near))

    # Stage 4: Lower-right foreground circular-bokeh frosted steel girder/railing + 35mm analog optics
    fg_girder = paint_foreground_bokeh_girder(rng)
    canvas = over_premul(canvas, fg_girder)
    final_rgb = apply_35mm_finish(canvas, rng)

    elapsed = time.perf_counter() - t0
    print(f"[fjord-02] Rendered {W}x{H} in {elapsed:.2f}s")

    u8 = np.round(np.clip(final_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    full_im = Image.fromarray(u8, mode="RGB")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        full_im.save(out_dir / "w02_full.png", optimize=True)
        crops = {
            "w02_crop_aperture.png": (int(W * 0.42), int(H * 0.14)),     # Canopy lamps + catenary cables + craggy snowy fjord mountains
            "w02_crop_midground.png": (int(W * 0.18), int(H * 0.56)),    # Rain-and-frost Voronoi basalt paving + bench + shelter + lamp reflections
            "w02_crop_foreground.png": (W - 600, H - 450),               # Lower-right circular disc bokeh frosted steel girder/railing + wet rails
        }
        for cname, (cx0, cy0) in crops.items():
            x0 = max(0, min(W - 600, cx0))
            y0 = max(0, min(H - 450, cy0))
            full_im.crop((x0, y0, x0 + 600, y0 + 450)).save(out_dir / cname)

    if webp_targets:
        for wp in webp_targets:
            wp.parent.mkdir(parents=True, exist_ok=True)
            full_im.save(wp, format="WEBP", quality=90, method=6)
            print(f"[fjord-02] Exported 2048x1152 WebP: {wp} ({wp.stat().st_size / 1024:.1f} KiB)")

    return final_rgb, {"elapsed": elapsed, "vp_ratio": cam.vp_x / W}


def main() -> None:
    ap = argparse.ArgumentParser(description="Render Fjord 02 — Glacier Station Overpass (2048x1152)")
    ap.add_argument("--seed", type=int, default=202602)
    ap.add_argument("--out-dir", type=pathlib.Path, default=None)
    ap.add_argument("--no-export", action="store_true", help="Skip exporting repository WebP assets")
    args = ap.parse_args()

    targets = None if args.no_export else [
        REPO_ROOT / "session/themes/fjord/walls/wall-02.webp",
        REPO_ROOT / "docs/assets/walls/fjord-02.webp",
    ]
    render_scene(seed=args.seed, out_dir=args.out_dir, webp_targets=targets)


if __name__ == "__main__":
    main()

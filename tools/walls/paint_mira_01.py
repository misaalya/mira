#!/usr/bin/env python3
"""
Mira 01 — "Obsidian Rain Alley" (2048 x 1152)

Pure-Python 3D-grounded hand-painted wallpaper pipeline (zero image models,
zero external image assets, 100% deterministic):
  Stage 1: Off-axis metric Camera3D (VP ~60.6% width), exact near-plane-safe
           3x3 UV-to-screen homographies with z_cam > 0.05 masking and
           exponential rain-mist fog, plus vectorized irregular Voronoi
           (d2 - d1) wet basalt cobblestones and board-formed concrete walls.
  Stage 2: Unified 3D directional moonlight/skylight vector (SUN_DIR) traced
           backward with 6-step pitched-eave height refinement + exact 3D
           point-light solver for the hanging acid-yellow (#d9d84a) paper
           lantern & warm shop transom, paired with Fresnel capillary-rippled
           sky reflections and 3D plane-mirrored wet cobblestone reflections.
  Stage 3: 4 depth-interleaved 2x supersampled PyCairo bands (Lanczos-downsampled)
           with tapered Catmull-Rom splines, weathered concrete steps, glowing
           ribbed chochin lantern, parked silhouette bicycle, utility poles &
           catenary overhead power lines, rain streaks, and isotropic
           fbm_gouache mottling, flat-brush dabs, paper tooth & shadow bounce.
  Stage 4: Foreground overhanging eave & wet rain-bough silhouette in the
           upper-right corner convolved in premultiplied RGBA with a circular
           aperture disc bokeh kernel (r = 7.8) + 1 px opposite R/B shift,
           followed by 35mm multi-scale halation, radial CA, optical vignette,
           shadow lift + C1 tanh highlight shoulder, luminance-weighted
           silver-halide grain, and faint analog YIQ transfer patina.
  Stage 5: Deterministic render (<60s CPU) exporting 2048x1152 WebP assets
           plus full PNG and 1:1 (600x450) macro audit crops.
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

# Mira monochrome + single warm acid-yellow (#d9d84a) light accent
ACCENT_YELLOW = np.array([0.851, 0.847, 0.290], dtype=np.float32)  # #d9d84a
ACCENT_CORE = np.array([0.968, 0.962, 0.635], dtype=np.float32)
ACCENT_WARM = np.array([0.780, 0.755, 0.220], dtype=np.float32)

# Stage 2: Unified normalized 3D nocturnal skylight vector (z < 0)
_raw_sun = np.array([0.34, -0.20, -0.919], dtype=np.float64)
SUN_DIR = _raw_sun / np.linalg.norm(_raw_sun)

# Primary hero point-light: hanging acid-yellow paper lantern & warm transom center
LANTERN_POS = np.array([-1.30, 11.05, 2.02], dtype=np.float64)
TRANSOM_POS = np.array([-2.16, 13.30, 2.38], dtype=np.float64)

# Architectural world planes (meters)
X_WALL_L = -2.20
X_WALL_R = 2.65
Y_WALL_END = 27.5
Z_GROUND = 0.0


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
    if sigma <= 4.0:
        return gaussian_blur_1d(gaussian_blur_1d(arr, sigma, axis=1), sigma, axis=0)
    h, w = arr.shape[:2]
    factor = 2 if sigma < 12 else (4 if sigma < 40 else 8)
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
        return np.asarray(
            Image.fromarray(arr.astype(np.float32), mode="F").resize((out_w, out_h), Image.Resampling.BICUBIC),
            dtype=np.float32,
        )
    out = np.empty((out_h, out_w, arr.shape[2]), dtype=np.float32)
    for c in range(arr.shape[2]):
        out[:, :, c] = np.asarray(
            Image.fromarray(arr[:, :, c].astype(np.float32), mode="F").resize((out_w, out_h), Image.Resampling.BICUBIC),
            dtype=np.float32,
        )
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
    return (
        (src[y0, x0] * (1.0 - fx) + src[y0, x1] * fx) * (1.0 - fy)
        + (src[y1, x0] * (1.0 - fx) + src[y1, x1] * fx) * fy
    )


def fbm_gouache(
    h: int,
    w: int,
    rng: np.random.Generator,
    scales=((72, 68), (26, 29), (10.5, 9.5), (4.5, 4.8)),
) -> np.ndarray:
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


def flat_brush_dabs(h: int, w: int, rng: np.random.Generator, count: int = 90) -> np.ndarray:
    qh, qw = max(16, h // 4), max(16, w // 4)
    arr_sum = np.zeros((qh, qw), dtype=np.float32)
    s_one = cairo.ImageSurface(cairo.FORMAT_A8, qw, qh)
    c1 = cairo.Context(s_one)
    stride = s_one.get_stride()
    for _ in range(count):
        cx = float(rng.uniform(0, qw))
        cy = float(rng.uniform(0, qh))
        ang = float(rng.uniform(0, math.pi))
        rx = float(rng.uniform(qw * 0.032, qw * 0.105))
        ry = float(rng.uniform(rx * 0.28, rx * 0.62))
        n_pts = int(rng.integers(5, 8))
        strength = float(rng.uniform(0.18, 0.44)) * (1.0 if rng.random() > 0.5 else -1.0)
        c1.save()
        c1.set_operator(cairo.OPERATOR_CLEAR)
        c1.paint()
        c1.restore()
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
        fog_start: float = 6.5,
        fog_density: float = 0.052,
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

        v_f = valid[..., None].astype(np.float32)
        alpha = sampled[:, :, 3:4] * v_f
        rgb_valid = sampled[:, :, :3] * v_f

        fg_rgb = rgb_valid * (1.0 - fog_t[..., None]) + (fog_rgb[None, None, :] * alpha) * fog_t[..., None]
        patch = np.concatenate([fg_rgb, alpha], axis=-1)
        dst_rgb[y0:y1, x0:x1] = over_premul(dst_rgb[y0:y1, x0:x1], patch)
        return dst_rgb


# =============================================================================
# Stage 2 — Unified 3D Eave/Wall Light Solver & Lantern Illumination
# =============================================================================

def trace_eave_moon_visibility(wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> np.ndarray:
    """
    Trace world points backward along -SUN_DIR onto the pitched shop eave
    (X in [-2.20, -1.16], Y in [9.8, 16.4], pitched Z from 3.16 down to 2.86)
    and upper left roof overhang using 6 fixed height-refinement steps.
    """
    inv_sz = 1.0 / (-SUN_DIR[2])
    z_eave = np.full_like(wx, 3.00, dtype=np.float64)
    for _ in range(6):
        t = np.maximum(0.0, (z_eave - wz) * inv_sz)
        rx = wx - SUN_DIR[0] * t
        slope_t = np.clip((rx - X_WALL_L) / 1.06, 0.0, 1.0)
        z_eave = 3.16 - 0.30 * slope_t

    t = np.maximum(0.0, (z_eave - wz) * inv_sz)
    rx = wx - SUN_DIR[0] * t
    ry = wy - SUN_DIR[1] * t
    under_eave = (wz < z_eave) & (rx >= X_WALL_L - 0.1) & (rx <= X_WALL_L + 1.06) & (ry >= 9.7) & (ry <= 16.4)

    t_roof = np.maximum(0.0, (5.05 - wz) * inv_sz)
    rrx = wx - SUN_DIR[0] * t_roof
    under_roof = (wz < 5.05) & (rrx <= X_WALL_L + 0.58)

    t_rw = np.maximum(0.0, (2.35 - wz) * inv_sz)
    rwx = wx - SUN_DIR[0] * t_rw
    under_rwall = (wz < 2.35) & (rwx >= X_WALL_R - 0.05) & (wx < X_WALL_R - 0.02)

    vis = np.ones_like(wx, dtype=np.float32)
    vis = np.where(under_roof, vis * 0.56, vis)
    vis = np.where(under_eave, vis * 0.28, vis)
    vis = np.where(under_rwall, vis * 0.45, vis)
    return vis


def eval_lantern_and_transom_light(
    wx: np.ndarray,
    wy: np.ndarray,
    wz: np.ndarray,
    nx: float,
    ny: float,
    nz: float,
) -> np.ndarray:
    """Physically grounded acid-yellow point-light irradiance from lantern + shop transom."""
    lx = LANTERN_POS[0] - wx
    ly = LANTERN_POS[1] - wy
    lz = LANTERN_POS[2] - wz
    d2 = lx * lx + ly * ly + lz * lz + 0.22
    d = np.sqrt(d2)
    ndotl = np.clip((lx * nx + ly * ny + lz * nz) / d, 0.0, 1.0)
    wrap = 0.25 + 0.75 * ndotl
    irr_l = (3.05 / (d2 + 0.62)) * wrap

    tx = TRANSOM_POS[0] - wx
    ty = TRANSOM_POS[1] - wy
    tz = TRANSOM_POS[2] - wz
    td2 = tx * tx + ty * ty + tz * tz + 0.38
    td = np.sqrt(td2)
    t_ndotl = np.clip((tx * nx + ty * ny + tz * nz) / td, 0.0, 1.0)
    emit_fwd = np.clip((wx - X_WALL_L) / (td + 1e-4), 0.0, 1.0)
    irr_t = (1.75 / (td2 + 1.10)) * (0.20 + 0.80 * t_ndotl) * (0.35 + 0.65 * emit_fwd)

    return np.clip(irr_l + irr_t, 0.0, 2.3).astype(np.float32)


def mirror_points_z0(pts: np.ndarray) -> np.ndarray:
    """Exact 3D reflection across ground plane Z = 0: pts - 2 * dot(pts - p0, n) * n."""
    n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    p0 = np.array([0.0, 0.0, Z_GROUND], dtype=np.float64)
    dots = np.sum((pts - p0) * n, axis=-1, keepdims=True)
    return pts - 2.0 * dots * n


# =============================================================================
# Stage 1 & 2 — Nocturnal Silver-Obsidian Sky, Concrete Walls & Voronoi Cobbles
# =============================================================================

def paint_nocturnal_mist_sky(cam: Camera3D, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Moody monochrome obsidian zenith fading to luminous silver-slate rain mist at the alley corner."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = xx / W
    v = yy / H
    vh = float(cam.horizon_y / H)

    mottle = fbm_gouache(H, W, rng)
    dabs = flat_brush_dabs(H, W, rng, count=88)
    tooth = paper_tooth(H, W, rng)

    # Luminous nocturnal silver-slate overcast sky so dark power lines & foreground bokeh read crisply!
    c_zenith = np.array([0.044, 0.048, 0.056], dtype=np.float32)
    c_mid = np.array([0.152, 0.160, 0.175], dtype=np.float32)
    c_mist = np.array([0.315, 0.325, 0.342], dtype=np.float32)
    c_ground_base = np.array([0.026, 0.028, 0.032], dtype=np.float32)

    sky_t = np.clip(v / max(0.2, vh + 0.04), 0.0, 1.0)
    sky_curve = np.power(sky_t, 1.32)
    sky = (
        c_zenith[None, None, :] * np.power(1.0 - sky_curve, 1.15)[..., None]
        + c_mid[None, None, :] * (2.05 * sky_curve * (1.0 - sky_curve))[..., None]
        + c_mist[None, None, :] * np.power(sky_curve, 1.48)[..., None]
    )

    # Luminous silver city-mist bloom opening down the alley canyon & upper-right behind power lines/bough
    du = (u - (cam.vp_x / W + 0.05)) * 1.15
    dv = (v - (vh - 0.14)) * 1.85
    vp_glow = np.exp(-(du * du * 2.8 + dv * dv * 5.2)).astype(np.float32)
    sky += vp_glow[..., None] * np.array([0.165, 0.172, 0.184], dtype=np.float32)

    # Drifting painted silver-grey rain cloud breaks (isotropic gouache)
    cloud_mask = np.clip((vh - v + 0.06) / 0.44, 0.0, 1.0)
    cloud = np.clip(0.52 + 0.36 * mottle + 0.20 * dabs, 0.0, 1.0) * cloud_mask
    sky += (cloud * 0.065)[..., None] * np.array([0.85, 0.88, 0.94], dtype=np.float32)

    below = np.clip((v - vh) / 0.06, 0.0, 1.0)
    canvas = sky * (1.0 - below[..., None]) + c_ground_base[None, None, :] * below[..., None]
    canvas += (mottle * 0.008 + dabs * 0.006 + tooth * 0.004)[..., None]

    fog_rgb = np.array([0.148, 0.155, 0.166], dtype=np.float32)
    return np.clip(canvas, 0.0, 1.25), fog_rgb


def paint_voronoi_cobblestones_uv(
    rng: np.random.Generator,
    x_span: tuple[float, float],
    y_span: tuple[float, float],
    tw: int = 1024,
    th: int = 1280,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Wet irregular Voronoi basalt cobblestones in physical meters (d2 - d1 crevices),
    complete with Stage 2 traced eave shadows, acid-yellow lantern spill, silver
    skylight wet bevel glints, and a physically coherent rainwater puddle mask.
    """
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wx = x_span[0] + u * (x_span[1] - x_span[0])
    wy = y_span[0] + v * (y_span[1] - y_span[0])
    wz = np.full_like(wx, Z_GROUND)

    nx = resize_bicubic(rng.standard_normal((th // 32, tw // 28)).astype(np.float32), tw, th) * 0.042
    ny = resize_bicubic(rng.standard_normal((th // 32, tw // 28)).astype(np.float32), tw, th) * 0.048
    px = wx + nx
    py = wy + ny

    cell_x, cell_y = 0.34, 0.42
    ix = np.floor(px / cell_x).astype(np.int32)
    iy = np.floor(py / cell_y).astype(np.int32)

    jx_tab = rng.uniform(0.18, 0.82, size=512)
    jy_tab = rng.uniform(0.18, 0.82, size=512)
    tone_tab = rng.uniform(-0.032, 0.038, size=512)
    wet_tab = rng.uniform(0.70, 1.0, size=512)

    d1 = np.full((th, tw), 1e9, dtype=np.float64)
    d2 = np.full((th, tw), 1e9, dtype=np.float64)
    best_id = np.zeros((th, tw), dtype=np.int32)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cx, cy = ix + dx, iy + dy
            h_id = ((cx * 73856093) ^ (cy * 19349663)) & 511
            sx = (cx + jx_tab[h_id]) * cell_x
            sy = (cy + jy_tab[h_id]) * cell_y
            dist = np.hypot(px - sx, py - sy)
            closer = dist < d1
            d2 = np.where(closer, d1, np.minimum(d2, dist))
            d1 = np.where(closer, dist, d1)
            best_id = np.where(closer, h_id, best_id)

    edge_dist = (d2 - d1).astype(np.float32)
    stone_mask = np.clip((edge_dist - 0.012) / 0.025, 0.0, 1.0)
    crown_rim = np.exp(-((edge_dist - 0.040) / 0.020) ** 2).astype(np.float32) * stone_mask

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=88)
    tooth = paper_tooth(th, tw, rng)
    sv = tone_tab[best_id].astype(np.float32)
    swet = wet_tab[best_id].astype(np.float32)

    # Rainwater puddle basins
    puddle_noise = resize_bicubic(rng.standard_normal((max(6, th // 72), max(6, tw // 64))).astype(np.float32), tw, th)
    puddle_detail = resize_bicubic(rng.standard_normal((max(8, th // 28), max(8, tw // 26))).astype(np.float32), tw, th)
    alley_trough = np.exp(-((wx + 0.15) / 1.95) ** 2).astype(np.float32)
    puddle = np.clip(0.40 + 0.42 * alley_trough + 0.28 * puddle_noise + 0.14 * puddle_detail, 0.0, 1.0)
    puddle = np.clip((puddle - 0.34) / 0.32, 0.0, 1.0)

    moon_vis = trace_eave_moon_visibility(wx, wy, wz)
    lan_irr = eval_lantern_and_transom_light(wx, wy, wz, 0.0, 0.0, 1.0)

    curb_l = np.clip(1.0 - np.abs(wx - (X_WALL_L + 0.38)) / 0.055, 0.0, 1.0).astype(np.float32)
    curb_r = np.clip(1.0 - np.abs(wx - (X_WALL_R - 0.32)) / 0.055, 0.0, 1.0).astype(np.float32)
    gutter = np.maximum(curb_l, curb_r)

    base_v = (0.056 + sv * 0.50 + mottle * 0.011 + dabs * 0.008) * (0.68 + 0.48 * moon_vis)
    base_v *= (1.0 - 0.26 * puddle)
    crevice_v = 0.013 + 0.006 * puddle

    v_col = stone_mask * base_v + (1.0 - stone_mask) * crevice_v
    v_col = v_col * (1.0 - 0.35 * gutter)

    rgb = np.stack([v_col * 0.96, v_col * 0.99, v_col * 1.05], axis=-1).astype(np.float32)

    # Cool silver skylight wet sheen along basalt stone crowns
    rgb += (crown_rim * moon_vis * swet * 0.068)[..., None] * np.array([0.88, 0.92, 0.99], dtype=np.float32)

    # Acid-yellow (#d9d84a) warm diffuse + wet stone specular spill from hanging lantern & transom
    lan_spec = (0.45 * stone_mask + 0.95 * crown_rim * swet) * np.power(np.clip(lan_irr, 0.0, 1.5), 1.15)
    rgb += (lan_irr * (0.14 * stone_mask + 0.04) + lan_spec * 0.14)[..., None] * ACCENT_YELLOW[None, None, :]

    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    rgb += np.clip((0.28 - lum) / 0.28, 0.0, 1.0)[..., None] * np.array([0.009, 0.009, 0.007], dtype=np.float32)
    rgb += tooth[..., None] * 0.005

    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.2)
    rgba[:, :, 3] = 1.0

    wet_mask = np.clip(puddle * 0.88 + stone_mask * swet * 0.42 * (1.0 - puddle), 0.0, 1.0)
    wet_mask *= (0.25 + 0.75 * stone_mask + 0.75 * puddle)
    wet_rgba = np.zeros((th, tw, 4), dtype=np.float32)
    wet_rgba[:, :, 0] = np.clip(wet_mask, 0.0, 1.0)
    wet_rgba[:, :, 1] = puddle
    wet_rgba[:, :, 2] = stone_mask
    wet_rgba[:, :, 3] = 1.0
    return rgba, wet_rgba


def paint_left_concrete_facade_uv(
    rng: np.random.Generator,
    y_span: tuple[float, float],
    z_span: tuple[float, float],
    tw: int = 1280,
    th: int = 768,
) -> np.ndarray:
    """
    Weathered Japanese board-formed concrete backstreet facade with formwork seams,
    tie-rod holes, rain staining, dark recessed shop doorway, glowing warm acid-yellow
    (#d9d84a) wooden-lattice transom window, and Stage 2 traced eave shadow + lantern glow.
    """
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wy = y_span[0] + u * (y_span[1] - y_span[0])
    wz = z_span[0] + v * (z_span[1] - z_span[0])
    wx = np.full_like(wy, X_WALL_L)

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=88)
    tooth = paper_tooth(th, tw, rng)

    panel_y = np.mod(wy, 1.80)
    panel_z = np.mod(wz, 0.90)
    seam_h = np.exp(-((np.minimum(panel_z, 0.90 - panel_z)) / 0.011) ** 2).astype(np.float32)
    seam_v = np.exp(-((np.minimum(panel_y, 1.80 - panel_y)) / 0.013) ** 2).astype(np.float32)
    plank_z = np.mod(wz, 0.18)
    plank_seam = np.exp(-((np.minimum(plank_z, 0.18 - plank_z)) / 0.005) ** 2).astype(np.float32)
    plank_id = np.floor(wz / 0.18).astype(np.int32)
    plank_tone = (((plank_id * 1103515245 + 12345) & 255).astype(np.float32) / 255.0 - 0.5) * 0.018

    ty_d = np.abs(np.mod(wy + 0.45, 0.90) - 0.45)
    tz_d = np.abs(np.mod(wz + 0.225, 0.45) - 0.225)
    tie_hole = np.exp(-((ty_d * ty_d + tz_d * tz_d) / (0.024 * 0.024))).astype(np.float32)

    damp_bot = np.exp(-np.maximum(0.0, wz - 0.12) / 0.62).astype(np.float32)
    rain_streak = resize_bicubic(rng.standard_normal((max(6, th // 96), max(12, tw // 12))).astype(np.float32), tw, th)
    rain_streak = np.clip(rain_streak * 0.42 + 0.22, 0.0, 1.0) * np.clip((wz - 0.4) / 3.2, 0.0, 1.0)

    moon_vis = trace_eave_moon_visibility(wx, wy, wz)
    lan_irr = eval_lantern_and_transom_light(wx, wy, wz, 1.0, 0.0, 0.0)

    val = (0.114 + plank_tone + mottle * 0.017 + dabs * 0.011) * (0.64 + 0.50 * moon_vis)
    val *= (1.0 - 0.34 * damp_bot - 0.11 * rain_streak)
    val *= (1.0 - 0.42 * seam_h - 0.36 * seam_v - 0.15 * plank_seam - 0.52 * tie_hole)

    coping = (wz >= 4.85) & (wz <= 5.18)
    val = np.where(coping, val * 1.24 + 0.020, val)

    rgb = np.stack([val * 0.96, val * 0.98, val * 1.03], axis=-1).astype(np.float32)
    rgb += (lan_irr * 0.23 * (1.0 - 0.35 * seam_h))[..., None] * ACCENT_YELLOW[None, None, :]

    # Shopfront recessed timber doorway (Y in [12.02, 14.58], Z in [0.46, 2.74])
    in_frame = (wy >= 11.94) & (wy <= 14.66) & (wz >= 0.44) & (wz <= 2.82)
    in_door = (wy >= 12.06) & (wy <= 14.54) & (wz >= 0.48) & (wz <= 2.02)
    in_transom = (wy >= 12.08) & (wy <= 14.52) & (wz >= 2.08) & (wz <= 2.70)

    frame_col = np.array([0.022, 0.022, 0.024], dtype=np.float32)
    door_slat = 0.5 + 0.5 * np.sin(wy * 58.0).astype(np.float32)
    door_col = np.stack([
        0.028 + 0.010 * door_slat + lan_irr * 0.06,
        0.028 + 0.010 * door_slat + lan_irr * 0.06,
        0.026 + 0.008 * door_slat + lan_irr * 0.02,
    ], axis=-1).astype(np.float32)
    in_door_pane = (wy >= 12.26) & (wy <= 14.34) & (wz >= 1.18) & (wz <= 1.88) & (np.abs(wy - 13.30) > 0.06)
    door_col = np.where(in_door_pane[..., None], door_col + ACCENT_WARM[None, None, :] * 0.14, door_col)

    rgb = np.where(in_frame[..., None], frame_col, rgb)
    rgb = np.where(in_door[..., None], door_col, rgb)

    # Glowing warm acid-yellow (#d9d84a) shoji / wooden-mullion shop transom
    tu = np.clip((wy - 12.08) / (14.52 - 12.08), 0.0, 1.0).astype(np.float32)
    tv = np.clip((wz - 2.08) / (2.70 - 2.08), 0.0, 1.0).astype(np.float32)
    tr_glow = np.exp(-((tu - 0.46) ** 2 * 2.2 + (tv - 0.35) ** 2 * 1.8)).astype(np.float32)
    paper_var = 0.92 + 0.08 * mottle
    transom_col = (
        ACCENT_WARM[None, None, :] * 0.58
        + ACCENT_YELLOW[None, None, :] * (0.46 * tr_glow)[..., None]
        + ACCENT_CORE[None, None, :] * (0.36 * np.power(tr_glow, 2.0))[..., None]
    ) * paper_var[..., None]

    v_bar = np.abs(np.mod(tu * 6.0 + 0.5, 1.0) - 0.5) < 0.062
    h_bar = np.abs(tv - 0.50) < 0.048
    mullion = v_bar | h_bar
    transom_col = np.where(mullion[..., None], np.array([0.040, 0.038, 0.024], dtype=np.float32), transom_col)
    rgb = np.where(in_transom[..., None], transom_col, rgb)

    # Upper second-floor unlit dark window
    in_win2 = (wy >= 5.35) & (wy <= 7.25) & (wz >= 3.22) & (wz <= 4.38)
    win2_glass = (wy >= 5.48) & (wy <= 7.12) & (wz >= 3.34) & (wz <= 4.26)
    win2_refl = np.clip(0.068 + 0.055 * (wz - 3.34), 0.04, 0.14).astype(np.float32)
    rgb = np.where(in_win2[..., None], np.array([0.024, 0.026, 0.030], dtype=np.float32), rgb)
    rgb = np.where(
        win2_glass[..., None],
        np.stack([win2_refl * 0.92, win2_refl * 0.96, win2_refl * 1.05], axis=-1),
        rgb,
    )
    rgb = np.where((win2_glass & (np.abs(wy - 6.30) < 0.04))[..., None], np.array([0.02, 0.022, 0.025], dtype=np.float32), rgb)

    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    rgb += np.clip((0.28 - lum) / 0.28, 0.0, 1.0)[..., None] * np.array([0.009, 0.009, 0.007], dtype=np.float32)
    rgb += tooth[..., None] * 0.005

    roof_h = np.where(wy < 16.5, 5.18, np.where(wy < 22.0, 4.45, 5.05))
    alpha = np.clip((roof_h - wz) / 0.06, 0.0, 1.0).astype(np.float32)

    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.38) * alpha[..., None]
    rgba[:, :, 3] = alpha
    return rgba


def paint_right_retaining_wall_uv(
    rng: np.random.Generator,
    y_span: tuple[float, float],
    z_span: tuple[float, float],
    tw: int = 1024,
    th: int = 512,
) -> np.ndarray:
    """Quiet board-formed concrete garden/retaining wall along right alley edge catching subtle lantern bounce."""
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wy = y_span[0] + u * (y_span[1] - y_span[0])
    wz = z_span[0] + v * (z_span[1] - z_span[0])
    wx = np.full_like(wy, X_WALL_R)

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=82)
    tooth = paper_tooth(th, tw, rng)

    seam_h = np.exp(-((np.minimum(np.mod(wz, 0.78), 0.78 - np.mod(wz, 0.78))) / 0.012) ** 2).astype(np.float32)
    seam_v = np.exp(-((np.minimum(np.mod(wy, 1.95), 1.95 - np.mod(wy, 1.95))) / 0.015) ** 2).astype(np.float32)
    ty_d = np.abs(np.mod(wy + 0.48, 0.975) - 0.48)
    tz_d = np.abs(np.mod(wz + 0.20, 0.78) - 0.39)
    tie_hole = np.exp(-((ty_d * ty_d + tz_d * tz_d) / (0.025 * 0.025))).astype(np.float32)

    damp_bot = np.exp(-np.maximum(0.0, wz - 0.08) / 0.55).astype(np.float32)
    lan_irr = eval_lantern_and_transom_light(wx, wy, wz, -1.0, 0.0, 0.0)

    val = 0.092 + mottle * 0.015 + dabs * 0.010
    val *= (1.0 - 0.30 * damp_bot)
    val *= (1.0 - 0.38 * seam_h - 0.32 * seam_v - 0.45 * tie_hole)
    coping = np.clip((wz - 2.18) / 0.14, 0.0, 1.0).astype(np.float32)
    val += coping * 0.055

    rgb = np.stack([val * 0.96, val * 0.99, val * 1.04], axis=-1).astype(np.float32)
    rgb += (lan_irr * 0.17)[..., None] * ACCENT_YELLOW[None, None, :]

    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    rgb += np.clip((0.28 - lum) / 0.28, 0.0, 1.0)[..., None] * np.array([0.008, 0.008, 0.006], dtype=np.float32)
    rgb += tooth[..., None] * 0.005

    wall_top = np.where(wy < 18.0, 2.32, 3.25)
    alpha = np.clip((wall_top - wz) / 0.06, 0.0, 1.0).astype(np.float32)
    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.1) * alpha[..., None]
    rgba[:, :, 3] = alpha
    return rgba


def paint_far_corner_wall_uv(rng: np.random.Generator, tw: int = 640, th: int = 512) -> np.ndarray:
    """Misty far cross-street concrete building closing the backstreet corner at Y = Y_WALL_END."""
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    mottle = fbm_gouache(th, tw, rng)
    tooth = paper_tooth(th, tw, rng)

    val = 0.088 + 0.022 * v + mottle * 0.014 + tooth * 0.005
    wx = -3.2 + u * 7.2
    wz = v * 6.0
    w_grid = (np.abs(np.mod(wx + 0.4, 1.4) - 0.7) < 0.32) & (wz > 2.2) & (wz < 3.6)
    val = np.where(w_grid, val * 0.60, val)
    rgb = np.stack([val * 0.96, val * 0.98, val * 1.03], axis=-1)

    roof = np.where(wx < 0.8, 5.25, 4.15)
    alpha = np.clip((roof - wz) / 0.08, 0.0, 1.0).astype(np.float32)
    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.0) * alpha[..., None]
    rgba[:, :, 3] = alpha
    return rgba


# =============================================================================
# Stage 2 — Fresnel Wet Cobblestone Reflections (Capillary Sky + Mirrored 3D)
# =============================================================================

def composite_wet_cobblestone_reflections(
    canvas: np.ndarray,
    cam: Camera3D,
    wet_screen: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Combine Fresnel-weighted capillary-rippled sky/building reflection (y_src = 2*horizon_y - y)
    with explicit 3D plane-mirrored quads (Z = 0) of the glowing acid-yellow lantern & transom!
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    wet_mask = np.clip(wet_screen[:, :, 0], 0.0, 1.0)
    puddle_mask = np.clip(wet_screen[:, :, 1], 0.0, 1.0)
    below_horizon = np.clip((yy - (cam.horizon_y + 2.0)) / 18.0, 0.0, 1.0)
    active = wet_mask * below_horizon

    # Horizontal capillary rain-puddle ripples (ONLY on water/wet-stone!)
    rip1 = resize_bicubic(rng.standard_normal((max(8, H // 4), max(8, W // 48))).astype(np.float32), W, H)
    rip2 = resize_bicubic(rng.standard_normal((max(8, H // 2), max(8, W // 22))).astype(np.float32), W, H)
    rip = 0.65 * rip1 + 0.35 * rip2

    dist_t = np.clip((yy - cam.horizon_y) / max(1.0, H - cam.horizon_y), 0.0, 1.0)
    fresnel_min, fresnel_max = 0.16, 0.76
    fresnel = fresnel_min + (fresnel_max - fresnel_min) * np.power(1.0 - dist_t, 1.4)

    amp_x = (3.8 + 7.2 * dist_t) * (0.45 + 0.55 * (1.0 - puddle_mask))
    amp_y = (2.4 + 4.6 * dist_t) * (0.55 + 0.45 * (1.0 - puddle_mask))
    map_x = xx + rip * amp_x
    map_y = np.clip(2.0 * cam.horizon_y - yy + rip * amp_y, 0.0, H - 1.001)
    refl_scene = remap_bilinear(canvas, map_x, map_y)
    refl_soft = gaussian_blur_2d(refl_scene, 3.0)
    refl_mix = refl_scene * puddle_mask[..., None] + refl_soft * (1.0 - puddle_mask[..., None])

    # Explicit 3D plane-mirrored lantern & transom glow across Z = 0
    m_lan = mirror_points_z0(LANTERN_POS[None, :])[0]
    m_tr = mirror_points_z0(TRANSOM_POS[None, :])[0]
    ul, vl, _ = cam.project(m_lan)
    ut, vt, _ = cam.project(m_tr)
    ub_l, vb_l, _ = cam.project(np.array([LANTERN_POS[0], LANTERN_POS[1], 0.0]))
    ub_t, vb_t, _ = cam.project(np.array([TRANSOM_POS[0], TRANSOM_POS[1], 0.0]))

    rx = xx + rip * (8.2 + 8.8 * dist_t)
    ry = yy + rip * 4.2

    dx_l = (rx - float(ul)) / 42.0
    dy_l = (ry - float(vl)) / 110.0
    lan_mirror_blob = np.exp(-0.5 * (dx_l * dx_l + dy_l * dy_l)).astype(np.float32)

    in_streak_y = np.clip((yy - (float(vb_l) - 18.0)) / 36.0, 0.0, 1.0)
    streak_w = 24.0 + 36.0 * np.clip((yy - float(vb_l)) / 320.0, 0.0, 1.0)
    dx_st = (rx - (float(ub_l) + (yy - float(vb_l)) * 0.028)) / streak_w
    lan_streak = np.exp(-0.5 * dx_st * dx_st) * in_streak_y * (0.48 + 0.52 * np.exp(-((yy - float(vl)) / 220.0) ** 2))

    dx_t = (rx - float(ut)) / 56.0
    dy_t = (ry - float(vt)) / 52.0
    tr_mirror = np.exp(-0.5 * (dx_t * dx_t + dy_t * dy_t)).astype(np.float32)
    tr_streak_y = np.clip((yy - (float(vb_t) - 12.0)) / 32.0, 0.0, 1.0)
    tr_streak = np.exp(-0.5 * ((rx - float(ut)) / 46.0) ** 2) * tr_streak_y * np.exp(-((yy - float(vt)) / 160.0) ** 2)

    y_refl_int = (
        lan_mirror_blob * (0.85 * puddle_mask + 0.25)
        + lan_streak * (0.58 * puddle_mask + 0.22)
        + tr_mirror * (0.62 * puddle_mask + 0.18)
        + tr_streak * (0.34 * puddle_mask + 0.12)
    ) * active
    shimmer = np.clip(0.68 + 0.55 * np.sin(yy * 0.34 + rip * 4.2), 0.25, 1.35).astype(np.float32)
    y_refl_int *= shimmer

    yellow_refl = (
        y_refl_int[..., None] * ACCENT_YELLOW[None, None, :] * 0.72
        + np.power(np.clip(y_refl_int, 0.0, 1.2), 2.1)[..., None] * ACCENT_CORE[None, None, :] * 0.45
    )

    refl_w = np.clip(active * fresnel * (0.52 + 0.48 * puddle_mask), 0.0, 0.82)
    out = canvas * (1.0 - refl_w[..., None] * 0.62) + refl_mix * (refl_w[..., None] * 0.76)
    out += yellow_refl
    return np.clip(out, 0.0, 1.45)


# =============================================================================
# Stage 3 — 2x Supersampled PyCairo Helpers & Depth-Interleaved Scene Props
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
    """Lanczos-downsample 2x Cairo premultiplied ARGB32 cleanly into 1x premul float32 RGBA."""
    surf.flush()
    buf = np.frombuffer(surf.get_data(), dtype=np.uint8).reshape(H * 2, W * 2, 4)
    rgba_premul_2x = np.stack([buf[:, :, 2], buf[:, :, 1], buf[:, :, 0], buf[:, :, 3]], axis=-1)
    out = np.empty((H, W, 4), dtype=np.float32)
    for c in range(4):
        im = Image.fromarray(rgba_premul_2x[:, :, c], mode="L").resize((W, H), Image.Resampling.LANCZOS)
        out[:, :, c] = np.asarray(im, dtype=np.float32) / 255.0
    a = np.clip(out[:, :, 3:4], 0.0, 1.0)
    rgb = np.where(a > 1e-4, np.clip(out[:, :, :3] / np.maximum(a, 1e-4), 0.0, 1.0), 0.0)
    out[:, :, :3] = rgb * a
    out[:, :, 3] = a[:, :, 0]
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
    tremor: float = 0.28,
) -> None:
    if len(pts) < 2:
        return
    decay = float(np.clip(1.0 / (1.0 + z_depth / (fade_dist * 0.45)), 0.25, 1.0))
    w_eff = base_w * decay
    ext = [pts[0]] + pts + [pts[-1]]
    dense: list[tuple[float, float]] = []
    steps = 7
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for s in range(steps):
            t = s / steps
            t2, t3 = t * t, t * t * t
            x = 0.5 * (
                (2 * p1[0])
                + (-p0[0] + p2[0]) * t
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2 * p1[1])
                + (-p0[1] + p2[1]) * t
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
            )
            if tremor > 0:
                x += float(rng.normal(0.0, tremor * decay))
                y += float(rng.normal(0.0, tremor * decay))
            dense.append((x, y))
    dense.append(pts[-1])

    ctx.set_source_rgba(*rgba)
    if not taper:
        ctx.set_line_width(max(0.45, w_eff))
        ctx.move_to(*dense[0])
        for pt in dense[1:]:
            ctx.line_to(*pt)
        ctx.stroke()
        return

    n = len(dense) - 1
    for i in range(n):
        frac = (i + 0.5) / max(1, n)
        wt = 0.35 + 0.65 * math.sin(frac * math.pi)
        ctx.set_line_width(max(0.45, w_eff * wt))
        ctx.move_to(*dense[i])
        ctx.line_to(*dense[i + 1])
        ctx.stroke()


def stroke_straight_3d(
    ctx: cairo.Context,
    cam: Camera3D,
    pts3: list[np.ndarray],
    base_w: float,
    z_depth: float,
    rgba: tuple[float, float, float, float],
    fade_dist: float = 24.0,
) -> None:
    """Crisp straight architectural 3D polyline without spline overshoot."""
    proj = cam.project_polyline(np.array(pts3, dtype=np.float64))
    if len(proj) < 2:
        return
    decay = float(np.clip(1.0 / (1.0 + z_depth / (fade_dist * 0.45)), 0.25, 1.0))
    ctx.set_source_rgba(*rgba)
    ctx.set_line_width(max(0.5, base_w * decay))
    ctx.move_to(proj[0][0], proj[0][1])
    for px, py, _ in proj[1:]:
        ctx.line_to(px, py)
    ctx.stroke()


def fog_mix(c3: Iterable[float], dist: float, fog_rgb: np.ndarray, fog_start: float = 6.5, fog_density: float = 0.052) -> tuple[float, float, float]:
    ft = 1.0 - math.exp(-max(0.0, dist - fog_start) * fog_density)
    c = np.asarray(c3, dtype=np.float64) * (1.0 - ft) + fog_rgb * ft
    return (float(c[0]), float(c[1]), float(c[2]))


def fill_3d_quad(
    ctx: cairo.Context,
    cam: Camera3D,
    pts3: list[np.ndarray],
    rgba: tuple[float, float, float, float],
    stroke_rgba: tuple[float, float, float, float] | None = None,
    stroke_w: float = 1.0,
) -> None:
    proj = [cam.project(p) for p in pts3]
    if any(z < Z_NEAR for _, _, z in proj):
        return
    ctx.move_to(float(proj[0][0]), float(proj[0][1]))
    for pu, pv, _ in proj[1:]:
        ctx.line_to(float(pu), float(pv))
    ctx.close_path()
    ctx.set_source_rgba(*rgba)
    if stroke_rgba is not None:
        ctx.fill_preserve()
        ctx.set_source_rgba(*stroke_rgba)
        ctx.set_line_width(stroke_w)
        ctx.stroke()
    else:
        ctx.fill()


def draw_utility_pole(
    ctx: cairo.Context,
    cam: Camera3D,
    base_x: float,
    base_y: float,
    height_m: float,
    radius_m: float,
    fog_rgb: np.ndarray,
    rng: np.random.Generator,
    with_transformer: bool = True,
) -> list[np.ndarray]:
    """Japanese concrete utility pole with visible steel crossarms, ceramic insulators & pole transformer."""
    p_bot = np.array([base_x, base_y, 0.0])
    dist = float(np.linalg.norm(p_bot - cam.eye))
    col_dark = fog_mix([0.022, 0.024, 0.028], dist, fog_rgb)
    col_rim = fog_mix([0.16, 0.17, 0.185], dist, fog_rgb)

    r_top = radius_m * 0.72
    pts_pole = [
        np.array([base_x - radius_m, base_y, 0.0]),
        np.array([base_x + radius_m, base_y, 0.0]),
        np.array([base_x + r_top, base_y, height_m]),
        np.array([base_x - r_top, base_y, height_m]),
    ]
    fill_3d_quad(ctx, cam, pts_pole, (*col_dark, 1.0))

    rim_x = base_x - radius_m if base_x > 0 else base_x + radius_m
    stroke_straight_3d(ctx, cam, [np.array([rim_x, base_y, 0.0]), np.array([rim_x * 0.99, base_y, height_m])], 1.8, dist, (*col_rim, 0.55))

    anchors: list[np.ndarray] = []
    # Crossarms placed clearly inside the open misty silver sky (height_m - 0.45 and - 1.15)
    for arm_z, arm_w in ((height_m - 0.35, 0.88), (height_m - 1.05, 0.74)):
        a0 = np.array([base_x - arm_w, base_y, arm_z])
        a1 = np.array([base_x + arm_w, base_y, arm_z])
        stroke_straight_3d(ctx, cam, [a0, a1], 5.2, dist, (*col_dark, 0.98))
        for sgn in (-1.0, 1.0):
            stroke_straight_3d(
                ctx, cam,
                [np.array([base_x, base_y, arm_z - 0.38]), np.array([base_x + sgn * arm_w * 0.64, base_y, arm_z])],
                2.0, dist, (*col_dark, 0.94),
            )
        for frac in (-0.84, -0.36, 0.36, 0.84):
            pt_ins = np.array([base_x + frac * arm_w, base_y, arm_z + 0.08])
            anchors.append(pt_ins)
            ui, vi, zi = cam.project(pt_ins)
            if zi > Z_NEAR:
                ri = max(1.6, 30.0 / float(zi))
                ctx.set_source_rgba(*col_dark, 0.98)
                ctx.arc(float(ui), float(vi), ri, 0, math.tau)
                ctx.fill()

    if with_transformer:
        tz0, tz1 = height_m - 2.15, height_m - 1.38
        tx_c = base_x - 0.34 if base_x > 0 else base_x + 0.34
        t_quad = [
            np.array([tx_c - 0.20, base_y, tz0]),
            np.array([tx_c + 0.20, base_y, tz0]),
            np.array([tx_c + 0.20, base_y, tz1]),
            np.array([tx_c - 0.20, base_y, tz1]),
        ]
        fill_3d_quad(ctx, cam, t_quad, (*col_dark, 0.98), (*col_rim, 0.36), 1.1)
    return anchors


def draw_catenary_wires(
    ctx: cairo.Context,
    cam: Camera3D,
    wire_pairs: list[tuple[np.ndarray, np.ndarray, float, float]],
    fog_rgb: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Slender dark overhead Japanese power lines with natural gravity catenary sag & near-plane clipping."""
    for p0, p1, sag, width in wire_pairs:
        pts3 = []
        n_seg = 22
        for i in range(n_seg + 1):
            t = i / n_seg
            p = p0 * (1.0 - t) + p1 * t
            p[2] -= sag * 4.0 * t * (1.0 - t)
            pts3.append(p)
        proj = cam.project_polyline(np.array(pts3, dtype=np.float64))
        if len(proj) < 2:
            continue
        mid_z = float(np.mean([p[2] for p in proj]))
        col = fog_mix([0.014, 0.016, 0.020], mid_z, fog_rgb)
        stroke_catmull_rom(
            ctx,
            [(p[0], p[1]) for p in proj],
            base_w=width,
            z_depth=mid_z,
            rgba=(*col, 0.94),
            rng=rng,
            taper=False,
            tremor=0.08,
        )


def draw_concrete_steps(
    ctx: cairo.Context,
    cam: Camera3D,
    fog_rgb: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Three weathered poured-concrete entry steps projecting from the left facade beneath the glowing transom."""
    y0, y1 = 11.82, 14.74
    n_steps = 3
    step_h = 0.155
    step_d = 0.24
    for s in range(n_steps):
        z_bot = s * step_h
        z_top = (s + 1) * step_h
        x_front = X_WALL_L + (n_steps - s) * step_d
        x_back = X_WALL_L + (n_steps - s - 1) * step_d

        mid_p = np.array([x_front, 0.5 * (y0 + y1), z_top])
        dist = float(np.linalg.norm(mid_p - cam.eye))
        lan_l = float(eval_lantern_and_transom_light(
            np.array([x_front]), np.array([0.5 * (y0 + y1)]), np.array([z_top]), 0.0, 0.0, 1.0
        )[0])

        tread_base = np.array([0.082, 0.084, 0.090]) + ACCENT_YELLOW * (0.28 * lan_l)
        tread_c = fog_mix(tread_base, dist, fog_rgb)
        fill_3d_quad(
            ctx, cam,
            [np.array([x_back, y0, z_top]), np.array([x_front, y0, z_top]),
             np.array([x_front, y1, z_top]), np.array([x_back, y1, z_top])],
            (*tread_c, 1.0),
        )

        riser_base = np.array([0.034, 0.036, 0.040]) + ACCENT_YELLOW * (0.10 * lan_l)
        riser_c = fog_mix(riser_base, dist, fog_rgb)
        fill_3d_quad(
            ctx, cam,
            [np.array([x_front, y0, z_bot]), np.array([x_front, y1, z_bot]),
             np.array([x_front, y1, z_top]), np.array([x_front, y0, z_top])],
            (*riser_c, 1.0),
        )

        cheek_base = np.array([0.046, 0.048, 0.052]) + ACCENT_YELLOW * (0.14 * lan_l)
        cheek_c = fog_mix(cheek_base, dist, fog_rgb)
        fill_3d_quad(
            ctx, cam,
            [np.array([x_back, y0, z_bot]), np.array([x_front, y0, z_bot]),
             np.array([x_front, y0, z_top]), np.array([x_back, y0, z_top])],
            (*cheek_c, 1.0),
        )

        # Crisp wet acid-yellow highlight along front step nosing + hairline recessed shadow under nosing
        hi_c = fog_mix(np.array([0.24, 0.25, 0.26]) + ACCENT_YELLOW * (0.42 * lan_l), dist, fog_rgb)
        stroke_straight_3d(ctx, cam, [np.array([x_front, y0, z_top]), np.array([x_front, y1, z_top])], 2.0, dist, (*hi_c, 0.85))
        sh_c = fog_mix([0.015, 0.016, 0.018], dist, fog_rgb)
        stroke_straight_3d(ctx, cam, [np.array([x_front, y0, z_top - 0.02]), np.array([x_front, y1, z_top - 0.02])], 1.4, dist, (*sh_c, 0.72))


def draw_shop_eave_and_downpipe(
    ctx: cairo.Context,
    cam: Camera3D,
    fog_rgb: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Projecting corrugated/timber shop eave above the doorway + crisp straight steel downpipe with wall straps."""
    y0, y1 = 9.7, 16.3
    x0, x1 = X_WALL_L, X_WALL_L + 1.04
    z_wall, z_tip = 3.16, 2.86
    dist = float(np.linalg.norm(np.array([x1, 12.5, z_tip]) - cam.eye))

    soffit_c = fog_mix(np.array([0.042, 0.043, 0.045]) + ACCENT_YELLOW * 0.16, dist, fog_rgb)
    fill_3d_quad(
        ctx, cam,
        [np.array([x0, y0, z_wall]), np.array([x1, y0, z_tip]),
         np.array([x1, y1, z_tip]), np.array([x0, y1, z_wall])],
        (*soffit_c, 1.0),
    )

    rafter_c = fog_mix([0.020, 0.021, 0.024], dist, fog_rgb)
    for yr in np.linspace(y0 + 0.35, y1 - 0.35, 11):
        stroke_straight_3d(ctx, cam, [np.array([x0, yr, z_wall - 0.03]), np.array([x1, yr, z_tip - 0.02])], 2.2, dist, (*rafter_c, 0.86))

    fascia_c = fog_mix([0.018, 0.020, 0.024], dist, fog_rgb)
    fill_3d_quad(
        ctx, cam,
        [np.array([x1, y0, z_tip - 0.09]), np.array([x1, y1, z_tip - 0.09]),
         np.array([x1, y1, z_tip + 0.06]), np.array([x1, y0, z_tip + 0.06])],
        (*fascia_c, 1.0),
    )
    fill_3d_quad(
        ctx, cam,
        [np.array([x0, y0, z_wall - 0.07]), np.array([x1, y0, z_tip - 0.09]),
         np.array([x1, y0, z_tip + 0.06]), np.array([x0, y0, z_wall + 0.08])],
        (*fascia_c, 1.0),
    )
    rim_c = fog_mix([0.26, 0.27, 0.29], dist, fog_rgb)
    stroke_straight_3d(ctx, cam, [np.array([x1, y0, z_tip + 0.06]), np.array([x1, y1, z_tip + 0.06])], 1.8, dist, (*rim_c, 0.68))

    # Crisp straight architectural steel downpipe (gutter -> angled gooseneck elbow -> vertical wall riser -> drain)
    yp = y0 + 0.08
    pipe_pts = [
        np.array([x1 - 0.04, yp, z_tip - 0.07]),
        np.array([x1 - 0.04, yp, z_tip - 0.17]),
        np.array([x0 + 0.11, yp, z_tip - 0.56]),
        np.array([x0 + 0.09, yp, 0.02]),
    ]
    p_dist = float(np.linalg.norm(pipe_pts[2] - cam.eye))
    p_col = fog_mix([0.022, 0.024, 0.028], p_dist, fog_rgb)
    p_hi = fog_mix(np.array([0.18, 0.19, 0.20]) + ACCENT_YELLOW * 0.24, p_dist, fog_rgb)
    stroke_straight_3d(ctx, cam, pipe_pts, 6.8, p_dist, (*p_col, 0.98))
    stroke_straight_3d(ctx, cam, [p + np.array([0.022, 0.0, 0.0]) for p in pipe_pts], 1.5, p_dist, (*p_hi, 0.58))

    # Wall mounting pipe-strap brackets along the vertical downpipe
    for zb in (0.65, 1.45, 2.20):
        stroke_straight_3d(ctx, cam, [np.array([x0, yp, zb]), np.array([x0 + 0.15, yp, zb])], 3.2, p_dist, (*p_col, 0.96))


def draw_hanging_paper_lantern(
    ctx: cairo.Context,
    cam: Camera3D,
    fog_rgb: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """
    Hero single acid-yellow (#d9d84a) Japanese chochin paper lantern suspended cleanly
    by a forged-iron hanger rod directly beneath the projecting timber shop eave!
    """
    lx, ly, lz = LANTERN_POS
    dist = float(np.linalg.norm(LANTERN_POS - cam.eye))

    # Clean vertical forged-iron suspension rod directly down from the projecting eave soffit!
    eave_z_at_lx = 3.16 - 0.30 * ((lx - X_WALL_L) / 1.04)
    bracket_c = fog_mix([0.016, 0.017, 0.020], dist, fog_rgb)
    stroke_straight_3d(
        ctx, cam,
        [np.array([lx, ly, eave_z_at_lx]), np.array([lx, ly, lz + 0.24])],
        3.2, dist, (*bracket_c, 0.98),
    )
    # Small horizontal forged-iron roof cleat right under the eave soffit
    stroke_straight_3d(
        ctx, cam,
        [np.array([lx - 0.14, ly, eave_z_at_lx - 0.02]), np.array([lx + 0.14, ly, eave_z_at_lx - 0.02])],
        4.0, dist, (*bracket_c, 0.96),
    )

    uc, vc, zc = cam.project(LANTERN_POS)
    if zc < Z_NEAR:
        return
    _, v_top, _ = cam.project(np.array([lx, ly, lz + 0.245]))
    _, v_bot, _ = cam.project(np.array([lx, ly, lz - 0.245]))
    u_rad, _, _ = cam.project(np.array([lx + 0.155, ly, lz]))

    cx, cy = float(uc), float(vc)
    rx = max(14.0, abs(float(u_rad) - cx))
    ry = max(24.0, abs(float(v_bot) - float(v_top)) * 0.5)

    # Soft luminous atmospheric halo around the lantern
    g_halo = cairo.RadialGradient(cx, cy, rx * 0.2, cx, cy, ry * 3.1)
    g_halo.add_color_stop_rgba(0.0, float(ACCENT_CORE[0]), float(ACCENT_CORE[1]), float(ACCENT_CORE[2]), 0.44)
    g_halo.add_color_stop_rgba(0.32, float(ACCENT_YELLOW[0]), float(ACCENT_YELLOW[1]), float(ACCENT_YELLOW[2]), 0.22)
    g_halo.add_color_stop_rgba(0.70, float(ACCENT_WARM[0]), float(ACCENT_WARM[1]), float(ACCENT_WARM[2]), 0.06)
    g_halo.add_color_stop_rgba(1.0, float(ACCENT_WARM[0]), float(ACCENT_WARM[1]), float(ACCENT_WARM[2]), 0.0)
    ctx.set_source(g_halo)
    ctx.arc(cx, cy, ry * 3.1, 0, math.tau)
    ctx.fill()

    # Main glowing oval chochin washi paper body
    ctx.save()
    ctx.translate(cx, cy)
    ctx.scale(rx, ry)
    ctx.move_to(0.0, -0.96)
    ctx.curve_to(0.76, -0.94, 1.06, -0.45, 1.04, 0.0)
    ctx.curve_to(1.06, 0.48, 0.76, 0.94, 0.0, 0.96)
    ctx.curve_to(-0.76, 0.94, -1.06, 0.48, -1.04, 0.0)
    ctx.curve_to(-1.06, -0.45, -0.76, -0.94, 0.0, -0.96)
    ctx.close_path()

    g_body = cairo.RadialGradient(0.06, -0.05, 0.05, 0.0, 0.0, 1.15)
    g_body.add_color_stop_rgba(0.0, 0.99, 0.985, 0.80, 1.0)
    g_body.add_color_stop_rgba(0.42, float(ACCENT_CORE[0]), float(ACCENT_CORE[1]), float(ACCENT_CORE[2]), 1.0)
    g_body.add_color_stop_rgba(0.78, float(ACCENT_YELLOW[0]), float(ACCENT_YELLOW[1]), float(ACCENT_YELLOW[2]), 1.0)
    g_body.add_color_stop_rgba(1.0, 0.56, 0.54, 0.16, 1.0)
    ctx.set_source(g_body)
    ctx.fill_preserve()
    ctx.clip()

    # Curved horizontal bamboo ribs
    ctx.set_source_rgba(0.28, 0.27, 0.09, 0.46)
    ctx.set_line_width(0.032)
    for k in range(-4, 5):
        yk = k * 0.19
        ctx.move_to(-1.12, yk - 0.03)
        ctx.curve_to(-0.45, yk + 0.06, 0.45, yk + 0.06, 1.12, yk - 0.03)
        ctx.stroke()

    # Minimal ink calligraphy mark
    ctx.set_source_rgba(0.06, 0.06, 0.04, 0.72)
    ctx.set_line_width(0.072)
    ctx.move_to(-0.08, -0.44)
    ctx.curve_to(-0.02, -0.15, 0.02, 0.14, -0.04, 0.46)
    ctx.stroke()
    ctx.set_line_width(0.052)
    ctx.move_to(-0.34, -0.24)
    ctx.curve_to(-0.04, -0.30, 0.20, -0.26, 0.34, -0.20)
    ctx.stroke()
    ctx.move_to(-0.26, 0.12)
    ctx.curve_to(0.0, 0.08, 0.18, 0.10, 0.28, 0.16)
    ctx.stroke()
    ctx.restore()

    # Lacquered top/bottom metal collars & delicate hanging silk tassel
    ctx.save()
    ctx.translate(cx, cy)
    ctx.set_source_rgba(*bracket_c, 0.98)
    ctx.rectangle(-rx * 0.54, -ry * 1.05, rx * 1.08, ry * 0.16)
    ctx.fill()
    ctx.rectangle(-rx * 0.48, ry * 0.89, rx * 0.96, ry * 0.15)
    ctx.fill()
    ctx.set_line_width(max(1.5, rx * 0.08))
    ctx.move_to(0.0, ry * 1.04)
    ctx.curve_to(rx * 0.05, ry * 1.24, -rx * 0.04, ry * 1.40, rx * 0.03, ry * 1.56)
    ctx.stroke()
    ctx.restore()


def draw_parked_bicycle_3d(
    ctx: cairo.Context,
    cam: Camera3D,
    fog_rgb: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """
    Parked Japanese city silhouette bicycle leaning against the left concrete wall,
    with soft feathered tire contact shadows, spoked wheels, chainring, mudguards,
    front basket, handlebars, and crisp silver/acid-yellow specular catchlights.
    """
    y_rear = 7.15
    y_front = 8.24
    x_base = X_WALL_L + 0.35
    lean = -0.14
    r_wheel = 0.335

    def bike_pt(y_m: float, z_m: float, x_off: float = 0.0) -> np.ndarray:
        return np.array([x_base + lean * z_m + x_off, y_m, z_m], dtype=np.float64)

    rear_hub = bike_pt(y_rear, r_wheel)
    front_hub = bike_pt(y_front, r_wheel)
    bb = bike_pt(y_rear + 0.44, 0.28)
    seat_top = bike_pt(y_rear + 0.32, 0.83)
    head_bot = bike_pt(y_front - 0.10, 0.72)
    head_top = bike_pt(y_front - 0.14, 0.92)
    stem_top = bike_pt(y_front - 0.16, 1.04)
    saddle_l = bike_pt(y_rear + 0.19, 0.92)
    saddle_r = bike_pt(y_rear + 0.45, 0.91)

    dist = float(np.linalg.norm(bb - cam.eye))
    ink = fog_mix([0.014, 0.015, 0.018], dist, fog_rgb)
    rim_y = fog_mix(np.array([0.24, 0.25, 0.26]) + ACCENT_YELLOW * 0.50, dist, fog_rgb)

    # Soft feathered elliptical contact shadows right beneath both tires (NO hard black rectangle!)
    for wy_sh in (y_rear, y_front, 0.5 * (y_rear + y_front)):
        us, vs, zs = cam.project(np.array([x_base, wy_sh, 0.005]))
        if zs > Z_NEAR:
            rad_x = 58.0 if wy_sh != 0.5 * (y_rear + y_front) else 88.0
            rad_y = rad_x * 0.26
            ctx.save()
            ctx.translate(float(us), float(vs))
            ctx.scale(1.0, rad_y / rad_x)
            gs = cairo.RadialGradient(0.0, 0.0, 2.0, 0.0, 0.0, rad_x)
            gs.add_color_stop_rgba(0.0, 0.006, 0.007, 0.009, 0.58)
            gs.add_color_stop_rgba(0.55, 0.006, 0.007, 0.009, 0.25)
            gs.add_color_stop_rgba(1.0, 0.006, 0.007, 0.009, 0.0)
            ctx.set_source(gs)
            ctx.arc(0.0, 0.0, rad_x, 0, math.tau)
            ctx.fill()
            ctx.restore()

    def draw_wheel(hub_y: float) -> None:
        hub = bike_pt(hub_y, r_wheel)
        rim_pts = [
            bike_pt(hub_y + math.cos(math.tau * k / 32.0) * r_wheel, r_wheel + math.sin(math.tau * k / 32.0) * r_wheel)
            for k in range(33)
        ]
        # 14 metallic spokes
        for k in range(14):
            ang = math.tau * k / 14.0 + 0.12
            p_edge = bike_pt(hub_y + math.cos(ang) * (r_wheel * 0.94), r_wheel + math.sin(ang) * (r_wheel * 0.94))
            stroke_straight_3d(ctx, cam, [hub, p_edge], 1.25, dist, (*ink, 0.65))
        # Hub center cap
        uh, vh, zh = cam.project(hub)
        if zh > Z_NEAR:
            ctx.set_source_rgba(*ink, 0.98)
            ctx.arc(float(uh), float(vh), 3.6, 0, math.tau)
            ctx.fill()
        # Bold rubber tire
        r_proj = cam.project_polyline(np.array(rim_pts))
        if len(r_proj) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in r_proj], 8.2, dist, (*ink, 0.99), rng, taper=False, tremor=0.06)
            hi_pts = [
                bike_pt(hub_y + math.cos(ang) * (r_wheel * 0.89), r_wheel + math.sin(ang) * (r_wheel * 0.89))
                for ang in np.linspace(-0.75, 1.55, 14)
            ]
            hi_proj = cam.project_polyline(np.array(hi_pts))
            if len(hi_proj) >= 2:
                stroke_catmull_rom(ctx, [(p[0], p[1]) for p in hi_proj], 1.8, dist, (*rim_y, 0.72), rng, taper=True, tremor=0.05)

        # Curved city mudguard fender + specular catchlight
        fend_pts = [
            bike_pt(hub_y + math.cos(ang) * (r_wheel + 0.038), r_wheel + math.sin(ang) * (r_wheel + 0.038))
            for ang in np.linspace(0.32, 2.80, 16)
        ]
        f_proj = cam.project_polyline(np.array(fend_pts))
        if len(f_proj) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in f_proj], 5.2, dist, (*ink, 0.98), rng, taper=False, tremor=0.06)
            stroke_catmull_rom(ctx, [(p[0] + 1.0, p[1] - 1.1) for p in f_proj], 1.7, dist, (*rim_y, 0.62), rng, taper=True, tremor=0.04)

    draw_wheel(y_rear)
    draw_wheel(y_front)

    # Chainring disc & crank arm at bottom bracket
    cr_pts = [
        bike_pt(y_rear + 0.44 + math.cos(math.tau * k / 18.0) * 0.088, 0.28 + math.sin(math.tau * k / 18.0) * 0.088, x_off=0.03)
        for k in range(19)
    ]
    cr_proj = cam.project_polyline(np.array(cr_pts))
    if len(cr_proj) >= 2:
        ctx.set_source_rgba(*ink, 0.96)
        ctx.move_to(cr_proj[0][0], cr_proj[0][1])
        for px, py, _ in cr_proj[1:]:
            ctx.line_to(px, py)
        ctx.close_path()
        ctx.fill()
    stroke_straight_3d(ctx, cam, [bb, bike_pt(y_rear + 0.56, 0.16, x_off=0.07)], 4.0, dist, (*ink, 0.98))

    # Bold Japanese city bike steel frame tubes + warm lantern rim catchlights
    tubes = [
        (rear_hub, bb, 4.6),
        (rear_hub, seat_top, 4.4),
        (bb, seat_top, 6.2),
        (seat_top, bike_pt(y_rear + 0.31, 0.92), 4.4),
        (bb, head_bot, 6.5),
        (bike_pt(y_rear + 0.36, 0.62), head_top, 5.4),
        (head_bot, head_top, 6.8),
        (head_top, stem_top, 5.0),
        (head_bot, front_hub, 5.6),
        (bike_pt(y_rear - 0.25, 0.74), bike_pt(y_rear + 0.29, 0.75), 3.8),
        (rear_hub, bike_pt(y_rear - 0.18, 0.74), 3.0),
        (rear_hub, bike_pt(y_rear - 0.12, 0.01, x_off=0.14), 3.4),
    ]
    for pa, pb, w_t in tubes:
        stroke_straight_3d(ctx, cam, [pa, pb], w_t, dist, (*ink, 0.99))
        stroke_straight_3d(ctx, cam, [pa + np.array([0.01, 0.0, 0.01]), pb + np.array([0.01, 0.0, 0.01])], max(1.2, w_t * 0.28), dist, (*rim_y, 0.58))

    # Swept-back city handlebars + brake grips
    for x_side in (-0.23, 0.23):
        hb = np.array([
            stem_top,
            bike_pt(y_front - 0.16, 1.06, x_off=x_side * 0.55),
            bike_pt(y_front - 0.29, 1.04, x_off=x_side),
        ])
        hseg = cam.project_polyline(hb)
        if len(hseg) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in hseg], 4.8, dist, (*ink, 0.99), rng, taper=False, tremor=0.06)
            stroke_catmull_rom(ctx, [(p[0], p[1] - 1.0) for p in hseg], 1.5, dist, (*rim_y, 0.62), rng, taper=True, tremor=0.04)

    # Contoured saddle + warm rim glint
    s_seg = cam.project_polyline(np.array([saddle_l, bike_pt(y_rear + 0.32, 0.95), saddle_r]))
    if len(s_seg) >= 2:
        stroke_catmull_rom(ctx, [(p[0], p[1]) for p in s_seg], 11.5, dist, (*ink, 0.99), rng, taper=True, tremor=0.06)
        stroke_catmull_rom(ctx, [(p[0], p[1] - 2.0) for p in s_seg], 2.0, dist, (*rim_y, 0.65), rng, taper=True, tremor=0.04)

    # Wire front bicycle basket (subtle 3D wire mesh)
    b_bot = [
        bike_pt(y_front - 0.04, 0.77, -0.14),
        bike_pt(y_front + 0.22, 0.77, -0.14),
        bike_pt(y_front + 0.22, 0.77, 0.14),
        bike_pt(y_front - 0.04, 0.77, 0.14),
    ]
    b_top = [
        bike_pt(y_front - 0.06, 0.98, -0.17),
        bike_pt(y_front + 0.26, 0.98, -0.17),
        bike_pt(y_front + 0.26, 0.98, 0.17),
        bike_pt(y_front - 0.06, 0.98, 0.17),
    ]
    for i in range(4):
        j = (i + 1) % 4
        stroke_straight_3d(ctx, cam, [b_bot[i], b_bot[j]], 2.0, dist, (*ink, 0.92))
        stroke_straight_3d(ctx, cam, [b_top[i], b_top[j]], 2.2, dist, (*ink, 0.94))
        stroke_straight_3d(ctx, cam, [b_bot[i], b_top[i]], 1.8, dist, (*ink, 0.90))
        # Subtle top rim catchlight
        stroke_straight_3d(ctx, cam, [b_top[i], b_top[j]], 0.9, dist, (*rim_y, 0.38))


def draw_utility_meter_box(ctx: cairo.Context, cam: Camera3D, fog_rgb: np.ndarray) -> None:
    """Authentic Japanese electrical meter box + vertical steel conduit pipe on the left concrete facade."""
    m_dist = float(np.linalg.norm(np.array([X_WALL_L + 0.12, 6.2, 1.62]) - cam.eye))
    box_front = fog_mix([0.032, 0.034, 0.038], m_dist, fog_rgb)
    box_side = fog_mix([0.052, 0.055, 0.060], m_dist, fog_rgb)
    box_hi = fog_mix([0.15, 0.16, 0.17], m_dist, fog_rgb)
    pipe_c = fog_mix([0.026, 0.028, 0.032], m_dist, fog_rgb)

    # Vertical steel electrical conduit rising cleanly from box up the board-formed wall
    stroke_straight_3d(ctx, cam, [np.array([X_WALL_L + 0.06, 6.18, 1.88]), np.array([X_WALL_L + 0.06, 6.18, 4.90])], 4.6, m_dist, (*pipe_c, 0.95))
    stroke_straight_3d(ctx, cam, [np.array([X_WALL_L + 0.08, 6.18, 1.88]), np.array([X_WALL_L + 0.08, 6.18, 4.90])], 1.2, m_dist, (*box_hi, 0.45))

    # Box faces (front +Y-facing cheek + side +X face + top overhang lid)
    fill_3d_quad(
        ctx, cam,
        [np.array([X_WALL_L + 0.17, 5.98, 1.38]), np.array([X_WALL_L + 0.17, 6.40, 1.38]),
         np.array([X_WALL_L + 0.17, 6.40, 1.86]), np.array([X_WALL_L + 0.17, 5.98, 1.86])],
        (*box_front, 0.98), (*box_hi, 0.42), 1.1,
    )
    fill_3d_quad(
        ctx, cam,
        [np.array([X_WALL_L, 5.98, 1.38]), np.array([X_WALL_L + 0.17, 5.98, 1.38]),
         np.array([X_WALL_L + 0.17, 5.98, 1.86]), np.array([X_WALL_L, 5.98, 1.86])],
        (*box_side, 0.98),
    )
    # Recessed circular glass watt-meter dial ring
    um, vm, zm = cam.project(np.array([X_WALL_L + 0.175, 6.19, 1.66]))
    if zm > Z_NEAR:
        ctx.save()
        ctx.translate(float(um), float(vm))
        ctx.scale(0.55, 1.0)
        ctx.set_source_rgba(*box_hi, 0.42)
        ctx.set_line_width(1.6)
        ctx.arc(0.0, 0.0, 15.0, 0, math.tau)
        ctx.stroke()
        ctx.restore()


def draw_rain_streaks_and_ripples(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
) -> None:
    """Delicate diagonal night rain streaks (brighter near lantern halo) & concentric puddle rings."""
    ul, vl, _ = cam.project(LANTERN_POS)
    for _ in range(360):
        x = float(rng.uniform(0, W))
        y = float(rng.uniform(0, H))
        ln = float(rng.uniform(22.0, 68.0))
        dx = -ln * 0.105
        dy = ln
        d_lan = math.hypot((x - float(ul)) / 320.0, (y - float(vl)) / 280.0)
        warm_t = math.exp(-d_lan * d_lan * 1.4)
        alpha = float(rng.uniform(0.035, 0.115)) + warm_t * 0.14
        r = 0.72 * (1.0 - warm_t) + float(ACCENT_CORE[0]) * warm_t
        g = 0.76 * (1.0 - warm_t) + float(ACCENT_CORE[1]) * warm_t
        b = 0.82 * (1.0 - warm_t) + float(ACCENT_CORE[2]) * warm_t
        ctx.set_source_rgba(r, g, b, min(0.30, alpha))
        ctx.set_line_width(float(rng.uniform(0.55, 1.10)))
        ctx.move_to(x, y)
        ctx.line_to(x + dx, y + dy)
        ctx.stroke()

    for _ in range(28):
        wy = float(rng.uniform(3.8, 16.0))
        wx = float(rng.uniform(-1.55, 1.95))
        u, v, z = cam.project(np.array([wx, wy, 0.01]))
        if z < Z_NEAR or v < cam.horizon_y + 20 or v > H - 15 or u < 20 or u > W - 20:
            continue
        rad_m = float(rng.uniform(0.05, 0.14))
        u2, _, _ = cam.project(np.array([wx + rad_m, wy, 0.01]))
        rx = max(4.0, abs(float(u2) - float(u)))
        ry = rx * float(np.clip((v - cam.horizon_y) / (H - cam.horizon_y) * 0.34, 0.08, 0.34))
        lan_w = math.exp(-((wx - LANTERN_POS[0]) ** 2 + (wy - LANTERN_POS[1]) ** 2) / 14.0)
        col = np.array([0.48, 0.52, 0.58]) * (1.0 - lan_w) + ACCENT_CORE * lan_w
        ctx.save()
        ctx.translate(float(u), float(v))
        ctx.scale(1.0, ry / max(1.0, rx))
        ctx.set_source_rgba(float(col[0]), float(col[1]), float(col[2]), 0.16 + 0.18 * lan_w)
        ctx.set_line_width(1.1)
        ctx.arc(0.0, 0.0, rx, 0, math.tau)
        ctx.stroke()
        ctx.restore()


# =============================================================================
# Stage 4 — Foreground Circular Disc Bokeh Eave/Bough & 35mm Analog Optics
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


def convolve_fft_2d_bbox(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    alpha = img[:, :, 3]
    ys, xs = np.where(alpha > 1e-5)
    if len(ys) == 0:
        return np.zeros_like(img, dtype=np.float32)
    kh, kw = kernel.shape
    pad = max(kh, kw)
    ih, iw = img.shape[:2]
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(ih, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(iw, int(xs.max()) + pad + 1)
    sub = img[y0:y1, x0:x1]
    sh, sw = sub.shape[:2]
    fh = 1 << int(math.ceil(math.log2(sh + kh - 1)))
    fw = 1 << int(math.ceil(math.log2(sw + kw - 1)))
    k_pad = np.zeros((fh, fw), dtype=np.float32)
    k_pad[:kh, :kw] = kernel
    k_pad = np.roll(np.roll(k_pad, -(kh // 2), axis=0), -(kw // 2), axis=1)
    K = np.fft.rfft2(k_pad)
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        p = np.zeros((fh, fw), dtype=np.float32)
        p[:sh, :sw] = sub[:, :, c]
        out[y0:y1, x0:x1, c] = np.fft.irfft2(np.fft.rfft2(p) * K, s=(fh, fw))[:sh, :sw]
    return out


def paint_foreground_bokeh_eave_bough(rng: np.random.Generator) -> np.ndarray:
    """
    Close out-of-focus overhanging dark tiled eave + rain-beaded maple/bamboo bough
    framing the upper-right corner, convolved in premultiplied RGBA with circular
    aperture disc kernel (r = 7.8) + ~1 px opposite R/B lateral shift.
    """
    surf, ctx = new_cairo_2x()

    eave_poly = [
        (W + 30, -30),
        (W - 560, -30),
        (W - 505, 56),
        (W - 395, 114),
        (W - 220, 174),
        (W + 30, 232),
    ]
    ctx.set_source_rgba(0.011, 0.012, 0.015, 0.98)
    ctx.move_to(*eave_poly[0])
    for p in eave_poly[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    ctx.fill()

    for rx_off, ry_off in ((W - 430, 94), (W - 315, 138), (W - 185, 184), (W - 60, 220)):
        ctx.save()
        ctx.translate(rx_off, ry_off)
        ctx.rotate(0.32)
        ctx.set_source_rgba(0.010, 0.011, 0.013, 0.98)
        ctx.rectangle(-16, -8, 42, 24)
        ctx.fill()
        ctx.restore()

    ctx.set_source_rgba(0.28, 0.30, 0.34, 0.50)
    ctx.set_line_width(3.2)
    ctx.move_to(W - 505, 56)
    ctx.line_to(W - 395, 114)
    ctx.line_to(W - 220, 174)
    ctx.line_to(W + 10, 228)
    ctx.stroke()

    boughs = [
        [(W - 30, 192), (W - 210, 224), (W - 385, 294), (W - 525, 355)],
        [(W - 115, 142), (W - 295, 152), (W - 465, 202), (W - 610, 228)],
        [(W + 10, 272), (W - 140, 328), (W - 275, 412)],
    ]
    for bpts in boughs:
        stroke_catmull_rom(ctx, bpts, base_w=6.5, z_depth=0.9, rgba=(0.011, 0.012, 0.015, 0.97), rng=rng, taper=True, tremor=0.5)
        for k in range(len(bpts) - 1):
            p0, p1 = bpts[k], bpts[k + 1]
            for _ in range(9):
                t = float(rng.uniform(0.12, 0.96))
                bx = p0[0] * (1.0 - t) + p1[0] * t
                by = p0[1] * (1.0 - t) + p1[1] * t
                ang = float(rng.uniform(1.65, 2.85))
                ln = float(rng.uniform(44.0, 96.0))
                wid = float(rng.uniform(8.5, 17.0))
                droop = float(rng.uniform(8.0, 22.0))
                nick = float(rng.uniform(-3.0, 3.0))

                ctx.save()
                ctx.translate(bx, by)
                ctx.rotate(ang)
                ctx.set_source_rgba(0.012, 0.013, 0.016, 0.96)
                ctx.move_to(0.0, 0.0)
                ctx.curve_to(ln * 0.34, -wid + nick, ln * 0.72, -wid * 0.48 + droop * 0.4, ln, droop)
                ctx.curve_to(ln * 0.68, wid * 0.42 + droop * 0.5, ln * 0.32, wid + nick, 0.0, 0.0)
                ctx.close_path()
                ctx.fill()
                ctx.restore()

    fg_1x = finish_cairo_2x(surf)
    blurred = convolve_fft_2d_bbox(fg_1x, make_disc_kernel(radius=7.8))
    out = blurred.copy()
    out[:, 1:, 0] = blurred[:, :-1, 0]
    out[:, :-1, 2] = blurred[:, 1:, 2]
    return np.clip(out, 0.0, 1.0)


def apply_35mm_finish(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = rgb.astype(np.float32).copy()

    # 1. Multi-scale highlight halation (sigma = 6.5, 24, 72 px) above lum > 0.72
    lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    hl_mask = np.clip((lum - 0.72) / 0.28, 0.0, 1.6)
    hl_src = img * hl_mask[..., None]
    tint = np.array([1.05, 1.04, 0.80], dtype=np.float32)
    hal = (
        0.26 * gaussian_blur_2d(hl_src, 6.5)
        + 0.20 * gaussian_blur_2d(hl_src, 24.0)
        + 0.14 * gaussian_blur_2d(hl_src, 72.0)
    ) * tint[None, None, :]
    img = 1.0 - (1.0 - np.clip(img, 0.0, 1.0)) * (1.0 - np.clip(hal, 0.0, 0.85))

    # 2. Radial corner chromatic aberration + gentle optical vignette
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (xx - W * 0.5) / (W * 0.5)
    ny = (yy - H * 0.5) / (H * 0.5)
    r2 = nx * nx + ny * ny
    ca_scale = 0.00105
    img[:, :, 0] = remap_bilinear(img[:, :, 0], xx + nx * r2 * (W * ca_scale), yy + ny * r2 * (H * ca_scale))
    img[:, :, 2] = remap_bilinear(img[:, :, 2], xx - nx * r2 * (W * ca_scale), yy - ny * r2 * (H * ca_scale))
    img *= np.clip(1.0 - 0.23 * (r2 ** 1.18), 0.68, 1.0)[..., None]

    # 3. Linear shadow black lift + C1-continuous tanh highlight shoulder (knee=0.80, ceiling=0.965)
    img = 0.011 + img * 0.989
    knee, ceiling = 0.80, 0.965
    span = ceiling - knee
    img = np.where(img > knee, knee + span * np.tanh((img - knee) / span), img)

    # 4. Luminance-weighted silver-halide grain
    lum2 = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    env = np.clip(1.12 - 0.95 * np.power(np.clip(lum2, 0.0, 1.0), 1.55), 0.12, 1.12)
    grain = rng.normal(0.0, 0.0095, size=(H, W, 3)).astype(np.float32)
    mono_g = rng.normal(0.0, 0.0085, size=(H, W, 1)).astype(np.float32)
    img = np.clip(img + (0.72 * mono_g + 0.28 * grain) * env[..., None], 0.0, 1.0)

    # 5. Faint analog transfer patina: horizontal YIQ chroma softening (1.0 px) + reflect101
    #    + slight luma ring + sub-pixel per-line horizontal tape jitter + <=0.8% scanlines
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    I = 0.596 * R - 0.274 * G - 0.322 * B
    Q = 0.211 * R - 0.523 * G + 0.312 * B

    I_soft = gaussian_blur_1d(I, 1.0, axis=1)
    Q_soft = gaussian_blur_1d(Q, 1.0, axis=1)
    Y_ring = Y + 0.038 * (Y - gaussian_blur_1d(Y, 1.1, axis=1))

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
    audit_dir: pathlib.Path | None = None,
    webp_outputs: list[pathlib.Path] | None = None,
) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # Stage 1: Off-axis Camera3D (VP ~60.6% width, horizon ~45.4% height)
    cam = Camera3D(W, H, eye=(0.55, 0.0, 1.56), target=(-0.75, 11.5, 1.24), fov_x_deg=56.0)
    print(f"[mira-01] Camera VP: ({cam.vp_x:.1f}, {cam.vp_y:.1f}) -> {cam.vp_x / W * 100:.1f}% width")

    # Stage 1 & 2: Nocturnal silver-obsidian rain-mist sky
    canvas, fog_rgb = paint_nocturnal_mist_sky(cam, rng)

    # Far cross-street building closing the misty backstreet corner
    far_uv = paint_far_corner_wall_uv(rng)
    canvas = cam.warp_plane_over(
        canvas, far_uv,
        p0=np.array([-3.2, Y_WALL_END, 0.0]),
        e1=np.array([7.2, 0.0, 0.0]),
        e2=np.array([0.0, 0.0, 6.0]),
        fog_rgb=fog_rgb, fog_start=6.5, fog_density=0.052,
    )

    # Wet irregular Voronoi basalt cobblestone street + wetness mask
    cobble_uv, wet_uv = paint_voronoi_cobblestones_uv(rng, (-2.6, 3.3), (1.8, 29.0))
    canvas = cam.warp_plane_over(
        canvas, cobble_uv,
        p0=np.array([-2.6, 1.8, Z_GROUND]),
        e1=np.array([5.9, 0.0, 0.0]),
        e2=np.array([0.0, 27.2, 0.0]),
        fog_rgb=fog_rgb, fog_start=7.0, fog_density=0.052,
    )
    wet_screen = cam.warp_plane_over(
        np.zeros((H, W, 3), dtype=np.float32), wet_uv,
        p0=np.array([-2.6, 1.8, Z_GROUND]),
        e1=np.array([5.9, 0.0, 0.0]),
        e2=np.array([0.0, 27.2, 0.0]),
        fog_rgb=np.zeros(3, dtype=np.float32), fog_start=999.0, fog_density=0.0,
    )

    # Left weathered board-formed concrete facade (with warm acid-yellow shop transom)
    left_uv = paint_left_concrete_facade_uv(rng, (1.8, 28.0), (0.0, 5.4))
    canvas = cam.warp_plane_over(
        canvas, left_uv,
        p0=np.array([X_WALL_L, 1.8, 0.0]),
        e1=np.array([0.0, 26.2, 0.0]),
        e2=np.array([0.0, 0.0, 5.4]),
        fog_rgb=fog_rgb, fog_start=6.5, fog_density=0.052,
    )

    # Right board-formed concrete retaining wall
    right_uv = paint_right_retaining_wall_uv(rng, (2.6, 28.0), (0.0, 3.6))
    canvas = cam.warp_plane_over(
        canvas, right_uv,
        p0=np.array([X_WALL_R, 2.6, 0.0]),
        e1=np.array([0.0, 25.4, 0.0]),
        e2=np.array([0.0, 0.0, 3.6]),
        fog_rgb=fog_rgb, fog_start=6.5, fog_density=0.052,
    )

    # Stage 3 Band 0 (Far Utility Pole & Backstreet Silhouettes in Silver Mist, Y = 18..27 m)
    s_far, c_far = new_cairo_2x()
    far_anchors = draw_utility_pole(c_far, cam, base_x=2.26, base_y=21.0, height_m=5.75, radius_m=0.11, fog_rgb=fog_rgb, rng=rng, with_transformer=False)
    far_eave_c = fog_mix([0.025, 0.027, 0.031], 21.0, fog_rgb)
    fill_3d_quad(
        c_far, cam,
        [np.array([X_WALL_R + 0.1, 18.2, 3.35]), np.array([X_WALL_R - 0.65, 18.2, 3.12]),
         np.array([X_WALL_R - 0.65, 25.5, 3.12]), np.array([X_WALL_R + 0.1, 25.5, 3.35])],
        (*far_eave_c, 0.96),
    )
    canvas = over_premul(canvas, finish_cairo_2x(s_far))

    # Stage 2: Composite Fresnel wet cobblestone reflections (sky + buildings + 3D mirrored lantern & transom)
    canvas = composite_wet_cobblestone_reflections(canvas, cam, wet_screen, rng)

    # Stage 3 Band 1 (Midground Concrete Steps, Shop Eave, Utility Pole, Catenary Power Lines & Hero Lantern)
    s_mid, c_mid = new_cairo_2x()
    draw_concrete_steps(c_mid, cam, fog_rgb, rng)
    draw_shop_eave_and_downpipe(c_mid, cam, fog_rgb, rng)
    mid_anchors = draw_utility_pole(c_mid, cam, base_x=2.34, base_y=12.4, height_m=5.45, radius_m=0.125, fog_rgb=fog_rgb, rng=rng, with_transformer=True)

    # Graceful catenary overhead power lines cutting cleanly across the silver misty night sky!
    wires: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    for k in range(min(len(far_anchors), len(mid_anchors))):
        wires.append((far_anchors[k], mid_anchors[k], 0.36 + 0.05 * (k % 3), 2.1))
    # Cross-alley feeder lines arching boldly across the open misty sky from mid pole to left building roofline!
    wires.append((mid_anchors[0], np.array([X_WALL_L + 0.05, 15.6, 4.95]), 0.44, 2.0))
    wires.append((mid_anchors[1], np.array([X_WALL_L + 0.05, 13.8, 4.82]), 0.52, 1.9))
    wires.append((mid_anchors[4], np.array([X_WALL_L + 0.05, 12.2, 4.45]), 0.46, 1.85))
    wires.append((mid_anchors[5], np.array([X_WALL_L + 0.05, 9.6, 4.65]), 0.50, 1.85))
    # Forward lines sweeping overhead across the upper sky frame
    wires.append((mid_anchors[0], np.array([0.85, 2.5, 4.95]), 0.45, 2.2))
    wires.append((mid_anchors[1], np.array([1.75, 2.2, 5.25]), 0.42, 2.15))
    wires.append((mid_anchors[4], np.array([-0.45, 3.2, 4.68]), 0.52, 2.05))
    wires.append((mid_anchors[5], np.array([-1.65, 4.4, 4.78]), 0.46, 1.95))
    draw_catenary_wires(c_mid, cam, wires, fog_rgb, rng)

    draw_hanging_paper_lantern(c_mid, cam, fog_rgb, rng)
    mid_1x = finish_cairo_2x(s_mid)
    m_mot = fbm_gouache(H, W, rng)
    m_dabs = flat_brush_dabs(H, W, rng, count=82)
    ma = mid_1x[:, :, 3:4]
    mid_1x[:, :, :3] = np.clip(mid_1x[:, :, :3] + ma * (m_mot * 0.011 + m_dabs * 0.007)[..., None], 0.0, 1.38)
    canvas = over_premul(canvas, mid_1x)

    # Volumetric acid-yellow atmospheric rain-mist halo around the glowing lantern & shop transom
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ul, vl, _ = cam.project(LANTERN_POS)
    ut, vt, _ = cam.project(TRANSOM_POS)
    d_lan = np.hypot((xx - float(ul)) / 245.0, (yy - float(vl)) / 215.0)
    d_tr = np.hypot((xx - float(ut)) / 210.0, (yy - float(vt)) / 145.0)
    mist_glow = (0.28 * np.exp(-d_lan * d_lan * 1.15) + 0.16 * np.exp(-d_tr * d_tr * 1.25)) * (0.86 + 0.14 * m_mot)
    mist_rgb = np.clip(mist_glow, 0.0, 0.7)[..., None] * ACCENT_YELLOW[None, None, :]
    canvas = 1.0 - (1.0 - np.clip(canvas, 0.0, 1.0)) * (1.0 - np.clip(mist_rgb, 0.0, 0.75))

    # Stage 3 Band 2 (Near-Midground Hero Parked Silhouette Bicycle, Utility Meter & Rain Streaks)
    s_hero, c_hero = new_cairo_2x()
    draw_parked_bicycle_3d(c_hero, cam, fog_rgb, rng)
    draw_utility_meter_box(c_hero, cam, fog_rgb)
    draw_rain_streaks_and_ripples(c_hero, cam, rng)
    canvas = over_premul(canvas, finish_cairo_2x(s_hero))

    # Stage 4: Upper-right foreground circular disc bokeh overhanging eave & rain bough + 35mm optics
    fg_bokeh = paint_foreground_bokeh_eave_bough(rng)
    canvas = over_premul(canvas, fg_bokeh)

    final_rgb = apply_35mm_finish(canvas, rng)
    elapsed = time.perf_counter() - t0
    print(f"[mira-01] Rendered {W}x{H} in {elapsed:.2f}s")

    u8 = np.round(np.clip(final_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    full_im = Image.fromarray(u8, mode="RGB")

    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        full_im.save(audit_dir / "w01_full.png", optimize=True)
        crops = {
            "w01_crop_aperture.png": (int(ul) - 240, int(vl) - 210),
            "w01_crop_midground.png": (int(ul) - 260, H - 450),
            "w01_crop_foreground.png": (430, 560),
            "w01_crop_bokeh.png": (W - 600, 0),
        }
        for cname, (cx0, cy0) in crops.items():
            x0 = max(0, min(W - 600, cx0))
            y0 = max(0, min(H - 450, cy0))
            full_im.crop((x0, y0, x0 + 600, y0 + 450)).save(audit_dir / cname)

    if webp_outputs:
        for wp in webp_outputs:
            wp.parent.mkdir(parents=True, exist_ok=True)
            full_im.save(wp, format="WEBP", quality=90, method=6)
            print(f"[mira-01] Saved {wp} ({wp.stat().st_size / 1024:.1f} KiB)")

    return final_rgb, {"elapsed": elapsed, "vp_ratio": cam.vp_x / W}


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Render Mira 01 — Obsidian Rain Alley (2048x1152)")
    ap.add_argument("--seed", type=int, default=202601)
    ap.add_argument("--audit-dir", type=pathlib.Path, default=None)
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    outputs = None if args.no_export else [
        repo_root / "session" / "themes" / "mira" / "walls" / "wall-01.webp",
        repo_root / "docs" / "assets" / "walls" / "mira-01.webp",
    ]
    render_scene(seed=args.seed, audit_dir=args.audit_dir, webp_outputs=outputs)


if __name__ == "__main__":
    main()

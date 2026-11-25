#!/usr/bin/env python3
"""
Ember 02 ("Lantern Shrine Stairway") — Pure-Python 3D-Grounded 1990s Anime Scene Pipeline (2048 x 1152)

Five-stage deterministic painting pipeline (zero image models, zero external assets):
  Stage 1: Metric right-handed Camera3D ascending a rain-slicked Japanese mountain
           shrine stairway (off-axis vanishing point ~59% width), exact 3x3
           UV-to-screen plane homographies masked by valid UV and exact ray-plane
           depth z_cam > 0.05 with exponential twilight valley haze, near-plane 3D
           line clipping, and irregular Voronoi (d2 - d1) wet volcanic basalt
           stair treads, risers, retaining wall, and mossy right parapet.
  Stage 2: Single locked normalized low-sun SUN_DIR (z < 0) tracing stone treads,
           risers, and shrine walls backward along -SUN_DIR (6 height-refinement
           steps for pitched overhanging dark-lacquer shrine eaves) for aligned
           cast shadows and volumetric valley sunbeams, unified with physically
           placed 3D mossy stone toro lantern firebox lights casting dramatic
           inverse-square golden-vermilion pools and Fresnel rain-puddle reflections
           across the dark rain-slicked masonry.
  Stage 3: 4 depth-interleaved 2x supersampled PyCairo bands (Lanczos-downsampled)
           featuring distant misty shrine sanctuary & summit vermilion torii gate,
           dramatic overhanging dark-lacquer wooden shrine eaves, hanging bronze
           shrine lantern & vermilion cedar pillars on the left, mossy stone toro
           lanterns flanking the stairway, per-instance seeded cubic-Bezier autumn
           maple branches & drifting momiji leaves — broken up by isotropic
           fbm_gouache, flat-brush dabs, cold-press paper tooth, and warm shadow
           bounce (lum < 0.28).
  Stage 4: Foreground upper-right overhanging autumn maple bough silhouette
           convolved in premultiplied RGBA with a circular aperture disc kernel
           (r = 7.8, subtle spherical aberration rim) + 1 px opposite R/B lateral shift,
           followed by 35mm multi-scale halation, radial CA, optical vignette, shadow
           black lift + C1 tanh highlight shoulder, luminance-weighted silver-halide
           grain, and faint analog YIQ transfer patina.
  Stage 5: Deterministic render (<60s CPU) exporting full 2048x1152 WebP assets plus
           optional full PNG and three 1:1 (600x450) macro audit crops.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import shutil
import time
from typing import Iterable

import cairo
import numpy as np
from PIL import Image

W, H = 2048, 1152
Z_NEAR = 0.05

# Stage 2: Single locked normalized twilight sunset travel vector (z < 0)
_raw_sun = np.array([-0.54, -0.795, -0.275], dtype=np.float64)
SUN_DIR = _raw_sun / np.linalg.norm(_raw_sun)

# Metric 3D Stairway Geometry (meters)
STAIR_Y0 = 3.20
STAIR_N_STEPS = 28
STEP_TREAD = 0.64
STEP_RISE = 0.155
STAIR_HALF_W = 1.62
STAIR_SLOPE = STEP_RISE / STEP_TREAD  # ~0.2422

# Left overhanging dark-lacquer shrine pavilion geometry (shared by shadow solver & PyCairo eaves)
SHRINE_X_WALL = -2.42
SHRINE_X_EAVE = 0.05  # sweeps dramatically further across upper-left frame!
SHRINE_Y0 = 2.8
SHRINE_Y1 = 19.8
SHRINE_EAVE_H_ABOVE_STAIR = 2.88
SHRINE_ROOF_PITCH = 0.42


def stair_z(y: np.ndarray | float) -> np.ndarray | float:
    y_cl = np.clip(y, STAIR_Y0, STAIR_Y0 + STAIR_N_STEPS * STEP_TREAD)
    return 0.36 + (y_cl - STAIR_Y0) * STAIR_SLOPE


def step_tread_z(y: float) -> float:
    """Exact horizontal tread elevation of the stone step containing world Y."""
    idx = max(0, min(STAIR_N_STEPS - 1, int(math.floor((y - STAIR_Y0) / STEP_TREAD))))
    return 0.36 + (idx + 1) * STEP_RISE


# Flanking mossy stone toro lanterns planted solidly on discrete outer stair treads (x = +/-1.34)!
LANTERN_SPECS = [
    (1.34, 6.72, step_tread_z(6.72) + 1.28, 1.62),
    (-1.34, 8.64, step_tread_z(8.64) + 1.26, 1.54),
    (1.34, 11.20, step_tread_z(11.20) + 1.26, 1.44),
    (-1.34, 13.76, step_tread_z(13.76) + 1.24, 1.36),
    (1.34, 16.32, step_tread_z(16.32) + 1.24, 1.28),
    (-1.34, 18.24, step_tread_z(18.24) + 1.22, 1.20),
    (1.34, 20.16, step_tread_z(20.16) + 1.22, 1.12),
]

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_THEME_WEBP = REPO_ROOT / "session" / "themes" / "ember" / "walls" / "wall-02.webp"
DEFAULT_DOCS_WEBP = REPO_ROOT / "docs" / "assets" / "walls" / "ember-02.webp"


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


def flat_brush_dabs(h: int, w: int, rng: np.random.Generator, count: int = 88) -> np.ndarray:
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
        c1.set_operator(cairo.OPERATOR_OVER)
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
    arr_sum = gaussian_blur_2d(arr_sum, 3.2)
    arr_sum -= arr_sum.mean()
    arr_sum /= float(arr_sum.std()) + 1e-6
    return resize_bicubic(arr_sum, w, h)


def over_premul(dst_rgb: np.ndarray, src_premul_rgba: np.ndarray) -> np.ndarray:
    a = np.clip(src_premul_rgba[:, :, 3:4], 0.0, 1.0)
    return src_premul_rgba[:, :, :3] + dst_rgb * (1.0 - a)


def apply_warm_shadow_bounce(rgb: np.ndarray, strength: float = 0.018) -> np.ndarray:
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    mask = np.clip((0.28 - lum) / 0.28, 0.0, 1.0)[..., None]
    bounce = np.array([1.28, 0.52, 0.28], dtype=np.float32) * strength
    return rgb + mask * bounce


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
        self.right_xy = np.array([right[0], right[1], 0.0], dtype=np.float64)
        self.right_xy /= np.linalg.norm(self.right_xy)
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
        fog_density: float = 0.044,
    ) -> np.ndarray:
        th, tw = tex_premul_rgba.shape[:2]
        H_mat = self.plane_homography_uv_to_screen(p0, e1, e2, tw, th)
        try:
            H_inv = np.linalg.inv(H_mat).astype(np.float32)
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

        yy = np.arange(y0, y1, dtype=np.float32)[:, None]
        xx = np.arange(x0, x1, dtype=np.float32)[None, :]
        denom = H_inv[2, 0] * xx + H_inv[2, 1] * yy + H_inv[2, 2]
        safe_d = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
        u_px = (H_inv[0, 0] * xx + H_inv[0, 1] * yy + H_inv[0, 2]) / safe_d
        v_px = (H_inv[1, 0] * xx + H_inv[1, 1] * yy + H_inv[1, 2]) / safe_d

        u_n = u_px * (1.0 / tw)
        v_n = v_px * (1.0 / th)
        valid = (u_n >= 0.0) & (u_n <= 1.0) & (v_n >= 0.0) & (v_n <= 1.0)
        if not np.any(valid):
            return dst_rgb

        r_e1 = (self.R @ e1).astype(np.float32)
        r_e2 = (self.R @ e2).astype(np.float32)
        r_p0 = (self.R @ (p0 - self.eye)).astype(np.float32)
        cam_x = r_p0[0] + u_n * r_e1[0] + v_n * r_e2[0]
        cam_y = r_p0[1] + u_n * r_e1[1] + v_n * r_e2[1]
        cam_z = r_p0[2] + u_n * r_e1[2] + v_n * r_e2[2]
        valid &= cam_z > Z_NEAR
        if not np.any(valid):
            return dst_rgb

        dist = np.sqrt(cam_x * cam_x + cam_y * cam_y + cam_z * cam_z)
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
# Stage 2 — Unified 3D Sunset Shadow Solver & Toro Lantern Pool Field
# =============================================================================

def trace_shrine_sun_visibility(wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> np.ndarray:
    """
    Trace any 3D shrine point backward along -SUN_DIR onto the pitched overhanging
    dark-lacquer shrine eaves (6 fixed height-refinement steps) AND the right mossy
    stone parapet wall.
    """
    inv_sz = np.float32(1.0 / (-SUN_DIR[2]))
    sx = np.float32(SUN_DIR[0])
    sy = np.float32(SUN_DIR[1])
    z_roof = (stair_z(wy) + SHRINE_EAVE_H_ABOVE_STAIR).astype(np.float32)
    for _ in range(6):
        t = np.maximum(0.0, (z_roof - wz) * inv_sz)
        rx = wx - sx * t
        ry = wy - sy * t
        z_roof = (stair_z(ry) + SHRINE_EAVE_H_ABOVE_STAIR + np.maximum(0.0, (SHRINE_X_EAVE - rx)) * SHRINE_ROOF_PITCH).astype(np.float32)
    t = np.maximum(0.0, (z_roof - wz) * inv_sz)
    rx = wx - sx * t
    ry = wy - sy * t
    under_eave = (wz < z_roof - 0.02) & (rx >= -5.5) & (rx <= SHRINE_X_EAVE) & (ry >= SHRINE_Y0) & (ry <= SHRINE_Y1)
    eave_soft = np.clip((rx - (SHRINE_X_EAVE - 0.20)) / 0.20, 0.0, 1.0)
    vis = np.where(under_eave, 0.18 + 0.82 * eave_soft, 1.0).astype(np.float32)

    t_p = np.where(wx < 1.72, (1.72 - wx) / np.float32(-SUN_DIR[0]), -1.0)
    py_h = wy - sy * t_p
    pz_h = wz - np.float32(SUN_DIR[2]) * t_p
    p_top = (stair_z(py_h) + 0.62).astype(np.float32)
    in_parapet = (t_p > 0.0) & (pz_h < p_top) & (py_h >= STAIR_Y0) & (py_h <= STAIR_Y0 + STAIR_N_STEPS * STEP_TREAD)
    p_soft = np.clip((pz_h - (p_top - 0.14)) / 0.14, 0.0, 1.0)
    return np.minimum(vis, np.where(in_parapet, 0.28 + 0.72 * p_soft, 1.0)).astype(np.float32)


def eval_toro_lantern_pools(
    wx: np.ndarray,
    wy: np.ndarray,
    wz: np.ndarray,
    is_horizontal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Warm inverse-square diffuse pools + wet Fresnel mirror reflection streaks from
    all flanking mossy stone toro lanterns in exact 3D world meters.
    """
    diff = np.zeros_like(wx, dtype=np.float32)
    spec = np.zeros_like(wx, dtype=np.float32)
    for lx, ly, lz, inten in LANTERN_SPECS:
        dx = wx - np.float32(lx)
        dy = wy - np.float32(ly)
        dz = wz - np.float32(lz)
        d2 = dx * dx + dy * dy + dz * dz
        diff += np.float32(inten) / (1.0 + 0.68 * d2)
        if is_horizontal:
            sx = dx * np.float32(1.0 / 0.68)
            sy = (dy + np.float32(0.55)) * np.float32(1.0 / 1.48)
            s_glow = np.exp(-0.5 * (sx * sx + sy * sy)) * (np.float32(inten) / (1.0 + 0.34 * (dx * dx + dy * dy)))
            spec += s_glow
    return np.clip(diff, 0.0, 1.9), np.clip(spec, 0.0, 1.8)


# =============================================================================
# Stage 1 & 2 — Twilight Mountain Valley Sky, Painterly Sunset Clouds & Cedar Ridges
# =============================================================================

def paint_twilight_valley_backdrop(cam: Camera3D, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Sweeping 1990s anime twilight mountain sky, billowing vermilion-gold valley clouds,
    layered misty volcanic ridges, and jagged natural Japanese cedar (sugi) tree lines.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = xx / W
    v = yy / H
    hy = float(cam.horizon_y)
    hv = hy / H

    mottle = fbm_gouache(H, W, rng)
    dabs = flat_brush_dabs(H, W, rng, count=90)
    tooth = paper_tooth(H, W, rng)

    sun_p = cam.K @ (cam.R @ (-SUN_DIR))
    sun_u = float(np.clip(sun_p[0] / sun_p[2] / W, 0.68, 0.78))
    sun_v = float(np.clip(sun_p[1] / sun_p[2] / H, 0.44, 0.56))

    t_y = np.clip(v / max(0.05, hv), 0.0, 1.0)
    c_zen = np.array([0.048, 0.018, 0.038], dtype=np.float32)
    c_plum = np.array([0.195, 0.050, 0.082], dtype=np.float32)
    c_crim = np.array([0.640, 0.185, 0.140], dtype=np.float32)
    c_verm = np.array([0.940, 0.425, 0.195], dtype=np.float32)
    c_gold = np.array([1.045, 0.725, 0.355], dtype=np.float32)

    s1 = np.clip(t_y / 0.36, 0.0, 1.0)[..., None]
    s2 = np.clip((t_y - 0.32) / 0.30, 0.0, 1.0)[..., None]
    s3 = np.clip((t_y - 0.58) / 0.26, 0.0, 1.0)[..., None]
    s4 = np.clip((t_y - 0.80) / 0.20, 0.0, 1.0)[..., None]

    sky = c_zen * (1.0 - s1) + c_plum * s1
    sky = sky * (1.0 - s2) + c_crim * s2
    sky = sky * (1.0 - s3) + c_verm * s3
    sky = sky * (1.0 - s4) + c_gold * s4

    du = (u - sun_u) * (W / H)
    dv = v - sun_v
    d_sun = np.hypot(du, dv)
    glow_w = np.exp(-(d_sun / 0.46) ** 1.55).astype(np.float32)
    glow_c = np.exp(-(d_sun / 0.15) ** 1.8).astype(np.float32)
    sky += glow_w[..., None] * np.array([0.38, 0.155, 0.062], dtype=np.float32)
    sky += glow_c[..., None] * np.array([0.52, 0.32, 0.13], dtype=np.float32)

    # Rich multi-scale hand-painted 1990s anime twilight valley cloud layers
    cb_big = resize_bicubic(rng.standard_normal((16, 26)).astype(np.float32), W, H)
    cb_mid = resize_bicubic(rng.standard_normal((34, 56)).astype(np.float32), W, H)
    cb_sm = resize_bicubic(rng.standard_normal((72, 118)).astype(np.float32), W, H)
    cb_fine = resize_bicubic(rng.standard_normal((144, 232)).astype(np.float32), W, H)
    cf = 0.48 * cb_big + 0.28 * cb_mid + 0.16 * cb_sm + 0.08 * cb_fine
    cf = (cf - cf.mean()) / (float(cf.std()) + 1e-6)

    slant = (u - 0.65) * 0.12
    b1 = np.exp(-((v - 0.21 + slant * 0.8 + 0.048 * cb_big) / 0.135) ** 2)
    b2 = np.exp(-((v - 0.42 + slant * 0.5 + 0.036 * cb_big) / 0.105) ** 2)
    b3 = np.exp(-((v - 0.59 + slant * 0.2 + 0.022 * cb_big) / 0.082) ** 2)

    raw_c = (b1 * 0.80 + b2 * 0.92 + b3 * 0.90) * np.clip(0.58 + 0.54 * cf, 0.0, 1.38)
    cl_a = np.clip((raw_c - 0.22) / 0.54, 0.0, 1.0)
    cl_sh = np.roll(np.roll(cl_a, -16, axis=0), -18, axis=1)
    cl_rim = np.clip((cl_a - cl_sh * 0.82) * 1.65, 0.0, 1.0) * (0.40 + 0.92 * glow_w)

    cl_dark = np.array([0.105, 0.034, 0.056], dtype=np.float32)[None, None, :]
    cl_mid = np.array([0.52, 0.155, 0.125], dtype=np.float32)[None, None, :]
    cl_lit = np.array([1.06, 0.64, 0.28], dtype=np.float32)[None, None, :]
    prox = np.clip(0.22 + 0.90 * glow_w, 0.0, 1.0)[..., None]

    cl_col = cl_dark * (1.0 - prox * 0.54) + cl_mid * (prox * 0.64) + cl_rim[..., None] * (cl_lit * 0.86)
    cl_col += (mottle * 0.017 + dabs * 0.015)[..., None]
    sky = sky * (1.0 - cl_a[..., None] * 0.84) + cl_col * (cl_a[..., None] * 0.84)

    # Multi-octave organic mountain valley ridges + realistic serrated cedar forest tree-line
    u_r = np.linspace(0.0, 1.0, W, dtype=np.float32)

    def fbm_1d(rgen: np.random.Generator, octaves) -> np.ndarray:
        out = np.zeros(W, dtype=np.float32)
        for nk, amp in octaves:
            vals = rgen.standard_normal(nk).astype(np.float32)
            out += amp * np.interp(np.linspace(0, nk - 1, W), np.arange(nk), vals)
        return out

    m1_n = fbm_1d(rng, ((12, 0.55), (28, 0.28), (64, 0.17)))
    m2_n = fbm_1d(rng, ((16, 0.45), (42, 0.28), (110, 0.16), (260, 0.11)))

    # Distant misty blue-crimson mountain range
    m1_h = 0.565 - 0.082 * np.exp(-((u_r - 0.88) / 0.18) ** 2) - 0.048 * np.exp(-((u_r - 0.58) / 0.17) ** 2) - 0.016 * np.abs(m1_n)
    # Nearer pine-clad dark mountain ridge sloping down into glowing valley mist (organic cedar crowns!)
    m2_h = 0.685 - 0.105 * np.exp(-((u_r - 0.96) / 0.22) ** 2) - 0.032 * np.exp(-((u_r - 0.72) / 0.13) ** 2) - 0.018 * np.abs(m2_n)
    # Summit shrine sanctuary forest backdrop directly behind the Torii gate (u ~ 0.35..0.70, v ~ 0.44..0.60)
    summit_trees = 0.435 + 0.032 * np.abs(fbm_1d(rng, ((24, 0.48), (68, 0.32), (160, 0.20)))) + 0.08 * np.clip((u_r - 0.60) / 0.16, 0.0, 1.0)

    a_m1 = np.clip((v - m1_h[None, :]) / 0.0025, 0.0, 1.0)
    c_m1 = np.array([0.26, 0.095, 0.100], dtype=np.float32)[None, None, :] + glow_w[..., None] * np.array([0.24, 0.11, 0.050], dtype=np.float32)
    sky = sky * (1.0 - a_m1[..., None]) + c_m1 * a_m1[..., None]

    # Glowing luminous vermilion-amber valley cloud sea mist floating between mountain 1 & 2!
    mist_band = np.exp(-((v - m2_h[None, :] + 0.032) / 0.058) ** 2).astype(np.float32) * (0.55 + 0.45 * cb_big)
    sky += np.clip(mist_band, 0.0, 1.0)[..., None] * np.array([0.52, 0.24, 0.11], dtype=np.float32) * (0.55 + 0.75 * glow_w[..., None])

    # Soft misty summit sanctuary cedar silhouette behind Torii (keeps upper Torii glowing against sky!)
    a_sum = np.clip((v - summit_trees[None, :]) / 0.004, 0.0, 1.0) * np.clip((0.72 - u_r[None, :]) / 0.08, 0.0, 1.0) * np.clip((v - 0.40) / 0.08, 0.0, 1.0)
    c_sum = np.array([0.145, 0.055, 0.062], dtype=np.float32)[None, None, :] + glow_w[..., None] * np.array([0.14, 0.06, 0.03], dtype=np.float32)
    sky = sky * (1.0 - a_sum[..., None] * 0.78) + c_sum * (a_sum[..., None] * 0.78)

    a_m2 = np.clip((v - m2_h[None, :]) / 0.0022, 0.0, 1.0)
    c_m2 = np.array([0.075, 0.028, 0.038], dtype=np.float32)[None, None, :] + glow_w[..., None] * np.array([0.08, 0.036, 0.020], dtype=np.float32)
    c_m2 += (mottle * 0.011 + dabs * 0.008)[..., None]
    sky = sky * (1.0 - a_m2[..., None]) + c_m2 * a_m2[..., None]

    # Deep volcanic valley shadow toward bottom-right
    deep_v = np.clip((v - 0.76) / 0.24, 0.0, 1.0)[..., None]
    sky = sky * (1.0 - deep_v * 0.62) + np.array([0.034, 0.014, 0.018], dtype=np.float32) * (deep_v * 0.62)

    sky += (mottle * 0.010 + dabs * 0.009 + tooth * 0.005)[..., None]
    fog_rgb = np.array([0.36, 0.145, 0.110], dtype=np.float32)
    return np.clip(sky, 0.0, 1.5), fog_rgb


# =============================================================================
# Stage 1 & 2 — Irregular Voronoi Basalt Masonry Atlas (Chiseled Dark Wet Stone!)
# =============================================================================

def eval_voronoi_field(
    a_m: np.ndarray,
    b_m: np.ndarray,
    cell_a: float,
    cell_b: float,
    rng: np.random.Generator,
    perturb: float = 0.017,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized 3x3 Voronoi cell neighborhood in physical meters with subtle chiseled
    stone perturbation (d2 - d1 mortar crevices, natural stone-to-stone tone variation).
    """
    th, tw = a_m.shape
    nx = resize_bicubic(rng.standard_normal((max(4, th // 22), max(4, tw // 22))).astype(np.float32), tw, th) * np.float32(perturb)
    ny = resize_bicubic(rng.standard_normal((max(4, th // 22), max(4, tw // 22))).astype(np.float32), tw, th) * np.float32(perturb)
    px, py = a_m + nx, b_m + ny

    ca, cb = np.float32(cell_a), np.float32(cell_b)
    ix = np.floor(px / ca).astype(np.int32)
    iy = np.floor(py / cb).astype(np.int32)

    jx_tab = rng.uniform(0.22, 0.78, size=256).astype(np.float32)
    jy_tab = rng.uniform(0.22, 0.78, size=256).astype(np.float32)
    tone_tab = rng.uniform(-0.045, 0.055, size=256).astype(np.float32)
    warm_tab = rng.uniform(-0.014, 0.024, size=256).astype(np.float32)

    d1 = np.full((th, tw), 1e9, dtype=np.float32)
    d2 = np.full((th, tw), 1e9, dtype=np.float32)
    best_id = np.zeros((th, tw), dtype=np.int32)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cx, cy = ix + dx, iy + dy
            h_id = ((cx * 73856093) ^ (cy * 19349663)) & 255
            sx = (cx + jx_tab[h_id]) * ca
            sy = (cy + jy_tab[h_id]) * cb
            dist = np.hypot(px - sx, py - sy)
            closer = dist < d1
            d2 = np.where(closer, d1, np.minimum(d2, dist))
            d1 = np.where(closer, dist, d1)
            best_id = np.where(closer, h_id, best_id)

    # Crisp natural hairline basalt stone joints (never thick cartoon black webs!)
    mortar = np.clip(((d2 - d1) - 0.006) / 0.014, 0.0, 1.0)
    bevel = np.clip(((d2 - d1) - 0.014) / 0.036, 0.0, 1.0)
    return mortar, bevel, tone_tab[best_id], warm_tab[best_id]


def paint_stairway_tread_atlas(rng: np.random.Generator) -> np.ndarray:
    """
    Deep volcanic obsidian-charcoal rain-slicked Voronoi basalt stone steps so the
    glowing golden-vermilion toro lantern pools & mirror puddles radiate dramatically!
    """
    tw, th = 768, 1280
    y_end = STAIR_Y0 + STAIR_N_STEPS * STEP_TREAD
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    wx = np.float32(-STAIR_HALF_W) + (xx / (tw - 1)) * np.float32(2.0 * STAIR_HALF_W)
    wy = np.float32(STAIR_Y0) + (yy / (th - 1)) * np.float32(y_end - STAIR_Y0)
    wz = stair_z(wy).astype(np.float32)

    mortar, bevel, sv, sw = eval_voronoi_field(wx, wy, cell_a=0.58, cell_b=0.56, rng=rng, perturb=0.016)
    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=88)
    tooth = paper_tooth(th, tw, rng)

    # Dark wet obsidian-charcoal volcanic basalt slabs (subtle warm stone mineral variation!)
    stone = np.zeros((th, tw, 3), dtype=np.float32)
    stone[:, :, 0] = 0.088 + sv * 0.45 + sw * 0.48 + mottle * 0.016 + dabs * 0.013
    stone[:, :, 1] = 0.062 + sv * 0.36 + sw * 0.22 + mottle * 0.012 + dabs * 0.010
    stone[:, :, 2] = 0.064 + sv * 0.32 + mottle * 0.011 + dabs * 0.009

    # Subtle natural mortar shadow (dark slate-charcoal, harmoniously blending with dark basalt!)
    crevice = np.array([0.024, 0.015, 0.016], dtype=np.float32)
    rgb = mortar[..., None] * stone + (1.0 - mortar[..., None]) * crevice

    # Velvety olive-moss clinging naturally along shaded outer stair edges & crevices
    flank = np.clip((np.abs(wx) - 0.92) / 0.72, 0.0, 1.0)
    moss = np.clip((flank * 0.85 + (1.0 - mortar) * 0.38 + mottle * 0.28 - 0.28) * 1.4, 0.0, 0.78)
    rgb = rgb * (1.0 - moss[..., None]) + np.array([0.056, 0.082, 0.038], dtype=np.float32) * moss[..., None]

    vis = trace_shrine_sun_visibility(wx, wy, wz)
    l_diff, l_spec = eval_toro_lantern_pools(wx, wy, wz, is_horizontal=True)

    step_f = np.mod((wy - np.float32(STAIR_Y0)) / np.float32(STEP_TREAD), 1.0)
    nosing_hi = np.exp(-((step_f - 0.055) / 0.060) ** 2) * (0.4 + 0.6 * mortar)

    # Worn glossy rain puddles down the center of ancient shrine steps reflecting crimson twilight & lanterns!
    center_wear = np.exp(-((wx / 0.96) ** 2))
    puddle = np.clip(0.24 + 0.44 * mottle + 0.40 * center_wear, 0.0, 1.0) * (0.5 + 0.5 * mortar)

    dist_t = np.clip((wy - np.float32(STAIR_Y0)) / np.float32(y_end - STAIR_Y0), 0.0, 1.0)
    fresnel = 0.22 + (0.76 - 0.22) * np.power(1.0 - dist_t, 1.4)

    rgb *= (0.34 + 0.66 * vis[..., None])

    # Deep crimson-vermilion twilight sky gloss reflected across rain-slicked center puddles
    sky_refl = np.array([0.44, 0.165, 0.115], dtype=np.float32)[None, None, :] * (0.38 + 0.62 * vis[..., None])
    wet_sky = (puddle * fresnel * 0.44)[..., None]
    rgb = rgb * (1.0 - wet_sky) + sky_refl * wet_sky

    # DRAMATIC glowing golden-vermilion pools & glossy mirror streaks from flanking stone toro lanterns!
    rgb += (l_diff * (0.70 + 0.30 * mortar))[..., None] * np.array([0.66, 0.28, 0.085], dtype=np.float32)
    rgb += (l_spec * puddle)[..., None] * np.array([1.05, 0.56, 0.17], dtype=np.float32)

    # Crisp glistening wet highlight along front stone step nosings
    rgb += (nosing_hi * (0.20 + 0.48 * vis + 0.85 * l_diff))[..., None] * np.array([0.48, 0.24, 0.10], dtype=np.float32)
    rgb += (np.clip(1.0 - bevel, 0.0, 1.0) * mortar * (0.14 + 0.45 * l_diff))[..., None] * np.array([0.30, 0.14, 0.06], dtype=np.float32)

    rgb = apply_warm_shadow_bounce(rgb, strength=0.016)
    rgb += tooth[..., None] * 0.007

    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.45)
    return rgba


def paint_stairway_riser_atlas(rng: np.random.Generator) -> np.ndarray:
    """Deep shadowed wet volcanic basalt vertical step risers catching soft lantern spill."""
    tw, th = 768, 448
    y_end = STAIR_Y0 + STAIR_N_STEPS * STEP_TREAD
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    wx = np.float32(-STAIR_HALF_W) + (xx / (tw - 1)) * np.float32(2.0 * STAIR_HALF_W)
    wy = np.float32(STAIR_Y0) + (yy / (th - 1)) * np.float32(y_end - STAIR_Y0)
    wz = (stair_z(wy) - STEP_RISE * 0.5).astype(np.float32)

    mortar, _, sv, sw = eval_voronoi_field(wx, wy * 0.45, cell_a=0.58, cell_b=0.28, rng=rng, perturb=0.012)
    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=80)
    tooth = paper_tooth(th, tw, rng)

    stone = np.zeros((th, tw, 3), dtype=np.float32)
    stone[:, :, 0] = 0.052 + sv * 0.28 + sw * 0.28 + mottle * 0.012 + dabs * 0.009
    stone[:, :, 1] = 0.034 + sv * 0.22 + mottle * 0.009 + dabs * 0.007
    stone[:, :, 2] = 0.036 + sv * 0.20 + mottle * 0.008 + dabs * 0.006

    rgb = mortar[..., None] * stone + (1.0 - mortar[..., None]) * np.array([0.016, 0.009, 0.010], dtype=np.float32)
    l_diff, _ = eval_toro_lantern_pools(wx, wy, wz, is_horizontal=False)
    rgb += (l_diff * (0.75 + 0.25 * mortar))[..., None] * np.array([0.36, 0.145, 0.048], dtype=np.float32)
    rgb = apply_warm_shadow_bounce(rgb, strength=0.015)
    rgb += tooth[..., None] * 0.006

    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.25)
    return rgba


def paint_parapet_wall_uv(
    rng: np.random.Generator,
    side: str,  # "right_inner" | "right_top" | "left_wall"
    tw: int = 896,
    th: int = 288,
) -> np.ndarray:
    """Chiseled mossy dark volcanic basalt shrine retaining curb & parapet coping (zero stretching!)."""
    y_end = STAIR_Y0 + STAIR_N_STEPS * STEP_TREAD
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wy = np.float32(STAIR_Y0) + u * np.float32(y_end - STAIR_Y0)

    if side == "right_top":
        wx = np.float32(STAIR_HALF_W) + v * np.float32(0.42)
        wz = (stair_z(wy) + 0.62).astype(np.float32)
        # Uniform square-ish metric basalt cap blocks along the parapet coping
        a_m, b_m = wy, wx
        ca, cb, pert = 0.72, 0.46, 0.011
    elif side == "right_inner":
        wx = np.full_like(wy, STAIR_HALF_W)
        wz = (stair_z(wy) + v * 0.62).astype(np.float32)
        a_m, b_m = wy, v * np.float32(0.62)
        ca, cb, pert = 0.68, 0.34, 0.014
    else:
        wx = np.full_like(wy, -STAIR_HALF_W)
        wz = (stair_z(wy) + v * 0.72).astype(np.float32)
        a_m, b_m = wy, v * np.float32(0.72)
        ca, cb, pert = 0.68, 0.36, 0.014

    mortar, bevel, sv, sw = eval_voronoi_field(a_m, b_m, cell_a=ca, cell_b=cb, rng=rng, perturb=pert)
    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=82)
    tooth = paper_tooth(th, tw, rng)

    stone = np.zeros((th, tw, 3), dtype=np.float32)
    stone[:, :, 0] = 0.088 + sv * 0.42 + sw * 0.45 + mottle * 0.015 + dabs * 0.012
    stone[:, :, 1] = 0.060 + sv * 0.32 + mottle * 0.011 + dabs * 0.009
    stone[:, :, 2] = 0.062 + sv * 0.28 + mottle * 0.010 + dabs * 0.008
    rgb = mortar[..., None] * stone + (1.0 - mortar[..., None]) * np.array([0.020, 0.011, 0.012], dtype=np.float32)

    moss_bias = 0.36 if side == "right_top" else 0.14
    moss = np.clip((moss_bias + (1.0 - mortar) * 0.48 + mottle * 0.34), 0.0, 0.80)
    rgb = rgb * (1.0 - moss[..., None]) + np.array([0.060, 0.092, 0.040], dtype=np.float32) * moss[..., None]

    vis = trace_shrine_sun_visibility(wx, wy, wz)
    l_diff, _ = eval_toro_lantern_pools(wx, wy, wz, is_horizontal=(side == "right_top"))

    if side == "right_top":
        rgb += (mortar * (0.35 + 0.65 * vis))[..., None] * np.array([0.28, 0.125, 0.060], dtype=np.float32)
        rgb += (np.clip(1.0 - bevel, 0.0, 1.0) * mortar)[..., None] * np.array([0.24, 0.11, 0.052], dtype=np.float32)
    elif side == "left_wall":
        rgb *= (0.42 + 0.58 * vis[..., None])
        rgb += (vis * mortar)[..., None] * np.array([0.16, 0.068, 0.032], dtype=np.float32)
    else:
        rgb *= 0.48

    rgb += (l_diff * (0.65 + 0.35 * mortar))[..., None] * np.array([0.52, 0.23, 0.075], dtype=np.float32)
    rgb = apply_warm_shadow_bounce(rgb, strength=0.016)
    rgb += tooth[..., None] * 0.007

    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.35)
    return rgba


def paint_shrine_facade_wall_uv(rng: np.random.Generator) -> np.ndarray:
    """
    Left shrine pavilion dark obsidian-lacquer cedar wall with subtle deep-cinnabar
    structural rails & warm glowing koshi wood-lattice sanctuary windows.
    """
    tw, th = 896, 448
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wy = np.float32(SHRINE_Y0) + u * np.float32(SHRINE_Y1 - SHRINE_Y0)
    z_base = stair_z(wy).astype(np.float32) + np.float32(0.55)
    wz = z_base + v * np.float32(2.85)
    wx = np.full_like(wy, SHRINE_X_WALL)

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=82)
    tooth = paper_tooth(th, tw, rng)

    bid = np.floor(wy / 0.24).astype(np.int32)
    bf = (wy / 0.24) - bid
    b_seam = np.clip(np.minimum(bf, 1.0 - bf) / 0.055, 0.0, 1.0)

    # Rich dark charred-lacquer shrine cedar wall (keeps vermilion pillars & lantern fire heroic!)
    wall = np.zeros((th, tw, 3), dtype=np.float32)
    wall[:, :, 0] = 0.078 + mottle * 0.014 + dabs * 0.010
    wall[:, :, 1] = 0.032 + mottle * 0.009 + dabs * 0.007
    wall[:, :, 2] = 0.032 + mottle * 0.008 + dabs * 0.006
    wall = b_seam[..., None] * wall + (1.0 - b_seam[..., None]) * np.array([0.014, 0.006, 0.008], dtype=np.float32)

    # Restrained deep Kyoto cinnabar-lacquer structural rails (nuki) — rich & deep, NOT neon stripes!
    is_beam = (np.abs(v - 0.14) < 0.038) | (np.abs(v - 0.78) < 0.044)
    verm = np.array([0.34, 0.072, 0.052], dtype=np.float32) + mottle[..., None] * 0.016
    wall = np.where(is_beam[..., None], verm, wall)

    vis = trace_shrine_sun_visibility(wx, wy, wz)
    wall *= (0.34 + 0.66 * vis[..., None])
    wall += (vis * b_seam)[..., None] * np.array([0.14, 0.052, 0.025], dtype=np.float32)

    # Warm candlelit shoji/koshi lattice windows tucked along the shrine corridor
    for wy0, wy1 in ((6.2, 8.2), (10.8, 12.8), (15.4, 17.3)):
        in_w = (wy >= wy0) & (wy <= wy1) & (v >= 0.22) & (v <= 0.70)
        fu = (wy - wy0) / (wy1 - wy0)
        fv = (v - 0.22) / 0.48
        lat = (np.mod(fu * 7.0, 1.0) < 0.19) | (np.mod(fv * 4.0, 1.0) < 0.15)
        shoji_c = np.array([0.96, 0.48, 0.15], dtype=np.float32)[None, None, :] * (0.55 + 0.45 * (1.0 - fv)[..., None])
        lat_c = np.array([0.048, 0.016, 0.018], dtype=np.float32)
        wall = np.where(in_w[..., None], np.where(lat[..., None], lat_c, shoji_c), wall)

    l_diff, _ = eval_toro_lantern_pools(wx, wy, wz, is_horizontal=False)
    wall += l_diff[..., None] * np.array([0.32, 0.135, 0.048], dtype=np.float32)
    wall = apply_warm_shadow_bounce(wall, strength=0.016)
    wall += tooth[..., None] * 0.007

    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(wall, 0.0, 1.4)
    return rgba


# =============================================================================
# Stage 2 & 3 — Volumetric Twilight Valley Sunbeams Aligned With SUN_DIR
# =============================================================================

def paint_valley_sunbeams(cam: Camera3D, rng: np.random.Generator, intensity: float = 0.36) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    p_ref = np.array([0.0, 12.0, 3.2], dtype=np.float64)
    u0, v0, _ = cam.project(p_ref)
    u1, v1, _ = cam.project(p_ref + SUN_DIR * 3.0)
    dx, dy = float(u1 - u0), float(v1 - v0)
    dlen = math.hypot(dx, dy) + 1e-6
    rx, ry = dx / dlen, dy / dlen
    nx, ny = -ry, rx

    s_coord = (xx - W * 0.78) * rx + (yy - H * 0.44) * ry
    p_coord = (xx - W * 0.78) * nx + (yy - H * 0.44) * ny

    acc = np.zeros((H, W), dtype=np.float32)
    for cp, wp, st in [(-240, 115, 0.40), (-55, 95, 0.56), (135, 110, 0.48), (310, 95, 0.34)]:
        w_eff = wp * (1.0 + np.maximum(0.0, s_coord) / (W * 1.1))
        beam = np.exp(-0.5 * ((p_coord - cp) / w_eff) ** 2)
        along = np.clip((s_coord + 260.0) / 320.0, 0.0, 1.0) * np.exp(-np.maximum(0.0, s_coord) / (W * 0.72))
        acc += beam * along * st

    turb = 0.85 + 0.15 * resize_bicubic(fbm_gouache(H // 4, W // 4, rng), W, H)
    acc = np.clip(acc * turb * intensity, 0.0, 0.72)
    return acc[..., None] * np.array([1.05, 0.50, 0.22], dtype=np.float32)[None, None, :]


# =============================================================================
# Stage 3 — Fast Tight-BBox 2x Supersampled PyCairo Shrine Props & Autumn Maples
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
    a_full = buf[:, :, 3]
    nz_y = np.any(a_full > 0, axis=1)
    nz_x = np.any(a_full > 0, axis=0)
    out = np.zeros((H, W, 4), dtype=np.float32)
    if not np.any(nz_y) or not np.any(nz_x):
        return out

    y0 = max(0, (int(np.argmax(nz_y)) // 2) - 2)
    y1 = min(H, ((H * 2 - int(np.argmax(nz_y[::-1])) + 1) // 2) + 2)
    x0 = max(0, (int(np.argmax(nz_x)) // 2) - 2)
    x1 = min(W, ((W * 2 - int(np.argmax(nz_x[::-1])) + 1) // 2) + 2)
    bw, bh = x1 - x0, y1 - y0
    sub = buf[y0 * 2 : y1 * 2, x0 * 2 : x1 * 2]

    a1 = np.asarray(Image.fromarray(sub[:, :, 3], mode="L").resize((bw, bh), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    a1 = np.clip(a1, 0.0, 1.0)
    out[y0:y1, x0:x1, 3] = a1
    for out_c, src_c in ((0, 2), (1, 1), (2, 0)):
        c1 = np.asarray(Image.fromarray(sub[:, :, src_c], mode="L").resize((bw, bh), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
        straight = np.where(a1 > 1e-4, np.clip(c1 / np.maximum(a1, 1e-4), 0.0, 1.0), 0.0)
        out[y0:y1, x0:x1, out_c] = straight * a1
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
    tremor: float = 0.25,
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
        ctx.set_line_width(max(0.5, w_eff * wt))
        ctx.move_to(*dense[i])
        ctx.line_to(*dense[i + 1])
        ctx.stroke()


def stroke_arch_edges(
    ctx: cairo.Context,
    poly: list[tuple[float, float]],
    base_w: float,
    z_depth: float,
    rgba: tuple[float, float, float, float],
    rng: np.random.Generator,
    fade_dist: float = 24.0,
) -> None:
    """
    Crisp hand-inked architectural polygon outline stroked edge-by-edge (ZERO Catmull-Rom
    overshoot loops around sharp 90-degree stone masonry corners!).
    """
    decay = float(np.clip(1.0 / (1.0 + z_depth / (fade_dist * 0.45)), 0.25, 1.0))
    ctx.set_source_rgba(*rgba)
    ctx.set_line_width(max(0.6, base_w * decay))
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        mid_x = 0.5 * (a[0] + b[0]) + float(rng.normal(0.0, 0.22 * decay))
        mid_y = 0.5 * (a[1] + b[1]) + float(rng.normal(0.0, 0.22 * decay))
        ctx.move_to(*a)
        ctx.line_to(mid_x, mid_y)
        ctx.line_to(*b)
        ctx.stroke()


def project_quad_3d(cam: Camera3D, pts3: np.ndarray) -> list[tuple[float, float]] | None:
    poly: list[tuple[float, float]] = []
    n = len(pts3)
    for i in range(n):
        cl = cam.clip_segment_3d(pts3[i], pts3[(i + 1) % n])
        if cl is None:
            continue
        for p in cl:
            u, v, _ = cam.project(p)
            pt = (float(u), float(v))
            if not poly or math.hypot(pt[0] - poly[-1][0], pt[1] - poly[-1][1]) > 0.35:
                poly.append(pt)
    return poly if len(poly) >= 3 else None


def fill_poly(ctx: cairo.Context, poly: list[tuple[float, float]] | None) -> None:
    if not poly or len(poly) < 3:
        return
    ctx.move_to(*poly[0])
    for p in poly[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    ctx.fill()


def fog_mix(c3: Iterable[float], dist: float, fog_rgb: np.ndarray, fog_start: float = 6.5, fog_density: float = 0.044) -> tuple[float, float, float]:
    ft = 1.0 - math.exp(-max(0.0, dist - fog_start) * fog_density)
    c = np.asarray(c3, dtype=np.float64) * (1.0 - ft) + fog_rgb * ft
    return (float(c[0]), float(c[1]), float(c[2]))


def draw_summit_sanctuary_and_torii(
    ctx: cairo.Context,
    cam: Camera3D,
    y_torii: float,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
) -> None:
    """
    Distant misty shrine sanctuary gable silhouette & stone summit landing behind a
    monumental vermilion Torii gate perched at the crest of the stairway.
    """
    z0 = float(stair_z(y_torii))

    # 1. Distant misty main shrine sanctuary hall (haiden, y ~ 29 m) with CRISP concave upturned temple eaves!
    y_sanc = 29.0
    dist_s = float(np.linalg.norm(np.array([0.0, y_sanc, z0 + 2.0]) - cam.eye))
    # Sanctuary dark timber lower facade + subtle vermilion pillars + warm candlelit central altar doors!
    sanc_wall = np.array([
        [-3.2, y_sanc, z0 - 0.05], [3.2, y_sanc, z0 - 0.05],
        [3.2, y_sanc, z0 + 1.72], [-3.2, y_sanc, z0 + 1.72],
    ])
    ctx.set_source_rgba(*fog_mix([0.075, 0.028, 0.034], dist_s, fog_rgb), 0.94)
    fill_poly(ctx, project_quad_3d(cam, sanc_wall))
    # Warm glowing paper shoji sanctuary altar doorway centered through the Torii!
    sanc_door = np.array([
        [-0.78, y_sanc - 0.05, z0 + 0.25], [0.78, y_sanc - 0.05, z0 + 0.25],
        [0.78, y_sanc - 0.05, z0 + 1.48], [-0.78, y_sanc - 0.05, z0 + 1.48],
    ])
    ctx.set_source_rgba(*fog_mix([0.92, 0.44, 0.16], dist_s * 0.85, fog_rgb), 0.92)
    fill_poly(ctx, project_quad_3d(cam, sanc_door))
    # Flanking sanctuary veranda pillars
    for sx_p in (-2.4, -1.1, 1.1, 2.4):
        sp_q = np.array([
            [sx_p - 0.12, y_sanc - 0.08, z0], [sx_p + 0.12, y_sanc - 0.08, z0],
            [sx_p + 0.12, y_sanc - 0.08, z0 + 1.75], [sx_p - 0.12, y_sanc - 0.08, z0 + 1.75],
        ])
        ctx.set_source_rgba(*fog_mix([0.36, 0.085, 0.062], dist_s, fog_rgb), 0.95)
        fill_poly(ctx, project_quad_3d(cam, sp_q))

    # Crisp iconic CONCAVE upturned Japanese irimoya temple roof silhouette (upturned corner eaves + sharp ridge!)
    sanc_roof = np.array([
        [-4.55, y_sanc - 0.15, z0 + 1.98],  # crisp upturned left eave tip (sori!)
        [-3.40, y_sanc - 0.15, z0 + 1.72],  # sweeping concave left eave dip
        [3.40, y_sanc - 0.15, z0 + 1.72],   # sweeping concave right eave dip
        [4.55, y_sanc - 0.15, z0 + 1.98],   # crisp upturned right eave tip (sori!)
        [3.05, y_sanc - 0.10, z0 + 2.52],   # concave hip slope
        [1.75, y_sanc, z0 + 3.45],          # sharp horizontal ridge right end
        [-1.75, y_sanc, z0 + 3.45],         # sharp horizontal ridge left end
        [-3.05, y_sanc - 0.10, z0 + 2.52],  # concave hip slope
    ])
    ctx.set_source_rgba(*fog_mix([0.055, 0.020, 0.028], dist_s, fog_rgb), 0.96)
    fill_poly(ctx, project_quad_3d(cam, sanc_roof))
    # Horizontal roof ridge beam (mune) catching warm twilight sunset rim!
    ridge_seg = cam.project_polyline(np.array([[-1.95, y_sanc, z0 + 3.52], [1.95, y_sanc, z0 + 3.52]]))
    if len(ridge_seg) >= 2:
        stroke_catmull_rom(ctx, [(p[0], p[1]) for p in ridge_seg], base_w=2.4, z_depth=dist_s,
                           rgba=(*fog_mix([0.96, 0.52, 0.22], dist_s, fog_rgb), 0.72), rng=rng, taper=False, tremor=0.10)

    # Warm sanctuary altar glow spill directly centered beneath Torii!
    us, vs, zs = cam.project(np.array([0.0, y_sanc - 0.5, z0 + 0.95]))
    if zs > 2.0:
        rs = max(4.0, float(cam.K[0, 0] * 0.30 / zs))
        g_sl = cairo.RadialGradient(float(us), float(vs), rs * 0.15, float(us), float(vs), rs * 4.8)
        g_sl.add_color_stop_rgba(0.0, 1.0, 0.84, 0.44, 0.74)
        g_sl.add_color_stop_rgba(0.45, 0.96, 0.46, 0.16, 0.28)
        g_sl.add_color_stop_rgba(1.0, 0.85, 0.24, 0.08, 0.0)
        ctx.set_source(g_sl)
        ctx.arc(float(us), float(vs), rs * 4.8, 0, math.tau)
        ctx.fill()

    # 2. Wet stone summit courtyard ground plane spanning y = y_torii - 0.3 .. 29.5 (no abrupt cliff!)
    courtyard = np.array([
        [-3.4, y_torii - 0.2, z0 + 0.02],
        [3.4, y_torii - 0.2, z0 + 0.02],
        [4.5, y_sanc, z0 + 0.02],
        [-4.5, y_sanc, z0 + 0.02],
    ])
    ctx.set_source_rgba(*fog_mix([0.14, 0.068, 0.056], dist_s * 0.85, fog_rgb), 0.96)
    fill_poly(ctx, project_quad_3d(cam, courtyard))

    # 3. Monumental Vermilion Torii Gate perched at summit crest
    dist = float(np.linalg.norm(np.array([0.0, y_torii, z0 + 2.2]) - cam.eye))
    p_h = 3.55
    half_span = 1.48

    for sgn in (-1.0, 1.0):
        xb = sgn * half_span
        xt = sgn * (half_span - 0.14)
        q_pl = np.array([
            [xb - 0.24, y_torii, z0], [xb + 0.24, y_torii, z0],
            [xb + 0.20, y_torii, z0 + 0.45], [xb - 0.20, y_torii, z0 + 0.45],
        ])
        ctx.set_source_rgba(*fog_mix([0.042, 0.018, 0.020], dist, fog_rgb), 0.98)
        fill_poly(ctx, project_quad_3d(cam, q_pl))

        q_col = np.array([
            [xb - 0.17, y_torii, z0 + 0.42], [xb + 0.17, y_torii, z0 + 0.42],
            [xt + 0.15, y_torii, z0 + p_h], [xt - 0.15, y_torii, z0 + p_h],
        ])
        pc = project_quad_3d(cam, q_col)
        if pc is not None:
            xs = [p[0] for p in pc]
            g = cairo.LinearGradient(min(xs), pc[0][1], max(xs), pc[0][1])
            g.add_color_stop_rgba(0.0, *fog_mix([0.24, 0.050, 0.046], dist, fog_rgb), 0.98)
            g.add_color_stop_rgba(0.65, *fog_mix([0.66, 0.160, 0.100], dist, fog_rgb), 0.98)
            g.add_color_stop_rgba(1.0, *fog_mix([0.98, 0.46, 0.21], dist, fog_rgb), 0.98)
            ctx.set_source(g)
            fill_poly(ctx, pc)

    q_nuki = np.array([
        [-half_span - 0.54, y_torii, z0 + p_h - 0.72], [half_span + 0.54, y_torii, z0 + p_h - 0.72],
        [half_span + 0.54, y_torii, z0 + p_h - 0.46], [-half_span - 0.54, y_torii, z0 + p_h - 0.46],
    ])
    ctx.set_source_rgba(*fog_mix([0.56, 0.140, 0.092], dist, fog_rgb), 0.98)
    fill_poly(ctx, project_quad_3d(cam, q_nuki))

    q_plq = np.array([
        [-0.24, y_torii - 0.04, z0 + p_h - 0.64], [0.24, y_torii - 0.04, z0 + p_h - 0.64],
        [0.24, y_torii - 0.04, z0 + p_h + 0.04], [-0.24, y_torii - 0.04, z0 + p_h + 0.04],
    ])
    ctx.set_source_rgba(*fog_mix([0.052, 0.020, 0.022], dist, fog_rgb), 0.98)
    fill_poly(ctx, project_quad_3d(cam, q_plq))

    n_k = 13
    xs_k = np.linspace(-half_span - 0.90, half_span + 0.90, n_k)
    curve = 0.25 * np.power(np.abs(xs_k) / (half_span + 0.90), 2.3)
    bot_k = np.column_stack([xs_k, np.full(n_k, y_torii), z0 + p_h - 0.04 + curve])
    mid_k = np.column_stack([xs_k, np.full(n_k, y_torii), z0 + p_h + 0.24 + curve])
    top_k = np.column_stack([xs_k * 1.04, np.full(n_k, y_torii), z0 + p_h + 0.46 + curve * 1.15])

    ctx.set_source_rgba(*fog_mix([0.74, 0.195, 0.110], dist, fog_rgb), 0.98)
    fill_poly(ctx, project_quad_3d(cam, np.vstack([bot_k, mid_k[::-1]])))
    ctx.set_source_rgba(*fog_mix([0.058, 0.022, 0.026], dist, fog_rgb), 0.98)
    fill_poly(ctx, project_quad_3d(cam, np.vstack([mid_k, top_k[::-1]])))

    tseg = cam.project_polyline(top_k)
    if len(tseg) >= 2:
        stroke_catmull_rom(ctx, [(p[0], p[1]) for p in tseg], base_w=2.4, z_depth=dist,
                           rgba=(*fog_mix([1.0, 0.62, 0.28], dist, fog_rgb), 0.80), rng=rng, taper=True, tremor=0.14)


def draw_stone_toro_lantern(
    ctx: cairo.Context,
    cam: Camera3D,
    lx: float,
    ly: float,
    lz_fire: float,
    intensity: float,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
) -> None:
    """
    Hand-painted 1990s anime mossy volcanic stone toro lantern (kasuga-doro) with
    CLEAN edge-by-edge hand-inked architectural masonry lines (zero spline overshoot!).
    """
    z_base = step_tread_z(ly) + 0.01
    p_fire = np.array([lx, ly, lz_fire], dtype=np.float64)
    dist = float(np.linalg.norm(p_fire - cam.eye))
    uf, vf, zf = cam.project(p_fire)
    if zf < 1.2 or uf < -180 or uf > W + 180:
        return

    px_m = float(cam.K[0, 0] / zf)

    # 1. Warm atmospheric radial lantern glow halo
    halo_r = px_m * 1.60 * (0.85 + 0.15 * intensity)
    g_h = cairo.RadialGradient(float(uf), float(vf), px_m * 0.05, float(uf), float(vf), halo_r)
    g_h.add_color_stop_rgba(0.0, 1.0, 0.88, 0.46, 0.66)
    g_h.add_color_stop_rgba(0.24, 0.98, 0.50, 0.15, 0.30)
    g_h.add_color_stop_rgba(0.60, 0.88, 0.26, 0.075, 0.11)
    g_h.add_color_stop_rgba(1.0, 0.75, 0.16, 0.05, 0.0)
    ctx.set_source(g_h)
    ctx.arc(float(uf), float(vf), halo_r, 0, math.tau)
    ctx.fill()

    def proj_pt(dx: float, dz: float) -> tuple[float, float]:
        u, v, _ = cam.project(np.array([lx + dx, ly, z_base + dz], dtype=np.float64))
        return (float(u), float(v))

    def draw_tier(w_bot: float, w_top: float, z0: float, z1: float, c_left, c_right) -> None:
        pts = [proj_pt(-w_bot, z0), proj_pt(w_bot, z0), proj_pt(w_top, z1), proj_pt(-w_top, z1)]
        xs = [p[0] for p in pts]
        g = cairo.LinearGradient(min(xs), pts[0][1], max(xs), pts[0][1])
        g.add_color_stop_rgba(0.0, *fog_mix(c_left, dist, fog_rgb), 0.99)
        g.add_color_stop_rgba(1.0, *fog_mix(c_right, dist, fog_rgb), 0.99)
        ctx.set_source(g)
        fill_poly(ctx, pts)
        # Clean edge-by-edge hand-inked architectural outline (ZERO looping corners!)
        stroke_arch_edges(ctx, pts, base_w=1.45, z_depth=dist,
                          rgba=(*fog_mix([0.022, 0.010, 0.012], dist, fog_rgb), 0.78), rng=rng)

    # Ground contact shadow beneath stone plinth
    b_l, b_r = proj_pt(-0.34, -0.01), proj_pt(0.34, -0.01)
    ctx.save()
    ctx.translate(0.5 * (b_l[0] + b_r[0]), b_l[1] + 2.0)
    ctx.scale(max(4.0, 0.56 * abs(b_r[0] - b_l[0])), max(2.0, 0.14 * abs(b_r[0] - b_l[0])))
    ctx.set_source_rgba(0.015, 0.007, 0.008, 0.65)
    ctx.arc(0, 0, 1.0, 0, math.tau)
    ctx.fill()
    ctx.restore()

    # Stepped mossy basalt plinth, shaft & lotus pedestal
    draw_tier(0.32, 0.29, 0.00, 0.18, [0.052, 0.036, 0.036], [0.22, 0.11, 0.072])
    draw_tier(0.24, 0.19, 0.18, 0.34, [0.062, 0.048, 0.038], [0.26, 0.13, 0.078])
    draw_tier(0.145, 0.135, 0.34, 0.94, [0.065, 0.045, 0.042], [0.28, 0.14, 0.082])
    draw_tier(0.15, 0.28, 0.94, 1.08, [0.13, 0.070, 0.048], [0.58, 0.27, 0.115])
    draw_tier(0.28, 0.26, 1.08, 1.15, [0.20, 0.10, 0.062], [0.78, 0.38, 0.15])

    # Luminous carved windowed firebox (hibukuro: z = 1.15..1.48)
    fb_pts = [proj_pt(-0.21, 1.15), proj_pt(0.21, 1.15), proj_pt(0.21, 1.48), proj_pt(-0.21, 1.48)]
    g_fb = cairo.RadialGradient(float(uf), float(vf), px_m * 0.02, float(uf), float(vf), px_m * 0.24)
    g_fb.add_color_stop_rgba(0.0, 1.0, 0.98, 0.84, 1.0)
    g_fb.add_color_stop_rgba(0.50, 1.0, 0.72, 0.26, 1.0)
    g_fb.add_color_stop_rgba(1.0, 0.92, 0.34, 0.10, 1.0)
    ctx.set_source(g_fb)
    fill_poly(ctx, fb_pts)

    mull_c = fog_mix([0.048, 0.020, 0.022], dist, fog_rgb)
    for mx_off, mw in ((-0.20, 0.042), (-0.07, 0.022), (0.07, 0.022), (0.20, 0.042)):
        mp = [proj_pt(mx_off - mw, 1.15), proj_pt(mx_off + mw, 1.15), proj_pt(mx_off + mw, 1.48), proj_pt(mx_off - mw, 1.48)]
        ctx.set_source_rgba(*mull_c, 0.96)
        fill_poly(ctx, mp)
    hb = [proj_pt(-0.21, 1.30), proj_pt(0.21, 1.30), proj_pt(0.21, 1.335), proj_pt(-0.21, 1.335)]
    ctx.set_source_rgba(*mull_c, 0.92)
    fill_poly(ctx, hb)

    # Curved overhanging stone umbrella roof (kasa) with upturned eaves
    kasa_pts = [
        proj_pt(-0.46, 1.50),
        proj_pt(-0.38, 1.46),
        proj_pt(0.38, 1.46),
        proj_pt(0.46, 1.50),
        proj_pt(0.42, 1.57),
        proj_pt(0.22, 1.66),
        proj_pt(0.00, 1.76),
        proj_pt(-0.22, 1.66),
        proj_pt(-0.42, 1.57),
    ]
    xs_k = [p[0] for p in kasa_pts]
    g_k = cairo.LinearGradient(min(xs_k), kasa_pts[6][1], max(xs_k), kasa_pts[0][1])
    g_k.add_color_stop_rgba(0.0, *fog_mix([0.062, 0.048, 0.042], dist, fog_rgb), 0.99)
    g_k.add_color_stop_rgba(0.55, *fog_mix([0.15, 0.082, 0.058], dist, fog_rgb), 0.99)
    g_k.add_color_stop_rgba(1.0, *fog_mix([0.56, 0.25, 0.115], dist, fog_rgb), 0.99)
    ctx.set_source(g_k)
    fill_poly(ctx, kasa_pts)

    # Warm orange fire glow illuminating the carved stone underside of kasa
    under_k = [proj_pt(-0.34, 1.46), proj_pt(0.34, 1.46), proj_pt(0.25, 1.50), proj_pt(-0.25, 1.50)]
    ctx.set_source_rgba(*fog_mix([0.96, 0.48, 0.16], dist, fog_rgb), 0.88)
    fill_poly(ctx, under_k)

    # Velvety olive-moss cap cushions resting organically along upper roof slopes
    moss_l = [proj_pt(-0.36, 1.57), proj_pt(-0.14, 1.68), proj_pt(-0.04, 1.72), proj_pt(-0.22, 1.61)]
    ctx.set_source_rgba(*fog_mix([0.095, 0.135, 0.052], dist, fog_rgb), 0.86)
    fill_poly(ctx, moss_l)

    # Lotus jewel finial (hoju)
    hoju = [proj_pt(-0.08, 1.74), proj_pt(0.08, 1.74), proj_pt(0.05, 1.84), proj_pt(0.0, 1.94), proj_pt(-0.05, 1.84)]
    ctx.set_source_rgba(*fog_mix([0.16, 0.082, 0.060], dist, fog_rgb), 0.98)
    fill_poly(ctx, hoju)
    stroke_arch_edges(ctx, hoju, base_w=1.2, z_depth=dist, rgba=(*fog_mix([0.025, 0.012, 0.014], dist, fog_rgb), 0.72), rng=rng)
    stroke_arch_edges(ctx, kasa_pts, base_w=1.4, z_depth=dist, rgba=(*fog_mix([0.022, 0.010, 0.012], dist, fog_rgb), 0.76), rng=rng)
    # Warm curved sunset rim highlight along right roof slope
    stroke_catmull_rom(ctx, kasa_pts[3:7], base_w=2.0, z_depth=dist,
                       rgba=(*fog_mix([0.98, 0.56, 0.24], dist, fog_rgb), 0.74), rng=rng, taper=True, tremor=0.14)


def draw_left_shrine_eaves_and_pillars(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
) -> None:
    """
    Monumental dark-lacquer Japanese mountain shrine eaves overhanging dramatically
    across the upper-left, heavy vermilion-lacquered cedar pillars seated solidly on
    basalt plinths behind the left curb, exposed taruki rafters with bronze end-caps,
    carved bracket complexes, and a glowing hanging bronze temple lantern (tsuri-doro)!
    """
    # 0. Continuous dark-lacquer wooden veranda floor sill resting atop the left basalt curb (grounding every pillar!)
    sill_q = np.array([
        [-2.42, SHRINE_Y0, float(stair_z(SHRINE_Y0)) + 0.68],
        [-1.60, SHRINE_Y0, float(stair_z(SHRINE_Y0)) + 0.68],
        [-1.60, SHRINE_Y1, float(stair_z(SHRINE_Y1)) + 0.68],
        [-2.42, SHRINE_Y1, float(stair_z(SHRINE_Y1)) + 0.68],
    ])
    ctx.set_source_rgba(0.055, 0.022, 0.022, 0.99)
    fill_poly(ctx, project_quad_3d(cam, sill_q))
    sill_edge = np.array([
        [-1.60, SHRINE_Y0, float(stair_z(SHRINE_Y0)) + 0.58],
        [-1.60, SHRINE_Y1, float(stair_z(SHRINE_Y1)) + 0.58],
        [-1.60, SHRINE_Y1, float(stair_z(SHRINE_Y1)) + 0.72],
        [-1.60, SHRINE_Y0, float(stair_z(SHRINE_Y0)) + 0.72],
    ])
    ctx.set_source_rgba(0.040, 0.016, 0.018, 0.99)
    fill_poly(ctx, project_quad_3d(cam, sill_edge))

    # 1. Massive vermilion-lacquered cedar shrine pillars seated solidly on top of the left veranda terrace (x = -1.96)
    for py in (18.6, 14.4, 10.2, 6.2):
        zb = float(stair_z(py)) + 0.72  # seated flush ON TOP of the stone curb & timber terrace sill!
        zt = zb + SHRINE_EAVE_H_ABOVE_STAIR + 0.34
        dist = float(np.linalg.norm(np.array([-1.96, py, zb + 1.5]) - cam.eye))
        rad = 0.21
        rvec = cam.right_xy * rad

        # Square dark-basalt foundation plinth seated cleanly on top of veranda sill
        q_pl = np.array([
            [-1.96 - rad * 1.24, py, zb - 0.02],
            [-1.96 + rad * 1.24, py, zb - 0.02],
            [-1.96 + rad * 1.12, py, zb + 0.24],
            [-1.96 - rad * 1.12, py, zb + 0.24],
        ])
        ctx.set_source_rgba(*fog_mix([0.048, 0.024, 0.024], dist, fog_rgb), 0.99)
        fill_poly(ctx, project_quad_3d(cam, q_pl))

        # Round Kyoto vermilion cedar pillar using cam.right_xy for exact cylindrical silhouette!
        p_b = np.array([-1.96, py, zb + 0.22])
        p_t = np.array([-1.96, py, zt])
        pp = project_quad_3d(cam, np.array([p_b - rvec, p_b + rvec, p_t + rvec, p_t - rvec]))
        if pp is not None:
            xs = [p[0] for p in pp]
            g_p = cairo.LinearGradient(min(xs), pp[0][1], max(xs), pp[0][1])
            g_p.add_color_stop_rgba(0.0, *fog_mix([0.14, 0.028, 0.028], dist, fog_rgb), 1.0)
            g_p.add_color_stop_rgba(0.55, *fog_mix([0.54, 0.120, 0.080], dist, fog_rgb), 1.0)
            g_p.add_color_stop_rgba(0.86, *fog_mix([0.86, 0.27, 0.135], dist, fog_rgb), 1.0)
            g_p.add_color_stop_rgba(1.0, *fog_mix([1.02, 0.56, 0.24], dist, fog_rgb), 1.0)
            ctx.set_source(g_p)
            fill_poly(ctx, pp)

            # Black-lacquer + gold copper ornamental collar near pillar capital
            p_c1 = np.array([-1.96, py, zt - 0.72])
            p_c2 = np.array([-1.96, py, zt - 0.54])
            ctx.set_source_rgba(*fog_mix([0.038, 0.015, 0.018], dist, fog_rgb), 0.98)
            fill_poly(ctx, project_quad_3d(cam, np.array([p_c1 - rvec * 1.04, p_c1 + rvec * 1.04, p_c2 + rvec * 1.04, p_c2 - rvec * 1.04])))

            rseg = cam.project_polyline(np.array([p_b + rvec, p_t + rvec - np.array([0, 0, 0.2])]))
            if len(rseg) >= 2:
                stroke_catmull_rom(ctx, [(p[0], p[1]) for p in rseg], base_w=2.4, z_depth=dist,
                                   rgba=(*fog_mix([1.02, 0.60, 0.28], dist, fog_rgb), 0.80), rng=rng, taper=False, tremor=0.14)

    # 2. Heavy dark-lacquer + vermilion bracket tie-beam (koryo) linking the shrine pillars
    z_e0 = float(stair_z(SHRINE_Y0)) + SHRINE_EAVE_H_ABOVE_STAIR
    z_e1 = float(stair_z(SHRINE_Y1)) + SHRINE_EAVE_H_ABOVE_STAIR
    z_in0 = z_e0 + (SHRINE_X_EAVE - SHRINE_X_WALL) * SHRINE_ROOF_PITCH
    z_in1 = z_e1 + (SHRINE_X_EAVE - SHRINE_X_WALL) * SHRINE_ROOF_PITCH

    beam_pts = np.array([
        [-1.98, SHRINE_Y0, z_e0 + 0.34],
        [-1.98, SHRINE_Y1, z_e1 + 0.34],
        [-1.98, SHRINE_Y1, z_e1 + 0.64],
        [-1.98, SHRINE_Y0, z_e0 + 0.64],
    ])
    ctx.set_source_rgba(0.42, 0.095, 0.068, 0.99)
    fill_poly(ctx, project_quad_3d(cam, beam_pts))

    # 3. Deep sweeping overhanging dark-lacquer wooden shrine eave soffit!
    soffit = np.array([
        [SHRINE_X_WALL - 0.25, SHRINE_Y0, z_in0],
        [SHRINE_X_EAVE, SHRINE_Y0, z_e0],
        [SHRINE_X_EAVE, SHRINE_Y1, z_e1],
        [SHRINE_X_WALL - 0.25, SHRINE_Y1, z_in1],
    ])
    sp = project_quad_3d(cam, soffit)
    if sp is not None:
        g_s = cairo.LinearGradient(sp[0][0], sp[0][1], sp[2][0], sp[2][1])
        g_s.add_color_stop_rgba(0.0, 0.035, 0.012, 0.015, 0.99)
        g_s.add_color_stop_rgba(0.65, 0.078, 0.025, 0.024, 0.99)
        g_s.add_color_stop_rgba(1.0, *fog_mix([0.13, 0.042, 0.038], 18.0, fog_rgb), 0.99)
        ctx.set_source(g_s)
        fill_poly(ctx, sp)

    # Exposed rhythmic parallel shrine roof rafters (taruki) + warm gilded bronze end-caps
    for ry in np.arange(SHRINE_Y0 + 0.4, SHRINE_Y1 - 0.2, 0.54):
        zr_e = float(stair_z(ry)) + SHRINE_EAVE_H_ABOVE_STAIR
        zr_in = zr_e + (SHRINE_X_EAVE - SHRINE_X_WALL) * SHRINE_ROOF_PITCH
        dist = float(np.linalg.norm(np.array([SHRINE_X_EAVE, ry, zr_e]) - cam.eye))
        seg = cam.project_polyline(np.array([
            [SHRINE_X_WALL - 0.15, ry, zr_in - 0.03],
            [SHRINE_X_EAVE + 0.04, ry, zr_e - 0.04],
        ]))
        if len(seg) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in seg], base_w=4.6, z_depth=dist,
                               rgba=(*fog_mix([0.022, 0.008, 0.010], dist, fog_rgb), 0.96), rng=rng, taper=False, tremor=0.12)
        ue, ve, ze = cam.project(np.array([SHRINE_X_EAVE + 0.03, ry, zr_e - 0.04]))
        if ze > 1.2 and 0 <= ue <= W and 0 <= ve <= H:
            rc = max(1.6, float(cam.K[0, 0] * 0.045 / ze))
            ctx.set_source_rgba(*fog_mix([0.92, 0.52, 0.20], dist, fog_rgb), 0.90)
            ctx.rectangle(float(ue) - rc * 0.6, float(ve) - rc * 0.6, rc * 1.2, rc * 1.2)
            ctx.fill()

    # Thick dark-lacquer fascia board & tiled eave edge (kawara)
    fascia = np.array([
        [SHRINE_X_EAVE, SHRINE_Y0, z_e0 - 0.04],
        [SHRINE_X_EAVE, SHRINE_Y1, z_e1 - 0.04],
        [SHRINE_X_EAVE - 0.04, SHRINE_Y1, z_e1 + 0.30],
        [SHRINE_X_EAVE - 0.04, SHRINE_Y0, z_e0 + 0.30],
    ])
    fp = project_quad_3d(cam, fascia)
    if fp is not None:
        g_f = cairo.LinearGradient(fp[0][0], fp[0][1], fp[1][0], fp[1][1])
        g_f.add_color_stop_rgba(0.0, 0.045, 0.016, 0.020, 1.0)
        g_f.add_color_stop_rgba(0.65, 0.14, 0.048, 0.042, 1.0)
        g_f.add_color_stop_rgba(1.0, *fog_mix([0.34, 0.13, 0.082], 18.0, fog_rgb), 1.0)
        ctx.set_source(g_f)
        fill_poly(ctx, fp)

    # Dark volcanic-slate tiled shrine roof slope sweeping grandly off top-left
    roof_top = np.array([
        [SHRINE_X_EAVE - 0.04, SHRINE_Y0, z_e0 + 0.28],
        [SHRINE_X_EAVE - 0.04, SHRINE_Y1, z_e1 + 0.28],
        [SHRINE_X_WALL - 2.6, SHRINE_Y1, z_in1 + 1.65],
        [SHRINE_X_WALL - 2.6, SHRINE_Y0, z_in0 + 1.65],
    ])
    ctx.set_source_rgba(0.030, 0.012, 0.018, 0.99)
    fill_poly(ctx, project_quad_3d(cam, roof_top))

    eseg = cam.project_polyline(np.array([
        [SHRINE_X_EAVE, SHRINE_Y0 + 0.4, z_e0 + 0.14],
        [SHRINE_X_EAVE, SHRINE_Y1, z_e1 + 0.14],
    ]))
    if len(eseg) >= 2:
        stroke_catmull_rom(ctx, [(p[0], p[1]) for p in eseg], base_w=2.8, z_depth=6.0,
                           rgba=(0.98, 0.54, 0.24, 0.78), rng=rng, taper=False, tremor=0.14)

    # 4. Graceful Hanging Bronze Shrine Lantern (tsuri-doro) suspended beneath the overhanging eave!
    hl_x, hl_y = SHRINE_X_EAVE - 0.48, 7.60
    z_hl_eave = float(stair_z(hl_y)) + SHRINE_EAVE_H_ABOVE_STAIR + 0.12
    z_hl_body = z_hl_eave - 0.62
    dist_hl = float(np.linalg.norm(np.array([hl_x, hl_y, z_hl_body]) - cam.eye))
    uh, vh, zh = cam.project(np.array([hl_x, hl_y, z_hl_body]))
    if zh > 1.5:
        px_h = float(cam.K[0, 0] / zh)
        # Suspension iron chain rod
        cseg = cam.project_polyline(np.array([[hl_x, hl_y, z_hl_eave], [hl_x, hl_y, z_hl_body + 0.24]]))
        if len(cseg) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in cseg], base_w=2.6, z_depth=dist_hl,
                               rgba=(0.04, 0.018, 0.018, 0.95), rng=rng, taper=False, tremor=0.10)
        # Warm glowing radial halo around hanging temple lantern
        g_hl = cairo.RadialGradient(float(uh), float(vh), px_h * 0.04, float(uh), float(vh), px_h * 0.95)
        g_hl.add_color_stop_rgba(0.0, 1.0, 0.88, 0.50, 0.68)
        g_hl.add_color_stop_rgba(0.35, 0.98, 0.48, 0.16, 0.26)
        g_hl.add_color_stop_rgba(1.0, 0.86, 0.22, 0.08, 0.0)
        ctx.set_source(g_hl)
        ctx.arc(float(uh), float(vh), px_h * 0.95, 0, math.tau)
        ctx.fill()
        # Hexagonal bronze hood + glowing amber core + pointed bronze finial below
        rw, rh = px_h * 0.19, px_h * 0.16
        # Glowing chamber
        ctx.set_source_rgba(1.0, 0.82, 0.36, 0.96)
        ctx.rectangle(float(uh) - rw * 0.52, float(vh) - rh * 0.52, rw * 1.04, rh * 1.04)
        ctx.fill()
        # Dark bronze hood roof
        ctx.set_source_rgba(0.055, 0.022, 0.022, 0.98)
        ctx.move_to(float(uh) - rw * 1.15, float(vh) - rh * 0.45)
        ctx.line_to(float(uh), float(vh) - rh * 1.25)
        ctx.line_to(float(uh) + rw * 1.15, float(vh) - rh * 0.45)
        ctx.close_path()
        ctx.fill()
        # Bronze base dish & lattice vertical bars
        ctx.rectangle(float(uh) - rw * 0.72, float(vh) + rh * 0.48, rw * 1.44, rh * 0.24)
        ctx.fill()
        ctx.set_line_width(max(1.2, rw * 0.12))
        for dx_b in (-0.52, 0.0, 0.52):
            ctx.move_to(float(uh) + rw * dx_b, float(vh) - rh * 0.52)
            ctx.line_to(float(uh) + rw * dx_b, float(vh) + rh * 0.52)
            ctx.stroke()


def draw_momiji_leaf(
    ctx: cairo.Context,
    cx: float,
    cy: float,
    size: float,
    angle: float,
    rgba: tuple[float, float, float, float],
    rng: np.random.Generator,
    backlit_rim: tuple[float, float, float, float] | None = None,
) -> None:
    """5-pointed palmate Japanese autumn maple leaf (momiji) via seeded cubic Beziers."""
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(angle)
    ctx.set_source_rgba(*rgba)

    lobe_specs = [
        (-1.82, 0.50),
        (-0.86, 0.82),
        (0.00, 1.00),
        (0.86, 0.82),
        (1.82, 0.50),
    ]
    ctx.move_to(0.0, size * 0.12)
    for la, lscale in lobe_specs:
        r_tip = size * lscale * float(rng.uniform(0.88, 1.12))
        a_j = la + float(rng.uniform(-0.08, 0.08))
        w_l = 0.21 * r_tip
        dx, dy = math.sin(a_j), -math.cos(a_j)
        px, py = math.cos(a_j), math.sin(a_j)
        ctx.curve_to(
            dx * r_tip * 0.35 - px * w_l * 1.15, dy * r_tip * 0.35 - py * w_l * 1.15,
            dx * r_tip * 0.76 - px * w_l * 0.55, dy * r_tip * 0.76 - py * w_l * 0.55,
            dx * r_tip, dy * r_tip,
        )
        ctx.curve_to(
            dx * r_tip * 0.76 + px * w_l * 0.55, dy * r_tip * 0.76 + py * w_l * 0.55,
            dx * r_tip * 0.35 + px * w_l * 1.10, dy * r_tip * 0.35 + py * w_l * 1.10,
            dx * size * 0.16 + px * size * 0.10, dy * size * 0.16 + py * size * 0.10,
        )
    ctx.close_path()
    ctx.fill()

    if backlit_rim is not None:
        ctx.set_source_rgba(*backlit_rim)
        ctx.set_line_width(max(0.7, size * 0.045))
        ctx.move_to(0.0, size * 0.12)
        ctx.line_to(0.0, -size * 0.75)
        ctx.stroke()
    ctx.restore()


def draw_midground_maple_branches_and_leaves(
    ctx: cairo.Context,
    rng: np.random.Generator,
    drifting_only: bool = False,
) -> None:
    if not drifting_only:
        branches = [
            [(W + 25, 595), (W - 140, 565), (W - 295, 588), (W - 430, 548)],
            [(W - 180, 574), (W - 265, 515), (W - 355, 488)],
            [(W - 285, 582), (W - 375, 622), (W - 465, 635)],
        ]
        for bpts in branches:
            stroke_catmull_rom(ctx, bpts, base_w=7.5, z_depth=9.0, rgba=(0.048, 0.018, 0.020, 0.94), rng=rng, taper=True, tremor=0.32)
        cluster_centers = [
            (W - 125, 548, 34), (W - 215, 558, 30), (W - 310, 520, 28),
            (W - 375, 492, 24), (W - 420, 552, 25), (W - 385, 615, 26),
            (W - 460, 628, 22), (W - 260, 596, 28),
        ]
        for ccx, ccy, bsz in cluster_centers:
            for _ in range(int(rng.integers(5, 9))):
                lx = ccx + float(rng.normal(0, bsz * 0.85))
                ly = ccy + float(rng.normal(0, bsz * 0.65))
                sz = bsz * float(rng.uniform(0.55, 1.05))
                ang = float(rng.uniform(0, math.tau))
                lit = rng.random() > 0.42
                col = (0.86, 0.24, 0.11, 0.92) if lit else (0.26, 0.050, 0.044, 0.94)
                rim = (1.0, 0.64, 0.24, 0.65) if lit else None
                draw_momiji_leaf(ctx, lx, ly, sz, ang, col, rng, backlit_rim=rim)
        return

    # Drifting autumn maple leaves floating gracefully through the twilight breeze!
    drifters = [
        (890, 290, 15.0), (1055, 360, 18.0), (1260, 275, 13.5), (1440, 395, 16.5),
        (1170, 515, 21.0), (960, 560, 19.0), (1385, 575, 17.5), (1540, 680, 22.0),
        (815, 720, 24.0), (1120, 795, 25.0), (1325, 865, 20.0), (675, 875, 28.0),
        (1015, 960, 26.0), (1490, 935, 23.0), (760, 445, 14.5), (1620, 485, 15.5),
        (1245, 650, 16.0), (910, 845, 21.0),
    ]
    for lx, ly, sz in drifters:
        ang = float(rng.uniform(-math.pi, math.pi))
        pick = rng.random()
        if pick > 0.55:
            col = (0.94, 0.36, 0.14, 0.92)
            rim = (1.0, 0.74, 0.34, 0.70)
        elif pick > 0.22:
            col = (0.64, 0.13, 0.085, 0.92)
            rim = (0.98, 0.44, 0.18, 0.55)
        else:
            col = (0.20, 0.042, 0.040, 0.92)
            rim = None
        draw_momiji_leaf(ctx, lx, ly, sz, ang, col, rng, backlit_rim=rim)

    for _ in range(90):
        mx = float(rng.uniform(220, W - 80))
        my = float(rng.uniform(180, H - 60))
        mr = float(rng.uniform(0.8, 2.2))
        ctx.set_source_rgba(1.0, 0.74, 0.34, float(rng.uniform(0.14, 0.42)))
        ctx.arc(mx, my, mr, 0, math.tau)
        ctx.fill()


# =============================================================================
# Stage 4 — Foreground Circular Disc Bokeh Autumn Maple Bough & 35mm Analog Optics
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
    nz = np.any(img[:, :, 3] > 1e-4, axis=1)
    nx = np.any(img[:, :, 3] > 1e-4, axis=0)
    if not np.any(nz) or not np.any(nx):
        return np.zeros_like(img, dtype=np.float32)
    kh, kw = kernel.shape
    pad = max(kh, kw) + 8
    y0 = max(0, int(np.argmax(nz)) - pad)
    y1 = min(img.shape[0], int(img.shape[0] - np.argmax(nz[::-1])) + pad)
    x0 = max(0, int(np.argmax(nx)) - pad)
    x1 = min(img.shape[1], int(img.shape[1] - np.argmax(nx[::-1])) + pad)

    sub = img[y0:y1, x0:x1]
    ih, iw = sub.shape[:2]
    fh = 1 << int(math.ceil(math.log2(ih + kh - 1)))
    fw = 1 << int(math.ceil(math.log2(iw + kw - 1)))
    k_pad = np.zeros((fh, fw), dtype=np.float32)
    k_pad[:kh, :kw] = kernel
    k_pad = np.roll(np.roll(k_pad, -(kh // 2), axis=0), -(kw // 2), axis=1)
    K = np.fft.rfft2(k_pad)

    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        p = np.zeros((fh, fw), dtype=np.float32)
        p[:ih, :iw] = sub[:, :, c]
        out[y0:y1, x0:x1, c] = np.fft.irfft2(np.fft.rfft2(p) * K, s=(fh, fw))[:ih, :iw]
    return out


def paint_foreground_bokeh_maple_bough(rng: np.random.Generator) -> np.ndarray:
    surf, ctx = new_cairo_2x()

    main_bough = [
        (W + 30, 62),
        (W - 145, 115),
        (W - 340, 148),
        (W - 515, 118),
        (W - 635, 138),
    ]
    twig_1 = [(W - 220, 128), (W - 315, 225), (W - 410, 275)]
    twig_2 = [(W - 125, 108), (W - 185, 245), (W - 235, 338)]
    twig_3 = [(W - 365, 142), (W - 455, 62), (W - 540, 28)]
    twig_4 = [(W - 95, 92), (W - 55, 235), (W - 88, 355)]

    stroke_catmull_rom(ctx, main_bough, base_w=17.0, z_depth=0.4, rgba=(0.022, 0.008, 0.010, 0.98), rng=rng, taper=True, tremor=0.45)
    for tw in (twig_1, twig_2, twig_3, twig_4):
        stroke_catmull_rom(ctx, tw, base_w=9.5, z_depth=0.4, rgba=(0.025, 0.010, 0.012, 0.96), rng=rng, taper=True, tremor=0.40)

    leaf_clusters = [
        (W - 620, 145, 58, 5),
        (W - 520, 115, 66, 6),
        (W - 495, 42, 62, 5),
        (W - 415, 265, 56, 6),
        (W - 335, 175, 72, 7),
        (W - 235, 325, 58, 6),
        (W - 195, 215, 74, 7),
        (W - 115, 118, 78, 8),
        (W - 78, 325, 64, 6),
        (W - 45, 210, 72, 6),
    ]
    for cx, cy, base_sz, cnt in leaf_clusters:
        for _ in range(cnt):
            lx = cx + float(rng.normal(0, base_sz * 0.62))
            ly = cy + float(rng.normal(0, base_sz * 0.55))
            sz = base_sz * float(rng.uniform(0.65, 1.15))
            ang = float(rng.uniform(-math.pi, math.pi))
            if rng.random() > 0.50:
                col = (0.36, 0.058, 0.036, 0.92)
                rim = (0.90, 0.30, 0.11, 0.60)
            else:
                col = (0.044, 0.013, 0.016, 0.97)
                rim = (0.52, 0.14, 0.068, 0.35)
            draw_momiji_leaf(ctx, lx, ly, sz, ang, col, rng, backlit_rim=rim)

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
    tint = np.array([1.15, 0.92, 0.74], dtype=np.float32)
    hal = (
        0.26 * gaussian_blur_2d(hl_src, 6.5)
        + 0.19 * gaussian_blur_2d(hl_src, 24.0)
        + 0.12 * gaussian_blur_2d(hl_src, 72.0)
    ) * tint[None, None, :]
    img = 1.0 - (1.0 - np.clip(img, 0.0, 1.0)) * (1.0 - np.clip(hal, 0.0, 0.85))

    # 2. Radial corner chromatic aberration + gentle optical vignette
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (xx - W * 0.5) / (W * 0.5)
    ny = (yy - H * 0.5) / (H * 0.5)
    r2 = nx * nx + ny * ny
    ca_scale = 0.0011
    img[:, :, 0] = remap_bilinear(img[:, :, 0], xx + nx * r2 * (W * ca_scale), yy + ny * r2 * (H * ca_scale))
    img[:, :, 2] = remap_bilinear(img[:, :, 2], xx - nx * r2 * (W * ca_scale), yy - ny * r2 * (H * ca_scale))
    img *= np.clip(1.0 - 0.22 * (r2 ** 1.15), 0.70, 1.0)[..., None]

    # 3. Linear shadow black lift + C1-continuous tanh highlight shoulder (knee=0.80, ceiling=0.965)
    img = 0.011 + img * 0.989
    knee, ceiling = 0.80, 0.965
    span = ceiling - knee
    img = np.where(img > knee, knee + span * np.tanh((img - knee) / span), img)

    # 4. Luminance-weighted silver-halide grain
    lum2 = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    env = np.clip(1.12 - 0.95 * np.power(np.clip(lum2, 0.0, 1.0), 1.55), 0.12, 1.12)
    grain = rng.normal(0.0, 0.0105, size=(H, W, 3)).astype(np.float32)
    mono_g = rng.normal(0.0, 0.0085, size=(H, W, 1)).astype(np.float32)
    img = np.clip(img + (0.65 * mono_g + 0.35 * grain) * env[..., None], 0.0, 1.0)

    # 5. Faint analog transfer patina: horizontal YIQ chroma softening (1.0 px) + reflect101
    #    + slight luma ring + sub-pixel per-line horizontal tape jitter + <=0.8% scanlines
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    I = 0.596 * R - 0.274 * G - 0.322 * B
    Q = 0.211 * R - 0.523 * G + 0.312 * B

    I_soft = gaussian_blur_1d(I, 1.0, axis=1)
    Q_soft = gaussian_blur_1d(Q, 1.0, axis=1)
    Y_ring = Y + 0.04 * (Y - gaussian_blur_1d(Y, 1.1, axis=1))

    out = np.empty_like(img)
    out[:, :, 0] = Y_ring + 0.956 * I_soft + 0.621 * Q_soft
    out[:, :, 1] = Y_ring - 0.272 * I_soft - 0.647 * Q_soft
    out[:, :, 2] = Y_ring - 1.106 * I_soft + 1.703 * Q_soft

    line_shift = gaussian_blur_1d(rng.normal(0.0, 0.17, size=(H, 1)).astype(np.float32), 1.2, axis=0)
    jx = np.broadcast_to(np.arange(W, dtype=np.float32)[None, :], (H, W)) + line_shift
    jy = np.broadcast_to(np.arange(H, dtype=np.float32)[:, None], (H, W))
    out = remap_bilinear(out, jx, jy)
    out *= (1.0 - 0.007 * (0.5 - 0.5 * np.cos(np.arange(H, dtype=np.float32) * math.pi)))[:, None, None]
    return np.clip(out, 0.0, 1.0)


# =============================================================================
# Full Scene Assembly & Deterministic Export
# =============================================================================

def render_scene(
    seed: int = 199711,
    audit_dir: pathlib.Path | None = None,
    webp_targets: list[pathlib.Path] | None = None,
) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # Stage 1: Off-axis pinhole Camera3D ascending the rain-slicked shrine stairway (VP ~59.3% width)
    cam = Camera3D(W, H, eye=(0.82, 0.0, 1.56), target=(-0.62, 14.0, 2.74), fov_x_deg=58.0)
    print(f"[ember-02] Camera VP: ({cam.vp_x:.1f}, {cam.vp_y:.1f}) -> {cam.vp_x / W * 100:.1f}% width, horizon {cam.horizon_y / H * 100:.1f}% height")

    # Stage 1 & 2: Dramatic twilight mountain valley sky, painterly sunset clouds & cedar ridges
    canvas, fog_rgb = paint_twilight_valley_backdrop(cam, rng)

    # Stage 1 & 2: Exact 3x3 Plane-Homography Stepped Voronoi Basalt Stairway + Parapets + Shrine Wall
    tread_atlas = paint_stairway_tread_atlas(rng)
    riser_atlas = paint_stairway_riser_atlas(rng)
    y_end = STAIR_Y0 + STAIR_N_STEPS * STEP_TREAD
    z_end = float(stair_z(y_end))

    shrine_wall_uv = paint_shrine_facade_wall_uv(rng)
    canvas = cam.warp_plane_over(
        canvas, shrine_wall_uv,
        p0=np.array([SHRINE_X_WALL, SHRINE_Y0, float(stair_z(SHRINE_Y0)) + 0.55]),
        e1=np.array([0.0, SHRINE_Y1 - SHRINE_Y0, float(stair_z(SHRINE_Y1) - stair_z(SHRINE_Y0))]),
        e2=np.array([0.0, 0.0, 2.85]),
        fog_rgb=fog_rgb, fog_start=6.5, fog_density=0.044,
    )

    left_curb_uv = paint_parapet_wall_uv(rng, "left_wall")
    canvas = cam.warp_plane_over(
        canvas, left_curb_uv,
        p0=np.array([-STAIR_HALF_W, STAIR_Y0, float(stair_z(STAIR_Y0))]),
        e1=np.array([0.0, y_end - STAIR_Y0, z_end - float(stair_z(STAIR_Y0))]),
        e2=np.array([0.0, 0.0, 0.72]),
        fog_rgb=fog_rgb, fog_start=6.5, fog_density=0.044,
    )

    right_in_uv = paint_parapet_wall_uv(rng, "right_inner")
    right_top_uv = paint_parapet_wall_uv(rng, "right_top")
    canvas = cam.warp_plane_over(
        canvas, right_in_uv,
        p0=np.array([STAIR_HALF_W, STAIR_Y0, float(stair_z(STAIR_Y0))]),
        e1=np.array([0.0, y_end - STAIR_Y0, z_end - float(stair_z(STAIR_Y0))]),
        e2=np.array([0.0, 0.0, 0.62]),
        fog_rgb=fog_rgb, fog_start=6.5, fog_density=0.044,
    )
    canvas = cam.warp_plane_over(
        canvas, right_top_uv,
        p0=np.array([STAIR_HALF_W, STAIR_Y0, float(stair_z(STAIR_Y0)) + 0.62]),
        e1=np.array([0.0, y_end - STAIR_Y0, z_end - float(stair_z(STAIR_Y0))]),
        e2=np.array([0.42, 0.0, 0.0]),
        fog_rgb=fog_rgb, fog_start=6.5, fog_density=0.044,
    )

    th_t = tread_atlas.shape[0]
    th_r = riser_atlas.shape[0]
    for step_i in range(STAIR_N_STEPS - 1, -1, -1):
        ys0 = STAIR_Y0 + step_i * STEP_TREAD
        zs_bot = 0.36 + step_i * STEP_RISE
        zs_top = zs_bot + STEP_RISE

        rt0 = int(round((step_i / STAIR_N_STEPS) * th_t))
        rt1 = max(rt0 + 2, int(round(((step_i + 1) / STAIR_N_STEPS) * th_t)))
        rr0 = int(round((step_i / STAIR_N_STEPS) * th_r))
        rr1 = max(rr0 + 2, int(round(((step_i + 1) / STAIR_N_STEPS) * th_r)))

        canvas = cam.warp_plane_over(
            canvas, tread_atlas[rt0:rt1, :, :],
            p0=np.array([-STAIR_HALF_W, ys0, zs_top]),
            e1=np.array([2.0 * STAIR_HALF_W, 0.0, 0.0]),
            e2=np.array([0.0, STEP_TREAD, 0.0]),
            fog_rgb=fog_rgb, fog_start=6.5, fog_density=0.044,
        )
        canvas = cam.warp_plane_over(
            canvas, riser_atlas[rr0:rr1, :, :],
            p0=np.array([-STAIR_HALF_W, ys0, zs_bot]),
            e1=np.array([2.0 * STAIR_HALF_W, 0.0, 0.0]),
            e2=np.array([0.0, 0.0, STEP_RISE]),
            fog_rgb=fog_rgb, fog_start=6.5, fog_density=0.044,
        )

    g_mottle = fbm_gouache(H, W, rng)
    g_dabs = flat_brush_dabs(H, W, rng, count=88)
    g_tooth = paper_tooth(H, W, rng)

    # Stage 3 Band 0 (Distant Summit Sanctuary, Vermilion Torii Gate & Far Toro Lanterns)
    s_far, c_far = new_cairo_2x()
    draw_summit_sanctuary_and_torii(c_far, cam, y_torii=21.2, rng=rng, fog_rgb=fog_rgb)
    for spec in sorted([s for s in LANTERN_SPECS if s[1] >= 17.5], key=lambda x: -x[1]):
        draw_stone_toro_lantern(c_far, cam, *spec, rng, fog_rgb)
    far_1x = finish_cairo_2x(s_far)
    fa = far_1x[:, :, 3:4]
    far_1x[:, :, :3] = np.clip(far_1x[:, :, :3] + fa * (g_mottle * 0.015 + g_dabs * 0.010)[..., None], 0.0, 1.35)
    canvas = over_premul(canvas, far_1x)

    # Volumetric crimson-amber twilight valley sunbeams aligned with projected SUN_DIR
    beams = paint_valley_sunbeams(cam, rng, intensity=0.36)
    canvas = 1.0 - (1.0 - np.clip(canvas, 0.0, 1.0)) * (1.0 - np.clip(beams, 0.0, 0.85))

    # Stage 3 Band 1 (Left Dark-Lacquer Shrine Eaves, Vermilion Pillars, Mid Toro Lanterns & Cliff Maples)
    s_mid, c_mid = new_cairo_2x()
    draw_left_shrine_eaves_and_pillars(c_mid, cam, rng, fog_rgb)
    for spec in sorted([s for s in LANTERN_SPECS if 10.5 <= s[1] < 17.5], key=lambda x: -x[1]):
        draw_stone_toro_lantern(c_mid, cam, *spec, rng, fog_rgb)
    draw_midground_maple_branches_and_leaves(c_mid, rng, drifting_only=False)
    mid_1x = finish_cairo_2x(s_mid)
    ma = mid_1x[:, :, 3:4]
    mid_1x[:, :, :3] = np.clip(mid_1x[:, :, :3] + ma * (g_mottle * 0.018 + g_dabs * 0.013 + g_tooth * 0.006)[..., None], 0.0, 1.35)
    canvas = over_premul(canvas, mid_1x)

    # Stage 3 Band 2 (Hero Near Mossy Stone Toro Lanterns & Drifting Autumn Maple Leaves)
    s_near, c_near = new_cairo_2x()
    for spec in sorted([s for s in LANTERN_SPECS if s[1] < 10.5], key=lambda x: -x[1]):
        draw_stone_toro_lantern(c_near, cam, *spec, rng, fog_rgb)
    draw_midground_maple_branches_and_leaves(c_near, rng, drifting_only=True)
    near_1x = finish_cairo_2x(s_near)
    na = near_1x[:, :, 3:4]
    near_1x[:, :, :3] = np.clip(near_1x[:, :, :3] + na * (g_mottle * 0.016 + g_dabs * 0.011 + g_tooth * 0.006)[..., None], 0.0, 1.35)
    canvas = over_premul(canvas, near_1x)

    # Stage 4: Foreground upper-right overhanging circular-bokeh autumn maple bough + 35mm analog finish
    fg_bokeh = paint_foreground_bokeh_maple_bough(rng)
    canvas = over_premul(canvas, fg_bokeh)

    final_rgb = apply_35mm_finish(canvas, rng)
    elapsed = time.perf_counter() - t0
    print(f"[ember-02] Rendered {W}x{H} in {elapsed:.2f}s")

    u8 = np.round(np.clip(final_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    full_im = Image.fromarray(u8, mode="RGB")

    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        full_im.save(audit_dir / "w02_full.png", optimize=True)
        crops = {
            "w02_crop_aperture.png": (int(W * 0.44), int(H * 0.14)),      # Summit torii, crisp sanctuary roof & twilight valley clouds
            "w02_crop_midground.png": (int(W * 0.18), int(H * 0.36)),     # Left dark-lacquer eaves, vermilion pillars & lantern steps
            "w02_crop_foreground.png": (int(W * 0.62), int(H * 0.18)),    # Hero near mossy toro lantern + foreground circular-bokeh maple bough
        }
        for cname, (cx0, cy0) in crops.items():
            x0 = max(0, min(W - 600, cx0))
            y0 = max(0, min(H - 450, cy0))
            full_im.crop((x0, y0, x0 + 600, y0 + 450)).save(audit_dir / cname)

    targets = webp_targets if webp_targets is not None else [DEFAULT_THEME_WEBP, DEFAULT_DOCS_WEBP]
    if targets:
        first = targets[0]
        first.parent.mkdir(parents=True, exist_ok=True)
        full_im.save(first, format="WEBP", quality=90, method=6)
        print(f"[ember-02] Saved {W}x{H} WebP: {first} ({first.stat().st_size / 1024:.1f} KiB)")
        for extra in targets[1:]:
            extra.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(first, extra)
            print(f"[ember-02] Copied {W}x{H} WebP: {extra} ({extra.stat().st_size / 1024:.1f} KiB)")

    return final_rgb, {"elapsed": elapsed, "vp_ratio": cam.vp_x / W}


def main() -> None:
    ap = argparse.ArgumentParser(description="Render Ember 02 (Lantern Shrine Stairway) 2048x1152 wallpaper")
    ap.add_argument("--seed", type=int, default=199711)
    ap.add_argument("--audit-dir", type=pathlib.Path, default=None)
    ap.add_argument("--no-webp", action="store_true")
    args = ap.parse_args()
    targets = [] if args.no_webp else [DEFAULT_THEME_WEBP, DEFAULT_DOCS_WEBP]
    render_scene(seed=args.seed, audit_dir=args.audit_dir, webp_targets=targets)


if __name__ == "__main__":
    main()

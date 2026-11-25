#!/usr/bin/env python3
"""
Matcha 02 — "Mossy Shrine Canal" (2048 x 1152)

Pure-Python 3D-grounded 5-stage painting pipeline (zero image models, zero external assets):
  Stage 1: Metric right-handed Camera3D (off-axis vanishing point ~60.5% width),
           exact 3x3 UV-to-screen plane homographies masked by UV bounds and
           ray-plane depth z_cam > 0.05 with exponential forest morning mist,
           3D near-plane segment clipping, and vectorized 3x3 Voronoi (d2 - d1)
           wet mossy shrine cobblestones winding beside the canal embankment.
  Stage 2: Single normalized diagonal morning SUN_DIR (SUN_DIR[2] < 0) traced
           backward with 6 height-refinement steps onto the layered cedar canopy
           and weathered dark-wood torii gates (with smooth penumbra shadows!) so
           cobblestone light pools, torii & lantern cast shadows, and volumetric
           golden-olive sunbeams align geometrically; plus emerald-jade canal
           Fresnel reflections combining capillary-rippled horizon reflections
           with 3D plane-mirrored quads.
  Stage 3: 4 depth-interleaved 2x supersampled PyCairo bands (Lanczos-downsampled)
           featuring lush mossy left-bank earth mounds rooting ancient continuous
           Japanese cedar (sugi) trunks with organic sinuous bark, creeping root
           moss, and layered feathery conifer foliage clouds, three receding
           weathered dark-wood Myojin torii gates with moss ridges & shimenawa
           rope, glowing mossy Kasuga stone lanterns with carved stone mullions,
           weathered canal pilings, and floating water lilies with cream blossoms,
           enriched by isotropic fbm_gouache, flat-brush dabs, paper tooth, and
           warm shadow bounce.
  Stage 4: Lower-left foreground forest fern fronds convolved in premultiplied
           RGBA with a circular aperture disc kernel (r = 7.8, spherical-aberration
           rim) + 1 px opposite R/B lateral shift, followed by 35mm multi-scale
           halation, radial CA, optical vignette, shadow black lift + C1 tanh
           highlight shoulder, luminance-weighted silver-halide grain, and faint
           YIQ analog transfer patina.
  Stage 5: Deterministic CPU render (<60s) exporting 2048x1152 WebP assets and
           optional 1:1 (600x450) macro audit crops.
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

# Stage 2: Single locked normalized 3D morning light vector piercing canopy upper-left -> lower-right
_raw_sun = np.array([0.46, -0.26, -0.848], dtype=np.float64)
SUN_DIR = _raw_sun / np.linalg.norm(_raw_sun)

# World elevation constants (meters)
Z_CANAL = 0.00
Z_PATH = 0.28

# Receding weathered dark-wood torii gate placements: (y, x_left, x_right, z_top)
TORII_SPECS = (
    (22.5, 0.28, 3.08, 3.38),
    (14.6, 0.16, 3.00, 3.36),
    (8.1, 0.30, 3.16, 3.34),
)


def bank_x_at_y(wy: np.ndarray | float) -> np.ndarray | float:
    """Gently winding stone canal embankment curve separating canal (left) and walkway (right)."""
    return 0.08 + 0.36 * np.sin(wy * 0.23 - 0.55) + 0.14 * np.cos(wy * 0.58 + 0.4)


def far_left_bank_x(wy: np.ndarray | float) -> np.ndarray | float:
    """Organic mossy forested left bank of the shrine canal where ancient cedar trunks root."""
    return -2.55 + 0.42 * np.sin(wy * 0.28 + 0.8) + 0.22 * np.cos(wy * 0.65 - 0.3)


def canopy_height(sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """Stepped/undulating ancient Japanese cedar canopy height surface (meters)."""
    return 6.4 + 0.85 * np.sin(sx * 0.42 + sy * 0.24) + 0.55 * np.cos(sx * 0.82 - sy * 0.48)


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
    if sigma <= 7.0:
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
        rx = float(rng.uniform(qw * 0.035, qw * 0.105))
        ry = float(rng.uniform(rx * 0.28, rx * 0.62))
        n_pts = int(rng.integers(5, 8))
        strength = float(rng.uniform(0.18, 0.42)) * (1.0 if rng.random() > 0.5 else -1.0)
        c1.save()
        c1.set_operator(cairo.OPERATOR_CLEAR)
        c1.paint()
        c1.set_operator(cairo.OPERATOR_OVER)
        c1.translate(cx, cy)
        c1.rotate(ang)
        pts = []
        for k in range(n_pts):
            a = (math.tau * k) / n_pts + float(rng.uniform(-0.22, 0.22))
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


def apply_shadow_bounce(rgb: np.ndarray, bounce_rgb=(0.018, 0.028, 0.016)) -> np.ndarray:
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    w_sh = np.clip((0.28 - lum) / 0.28, 0.0, 1.0)
    return rgb + (w_sh * w_sh)[..., None] * np.array(bounce_rgb, dtype=np.float32)


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

    def clip_segment_3d(
        self, p0: np.ndarray, p1: np.ndarray, z_near: float = Z_NEAR
    ) -> tuple[np.ndarray, np.ndarray] | None:
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

    def clip_polygon_3d(self, pts: list[np.ndarray], z_near: float = Z_NEAR) -> list[np.ndarray]:
        if not pts:
            return []
        out: list[np.ndarray] = []
        n = len(pts)
        for i in range(n):
            p0, p1 = pts[i], pts[(i + 1) % n]
            z0 = float((self.R @ (p0 - self.eye))[2])
            z1 = float((self.R @ (p1 - self.eye))[2])
            in0, in1 = z0 >= z_near, z1 >= z_near
            if in0 and in1:
                out.append(p1)
            elif in0 and not in1:
                t = (z_near - z0) / (z1 - z0)
                out.append(p0 + t * (p1 - p0))
            elif (not in0) and in1:
                t = (z_near - z0) / (z1 - z0)
                out.append(p0 + t * (p1 - p0))
                out.append(p1)
        return out

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

    def plane_homography_uv_to_screen(
        self, p0: np.ndarray, e1: np.ndarray, e2: np.ndarray, uv_w: int, uv_h: int
    ) -> np.ndarray:
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
        fog_density: float = 0.062,
    ) -> np.ndarray:
        th, tw = tex_premul_rgba.shape[:2]
        H_mat = self.plane_homography_uv_to_screen(p0, e1, e2, tw, th)
        try:
            H_inv = np.linalg.inv(H_mat)
        except np.linalg.LinAlgError:
            return dst_rgb

        corners = [p0, p0 + e1, p0 + e1 + e2, p0 + e2]
        clipped = self.clip_polygon_3d(corners)
        if len(clipped) < 3:
            return dst_rgb
        xs, ys = [], []
        for pt in clipped:
            sx, sy, _ = self.project(pt)
            xs.append(float(sx))
            ys.append(float(sy))

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
# Stage 2 — Unified 3D Forest Canopy & Smooth Penumbra Torii Shadow Solver
# =============================================================================

def trace_shrine_canopy_sunlight(wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> np.ndarray:
    """
    Trace any 3D walkway/canal point backward along -SUN_DIR using 6 fixed
    height-refinement steps onto the undulating cedar canopy, plus smooth
    analytical penumbra shadows from all three torii gates & the hero stone lantern!
    """
    inv_sz = 1.0 / (-SUN_DIR[2])
    z_can = np.full_like(wx, 6.4, dtype=np.float64)
    for _ in range(6):
        t = np.maximum(0.0, (z_can - wz) * inv_sz)
        sx = wx - SUN_DIR[0] * t
        sy = wy - SUN_DIR[1] * t
        z_can = canopy_height(sx, sy)

    t = np.maximum(0.0, (z_can - wz) * inv_sz)
    sx = wx - SUN_DIR[0] * t
    sy = wy - SUN_DIR[1] * t

    gap_field = (
        0.52 * np.sin(sy * 0.44 - sx * 0.32 + 0.6)
        + 0.34 * np.cos(sx * 0.62 + sy * 0.28 - 0.8)
        + 0.24 * np.sin(sx * 1.15 - sy * 0.85)
    )
    corridor = np.exp(-((sx + 1.1) / 3.8) ** 2)
    sun = np.clip((gap_field + 0.38) * 1.28 * corridor, 0.0, 1.0).astype(np.float32)

    # Smooth penumbra shadows of all three torii lintels & pillars (ZERO blocky stair-stepping!)
    for ty, tx_l, tx_r, tz_top in TORII_SPECS:
        t_k = np.maximum(0.0, (tz_top - wz) * inv_sz)
        kx = wx - SUN_DIR[0] * t_k
        ky = wy - SUN_DIR[1] * t_k
        kx_m = np.clip((kx - (tx_l - 0.55)) / 0.14, 0.0, 1.0) * np.clip(((tx_r + 0.55) - kx) / 0.14, 0.0, 1.0)
        ky_m = np.clip((0.28 - np.abs(ky - ty)) / 0.14, 0.0, 1.0)
        sh_k = np.where(wz < tz_top, kx_m * ky_m, 0.0)

        t_n = np.maximum(0.0, (tz_top - 0.55 - wz) * inv_sz)
        nx_p = wx - SUN_DIR[0] * t_n
        ny_p = wy - SUN_DIR[1] * t_n
        nx_m = np.clip((nx_p - (tx_l - 0.38)) / 0.12, 0.0, 1.0) * np.clip(((tx_r + 0.38) - nx_p) / 0.12, 0.0, 1.0)
        ny_m = np.clip((0.18 - np.abs(ny_p - ty)) / 0.10, 0.0, 1.0)
        sh_n = np.where(wz < tz_top - 0.55, nx_m * ny_m, 0.0)

        sun *= (1.0 - 0.72 * np.maximum(sh_k, sh_n)).astype(np.float32)

        for px_c in (tx_l, tx_r):
            t_col = (wx - px_c) / SUN_DIR[0]
            hz = wz - t_col * SUN_DIR[2]
            hy = wy - t_col * SUN_DIR[1]
            h_m = np.clip((hz - Z_PATH) / 0.10, 0.0, 1.0) * np.clip((tz_top - hz) / 0.15, 0.0, 1.0) * (t_col > 0.0)
            y_m = np.clip((0.18 - np.abs(hy - ty)) / 0.10, 0.0, 1.0)
            sun *= (1.0 - 0.68 * h_m * y_m).astype(np.float32)

    # Smooth penumbra shadow of hero stone lantern at (0.36, 6.2)
    t_l = (wx - 0.36) / SUN_DIR[0]
    hlz = wz - t_l * SUN_DIR[2]
    hly = wy - t_l * SUN_DIR[1]
    l_hm = np.clip((hlz - Z_PATH) / 0.08, 0.0, 1.0) * np.clip((1.92 - hlz) / 0.15, 0.0, 1.0) * (t_l > 0.0)
    l_ym = np.clip((0.30 - np.abs(hly - 6.2)) / 0.14, 0.0, 1.0)
    sun *= (1.0 - 0.72 * l_hm * l_ym).astype(np.float32)
    return np.clip(sun, 0.0, 1.0)


def mirror_pts_across_plane(pts: list[np.ndarray], z_plane: float = Z_CANAL) -> list[np.ndarray]:
    """Exact 3D plane mirror: pts - 2 * dot(pts - p0, n) * n for canal water plane z = z_plane."""
    p0 = np.array([0.0, 0.0, z_plane], dtype=np.float64)
    n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return [p - 2.0 * float(np.dot(p - p0, n)) * n for p in pts]


# =============================================================================
# Stage 1 & 3 — Atmospheric Misty Cedar Forest Backdrop & Voronoi Cobblestones
# =============================================================================

def paint_misty_forest_backdrop(cam: Camera3D, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Atmospheric deep mossy Japanese cedar forest mist (NO sine-wave hills!)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = xx / W
    v = yy / H

    mottle = fbm_gouache(H, W, rng)
    dabs = flat_brush_dabs(H, W, rng, count=94)
    tooth = paper_tooth(H, W, rng)

    # Rich deep Matcha forest shadow canopy -> misty sage -> luminous morning cream clearing
    c_top = np.array([0.068, 0.138, 0.105], dtype=np.float32)
    c_mid = np.array([0.30, 0.44, 0.34], dtype=np.float32)
    c_mist = np.array([0.78, 0.85, 0.68], dtype=np.float32)

    hv = float(cam.horizon_y / H)
    t_sky = np.clip(v / max(0.32, hv + 0.08), 0.0, 1.0)
    t1 = np.clip(t_sky / 0.56, 0.0, 1.0)
    t2 = np.clip((t_sky - 0.34) / 0.66, 0.0, 1.0)
    bg = (
        c_top[None, None, :] * (1.0 - t1[..., None])
        + c_mid[None, None, :] * (t1 * (1.0 - t2))[..., None]
        + c_mist[None, None, :] * (t1 * t2)[..., None]
    )

    # Luminous golden-cream upper-left morning sky opening aligned with -SUN_DIR!
    du = u - 0.28
    dv = v - 0.16
    sun_halo = np.exp(-(du * du * 4.2 + dv * dv * 7.6)).astype(np.float32)
    sun_core = np.exp(-(du * du * 22.0 + dv * dv * 38.0)).astype(np.float32)
    bg += sun_halo[..., None] * np.array([0.44, 0.46, 0.26], dtype=np.float32)
    bg += sun_core[..., None] * np.array([0.44, 0.42, 0.28], dtype=np.float32)

    # Soft radiant shrine sanctuary mist clearing right along the vanishing point (u ~ 0.60, v ~ 0.46)
    vp_glow = np.exp(-(((u - 0.60) / 0.20) ** 2 + ((v - 0.46) / 0.16) ** 2)).astype(np.float32)
    bg += vp_glow[..., None] * np.array([0.20, 0.22, 0.12], dtype=np.float32)

    # Soft atmospheric vertical distant cedar grove trunks & feathery canopy haze (irregular seeded widths!)
    for layer_i, (n_trunks, fog_a, col_dark, col_mist) in enumerate(
        [
            (26, 0.42, [0.18, 0.29, 0.22], [0.68, 0.76, 0.60]),
            (20, 0.56, [0.10, 0.19, 0.14], [0.46, 0.58, 0.44]),
            (14, 0.68, [0.048, 0.108, 0.078], [0.24, 0.35, 0.25]),
        ]
    ):
        trunk_acc = np.zeros(W, dtype=np.float32)
        for _ in range(n_trunks):
            tx = float(rng.uniform(-0.02, 1.02))
            # Keep the luminous central shrine corridor open & airy
            if abs(tx - 0.59) < (0.11 - 0.02 * layer_i):
                continue
            tw = float(rng.uniform(0.005, 0.015)) * (1.0 + 0.35 * layer_i)
            trunk_acc = np.maximum(trunk_acc, np.exp(-((u[0] - tx) / tw) ** 2))

        # Soft blurred canopy cloud hanging across top third, fading smoothly downward into ground mist
        canopy_drop = 0.22 + 0.14 * np.exp(-((u[0] - 0.45) / 0.28) ** 2)
        canopy_v = np.clip((canopy_drop[None, :] + 0.06 * mottle - v) / 0.18, 0.0, 1.0)
        vert_fade = np.clip((hv + 0.14 - v) / 0.14, 0.0, 1.0)
        mask = np.clip(np.maximum(trunk_acc[None, :] * 0.85, canopy_v * 0.72), 0.0, 1.0) * vert_fade * fog_a

        rel_h = np.clip(v / max(0.3, hv + 0.08), 0.0, 1.0)
        cd = np.array(col_dark, dtype=np.float32)
        cm = np.array(col_mist, dtype=np.float32)
        t_col = cd[None, None, :] * (1.0 - rel_h[..., None]) + cm[None, None, :] * rel_h[..., None]
        bg = bg * (1.0 - mask[..., None]) + t_col * mask[..., None]

    bg += (mottle * 0.016 + dabs * 0.012 + tooth * 0.005)[..., None]

    # Thick luminous morning mist pooling gracefully along the distant horizon/canal line!
    mist_band = np.exp(-((v - (hv + 0.005)) / 0.072) ** 2) * (0.48 + 0.24 * np.clip(mottle, -1.0, 1.0))
    bg = bg * (1.0 - mist_band[..., None]) + np.array([0.74, 0.82, 0.66], dtype=np.float32) * mist_band[..., None]

    fog_rgb = np.array([0.52, 0.63, 0.50], dtype=np.float32)
    return np.clip(bg, 0.0, 1.35), fog_rgb, mottle, dabs


def paint_voronoi_cobblestone_walkway_uv(rng: np.random.Generator, tw: int = 960, th: int = 1280) -> np.ndarray:
    """
    Stage 1 & 2: Winding mossy shrine walkway paved with irregular Voronoi (d2 - d1)
    wet cobblestones, smooth penumbra torii shadows, and seamless far-mist dissolution!
    """
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wx = -0.65 + u * (5.40 - (-0.65))
    wy = 1.2 + v * 28.8
    wz = np.full_like(wx, Z_PATH)

    nx = resize_bicubic(rng.standard_normal((th // 26, tw // 26)).astype(np.float32), tw, th) * 0.055
    ny = resize_bicubic(rng.standard_normal((th // 26, tw // 26)).astype(np.float32), tw, th) * 0.055
    px, py = wx + nx, wy + ny

    cell_x, cell_y = 0.44, 0.48
    ix = np.floor(px / cell_x).astype(np.int32)
    iy = np.floor(py / cell_y).astype(np.int32)

    jx_tab = rng.uniform(0.18, 0.82, size=256)
    jy_tab = rng.uniform(0.18, 0.82, size=256)
    tone_tab = rng.uniform(-0.055, 0.065, size=256)
    moss_tab = rng.uniform(0.0, 1.0, size=256)

    d1 = np.full((th, tw), 1e9, dtype=np.float64)
    d2 = np.full((th, tw), 1e9, dtype=np.float64)
    best_id = np.zeros((th, tw), dtype=np.int32)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cx, cy = ix + dx, iy + dy
            h_id = ((cx * 73856093) ^ (cy * 19349663)) & 255
            sx = (cx + jx_tab[h_id]) * cell_x
            sy = (cy + jy_tab[h_id]) * cell_y
            dist = np.hypot(px - sx, py - sy)
            closer = dist < d1
            d2 = np.where(closer, d1, np.minimum(d2, dist))
            d1 = np.where(closer, dist, d1)
            best_id = np.where(closer, h_id, best_id)

    edge_dist = (d2 - d1).astype(np.float32)
    cobble_mask = np.clip((edge_dist - 0.024) / 0.032, 0.0, 1.0)
    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=92)
    tooth = paper_tooth(th, tw, rng)

    bank_x = bank_x_at_y(wy)
    dist_from_bank = (wx - bank_x).astype(np.float32)
    shoulder_moss = np.clip((0.45 - dist_from_bank) / 0.45, 0.0, 1.0) + np.clip((dist_from_bank - 2.65) / 0.85, 0.0, 1.0)
    shoulder_moss = np.clip(shoulder_moss + mottle * 0.22, 0.0, 1.0)

    sv = tone_tab[best_id].astype(np.float32)
    stone = np.zeros((th, tw, 3), dtype=np.float32)
    stone[:, :, 0] = 0.155 + sv * 0.42 + mottle * 0.024 + dabs * 0.016
    stone[:, :, 1] = 0.195 + sv * 0.46 + mottle * 0.028 + dabs * 0.018
    stone[:, :, 2] = 0.172 + sv * 0.40 + mottle * 0.022 + dabs * 0.015

    moss = np.zeros((th, tw, 3), dtype=np.float32)
    moss[:, :, 0] = 0.088 + mottle * 0.030 + dabs * 0.022
    moss[:, :, 1] = 0.200 + mottle * 0.048 + dabs * 0.032
    moss[:, :, 2] = 0.086 + mottle * 0.020 + dabs * 0.014

    is_moss_cell = (moss_tab[best_id] < (0.14 + 0.72 * shoulder_moss)).astype(np.float32)
    eff_stone = cobble_mask * (1.0 - 0.88 * is_moss_cell)
    crevice_ao = np.exp(-((edge_dist - 0.020) / 0.014) ** 2) * (1.0 - is_moss_cell)

    rgb = eff_stone[..., None] * stone + (1.0 - eff_stone[..., None]) * moss
    rgb *= (1.0 - 0.48 * crevice_ao)[..., None]

    # Carved mossy curbstone lip bordering the winding canal
    curb = (dist_from_bank >= 0.0) & (dist_from_bank <= 0.24)
    curb_col = np.array([0.21, 0.25, 0.21], dtype=np.float32)[None, None, :] + (mottle * 0.022)[..., None]
    curb_joint = np.abs(np.mod(wy, 0.72) - 0.36) > 0.345
    curb_col = np.where(curb_joint[..., None], curb_col * 0.42, curb_col)
    rgb = np.where(curb[..., None], 0.74 * curb_col + 0.26 * moss, rgb)

    # Stage 2: Smooth 6-step -SUN_DIR canopy + torii gate + lantern penumbra shadows
    sun = trace_shrine_canopy_sunlight(wx, wy, wz)
    sun_stone = np.array([0.48, 0.50, 0.28], dtype=np.float32)
    sun_moss = np.array([0.38, 0.48, 0.16], dtype=np.float32)
    rgb += sun[..., None] * (eff_stone[..., None] * sun_stone + (1.0 - eff_stone[..., None]) * sun_moss)

    puddle = eff_stone * np.clip((mottle * 0.55 + 0.22) * (1.0 - shoulder_moss * 0.7), 0.0, 1.0)
    sky_refl = np.array([0.56, 0.66, 0.52], dtype=np.float32)
    rgb = rgb * (1.0 - 0.32 * puddle[..., None]) + sky_refl[None, None, :] * (0.32 * puddle[..., None])
    rgb += (puddle * sun)[..., None] * np.array([0.28, 0.30, 0.18], dtype=np.float32)

    # Warm candlelight spill around the hero stone lantern at (0.36, 6.2)
    d_lan = np.hypot(wx - 0.36, wy - 6.2).astype(np.float32)
    lan_glow = np.exp(-(d_lan / 1.15) ** 2) * 0.28
    rgb += lan_glow[..., None] * np.array([0.94, 0.80, 0.40], dtype=np.float32)

    rgb = apply_shadow_bounce(rgb)
    rgb += tooth[..., None] * 0.008

    # Smooth winding embankment alpha AND seamless far-mist dissolution (ZERO straight cutoff!)
    alpha_bank = np.clip((wx - bank_x) / 0.06, 0.0, 1.0).astype(np.float32)
    alpha_far = np.clip((27.5 - wy) / 5.5, 0.0, 1.0).astype(np.float32)
    alpha = alpha_bank * alpha_far
    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.35) * alpha[..., None]
    rgba[:, :, 3] = alpha
    return rgba


def paint_left_mossy_forest_bank_uv(rng: np.random.Generator, tw: int = 512, th: int = 1024) -> np.ndarray:
    """Lush mossy earthen left bank of the shrine canal so left-side cedar trunks root naturally in mossy soil!"""
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wx = -7.8 + u * (0.0 - (-7.8))
    wy = 2.0 + v * 26.0

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=82)
    tooth = paper_tooth(th, tw, rng)

    edge_x = far_left_bank_x(wy) + 0.28 * mottle
    rgb = np.zeros((th, tw, 3), dtype=np.float32)
    rgb[:, :, 0] = 0.085 + mottle * 0.028 + dabs * 0.022
    rgb[:, :, 1] = 0.195 + mottle * 0.045 + dabs * 0.032
    rgb[:, :, 2] = 0.082 + mottle * 0.018 + dabs * 0.014

    sun = trace_shrine_canopy_sunlight(wx, wy, np.full_like(wx, 0.22))
    rgb += sun[..., None] * np.array([0.28, 0.36, 0.14], dtype=np.float32)
    # Dark wet muddy shoreline transition
    wet = np.clip((wx - (edge_x - 0.40)) / 0.40, 0.0, 1.0).astype(np.float32)
    rgb *= (1.0 - 0.52 * wet)[..., None]
    rgb = apply_shadow_bounce(rgb)
    rgb += tooth[..., None] * 0.007

    alpha = np.clip((edge_x - wx) / 0.22, 0.0, 1.0).astype(np.float32) * np.clip((25.5 - wy) / 4.5, 0.0, 1.0).astype(np.float32)
    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.25) * alpha[..., None]
    rgba[:, :, 3] = alpha
    return rgba


# =============================================================================
# Stage 3 — 2x Supersampled PyCairo Shrine Architecture, Seamless Cedars & Lilies
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
        im = Image.fromarray(np.clip(rgb2[:, :, c], 0, 255).astype(np.uint8), mode="L").resize(
            (W, H), Image.Resampling.LANCZOS
        )
        out[:, :, c] = np.asarray(im, dtype=np.float32) / 255.0
    a1 = (
        np.asarray(Image.fromarray(rgba_2x[:, :, 3], mode="L").resize((W, H), Image.Resampling.LANCZOS), dtype=np.float32)
        / 255.0
    )
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
    tremor: float = 0.28,
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


def fill_3d_quad(
    ctx: cairo.Context,
    cam: Camera3D,
    pts3: list[np.ndarray],
    rgba: tuple[float, float, float, float],
    grad_rgba: tuple[float, float, float, float] | None = None,
) -> None:
    cl = cam.clip_polygon_3d(pts3)
    if len(cl) < 3:
        return
    proj = [cam.project(p) for p in cl]
    if grad_rgba is not None:
        g = cairo.LinearGradient(float(proj[0][0]), float(proj[0][1]), float(proj[1][0]), float(proj[1][1]))
        g.add_color_stop_rgba(0.0, *rgba)
        g.add_color_stop_rgba(1.0, *grad_rgba)
        ctx.set_source(g)
    else:
        ctx.set_source_rgba(*rgba)
    ctx.move_to(float(proj[0][0]), float(proj[0][1]))
    for pu, pv, _ in proj[1:]:
        ctx.line_to(float(pu), float(pv))
    ctx.close_path()
    ctx.fill()


def fog_mix(c3: Iterable[float], dist: float, fog_rgb: np.ndarray, fog_start: float = 5.0, fog_density: float = 0.062) -> tuple[float, float, float]:
    ft = 1.0 - math.exp(-max(0.0, dist - fog_start) * fog_density)
    c = np.asarray(c3, dtype=np.float64) * (1.0 - ft) + fog_rgb * ft
    return (float(c[0]), float(c[1]), float(c[2]))


def draw_stone_canal_embankment(ctx: cairo.Context, cam: Camera3D, rng: np.random.Generator, fog_rgb: np.ndarray) -> None:
    """Segmented 3D mossy stone canal wall (ishigaki) + rounded weathered mossy timber pilings."""
    n_seg = 42
    ys = np.linspace(25.5, 2.2, n_seg + 1)
    for i in range(n_seg):
        y0, y1 = float(ys[i]), float(ys[i + 1])
        x0, x1 = float(bank_x_at_y(y0)), float(bank_x_at_y(y1))
        dist = float(np.linalg.norm(np.array([x0, y0, 0.14]) - cam.eye))
        c_top = fog_mix([0.16, 0.24, 0.16], dist, fog_rgb)
        c_bot = fog_mix([0.048, 0.098, 0.078], dist, fog_rgb)
        q = [
            np.array([x0, y0, Z_CANAL - 0.02]),
            np.array([x1, y1, Z_CANAL - 0.02]),
            np.array([x1, y1, Z_PATH + 0.01]),
            np.array([x0, y0, Z_PATH + 0.01]),
        ]
        fill_3d_quad(ctx, cam, q, (*c_bot, 0.99), (*c_top, 0.99))

        if i % 2 == 0:
            seg = cam.project_polyline(np.array([q[0], q[3]]))
            if len(seg) >= 2:
                stroke_catmull_rom(ctx, [(p[0], p[1]) for p in seg], base_w=1.2, z_depth=dist, rgba=(0.03, 0.06, 0.04, 0.65), rng=rng, taper=False, tremor=0.18)

    # Weathered mossy wooden canal retention pilings with rounded tops & damp moss band
    for py_p in (18.8, 14.8, 11.2, 8.3, 5.8):
        px_p = float(bank_x_at_y(py_p)) - 0.14
        dist = float(np.linalg.norm(np.array([px_p, py_p, 0.22]) - cam.eye))
        r = 0.046
        u_b, v_b, z_b = cam.project(np.array([px_p, py_p, Z_CANAL]))
        u_t, v_t, z_t = cam.project(np.array([px_p, py_p, Z_PATH + 0.16]))
        if z_b < Z_NEAR or z_t < Z_NEAR:
            continue
        rw = max(2.0, r * cam.K[0, 0] / float(z_b))
        c_l = fog_mix([0.28, 0.27, 0.18], dist, fog_rgb)
        c_r = fog_mix([0.075, 0.088, 0.065], dist, fog_rgb)
        c_m = fog_mix([0.16, 0.32, 0.14], dist, fog_rgb)
        g = cairo.LinearGradient(float(u_b) - rw, float(v_t), float(u_b) + rw, float(v_t))
        g.add_color_stop_rgba(0.0, *c_l, 0.98)
        g.add_color_stop_rgba(0.65, *c_r, 0.98)
        g.add_color_stop_rgba(1.0, *(v * 0.72 for v in c_r), 0.98)
        ctx.set_source(g)
        ctx.move_to(float(u_b) - rw, float(v_b) + 2.0)
        ctx.line_to(float(u_t) - rw * 0.94, float(v_t))
        ctx.arc(float(u_t), float(v_t), rw * 0.94, math.pi, 0.0)
        ctx.line_to(float(u_b) + rw, float(v_b) + 2.0)
        ctx.close_path()
        ctx.fill()
        # Damp green moss waterline collar
        ctx.set_source_rgba(*c_m, 0.72)
        ctx.rectangle(float(u_b) - rw, float(v_b) - rw * 0.9, rw * 2.0, rw * 0.95)
        ctx.fill()


def draw_mossy_cedar_trunk(
    ctx: cairo.Context,
    cam: Camera3D,
    base_pos: np.ndarray,
    radius: float,
    height: float,
    seed: int,
    fog_rgb: np.ndarray,
    mirrored: bool = False,
) -> None:
    """
    Seamless continuous towering mossy Japanese cedar (sugi) trunk (ZERO horizontal
    quad cracks!), natural flared mossy root mound, continuous sinuous bark fissures,
    creeping ground moss (NO polka-dot ovals!), and connected feathery conifer clouds!
    """
    srng = np.random.default_rng(seed)
    dist = float(np.linalg.norm(base_pos - cam.eye))
    alpha_m = 0.56 if mirrored else 0.99

    c_lit = fog_mix([0.38, 0.36, 0.22], dist, fog_rgb)
    c_mid = fog_mix([0.135, 0.125, 0.095], dist, fog_rgb)
    c_dark = fog_mix([0.048, 0.064, 0.052], dist, fog_rgb)
    c_moss_sh = fog_mix([0.12, 0.25, 0.12], dist, fog_rgb)
    c_moss_hi = fog_mix([0.44, 0.59, 0.24], dist, fog_rgb)

    lean_x = float(srng.uniform(-0.14, 0.14))
    n_pts = 14
    left_edge, right_edge = [], []
    for i in range(n_pts):
        t = i / (n_pts - 1)
        # Organic flared buttress roots near base (t < 0.14), smooth slight taper upward
        root_flare = 0.42 * math.exp(-t * 14.0)
        r_t = radius * (1.0 - 0.28 * t + root_flare)
        sway = lean_x * t + 0.03 * math.sin(t * 5.0 + seed)
        p_l = base_pos + np.array([sway - r_t, 0.0, height * t])
        p_r = base_pos + np.array([sway + r_t, 0.0, height * t])
        if mirrored:
            p_l, p_r = mirror_pts_across_plane([p_l, p_r], Z_CANAL)
        ul, vl, zl = cam.project(p_l)
        ur, vr, zr = cam.project(p_r)
        if zl < Z_NEAR or zr < Z_NEAR:
            continue
        left_edge.append((float(ul), float(vl)))
        right_edge.append((float(ur), float(vr)))

    if len(left_edge) < 3:
        return

    # 1. ONE continuous smooth tapered trunk polygon with natural flared roots merging into mossy soil!
    xs_all = [p[0] for p in left_edge + right_edge]
    bx0, by0 = 0.5 * (left_edge[0][0] + right_edge[0][0]), left_edge[0][1]
    bw0 = abs(right_edge[0][0] - left_edge[0][0])

    # Soft dark root contact shadow tucked directly beneath the cedar buttress roots (NO giant dome hills!)
    if not mirrored:
        ctx.save()
        ctx.translate(bx0, by0 + bw0 * 0.05)
        ctx.scale(1.0, 0.22)
        g_ao = cairo.RadialGradient(0, 0, bw0 * 0.15, 0, 0, bw0 * 0.92)
        g_ao.add_color_stop_rgba(0.0, 0.025, 0.048, 0.032, 0.72)
        g_ao.add_color_stop_rgba(1.0, 0.025, 0.048, 0.032, 0.0)
        ctx.set_source(g_ao)
        ctx.arc(0, 0, bw0 * 0.92, 0, math.tau)
        ctx.fill()
        ctx.restore()

    g = cairo.LinearGradient(min(xs_all), left_edge[0][1], max(xs_all), left_edge[0][1])
    g.add_color_stop_rgba(0.0, *c_lit, alpha_m)
    g.add_color_stop_rgba(0.24, *c_mid, alpha_m)
    g.add_color_stop_rgba(0.72, *c_dark, alpha_m)
    g.add_color_stop_rgba(1.0, *(v * 0.82 for v in c_dark), alpha_m)
    ctx.set_source(g)
    # Organic scalloped curved root bottom edge instead of a flat horizontal line!
    sign_m = -1.0 if mirrored else 1.0
    ctx.move_to(left_edge[0][0] - bw0 * 0.16, left_edge[0][1] + sign_m * bw0 * 0.07)
    ctx.curve_to(left_edge[0][0] - bw0 * 0.04, left_edge[0][1] - sign_m * bw0 * 0.08, left_edge[1][0], left_edge[1][1] + sign_m * bw0 * 0.12, left_edge[1][0], left_edge[1][1])
    for p in left_edge[2:]:
        ctx.line_to(*p)
    for p in reversed(right_edge[1:]):
        ctx.line_to(*p)
    ctx.curve_to(right_edge[1][0], right_edge[1][1] + sign_m * bw0 * 0.12, right_edge[0][0] + bw0 * 0.04, right_edge[0][1] - sign_m * bw0 * 0.08, right_edge[0][0] + bw0 * 0.16, right_edge[0][1] + sign_m * bw0 * 0.07)
    ctx.curve_to(bx0 + bw0 * 0.24, by0 + sign_m * bw0 * 0.14, bx0 - bw0 * 0.24, by0 + sign_m * bw0 * 0.14, left_edge[0][0] - bw0 * 0.16, left_edge[0][1] + sign_m * bw0 * 0.07)
    ctx.close_path()
    ctx.fill()

    if not mirrored:
        # 2. Continuous fine sinuous vertical cedar bark fissures (soft & painterly!)
        n_bark = 5
        for b_k in range(n_bark):
            u_f = 0.14 + 0.72 * (b_k / max(1, n_bark - 1)) + float(srng.uniform(-0.05, 0.05))
            b_pts = []
            for j in range(len(left_edge)):
                lx, ly = left_edge[j]
                rx, ry = right_edge[j]
                w_loc = rx - lx
                wob = math.sin(j * 0.9 + b_k * 2.1 + seed) * w_loc * 0.035
                b_pts.append((lx + u_f * w_loc + wob, 0.5 * (ly + ry)))
            stroke_catmull_rom(ctx, b_pts, base_w=1.35, z_depth=dist, rgba=(0.025, 0.036, 0.028, 0.38), rng=srng, taper=False, tremor=0.32)

        # 3. Soft feathered velvety bark moss patches clinging organically along the damp left bark & roots!
        ctx.save()
        ctx.move_to(*left_edge[0])
        for p in left_edge[1:]:
            ctx.line_to(*p)
        for p in reversed(right_edge):
            ctx.line_to(*p)
        ctx.close_path()
        ctx.clip()
        for m_i in range(11):
            frac_h = float(srng.uniform(0.0, 0.42))
            idx_e = min(len(left_edge) - 2, int(frac_h * (len(left_edge) - 1)))
            lx, ly = left_edge[idx_e]
            rx, _ = right_edge[idx_e]
            w_loc = rx - lx
            mx = lx + w_loc * float(srng.uniform(0.04, 0.34 * (1.0 - frac_h * 1.4)))
            my = ly + float(srng.uniform(-w_loc * 0.25, w_loc * 0.25))
            mrx = w_loc * float(srng.uniform(0.16, 0.34))
            mry = mrx * float(srng.uniform(1.1, 1.9))
            gm = cairo.RadialGradient(-mrx * 0.18, 0.0, mrx * 0.08, 0.0, 0.0, mrx)
            gm.add_color_stop_rgba(0.0, *c_moss_hi, 0.62)
            gm.add_color_stop_rgba(0.55, *c_moss_sh, 0.48)
            gm.add_color_stop_rgba(1.0, *c_moss_sh, 0.0)
            ctx.save()
            ctx.translate(mx, my)
            ctx.scale(1.0, mry / max(1.0, mrx))
            ctx.set_source(gm)
            ctx.arc(0, 0, mrx, 0, math.tau)
            ctx.fill()
            ctx.restore()
        ctx.restore()

        # 4. Delicate tapered pointed wild forest sedge blades tucking naturally around the cedar roots!
        for k_f in range(14):
            fx = bx0 + float(srng.uniform(-bw0 * 0.68, bw0 * 0.68))
            fy = by0 + bw0 * float(srng.uniform(0.04, 0.12))
            fl = bw0 * float(srng.uniform(0.22, 0.48))
            bw_blade = max(1.4, bw0 * float(srng.uniform(0.018, 0.032)))
            ang = -math.pi * 0.5 + float(srng.uniform(-0.72, 0.72))
            ca, sa = math.cos(ang), math.sin(ang)
            tx = fx + ca * fl
            ty = fy + sa * fl + fl * 0.12
            col_f = c_moss_hi if (fx < bx0 and k_f % 3 == 0) else c_moss_sh
            ctx.set_source_rgba(*col_f, 0.86)
            ctx.move_to(fx - bw_blade, fy)
            ctx.curve_to(fx + ca * fl * 0.35 - bw_blade * 0.6, fy + sa * fl * 0.45, fx + ca * fl * 0.72 - bw_blade * 0.2, fy + sa * fl * 0.80, tx, ty)
            ctx.curve_to(fx + ca * fl * 0.72 + bw_blade * 0.2, fy + sa * fl * 0.80, fx + ca * fl * 0.35 + bw_blade * 0.6, fy + sa * fl * 0.45, fx + bw_blade, fy)
            ctx.close_path()
            ctx.fill()

        # 5. Connected woody cedar limbs + lush layered soft feathery conifer foliage clouds!
        c_fol_deep = fog_mix([0.042, 0.105, 0.068], dist, fog_rgb)
        c_fol_mid = fog_mix([0.115, 0.225, 0.125], dist, fog_rgb)
        c_fol_lit = fog_mix([0.42, 0.56, 0.23], dist, fog_rgb)
        for b_i in range(4):
            bz = height * float(srng.uniform(0.46, 0.88))
            side = -1.0 if base_pos[0] > 0.5 else 1.0
            if b_i == 3:
                side = -side
            b_len = float(srng.uniform(1.6, 3.0))
            b_start = base_pos + np.array([lean_x * (bz / height), 0.0, bz])
            b_mid = b_start + np.array([side * b_len * 0.55, 0.0, -float(srng.uniform(0.15, 0.45))])
            b_end = b_start + np.array([side * b_len, 0.0, -float(srng.uniform(0.28, 0.75))])
            seg = cam.project_polyline(np.array([b_start, b_mid, b_end]))
            if len(seg) < 2:
                continue
            # Woody branch limb connecting trunk to foliage cloud
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in seg], base_w=7.5, z_depth=dist, rgba=(*c_dark, 0.92), rng=srng, taper=True, tremor=0.25)

            # Soft layered scalloped Japanese cedar (sugi) foliage puffs draping gracefully off the limb!
            n_cl = 7
            for k in range(n_cl):
                f = 0.22 + 0.78 * (k / (n_cl - 1))
                p_cur = b_start * (1.0 - f) ** 2 + b_mid * 2.0 * (1.0 - f) * f + b_end * (f ** 2)
                fu, fv, fz = cam.project(p_cur)
                if fz < Z_NEAR:
                    continue
                rad_px = float(srng.uniform(0.36, 0.68)) * (1.15 - 0.35 * f) * cam.K[0, 0] / float(fz)
                # Multi-puff scalloped conifer cloud (shadow base -> mid foliage -> sunlit upper-left rim!)
                for puff_i in range(6):
                    ox = float(srng.uniform(-rad_px * 0.9, rad_px * 0.9))
                    oy = float(srng.uniform(-rad_px * 0.42, rad_px * 0.48))
                    prx = rad_px * float(srng.uniform(0.42, 0.72))
                    pry = prx * float(srng.uniform(0.42, 0.62))
                    # Upper-left puffs catch golden-olive sun; lower-right puffs sit in deep pine shadow
                    lit_w = float(np.clip(0.50 - (ox / max(1.0, rad_px)) * 0.35 - (oy / max(1.0, rad_px)) * 0.45, 0.0, 1.0))
                    col = (
                        tuple(c_fol_deep[c] * (1 - lit_w * 2) + c_fol_mid[c] * (lit_w * 2) for c in range(3))
                        if lit_w < 0.5
                        else tuple(c_fol_mid[c] * (2 - lit_w * 2) + c_fol_lit[c] * (lit_w * 2 - 1) for c in range(3))
                    )
                    ctx.save()
                    ctx.translate(float(fu) + ox, float(fv) + oy)
                    ctx.scale(1.0, pry / max(1.0, prx))
                    # Scalloped organic pine bough silhouette (7 gently notched lobes)
                    ctx.set_source_rgba(*col, 0.92)
                    n_lobes = 7
                    for lb in range(n_lobes):
                        a0 = (math.tau * lb) / n_lobes
                        a1 = (math.tau * (lb + 0.5)) / n_lobes
                        a2 = (math.tau * (lb + 1.0)) / n_lobes
                        r_out = prx * float(srng.uniform(0.92, 1.18))
                        r_in = prx * 0.72
                        if lb == 0:
                            ctx.move_to(math.cos(a0) * r_in, math.sin(a0) * r_in)
                        ctx.curve_to(math.cos(a1) * r_out, math.sin(a1) * r_out, math.cos(a1 + 0.15) * r_out, math.sin(a1 + 0.15) * r_out, math.cos(a2) * r_in, math.sin(a2) * r_in)
                    ctx.close_path()
                    ctx.fill()
                    ctx.restore()


def draw_weathered_torii_gate(
    ctx: cairo.Context,
    cam: Camera3D,
    y_pos: float,
    x_left: float,
    x_right: float,
    z_top: float,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    mirrored: bool = False,
    with_shimenawa: bool = True,
) -> None:
    """Weathered dark-wood rural Kyoto Myojin torii gate with moss cushions, kusabi wedges & shimenawa."""
    cx = 0.5 * (x_left + x_right)
    span = x_right - x_left
    dist = float(np.linalg.norm(np.array([cx, y_pos, 1.8]) - cam.eye))
    alpha_m = 0.58 if mirrored else 0.99

    c_dark = fog_mix([0.088, 0.078, 0.066], dist, fog_rgb)
    c_mid = fog_mix([0.175, 0.152, 0.118], dist, fog_rgb)
    c_lit = fog_mix([0.48, 0.46, 0.28], dist, fog_rgb)
    c_moss = fog_mix([0.20, 0.38, 0.16], dist, fog_rgb)
    c_moss_hi = fog_mix([0.58, 0.72, 0.30], dist, fog_rgb)
    c_stone = fog_mix([0.12, 0.15, 0.13], dist, fog_rgb)

    def m_pts(pts: list[np.ndarray]) -> list[np.ndarray]:
        return mirror_pts_across_plane(pts, Z_CANAL) if mirrored else pts

    r_col = 0.105
    cant = 0.09
    z_base = Z_PATH
    z_nuki = z_top - 0.58

    for side, xb in ((-1.0, x_left), (1.0, x_right)):
        xt = xb - side * cant
        q_ped = m_pts([
            np.array([xb - r_col * 1.45, y_pos, z_base - 0.02]),
            np.array([xb + r_col * 1.45, y_pos, z_base - 0.02]),
            np.array([xb + r_col * 1.25, y_pos, z_base + 0.34]),
            np.array([xb - r_col * 1.25, y_pos, z_base + 0.34]),
        ])
        fill_3d_quad(ctx, cam, q_ped, (*c_stone, alpha_m), (*(v * 0.72 for v in c_stone), alpha_m))
        q_col = m_pts([
            np.array([xb - r_col, y_pos, z_base + 0.30]),
            np.array([xb + r_col, y_pos, z_base + 0.30]),
            np.array([xt + r_col * 0.88, y_pos, z_top - 0.12]),
            np.array([xt - r_col * 0.88, y_pos, z_top - 0.12]),
        ])
        fill_3d_quad(ctx, cam, q_col, (*c_lit, alpha_m), (*c_dark, alpha_m))
        # Traditional wooden joint wedges (kusabi) & collar (daiwa) at nuki & top!
        q_wed = m_pts([
            np.array([xt - r_col * 1.35, y_pos - 0.01, z_nuki - 0.12]),
            np.array([xt + r_col * 1.35, y_pos - 0.01, z_nuki - 0.12]),
            np.array([xt + r_col * 1.35, y_pos - 0.01, z_nuki + 0.13]),
            np.array([xt - r_col * 1.35, y_pos - 0.01, z_nuki + 0.13]),
        ])
        fill_3d_quad(ctx, cam, q_wed, (*c_dark, alpha_m))

        if not mirrored:
            seg_rim = cam.project_polyline(np.array([q_col[0], q_col[3]]))
            if len(seg_rim) >= 2:
                stroke_catmull_rom(ctx, [(p[0], p[1]) for p in seg_rim], base_w=2.2, z_depth=dist, rgba=(*c_lit, 0.86), rng=rng, taper=False, tremor=0.20)

    q_nuki = m_pts([
        np.array([x_left - 0.44, y_pos, z_nuki - 0.08]),
        np.array([x_right + 0.44, y_pos, z_nuki - 0.08]),
        np.array([x_right + 0.44, y_pos, z_nuki + 0.09]),
        np.array([x_left - 0.44, y_pos, z_nuki + 0.09]),
    ])
    fill_3d_quad(ctx, cam, q_nuki, (*c_mid, alpha_m), (*c_dark, alpha_m))

    q_gak = m_pts([
        np.array([cx - 0.15, y_pos - 0.02, z_nuki + 0.06]),
        np.array([cx + 0.15, y_pos - 0.02, z_nuki + 0.06]),
        np.array([cx + 0.13, y_pos - 0.02, z_top - 0.16]),
        np.array([cx - 0.13, y_pos - 0.02, z_top - 0.16]),
    ])
    fill_3d_quad(ctx, cam, q_gak, (*c_lit, alpha_m), (*c_dark, alpha_m))

    n_k = 14
    bot_k, top_k, moss_k = [], [], []
    x_k0, x_k1 = x_left - 0.62, x_right + 0.62
    for i in range(n_k + 1):
        t = (i / n_k) * 2.0 - 1.0
        xk = cx + t * (0.5 * (x_k1 - x_k0))
        curve_up = 0.22 * (t * t) * (0.6 + 0.4 * abs(t))
        zb = z_top - 0.18 + curve_up * 0.78
        zt = z_top + 0.05 + curve_up
        bot_k.append(np.array([xk, y_pos, zb]))
        top_k.append(np.array([xk, y_pos, zt]))
        moss_h = 0.075 * (0.55 + 0.45 * math.sin(i * 1.45 + y_pos)) * (1.0 - 0.40 * abs(t))
        moss_k.append(np.array([xk, y_pos, zt + moss_h]))

    poly_k = m_pts(bot_k + list(reversed(top_k)))
    cl_k = cam.clip_polygon_3d(poly_k)
    if len(cl_k) >= 3:
        pr_k = [cam.project(p) for p in cl_k]
        ys_k = [float(p[1]) for p in pr_k]
        g_k = cairo.LinearGradient(float(pr_k[0][0]), min(ys_k), float(pr_k[0][0]), max(ys_k))
        g_k.add_color_stop_rgba(0.0, *c_lit, alpha_m)
        g_k.add_color_stop_rgba(0.38, *c_mid, alpha_m)
        g_k.add_color_stop_rgba(1.0, *c_dark, alpha_m)
        ctx.set_source(g_k)
        ctx.move_to(float(pr_k[0][0]), float(pr_k[0][1]))
        for pu, pv, _ in pr_k[1:]:
            ctx.line_to(float(pu), float(pv))
        ctx.close_path()
        ctx.fill()

    poly_m = m_pts(top_k + list(reversed(moss_k)))
    cl_m = cam.clip_polygon_3d(poly_m)
    if len(cl_m) >= 3:
        pr_m = [cam.project(p) for p in cl_m]
        xs_m = [float(p[0]) for p in pr_m]
        g_m = cairo.LinearGradient(min(xs_m), float(pr_m[0][1]), max(xs_m), float(pr_m[0][1]))
        g_m.add_color_stop_rgba(0.0, *c_moss_hi, alpha_m)
        g_m.add_color_stop_rgba(0.65, *c_moss, alpha_m)
        g_m.add_color_stop_rgba(1.0, *c_moss, alpha_m)
        ctx.set_source(g_m)
        ctx.move_to(float(pr_m[0][0]), float(pr_m[0][1]))
        for pu, pv, _ in pr_m[1:]:
            ctx.line_to(float(pu), float(pv))
        ctx.close_path()
        ctx.fill()

    if with_shimenawa and not mirrored:
        rope_pts = []
        for k in range(11):
            t = k / 10.0
            rx = (x_left + 0.08) + t * (span - 0.16)
            sag = 0.24 * math.sin(t * math.pi)
            ru, rv, rz = cam.project(np.array([rx, y_pos - 0.03, z_nuki - 0.08 - sag]))
            if rz > Z_NEAR:
                rope_pts.append((float(ru), float(rv)))
        if len(rope_pts) >= 3:
            c_rope = fog_mix([0.52, 0.46, 0.28], dist, fog_rgb)
            c_rope_sh = fog_mix([0.22, 0.18, 0.11], dist, fog_rgb)
            stroke_catmull_rom(ctx, rope_pts, base_w=8.5, z_depth=dist, rgba=(*c_rope_sh, 0.92), rng=rng, taper=True, tremor=0.25)
            stroke_catmull_rom(ctx, [(x, y - 1.2) for x, y in rope_pts], base_w=4.8, z_depth=dist, rgba=(*c_rope, 0.92), rng=rng, taper=True, tremor=0.20)
            c_shide = fog_mix([0.96, 0.96, 0.88], dist, fog_rgb)
            for idx_s in (2, 4, 6, 8):
                sx, sy = rope_pts[idx_s]
                sc = max(4.0, 0.16 * cam.K[0, 0] / max(2.0, dist))
                ctx.set_source_rgba(*c_shide, 0.94)
                ctx.move_to(sx, sy + 2.0)
                ctx.line_to(sx - sc * 0.35, sy + sc * 0.55)
                ctx.line_to(sx + sc * 0.30, sy + sc * 0.85)
                ctx.line_to(sx - sc * 0.25, sy + sc * 1.45)
                ctx.line_to(sx + sc * 0.38, sy + sc * 1.35)
                ctx.line_to(sx + sc * 0.05, sy + sc * 0.72)
                ctx.line_to(sx + sc * 0.42, sy + sc * 0.42)
                ctx.close_path()
                ctx.fill()


def draw_kasuga_stone_lantern(
    ctx: cairo.Context,
    cam: Camera3D,
    base_pos: np.ndarray,
    scale: float,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    mirrored: bool = False,
) -> None:
    """Mossy hexagonal Kyoto Kasuga stone lantern with clean luminous firebox & carved stone lattice!"""
    dist = float(np.linalg.norm(base_pos - cam.eye))
    alpha_m = 0.62 if mirrored else 0.99
    c_lit = fog_mix([0.56, 0.61, 0.42], dist, fog_rgb)
    c_mid = fog_mix([0.27, 0.32, 0.26], dist, fog_rgb)
    c_dark = fog_mix([0.10, 0.14, 0.11], dist, fog_rgb)
    c_moss = fog_mix([0.19, 0.37, 0.15], dist, fog_rgb)

    def proj_pt(dx: float, dz: float) -> tuple[float, float, float]:
        p = base_pos + np.array([dx * scale, 0.0, dz * scale], dtype=np.float64)
        if mirrored:
            p = mirror_pts_across_plane([p], Z_CANAL)[0]
        u, v, z = cam.project(p)
        return float(u), float(v), float(z)

    def poly_rel(pts_rel: list[tuple[float, float]], left_c, mid_c, right_c=None):
        pr = [proj_pt(dx, dz) for dx, dz in pts_rel]
        if any(z < Z_NEAR for _, _, z in pr):
            return
        xs = [p[0] for p in pr]
        ys = [p[1] for p in pr]
        g = cairo.LinearGradient(min(xs), min(ys), max(xs), min(ys))
        g.add_color_stop_rgba(0.0, *left_c, alpha_m)
        g.add_color_stop_rgba(0.55, *mid_c, alpha_m)
        g.add_color_stop_rgba(1.0, *(right_c if right_c is not None else c_dark), alpha_m)
        ctx.set_source(g)
        ctx.move_to(pr[0][0], pr[0][1])
        for p in pr[1:]:
            ctx.line_to(p[0], p[1])
        ctx.close_path()
        ctx.fill()

    # Pedestal, tapered mossy shaft (sao), carved lotus platform (chudai), stone firebox (hibukuro)
    poly_rel([(-0.36, 0.0), (-0.30, 0.20), (0.30, 0.20), (0.36, 0.0)], c_mid, c_dark)
    poly_rel([(-0.125, 0.18), (-0.102, 0.86), (0.102, 0.86), (0.125, 0.18)], c_lit, c_mid)
    poly_rel([(-0.28, 0.84), (-0.35, 0.99), (0.35, 0.99), (0.28, 0.84)], c_lit, c_mid)
    poly_rel([(-0.23, 0.99), (-0.23, 1.34), (0.23, 1.34), (0.23, 0.99)], c_lit, c_mid)

    # Pure luminous golden-cream candle flame inside firebox (NO dark shadow stop inside flame!)
    cu, cv, cz = proj_pt(0.0, 1.16)
    if cz > Z_NEAR:
        rad = max(4.0, 0.18 * scale * cam.K[0, 0] / cz)
        glow = cairo.RadialGradient(cu, cv, rad * 0.1, cu, cv, rad * 2.9)
        glow.add_color_stop_rgba(0.0, 1.0, 0.97, 0.80, 0.96 * alpha_m)
        glow.add_color_stop_rgba(0.38, 0.94, 0.84, 0.44, 0.54 * alpha_m)
        glow.add_color_stop_rgba(1.0, 0.82, 0.70, 0.28, 0.0)
        ctx.set_source(glow)
        ctx.arc(cu, cv, rad * 2.9, 0, math.tau)
        ctx.fill()

        # Pure glowing warm paper/candle pane
        poly_rel(
            [(-0.10, 1.04), (-0.10, 1.28), (0.10, 1.28), (0.10, 1.04)],
            (0.99, 0.93, 0.66),
            (1.00, 0.98, 0.88),
            (0.98, 0.90, 0.58),
        )
        # Delicate carved stone mullion bar silhouetted cleanly across the warm candle window!
        if not mirrored:
            p_m1, p_m2 = proj_pt(0.0, 1.04), proj_pt(0.0, 1.28)
            p_h1, p_h2 = proj_pt(-0.10, 1.16), proj_pt(0.10, 1.16)
            ctx.set_source_rgba(*c_dark, 0.75)
            ctx.set_line_width(max(1.2, 0.016 * scale * cam.K[0, 0] / cz))
            ctx.move_to(p_m1[0], p_m1[1])
            ctx.line_to(p_m2[0], p_m2[1])
            ctx.move_to(p_h1[0], p_h1[1])
            ctx.line_to(p_h2[0], p_h2[1])
            ctx.stroke()

    # Pagoda-curved hexagonal stone roof (kasa) + upturned eaves + lotus finial (hoju) + velvety moss!
    poly_rel(
        [(-0.50, 1.34), (-0.36, 1.49), (-0.14, 1.63), (0.14, 1.63), (0.36, 1.49), (0.50, 1.34), (0.39, 1.29), (-0.39, 1.29)],
        c_lit,
        c_mid,
    )
    poly_rel([(-0.09, 1.62), (-0.05, 1.78), (0.0, 1.86), (0.05, 1.78), (0.09, 1.62)], c_lit, c_mid)
    poly_rel([(-0.40, 1.42), (-0.18, 1.60), (0.12, 1.59), (0.30, 1.46), (0.04, 1.44), (-0.24, 1.39)], c_lit, c_moss, c_moss)
    poly_rel([(-0.34, 0.04), (-0.26, 0.21), (0.06, 0.20), (0.24, 0.06)], c_moss, c_moss, c_dark)


def draw_water_lilies(ctx: cairo.Context, cam: Camera3D, rng: np.random.Generator, fog_rgb: np.ndarray) -> None:
    """Floating emerald-olive water lily pads (suiren) with organic veins & delicate cream blossoms."""
    clusters = [
        (-0.85, 5.4, 7, 2, 501),
        (-1.60, 6.6, 8, 2, 502),
        (-0.62, 8.2, 6, 1, 503),
        (-1.75, 9.8, 7, 2, 504),
        (-1.05, 12.2, 6, 1, 505),
        (-1.95, 7.6, 5, 1, 506),
        (-0.68, 5.15, 4, 1, 507),
    ]
    for cx, cy, n_pads, n_blos, cseed in clusters:
        srng = np.random.default_rng(cseed)
        pads = []
        for _ in range(n_pads):
            px = cx + float(srng.uniform(-0.54, 0.54))
            py = cy + float(srng.uniform(-0.60, 0.60))
            max_x = float(bank_x_at_y(py)) - 0.26
            min_x = float(far_left_bank_x(py)) + 0.35
            px = float(np.clip(px, min_x, max_x))
            prad = float(srng.uniform(0.11, 0.21))
            pads.append((px, py, prad, float(srng.uniform(0, math.tau))))
        pads.sort(key=lambda t: -t[1])

        for i, (px, py, prad, notch_a) in enumerate(pads):
            ux, uy, uz = cam.project(np.array([px, py, Z_CANAL + 0.004]))
            if uz < Z_NEAR or ux < -40 or ux > W + 40 or uy < cam.horizon_y or uy > H + 30:
                continue
            dist = float(np.linalg.norm(np.array([px, py, Z_CANAL]) - cam.eye))
            r_px = prad * cam.K[0, 0] / float(uz)
            aspect = float(np.clip(0.24 + 0.14 * (1.0 - min(1.0, (dist - 4.0) / 14.0)), 0.18, 0.40))

            sun_v = float(trace_shrine_canopy_sunlight(np.array([px]), np.array([py]), np.array([Z_CANAL]))[0])
            c_pad_sh = fog_mix([0.08 + 0.10 * sun_v, 0.20 + 0.16 * sun_v, 0.11 + 0.06 * sun_v], dist, fog_rgb)
            c_pad_hi = fog_mix([0.24 + 0.34 * sun_v, 0.42 + 0.34 * sun_v, 0.17 + 0.18 * sun_v], dist, fog_rgb)

            ctx.save()
            ctx.translate(float(ux), float(uy))
            ctx.scale(1.0, aspect)
            ctx.set_source_rgba(0.02, 0.06, 0.04, 0.56)
            ctx.arc(2.0, 4.0, r_px * 1.04, 0, math.tau)
            ctx.fill()

            g = cairo.RadialGradient(-r_px * 0.28, -r_px * 0.22, r_px * 0.1, 0, 0, r_px)
            g.add_color_stop_rgba(0.0, *c_pad_hi, 0.98)
            g.add_color_stop_rgba(0.75, *c_pad_sh, 0.98)
            g.add_color_stop_rgba(1.0, *(v * 0.72 for v in c_pad_sh), 0.98)
            ctx.set_source(g)
            ctx.move_to(0, 0)
            ctx.arc(0, 0, r_px, notch_a + 0.24, notch_a + math.tau - 0.24)
            ctx.close_path()
            ctx.fill()

            # Delicate radiating leaf veins + warm sunlit pad lip
            ctx.set_source_rgba(*c_pad_hi, 0.36)
            ctx.set_line_width(max(0.7, r_px * 0.035))
            for vk in range(7):
                va = notch_a + 0.45 + (math.tau - 0.90) * (vk / 6.0)
                ctx.move_to(0, 0)
                ctx.line_to(math.cos(va) * r_px * 0.86, math.sin(va) * r_px * 0.86)
                ctx.stroke()
            ctx.arc(0, 0, r_px * 0.94, notch_a + 0.32, notch_a + math.tau - 0.32)
            ctx.stroke()
            ctx.restore()

            if i < n_blos and r_px > 9.0:
                bx = float(ux) + float(srng.uniform(-r_px * 0.16, r_px * 0.16))
                by = float(uy) - r_px * aspect * 0.25
                br = r_px * float(srng.uniform(0.32, 0.44))
                c_pet_out = fog_mix([0.84, 0.87, 0.76], dist, fog_rgb)
                c_pet_in = fog_mix([0.98, 0.98, 0.92], dist, fog_rgb)
                c_stam = fog_mix([0.88, 0.78, 0.34], dist, fog_rgb)

                for p_k in range(7):
                    pa = -math.pi * 0.12 - (math.pi * 0.76) * (p_k / 6.0) + float(srng.uniform(-0.08, 0.08))
                    plen = br * float(srng.uniform(0.88, 1.12))
                    pw = plen * 0.34
                    ca, sa = math.cos(pa), math.sin(pa)
                    px_n, py_n = -sa * pw, ca * pw
                    col = c_pet_in if (1 <= p_k <= 5) else c_pet_out
                    ctx.set_source_rgba(*col, 0.96)
                    ctx.move_to(bx, by)
                    ctx.curve_to(bx + ca * plen * 0.4 + px_n, by + sa * plen * 0.4 + py_n, bx + ca * plen * 0.8 + px_n * 0.4, by + sa * plen * 0.8 + py_n * 0.4, bx + ca * plen, by + sa * plen)
                    ctx.curve_to(bx + ca * plen * 0.8 - px_n * 0.4, by + sa * plen * 0.8 - py_n * 0.4, bx + ca * plen * 0.4 - px_n, by + sa * plen * 0.4 - py_n, bx, by)
                    ctx.close_path()
                    ctx.fill()
                ctx.set_source_rgba(*c_stam, 0.96)
                ctx.arc(bx, by - br * 0.18, max(1.5, br * 0.18), 0, math.tau)
                ctx.fill()


# =============================================================================
# Stage 2 — Calm Emerald-Jade Shrine Canal with Fresnel & 3D Plane Reflections
# =============================================================================

def paint_canal_fresnel_water(
    canvas: np.ndarray,
    ref_backdrop: np.ndarray,
    mirrored_3d_rgba: np.ndarray,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    mottle_full: np.ndarray,
) -> np.ndarray:
    """Calm emerald-jade shrine canal with capillary-rippled horizon + 3D mirrored Fresnel reflections."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    rx_c = (xx - cam.K[0, 2]) / cam.K[0, 0]
    ry_c = (yy - cam.K[1, 2]) / cam.K[1, 1]
    RT = cam.R.T
    dx_w = RT[0, 0] * rx_c + RT[0, 1] * ry_c + RT[0, 2]
    dy_w = RT[1, 0] * rx_c + RT[1, 1] * ry_c + RT[1, 2]
    dz_w = RT[2, 0] * rx_c + RT[2, 1] * ry_c + RT[2, 2]

    hit = dz_w < -0.004
    t_w = np.where(hit, (Z_CANAL - cam.eye[2]) / np.where(hit, dz_w, -1.0), -1.0)
    wx = cam.eye[0] + dx_w * t_w
    wy = cam.eye[1] + dy_w * t_w

    bank_x = bank_x_at_y(wy)
    valid = hit & (t_w > Z_NEAR) & (wy > 1.2) & (wy < 34.0) & (wx <= bank_x + 0.16)
    canal_mask = (
        valid.astype(np.float32)
        * np.clip((bank_x + 0.16 - wx) / 0.12, 0.0, 1.0).astype(np.float32)
        * np.clip((30.0 - wy) / 5.0, 0.0, 1.0).astype(np.float32)
    )
    if not np.any(canal_mask > 1e-4):
        return canvas

    # Horizontal capillary canal ripples (ONLY used on water!)
    rip1 = resize_bicubic(rng.standard_normal((H // 5, W // 46)).astype(np.float32), W, H)
    rip2 = resize_bicubic(rng.standard_normal((H // 3, W // 20)).astype(np.float32), W, H)
    rip = 0.60 * rip1 + 0.40 * rip2

    dist_t = np.clip((yy.astype(np.float32) - float(cam.horizon_y)) / max(1.0, H - float(cam.horizon_y)), 0.0, 1.0)
    rip_scale = (1.4 + 6.8 * dist_t).astype(np.float32)

    map_x = xx.astype(np.float32) + rip * 0.88 * rip_scale
    map_y = (2.0 * float(cam.horizon_y) - yy.astype(np.float32)) + rip * 1.20 * rip_scale

    refl_bg = remap_bilinear(ref_backdrop, map_x, np.clip(map_y, 0.0, H - 1.001))
    mx_3d = xx.astype(np.float32) + rip * 0.78 * rip_scale
    my_3d = yy.astype(np.float32) + rip * 0.68 * rip_scale
    refl_3d = remap_bilinear(mirrored_3d_rgba, mx_3d, my_3d)
    refl_all = over_premul(refl_bg, refl_3d) * np.array([0.76, 0.90, 0.80], dtype=np.float32)[None, None, :]

    bed_col = np.array([0.030, 0.115, 0.092], dtype=np.float32)[None, None, :] + (mottle_full * 0.020)[..., None] * np.array(
        [0.35, 0.85, 0.65], dtype=np.float32
    )
    fresnel = 0.26 + (0.86 - 0.26) * np.power(1.0 - dist_t, 1.4)
    water_rgb = bed_col * (1.0 - fresnel[..., None]) + refl_all * fresnel[..., None]

    sun_w = trace_shrine_canopy_sunlight(wx, wy, np.full_like(wx, Z_CANAL))
    shimmer = np.clip((rip + 0.55) * 0.75, 0.0, 1.0) * sun_w
    water_rgb += shimmer[..., None] * np.array([0.24, 0.30, 0.14], dtype=np.float32)
    glint = np.clip((rip - 0.92) * 2.4, 0.0, 1.0) * sun_w
    water_rgb += glint[..., None] * np.array([0.50, 0.52, 0.30], dtype=np.float32)

    near_wall = np.exp(-np.maximum(0.0, (bank_x - wx).astype(np.float32)) / 0.32)
    water_rgb *= (1.0 - 0.38 * near_wall)[..., None]

    fog_t = 1.0 - np.exp(-np.maximum(0.0, t_w.astype(np.float32) - 5.5) * 0.056)
    water_rgb = water_rgb * (1.0 - fog_t[..., None]) + fog_rgb[None, None, :] * fog_t[..., None]
    return canvas * (1.0 - canal_mask[..., None]) + water_rgb * canal_mask[..., None]


# =============================================================================
# Stage 2 & 3 — Diagonal Golden-Olive Forest Sunbeams Aligned With SUN_DIR
# =============================================================================

def paint_forest_sunbeams(cam: Camera3D, rng: np.random.Generator, intensity: float = 0.52, front: bool = False) -> np.ndarray:
    """Diagonal golden-olive morning sunbeams piercing from upper-left strictly parallel to SUN_DIR."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    p_ref = np.array([0.8, 8.0, 2.2], dtype=np.float64)
    u0, v0, _ = cam.project(p_ref)
    u1, v1, _ = cam.project(p_ref + SUN_DIR * 3.0)
    dx, dy = float(u1 - u0), float(v1 - v0)
    dl = math.hypot(dx, dy) + 1e-6
    rx, ry = dx / dl, dy / dl
    nx, ny = -ry, rx

    s_coord = (xx - W * 0.30) * rx + (yy - H * 0.06) * ry
    p_coord = (xx - W * 0.30) * nx + (yy - H * 0.06) * ny

    acc = np.zeros((H, W), dtype=np.float32)
    shafts = (
        [(-310, 56, 0.42), (-145, 68, 0.68), (30, 58, 0.74), (195, 74, 0.62), (370, 68, 0.40)]
        if not front
        else [(-200, 42, 0.34), (-20, 52, 0.52), (155, 44, 0.46), (310, 50, 0.30)]
    )
    for cp, wp, st in shafts:
        w_eff = wp * (1.0 + np.maximum(0.0, s_coord) / (H * 1.40))
        beam = np.exp(-0.5 * ((p_coord - cp) / w_eff) ** 2)
        vert = np.clip((s_coord + 260.0) / 280.0, 0.0, 1.0) * np.exp(-np.maximum(0.0, s_coord) / (H * 1.12))
        acc += beam * vert * st

    turb = 0.80 + 0.20 * resize_bicubic(fbm_gouache(H // 4, W // 4, rng), W, H)
    acc = np.clip(acc * turb * intensity, 0.0, 1.0)
    col = np.array([0.94, 0.95, 0.66], dtype=np.float32) if front else np.array([0.80, 0.88, 0.52], dtype=np.float32)
    return acc[..., None] * col[None, None, :]


# =============================================================================
# Stage 4 — Foreground Circular-Bokeh Fern Fronds & 35mm Analog Optics
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
    """Exact FFT 2D convolution restricted to padded non-zero bounding box (identical output, ~15x faster!)."""
    ih, iw = img.shape[:2]
    kh, kw = kernel.shape
    nz = np.any(img > 1e-6, axis=2)
    if not np.any(nz):
        return np.zeros_like(img, dtype=np.float32)
    ys, xs = np.where(nz)
    pad = max(kh, kw) + 4
    y0, y1 = max(0, int(ys.min()) - pad), min(ih, int(ys.max()) + pad + 1)
    x0, x1 = max(0, int(xs.min()) - pad), min(iw, int(xs.max()) + pad + 1)
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


def draw_fern_frond(
    ctx: cairo.Context,
    spine_pts: list[tuple[float, float]],
    max_pinna: float,
    srng: np.random.Generator,
    backlit: float,
) -> None:
    """Per-seed arching Japanese forest fern frond (shida) with serrated paired Bezier pinnae."""
    ext = [spine_pts[0]] + spine_pts + [spine_pts[-1]]
    dense: list[tuple[float, float]] = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for s in range(12):
            t = s / 12.0
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            dense.append((x, y))
    dense.append(spine_pts[-1])

    c_stem = (0.042 + 0.08 * backlit, 0.112 + 0.14 * backlit, 0.058 + 0.05 * backlit, 0.96)
    ctx.set_source_rgba(*c_stem)
    n = len(dense) - 1
    for i in range(n):
        t = i / max(1, n)
        ctx.set_line_width(max(1.2, 7.2 * (1.0 - 0.82 * t)))
        ctx.move_to(*dense[i])
        ctx.line_to(*dense[i + 1])
        ctx.stroke()

    n_pairs = 22
    for k in range(2, n_pairs):
        frac = k / n_pairs
        idx = min(n - 1, int(frac * n))
        x, y = dense[idx]
        x_n, y_n = dense[min(n, idx + 2)]
        tx, ty = x_n - x, y_n - y
        tl = math.hypot(tx, ty) + 1e-6
        tx, ty = tx / tl, ty / tl
        nx_v, ny_v = -ty, tx

        env = math.sin(math.pow(frac, 0.72) * math.pi)
        plen = max_pinna * env * float(srng.uniform(0.88, 1.12))
        pwid = plen * float(srng.uniform(0.18, 0.24))

        for side in (-1.0, 1.0):
            dx = nx_v * side * 0.86 + tx * 0.50
            dy = ny_v * side * 0.86 + ty * 0.50
            dl = math.hypot(dx, dy) + 1e-6
            dx, dy = dx / dl, dy / dl
            px_v, py_v = -dy * pwid, dx * pwid
            droop = plen * float(srng.uniform(0.06, 0.16))

            tip_x = x + dx * plen
            tip_y = y + dy * plen + droop
            lit_l = backlit * (0.55 + 0.45 * frac)
            c_base = (0.040 + 0.10 * lit_l, 0.112 + 0.20 * lit_l, 0.055 + 0.07 * lit_l, 0.96)
            c_tip = (0.18 + 0.48 * lit_l, 0.34 + 0.46 * lit_l, 0.12 + 0.22 * lit_l, 0.95)
            g = cairo.LinearGradient(x, y, tip_x, tip_y)
            g.add_color_stop_rgba(0.0, *c_base)
            g.add_color_stop_rgba(1.0, *c_tip)
            ctx.set_source(g)
            ctx.move_to(x, y)
            ctx.curve_to(x + dx * plen * 0.38 + px_v, y + dy * plen * 0.38 + py_v, x + dx * plen * 0.75 + px_v * 0.48, y + dy * plen * 0.75 + py_v * 0.48 + droop * 0.6, tip_x, tip_y)
            ctx.curve_to(x + dx * plen * 0.75 - px_v * 0.48, y + dy * plen * 0.75 - py_v * 0.48 + droop * 0.6, x + dx * plen * 0.38 - px_v, y + dy * plen * 0.38 - py_v, x, y)
            ctx.close_path()
            ctx.fill()


def paint_foreground_bokeh_fern_fronds(rng: np.random.Generator) -> np.ndarray:
    """Stage 4: Near-camera mossy fern fronds framing lower-left corner with r=7.8 disc bokeh."""
    surf, ctx = new_cairo_2x()
    fronds = [
        ([(-30, H + 45), (55, H - 155), (145, H - 335), (275, H - 460), (405, H - 515)], 108.0, 8101, 0.82),
        ([(-45, H - 25), (85, H - 115), (235, H - 195), (395, H - 235), (535, H - 220)], 98.0, 8102, 0.64),
        ([(-25, H + 55), (120, H - 40), (285, H - 95), (455, H - 102), (610, H - 62)], 92.0, 8103, 0.42),
        ([(-40, H - 180), (42, H - 310), (108, H - 455), (175, H - 585)], 88.0, 8104, 0.76),
        ([(-15, H + 40), (22, H - 85), (75, H - 215), (178, H - 315)], 82.0, 8105, 0.32),
    ]
    for spts, mp, fseed, bl in fronds:
        draw_fern_frond(ctx, spts, mp, np.random.default_rng(fseed), bl)

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
    hl_mask = np.clip((lum - 0.72) / 0.28, 0.0, 1.5)
    hl_src = img * hl_mask[..., None]
    tint = np.array([1.03, 1.06, 0.85], dtype=np.float32)
    hal = (
        0.22 * gaussian_blur_2d(hl_src, 6.5)
        + 0.16 * gaussian_blur_2d(hl_src, 24.0)
        + 0.10 * gaussian_blur_2d(hl_src, 72.0)
    ) * tint[None, None, :]
    img = 1.0 - (1.0 - np.clip(img, 0.0, 1.0)) * (1.0 - np.clip(hal, 0.0, 0.82))

    # 2. Radial corner chromatic aberration + gentle optical vignette
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (xx - W * 0.5) / (W * 0.5)
    ny = (yy - H * 0.5) / (H * 0.5)
    r2 = nx * nx + ny * ny
    ca_scale = 0.00105
    img[:, :, 0] = remap_bilinear(img[:, :, 0], xx + nx * r2 * (W * ca_scale), yy + ny * r2 * (H * ca_scale))
    img[:, :, 2] = remap_bilinear(img[:, :, 2], xx - nx * r2 * (W * ca_scale), yy - ny * r2 * (H * ca_scale))
    img *= np.clip(1.0 - 0.24 * (r2 ** 1.15), 0.68, 1.0)[..., None]

    # 3. Linear shadow black lift + C1-continuous tanh highlight shoulder (knee=0.80, ceiling=0.965)
    img = 0.014 + img * 0.986
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
    #    + slight luma ring + sub-pixel per-line horizontal tape jitter + <= 0.8% scanlines
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
    out *= (1.0 - 0.007 * (0.5 - 0.5 * np.cos(np.arange(H, dtype=np.float32) * math.pi)))[:, None, None]
    return np.clip(out, 0.0, 1.0)


# =============================================================================
# Full Scene Assembly & Deterministic Export
# =============================================================================

def render_scene(
    seed: int = 202602,
    audit_dir: pathlib.Path | None = None,
    export_webps: list[pathlib.Path] | None = None,
) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # Stage 1: Off-axis Camera3D (VP ~60.5% width, canal open on left, torii walkway on right)
    cam = Camera3D(W, H, eye=(0.95, 0.0, 1.56), target=(-0.45, 12.0, 1.44), fov_x_deg=58.0)
    print(f"[matcha-02] Camera VP: ({cam.vp_x:.1f}, {cam.vp_y:.1f}) -> {cam.vp_x / W * 100:.1f}% width")

    # Atmospheric deep mossy Japanese cedar forest mist backdrop
    canvas, fog_rgb, m_global, d_global = paint_misty_forest_backdrop(cam, rng)

    # Ancient mossy cedar trunks rooted firmly ON the mossy left forest bank (x < far_left_bank_x) and right walkway shoulder!
    far_cedars = [
        (np.array([float(far_left_bank_x(21.5)) - 0.75, 21.5, 0.18]), 0.34, 9.2, 301),
        (np.array([4.0, 21.0, Z_PATH]), 0.36, 9.6, 302),
        (np.array([float(far_left_bank_x(18.0)) - 0.90, 18.0, 0.18]), 0.38, 9.5, 303),
        (np.array([4.2, 17.5, Z_PATH]), 0.38, 9.8, 304),
    ]
    mid_cedars = [
        (np.array([float(far_left_bank_x(13.8)) - 0.65, 13.8, 0.20]), 0.42, 9.4, 311),
        (np.array([3.85, 13.5, Z_PATH]), 0.44, 9.8, 312),
        (np.array([float(far_left_bank_x(10.2)) - 0.75, 10.2, 0.20]), 0.46, 9.6, 313),
        (np.array([4.15, 10.4, Z_PATH]), 0.46, 9.8, 314),
    ]
    near_cedars = [
        (np.array([float(far_left_bank_x(6.5)) - 0.65, 6.5, 0.22]), 0.48, 9.2, 321),
        (np.array([3.95, 6.2, Z_PATH]), 0.50, 9.5, 322),
    ]
    lantern_far = np.array([0.42, 12.8, Z_PATH], dtype=np.float64)
    lantern_hero = np.array([0.36, 6.2, Z_PATH], dtype=np.float64)

    # Stage 3 Band 0 (Far Shrine Sanctuary: Torii #3 & Far Mossy Cedar Trunks)
    s_far, c_far = new_cairo_2x()
    for bp, rad, ht, cseed in far_cedars:
        draw_mossy_cedar_trunk(c_far, cam, bp, rad, ht, cseed, fog_rgb, mirrored=False)
    ty, txl, txr, tzt = TORII_SPECS[0]
    draw_weathered_torii_gate(c_far, cam, ty, txl, txr, tzt, rng, fog_rgb, mirrored=False, with_shimenawa=False)
    canvas = over_premul(canvas, finish_cairo_2x(s_far))

    # Snapshot horizon reflection source BEFORE painting water
    ref_backdrop = canvas.copy()

    # Dedicated 3D plane-mirrored quads pass (torii gates, cedar trunks, glowing stone lanterns!)
    s_ref, c_ref = new_cairo_2x()
    for bp, rad, ht, cseed in far_cedars + mid_cedars + near_cedars:
        draw_mossy_cedar_trunk(c_ref, cam, bp, rad, ht, cseed, fog_rgb, mirrored=True)
    for ty, txl, txr, tzt in TORII_SPECS:
        draw_weathered_torii_gate(c_ref, cam, ty, txl, txr, tzt, rng, fog_rgb, mirrored=True, with_shimenawa=False)
    draw_kasuga_stone_lantern(c_ref, cam, lantern_far, 0.85, rng, fog_rgb, mirrored=True)
    draw_kasuga_stone_lantern(c_ref, cam, lantern_hero, 0.88, rng, fog_rgb, mirrored=True)
    mirrored_3d = finish_cairo_2x(s_ref)

    # Stage 2: Calm emerald-jade shrine canal with capillary-rippled Fresnel reflections
    canvas = paint_canal_fresnel_water(canvas, ref_backdrop, mirrored_3d, cam, rng, fog_rgb, m_global)

    # Stage 1: Left mossy earthen forest bank + canal stone embankment + wet Voronoi cobblestone walkway!
    left_bank_uv = paint_left_mossy_forest_bank_uv(rng)
    canvas = cam.warp_plane_over(
        canvas,
        left_bank_uv,
        p0=np.array([-7.8, 2.0, 0.16]),
        e1=np.array([7.8, 0.0, 0.0]),
        e2=np.array([0.0, 26.0, 0.0]),
        fog_rgb=fog_rgb,
        fog_start=5.5,
        fog_density=0.056,
    )

    s_emb, c_emb = new_cairo_2x()
    draw_stone_canal_embankment(c_emb, cam, rng, fog_rgb)
    canvas = over_premul(canvas, finish_cairo_2x(s_emb))

    walkway_uv = paint_voronoi_cobblestone_walkway_uv(rng)
    canvas = cam.warp_plane_over(
        canvas,
        walkway_uv,
        p0=np.array([-0.65, 1.2, Z_PATH]),
        e1=np.array([5.40 - (-0.65), 0.0, 0.0]),
        e2=np.array([0.0, 28.8, 0.0]),
        fog_rgb=fog_rgb,
        fog_start=5.5,
        fog_density=0.056,
    )

    # Stage 3 Band 1 (Midground Torii #2, Towering Seamless Mossy Cedars & Midground Lantern)
    s_mid, c_mid = new_cairo_2x()
    for bp, rad, ht, cseed in mid_cedars:
        draw_mossy_cedar_trunk(c_mid, cam, bp, rad, ht, cseed, fog_rgb, mirrored=False)
    ty, txl, txr, tzt = TORII_SPECS[1]
    draw_weathered_torii_gate(c_mid, cam, ty, txl, txr, tzt, rng, fog_rgb, mirrored=False, with_shimenawa=True)
    draw_kasuga_stone_lantern(c_mid, cam, lantern_far, 0.85, rng, fog_rgb, mirrored=False)
    mid_1x = finish_cairo_2x(s_mid)
    ma = mid_1x[:, :, 3:4]
    mid_1x[:, :, :3] = np.clip(mid_1x[:, :, :3] + ma * (m_global * 0.018 + d_global * 0.012)[..., None], 0.0, 1.25)
    canvas = over_premul(canvas, mid_1x)

    # Back diagonal golden-olive morning sunbeams piercing canopy aligned with SUN_DIR
    rays_back = paint_forest_sunbeams(cam, rng, intensity=0.50, front=False)
    canvas = 1.0 - (1.0 - np.clip(canvas, 0.0, 1.0)) * (1.0 - np.clip(rays_back, 0.0, 0.85))

    # Stage 3 Band 2 (Hero Near Torii #1, Hero Mossy Stone Lantern, Near Cedars & Floating Water Lilies)
    s_hero, c_hero = new_cairo_2x()
    draw_water_lilies(c_hero, cam, rng, fog_rgb)
    for bp, rad, ht, cseed in near_cedars:
        draw_mossy_cedar_trunk(c_hero, cam, bp, rad, ht, cseed, fog_rgb, mirrored=False)
    ty, txl, txr, tzt = TORII_SPECS[2]
    draw_weathered_torii_gate(c_hero, cam, ty, txl, txr, tzt, rng, fog_rgb, mirrored=False, with_shimenawa=True)
    draw_kasuga_stone_lantern(c_hero, cam, lantern_hero, 0.88, rng, fog_rgb, mirrored=False)

    hero_1x = finish_cairo_2x(s_hero)
    h_tooth = paper_tooth(H, W, rng)
    ha = hero_1x[:, :, 3:4]
    hero_1x[:, :, :3] = np.clip(hero_1x[:, :, :3] + ha * (m_global * 0.016 + h_tooth * 0.007)[..., None], 0.0, 1.25)
    canvas = over_premul(canvas, hero_1x)

    # Front delicate golden-olive sunbeam veil + drifting backlit forest dust motes
    rays_front = paint_forest_sunbeams(cam, rng, intensity=0.32, front=True)
    canvas = 1.0 - (1.0 - np.clip(canvas, 0.0, 1.0)) * (1.0 - np.clip(rays_front, 0.0, 0.82))

    s_motes, c_motes = new_cairo_2x()
    for _ in range(110):
        mx = float(rng.uniform(100, W - 100))
        my = float(rng.uniform(100, H - 100))
        mr = float(rng.uniform(0.9, 2.2))
        mv = float(rng.uniform(0.18, 0.56)) * (0.5 + 0.5 * math.exp(-((mx / W - 0.45) / 0.34) ** 2))
        c_motes.set_source_rgba(0.95, 0.96, 0.74, mv)
        c_motes.arc(mx, my, mr, 0, math.tau)
        c_motes.fill()
    canvas = over_premul(canvas, finish_cairo_2x(s_motes))

    # Stage 4: Lower-left foreground circular-bokeh fern fronds + 35mm analog finish
    fg_bokeh = paint_foreground_bokeh_fern_fronds(rng)
    canvas = over_premul(canvas, fg_bokeh)

    final_rgb = apply_35mm_finish(canvas, rng)
    elapsed = time.perf_counter() - t0
    print(f"[matcha-02] Rendered {W}x{H} in {elapsed:.2f}s")

    u8 = np.round(np.clip(final_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    full_im = Image.fromarray(u8, mode="RGB")

    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        full_im.save(audit_dir / "w02_full.png", optimize=True)
        crops = {
            "w02_crop_aperture.png": (int(W * 0.24), int(H * 0.06)),        # Sunlit cedar canopy, feathery boughs & seamless bark
            "w02_crop_midground.png": (int(W * 0.44), int(H * 0.42)),       # Hero torii gate, glowing stone lantern with mullions & Voronoi stones
            "w02_crop_foreground.png": (int(W * 0.04), H - 450),            # Lower-left circular-bokeh ferns, mossy left bank & water lilies
        }
        for cname, (cx0, cy0) in crops.items():
            x0 = max(0, min(W - 600, cx0))
            y0 = max(0, min(H - 450, cy0))
            full_im.crop((x0, y0, x0 + 600, y0 + 450)).save(audit_dir / cname)

    if export_webps:
        for wp in export_webps:
            wp.parent.mkdir(parents=True, exist_ok=True)
            full_im.save(wp, format="WEBP", quality=90, method=6)
            print(f"[matcha-02] Exported {wp} ({wp.stat().st_size / 1024:.1f} KiB)")

    return final_rgb, {"elapsed": elapsed, "vp_ratio": cam.vp_x / W}


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    default_webps = [
        repo_root / "session" / "themes" / "matcha" / "walls" / "wall-02.webp",
        repo_root / "docs" / "assets" / "walls" / "matcha-02.webp",
    ]
    ap = argparse.ArgumentParser(description='Render Matcha 02 — "Mossy Shrine Canal" (2048x1152)')
    ap.add_argument("--seed", type=int, default=202602)
    ap.add_argument("--audit-dir", type=pathlib.Path, default=None)
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()
    render_scene(seed=args.seed, audit_dir=args.audit_dir, export_webps=None if args.no_export else default_webps)


if __name__ == "__main__":
    main()

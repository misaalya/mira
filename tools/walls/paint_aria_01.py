#!/usr/bin/env python3
"""
Aria 01 — "Aurora Rooftop Observatory"
Pure-Python 3D-Grounded Nocturnal Anime Scene Painting Pipeline (2048 x 1152)

Five-stage deterministic pipeline (zero image models, zero external assets):
  Stage 1: Metric right-handed Camera3D with off-axis vanishing point (~61.9%
           width), exact 3x3 UV-to-screen homography masked by valid UV bounds
           and ray-plane depth z_cam > 0.05 with exponential atmospheric fog,
           near-plane 3D line clipping, and irregular Voronoi (d2 - d1) wet
           concrete rooftop tiles.
  Stage 2: Unified normalized 3D light travel vector SUN_DIR (z < 0) tracing
           rooftop floor coordinates backward along -SUN_DIR with 6 fixed
           height-refinement steps onto the pitched/stepped observatory
           penthouse overhang & glowing skylight mullions; shallow concrete
           puddles combining capillary-rippled horizon reflections with 3D
           plane-mirrored architectural quads weighted by Fresnel angle.
  Stage 3: 4 depth-interleaved 2x supersampled PyCairo bands (Lanczos-
           downsampled) for the sleek observatory dome & penthouse, glowing
           architectural skylight studio, perimeter glass/steel railings,
           rooftop bench with glowing lantern, tripod refractor telescope, and
           seeded cubic-Bezier rooftop grasses, broken up by isotropic
           fbm_gouache, flat-brush dabs, cold-press paper tooth, and warm
           bounce light in deep shadows.
  Stage 4: Foreground lower-right concrete parapet & directional rooftop
           antenna silhouette convolved in premultiplied RGBA with a circular
           aperture disc bokeh kernel (r = 7.8) + 1 px opposite R/B lateral
           shift, followed by 35mm multi-scale halation, radial CA, vignette,
           shadow lift + C1 tanh highlight shoulder, luminance-weighted
           silver-halide grain, and faint analog YIQ transfer patina.
  Stage 5: Deterministic render exporting 2048x1152 WebP (quality=90, method=6)
           to session/themes/aria/walls/wall-01.webp and
           docs/assets/walls/aria-01.webp, plus optional 1:1 macro audit crops.
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

# Stage 2: Single normalized 3D celestial light travel vector (SUN_DIR[2] < 0)
_RAW_LIGHT = np.array([0.42, -0.24, -0.875], dtype=np.float64)
SUN_DIR = _RAW_LIGHT / np.linalg.norm(_RAW_LIGHT)

# World vertical & facade references (meters)
Z_DECK = 0.0
X_WALL = -1.15
Z_ROOF = 3.22  # Sleek modern penthouse roof height so observatory dome rises majestically above!


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
    if sigma <= 4.5:
        return gaussian_blur_1d(gaussian_blur_1d(arr, sigma, axis=1), sigma, axis=0)
    h, w = arr.shape[:2]
    factor = 2 if sigma < 12.0 else (4 if sigma < 40.0 else 8)
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


def flat_brush_dabs(h: int, w: int, rng: np.random.Generator, count: int = 92) -> np.ndarray:
    qh, qw = max(16, h // 4), max(16, w // 4)
    arr_sum = np.zeros((qh, qw), dtype=np.float32)
    s_one = cairo.ImageSurface(cairo.FORMAT_A8, qw, qh)
    c1 = cairo.Context(s_one)
    stride = s_one.get_stride()
    for _ in range(count):
        c1.set_operator(cairo.OPERATOR_CLEAR)
        c1.paint()
        c1.set_operator(cairo.OPERATOR_SOURCE)
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
    arr_sum = gaussian_blur_2d(arr_sum, 3.2)
    arr_sum -= arr_sum.mean()
    arr_sum /= float(arr_sum.std()) + 1e-6
    return resize_bicubic(arr_sum, w, h)


def over_premul(dst_rgb: np.ndarray, src_premul_rgba: np.ndarray) -> np.ndarray:
    a = np.clip(src_premul_rgba[:, :, 3:4], 0.0, 1.0)
    return src_premul_rgba[:, :, :3] + dst_rgb * (1.0 - a)


def screen_blend(dst_rgb: np.ndarray, add_rgb: np.ndarray) -> np.ndarray:
    d = np.clip(dst_rgb, 0.0, 1.0)
    a = np.clip(add_rgb, 0.0, 0.92)
    return 1.0 - (1.0 - d) * (1.0 - a)


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
        fog_start: float = 6.0,
        fog_density: float = 0.030,
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

    def ray_plane_z0(self, xx: np.ndarray, yy: np.ndarray, z_plane: float = 0.0):
        rx_c = (xx.astype(np.float64) - self.K[0, 2]) / self.K[0, 0]
        ry_c = (yy.astype(np.float64) - self.K[1, 2]) / self.K[1, 1]
        RT = self.R.T
        dx_w = RT[0, 0] * rx_c + RT[0, 1] * ry_c + RT[0, 2]
        dy_w = RT[1, 0] * rx_c + RT[1, 1] * ry_c + RT[1, 2]
        dz_w = RT[2, 0] * rx_c + RT[2, 1] * ry_c + RT[2, 2]
        hit = dz_w < -1e-4
        t = np.where(hit, (z_plane - self.eye[2]) / np.where(hit, dz_w, -1.0), -1.0)
        hit &= t > Z_NEAR
        wx = self.eye[0] + dx_w * t
        wy = self.eye[1] + dy_w * t
        return hit, wx, wy, t


def mirror_quad_z(pts: np.ndarray, z_plane: float = Z_DECK) -> np.ndarray:
    p0 = np.array([0.0, 0.0, z_plane], dtype=np.float64)
    n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    d = np.sum((pts - p0) * n, axis=-1, keepdims=True)
    return pts - 2.0 * d * n


# =============================================================================
# Stage 2 — Unified 3D Light Solver (Observatory Roof & Skylight Shadows)
# =============================================================================

def penthouse_roof_z(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Stepped & gently pitched architectural observatory penthouse roof overhang height."""
    pitch = (Z_ROOF - 0.04) + 0.14 * np.clip((x + 4.8) / 4.2, 0.0, 1.0)
    clerestory = np.where((x > -3.4) & (x < -1.15) & (y > 4.6) & (y < 11.4), 0.36, 0.0)
    return pitch + clerestory


def trace_rooftop_lighting(wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Trace rooftop deck coordinates backward along -SUN_DIR using 6 fixed
    height-refinement steps onto the pitched/stepped observatory penthouse roof
    and cantilevered skylight overhang. Returns (ambient_shadow, skylight_patch).
    """
    inv_sz = 1.0 / (-SUN_DIR[2])
    z_roof = np.full_like(wx, Z_ROOF + 0.08, dtype=np.float64)
    for _ in range(6):
        t = np.maximum(0.0, (z_roof - wz) * inv_sz)
        rx = wx - SUN_DIR[0] * t
        ry = wy - SUN_DIR[1] * t
        z_roof = penthouse_roof_z(rx, ry)

    t = np.maximum(0.0, (z_roof - wz) * inv_sz)
    rx = wx - SUN_DIR[0] * t
    ry = wy - SUN_DIR[1] * t

    edge_dx = np.clip((-0.55 - rx) / 0.24, 0.0, 1.0)
    edge_dy0 = np.clip((ry - 2.2) / 0.25, 0.0, 1.0)
    edge_dy1 = np.clip((13.6 - ry) / 0.25, 0.0, 1.0)
    oh_mask = edge_dx * edge_dy0 * edge_dy1

    # Glowing panoramic observatory skylight window spill cast across damp concrete tiles
    t_wall = np.where(wx > X_WALL, (wx - X_WALL) / max(1e-5, SUN_DIR[0]), -1.0)
    wy_w = wy - SUN_DIR[1] * t_wall
    wz_w = wz - SUN_DIR[2] * t_wall
    in_win = (t_wall > 0.0) & (wy_w > 4.2) & (wy_w < 11.8) & (wz_w > 0.72) & (wz_w < 2.75)
    mullions = (
        (np.abs(wy_w - 6.1) < 0.085)
        | (np.abs(wy_w - 8.0) < 0.085)
        | (np.abs(wy_w - 9.9) < 0.085)
        | (np.abs(wz_w - 1.85) < 0.065)
    )
    win_falloff = np.exp(-np.maximum(0.0, wx - X_WALL) * 0.44)
    skylight_spill = np.where(in_win & (~mullions), 1.0, 0.0) * win_falloff

    # Cast shadow from the rooftop bench with soft penumbra edges
    t_b = np.maximum(0.0, (0.46 - wz) * inv_sz)
    bx = wx - SUN_DIR[0] * t_b
    by = wy - SUN_DIR[1] * t_b
    bx_s = np.clip((bx - 1.30) / 0.10, 0.0, 1.0) * np.clip((3.20 - bx) / 0.10, 0.0, 1.0)
    by_s = np.clip((by - 7.10) / 0.10, 0.0, 1.0) * np.clip((7.76 - by) / 0.10, 0.0, 1.0)
    bench_sh = 1.0 - 0.44 * (bx_s * by_s)

    shadow = (1.0 - 0.52 * oh_mask) * bench_sh
    return shadow.astype(np.float32), skylight_spill.astype(np.float32)


# =============================================================================
# Stage 1 & 3 — Silky Sweeping Prism-Mint Aurora, Starry Sky & Coastal Valley
# =============================================================================

def paint_sky_aurora_valley(
    cam: Camera3D, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Paint Aria's midnight navy (#060b1c) -> twilight indigo -> periwinkle sky,
    silky sweeping folded prism-mint (#63ebb9) aurora ribbons (zero barcode noise!),
    crisp spectral starfield, and calm coastal valley with misty mountain ridges.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = xx / W
    v = yy / H
    hy = cam.horizon_y / H

    mottle = fbm_gouache(H, W, rng)
    dabs = flat_brush_dabs(H, W, rng, count=94)
    tooth = paper_tooth(H, W, rng)

    sky_t = np.clip(v / max(0.05, hy + 0.03), 0.0, 1.0)

    c_zenith = np.array([0.020, 0.039, 0.106], dtype=np.float32)
    c_mid = np.array([0.068, 0.128, 0.295], dtype=np.float32)
    c_peri = np.array([0.220, 0.355, 0.620], dtype=np.float32)
    c_glow = np.array([0.300, 0.680, 0.680], dtype=np.float32)

    s1 = np.power(sky_t, 1.24)
    s2 = np.power(np.clip((sky_t - 0.46) / 0.54, 0.0, 1.0), 1.55)
    sky = (
        c_zenith[None, None, :] * (1.0 - s1[..., None])
        + c_mid[None, None, :] * (s1 * (1.0 - s2))[..., None]
        + c_peri[None, None, :] * (s1 * s2)[..., None]
    )

    horiz_band = np.exp(-((v - (hy - 0.015)) / 0.090) ** 2) * np.exp(-((u - 0.64) / 0.56) ** 2)
    sky += horiz_band[..., None] * (c_glow[None, None, :] * 0.34)
    sky += np.exp(-((v - hy) / 0.044) ** 2)[..., None] * np.array([0.16, 0.25, 0.42], dtype=np.float32)

    # Delicate diagonal Milky Way star-dust veil
    mw_dist = v - (0.34 - 0.30 * u)
    mw = np.exp(-(mw_dist / 0.15) ** 2) * (0.45 + 0.55 * np.clip(mottle * 0.55 + 0.5, 0.0, 1.0)) * (1.0 - s2 * 0.85)
    sky += mw[..., None] * np.array([0.045, 0.072, 0.155], dtype=np.float32)

    sky += (mottle * 0.013 + dabs * 0.009 + tooth * 0.0045)[..., None]

    # -------------------------------------------------------------------------
    # Silky Sweeping Anime Prism-Mint (#63ebb9) Aurora Ribbons
    # Smooth luminous magnetic curtains with delicate upward rays & violet crown
    # -------------------------------------------------------------------------
    u_sheared = u + (v - 0.25) * 0.13
    silk_rays = (
        0.72
        + 0.16 * np.sin(u_sheared * 24.0 + 1.9 * np.sin(u_sheared * 5.8))
        + 0.11 * np.sin(u_sheared * 49.0 - 1.3 * np.cos(u_sheared * 10.5))
        + 0.07 * np.cos(u_sheared * 83.0 + 0.9 * np.sin(u_sheared * 16.0))
    ).astype(np.float32)
    soft_fold = resize_bicubic(rng.standard_normal((max(4, H // 180), max(8, W // 68))).astype(np.float32), W, H)
    curtain_silk = np.clip(silk_rays * (0.86 + 0.20 * soft_fold), 0.36, 1.30)

    aurora_rgb = np.zeros((H, W, 3), dtype=np.float32)

    # Three sweeping sinuous aurora ribbons arching gracefully across the starry indigo sky
    ribbons = [
        # Main sweeping hero ribbon (vivid prism-mint #63ebb9 lower hem -> aqua -> upper lavender-violet curtain)
        (
            0.312 - 0.145 * np.sin((u - 0.24) * 2.85) + 0.038 * np.sin(u * 6.2 - 0.8),
            0.195,
            0.017,
            1.05,
            0.16,
            1.02,
            np.array([0.388, 0.945, 0.740], dtype=np.float32),
            np.array([0.510, 0.440, 0.920], dtype=np.float32),
        ),
        # Upper sweeping secondary aurora veil arching gracefully across zenith
        (
            0.162 + 0.085 * np.sin((u - 0.14) * 3.1 + 0.9) + 0.022 * np.cos(u * 5.8),
            0.155,
            0.018,
            0.76,
            0.20,
            1.04,
            np.array([0.335, 0.900, 0.785], dtype=np.float32),
            np.array([0.640, 0.490, 0.950], dtype=np.float32),
        ),
        # Lower glowing horizon ribbon curling above the coastal mountains
        (
            0.398 - 0.052 * np.sin((u - 0.42) * 3.6 + 0.3),
            0.115,
            0.015,
            0.62,
            0.36,
            1.05,
            np.array([0.400, 0.950, 0.745], dtype=np.float32),
            np.array([0.350, 0.650, 0.900], dtype=np.float32),
        ),
    ]

    for curve_v, w_up, w_dn, strength, u_start, u_end, c_mint, c_top in ribbons:
        dv = v - curve_v
        sigma = np.where(dv < 0.0, w_up, w_dn)
        env = np.exp(-0.5 * (dv / sigma) ** 2)
        # Tall ethereal curtain vertical rays rising softly above the luminous lower edge
        ray_tail = np.where(dv < 0.0, np.exp(dv / (w_up * 1.15)), np.exp(-0.5 * (dv / w_dn) ** 2))
        arc_core = np.exp(-0.5 * (dv / (w_dn * 0.64)) ** 2)
        sweep = (
            np.clip((u - u_start) / 0.22, 0.0, 1.0)
            * np.clip((u_end - u) / 0.16, 0.0, 1.0)
            * (0.64 + 0.36 * np.sin(u * 5.2 + u_start * 9.0))
        )
        alt_t = np.clip(-dv / (w_up * 1.35), 0.0, 1.0)
        ribbon_col = c_mint[None, None, :] * (1.0 - alt_t[..., None]) + c_top[None, None, :] * alt_t[..., None]
        curt = (0.58 * env + 0.42 * ray_tail) * curtain_silk * sweep * strength
        contrib = curt[..., None] * ribbon_col * 0.76
        contrib += (arc_core * (0.72 + 0.28 * silk_rays) * sweep * strength)[..., None] * np.array(
            [0.74, 1.00, 0.91], dtype=np.float32
        ) * 0.56
        aurora_rgb += contrib

    aurora_mask = np.clip((hy + 0.01 - v) / 0.08, 0.0, 1.0)
    aurora_rgb *= aurora_mask[..., None]
    sky = screen_blend(sky, aurora_rgb * 0.88)
    sky += gaussian_blur_2d(aurora_rgb, 42.0) * 0.22

    # Crisp Multi-Magnitude Starry Sky
    star_layer = np.zeros((H, W, 3), dtype=np.float32)
    star_colors = [
        np.array([0.92, 0.96, 1.00], dtype=np.float32),
        np.array([0.72, 0.98, 0.91], dtype=np.float32),
        np.array([0.80, 0.88, 1.00], dtype=np.float32),
        np.array([1.00, 0.94, 0.86], dtype=np.float32),
    ]
    max_star_y = int(H * (hy - 0.02))
    for _ in range(480):
        sx = float(rng.uniform(4, W - 4))
        sy = float(rng.uniform(4, max_star_y))
        alt_vis = float(np.clip((hy - sy / H) / 0.34, 0.16, 1.0))
        mag = float(rng.power(0.32))
        bright = (0.16 + 0.84 * mag) * alt_vis
        col = star_colors[int(rng.integers(0, len(star_colors)))]
        ix, iy = int(sx), int(sy)
        rad = 1 if mag < 0.65 else (2 if mag < 0.90 else 3)
        y0, y1 = max(0, iy - rad - 1), min(H, iy + rad + 2)
        x0, x1 = max(0, ix - rad - 1), min(W, ix + rad + 2)
        gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        sig = 0.56 + 0.58 * mag
        spot = np.exp(-0.5 * ((gx - sx) ** 2 + (gy - sy) ** 2) / (sig * sig)) * bright
        star_layer[y0:y1, x0:x1] += spot[..., None] * col[None, None, :]
        if mag > 0.945:
            spike_len = int(9 + 11 * (mag - 0.945) / 0.055)
            for ds in range(-spike_len, spike_len + 1):
                w_sp = math.exp(-abs(ds) / (spike_len * 0.27)) * 0.56 * bright
                if 0 <= ix + ds < W:
                    star_layer[iy, ix + ds] += col * w_sp
                if 0 <= iy + ds < H:
                    star_layer[iy + ds, ix] += col * w_sp

    sky = np.clip(sky + star_layer, 0.0, 1.35)
    sky_for_refl = sky.copy()

    # Calm Distant Coastal Valley, Layered Mountain Ridges & Bay Reflections
    u_1d = np.arange(W, dtype=np.float32) / W
    r1 = (
        hy - 0.036
        - 0.032 * np.exp(-((u_1d - 0.84) / 0.19) ** 2)
        - 0.022 * np.exp(-((u_1d - 0.44) / 0.14) ** 2)
        - 0.010 * np.sin(u_1d * 19.0 + 0.4)
        - 0.005 * np.cos(u_1d * 41.0)
    )
    r2 = (
        hy - 0.012
        - 0.025 * np.exp(-((u_1d - 0.93) / 0.16) ** 2)
        - 0.018 * np.exp(-((u_1d - 0.36) / 0.12) ** 2)
        - 0.007 * np.sin(u_1d * 27.0 + 1.5)
        - 0.0035 * np.cos(u_1d * 57.0)
    )

    m1_mask = np.clip((v - r1[None, :]) / 0.0035, 0.0, 1.0)
    m2_mask = np.clip((v - r2[None, :]) / 0.0030, 0.0, 1.0)

    m1_col = np.array([0.056, 0.106, 0.232], dtype=np.float32)[None, None, :] + mottle[..., None] * 0.006
    m2_col = np.array([0.028, 0.056, 0.138], dtype=np.float32)[None, None, :] + mottle[..., None] * 0.005
    valley_mist = np.exp(-((v - (hy - 0.004)) / 0.022) ** 2)[..., None] * np.array([0.14, 0.32, 0.44], dtype=np.float32)

    sky = sky * (1.0 - m1_mask[..., None]) + (m1_col + valley_mist * 0.55) * m1_mask[..., None]
    sky = sky * (1.0 - m2_mask[..., None]) + (m2_col + valley_mist * 0.32) * m2_mask[..., None]

    # Calm Coastal Bay reflecting the silky prism-mint aurora
    bay_mask = np.clip((v - (hy + 0.003)) / 0.004, 0.0, 1.0)
    bay_rip = resize_bicubic(rng.standard_normal((H // 4, W // 48)).astype(np.float32), W, H) * 3.6
    bay_y_src = np.clip(2.0 * cam.horizon_y - yy + bay_rip, 0.0, H - 1.0)
    bay_refl = remap_bilinear(sky_for_refl, xx, bay_y_src)
    bay_base = np.array([0.018, 0.044, 0.108], dtype=np.float32)[None, None, :]
    bay_col = bay_base * 0.54 + bay_refl * 0.60
    sky = sky * (1.0 - bay_mask[..., None]) + bay_col * bay_mask[..., None]

    # Quiet twinkling distant coastal town lights along the bay shore
    for _ in range(120):
        tx = float(rng.uniform(W * 0.32, W * 0.99))
        ty = float(cam.horizon_y + rng.uniform(1.0, 13.0))
        if rng.random() < 0.32:
            ty -= float(rng.uniform(2.0, 11.0))
        tb = float(rng.uniform(0.35, 0.92))
        tcol = (
            np.array([0.98, 0.88, 0.66], dtype=np.float32)
            if rng.random() < 0.68
            else np.array([0.55, 0.96, 0.88], dtype=np.float32)
        )
        ix, iy = int(tx), int(ty)
        if 2 <= ix < W - 2 and 2 <= iy < H - 2:
            for dy_t in range(-2, 3):
                for dx_t in range(-2, 3):
                    w_t = math.exp(-0.5 * (dx_t * dx_t + dy_t * dy_t) / 1.12) * tb
                    sky[iy + dy_t, ix + dx_t] += tcol * w_t
            for step in range(2, 11):
                if iy + step < H:
                    sky[iy + step, ix] += tcol * (math.exp(-step * 0.28) * 0.28 * tb)

    fog_rgb = np.array([0.068, 0.132, 0.268], dtype=np.float32)
    return np.clip(sky, 0.0, 1.35), sky_for_refl, fog_rgb, mottle


# =============================================================================
# Stage 1 & 2 — Irregular Voronoi Concrete Rooftop Deck & Observatory Wall UV
# =============================================================================

def paint_voronoi_rooftop_uv(
    rng: np.random.Generator,
    x_span: tuple[float, float],
    y_span: tuple[float, float],
    tw: int = 1024,
    th: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Rain-slicked irregular Voronoi architectural concrete rooftop tiles + shallow puddle basins."""
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wx = x_span[0] + u * (x_span[1] - x_span[0])
    wy = y_span[0] + v * (y_span[1] - y_span[0])

    nx = resize_bicubic(rng.standard_normal((th // 32, tw // 32)).astype(np.float32), tw, th) * 0.055
    ny = resize_bicubic(rng.standard_normal((th // 32, tw // 32)).astype(np.float32), tw, th) * 0.055
    px = wx + nx
    py = wy + ny

    cell_x, cell_y = 1.12, 1.18
    ix = np.floor(px / cell_x).astype(np.int32)
    iy = np.floor(py / cell_y).astype(np.int32)

    jx_tab = rng.uniform(0.18, 0.82, size=256)
    jy_tab = rng.uniform(0.18, 0.82, size=256)
    tone_tab = rng.uniform(-0.028, 0.032, size=256)

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

    edge_d = (d2 - d1).astype(np.float32)
    slab_mask = np.clip((edge_d - 0.012) / 0.024, 0.0, 1.0)
    bevel_hi = np.clip((edge_d - 0.018) / 0.020, 0.0, 1.0) * np.clip((0.068 - edge_d) / 0.045, 0.0, 1.0)
    drain_trench = np.clip(1.0 - np.abs(wx - (-0.82)) / 0.085, 0.0, 1.0).astype(np.float32)

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=88)
    tooth = paper_tooth(th, tw, rng)
    sv = tone_tab[best_id].astype(np.float32)

    base = np.zeros((th, tw, 3), dtype=np.float32)
    base[:, :, 0] = 0.052 + sv * 0.45 + mottle * 0.011 + dabs * 0.008
    base[:, :, 1] = 0.082 + sv * 0.55 + mottle * 0.015 + dabs * 0.010
    base[:, :, 2] = 0.148 + sv * 0.62 + mottle * 0.018 + dabs * 0.012

    crevice_col = np.array([0.014, 0.024, 0.056], dtype=np.float32)
    rgb = slab_mask[..., None] * base + (1.0 - slab_mask[..., None]) * crevice_col
    rgb += bevel_hi[..., None] * np.array([0.026, 0.072, 0.078], dtype=np.float32)
    rgb = rgb * (1.0 - 0.65 * drain_trench[..., None]) + np.array([0.016, 0.028, 0.052], dtype=np.float32) * (
        0.65 * drain_trench[..., None]
    )

    # Stage 2: Unified SUN_DIR shadows & glowing skylight projection
    wz = np.full_like(wx, Z_DECK)
    shadow, skylight_spill = trace_rooftop_lighting(wx, wy, wz)
    rgb *= shadow[..., None]
    rgb += (shadow * slab_mask)[..., None] * np.array([0.020, 0.058, 0.068], dtype=np.float32)
    rgb += (skylight_spill * slab_mask)[..., None] * np.array([0.15, 0.33, 0.30], dtype=np.float32)

    # Warm bounce in deep shadows (lum < 0.28) + paper tooth
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    rgb += np.clip((0.28 - lum) / 0.28, 0.0, 1.0)[..., None] * np.array([0.012, 0.011, 0.009], dtype=np.float32)
    rgb += tooth[..., None] * 0.006

    # Organic shallow rain puddles with natural capillary damp-stone transition (no hard cartoon rings!)
    puddle_field = fbm_gouache(th, tw, rng, scales=((160, 150), (74, 72), (32, 30), (14, 14)))
    pool_bias = (
        0.54 * np.exp(-((wx - 0.68) ** 2 / 3.5 + (wy - 6.8) ** 2 / 14.5))
        + 0.48 * np.exp(-((wx - 2.38) ** 2 / 3.8 + (wy - 10.6) ** 2 / 16.0))
        + 0.34 * np.exp(-((wx - 0.85) ** 2 / 2.2 + (wy - 3.6) ** 2 / 4.2))
    ).astype(np.float32)
    margin = (
        np.clip((wx - (X_WALL + 0.25)) / 0.35, 0.0, 1.0)
        * np.clip((5.05 - wx) / 0.45, 0.0, 1.0)
        * np.clip((15.2 - wy) / 0.55, 0.0, 1.0)
    ).astype(np.float32)
    raw_p = (puddle_field * 0.46 + pool_bias - 0.15) * margin
    # Subtle wide damp darkening halo around puddles + smooth water film inside
    damp_halo = np.clip((raw_p + 0.12) * 3.2, 0.0, 1.0)
    puddle_mask = np.clip(raw_p * 3.8, 0.0, 1.0)
    rgb *= (1.0 - 0.24 * damp_halo[..., None] - 0.22 * puddle_mask[..., None])

    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.2)
    rgba[:, :, 3] = 1.0
    return rgba, puddle_mask


def paint_observatory_wall_uv(
    rng: np.random.Generator,
    y_span: tuple[float, float],
    z_span: tuple[float, float],
    tw: int = 1024,
    th: int = 512,
) -> np.ndarray:
    """
    Sleek dark poured-concrete observatory penthouse facade (x = X_WALL) with
    hand-crafted architectural interior behind its tall panoramic windows:
    wooden ceiling slats, pendant lamps, multi-shelf astronomy library with
    varied book silhouettes, glowing telemetry monitors, and a warm reading lamp!
    """
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wy = y_span[0] + u * (y_span[1] - y_span[0])
    wz = z_span[0] + v * (z_span[1] - z_span[0])

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=84)
    tooth = paper_tooth(th, tw, rng)

    base = np.zeros((th, tw, 3), dtype=np.float32)
    base[:, :, 0] = 0.036 + mottle * 0.008 + dabs * 0.006
    base[:, :, 1] = 0.056 + mottle * 0.011 + dabs * 0.008
    base[:, :, 2] = 0.112 + mottle * 0.014 + dabs * 0.010

    seam_v = np.exp(-(((wy - 2.4) % 1.9 - 0.95) ** 2) / 0.0005)
    seam_h = np.exp(-(((wz) % 1.15 - 0.575) ** 2) / 0.0005)
    seam = np.clip(seam_v + seam_h, 0.0, 1.0)
    rgb = base * (1.0 - 0.55 * seam[..., None]) + np.array([0.012, 0.020, 0.046], dtype=np.float32) * (0.55 * seam[..., None])

    # Architectural tie-rod holes
    tie_y = np.abs(((wy - 2.4) % 1.9) - 0.45)
    tie_y2 = np.abs(((wy - 2.4) % 1.9) - 1.45)
    tie_z = np.abs((wz % 1.15) - 0.28)
    ties = ((np.hypot(tie_y, tie_z) < 0.030) | (np.hypot(tie_y2, tie_z) < 0.030)).astype(np.float32)
    rgb *= (1.0 - 0.55 * ties[..., None])

    # Panoramic Observatory Studio Window Bay: y in [4.2, 11.8], z in [0.72, 2.76]
    in_win = (wy >= 4.20) & (wy <= 11.80) & (wz >= 0.72) & (wz <= 2.76)

    # Render crisp hand-drawn observatory studio interior onto a PyCairo UV buffer!
    s_int = cairo.ImageSurface(cairo.FORMAT_ARGB32, tw, th)
    ci = cairo.Context(s_int)
    ci.set_antialias(cairo.ANTIALIAS_BEST)

    def yz_to_px(y_m: float, z_m: float) -> tuple[float, float]:
        px = (y_m - y_span[0]) / (y_span[1] - y_span[0]) * (tw - 1)
        py = (z_m - z_span[0]) / (z_span[1] - z_span[0]) * (th - 1)
        return px, py

    # Note: in UV space, v=0 (row 0) corresponds to z = z_span[0] (bottom = 0.0), v=1 (row th-1) to z_top!
    # Interior warm-cyan observatory backdrop gradient
    x_w0, y_w0 = yz_to_px(4.20, 0.72)
    x_w1, y_w1 = yz_to_px(11.80, 2.76)
    g_bg = cairo.LinearGradient(x_w0, y_w1, x_w0, y_w0)
    g_bg.add_color_stop_rgba(0.0, 0.14, 0.34, 0.44, 1.0)     # top ceiling cool indigo-cyan
    g_bg.add_color_stop_rgba(0.50, 0.22, 0.50, 0.56, 1.0)    # mid studio cyan-mint light
    g_bg.add_color_stop_rgba(1.0, 0.16, 0.28, 0.38, 1.0)     # lower desk space
    ci.set_source(g_bg)
    ci.rectangle(x_w0, y_w0, x_w1 - x_w0, y_w1 - y_w0)
    ci.fill()

    # Glowing overhead ceiling skylight trough + architectural steel ceiling beams
    for beam_y in (4.95, 6.10, 7.25, 8.40, 9.55, 10.70):
        bx0, by0 = yz_to_px(beam_y - 0.05, 2.48)
        bx1, by1 = yz_to_px(beam_y + 0.05, 2.76)
        ci.set_source_rgba(0.06, 0.12, 0.22, 0.78)
        ci.rectangle(bx0, by0, bx1 - bx0, by1 - by0)
        ci.fill()

    # Recessed luminous prism-mint ceiling light panel strips (casting soft glow downward)
    for lx_m in (5.5, 7.8, 10.2):
        lx, ly = yz_to_px(lx_m, 2.58)
        g_cl = cairo.RadialGradient(lx, ly, 4.0, lx, ly, 165.0)
        g_cl.add_color_stop_rgba(0.0, 0.74, 0.99, 0.92, 0.85)
        g_cl.add_color_stop_rgba(0.35, 0.36, 0.86, 0.78, 0.42)
        g_cl.add_color_stop_rgba(1.0, 0.18, 0.48, 0.58, 0.0)
        ci.set_source(g_cl)
        ci.arc(lx, ly, 165.0, 0, math.tau)
        ci.fill()

    # Cozy warm amber architectural desk lamp glow inside the center bay (y ~ 7.35, z ~ 1.48)
    dx_l, dy_l = yz_to_px(7.35, 1.46)
    g_warm = cairo.RadialGradient(dx_l, dy_l, 5.0, dx_l, dy_l, 155.0)
    g_warm.add_color_stop_rgba(0.0, 1.00, 0.92, 0.74, 0.90)
    g_warm.add_color_stop_rgba(0.38, 0.88, 0.72, 0.48, 0.44)
    g_warm.add_color_stop_rgba(1.0, 0.36, 0.46, 0.48, 0.0)
    ci.set_source(g_warm)
    ci.arc(dx_l, dy_l, 155.0, 0, math.tau)
    ci.fill()

    # Sleek hanging pendant lamp fixtures descending cleanly from ceiling (z = 2.76 down to z = 2.15)
    for py_m in (5.35, 8.95):
        px_c, py_top = yz_to_px(py_m, 2.76)
        _, py_sh = yz_to_px(py_m, 2.18)
        ci.set_source_rgba(0.05, 0.09, 0.18, 0.88)
        ci.set_line_width(2.0)
        ci.move_to(px_c, py_top)
        ci.line_to(px_c, py_sh)
        ci.stroke()
        # Conical modern pendant shade
        ci.move_to(px_c - 18, py_sh - 10)
        ci.line_to(px_c + 18, py_sh - 10)
        ci.line_to(px_c + 7, py_sh + 6)
        ci.line_to(px_c - 7, py_sh + 6)
        ci.close_path()
        ci.fill()
        # Soft warm-mint bulb glow under shade
        g_pb = cairo.RadialGradient(px_c, py_sh - 14, 2.0, px_c, py_sh - 24, 65.0)
        g_pb.add_color_stop_rgba(0.0, 0.96, 0.98, 0.88, 0.72)
        g_pb.add_color_stop_rgba(1.0, 0.42, 0.85, 0.78, 0.0)
        ci.set_source(g_pb)
        ci.arc(px_c, py_sh - 24, 65.0, 0, math.tau)
        ci.fill()

    # Bay 1 (y = 4.35..5.92): Multi-shelf astronomy library with realistic varied book rows!
    sx0, sz0 = yz_to_px(4.38, 0.72)
    sx1, sz1 = yz_to_px(5.88, 2.12)
    ci.set_source_rgba(0.045, 0.088, 0.165, 0.84)
    # Vertical bookcase stiles
    for sy_m in (4.38, 5.12, 5.84):
        bx, _ = yz_to_px(sy_m, 0.72)
        ci.rectangle(bx, sz0, 4.5, sz1 - sz0)
        ci.fill()
    # 4 horizontal bookshelves populated with organically varied books & star atlases
    for sh_z in (0.98, 1.32, 1.66, 2.00):
        bx0, bz = yz_to_px(4.38, sh_z)
        bx1, _ = yz_to_px(5.88, sh_z)
        ci.set_source_rgba(0.042, 0.082, 0.155, 0.86)
        ci.rectangle(bx0, bz - 3.0, bx1 - bx0, 5.5)
        ci.fill()
        cur_x = bx0 + 6.0
        while cur_x < bx1 - 9.0:
            bw = float(rng.uniform(4.0, 9.5))
            if rng.random() < 0.16:
                cur_x += bw + 4.0  # natural shelf breathing gap
                continue
            bh = float(rng.uniform(14.0, 28.0))
            ci.set_source_rgba(0.055, 0.105, 0.195, 0.78)
            ci.rectangle(cur_x, bz + 2.5, bw, bh)
            ci.fill()
            cur_x += bw + 1.4

    # Bay 2 & 3 (y = 6.25..9.75): Sleek observatory desk, dual glowing star-chart monitors & desk lamp!
    dx0, dz_top = yz_to_px(6.25, 1.18)
    dx1, dz_bot = yz_to_px(9.75, 0.72)
    ci.set_source_rgba(0.042, 0.078, 0.148, 0.88)
    ci.rectangle(dx0, dz_bot, dx1 - dx0, dz_top - dz_bot)
    ci.fill()
    # Desk surface highlight catching warm lamp
    ci.set_source_rgba(0.75, 0.82, 0.74, 0.42)
    ci.rectangle(dx0, dz_top - 3.0, dx1 - dx0, 3.0)
    ci.fill()
    # Dual curved telemetry monitors displaying subtle mint star-chart curves
    for my_m in (8.25, 9.05):
        mx0, mz0 = yz_to_px(my_m - 0.30, 1.28)
        mx1, mz1 = yz_to_px(my_m + 0.30, 1.66)
        # Stand
        ci.set_source_rgba(0.032, 0.062, 0.125, 0.92)
        ci.rectangle(0.5 * (mx0 + mx1) - 4, dz_top, 8, mz0 - dz_top)
        ci.fill()
        # Bezel & dark glass telemetry screen with glowing prism-mint orbital star-chart traces
        ci.rectangle(mx0, mz0, mx1 - mx0, mz1 - mz0)
        ci.fill()
        ci.set_source_rgba(0.08, 0.26, 0.32, 0.95)
        ci.rectangle(mx0 + 3, mz0 + 3, (mx1 - mx0) - 6, (mz1 - mz0) - 6)
        ci.fill()
        # Subtle glowing star-chart ellipse & telemetry wave inside monitor
        mcx, mcy = 0.5 * (mx0 + mx1), 0.5 * (mz0 + mz1)
        ci.set_source_rgba(0.45, 0.96, 0.84, 0.85)
        ci.set_line_width(1.3)
        ci.arc(mcx, mcy + 4, 9.5, 0, math.tau)
        ci.stroke()
        ci.move_to(mx0 + 6, mz0 + 10)
        ci.line_to(mx1 - 8, mz0 + 10)
        ci.move_to(mx0 + 6, mz0 + 15)
        ci.line_to(mx0 + 22, mz0 + 15)
        ci.stroke()
    # Architect desk lamp silhouette on left of desk (y ~ 7.10)
    alx, alz = yz_to_px(7.05, 1.18)
    _, alz_head = yz_to_px(7.05, 1.56)
    ci.set_source_rgba(0.035, 0.065, 0.130, 0.92)
    ci.set_line_width(3.2)
    ci.move_to(alx, alz)
    ci.line_to(alx - 8, alz + 24)
    ci.line_to(alx + 14, alz_head)
    ci.stroke()
    ci.arc(alx + 18, alz_head - 2, 10.0, 0.2, math.pi * 0.95)
    ci.fill()

    # Bay 4 (y = 10.05..11.65): Subtle indoor observatory telescope pier / tall potted ficus silhouette
    px_tr, pz_bot = yz_to_px(10.85, 0.72)
    _, pz_top = yz_to_px(10.85, 1.95)
    ci.set_source_rgba(0.048, 0.095, 0.175, 0.74)
    ci.rectangle(px_tr - 14, pz_bot, 28, pz_top - pz_bot)
    ci.fill()

    # Diagonal soft glass reflection sheen across panoramic panes
    ci.set_source_rgba(0.45, 0.94, 0.82, 0.12)
    ci.move_to(x_w0 + 80, y_w0)
    ci.line_to(x_w0 + 230, y_w0)
    ci.line_to(x_w0 + 380, y_w1)
    ci.line_to(x_w0 + 230, y_w1)
    ci.close_path()
    ci.fill()

    s_int.flush()
    ibuf = np.frombuffer(s_int.get_data(), dtype=np.uint8).reshape(th, tw, 4)
    interior = np.stack([ibuf[:, :, 2], ibuf[:, :, 1], ibuf[:, :, 0]], axis=-1).astype(np.float32) / 255.0

    # Window dark steel perimeter frames & mullions
    win_frame = (
        ((wy >= 4.08) & (wy <= 11.92) & (wz >= 0.62) & (wz <= 2.86) & (~in_win))
        | (in_win & ((np.abs(wy - 6.10) < 0.060) | (np.abs(wy - 8.00) < 0.060) | (np.abs(wy - 9.90) < 0.060)))
        | (in_win & (np.abs(wz - 1.85) < 0.048))
    )

    rgb = np.where(in_win[..., None], interior, rgb)
    frame_col = np.array([0.016, 0.028, 0.060], dtype=np.float32)
    rgb = np.where(win_frame[..., None], frame_col, rgb)

    # Metallic sill catching faint aurora reflection
    sill = ((wy >= 4.05) & (wy <= 11.95) & (wz >= 0.58) & (wz <= 0.65)).astype(np.float32)
    rgb += sill[..., None] * np.array([0.11, 0.25, 0.26], dtype=np.float32)

    # Under-eave overhang shadow
    under_eave = np.clip((wz - (Z_ROOF - 0.35)) / 0.38, 0.0, 1.0)
    rgb *= (1.0 - 0.45 * under_eave[..., None])

    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    rgb += np.clip((0.28 - lum) / 0.28, 0.0, 1.0)[..., None] * np.array([0.012, 0.010, 0.009], dtype=np.float32)
    rgb += tooth[..., None] * 0.006

    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.35)
    rgba[:, :, 3] = 1.0
    return rgba


# =============================================================================
# Stage 3 — 2x Supersampled PyCairo Helpers & Architectural / Organic Props
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
        np.asarray(
            Image.fromarray(rgba_2x[:, :, 3], mode="L").resize((W, H), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
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
    fade_dist: float = 20.0,
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
    n = len(dense) - 1
    for i in range(n):
        frac = (i + 0.5) / max(1, n)
        wt = (0.38 + 0.62 * math.sin(frac * math.pi)) if taper else 1.0
        ctx.set_line_width(max(0.45, w_eff * wt))
        ctx.move_to(*dense[i])
        ctx.line_to(*dense[i + 1])
        ctx.stroke()


def fill_3d_quad(
    ctx: cairo.Context,
    cam: Camera3D,
    pts3: list[np.ndarray] | np.ndarray,
    rgba: tuple[float, float, float, float],
    stroke_rgba: tuple[float, float, float, float] | None = None,
    stroke_w: float = 1.2,
) -> bool:
    arr = np.asarray(pts3, dtype=np.float64)
    poly: list[tuple[float, float]] = []
    n = len(arr)
    for i in range(n):
        cl = cam.clip_segment_3d(arr[i], arr[(i + 1) % n])
        if cl is None:
            continue
        ua, va, _ = cam.project(cl[0])
        ub, vb, _ = cam.project(cl[1])
        if not poly or math.hypot(poly[-1][0] - float(ua), poly[-1][1] - float(va)) > 0.5:
            poly.append((float(ua), float(va)))
        poly.append((float(ub), float(vb)))
    if len(poly) < 3:
        return False
    ctx.set_source_rgba(*rgba)
    ctx.move_to(*poly[0])
    for p in poly[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    if stroke_rgba is not None:
        ctx.fill_preserve()
        ctx.set_source_rgba(*stroke_rgba)
        ctx.set_line_width(stroke_w)
        ctx.stroke()
    else:
        ctx.fill()
    return True


def stroke_3d_seg(
    ctx: cairo.Context,
    cam: Camera3D,
    p0: np.ndarray,
    p1: np.ndarray,
    base_w: float,
    rgba: tuple[float, float, float, float],
    rng: np.random.Generator,
    use_catmull: bool = False,
    taper: bool = False,
    tremor: float = 0.14,
) -> None:
    cl = cam.clip_segment_3d(np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64))
    if cl is None:
        return
    a, b = cl
    ua, va, za = cam.project(a)
    ub, vb, zb = cam.project(b)
    z_mid = 0.5 * (float(za) + float(zb))
    if use_catmull:
        mid = 0.5 * (a + b)
        um, vm, _ = cam.project(mid)
        stroke_catmull_rom(
            ctx,
            [(float(ua), float(va)), (float(um), float(vm)), (float(ub), float(vb))],
            base_w=base_w,
            z_depth=z_mid,
            rgba=rgba,
            rng=rng,
            taper=taper,
            tremor=tremor,
        )
    else:
        decay = float(np.clip(1.0 / (1.0 + z_mid / 9.0), 0.28, 1.0))
        ctx.set_source_rgba(*rgba)
        ctx.set_line_width(max(0.55, base_w * decay))
        ctx.move_to(float(ua), float(va))
        ctx.line_to(float(ub), float(vb))
        ctx.stroke()


def draw_observatory_dome_and_penthouse_shell(
    ctx: cairo.Context, cam: Camera3D, rng: np.random.Generator
) -> None:
    """
    Draw the sleek architectural observatory dome perched visibly atop the rear
    penthouse roof deck overlooking the aurora sky, plus the watertight
    cantilevered concrete roof slab, recessed soffit glow strip, and perimeter curbs.
    """
    # 1. Iconic Observatory Dome rising prominently above the rear penthouse roof!
    # Positioned at x=-2.25, y=12.4, z_base=Z_ROOF+0.15 so the entire illuminated hemisphere
    # with its open observation slit is cleanly framed against the aurora sky!
    dome_c = np.array([-2.32, 12.2, Z_ROOF + 0.18], dtype=np.float64)
    dome_r = 1.62

    uc, vc, zc = cam.project(dome_c + np.array([0.0, 0.0, 0.30]))
    u_top, v_top, _ = cam.project(dome_c + np.array([0.0, 0.0, 0.30 + dome_r]))
    u_rt, v_rt, _ = cam.project(dome_c + np.array([dome_r, 0.0, 0.30]))
    rx_px = abs(float(u_rt) - float(uc))
    ry_px = abs(float(vc) - float(v_top))

    # Cylindrical concrete/steel dome drum pedestal
    drum_h = ry_px * 0.28
    g_drum = cairo.LinearGradient(float(uc) - rx_px, float(vc), float(uc) + rx_px, float(vc))
    g_drum.add_color_stop_rgba(0.0, 0.022, 0.042, 0.095, 1.0)
    g_drum.add_color_stop_rgba(0.75, 0.068, 0.125, 0.228, 1.0)
    g_drum.add_color_stop_rgba(1.0, 0.140, 0.310, 0.365, 1.0)
    ctx.set_source(g_drum)
    ctx.rectangle(float(uc) - rx_px, float(vc) - 2.0, rx_px * 2.0, drum_h + 18.0)
    ctx.fill()

    # Sleek metallic-indigo hemispherical dome catching luminous prism-mint aurora reflection!
    ctx.save()
    ctx.translate(float(uc), float(vc))
    ctx.scale(rx_px, ry_px)
    g_dome = cairo.RadialGradient(0.46, -0.58, 0.06, 0.05, -0.10, 1.12)
    g_dome.add_color_stop_rgba(0.0, 0.44, 0.90, 0.80, 1.0)      # bright prism-mint aurora zenith sheen
    g_dome.add_color_stop_rgba(0.28, 0.18, 0.38, 0.54, 1.0)     # periwinkle-teal metallic midtone
    g_dome.add_color_stop_rgba(0.68, 0.052, 0.095, 0.195, 1.0)
    g_dome.add_color_stop_rgba(1.0, 0.022, 0.042, 0.098, 1.0)
    ctx.set_source(g_dome)
    ctx.arc(0.0, 0.0, 1.0, math.pi, 2.0 * math.pi)
    ctx.close_path()
    ctx.fill()

    # Architectural panel ribs curving across dome
    ctx.set_source_rgba(0.025, 0.050, 0.110, 0.55)
    ctx.set_line_width(0.012)
    for rib_x in (-0.55, -0.18, 0.58):
        ctx.move_to(rib_x, 0.0)
        ctx.curve_to(rib_x * 0.95, -0.45, rib_x * 0.45, -0.84, 0.0, -0.995)
        ctx.stroke()

    # Open observation slit facing upper-right aurora + glowing mint-cyan interior rim
    ctx.set_source_rgba(0.014, 0.026, 0.064, 0.94)
    ctx.set_line_width(0.145)
    ctx.move_to(0.24, -0.02)
    ctx.curve_to(0.25, -0.52, 0.12, -0.88, -0.05, -0.99)
    ctx.stroke()

    # Subtle glowing mint rim along slit shutter edge
    ctx.set_source_rgba(0.55, 0.98, 0.86, 0.78)
    ctx.set_line_width(0.024)
    ctx.move_to(0.32, -0.02)
    ctx.curve_to(0.33, -0.50, 0.19, -0.86, 0.03, -0.98)
    ctx.stroke()

    # Crisp aurora backlight rim around right-upper dome silhouette (delicate specular edge)
    ctx.set_source_rgba(0.58, 0.96, 0.86, 0.52)
    ctx.set_line_width(0.009)
    ctx.arc(0.0, 0.0, 0.996, math.pi * 1.48, math.pi * 1.98)
    ctx.stroke()
    ctx.restore()

    # Base collar ring & tiny red/mint status indicator on observatory dome
    ctx.set_source_rgba(0.45, 0.92, 0.80, 0.55)
    ctx.set_line_width(2.0)
    ctx.move_to(float(uc) - rx_px, float(vc))
    ctx.line_to(float(uc) + rx_px, float(vc))
    ctx.stroke()

    # 2. Cantilevered Dark Concrete Penthouse Roof Slab (watertight overlap with wall at z = Z_ROOF)
    soffit = [
        np.array([X_WALL - 0.08, 2.15, Z_ROOF - 0.02]),
        np.array([-0.52, 2.15, Z_ROOF + 0.04]),
        np.array([-0.52, 13.7, Z_ROOF + 0.04]),
        np.array([X_WALL - 0.08, 13.7, Z_ROOF - 0.02]),
    ]
    fill_3d_quad(ctx, cam, soffit, (0.024, 0.044, 0.098, 1.0))

    # Recessed LED soffit architectural wash strip
    eave_strip = [
        np.array([-0.90, 4.1, Z_ROOF + 0.01]),
        np.array([-0.80, 4.1, Z_ROOF + 0.02]),
        np.array([-0.80, 11.9, Z_ROOF + 0.02]),
        np.array([-0.90, 11.9, Z_ROOF + 0.01]),
    ]
    fill_3d_quad(ctx, cam, eave_strip, (0.48, 0.95, 0.82, 0.88))

    # Fascia right side & slight top roof parapet edge
    fascia = [
        np.array([-0.52, 2.15, Z_ROOF + 0.04]),
        np.array([-0.52, 13.7, Z_ROOF + 0.04]),
        np.array([-0.52, 13.7, Z_ROOF + 0.36]),
        np.array([-0.52, 2.15, Z_ROOF + 0.36]),
    ]
    fill_3d_quad(ctx, cam, fascia, (0.036, 0.064, 0.132, 1.0), stroke_rgba=(0.014, 0.026, 0.060, 0.92), stroke_w=1.2)

    # Crisp prism-mint aurora rim highlight along roof cornice
    stroke_3d_seg(
        ctx, cam, np.array([-0.52, 2.15, Z_ROOF + 0.36]), np.array([-0.52, 13.7, Z_ROOF + 0.36]),
        base_w=2.4, rgba=(0.48, 0.94, 0.80, 0.84), rng=rng, use_catmull=True, tremor=0.12,
    )

    # Front & rear penthouse mass enclosures
    front_face = [
        np.array([-4.8, 2.15, 0.0]),
        np.array([X_WALL, 2.15, 0.0]),
        np.array([X_WALL, 2.15, Z_ROOF + 0.02]),
        np.array([-4.8, 2.15, Z_ROOF - 0.05]),
    ]
    fill_3d_quad(ctx, cam, front_face, (0.020, 0.038, 0.086, 1.0))

    rear_corner = [
        np.array([-3.6, 13.7, 0.0]),
        np.array([X_WALL, 13.7, 0.0]),
        np.array([X_WALL, 13.7, Z_ROOF + 0.02]),
        np.array([-3.6, 13.7, Z_ROOF + 0.02]),
    ]
    fill_3d_quad(ctx, cam, rear_corner, (0.026, 0.050, 0.110, 1.0))

    # Low concrete perimeter curbs
    far_curb_front = [
        np.array([-1.15, 15.55, 0.0]),
        np.array([5.35, 15.55, 0.0]),
        np.array([5.35, 15.55, 0.19]),
        np.array([-1.15, 15.55, 0.19]),
    ]
    far_curb_top = [
        np.array([-1.15, 15.55, 0.19]),
        np.array([5.35, 15.55, 0.19]),
        np.array([5.35, 15.88, 0.19]),
        np.array([-1.15, 15.88, 0.19]),
    ]
    fill_3d_quad(ctx, cam, far_curb_front, (0.028, 0.052, 0.112, 1.0))
    fill_3d_quad(ctx, cam, far_curb_top, (0.082, 0.172, 0.242, 0.95))

    right_curb_in = [
        np.array([5.25, 2.0, 0.0]),
        np.array([5.25, 15.65, 0.0]),
        np.array([5.25, 15.65, 0.19]),
        np.array([5.25, 2.0, 0.19]),
    ]
    right_curb_top = [
        np.array([5.25, 2.0, 0.19]),
        np.array([5.58, 2.0, 0.19]),
        np.array([5.58, 15.65, 0.19]),
        np.array([5.25, 15.65, 0.19]),
    ]
    fill_3d_quad(ctx, cam, right_curb_in, (0.026, 0.048, 0.108, 1.0))
    fill_3d_quad(ctx, cam, right_curb_top, (0.076, 0.162, 0.228, 0.95))


def draw_perimeter_railing(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    mirrored: bool = False,
    alpha_scale: float = 1.0,
) -> None:
    z_sign = -1.0 if mirrored else 1.0
    z_base = 0.18 * z_sign
    z_top = 1.08 * z_sign
    z_mid = 0.62 * z_sign

    def pt(x: float, y: float, z: float) -> np.ndarray:
        return np.array([x, y, z], dtype=np.float64)

    far_xs = np.linspace(-1.05, 5.28, 7)
    for i in range(len(far_xs) - 1):
        quad = [
            pt(far_xs[i], 15.62, z_base + 0.05 * z_sign),
            pt(far_xs[i + 1], 15.62, z_base + 0.05 * z_sign),
            pt(far_xs[i + 1], 15.62, z_top - 0.08 * z_sign),
            pt(far_xs[i], 15.62, z_top - 0.08 * z_sign),
        ]
        fill_3d_quad(ctx, cam, quad, (0.20, 0.56, 0.62, 0.13 * alpha_scale))

    right_ys = np.linspace(3.0, 15.62, 8)
    for i in range(len(right_ys) - 1):
        quad = [
            pt(5.28, right_ys[i], z_base + 0.05 * z_sign),
            pt(5.28, right_ys[i + 1], z_base + 0.05 * z_sign),
            pt(5.28, right_ys[i + 1], z_top - 0.08 * z_sign),
            pt(5.28, right_ys[i], z_top - 0.08 * z_sign),
        ]
        fill_3d_quad(ctx, cam, quad, (0.18, 0.48, 0.58, 0.12 * alpha_scale))

    post_col = (0.028, 0.052, 0.115, 0.94 * alpha_scale)
    rail_hi = (0.46, 0.94, 0.80, 0.82 * alpha_scale)
    wire_col = (0.28, 0.56, 0.68, 0.55 * alpha_scale)

    for x in far_xs:
        stroke_3d_seg(ctx, cam, pt(x, 15.62, z_base), pt(x, 15.62, z_top), 3.6, post_col, rng)
        stroke_3d_seg(ctx, cam, pt(x + 0.02, 15.62, z_base), pt(x + 0.02, 15.62, z_top), 1.1, rail_hi, rng)

    stroke_3d_seg(ctx, cam, pt(-1.05, 15.62, z_top), pt(5.28, 15.62, z_top), 4.4, post_col, rng)
    stroke_3d_seg(
        ctx, cam, pt(-1.05, 15.62, z_top + 0.015 * z_sign), pt(5.28, 15.62, z_top + 0.015 * z_sign),
        1.7, rail_hi, rng, use_catmull=not mirrored, tremor=0.10,
    )
    stroke_3d_seg(ctx, cam, pt(-1.05, 15.62, z_mid), pt(5.28, 15.62, z_mid), 1.2, wire_col, rng)

    for y in right_ys:
        stroke_3d_seg(ctx, cam, pt(5.28, y, z_base), pt(5.28, y, z_top), 4.0, post_col, rng)
        stroke_3d_seg(ctx, cam, pt(5.26, y, z_base), pt(5.26, y, z_top), 1.2, rail_hi, rng)

    stroke_3d_seg(ctx, cam, pt(5.28, 2.4, z_top), pt(5.28, 15.62, z_top), 4.8, post_col, rng)
    stroke_3d_seg(
        ctx, cam, pt(5.28, 2.4, z_top + 0.018 * z_sign), pt(5.28, 15.62, z_top + 0.018 * z_sign),
        1.8, rail_hi, rng, use_catmull=not mirrored, tremor=0.10,
    )
    stroke_3d_seg(ctx, cam, pt(5.28, 2.4, z_mid), pt(5.28, 15.62, z_mid), 1.2, wire_col, rng)


def draw_rooftop_bench(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    mirrored: bool = False,
    alpha_scale: float = 1.0,
) -> None:
    z_sign = -1.0 if mirrored else 1.0

    def p(x: float, y: float, z: float) -> np.ndarray:
        return np.array([x, y, z * z_sign], dtype=np.float64)

    steel_col = (0.022, 0.040, 0.090, 0.96 * alpha_scale)
    steel_hi = (0.34, 0.72, 0.72, 0.55 * alpha_scale)

    for lx in (1.52, 2.98):
        stroke_3d_seg(ctx, cam, p(lx, 7.24, 0.0), p(lx, 7.24, 0.44), 5.6, steel_col, rng)
        stroke_3d_seg(ctx, cam, p(lx, 7.70, 0.0), p(lx, 7.70, 0.44), 5.2, steel_col, rng)
        stroke_3d_seg(ctx, cam, p(lx, 7.24, 0.43), p(lx, 7.70, 0.43), 4.6, steel_col, rng)
        stroke_3d_seg(ctx, cam, p(lx, 7.24, 0.14), p(lx, 7.70, 0.14), 3.0, steel_col, rng)
        stroke_3d_seg(ctx, cam, p(lx - 0.015, 7.24, 0.04), p(lx - 0.015, 7.24, 0.43), 1.3, steel_hi, rng)

    stroke_3d_seg(ctx, cam, p(1.52, 7.47, 0.39), p(2.98, 7.47, 0.39), 3.6, steel_col, rng)

    slat_ys = [(7.21, 7.31), (7.34, 7.44), (7.47, 7.57), (7.60, 7.71)]
    for idx, (y0, y1) in enumerate(slat_ys):
        front = [p(1.36, y0, 0.42), p(3.14, y0, 0.42), p(3.14, y0, 0.465), p(1.36, y0, 0.465)]
        fill_3d_quad(ctx, cam, front, (0.042, 0.056, 0.105, 0.96 * alpha_scale))
        top = [p(1.36, y0, 0.465), p(3.14, y0, 0.465), p(3.14, y1, 0.465), p(1.36, y1, 0.465)]
        tint = 0.02 * idx
        fill_3d_quad(
            ctx, cam, top,
            (0.088 + tint * 0.4, 0.175 + tint * 1.1, 0.225 + tint * 0.8, 0.96 * alpha_scale),
            stroke_rgba=(0.018, 0.032, 0.072, 0.72 * alpha_scale),
            stroke_w=0.95,
        )
        stroke_3d_seg(
            ctx, cam, p(1.38, y0 + 0.01, 0.468), p(3.12, y0 + 0.01, 0.468),
            1.4, (0.48, 0.94, 0.80, 0.62 * alpha_scale), rng, use_catmull=not mirrored, tremor=0.10,
        )

    # Small warm-glowing portable astronomy desk lantern resting on bench
    lan_c = p(1.66, 7.45, 0.465)
    lan_top = p(1.66, 7.45, 0.67)
    ub, vb, zb = cam.project(lan_c)
    ut, vt, zt = cam.project(lan_top)
    if zb > Z_NEAR and zt > Z_NEAR:
        cx_l, cy_l = float(0.5 * (ub + ut)), float(0.5 * (vb + vt))
        h_l = abs(float(vb - vt))
        w_l = h_l * 0.58
        g_h = cairo.RadialGradient(cx_l, cy_l, 1.0, cx_l, cy_l, h_l * 2.5)
        g_h.add_color_stop_rgba(0.0, 0.98, 0.94, 0.76, 0.62 * alpha_scale)
        g_h.add_color_stop_rgba(0.38, 0.48, 0.94, 0.78, 0.26 * alpha_scale)
        g_h.add_color_stop_rgba(1.0, 0.20, 0.52, 0.65, 0.0)
        ctx.set_source(g_h)
        ctx.arc(cx_l, cy_l, h_l * 2.5, 0, math.tau)
        ctx.fill()
        ctx.set_source_rgba(0.96, 0.96, 0.84, 0.94 * alpha_scale)
        ctx.rectangle(cx_l - w_l * 0.44, min(float(vt), float(vb)) + h_l * 0.16, w_l * 0.88, h_l * 0.68)
        ctx.fill()
        ctx.set_source_rgba(0.025, 0.045, 0.098, 0.96 * alpha_scale)
        ctx.rectangle(cx_l - w_l * 0.54, min(float(vt), float(vb)), w_l * 1.08, h_l * 0.18)
        ctx.rectangle(cx_l - w_l * 0.54, max(float(vt), float(vb)) - h_l * 0.16, w_l * 1.08, h_l * 0.16)
        ctx.fill()


def draw_tripod_telescope(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    mirrored: bool = False,
    alpha_scale: float = 1.0,
) -> None:
    z_sign = -1.0 if mirrored else 1.0
    cx, cy = 2.82, 11.45

    def p(x: float, y: float, z: float) -> np.ndarray:
        return np.array([x, y, z * z_sign], dtype=np.float64)

    head = p(cx, cy, 1.32)
    legs = [
        p(cx - 0.34, cy - 0.28, 0.0),
        p(cx + 0.36, cy - 0.22, 0.0),
        p(cx - 0.04, cy + 0.38, 0.0),
    ]
    dark_col = (0.022, 0.042, 0.095, 0.96 * alpha_scale)
    rim_col = (0.48, 0.94, 0.82, 0.72 * alpha_scale)
    for lf in legs:
        stroke_3d_seg(ctx, cam, lf, head, 4.2, dark_col, rng)
        stroke_3d_seg(ctx, cam, lf + np.array([0.012, 0, 0]), head + np.array([0.012, 0, 0]), 1.1, rim_col, rng)

    stroke_3d_seg(ctx, cam, head, p(cx - 0.22, cy - 0.12, 1.02), 3.2, dark_col, rng)

    eye_end = p(cx - 0.26, cy - 0.30, 1.14)
    obj_end = p(cx + 0.44, cy + 0.42, 1.78)
    stroke_3d_seg(ctx, cam, eye_end, obj_end, 11.5, (0.085, 0.155, 0.265, 0.98 * alpha_scale), rng)
    stroke_3d_seg(
        ctx, cam,
        eye_end + np.array([0.0, 0.0, 0.045 * z_sign]),
        obj_end + np.array([0.0, 0.0, 0.045 * z_sign]),
        2.3, (0.58, 0.96, 0.86, 0.88 * alpha_scale), rng, use_catmull=not mirrored, tremor=0.08,
    )
    stroke_3d_seg(
        ctx, cam,
        eye_end * 0.28 + obj_end * 0.72,
        obj_end + np.array([0.08, 0.08, 0.07 * z_sign]),
        13.2, dark_col, rng,
    )
    stroke_3d_seg(
        ctx, cam,
        eye_end + np.array([0.0, 0.0, 0.11 * z_sign]),
        eye_end * 0.45 + obj_end * 0.55 + np.array([0.0, 0.0, 0.11 * z_sign]),
        3.2, dark_col, rng,
    )


def draw_rooftop_planter_and_grasses(
    ctx: cairo.Context, cam: Camera3D, rng: np.random.Generator
) -> None:
    """Architectural concrete planter with bevelled rim & tapered seeded cubic-Bezier ornamental grasses."""
    front = [
        np.array([4.55, 9.4, 0.0]),
        np.array([5.18, 9.4, 0.0]),
        np.array([5.18, 9.4, 0.46]),
        np.array([4.55, 9.4, 0.46]),
    ]
    left_f = [
        np.array([4.55, 9.4, 0.0]),
        np.array([4.55, 12.2, 0.0]),
        np.array([4.55, 12.2, 0.46]),
        np.array([4.55, 9.4, 0.46]),
    ]
    top_rim = [
        np.array([4.52, 9.37, 0.46]),
        np.array([5.20, 9.37, 0.46]),
        np.array([5.20, 12.23, 0.46]),
        np.array([4.52, 12.23, 0.46]),
    ]
    soil = [
        np.array([4.61, 9.48, 0.465]),
        np.array([5.11, 9.48, 0.465]),
        np.array([5.11, 12.12, 0.465]),
        np.array([4.61, 12.12, 0.465]),
    ]
    fill_3d_quad(ctx, cam, front, (0.032, 0.056, 0.118, 1.0))
    fill_3d_quad(ctx, cam, left_f, (0.048, 0.088, 0.165, 1.0))
    fill_3d_quad(ctx, cam, top_rim, (0.095, 0.185, 0.250, 1.0), stroke_rgba=(0.02, 0.04, 0.08, 0.8), stroke_w=0.9)
    fill_3d_quad(ctx, cam, soil, (0.018, 0.032, 0.068, 1.0))

    # Crisp aurora highlight along planter top edge
    stroke_3d_seg(
        ctx, cam, np.array([4.52, 9.37, 0.468]), np.array([4.52, 12.23, 0.468]),
        1.5, (0.42, 0.86, 0.78, 0.65), rng,
    )

    # Tapered organic seeded cubic-Bezier ornamental grasses (filled tapering blades, NOT stiff sticks!)
    for _ in range(52):
        gx = float(rng.uniform(4.64, 5.08))
        gy = float(rng.uniform(9.55, 12.02))
        h_m = float(rng.uniform(0.28, 0.74))
        droop_x = float(rng.uniform(-0.38, 0.12))
        droop_y = float(rng.uniform(-0.22, 0.22))
        p0 = np.array([gx, gy, 0.46])
        p1 = np.array([gx + droop_x * 0.22, gy + droop_y * 0.22, 0.46 + h_m * 0.58])
        p2 = np.array([gx + droop_x * 0.72, gy + droop_y * 0.72, 0.46 + h_m * 1.02])
        p3 = np.array([gx + droop_x, gy + droop_y, 0.46 + h_m * float(rng.uniform(0.60, 0.88))])
        proj = [cam.project(pt) for pt in (p0, p1, p2, p3)]
        if any(z < Z_NEAR for _, _, z in proj):
            continue
        q0, q1, q2, q3 = [(float(u), float(v)) for u, v, _ in proj]
        wb = float(rng.uniform(1.8, 3.2))

        g_bl = cairo.LinearGradient(q0[0], q0[1], q3[0], q3[1])
        g_bl.add_color_stop_rgba(0.0, 0.018, 0.040, 0.088, 0.96)
        if rng.random() < 0.36:
            g_bl.add_color_stop_rgba(1.0, 0.24, 0.62, 0.62, 0.82)
        else:
            g_bl.add_color_stop_rgba(1.0, 0.065, 0.155, 0.235, 0.88)
        ctx.set_source(g_bl)
        ctx.move_to(q0[0] - wb * 0.5, q0[1])
        ctx.curve_to(q1[0] - wb * 0.34, q1[1], q2[0] - wb * 0.12, q2[1], q3[0], q3[1])
        ctx.curve_to(q2[0] + wb * 0.12, q2[1], q1[0] + wb * 0.34, q1[1], q0[0] + wb * 0.5, q0[1])
        ctx.close_path()
        ctx.fill()


def draw_near_roof_details(ctx: cairo.Context, cam: Camera3D, rng: np.random.Generator) -> None:
    stroke_3d_seg(
        ctx, cam, np.array([-1.09, 2.3, 0.24]), np.array([-1.09, 13.5, 0.24]),
        3.2, (0.026, 0.048, 0.102, 0.95), rng,
    )
    stroke_3d_seg(
        ctx, cam, np.array([-1.07, 2.3, 0.26]), np.array([-1.07, 13.5, 0.26]),
        1.1, (0.38, 0.85, 0.78, 0.65), rng,
    )

    # Low architectural bollard path light near far-right deck corner (clear of planter!)
    for by in (5.2, 13.8):
        bx = 4.92
        b0 = np.array([bx, by, 0.0])
        b1 = np.array([bx, by, 0.60])
        stroke_3d_seg(ctx, cam, b0, b1, 8.2, (0.024, 0.044, 0.098, 0.98), rng)
        ul, vl, zl = cam.project(np.array([bx, by, 0.52]))
        if zl > Z_NEAR:
            r_glow = max(8.0, 200.0 / float(zl))
            g = cairo.RadialGradient(float(ul), float(vl), 1.0, float(ul), float(vl), r_glow)
            g.add_color_stop_rgba(0.0, 0.82, 1.00, 0.94, 0.88)
            g.add_color_stop_rgba(0.28, 0.39, 0.92, 0.73, 0.42)
            g.add_color_stop_rgba(1.0, 0.16, 0.45, 0.52, 0.0)
            ctx.set_source(g)
            ctx.arc(float(ul), float(vl), r_glow, 0, math.tau)
            ctx.fill()


# =============================================================================
# Stage 2 — Fresnel Shallow Rooftop Puddle Reflections (Horizon + 3D Mirror)
# =============================================================================

def composite_fresnel_puddles(
    canvas: np.ndarray,
    sky_for_refl: np.ndarray,
    wall_uv: np.ndarray,
    puddle_mask_uv: np.ndarray,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
) -> np.ndarray:
    p0_d = np.array([-1.25, 0.2, Z_DECK], dtype=np.float64)
    e1_d = np.array([6.95, 0.0, 0.0], dtype=np.float64)
    e2_d = np.array([0.0, 15.6, 0.0], dtype=np.float64)

    pm_rgba = np.zeros((puddle_mask_uv.shape[0], puddle_mask_uv.shape[1], 4), dtype=np.float32)
    pm_rgba[:, :, 0] = puddle_mask_uv
    pm_rgba[:, :, 3] = puddle_mask_uv
    mask_buf = np.zeros((H, W, 3), dtype=np.float32)
    mask_buf = cam.warp_plane_over(
        mask_buf, pm_rgba, p0_d, e1_d, e2_d,
        fog_rgb=np.zeros(3, dtype=np.float32), fog_start=50.0, fog_density=0.0,
    )
    puddle_screen = np.clip(mask_buf[:, :, 0], 0.0, 1.0)
    if float(puddle_screen.max()) < 1e-3:
        return canvas

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rip_h1 = resize_bicubic(rng.standard_normal((H // 6, W // 46)).astype(np.float32), W, H)
    rip_h2 = resize_bicubic(rng.standard_normal((H // 3, W // 22)).astype(np.float32), W, H)
    rip = 0.62 * rip_h1 + 0.38 * rip_h2
    y_src = np.clip(2.0 * cam.horizon_y - yy + rip * 3.8, 0.0, H - 1.0)
    x_src = np.clip(xx + rip * 1.2, 0.0, W - 1.0)
    refl_world = remap_bilinear(sky_for_refl, x_src, y_src) * 0.94

    wall_quad = np.array([
        [X_WALL, 2.2, 0.0],
        [X_WALL, 13.7, 0.0],
        [X_WALL, 13.7, Z_ROOF + 0.05],
        [X_WALL, 2.2, Z_ROOF + 0.05],
    ], dtype=np.float64)
    mir_wall = mirror_quad_z(wall_quad, Z_DECK)
    refl_world = cam.warp_plane_over(
        refl_world, wall_uv, mir_wall[0], mir_wall[1] - mir_wall[0], mir_wall[3] - mir_wall[0],
        fog_rgb=fog_rgb, fog_start=6.0, fog_density=0.032,
    )

    s_mir, c_mir = new_cairo_2x()
    draw_perimeter_railing(c_mir, cam, rng, mirrored=True, alpha_scale=0.82)
    draw_tripod_telescope(c_mir, cam, rng, mirrored=True, alpha_scale=0.78)
    draw_rooftop_bench(c_mir, cam, rng, mirrored=True, alpha_scale=0.88)
    refl_world = over_premul(refl_world, finish_cairo_2x(s_mir))

    refl_rippled = remap_bilinear(refl_world, np.clip(xx + rip * 1.3, 0, W - 1), np.clip(yy + rip * 1.5, 0, H - 1))

    hit, wx, wy, t_ray = cam.ray_plane_z0(xx, yy, Z_DECK)
    dist_t = np.clip(1.0 - (t_ray.astype(np.float32) - 2.2) / 14.0, 0.0, 1.0)
    fresnel = 0.30 + (0.90 - 0.30) * np.power(1.0 - dist_t, 1.4)

    w_refl = puddle_screen * fresnel * hit.astype(np.float32)
    return canvas * (1.0 - w_refl[..., None]) + refl_rippled * w_refl[..., None]


# =============================================================================
# Stage 2 & 3 — Subtle Diagonal Aurora/Skylight Shafts Along SUN_DIR
# =============================================================================

def paint_aurora_shafts(cam: Camera3D, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    p_ref = np.array([0.8, 8.0, 2.2], dtype=np.float64)
    u0, v0, _ = cam.project(p_ref)
    u1, v1, _ = cam.project(p_ref + SUN_DIR * 3.0)
    dx, dy = float(u1 - u0), float(v1 - v0)
    dlen = math.hypot(dx, dy) + 1e-6
    rx, ry = dx / dlen, dy / dlen
    nx, ny = -ry, rx

    s_coord = (xx - W * 0.36) * rx + (yy - H * 0.24) * ry
    p_coord = (xx - W * 0.36) * nx + (yy - H * 0.24) * ny

    acc = np.zeros((H, W), dtype=np.float32)
    shafts = [(-180, 95, 0.12), (20, 120, 0.16), (260, 135, 0.13), (490, 110, 0.09)]
    for cp, wp, st in shafts:
        w_eff = wp * (1.0 + np.maximum(0.0, s_coord) / (H * 1.5))
        beam = np.exp(-0.5 * ((p_coord - cp) / w_eff) ** 2)
        vert = np.clip((s_coord + 280.0) / 300.0, 0.0, 1.0) * np.exp(-np.maximum(0.0, s_coord) / (H * 0.95))
        acc += beam * vert * st

    turb = resize_bicubic(0.84 + 0.16 * fbm_gouache(H // 4, W // 4, rng), W, H)
    acc = np.clip(acc * turb, 0.0, 0.48)
    return acc[..., None] * np.array([0.38, 0.90, 0.78], dtype=np.float32)


# =============================================================================
# Stage 4 — Foreground Circular Disc Bokeh & 35mm Analog Optics
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


def convolve_fft_2d_roi(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    alpha = img[:, :, 3]
    ys, xs = np.nonzero(alpha > 1e-5)
    if len(ys) == 0:
        return np.zeros_like(img, dtype=np.float32)
    kh, kw = kernel.shape
    pad = max(kh, kw) + 4
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(img.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(img.shape[1], int(xs.max()) + pad + 1)

    sub = img[y0:y1, x0:x1]
    ih, iw = sub.shape[:2]
    fh = 1 << int(math.ceil(math.log2(ih + kh - 1)))
    fw = 1 << int(math.ceil(math.log2(iw + kw - 1)))
    k_pad = np.zeros((fh, fw), dtype=np.float32)
    k_pad[:kh, :kw] = kernel
    k_pad = np.roll(np.roll(k_pad, -(kh // 2), axis=0), -(kw // 2), axis=1)
    K = np.fft.rfft2(k_pad)

    out_sub = np.empty_like(sub, dtype=np.float32)
    for c in range(sub.shape[2]):
        p = np.zeros((fh, fw), dtype=np.float32)
        p[:ih, :iw] = sub[:, :, c]
        out_sub[:, :, c] = np.fft.irfft2(np.fft.rfft2(p) * K, s=(fh, fw))[:ih, :iw]

    out = np.zeros_like(img, dtype=np.float32)
    out[y0:y1, x0:x1] = out_sub
    return out


def paint_foreground_bokeh_antenna_parapet(rng: np.random.Generator) -> np.ndarray:
    """
    Close out-of-focus concrete rooftop parapet coping & directional Yagi
    antenna mast framing the lower-right corner cleanly without crowding the
    midground planter, convolved with circular aperture disc bokeh (r = 7.8)
    + 1 px opposite R/B lateral shift.
    """
    surf, ctx = new_cairo_2x()

    parapet = [
        (W + 30, H + 30),
        (W - 390, H + 30),
        (W - 350, H - 62),
        (W - 265, H - 115),
        (W - 145, H - 155),
        (W + 30, H - 190),
    ]
    ctx.set_source_rgba(0.010, 0.020, 0.056, 0.98)
    ctx.move_to(*parapet[0])
    for pt in parapet[1:]:
        ctx.line_to(*pt)
    ctx.close_path()
    ctx.fill()

    # Subtle steel coping cap rim catching prism-mint aurora sheen
    ctx.set_source_rgba(0.28, 0.68, 0.68, 0.50)
    ctx.set_line_width(4.5)
    ctx.move_to(*parapet[1])
    for pt in parapet[2:]:
        ctx.line_to(*pt)
    ctx.stroke()

    # Slender rooftop antenna mast positioned cleanly near the right edge
    mast_base = (W - 95, H - 145)
    mast_top = (W - 142, H - 615)
    stroke_catmull_rom(
        ctx, [mast_base, (W - 118, H - 380), mast_top],
        base_w=9.5, z_depth=0.9, rgba=(0.012, 0.024, 0.062, 0.96), rng=rng, taper=False, tremor=0.30,
    )
    boom = [(W - 258, H - 538), (W - 132, H - 512), (W - 10, H - 486)]
    stroke_catmull_rom(
        ctx, boom, base_w=5.8, z_depth=0.9, rgba=(0.012, 0.024, 0.062, 0.95), rng=rng, taper=False, tremor=0.22,
    )
    for frac, half_span in ((0.14, 44), (0.38, 52), (0.64, 56), (0.86, 46)):
        bx = boom[0][0] * (1 - frac) + boom[2][0] * frac
        by = boom[0][1] * (1 - frac) + boom[2][1] * frac
        stroke_catmull_rom(
            ctx, [(bx - 10, by - half_span), (bx, by), (bx + 10, by + half_span)],
            base_w=4.2, z_depth=0.9, rgba=(0.014, 0.028, 0.068, 0.94), rng=rng, taper=False, tremor=0.18,
        )
    # Glowing mint-white nocturnal aviation beacon atop mast -> renders as a crisp circular bokeh disc!
    ctx.set_source_rgba(0.62, 0.99, 0.88, 0.96)
    ctx.arc(mast_top[0], mast_top[1], 5.8, 0, math.tau)
    ctx.fill()

    fg_1x = finish_cairo_2x(surf)
    blurred = convolve_fft_2d_roi(fg_1x, make_disc_kernel(radius=7.8))
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
    tint = np.array([0.88, 1.08, 1.02], dtype=np.float32)
    hal = (
        0.25 * gaussian_blur_2d(hl_src, 6.5)
        + 0.18 * gaussian_blur_2d(hl_src, 24.0)
        + 0.12 * gaussian_blur_2d(hl_src, 72.0)
    ) * tint[None, None, :]
    img = 1.0 - (1.0 - np.clip(img, 0.0, 1.0)) * (1.0 - np.clip(hal, 0.0, 0.85))

    # 2. Radial corner CA + gentle optical vignette
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (xx - W * 0.5) / (W * 0.5)
    ny = (yy - H * 0.5) / (H * 0.5)
    r2 = nx * nx + ny * ny
    ca_scale = 0.00105
    img[:, :, 0] = remap_bilinear(img[:, :, 0], xx + nx * r2 * (W * ca_scale), yy + ny * r2 * (H * ca_scale))
    img[:, :, 2] = remap_bilinear(img[:, :, 2], xx - nx * r2 * (W * ca_scale), yy - ny * r2 * (H * ca_scale))
    img *= np.clip(1.0 - 0.20 * (r2 ** 1.15), 0.72, 1.0)[..., None]

    # 3. Linear shadow black lift + C1-continuous tanh highlight shoulder (knee=0.80, ceiling=0.965)
    img = 0.013 + img * 0.987
    knee, ceiling = 0.80, 0.965
    span = ceiling - knee
    img = np.where(img > knee, knee + span * np.tanh((img - knee) / span), img)

    # 4. Luminance-weighted silver-halide grain
    lum2 = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    env = np.clip(1.12 - 0.95 * np.power(np.clip(lum2, 0.0, 1.0), 1.55), 0.12, 1.12)
    grain = rng.normal(0.0, 0.010, size=(H, W, 3)).astype(np.float32)
    mono_g = rng.normal(0.0, 0.0085, size=(H, W, 1)).astype(np.float32)
    img = np.clip(img + (0.65 * mono_g + 0.35 * grain) * env[..., None], 0.0, 1.0)

    # 5. Faint analog transfer patina: horizontal YIQ chroma softening (1.0 px) + reflect101
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
    seed: int = 202601,
    audit_dir: pathlib.Path | None = None,
    webp_outputs: list[pathlib.Path] | None = None,
) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # Stage 1: Off-axis Camera3D (vanishing point ~61.9% width, horizon ~55.2% height)
    cam = Camera3D(W, H, eye=(1.65, -1.2, 1.52), target=(-0.35, 14.0, 2.02), fov_x_deg=58.0)
    print(f"[aria-01] Camera VP: ({cam.vp_x:.1f}, {cam.vp_y:.1f}) -> {cam.vp_x / W * 100:.1f}% width, {cam.horizon_y / H * 100:.1f}% height")

    canvas, sky_for_refl, fog_rgb, m_arch = paint_sky_aurora_valley(cam, rng)

    # Stage 1 & 2: Rain-slicked Voronoi concrete rooftop terrace & observatory facade UVs
    deck_uv, puddle_mask_uv = paint_voronoi_rooftop_uv(rng, (-1.25, 5.70), (0.2, 15.8), tw=896, th=896)
    wall_uv = paint_observatory_wall_uv(rng, (2.2, 13.7), (0.0, Z_ROOF + 0.05), tw=1024, th=512)

    canvas = cam.warp_plane_over(
        canvas, deck_uv,
        p0=np.array([-1.25, 0.2, Z_DECK]),
        e1=np.array([6.95, 0.0, 0.0]),
        e2=np.array([0.0, 15.6, 0.0]),
        fog_rgb=fog_rgb, fog_start=6.0, fog_density=0.030,
    )

    # Stage 2: Composite Fresnel rain-puddle reflections (horizon aurora + 3D plane-mirrored quads/props)
    canvas = composite_fresnel_puddles(canvas, sky_for_refl, wall_uv, puddle_mask_uv, cam, rng, fog_rgb)

    # Stage 3 Band 0 (Observatory Dome, Penthouse Roof Overhang & Perimeter Curbs)
    s_b0, c_b0 = new_cairo_2x()
    draw_observatory_dome_and_penthouse_shell(c_b0, cam, rng)
    b0_1x = finish_cairo_2x(s_b0)
    a0 = b0_1x[:, :, 3:4]
    b0_1x[:, :, :3] = np.clip(b0_1x[:, :, :3] + a0 * (m_arch * 0.011)[..., None], 0.0, 1.2)
    canvas = over_premul(canvas, b0_1x)

    # Warp observatory concrete facade & hand-crafted studio windows cleanly behind fascia lip
    canvas = cam.warp_plane_over(
        canvas, wall_uv,
        p0=np.array([X_WALL, 2.2, 0.0]),
        e1=np.array([0.0, 11.5, 0.0]),
        e2=np.array([0.0, 0.0, Z_ROOF + 0.05]),
        fog_rgb=fog_rgb, fog_start=6.0, fog_density=0.028,
    )

    # Stage 3 Band 1 (Far & Right Perimeter Railings + Concrete Planter & Tapered Grasses)
    s_b1, c_b1 = new_cairo_2x()
    draw_perimeter_railing(c_b1, cam, rng, mirrored=False)
    draw_rooftop_planter_and_grasses(c_b1, cam, rng)
    canvas = over_premul(canvas, finish_cairo_2x(s_b1))

    # Atmospheric diagonal aurora/skylight shafts strictly parallel to SUN_DIR
    shafts = paint_aurora_shafts(cam, rng)
    canvas = screen_blend(canvas, shafts)

    # Stage 3 Band 2 (Midground Hero Props: Tripod Telescope & Rooftop Bench + Glowing Lantern)
    s_b2, c_b2 = new_cairo_2x()
    draw_tripod_telescope(c_b2, cam, rng, mirrored=False)
    draw_rooftop_bench(c_b2, cam, rng, mirrored=False)
    draw_near_roof_details(c_b2, cam, rng)
    b2_1x = finish_cairo_2x(s_b2)
    a2 = b2_1x[:, :, 3:4]
    b2_1x[:, :, :3] = np.clip(b2_1x[:, :, :3] + a2 * (m_arch * 0.010)[..., None], 0.0, 1.2)
    canvas = over_premul(canvas, b2_1x)

    # Stage 4: Foreground Circular Disc Bokeh Parapet & Antenna + 35mm Analog Finish
    fg_bokeh = paint_foreground_bokeh_antenna_parapet(rng)
    canvas = over_premul(canvas, fg_bokeh)
    final_rgb = apply_35mm_finish(canvas, rng)

    elapsed = time.perf_counter() - t0
    print(f"[aria-01] Rendered {W}x{H} in {elapsed:.2f}s")

    u8 = np.round(np.clip(final_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    full_im = Image.fromarray(u8, mode="RGB")

    if webp_outputs:
        for wp in webp_outputs:
            wp.parent.mkdir(parents=True, exist_ok=True)
            full_im.save(wp, format="WEBP", quality=90, method=6)
            print(f"[aria-01] Exported WebP: {wp} ({wp.stat().st_size / 1024:.1f} KiB)")

    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        full_im.save(audit_dir / "w01_full.png", optimize=True)
        crops = {
            "w01_crop_aperture.png": (int(W * 0.25), int(H * 0.02)),      # Observatory dome + silky prism-mint aurora
            "w01_crop_midground.png": (int(W * 0.16), int(H * 0.42)),     # Observatory studio windows + Voronoi puddles
            "w01_crop_foreground.png": (W - 600, H - 450),                # Bench/planter/parapet/antenna circular bokeh
        }
        for cname, (cx0, cy0) in crops.items():
            x0 = max(0, min(W - 600, cx0))
            y0 = max(0, min(H - 450, cy0))
            full_im.crop((x0, y0, x0 + 600, y0 + 450)).save(audit_dir / cname)

    return final_rgb, {"elapsed": elapsed, "vp_ratio": cam.vp_x / W}


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    default_webps = [
        repo_root / "session" / "themes" / "aria" / "walls" / "wall-01.webp",
        repo_root / "docs" / "assets" / "walls" / "aria-01.webp",
    ]
    ap = argparse.ArgumentParser(description="Render Aria 01 — Aurora Rooftop Observatory (2048x1152)")
    ap.add_argument("--seed", type=int, default=202601)
    ap.add_argument("--audit-dir", type=pathlib.Path, default=None)
    ap.add_argument("--no-webp", action="store_true")
    args = ap.parse_args()
    render_scene(
        seed=args.seed,
        audit_dir=args.audit_dir,
        webp_outputs=[] if args.no_webp else default_webps,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Aria 02 — "Starlit Tidal Railway"
Pure-Python 3D-Grounded Nocturnal Anime Scene Painting Pipeline (2048 x 1152)

Five-stage deterministic pipeline (zero image models, zero external assets):
  Stage 1: Metric right-handed Camera3D with off-axis vanishing point (~62.7%
           width) carrying the railway & basalt embankment along the left while
           opening vast mirror-calm tidal water on the right; exact 3x3 UV-to-
           screen plane homography masked by UV bounds and z_cam > 0.05 with
           exponential coastal mist fog, near-plane 3D line clipping, and
           irregular Voronoi (d2 - d1) stone embankment blocks extending
           seamlessly into the luminous coastal dawn horizon.
  Stage 2: Unified normalized 3D celestial light travel vector SUN_DIR (z < 0)
           tracing embankment and trackbed coordinates backward along -SUN_DIR
           with 6 fixed height-refinement steps; vast Fresnel capillary-rippled
           tidal water combining horizon dawn/star reflections
           (y_src = 2*horizon_y - y) with 3D plane-mirrored structures (poles,
           catenary wires, signal post, glowing mint-white signal lamp).
  Stage 3: 3+ depth-interleaved 2x supersampled PyCairo bands (Lanczos-
           downsampled) for receding weathered timber railway sleepers, glinting
           twin steel rails, leaning utility poles with sagging 3D catenary
           wires, and a single glowing mint-white distant signal lamp, broken up
           by isotropic fbm_gouache, flat-brush dabs, cold-press paper tooth,
           and warm bounce light in deep shadows.
  Stage 4: Foreground lower-right coastal reeds silhouette (per-instance seeded
           cubic Beziers) convolved in premultiplied RGBA with a circular
           aperture disc bokeh kernel (r = 7.8) + 1 px opposite R/B lateral
           shift, followed by 35mm multi-scale halation, radial CA, vignette,
           shadow lift + C1 tanh highlight shoulder, luminance-weighted
           silver-halide grain, and faint analog YIQ transfer patina.
  Stage 5: Deterministic render (<60s CPU) exporting 2048x1152 WebP
           (quality=90, method=6) to session/themes/aria/walls/wall-02.webp and
           docs/assets/walls/aria-02.webp, plus optional 1:1 macro audit crops.
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

# Stage 2: Single normalized 3D light travel vector (SUN_DIR[2] < 0)
_RAW_LIGHT = np.array([-0.46, -0.68, -0.571], dtype=np.float64)
SUN_DIR = _RAW_LIGHT / np.linalg.norm(_RAW_LIGHT)

# World metric geometry (meters)
Z_WATER = 0.0
Z_BED = 0.24
X_RAIL_L = -1.07
X_RAIL_R = 0.37
X_TRACK_C = 0.5 * (X_RAIL_L + X_RAIL_R)  # -0.35 m
Z_RAIL_TOP = Z_BED + 0.14

# Leaning utility pole positions along left embankment flank: (x_base, y_pos, height, lean_x, lean_y)
POLE_SPECS = [
    (-1.86, 4.7, 5.55, 0.065, 0.015),
    (-1.84, 12.2, 5.50, 0.085, -0.010),
    (-1.82, 21.0, 5.45, 0.055, 0.018),
    (-1.80, 31.5, 5.40, 0.075, -0.012),
    (-1.78, 44.5, 5.35, 0.060, 0.010),
    (-1.76, 61.0, 5.30, 0.070, 0.000),
    (-1.74, 82.0, 5.25, 0.065, 0.000),
    (-1.72, 112.0, 5.20, 0.060, 0.000),
]

SIGNAL_POS = np.array([1.68, 24.8, Z_WATER], dtype=np.float64)
SIGNAL_LAMP_Z = 3.55


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


def flat_brush_dabs(h: int, w: int, rng: np.random.Generator, count: int = 90) -> np.ndarray:
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
        fog_start: float = 8.0,
        fog_density: float = 0.022,
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


def mirror_quad_z(pts: np.ndarray, z_plane: float = Z_WATER) -> np.ndarray:
    p0 = np.array([0.0, 0.0, z_plane], dtype=np.float64)
    n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    d = np.sum((pts - p0) * n, axis=-1, keepdims=True)
    return pts - 2.0 * d * n


# =============================================================================
# Stage 2 — Unified 3D Light Solver (Embankment Slope & Utility Pole Shadows)
# =============================================================================

def trace_embankment_shadows(wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> np.ndarray:
    """
    Fast float32 3D light solver tracing embankment & trackbed points backward
    along -SUN_DIR (6 fixed height-refinement steps for overhead crossarms &
    sloped embankment surface) to cast aligned pole, signal, and rail shadows.
    """
    wx32 = wx.astype(np.float32)
    wy32 = wy.astype(np.float32)
    wz32 = wz.astype(np.float32)
    sx0 = np.float32(SUN_DIR[0])
    sy0 = np.float32(SUN_DIR[1])
    inv_sz = np.float32(1.0 / (-SUN_DIR[2]))

    # 6 fixed height-refinement steps tracing backward along -SUN_DIR to overhead catenary crossarm canopy
    z_ov = np.full_like(wx32, 4.4, dtype=np.float32)
    for _ in range(6):
        t_ov = np.maximum(0.0, (z_ov - wz32) * inv_sz)
        rx_ov = wx32 - sx0 * t_ov
        z_ov = np.where(rx_ov < -0.5, 4.95, 4.10).astype(np.float32)

    shadow = np.ones_like(wx32, dtype=np.float32)

    # Exact projected pole shaft shadows onto z = wz
    for px, py, ph, lx, ly in POLE_SPECS[:5]:
        ax = np.float32(lx + SUN_DIR[0] / (-SUN_DIR[2]))
        ay = np.float32(ly + SUN_DIR[1] / (-SUN_DIR[2]))
        denom = ax * ax + ay * ay + 1e-6
        dx0 = wx32 - (np.float32(px) - sx0 * wz32 * inv_sz)
        dy0 = wy32 - (np.float32(py) - sy0 * wz32 * inv_sz)
        h_proj = (dx0 * ax + dy0 * ay) / denom
        h_cl = np.clip(h_proj, 0.0, np.float32(ph))
        cx_s = np.float32(px) + ax * h_cl - sx0 * wz32 * inv_sz
        cy_s = np.float32(py) + ay * h_cl - sy0 * wz32 * inv_sz
        dist_p = np.hypot(wx32 - cx_s, wy32 - cy_s)
        in_pole = (h_proj > 0.0) & (h_proj < ph) & (dist_p < 0.16)
        shadow *= np.where(in_pole, 0.56, 1.0)

    # Signal mast & twin railhead cast shadows
    t_rl = np.maximum(0.0, (np.float32(Z_RAIL_TOP) - wz32) * inv_sz)
    rx_l = wx32 - sx0 * t_rl
    in_rail_sh = ((rx_l > X_RAIL_L - 0.04) & (rx_l < X_RAIL_L + 0.08)) | (
        (rx_l > X_RAIL_R - 0.04) & (rx_l < X_RAIL_R + 0.08)
    )
    shadow *= np.where(in_rail_sh, 0.70, 1.0)
    return shadow


# =============================================================================
# Stage 1 & 2 — Midnight-to-Periwinkle Starry Dawn Sky & Silky Tidal Water
# =============================================================================

def paint_sky_dawn_horizon(
    cam: Camera3D, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Paint the poetic midnight-navy (#060b1c) -> indigo -> periwinkle -> prism-mint
    starry dawn horizon, silky flowing aurora veils (zero harsh vertical striping!),
    soft anime backlit cirrus clouds, silver-mint crescent moon, and distant
    coastal headland silhouettes.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = xx / W
    v = yy / H
    hy = cam.horizon_y / H

    mottle = fbm_gouache(H, W, rng)
    dabs = flat_brush_dabs(H, W, rng, count=90)
    tooth = paper_tooth(H, W, rng)

    sky_t = np.clip(v / max(0.05, hy + 0.01), 0.0, 1.0)

    c_zenith = np.array([0.020, 0.039, 0.106], dtype=np.float32)
    c_indigo = np.array([0.072, 0.136, 0.312], dtype=np.float32)
    c_peri = np.array([0.275, 0.405, 0.705], dtype=np.float32)
    c_mint_dawn = np.array([0.420, 0.905, 0.775], dtype=np.float32)

    s1 = np.power(sky_t, 1.18)
    s2 = np.power(np.clip((sky_t - 0.44) / 0.56, 0.0, 1.0), 1.48)
    s3 = np.power(np.clip((sky_t - 0.74) / 0.26, 0.0, 1.0), 1.82)

    sky = (
        c_zenith[None, None, :] * (1.0 - s1[..., None])
        + c_indigo[None, None, :] * (s1 * (1.0 - s2))[..., None]
        + c_peri[None, None, :] * (s1 * s2 * (1.0 - s3))[..., None]
        + c_mint_dawn[None, None, :] * (s1 * s2 * s3)[..., None]
    )

    # Radiant prism-mint & frosted-white dawn aperture centered around u ~ 0.67
    dawn_core = np.exp(-(((u - 0.67) / 0.36) ** 2 + ((v - (hy - 0.012)) / 0.070) ** 2))
    dawn_wide = np.exp(-(((u - 0.64) / 0.58) ** 2 + ((v - (hy - 0.030)) / 0.145) ** 2))
    sky += dawn_wide[..., None] * np.array([0.16, 0.36, 0.44], dtype=np.float32)
    sky += dawn_core[..., None] * np.array([0.32, 0.58, 0.50], dtype=np.float32)

    # Silky sweeping aurora veils arching gently across upper sky (smooth silk folds, ZERO barcode lines!)
    u_sh = u - (v - 0.20) * 0.12
    silk_rays = (
        0.78
        + 0.13 * np.sin(u_sh * 26.0 + 1.5 * np.sin(u_sh * 5.8))
        + 0.08 * np.sin(u_sh * 48.0 - 1.1 * np.cos(u_sh * 9.5))
    ).astype(np.float32)
    soft_fold = resize_bicubic(rng.standard_normal((max(4, H // 180), max(8, W // 72))).astype(np.float32), W, H)
    curtain_silk = np.clip(silk_rays * (0.88 + 0.15 * soft_fold), 0.44, 1.22)

    for curve_v, w_up, w_dn, st, u_s, c_lo, c_hi in (
        (
            0.228 - 0.072 * np.sin((u - 0.24) * 3.1) + 0.020 * np.cos(u * 5.8),
            0.120, 0.020, 0.64, 0.14,
            np.array([0.385, 0.935, 0.745], dtype=np.float32),
            np.array([0.520, 0.470, 0.910], dtype=np.float32),
        ),
        (
            0.312 - 0.044 * np.cos((u - 0.36) * 3.8),
            0.092, 0.018, 0.44, 0.26,
            np.array([0.350, 0.885, 0.790], dtype=np.float32),
            np.array([0.420, 0.580, 0.930], dtype=np.float32),
        ),
    ):
        dv = v - curve_v
        sig = np.where(dv < 0.0, w_up, w_dn)
        env = np.exp(-0.5 * (dv / sig) ** 2)
        arc = np.exp(-0.5 * (dv / (w_dn * 0.70)) ** 2)
        sweep = np.clip((u - u_s) / 0.28, 0.0, 1.0) * np.clip((1.06 - u) / 0.18, 0.0, 1.0)
        alt_t = np.clip(-dv / (w_up * 1.40), 0.0, 1.0)
        col = c_lo[None, None, :] * (1.0 - alt_t[..., None]) + c_hi[None, None, :] * alt_t[..., None]
        wisp = (env * curtain_silk * sweep * st * np.clip((hy - v) / 0.09, 0.0, 1.0))[..., None] * col * 0.78
        wisp += (arc * sweep * st * 0.32 * np.clip((hy - v) / 0.09, 0.0, 1.0))[..., None] * np.array(
            [0.74, 0.99, 0.92], dtype=np.float32
        )
        sky = screen_blend(sky, wisp)

    # Graceful stratified anime twilight cirrus clouds (computed at half-res for speed & silky smoothness)
    c_half = fbm_gouache(H // 2, W // 2, rng, scales=((56, 52), (22, 20), (9, 8.5), (4, 3.8)))
    c_noise = resize_bicubic(c_half, W, H)
    # Gently wispy horizontal dawn cloud banks floating softly above the bright horizon
    cloud1 = np.exp(-((v - (hy - 0.095 + 0.016 * np.sin(u * 4.2))) / 0.036) ** 2) * np.clip(c_noise * 0.65 + 0.06, 0.0, 1.0)
    cloud2 = np.exp(-((v - (hy - 0.165 - 0.014 * np.cos(u * 3.5))) / 0.044) ** 2) * np.clip(c_noise * 0.58 - 0.12, 0.0, 1.0)
    c_lit = np.array([0.48, 0.90, 0.86], dtype=np.float32)[None, None, :]
    c_shd = np.array([0.13, 0.21, 0.43], dtype=np.float32)[None, None, :]
    under1 = np.clip((v - (hy - 0.112)) / 0.035, 0.0, 1.0)[..., None]
    under2 = np.clip((v - (hy - 0.182)) / 0.040, 0.0, 1.0)[..., None]
    sky = sky * (1.0 - 0.44 * cloud1[..., None]) + (c_shd * (1.0 - under1) + c_lit * under1) * (0.44 * cloud1[..., None])
    sky = sky * (1.0 - 0.34 * cloud2[..., None]) + (c_shd * (1.0 - under2) + (c_lit * 0.78) * under2) * (0.34 * cloud2[..., None])

    sky += (mottle * 0.012 + dabs * 0.0085 + tooth * 0.0045)[..., None]

    # Delicate luminous silver-mint crescent moon high in upper-right sky (u ~ 0.77, v ~ 0.165)
    mx, my = W * 0.768, H * 0.165
    mr = 21.0
    dm = np.hypot(xx - mx, yy - my)
    dm_cut = np.hypot(xx - (mx - 8.8), yy - (my - 5.2))
    moon_disc = np.clip((mr - dm + 0.8) / 1.4, 0.0, 1.0) * np.clip((dm_cut - (mr * 0.90)) / 1.5, 0.0, 1.0)
    moon_halo = np.exp(-((dm / 64.0) ** 2)) * 0.24 + np.exp(-((dm / 180.0) ** 2)) * 0.09
    sky += moon_halo[..., None] * np.array([0.46, 0.90, 0.85], dtype=np.float32)
    sky = sky * (1.0 - moon_disc[..., None]) + np.array([0.94, 0.99, 0.96], dtype=np.float32) * moon_disc[..., None]

    # Crisp spectral starfield
    star_layer = np.zeros((H, W, 3), dtype=np.float32)
    star_cols = [
        np.array([0.92, 0.96, 1.00], dtype=np.float32),
        np.array([0.70, 0.98, 0.90], dtype=np.float32),
        np.array([0.82, 0.88, 1.00], dtype=np.float32),
        np.array([1.00, 0.93, 0.84], dtype=np.float32),
    ]
    max_sy = int(H * (hy - 0.035))
    for _ in range(460):
        sx = float(rng.uniform(4, W - 4))
        sy = float(rng.uniform(4, max_sy))
        alt_vis = float(np.clip((hy - 0.02 - sy / H) / 0.34, 0.10, 1.0))
        mag = float(rng.power(0.30))
        bright = (0.15 + 0.85 * mag) * alt_vis
        col = star_cols[int(rng.integers(0, len(star_cols)))]
        ix, iy = int(sx), int(sy)
        rad = 1 if mag < 0.68 else (2 if mag < 0.91 else 3)
        y0, y1 = max(0, iy - rad - 1), min(H, iy + rad + 2)
        x0, x1 = max(0, ix - rad - 1), min(W, ix + rad + 2)
        gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        sig = 0.55 + 0.56 * mag
        spot = np.exp(-0.5 * ((gx - sx) ** 2 + (gy - sy) ** 2) / (sig * sig)) * bright
        star_layer[y0:y1, x0:x1] += spot[..., None] * col[None, None, :]
        if mag > 0.945:
            slen = int(8 + 10 * (mag - 0.945) / 0.055)
            for ds in range(-slen, slen + 1):
                w_sp = math.exp(-abs(ds) / (slen * 0.26)) * 0.52 * bright
                if 0 <= ix + ds < W:
                    star_layer[iy, ix + ds] += col * w_sp
                if 0 <= iy + ds < H:
                    star_layer[iy + ds, ix] += col * w_sp

    sky = np.clip(sky + star_layer, 0.0, 1.35)

    # Distant quiet misty coastal island/headland silhouettes along horizon
    u_1d = np.arange(W, dtype=np.float32) / W
    island = (
        hy
        - 0.017 * np.exp(-((u_1d - 0.89) / 0.11) ** 2)
        - 0.011 * np.exp(-((u_1d - 0.13) / 0.12) ** 2)
        - 0.0035 * np.sin(u_1d * 29.0 + 0.7) * np.exp(-((u_1d - 0.89) / 0.15) ** 2)
    )
    isl_mask = np.clip((v - island[None, :]) / 0.0028, 0.0, 1.0) * (v <= hy + 0.002)
    isl_col = np.array([0.058, 0.114, 0.240], dtype=np.float32)[None, None, :]
    sky = sky * (1.0 - isl_mask[..., None]) + isl_col * isl_mask[..., None]

    fog_rgb = np.array([0.195, 0.365, 0.505], dtype=np.float32)
    return sky, sky.copy(), fog_rgb, mottle


def paint_vast_tidal_water(
    sky_canvas: np.ndarray,
    sky_for_refl: np.ndarray,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    seabed_mottle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vast mirror-calm Fresnel capillary-rippled starlit tidal water with silky
    fine horizontal wave shimmer (ZERO hard binary blotches!), combining
    capillary-rippled horizon reflection (y_src = 2*horizon_y - y) with exact
    Fresnel angle reflectivity: fresnel_min + (fresnel_max - fresnel_min)*(1-dist_t)**1.4.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    hy_px = cam.horizon_y
    water_mask = np.clip((yy - hy_px + 0.8) / 2.2, 0.0, 1.0)

    hit, wx, wy, t_ray = cam.ray_plane_z0(xx, yy, Z_WATER)
    depth_f = np.clip((yy - hy_px) / max(1.0, H - hy_px), 0.0, 1.0)

    # Silky fine multi-octave horizontal capillary ripples (horizontal-only noise ONLY on water!)
    rip1 = resize_bicubic(rng.standard_normal((H // 3, W // 38)).astype(np.float32), W, H)
    rip2 = resize_bicubic(rng.standard_normal((H // 2, W // 18)).astype(np.float32), W, H)
    rip3 = resize_bicubic(rng.standard_normal((H // 2, W // 9)).astype(np.float32), W, H)
    rip = 0.52 * rip1 + 0.32 * rip2 + 0.16 * rip3

    # Gentle perspective tidal wavelets
    swell = np.sin(wy.astype(np.float32) * 1.9 + wx.astype(np.float32) * 0.25 + rip1 * 1.4) * 0.45
    rip_y = (rip * 5.2 + swell * 2.4) * (0.22 + 0.85 * np.power(depth_f, 0.65))
    rip_x = rip * 1.8 * (0.20 + 0.72 * depth_f)

    y_src = np.clip(2.0 * hy_px - yy + rip_y, 0.0, H - 1.0)
    x_src = np.clip(xx + rip_x, 0.0, W - 1.0)
    horizon_refl = remap_bilinear(sky_for_refl, x_src, y_src)

    # Deep translucent midnight-indigo water body
    water_body = np.zeros((H, W, 3), dtype=np.float32)
    water_body[:, :, 0] = 0.015 + seabed_mottle * 0.004
    water_body[:, :, 1] = 0.036 + seabed_mottle * 0.007
    water_body[:, :, 2] = 0.090 + seabed_mottle * 0.011

    # Exact Fresnel reflectivity
    dist_t = np.power(depth_f, 0.62)
    fresnel = 0.30 + (0.93 - 0.30) * np.power(1.0 - dist_t, 1.4)
    water_rgb = water_body * (1.0 - fresnel[..., None]) + (horizon_refl * 0.95) * fresnel[..., None]

    # Silky soft dawn shimmer across fine capillary wave slopes (mirror-calm liquid glass!)
    slope_raw = np.power(np.clip(0.46 + 0.32 * rip2 + 0.22 * rip3 + 0.20 * swell, 0.0, 1.0), 4.2)
    slope_hi = gaussian_blur_1d(slope_raw, 2.2, axis=1)
    glint_band = np.exp(-(((xx / W) - 0.67) / 0.28) ** 2) * np.exp(-depth_f * 1.45)
    water_rgb += (slope_hi * glint_band * 0.36)[..., None] * np.array([0.46, 0.95, 0.86], dtype=np.float32)

    # Soft coastal mist blending the water smoothly into the glowing dawn horizon
    h_mist = np.exp(-((yy - hy_px) / 24.0) ** 2) * 0.22
    water_rgb = water_rgb * (1.0 - h_mist[..., None]) + fog_rgb[None, None, :] * h_mist[..., None]

    out = sky_canvas * (1.0 - water_mask[..., None]) + water_rgb * water_mask[..., None]
    return out, rip, rip2


# =============================================================================
# Stage 1 & 2 — Irregular Voronoi Stone Embankment & Sloped Basalt Sea-Wall UV
# =============================================================================

def paint_voronoi_embankment_uv(
    rng: np.random.Generator,
    x_span: tuple[float, float],
    y_span: tuple[float, float],
    is_slope: bool = False,
    tw: int = 512,
    th: int = 768,
) -> np.ndarray:
    """
    Irregular Voronoi coastal basalt embankment blocks (d2 - d1 over 3x3 cell
    neighborhood, jittered physical meters + bicubic noise perturbation) with
    balanced UV density so near stones are crisp and isotropic at 1:1 pixel zoom.
    """
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wx = x_span[0] + u * (x_span[1] - x_span[0])
    wy = y_span[0] + v * (y_span[1] - y_span[0])
    wz = np.full_like(wx, Z_BED) if not is_slope else (Z_BED * (1.0 - u) - 0.04 * u)

    nx = resize_bicubic(rng.standard_normal((max(4, th // 28), max(4, tw // 28))).astype(np.float32), tw, th) * 0.05
    ny = resize_bicubic(rng.standard_normal((max(4, th // 28), max(4, tw // 28))).astype(np.float32), tw, th) * 0.05
    px = wx + nx
    py = wy + ny

    cell_x, cell_y = (0.42, 0.56) if is_slope else (0.58, 0.68)
    ix = np.floor(px / cell_x).astype(np.int32)
    iy = np.floor(py / cell_y).astype(np.int32)

    jx_tab = rng.uniform(0.18, 0.82, size=256)
    jy_tab = rng.uniform(0.18, 0.82, size=256)
    tone_tab = rng.uniform(-0.032, 0.036, size=256)

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
    stone_mask = np.clip((edge_d - 0.010) / 0.022, 0.0, 1.0)
    bevel_hi = np.clip((edge_d - 0.015) / 0.020, 0.0, 1.0) * np.clip((0.056 - edge_d) / 0.038, 0.0, 1.0)

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=82)
    tooth = paper_tooth(th, tw, rng)
    sv = tone_tab[best_id].astype(np.float32)

    base = np.zeros((th, tw, 3), dtype=np.float32)
    base[:, :, 0] = 0.048 + sv * 0.48 + mottle * 0.012 + dabs * 0.009
    base[:, :, 1] = 0.078 + sv * 0.56 + mottle * 0.016 + dabs * 0.011
    base[:, :, 2] = 0.142 + sv * 0.64 + mottle * 0.019 + dabs * 0.013

    crevice = np.array([0.012, 0.022, 0.052], dtype=np.float32)
    rgb = stone_mask[..., None] * base + (1.0 - stone_mask[..., None]) * crevice
    rgb += bevel_hi[..., None] * np.array([0.034, 0.092, 0.098], dtype=np.float32)

    if not is_slope:
        ballast_band = np.clip(1.0 - np.abs(wx - X_TRACK_C) / 1.25, 0.0, 1.0).astype(np.float32)
        bal_col = np.array([0.038, 0.062, 0.115], dtype=np.float32)[None, None, :] + tooth[..., None] * 0.018
        rgb = rgb * (1.0 - 0.45 * ballast_band[..., None]) + bal_col * (0.45 * ballast_band[..., None])
        alpha = np.ones((th, tw), dtype=np.float32)
    else:
        wet_t = np.clip((u.astype(np.float32) - 0.42) / 0.44, 0.0, 1.0)
        rgb *= (1.0 - 0.40 * wet_t[..., None])
        rgb += ((1.0 - wet_t) * stone_mask)[..., None] * np.array([0.044, 0.115, 0.118], dtype=np.float32)
        shore_edge = 0.88 + 0.050 * np.sin(wy.astype(np.float32) * 1.05) + 0.035 * mottle
        foam = np.exp(-((u.astype(np.float32) - (shore_edge - 0.02)) / 0.032) ** 2) * (
            0.55 + 0.45 * np.clip(mottle + 0.5, 0.0, 1.0)
        )
        rgb += foam[..., None] * np.array([0.38, 0.86, 0.82], dtype=np.float32) * 0.46
        alpha = np.clip((shore_edge - u.astype(np.float32)) / 0.06, 0.0, 1.0)

    sh = trace_embankment_shadows(wx, wy, wz)
    rgb *= sh[..., None]

    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    rgb += np.clip((0.28 - lum) / 0.28, 0.0, 1.0)[..., None] * np.array([0.012, 0.011, 0.009], dtype=np.float32)
    rgb += tooth[..., None] * 0.006

    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.25) * alpha[..., None]
    rgba[:, :, 3] = alpha
    return rgba


# =============================================================================
# Stage 3 — 2x Supersampled PyCairo Helpers, Railway, Poles, Wires & Signal
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
    fade_dist: float = 26.0,
    taper: bool = True,
    tremor: float = 0.18,
) -> None:
    if len(pts) < 2:
        return
    decay = float(np.clip(1.0 / (1.0 + z_depth / (fade_dist * 0.45)), 0.25, 1.0))
    w_eff = base_w * decay
    ext = [pts[0]] + pts + [pts[-1]]
    dense: list[tuple[float, float]] = []
    steps = 5
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
        wt = (0.42 + 0.58 * math.sin(frac * math.pi)) if taper else 1.0
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
    stroke_w: float = 1.0,
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
    fade_dist: float = 26.0,
) -> None:
    cl = cam.clip_segment_3d(np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64))
    if cl is None:
        return
    ua, va, za = cam.project(cl[0])
    ub, vb, zb = cam.project(cl[1])
    z_mid = 0.5 * (float(za) + float(zb))
    decay = float(np.clip(1.0 / (1.0 + z_mid / (fade_dist * 0.45)), 0.25, 1.0))
    ctx.set_source_rgba(*rgba)
    ctx.set_line_width(max(0.45, base_w * decay))
    ctx.move_to(float(ua), float(va))
    ctx.line_to(float(ub), float(vb))
    ctx.stroke()


def fog_mix(col: tuple[float, float, float], dist: float, fog_rgb: np.ndarray, fog_start=8.0, fog_density=0.018):
    ft = 1.0 - math.exp(-max(0.0, dist - fog_start) * fog_density)
    return (
        col[0] * (1.0 - ft) + float(fog_rgb[0]) * ft,
        col[1] * (1.0 - ft) + float(fog_rgb[1]) * ft,
        col[2] * (1.0 - ft) + float(fog_rgb[2]) * ft,
    )


def draw_railway_sleepers_and_rails(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    y_min: float,
    y_max: float,
    draw_rails: bool = True,
) -> None:
    """Weathered timber railway sleepers & glinting twin steel rails receding seamlessly to the dawn horizon."""
    # Dense spacing near camera, gently widening stride in deep far mist (> 65 m) for speed
    ys_near = np.arange(0.65, 65.0, 0.68)
    ys_far = np.arange(65.0, 210.0, 1.9)
    ys = np.concatenate([ys_near, ys_far])
    ys = ys[(ys >= y_min) & (ys < y_max)][::-1]

    for y in ys:
        dist = float(math.hypot(X_TRACK_C - cam.eye[0], y - cam.eye[1]))
        jitter_x = float(rng.uniform(-0.04, 0.04))
        half_len = 1.24 + float(rng.uniform(-0.04, 0.04))
        w_tie = 0.23
        h_tie = 0.085
        xl = X_TRACK_C - half_len + jitter_x
        xr = X_TRACK_C + half_len + jitter_x

        tv = float(rng.uniform(-0.015, 0.018))
        top_c = fog_mix((0.068 + tv, 0.088 + tv * 1.1, 0.142 + tv * 1.2), dist, fog_rgb)
        front_c = fog_mix((0.026 + tv * 0.4, 0.036 + tv * 0.5, 0.076 + tv * 0.6), dist, fog_rgb)
        right_c = fog_mix((0.058 + tv * 0.6, 0.102 + tv * 0.9, 0.156 + tv * 0.9), dist, fog_rgb)

        front_q = [
            np.array([xl, y - w_tie * 0.5, Z_BED]),
            np.array([xr, y - w_tie * 0.5, Z_BED]),
            np.array([xr, y - w_tie * 0.5, Z_BED + h_tie]),
            np.array([xl, y - w_tie * 0.5, Z_BED + h_tie]),
        ]
        fill_3d_quad(ctx, cam, front_q, (*front_c, 0.98))

        if y < 55.0:
            right_q = [
                np.array([xr, y - w_tie * 0.5, Z_BED]),
                np.array([xr, y + w_tie * 0.5, Z_BED]),
                np.array([xr, y + w_tie * 0.5, Z_BED + h_tie]),
                np.array([xr, y - w_tie * 0.5, Z_BED + h_tie]),
            ]
            fill_3d_quad(ctx, cam, right_q, (*right_c, 0.98))

        top_q = [
            np.array([xl, y - w_tie * 0.5, Z_BED + h_tie]),
            np.array([xr, y - w_tie * 0.5, Z_BED + h_tie]),
            np.array([xr, y + w_tie * 0.5, Z_BED + h_tie]),
            np.array([xl, y + w_tie * 0.5, Z_BED + h_tie]),
        ]
        fill_3d_quad(
            ctx, cam, top_q, (*top_c, 0.98),
            stroke_rgba=(0.015, 0.024, 0.055, 0.55) if y < 18.0 else None,
            stroke_w=0.8,
        )

        if y < 22.0:
            hi_c = fog_mix((0.36, 0.80, 0.76), dist, fog_rgb)
            stroke_3d_seg(
                ctx, cam,
                np.array([xl + 0.05, y + w_tie * 0.32, Z_BED + h_tie + 0.004]),
                np.array([xr - 0.05, y + w_tie * 0.32, Z_BED + h_tie + 0.004]),
                1.4, (*hi_c, 0.44),
            )
            for rx in (X_RAIL_L, X_RAIL_R):
                plate = [
                    np.array([rx - 0.11, y - 0.10, Z_BED + h_tie + 0.006]),
                    np.array([rx + 0.11, y - 0.10, Z_BED + h_tie + 0.006]),
                    np.array([rx + 0.11, y + 0.10, Z_BED + h_tie + 0.006]),
                    np.array([rx - 0.11, y + 0.10, Z_BED + h_tie + 0.006]),
                ]
                fill_3d_quad(ctx, cam, plate, (0.020, 0.032, 0.068, 0.92))

    if not draw_rails:
        return

    # Twin Glinting Steel Rails receding cleanly to y = 260 m (melts poetically into the dawn horizon!)
    rail_segs = np.geomspace(0.35, 260.0, 32)
    for rx in (X_RAIL_L, X_RAIL_R):
        for i in range(len(rail_segs) - 1):
            ya, yb = float(rail_segs[i]), float(rail_segs[i + 1])
            dist = float(math.hypot(rx - cam.eye[0], 0.5 * (ya + yb) - cam.eye[1]))
            web_c = fog_mix((0.026, 0.044, 0.088), dist, fog_rgb)
            web_q = [
                np.array([rx + 0.032, ya, Z_BED + 0.082]),
                np.array([rx + 0.032, yb, Z_BED + 0.082]),
                np.array([rx + 0.032, yb, Z_RAIL_TOP]),
                np.array([rx + 0.032, ya, Z_RAIL_TOP]),
            ]
            fill_3d_quad(ctx, cam, web_q, (*web_c, 0.98))

            head_c = fog_mix((0.36, 0.72, 0.78), dist, fog_rgb)
            head_q = [
                np.array([rx - 0.036, ya, Z_RAIL_TOP]),
                np.array([rx + 0.036, ya, Z_RAIL_TOP]),
                np.array([rx + 0.036, yb, Z_RAIL_TOP]),
                np.array([rx - 0.036, yb, Z_RAIL_TOP]),
            ]
            fill_3d_quad(ctx, cam, head_q, (*head_c, 0.98))

            glint_c = fog_mix((0.86, 1.00, 0.95), dist, fog_rgb, fog_density=0.012)
            stroke_3d_seg(
                ctx, cam,
                np.array([rx + 0.014, ya, Z_RAIL_TOP + 0.005]),
                np.array([rx + 0.014, yb, Z_RAIL_TOP + 0.005]),
                2.4 if rx == X_RAIL_R else 2.1,
                (*glint_c, 0.94),
                fade_dist=38.0,
            )


def pole_wire_anchor(pole_spec: tuple[float, float, float, float, float], wire_idx: int, z_sign: float = 1.0) -> np.ndarray:
    px, py, ph, lx, ly = pole_spec
    if wire_idx == 0:
        z_a, dx_arm = ph - 0.50, -0.62
    elif wire_idx == 1:
        z_a, dx_arm = ph - 0.50, 0.68
    elif wire_idx == 2:
        z_a, dx_arm = ph - 1.15, 1.48
    else:
        z_a, dx_arm = ph - 1.62, 1.48
    return np.array([px + lx * z_a + dx_arm, py + ly * z_a, z_a * z_sign], dtype=np.float64)


def draw_utility_poles_and_catenary(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    pole_indices: list[int],
    mirrored: bool = False,
    alpha_scale: float = 1.0,
) -> None:
    """
    Draw leaning coastal utility poles with cantilevered catenary brackets &
    sagging 3D catenary wires. Every wire span is strictly forward of z_cam > 0.6 m
    so Catmull-Rom tangents NEVER overshoot across the frame!
    """
    z_sign = -1.0 if mirrored else 1.0

    for idx in pole_indices:
        if idx < len(POLE_SPECS) - 1:
            p_curr = POLE_SPECS[idx]
            p_next = POLE_SPECS[idx + 1]
            dist_mid = float(math.hypot(p_curr[0] - cam.eye[0], 0.5 * (p_curr[1] + p_next[1]) - cam.eye[1]))
            wire_c = fog_mix((0.025, 0.048, 0.105), dist_mid, fog_rgb)
            wire_hi = fog_mix((0.38, 0.80, 0.78), dist_mid, fog_rgb)
            for w_idx, sag in ((0, 0.34), (1, 0.38), (2, 0.44), (3, 0.16)):
                a = pole_wire_anchor(p_curr, w_idx, z_sign)
                b = pole_wire_anchor(p_next, w_idx, z_sign)
                n_s = 13
                pts_2d: list[tuple[float, float]] = []
                z_acc = dist_mid
                for s in range(n_s):
                    t = s / (n_s - 1)
                    q = a * (1.0 - t) + b * t + np.array([0.0, 0.0, -sag * 4.0 * t * (1.0 - t) * z_sign])
                    uq, vq, zq = cam.project(q)
                    if float(zq) > 0.5:
                        pts_2d.append((float(uq), float(vq)))
                        z_acc = float(zq)
                if len(pts_2d) >= 2:
                    bw = 2.0 if w_idx in (2, 3) else 1.55
                    if not mirrored:
                        stroke_catmull_rom(
                            ctx, pts_2d, base_w=bw, z_depth=z_acc,
                            rgba=(*wire_c, 0.88 * alpha_scale), rng=rng, fade_dist=32.0, taper=False, tremor=0.08,
                        )
                        ctx.set_source_rgba(*wire_hi, 0.28 * alpha_scale)
                        ctx.set_line_width(max(0.45, 0.85 * float(np.clip(1.0 / (1.0 + z_acc / 14.0), 0.25, 1.0))))
                        ctx.move_to(*pts_2d[0])
                        for pp in pts_2d[1:]:
                            ctx.line_to(*pp)
                        ctx.stroke()
                    else:
                        ctx.set_source_rgba(*wire_c, 0.46 * alpha_scale)
                        ctx.set_line_width(max(0.45, bw * float(np.clip(1.0 / (1.0 + z_acc / 14.0), 0.25, 1.0))))
                        ctx.move_to(*pts_2d[0])
                        for pp in pts_2d[1:]:
                            ctx.line_to(*pp)
                        ctx.stroke()

            if not mirrored and idx < 4:
                a2, b2 = pole_wire_anchor(p_curr, 2, z_sign), pole_wire_anchor(p_next, 2, z_sign)
                a3, b3 = pole_wire_anchor(p_curr, 3, z_sign), pole_wire_anchor(p_next, 3, z_sign)
                for frac in (0.25, 0.50, 0.75):
                    pt_top = a2 * (1 - frac) + b2 * frac + np.array([0, 0, -0.44 * 4 * frac * (1 - frac)])
                    pt_bot = a3 * (1 - frac) + b3 * frac + np.array([0, 0, -0.16 * 4 * frac * (1 - frac)])
                    stroke_3d_seg(ctx, cam, pt_top, pt_bot, 0.9, (*wire_c, 0.60 * alpha_scale), fade_dist=24.0)

        # Safe overhead forward span from Pole 0 toward upper-left edge (sampled ONLY for y >= 0.4 m, well in front of camera!)
        if idx == 0 and not mirrored:
            p0 = POLE_SPECS[0]
            p_prev = (-1.88, -2.2, 5.60, 0.055, 0.012)
            for w_idx, sag in ((0, 0.30), (1, 0.34), (2, 0.38), (3, 0.14)):
                a = pole_wire_anchor(p_prev, w_idx, 1.0)
                b = pole_wire_anchor(p0, w_idx, 1.0)
                pts_2d = []
                # Start strictly at t = 0.42 (y >= 0.7 m, z_cam >= 2.2 m) so screen coords stay completely bounded!
                for s in range(10):
                    t = 0.42 + 0.58 * (s / 9.0)
                    q = a * (1.0 - t) + b * t + np.array([0.0, 0.0, -sag * 4.0 * t * (1.0 - t)])
                    uq, vq, zq = cam.project(q)
                    if float(zq) > 1.2 and -400.0 < float(uq) < W + 400.0 and -400.0 < float(vq) < H + 400.0:
                        pts_2d.append((float(uq), float(vq)))
                if len(pts_2d) >= 2:
                    stroke_catmull_rom(
                        ctx, pts_2d, base_w=2.3, z_depth=3.5,
                        rgba=(0.018, 0.036, 0.084, 0.90), rng=rng, taper=False, tremor=0.10,
                    )

    # Draw each leaning utility pole shaft, crossarms, cantilever boom & insulators
    for idx in pole_indices:
        px, py, ph, lx, ly = POLE_SPECS[idx]
        base = np.array([px, py, (Z_BED - 0.08) * z_sign], dtype=np.float64)
        top = np.array([px + lx * ph, py + ly * ph, ph * z_sign], dtype=np.float64)
        dist = float(math.hypot(px - cam.eye[0], py - cam.eye[1]))

        pole_c = fog_mix((0.022, 0.042, 0.094), dist, fog_rgb)
        rim_c = fog_mix((0.42, 0.90, 0.80), dist, fog_rgb)

        r_bot, r_top = 0.115, 0.072
        shaft = [
            base + np.array([-r_bot, 0.0, 0.0]),
            base + np.array([r_bot, 0.0, 0.0]),
            top + np.array([r_top, 0.0, 0.0]),
            top + np.array([-r_top, 0.0, 0.0]),
        ]
        fill_3d_quad(ctx, cam, shaft, (*pole_c, 0.96 * alpha_scale))
        stroke_3d_seg(
            ctx, cam,
            base + np.array([r_bot * 0.88, 0.0, 0.0]),
            top + np.array([r_top * 0.88, 0.0, 0.0]),
            2.4, (*rim_c, 0.72 * alpha_scale), fade_dist=28.0,
        )

        z_arm1 = ph - 0.50
        arm1_c = np.array([px + lx * z_arm1, py + ly * z_arm1, z_arm1 * z_sign])
        stroke_3d_seg(ctx, cam, arm1_c + np.array([-0.74, 0, 0]), arm1_c + np.array([0.82, 0, 0]), 5.6, (*pole_c, 0.96 * alpha_scale))
        stroke_3d_seg(ctx, cam, arm1_c + np.array([-0.74, 0, 0.03 * z_sign]), arm1_c + np.array([0.82, 0, 0.03 * z_sign]), 1.4, (*rim_c, 0.58 * alpha_scale))

        z_br = ph - 0.98
        br_pt = np.array([px + lx * z_br, py + ly * z_br, z_br * z_sign])
        stroke_3d_seg(ctx, cam, br_pt, arm1_c + np.array([-0.46, 0, 0]), 2.2, (*pole_c, 0.92 * alpha_scale))
        stroke_3d_seg(ctx, cam, br_pt, arm1_c + np.array([0.52, 0, 0]), 2.2, (*pole_c, 0.92 * alpha_scale))

        z_cat_lo = ph - 1.72
        z_cat_hi = ph - 1.05
        cat_lo = np.array([px + lx * z_cat_lo, py + ly * z_cat_lo, z_cat_lo * z_sign])
        cat_hi = np.array([px + lx * z_cat_hi, py + ly * z_cat_hi, z_cat_hi * z_sign])
        cat_tip = pole_wire_anchor(POLE_SPECS[idx], 2, z_sign)
        stroke_3d_seg(ctx, cam, cat_lo, cat_tip, 3.5, (*pole_c, 0.95 * alpha_scale))
        stroke_3d_seg(ctx, cam, cat_hi, cat_tip, 2.2, (*pole_c, 0.90 * alpha_scale))
        stroke_3d_seg(ctx, cam, cat_lo, cat_lo + np.array([1.54, 0, 0.08 * z_sign]), 2.3, (*pole_c, 0.90 * alpha_scale))

        if not mirrored and idx < 4:
            for w_i in (0, 1, 2):
                ip = pole_wire_anchor(POLE_SPECS[idx], w_i, z_sign)
                ui, vi, zi = cam.project(ip)
                if zi > Z_NEAR:
                    rad_i = max(1.4, 24.0 / float(zi))
                    ctx.set_source_rgba(0.60, 0.96, 0.89, 0.85)
                    ctx.arc(float(ui), float(vi), rad_i, 0, math.tau)
                    ctx.fill()


def draw_distant_signal_lamp(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    mirrored: bool = False,
    alpha_scale: float = 1.0,
) -> None:
    z_sign = -1.0 if mirrored else 1.0
    base = np.array([SIGNAL_POS[0], SIGNAL_POS[1], -0.05 * z_sign], dtype=np.float64)
    top = np.array([SIGNAL_POS[0], SIGNAL_POS[1], (SIGNAL_LAMP_Z + 0.55) * z_sign], dtype=np.float64)
    lamp_p = np.array([SIGNAL_POS[0] - 0.14, SIGNAL_POS[1], SIGNAL_LAMP_Z * z_sign], dtype=np.float64)

    dist = float(math.hypot(SIGNAL_POS[0] - cam.eye[0], SIGNAL_POS[1] - cam.eye[1]))
    post_c = fog_mix((0.022, 0.042, 0.095), dist, fog_rgb)
    rim_c = fog_mix((0.46, 0.92, 0.82), dist, fog_rgb)

    pier = [
        np.array([SIGNAL_POS[0] - 0.18, SIGNAL_POS[1], -0.05 * z_sign]),
        np.array([SIGNAL_POS[0] + 0.18, SIGNAL_POS[1], -0.05 * z_sign]),
        np.array([SIGNAL_POS[0] + 0.14, SIGNAL_POS[1], 0.42 * z_sign]),
        np.array([SIGNAL_POS[0] - 0.14, SIGNAL_POS[1], 0.42 * z_sign]),
    ]
    fill_3d_quad(ctx, cam, pier, (*post_c, 0.94 * alpha_scale))
    stroke_3d_seg(ctx, cam, base, top, 5.2, (*post_c, 0.96 * alpha_scale), fade_dist=34.0)
    stroke_3d_seg(
        ctx, cam, base + np.array([0.04, 0, 0]), top + np.array([0.04, 0, 0]),
        1.5, (*rim_c, 0.72 * alpha_scale), fade_dist=34.0,
    )

    u_l, v_l, z_l = cam.project(lamp_p)
    if z_l <= Z_NEAR:
        return
    lx, ly = float(u_l), float(v_l)
    sc = max(0.55, 22.0 / float(z_l))

    ctx.set_source_rgba(*post_c, 0.96 * alpha_scale)
    ctx.rectangle(lx - 7.5 * sc, ly - 15.0 * sc, 15.0 * sc, 30.0 * sc)
    ctx.fill()

    for r_g, rgba_g in (
        (78.0 * sc, (0.38, 0.94, 0.78, 0.28 * alpha_scale)),
        (34.0 * sc, (0.62, 0.99, 0.88, 0.54 * alpha_scale)),
        (13.0 * sc, (0.88, 1.00, 0.96, 0.88 * alpha_scale)),
    ):
        g = cairo.RadialGradient(lx, ly, 0.5, lx, ly, r_g)
        g.add_color_stop_rgba(0.0, *rgba_g)
        g.add_color_stop_rgba(1.0, rgba_g[0], rgba_g[1], rgba_g[2], 0.0)
        ctx.set_source(g)
        ctx.arc(lx, ly, r_g, 0, math.tau)
        ctx.fill()

    ctx.set_source_rgba(0.96, 1.00, 0.98, 0.98 * alpha_scale)
    ctx.arc(lx, ly, 4.6 * sc, 0, math.tau)
    ctx.fill()


def draw_tidal_water_markers(
    ctx: cairo.Context,
    cam: Camera3D,
    fog_rgb: np.ndarray,
    mirrored: bool = False,
    alpha_scale: float = 1.0,
) -> None:
    z_sign = -1.0 if mirrored else 1.0
    stakes = [
        (4.35, 13.8, 1.05, 0.08),
        (5.10, 15.2, 0.78, -0.05),
        (7.40, 28.5, 1.15, 0.06),
    ]
    for sx, sy, sh, tilt in stakes:
        dist = float(math.hypot(sx - cam.eye[0], sy - cam.eye[1]))
        col = fog_mix((0.025, 0.048, 0.104), dist, fog_rgb)
        rim = fog_mix((0.40, 0.88, 0.78), dist, fog_rgb)
        p0 = np.array([sx, sy, -0.03 * z_sign])
        p1 = np.array([sx + tilt * sh, sy, sh * z_sign])
        stroke_3d_seg(ctx, cam, p0, p1, 6.0, (*col, 0.94 * alpha_scale), fade_dist=28.0)
        stroke_3d_seg(ctx, cam, p0 + np.array([0.03, 0, 0]), p1 + np.array([0.03, 0, 0]), 1.6, (*rim, 0.65 * alpha_scale), fade_dist=28.0)
        if not mirrored:
            u0, v0, z0 = cam.project(np.array([sx, sy, 0.01]))
            if z0 > Z_NEAR:
                rw = max(5.0, 145.0 / float(z0))
                ctx.save()
                ctx.translate(float(u0), float(v0))
                ctx.scale(rw, rw * 0.22)
                ctx.set_source_rgba(0.45, 0.92, 0.82, 0.42)
                ctx.set_line_width(1.6 / rw)
                ctx.arc(0.0, 0.0, 1.0, 0, math.tau)
                ctx.stroke()
                ctx.restore()


# =============================================================================
# Stage 2 — Composite 3D Plane-Mirrored Reflections Onto Tidal Water
# =============================================================================

def composite_mirrored_structures_on_water(
    canvas: np.ndarray,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    rip: np.ndarray,
    rip2: np.ndarray,
) -> np.ndarray:
    s_mir, c_mir = new_cairo_2x()
    draw_utility_poles_and_catenary(c_mir, cam, rng, fog_rgb, list(range(len(POLE_SPECS))), mirrored=True, alpha_scale=0.80)
    draw_distant_signal_lamp(c_mir, cam, rng, fog_rgb, mirrored=True, alpha_scale=0.96)
    draw_tidal_water_markers(c_mir, cam, fog_rgb, mirrored=True, alpha_scale=0.80)
    mir_1x = finish_cairo_2x(s_mir)

    lamp_base = np.array([SIGNAL_POS[0] - 0.14, SIGNAL_POS[1], 0.0], dtype=np.float64)
    ub, vb, _ = cam.project(lamp_base)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    depth_f = np.clip((yy - cam.horizon_y) / max(1.0, H - cam.horizon_y), 0.0, 1.0)
    mx = np.clip(xx + rip * (2.6 + 5.2 * depth_f), 0.0, W - 1.0)
    my = np.clip(yy + rip * (1.4 + 2.4 * depth_f), 0.0, H - 1.0)
    mir_rip = remap_bilinear(mir_1x, mx, my)

    water_mask = np.clip((yy - cam.horizon_y) / 2.5, 0.0, 1.0)
    dist_t = np.power(depth_f, 0.62)
    fresnel = 0.30 + (0.93 - 0.30) * np.power(1.0 - dist_t, 1.4)

    mir_rip[:, :, :3] *= (water_mask * fresnel)[..., None]
    mir_rip[:, :, 3] *= (water_mask * fresnel)
    out = over_premul(canvas, mir_rip)

    # Shimmering broken-water vertical reflection column directly beneath the glowing mint-white signal lamp!
    bx_s, by_s = float(ub) + 2.0, float(vb)
    dy_s = np.maximum(0.0, yy - by_s)
    col_w = 5.5 + 0.060 * dy_s + np.abs(rip) * 3.6
    dx_s = xx - (bx_s + rip * (2.6 + 0.025 * dy_s))
    pillar = (
        np.exp(-0.5 * (dx_s / col_w) ** 2)
        * np.exp(-dy_s / 280.0)
        * (yy >= by_s - 2.0)
        * np.clip(0.64 + 0.42 * np.sin(dy_s * 0.52 + rip2 * 2.0), 0.20, 1.25)
    )
    pillar_rgb = pillar[..., None] * np.array([0.58, 1.00, 0.90], dtype=np.float32) * 0.86
    return screen_blend(out, pillar_rgb)


# =============================================================================
# Stage 4 — Foreground Circular Disc Bokeh Coastal Reeds & 35mm Analog Optics
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


def paint_foreground_bokeh_coastal_reeds(rng: np.random.Generator) -> np.ndarray:
    """
    Close out-of-focus coastal reeds & nodding feathered plumes framing the
    lower-right corner via per-instance seeded cubic-Bezier stalks (varying
    length, droop, and silhouette nicks), convolved in premultiplied RGBA with
    a circular aperture disc bokeh kernel (r = 7.8) + 1 px opposite R/B shift.
    """
    surf, ctx = new_cairo_2x()

    ctx.set_source_rgba(0.010, 0.022, 0.056, 0.96)
    ctx.move_to(W + 30, H + 30)
    ctx.line_to(W - 410, H + 30)
    ctx.curve_to(W - 300, H - 52, W - 150, H - 88, W + 30, H - 120)
    ctx.close_path()
    ctx.fill()

    for i in range(38):
        bx = float(rng.uniform(W - 360, W + 35))
        by = float(H + rng.uniform(-25, 35))
        edge_bias = np.clip((bx - (W - 360)) / 380.0, 0.15, 1.0)
        length = float(rng.uniform(190.0, 485.0)) * (0.55 + 0.55 * edge_bias)
        lean = float(rng.uniform(-230.0, -35.0))
        droop = float(rng.uniform(-42.0, 82.0))
        w_base = float(rng.uniform(6.0, 14.0))

        p0 = (bx, by)
        p1 = (bx + lean * 0.28, by - length * 0.44)
        p2 = (bx + lean * 0.76, by - length * 0.86)
        p3 = (bx + lean, by - length + droop)

        nick = float(rng.uniform(-3.0, 3.0))
        ctx.set_source_rgba(0.012, 0.026, 0.062, 0.95)
        ctx.move_to(p0[0] - w_base * 0.5, p0[1])
        ctx.curve_to(p1[0] - w_base * 0.36 + nick, p1[1], p2[0] - w_base * 0.15, p2[1], p3[0], p3[1])
        ctx.curve_to(p2[0] + w_base * 0.15, p2[1], p1[0] + w_base * 0.36, p1[1], p0[0] + w_base * 0.5, p0[1])
        ctx.close_path()
        ctx.fill()

        if rng.random() < 0.45:
            ctx.set_source_rgba(0.32, 0.82, 0.76, 0.48)
            ctx.set_line_width(2.5)
            ctx.move_to(p0[0] - w_base * 0.48, p0[1])
            ctx.curve_to(p1[0] - w_base * 0.36, p1[1], p2[0] - w_base * 0.15, p2[1], p3[0], p3[1])
            ctx.stroke()

        if i % 4 == 0 and length > 290:
            tx, ty = p3
            for k_pl in range(11):
                ang = float(rng.uniform(-2.75, -1.05))
                plen = float(rng.uniform(22.0, 56.0))
                ox = tx - (p3[0] - p2[0]) * (k_pl * 0.045)
                oy = ty - (p3[1] - p2[1]) * (k_pl * 0.045)
                ex = ox + math.cos(ang) * plen
                ey = oy + math.sin(ang) * plen
                ctx.set_source_rgba(0.018, 0.040, 0.088, 0.88)
                ctx.set_line_width(3.4)
                ctx.move_to(ox, oy)
                ctx.curve_to(ox + math.cos(ang) * plen * 0.5, oy + math.sin(ang) * plen * 0.4 - 6, ex, ey + 6, ex, ey + 10)
                ctx.stroke()
                if k_pl % 2 == 0:
                    ctx.set_source_rgba(0.42, 0.88, 0.80, 0.52)
                    ctx.set_line_width(1.6)
                    ctx.move_to(ox, oy)
                    ctx.line_to(ex, ey + 8)
                    ctx.stroke()

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
    tint = np.array([0.88, 1.08, 1.04], dtype=np.float32)
    hal = (
        0.24 * gaussian_blur_2d(hl_src, 6.5)
        + 0.17 * gaussian_blur_2d(hl_src, 24.0)
        + 0.11 * gaussian_blur_2d(hl_src, 72.0)
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
    seed: int = 202602,
    audit_dir: pathlib.Path | None = None,
    webp_outputs: list[pathlib.Path] | None = None,
) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # Stage 1: Off-axis Camera3D (vanishing point ~62.7% width, horizon ~50.9% height)
    cam = Camera3D(W, H, eye=(1.55, -1.5, 1.38), target=(-0.55, 14.0, 1.46), fov_x_deg=56.0)
    print(f"[aria-02] Camera VP: ({cam.vp_x:.1f}, {cam.vp_y:.1f}) -> {cam.vp_x / W * 100:.1f}% width, {cam.horizon_y / H * 100:.1f}% height")

    canvas, sky_for_refl, fog_rgb, m_gouache = paint_sky_dawn_horizon(cam, rng)

    # Stage 2: Vast Fresnel capillary-rippled starlit tidal water + 3D plane-mirrored reflections
    canvas, rip, rip2 = paint_vast_tidal_water(canvas, sky_for_refl, cam, rng, fog_rgb, m_gouache)
    canvas = composite_mirrored_structures_on_water(canvas, cam, rng, fog_rgb, rip, rip2)

    # Stage 1 & 2: Two-zone Voronoi basalt embankment (near crisp y=0.2..28m + far seamless mist dissolution to horizon!)
    near_top_uv = paint_voronoi_embankment_uv(rng, (-5.2, 0.98), (0.2, 28.0), is_slope=False, tw=512, th=768)
    near_slope_uv = paint_voronoi_embankment_uv(rng, (0.98, 1.92), (0.2, 28.0), is_slope=True, tw=384, th=768)
    far_top_uv = paint_voronoi_embankment_uv(rng, (-5.2, 0.98), (27.8, 420.0), is_slope=False, tw=256, th=384)
    far_slope_uv = paint_voronoi_embankment_uv(rng, (0.98, 1.92), (27.8, 420.0), is_slope=True, tw=192, th=384)
    # Dissolve far embankment softly into the misty horizon past y ~ 150 m (zero boxy cut-off edge!)
    fy = np.linspace(0.0, 1.0, far_top_uv.shape[0], dtype=np.float32)[:, None]
    far_fade = np.clip((1.0 - fy) / 0.62, 0.0, 1.0)
    far_top_uv *= far_fade[:, :, None]
    far_slope_uv *= far_fade[:, :, None]

    for uv_tex, p0_v, e1_v, e2_v, fd in (
        (far_slope_uv, np.array([0.98, 27.8, Z_BED]), np.array([0.94, 0.0, -0.28]), np.array([0.0, 392.2, 0.0]), 0.024),
        (far_top_uv, np.array([-5.2, 27.8, Z_BED]), np.array([6.18, 0.0, 0.0]), np.array([0.0, 392.2, 0.0]), 0.024),
        (near_slope_uv, np.array([0.98, 0.2, Z_BED]), np.array([0.94, 0.0, -0.28]), np.array([0.0, 27.8, 0.0]), 0.018),
        (near_top_uv, np.array([-5.2, 0.2, Z_BED]), np.array([6.18, 0.0, 0.0]), np.array([0.0, 27.8, 0.0]), 0.018),
    ):
        canvas = cam.warp_plane_over(
            canvas, uv_tex, p0_v, e1_v, e2_v,
            fog_rgb=fog_rgb, fog_start=8.0, fog_density=fd,
        )

    # Stage 3 Band 0 (Far Railway, Distant Glowing Mint-White Signal Lamp & Far Utility Poles, y >= 22 m)
    s_b0, c_b0 = new_cairo_2x()
    draw_railway_sleepers_and_rails(c_b0, cam, rng, fog_rgb, y_min=22.0, y_max=220.0, draw_rails=False)
    draw_utility_poles_and_catenary(c_b0, cam, rng, fog_rgb, [7, 6, 5, 4, 3], mirrored=False)
    draw_distant_signal_lamp(c_b0, cam, rng, fog_rgb, mirrored=False)
    draw_tidal_water_markers(c_b0, cam, fog_rgb, mirrored=False)
    canvas = over_premul(canvas, finish_cairo_2x(s_b0))

    # Stage 3 Band 1 (Midground & Near Timber Sleepers + Continuous Glinting Twin Steel Rails)
    s_b1, c_b1 = new_cairo_2x()
    draw_railway_sleepers_and_rails(c_b1, cam, rng, fog_rgb, y_min=0.4, y_max=22.0, draw_rails=True)
    b1_1x = finish_cairo_2x(s_b1)
    a1 = b1_1x[:, :, 3:4]
    b1_1x[:, :, :3] = np.clip(b1_1x[:, :, :3] + a1 * (m_gouache * 0.012)[..., None], 0.0, 1.2)
    canvas = over_premul(canvas, b1_1x)

    # Stage 3 Band 2 (Mid & Near Leaning Utility Poles + Clean Bounded Overhead Catenary Wires)
    s_b2, c_b2 = new_cairo_2x()
    draw_utility_poles_and_catenary(c_b2, cam, rng, fog_rgb, [2, 1, 0], mirrored=False)
    b2_1x = finish_cairo_2x(s_b2)
    a2 = b2_1x[:, :, 3:4]
    b2_1x[:, :, :3] = np.clip(b2_1x[:, :, :3] + a2 * (m_gouache * 0.010)[..., None], 0.0, 1.2)
    canvas = over_premul(canvas, b2_1x)

    # Stage 4: Foreground Circular Disc Bokeh Coastal Reeds + 35mm Analog Finish
    fg_bokeh = paint_foreground_bokeh_coastal_reeds(rng)
    canvas = over_premul(canvas, fg_bokeh)
    final_rgb = apply_35mm_finish(canvas, rng)

    elapsed = time.perf_counter() - t0
    print(f"[aria-02] Rendered {W}x{H} in {elapsed:.2f}s")

    u8 = np.round(np.clip(final_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    full_im = Image.fromarray(u8, mode="RGB")

    if webp_outputs:
        for wp in webp_outputs:
            wp.parent.mkdir(parents=True, exist_ok=True)
            full_im.save(wp, format="WEBP", quality=90, method=6)
            print(f"[aria-02] Exported WebP: {wp} ({wp.stat().st_size / 1024:.1f} KiB)")

    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        full_im.save(audit_dir / "w02_full.png", optimize=True)
        crops = {
            "w02_crop_aperture.png": (int(W * 0.50), int(H * 0.25)),      # Glowing dawn horizon + distant signal lamp
            "w02_crop_midground.png": (int(W * 0.14), int(H * 0.54)),     # Weathered sleepers, glinting steel rails & basalt stones
            "w02_crop_foreground.png": (W - 600, H - 450),                # Lower-right circular-bokeh coastal reeds + silky tidal water
        }
        for cname, (cx0, cy0) in crops.items():
            x0 = max(0, min(W - 600, cx0))
            y0 = max(0, min(H - 450, cy0))
            full_im.crop((x0, y0, x0 + 600, y0 + 450)).save(audit_dir / cname)

    return final_rgb, {"elapsed": elapsed, "vp_ratio": cam.vp_x / W}


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    default_webps = [
        repo_root / "session" / "themes" / "aria" / "walls" / "wall-02.webp",
        repo_root / "docs" / "assets" / "walls" / "aria-02.webp",
    ]
    ap = argparse.ArgumentParser(description="Render Aria 02 — Starlit Tidal Railway (2048x1152)")
    ap.add_argument("--seed", type=int, default=202602)
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

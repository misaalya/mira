#!/usr/bin/env python3
"""
Mira 02 — "Monolith Reflecting Atrium" (2048 x 1152)

Pure-Python 3D-grounded hand-painted wallpaper pipeline (zero image models,
zero external image assets, 100% deterministic):
  Stage 1: Off-axis metric Camera3D (VP ~58.6% width), exact near-plane-safe
           3x3 UV-to-screen homographies with z_cam > 0.05 masking and
           exponential twilight-mist fog, plus high-resolution irregular
           Voronoi (d2 - d1) wet basalt stone paving and board-formed
           brutalist concrete walls.
  Stage 2: Unified normalized 3D twilight skylight vector (SUN_DIR, z < 0)
           traced backward with 6-step stepped-beam height refinement through
           the slatted concrete skylight roof onto floor, walls & volumetric
           light shafts, paired with Fresnel capillary-rippled water canal &
           infinity horizon reflections plus exact 3D plane-mirrored colonnade
           & acid-yellow (#d9d84a) recessed beacon / warm slit window reflections.
  Stage 3: 4 depth-interleaved 2x supersampled PyCairo bands (Lanczos-downsampled)
           with monumental off-axis board-formed concrete pillars, transverse
           skylight louvre beams, recessed beacon plinth cleanly seated at the
           waterline, monolithic stone bench, and isotropic fbm_gouache mottling,
           flat-brush dabs, cold-press paper tooth & warm shadow bounce.
  Stage 4: Foreground brutalist concrete louvre silhouette framing the upper-left
           corner convolved in premultiplied RGBA with a circular aperture disc
           bokeh kernel (r = 7.8) + 1 px opposite R/B shift, followed by 35mm
           multi-scale halation, radial CA, optical vignette, shadow black lift
           + C1 tanh highlight shoulder, luminance-weighted silver-halide grain,
           and faint analog YIQ transfer patina.
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

# Mira monochrome obsidian + single luminous acid-yellow (#d9d84a) light accent
ACCENT_YELLOW = np.array([0.851, 0.847, 0.290], dtype=np.float32)  # #d9d84a
ACCENT_CORE = np.array([0.968, 0.962, 0.650], dtype=np.float32)
ACCENT_WARM = np.array([0.765, 0.745, 0.215], dtype=np.float32)

# Stage 2: Unified normalized 3D twilight skylight travel vector (z < 0)
_raw_sun = np.array([-0.40, -0.14, -0.906], dtype=np.float64)
SUN_DIR = _raw_sun / np.linalg.norm(_raw_sun)

# Architectural world geometry (meters)
Z_WATER = 0.0
Z_WALK = 0.09
Z_ROOF_BOT = 5.15
Z_ROOF_TOP = 5.68
X_WALL_L = -3.85
X_CANAL_L = -0.45
X_WALL_R = 3.28
Y_OPEN_END = 32.0

# Colonnade pillar geometry (aligned along X in [-1.04, -0.46], right at water's edge!)
PILLAR_X0, PILLAR_X1 = -1.04, -0.46
PILLAR_YS = [5.15, 9.05, 12.95, 16.85, 20.75, 24.65, 28.55]
PILLAR_HALF_Y = 0.29

# Transverse slatted skylight concrete beams
BEAM_PITCH = 1.85
BEAM_HALF_Y = 0.34

# Hero single acid-yellow (#d9d84a) architectural light complex on the right canal monolith
SLIT_Y0, SLIT_Y1 = 15.10, 15.85
SLIT_Z0, SLIT_Z1 = 0.55, 4.35
SLIT_POS = np.array([X_WALL_R - 0.04, 0.5 * (SLIT_Y0 + SLIT_Y1), 2.35], dtype=np.float64)
BEACON_X0, BEACON_X1 = X_WALL_R - 0.52, X_WALL_R + 0.02
BEACON_Y0, BEACON_Y1 = 12.65, 13.45
BEACON_POS = np.array([X_WALL_R - 0.38, 13.05, 0.62], dtype=np.float64)


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
        rx = float(rng.uniform(qw * 0.035, qw * 0.105))
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
    def __init__(self, w: int, h: int, eye: Iterable[float], target: Iterable[float], fov_x_deg: float = 57.0):
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
        fog_start: float = 8.0,
        fog_density: float = 0.042,
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
# Stage 2 — Unified 3D Slatted Skylight Ray Tracer & Acid-Yellow Light Solver
# =============================================================================

def trace_skylight_and_colonnade(wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> np.ndarray:
    """
    Trace any 3D point backward along -SUN_DIR through the stepped slatted concrete
    skylight roof (6 height-refinement steps across beam depth) and concrete
    colonnade pillars. Returns [0..1] crisp geometric skylight illumination factor.
    """
    inv_sz = 1.0 / (-SUN_DIR[2])

    z_hit = np.full_like(wx, Z_ROOF_BOT, dtype=np.float64)
    for _ in range(6):
        t = np.maximum(0.0, (z_hit - wz) * inv_sz)
        ry = wy - SUN_DIR[1] * t
        phase = np.abs(np.mod(ry + 0.5 * BEAM_PITCH, BEAM_PITCH) - 0.5 * BEAM_PITCH)
        in_beam = phase < BEAM_HALF_Y
        z_hit = np.where(in_beam, Z_ROOF_BOT, Z_ROOF_TOP)

    t_bot = np.maximum(0.0, (Z_ROOF_BOT - wz) * inv_sz)
    ry_bot = wy - SUN_DIR[1] * t_bot
    t_top = np.maximum(0.0, (Z_ROOF_TOP - wz) * inv_sz)
    ry_top = wy - SUN_DIR[1] * t_top
    rx_top = wx - SUN_DIR[0] * t_top

    ph_bot = np.abs(np.mod(ry_bot + 0.5 * BEAM_PITCH, BEAM_PITCH) - 0.5 * BEAM_PITCH)
    ph_top = np.abs(np.mod(ry_top + 0.5 * BEAM_PITCH, BEAM_PITCH) - 0.5 * BEAM_PITCH)

    slat_open_bot = np.clip((ph_bot - (BEAM_HALF_Y - 0.04)) / 0.12, 0.0, 1.0)
    slat_open_top = np.clip((ph_top - (BEAM_HALF_Y - 0.04)) / 0.12, 0.0, 1.0)
    slat_open = np.minimum(slat_open_bot, slat_open_top).astype(np.float32)

    in_skylight_x = np.clip((rx_top + 2.65) / 0.25, 0.0, 1.0) * np.clip((2.85 - rx_top) / 0.25, 0.0, 1.0)
    in_skylight_y = np.clip((ry_top - 2.4) / 0.3, 0.0, 1.0) * np.clip((Y_OPEN_END - ry_top) / 0.3, 0.0, 1.0)
    t_rw = (X_WALL_R - wx) / (-SUN_DIR[0])
    z_at_rw = wz + t_rw * (-SUN_DIR[2])
    blocked_rw = (t_rw > 0.01) & (z_at_rw < Z_ROOF_BOT)

    sky_vis = slat_open * (in_skylight_x * in_skylight_y).astype(np.float32)
    sky_vis = np.where(blocked_rw, 0.0, sky_vis)

    for py_c in PILLAR_YS:
        t_p0 = (PILLAR_X0 - wx) / (-SUN_DIR[0])
        t_p1 = (PILLAR_X1 - wx) / (-SUN_DIR[0])
        tmin = np.minimum(t_p0, t_p1)
        tmax = np.maximum(t_p0, t_p1)
        t_mid = np.maximum(0.0, 0.5 * (tmin + tmax))
        y_at_p = wy - SUN_DIR[1] * t_mid
        z_at_p = wz - SUN_DIR[2] * t_mid
        hit_p = (tmax > 0.02) & (t_mid > 0.0) & (np.abs(y_at_p - py_c) < PILLAR_HALF_Y) & (z_at_p < Z_ROOF_BOT) & (wx < PILLAR_X0 + 0.02)
        sky_vis = np.where(hit_p, sky_vis * 0.14, sky_vis)

    return np.clip(sky_vis, 0.0, 1.0).astype(np.float32)


def eval_beacon_and_slit_light(
    wx: np.ndarray,
    wy: np.ndarray,
    wz: np.ndarray,
    nx: float,
    ny: float,
    nz: float,
) -> np.ndarray:
    """Warm acid-yellow (#d9d84a) irradiance from the recessed water-edge beacon and tall slit window."""
    bx = BEACON_POS[0] - wx
    by = BEACON_POS[1] - wy
    bz = BEACON_POS[2] - wz
    bd2 = bx * bx + by * by + bz * bz + 0.25
    bd = np.sqrt(bd2)
    b_ndotl = np.clip((bx * nx + by * ny + bz * nz) / bd, 0.0, 1.0)
    irr_b = (2.15 / (bd2 + 0.65)) * (0.25 + 0.75 * b_ndotl)

    sx = SLIT_POS[0] - wx
    sy = SLIT_POS[1] - wy
    sz = np.clip(wz, SLIT_Z0, SLIT_Z1) - wz
    sd2 = sx * sx + sy * sy + sz * sz + 0.35
    sd = np.sqrt(sd2)
    s_ndotl = np.clip((sx * nx + sy * ny + sz * nz) / sd, 0.0, 1.0)
    irr_s = (2.40 / (sd2 + 1.05)) * (0.20 + 0.80 * s_ndotl)

    return np.clip(irr_b + irr_s, 0.0, 2.0).astype(np.float32)


def mirror_points_water(pts: np.ndarray) -> np.ndarray:
    """Exact 3D reflection across still canal water plane Z = Z_WATER."""
    n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    p0 = np.array([0.0, 0.0, Z_WATER], dtype=np.float64)
    dots = np.sum((pts - p0) * n, axis=-1, keepdims=True)
    return pts - 2.0 * dots * n


# =============================================================================
# Stage 1 & 2 — Deep Twilight Horizon Backdrop, Concrete Walls & Voronoi Paving
# =============================================================================

def paint_twilight_horizon_backdrop(cam: Camera3D, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Serene deep-twilight monochrome horizon visible through the open end of the
    brutalist atrium, with smooth anti-aliased multi-layered misty distant headlands
    and a seamless reflective twilight sea stretching cleanly from the horizon.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = xx / W
    v = yy / H
    vh = float(cam.horizon_y / H)

    mottle = fbm_gouache(H, W, rng)
    dabs = flat_brush_dabs(H, W, rng, count=88)
    tooth = paper_tooth(H, W, rng)

    c_zenith = np.array([0.034, 0.038, 0.046], dtype=np.float32)
    c_mid = np.array([0.132, 0.142, 0.158], dtype=np.float32)
    c_horiz = np.array([0.385, 0.395, 0.415], dtype=np.float32)

    sky_t = np.clip(v / max(0.2, vh), 0.0, 1.0)
    sky_p = np.power(sky_t, 1.48)
    sky = (
        c_zenith[None, None, :] * np.power(1.0 - sky_p, 1.20)[..., None]
        + c_mid[None, None, :] * (2.1 * sky_p * (1.0 - sky_p))[..., None]
        + c_horiz[None, None, :] * np.power(sky_p, 1.75)[..., None]
    )

    # Luminous misty twilight glow centered on the open atrium portal
    du = (u - cam.vp_x / W) * 2.1
    dv = (v - (vh - 0.05)) * 3.1
    portal_glow = np.exp(-(du * du * 3.4 + dv * dv * 7.2)).astype(np.float32)
    sky += portal_glow[..., None] * np.array([0.18, 0.19, 0.205], dtype=np.float32)

    cloud = np.clip(0.5 + 0.35 * mottle + 0.18 * dabs, 0.0, 1.0) * np.clip((vh - v) / 0.42, 0.0, 1.0)
    sky += (cloud * 0.052)[..., None] * np.array([0.86, 0.90, 0.96], dtype=np.float32)

    # Smooth anti-aliased asymmetric coastal headland & misty distant mountain ridge
    # Slopes gently from left toward open serene twilight sea on right (avoids symmetric lens shape!)
    u_rel = (u - cam.vp_x / W) / 0.16  # -1 at left portal edge, +1 at right portal edge
    r1_h = np.maximum(
        0.002,
        0.021
        - 0.007 * u_rel
        + 0.006 * np.sin(u_rel * 2.4 + 0.9)
        + 0.003 * np.cos(u_rel * 5.2 - 0.4)
        + 0.0018 * mottle,
    )
    a_r1 = np.clip((v - (vh - r1_h)) / 0.0045, 0.0, 1.0) * np.clip((vh + 0.002 - v) / 0.002, 0.0, 1.0)
    r1_mist = np.clip((vh - v) / 0.032, 0.0, 1.0)
    r1_col = np.array([0.245, 0.254, 0.268], dtype=np.float32)[None, None, :] * (1.0 - 0.15 * r1_mist[..., None])
    sky = sky * (1.0 - a_r1[..., None]) + r1_col * a_r1[..., None]

    # Closer low dark coastal headland rising gently on the right side only
    r2_h = np.maximum(
        0.0,
        0.0085 * np.clip(0.35 + 0.65 * u_rel, 0.0, 1.4)
        + 0.0035 * np.sin(u_rel * 3.8 + 1.2)
        + 0.0012 * mottle,
    )
    a_r2 = np.clip((v - (vh - r2_h)) / 0.0038, 0.0, 1.0) * np.clip((vh + 0.002 - v) / 0.002, 0.0, 1.0)
    r2_col = np.array([0.168, 0.175, 0.188], dtype=np.float32)[None, None, :]
    sky = sky * (1.0 - a_r2[..., None]) + r2_col * a_r2[..., None]

    # Soft broad silver twilight mist band dissolving the distant shoreline naturally
    haze_band = np.exp(-((v - vh) / 0.028) ** 2).astype(np.float32) * 0.13
    sky += haze_band[..., None] * np.array([0.78, 0.81, 0.86], dtype=np.float32)

    # Below horizon (v > vh): calm misty outer twilight sea mirroring sky with soft capillary horizontal softening
    v_mirror_y = np.clip(2.0 * cam.horizon_y - yy, 0.0, H - 1.001)
    sea_raw = remap_bilinear(sky, xx, v_mirror_y)
    sea_refl = gaussian_blur_1d(sea_raw, 2.2, axis=0) * 0.78 + np.array([0.032, 0.036, 0.042], dtype=np.float32)[None, None, :]
    below = np.clip((v - vh) / 0.005, 0.0, 1.0)
    canvas = sky * (1.0 - below[..., None]) + sea_refl * below[..., None]
    canvas += (mottle * 0.007 + dabs * 0.005 + tooth * 0.004)[..., None]

    fog_rgb = np.array([0.158, 0.166, 0.180], dtype=np.float32)
    return np.clip(canvas, 0.0, 1.28), fog_rgb


def paint_voronoi_paving_uv(
    rng: np.random.Generator,
    x_span: tuple[float, float],
    y_span: tuple[float, float],
    tw: int = 768,
    th: int = 1408,
) -> np.ndarray:
    """
    High-resolution wet architectural Voronoi basalt/travertine stone paving along
    the left colonnade promenade (extra Y resolution eliminates foreground aliasing),
    receiving exact Stage 2 slatted-skylight geometric shafts & pillar shadows.
    """
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wx = x_span[0] + u * (x_span[1] - x_span[0])
    wy = y_span[0] + v * (y_span[1] - y_span[0])
    wz = np.full_like(wx, Z_WALK)

    nx = resize_bicubic(rng.standard_normal((th // 36, tw // 24)).astype(np.float32), tw, th) * 0.044
    ny = resize_bicubic(rng.standard_normal((th // 36, tw // 24)).astype(np.float32), tw, th) * 0.052
    px = wx + nx
    py = wy + ny

    cell_x, cell_y = 0.62, 0.78
    ix = np.floor(px / cell_x).astype(np.int32)
    iy = np.floor(py / cell_y).astype(np.int32)

    jx_tab = rng.uniform(0.20, 0.80, size=512)
    jy_tab = rng.uniform(0.20, 0.80, size=512)
    tone_tab = rng.uniform(-0.026, 0.030, size=512)

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
    # Smooth anti-aliased Voronoi basalt crevices so foreground edges stay velvety at 1:1 macro zoom!
    stone_mask = np.clip((edge_dist - 0.014) / 0.030, 0.0, 1.0)
    stone_mask = stone_mask * stone_mask * (3.0 - 2.0 * stone_mask)
    bevel = np.exp(-((edge_dist - 0.048) / 0.022) ** 2).astype(np.float32) * stone_mask

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=86)
    tooth = paper_tooth(th, tw, rng)
    sv = tone_tab[best_id].astype(np.float32)

    sky_lit = trace_skylight_and_colonnade(wx, wy, wz)
    acc_irr = eval_beacon_and_slit_light(wx, wy, wz, 0.0, 0.0, 1.0)

    in_coping = (wx > X_CANAL_L - 0.32)
    coping_joint = np.exp(-((np.minimum(np.mod(wy, 1.2), 1.2 - np.mod(wy, 1.2))) / 0.014) ** 2).astype(np.float32)

    amb_v = 0.052 + sv * 0.44 + mottle * 0.011 + dabs * 0.008
    lit_v = amb_v + sky_lit * (0.205 + 0.068 * bevel)
    crevice_v = 0.018 + 0.028 * sky_lit

    v_val = stone_mask * lit_v + (1.0 - stone_mask) * crevice_v
    v_val = np.where(in_coping, (0.084 + sky_lit * 0.215 + mottle * 0.010) * (1.0 - 0.42 * coping_joint), v_val)

    rgb = np.stack([v_val * 0.95, v_val * 0.98, v_val * 1.04], axis=-1).astype(np.float32)
    rgb += (acc_irr * (0.11 * stone_mask + 0.05 * bevel))[..., None] * ACCENT_YELLOW[None, None, :]

    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    rgb += np.clip((0.28 - lum) / 0.28, 0.0, 1.0)[..., None] * np.array([0.009, 0.009, 0.007], dtype=np.float32)
    rgb += tooth[..., None] * 0.005

    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.25)
    rgba[:, :, 3] = 1.0
    return rgba


def paint_board_concrete_wall_uv(
    rng: np.random.Generator,
    x_plane: float,
    normal_x: float,
    y_span: tuple[float, float],
    z_span: tuple[float, float],
    has_slit_window: bool = False,
    tw: int = 1280,
    th: int = 768,
) -> np.ndarray:
    """Monolithic board-formed concrete wall with formwork panel grid, tie holes & optional acid-yellow slit window."""
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wy = y_span[0] + u * (y_span[1] - y_span[0])
    wz = z_span[0] + v * (z_span[1] - z_span[0])
    wx = np.full_like(wy, x_plane)

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=86)
    tooth = paper_tooth(th, tw, rng)

    seam_h = np.exp(-((np.minimum(np.mod(wz, 0.90), 0.90 - np.mod(wz, 0.90))) / 0.011) ** 2).astype(np.float32)
    seam_v = np.exp(-((np.minimum(np.mod(wy, 1.80), 1.80 - np.mod(wy, 1.80))) / 0.013) ** 2).astype(np.float32)
    plank_z = np.mod(wz, 0.15)
    plank_seam = np.exp(-((np.minimum(plank_z, 0.15 - plank_z)) / 0.0045) ** 2).astype(np.float32)
    plank_id = np.floor(wz / 0.15).astype(np.int32)
    plank_tone = (((plank_id * 1664525 + 1013904223) & 255).astype(np.float32) / 255.0 - 0.5) * 0.014

    ty_d = np.abs(np.mod(wy + 0.30, 0.60) - 0.30)
    tz_d = np.abs(np.mod(wz + 0.225, 0.45) - 0.225)
    tie_hole = np.exp(-((ty_d * ty_d + tz_d * tz_d) / (0.022 * 0.022))).astype(np.float32)
    damp_bot = np.exp(-np.maximum(0.0, wz - 0.06) / 0.48).astype(np.float32)

    sky_lit = trace_skylight_and_colonnade(wx + normal_x * 0.04, wy, wz) if normal_x > 0 else np.zeros_like(wx, dtype=np.float32)
    acc_irr = eval_beacon_and_slit_light(wx, wy, wz, normal_x, 0.0, 0.0)

    base_v = (0.088 if normal_x > 0 else 0.068) + plank_tone + mottle * 0.014 + dabs * 0.009
    base_v += np.clip((wz - 3.6) / 2.2, 0.0, 1.0).astype(np.float32) * 0.024
    val = base_v * (1.0 - 0.32 * damp_bot)
    val += sky_lit * 0.185
    val *= (1.0 - 0.42 * seam_h - 0.36 * seam_v - 0.14 * plank_seam - 0.54 * tie_hole)

    rgb = np.stack([val * 0.95, val * 0.98, val * 1.04], axis=-1).astype(np.float32)
    rgb += (acc_irr * 0.19 * (1.0 - 0.32 * seam_h))[..., None] * ACCENT_YELLOW[None, None, :]

    if has_slit_window:
        in_reveal = (wy >= SLIT_Y0 - 0.12) & (wy <= SLIT_Y1 + 0.12) & (wz >= SLIT_Z0 - 0.10) & (wz <= SLIT_Z1 + 0.10)
        in_slit = (wy >= SLIT_Y0) & (wy <= SLIT_Y1) & (wz >= SLIT_Z0) & (wz <= SLIT_Z1)

        near_d = np.hypot(
            np.maximum(0.0, np.maximum(SLIT_Y0 - wy, wy - SLIT_Y1)),
            np.maximum(0.0, np.maximum(SLIT_Z0 - wz, wz - SLIT_Z1)),
        ).astype(np.float32)
        jamb_glow = np.exp(-near_d / 0.85) * 0.34
        rgb += jamb_glow[..., None] * ACCENT_YELLOW[None, None, :]

        reveal_col = np.array([0.045, 0.044, 0.024], dtype=np.float32) + ACCENT_YELLOW * 0.22
        rgb = np.where(in_reveal[..., None], reveal_col[None, None, :], rgb)

        su = np.clip((wy - SLIT_Y0) / (SLIT_Y1 - SLIT_Y0), 0.0, 1.0).astype(np.float32)
        sv = np.clip((wz - SLIT_Z0) / (SLIT_Z1 - SLIT_Z0), 0.0, 1.0).astype(np.float32)
        slit_profile = np.exp(-((su - 0.5) / 0.52) ** 2) * (0.76 + 0.24 * np.sin(sv * math.pi))
        slit_col = (
            ACCENT_WARM[None, None, :] * 0.55
            + ACCENT_YELLOW[None, None, :] * (0.52 * slit_profile)[..., None]
            + ACCENT_CORE[None, None, :] * (0.42 * np.power(slit_profile, 1.8))[..., None]
        ) * (0.94 + 0.06 * mottle)[..., None]

        mullion = (np.abs(wz - 1.65) < 0.025) | (np.abs(wz - 2.95) < 0.025)
        slit_col = np.where(mullion[..., None], np.array([0.032, 0.032, 0.020], dtype=np.float32), slit_col)
        rgb = np.where(in_slit[..., None], slit_col, rgb)

    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    rgb += np.clip((0.28 - lum) / 0.28, 0.0, 1.0)[..., None] * np.array([0.009, 0.009, 0.007], dtype=np.float32)
    rgb += tooth[..., None] * 0.005

    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.45)
    rgba[:, :, 3] = 1.0
    return rgba


# =============================================================================
# Stage 2 & 3 — Volumetric Slatted Skylight Shafts & Fresnel Water Canal
# =============================================================================

def paint_volumetric_skylight_shafts(
    cam: Camera3D,
    rng: np.random.Generator,
    intensity: float = 0.48,
) -> np.ndarray:
    """
    Crisp geometric twilight volumetric light shafts pouring diagonally down-left
    from each slatted skylight slot strictly parallel to projected SUN_DIR.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    p_ref = np.array([-0.8, 12.0, 3.0], dtype=np.float64)
    u0, v0, _ = cam.project(p_ref)
    u1, v1, _ = cam.project(p_ref + SUN_DIR * 3.0)
    dx, dy = float(u1 - u0), float(v1 - v0)
    dlen = math.hypot(dx, dy) + 1e-6
    rx, ry = dx / dlen, dy / dlen
    nx, ny = -ry, rx

    acc = np.zeros((H, W), dtype=np.float32)
    for k in range(2, 13):
        y_slot = (k + 0.5) * BEAM_PITCH
        if y_slot > Y_OPEN_END - 1.5:
            break
        p_slot = np.array([0.55, y_slot, Z_ROOF_BOT], dtype=np.float64)
        us, vs, zs = cam.project(p_slot)
        if zs < Z_NEAR:
            continue
        s_coord = (xx - float(us)) * rx + (yy - float(vs)) * ry
        p_coord = (xx - float(us)) * nx + (yy - float(vs)) * ny

        w_base = max(12.0, 340.0 / float(zs))
        w_eff = w_base * (1.0 + np.maximum(0.0, s_coord) / (H * 1.9))
        beam = np.exp(-0.5 * (p_coord / w_eff) ** 4)
        vert = np.clip((s_coord + 10.0) / 70.0, 0.0, 1.0) * np.exp(-np.maximum(0.0, s_coord) / (H * 0.88))
        dist_att = math.exp(-max(0.0, float(zs) - 5.0) * 0.060)
        acc += beam * vert * dist_att * 0.36

    turb = 0.86 + 0.14 * resize_bicubic(fbm_gouache(H // 4, W // 4, rng), W, H)
    acc = np.clip(acc * turb * intensity, 0.0, 0.72)
    return acc[..., None] * np.array([0.76, 0.80, 0.88], dtype=np.float32)[None, None, :]


def composite_reflecting_water_canal(
    canvas: np.ndarray,
    cam: Camera3D,
    canal_mask_screen: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Still indoor reflecting water canal extending seamlessly to the infinity weir
    with Fresnel capillary ripples and exact 3D plane-mirrored colonnade & acid-yellow
    (#d9d84a) recessed beacon / warm slit window reflections!
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    mask = np.clip(canal_mask_screen[:, :, 0], 0.0, 1.0)
    below_h = np.clip((yy - (cam.horizon_y - 1.0)) / 5.0, 0.0, 1.0)
    w_mask = mask * below_h

    rip1 = resize_bicubic(rng.standard_normal((max(8, H // 4), max(8, W // 58))).astype(np.float32), W, H)
    rip2 = resize_bicubic(rng.standard_normal((max(8, H // 2), max(8, W // 26))).astype(np.float32), W, H)
    rip = 0.65 * rip1 + 0.35 * rip2

    dist_t = np.clip((yy - cam.horizon_y) / max(1.0, H - cam.horizon_y), 0.0, 1.0)
    fresnel_min, fresnel_max = 0.24, 0.88
    fresnel = fresnel_min + (fresnel_max - fresnel_min) * np.power(1.0 - dist_t, 1.4)

    amp_x = 1.4 + 5.0 * dist_t
    amp_y = 1.0 + 3.4 * dist_t
    map_x = xx + rip * amp_x
    map_y = np.clip(2.0 * cam.horizon_y - yy + rip * amp_y, 0.0, H - 1.001)
    refl_scene = remap_bilinear(canvas, map_x, map_y)

    water_body = np.array([0.015, 0.017, 0.021], dtype=np.float32)[None, None, :]
    water_col = water_body * (1.0 - fresnel[..., None]) + refl_scene * (fresnel[..., None] * 0.90)

    # Exact 3D plane-mirrored luminous acid-yellow (#d9d84a) beacon + tall vertical slit window
    m_slit_top = mirror_points_water(np.array([[X_WALL_R, 0.5 * (SLIT_Y0 + SLIT_Y1), SLIT_Z1]]))[0]
    m_slit_bot = mirror_points_water(np.array([[X_WALL_R, 0.5 * (SLIT_Y0 + SLIT_Y1), SLIT_Z0]]))[0]
    m_beacon = mirror_points_water(BEACON_POS[None, :])[0]

    ust, vst, _ = cam.project(m_slit_top)
    usb, vsb, _ = cam.project(m_slit_bot)
    ube, vbe, _ = cam.project(m_beacon)
    _, vb0, _ = cam.project(np.array([BEACON_POS[0], BEACON_POS[1], Z_WATER]))

    rx = xx + rip * (6.2 + 7.5 * dist_t)
    ry = yy + rip * 3.0

    slit_t = np.clip((ry - float(vsb)) / max(1.0, float(vst) - float(vsb)), 0.0, 1.0)
    slit_u_at_y = float(usb) + (float(ust) - float(usb)) * slit_t
    in_slit_y = np.clip((ry - (float(vsb) - 18.0)) / 24.0, 0.0, 1.0) * np.clip((float(vst) + 55.0 - ry) / 65.0, 0.0, 1.0)
    slit_w = 18.0 + 26.0 * slit_t
    slit_refl = np.exp(-0.5 * ((rx - slit_u_at_y) / slit_w) ** 2) * in_slit_y

    in_b_y = np.clip((ry - (float(vb0) - 6.0)) / 14.0, 0.0, 1.0) * np.exp(-np.maximum(0.0, ry - float(vbe)) / 225.0)
    b_w = 16.0 + 28.0 * np.clip((ry - float(vb0)) / 280.0, 0.0, 1.0)
    b_refl = np.exp(-0.5 * ((rx - float(ube)) / b_w) ** 2) * in_b_y

    shimmer = np.clip(0.62 + 0.58 * np.sin(yy * 0.42 + rip * 4.8), 0.18, 1.38).astype(np.float32)
    y_int = (slit_refl * 0.85 + b_refl * 0.95) * shimmer * w_mask

    wide_sheen = (
        np.exp(-0.5 * ((rx - slit_u_at_y) / (slit_w * 3.6)) ** 2) * in_slit_y * 0.16
        + np.exp(-0.5 * ((rx - float(ube)) / (b_w * 3.8)) ** 2) * in_b_y * 0.22
    ) * w_mask

    water_col += (y_int + wide_sheen)[..., None] * ACCENT_YELLOW[None, None, :] * 0.72
    water_col += np.power(np.clip(y_int, 0.0, 1.2), 2.0)[..., None] * ACCENT_CORE[None, None, :] * 0.48
    water_col += (np.clip(rip, 0.0, 1.0) * 0.018 * (0.4 + 0.6 * fresnel) * w_mask)[..., None]

    # Subtle razor-clean dark infinity-pool weir sill line right at Y = Y_OPEN_END!
    _, v_weir, _ = cam.project(np.array([1.2, Y_OPEN_END, Z_WATER]))
    weir_line = np.exp(-((yy - float(v_weir)) / 1.45) ** 2).astype(np.float32) * w_mask
    water_col = water_col * (1.0 - 0.58 * weir_line[..., None])

    out = canvas * (1.0 - w_mask[..., None]) + water_col * w_mask[..., None]
    return np.clip(out, 0.0, 1.45)


# =============================================================================
# Stage 3 — 2x Supersampled PyCairo Brutalist Colonnade, Skylight & Beacon
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
    fade_dist: float = 28.0,
    taper: bool = True,
    tremor: float = 0.20,
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


def fog_mix(c3: Iterable[float], dist: float, fog_rgb: np.ndarray, fog_start: float = 8.0, fog_density: float = 0.042) -> tuple[float, float, float]:
    ft = 1.0 - math.exp(-max(0.0, dist - fog_start) * fog_density)
    c = np.asarray(c3, dtype=np.float64) * (1.0 - ft) + fog_rgb * ft
    return (float(c[0]), float(c[1]), float(c[2]))


def fill_3d_quad_grad(
    ctx: cairo.Context,
    cam: Camera3D,
    pts3: list[np.ndarray],
    c_bot: tuple[float, float, float],
    c_top: tuple[float, float, float],
    alpha: float = 1.0,
) -> None:
    proj = [cam.project(p) for p in pts3]
    if any(z < Z_NEAR for _, _, z in proj):
        return
    g = cairo.LinearGradient(float(proj[0][0]), float(proj[0][1]), float(proj[3][0]), float(proj[3][1]))
    g.add_color_stop_rgba(0.0, *c_bot, alpha)
    g.add_color_stop_rgba(1.0, *c_top, alpha)
    ctx.set_source(g)
    ctx.move_to(float(proj[0][0]), float(proj[0][1]))
    for pu, pv, _ in proj[1:]:
        ctx.line_to(float(pu), float(pv))
    ctx.close_path()
    ctx.fill()


def draw_monolith_pillar(
    ctx: cairo.Context,
    cam: Camera3D,
    y_center: float,
    fog_rgb: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """
    Monolithic board-formed concrete colonnade pillar standing at X in [PILLAR_X0, PILLAR_X1],
    complete with formwork pour seams, subtle horizontal board-plank variation,
    tie-rod cone holes, crisp chamfer catchlight, and acid-yellow water bounce.
    """
    y0 = y_center - PILLAR_HALF_Y
    y1 = y_center + PILLAR_HALF_Y
    z0, z1 = Z_WALK, Z_ROOF_BOT + 0.08
    dist = float(np.linalg.norm(np.array([PILLAR_X1, y_center, 1.8]) - cam.eye))

    acc_r = float(eval_beacon_and_slit_light(
        np.array([PILLAR_X1]), np.array([y_center]), np.array([1.6]), 1.0, 0.0, 0.0
    )[0])

    front_pts = [
        np.array([PILLAR_X0, y0, z0]),
        np.array([PILLAR_X1, y0, z0]),
        np.array([PILLAR_X1, y0, z1]),
        np.array([PILLAR_X0, y0, z1]),
    ]
    right_pts = [
        np.array([PILLAR_X1, y0, z0]),
        np.array([PILLAR_X1, y1, z0]),
        np.array([PILLAR_X1, y1, z1]),
        np.array([PILLAR_X1, y0, z1]),
    ]

    # Clean architectural concrete shading (subtle skylight bounce on front face, luminous skylight+yellow on right!)
    c_f_bot = fog_mix([0.044, 0.047, 0.053], dist, fog_rgb)
    c_f_top = fog_mix([0.084, 0.089, 0.098], dist, fog_rgb)
    c_r_bot = fog_mix(np.array([0.084, 0.090, 0.100]) + ACCENT_YELLOW * (0.20 * acc_r), dist, fog_rgb)
    c_r_top = fog_mix(np.array([0.175, 0.184, 0.198]) + ACCENT_YELLOW * (0.14 * acc_r), dist, fog_rgb)

    fill_3d_quad_grad(ctx, cam, front_pts, c_f_bot, c_f_top, 1.0)
    fill_3d_quad_grad(ctx, cam, right_pts, c_r_bot, c_r_top, 1.0)

    # Subtle alternating board-formwork lift panels (0.90 m bands)
    for k_b, zb in enumerate(np.arange(Z_WALK, Z_ROOF_BOT - 0.1, 0.90)):
        zt = min(Z_ROOF_BOT + 0.04, zb + 0.90)
        if k_b % 2 == 0:
            tint_a = 0.05
            fill_3d_quad_grad(
                ctx, cam,
                [np.array([PILLAR_X0, y0, zb]), np.array([PILLAR_X1, y0, zb]),
                 np.array([PILLAR_X1, y0, zt]), np.array([PILLAR_X0, y0, zt])],
                (0.22, 0.23, 0.26), (0.18, 0.19, 0.22), tint_a,
            )

    seam_c = fog_mix([0.018, 0.020, 0.024], dist, fog_rgb)
    for zs in np.arange(0.90, Z_ROOF_BOT, 0.90):
        for pa, pb in (
            (np.array([PILLAR_X0, y0, zs]), np.array([PILLAR_X1, y0, zs])),
            (np.array([PILLAR_X1, y0, zs]), np.array([PILLAR_X1, y1, zs])),
        ):
            seg = cam.project_polyline(np.array([pa, pb]))
            if len(seg) >= 2:
                stroke_catmull_rom(ctx, [(p[0], p[1]) for p in seg], 1.5, dist, (*seam_c, 0.62), rng, taper=False, tremor=0.10)

    # Pairs of architectural concrete tie-rod holes per formwork panel
    for zt in np.arange(0.45, Z_ROOF_BOT - 0.1, 0.90):
        for pt_hole in (
            np.array([PILLAR_X0 + 0.16, y0, zt]),
            np.array([PILLAR_X1 - 0.16, y0, zt]),
            np.array([PILLAR_X1, y0 + 0.16, zt]),
            np.array([PILLAR_X1, y1 - 0.16, zt]),
        ):
            uh, vh, zh = cam.project(pt_hole)
            if zh > Z_NEAR:
                rh = max(1.2, 24.0 / float(zh))
                ctx.set_source_rgba(*seam_c, 0.72)
                ctx.arc(float(uh), float(vh), rh, 0, math.tau)
                ctx.fill()

    # Crisp chamfer corner catchlight facing water & skylight
    rim_c = fog_mix(np.array([0.29, 0.305, 0.325]) + ACCENT_YELLOW * (0.28 * acc_r), dist, fog_rgb)
    rseg = cam.project_polyline(np.array([[PILLAR_X1, y0, z0], [PILLAR_X1, y0, z1]]))
    if len(rseg) >= 2:
        stroke_catmull_rom(ctx, [(p[0], p[1]) for p in rseg], 2.1, dist, (*rim_c, 0.80), rng, taper=False, tremor=0.08)


def draw_skylight_beam(
    ctx: cairo.Context,
    cam: Camera3D,
    y_center: float,
    fog_rgb: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Transverse brutalist cast-concrete skylight beam spanning from X_WALL_L to X_WALL_R."""
    y0 = y_center - BEAM_HALF_Y
    y1 = y_center + BEAM_HALF_Y
    dist = float(np.linalg.norm(np.array([0.0, y_center, Z_ROOF_BOT]) - cam.eye))

    sof_c0 = fog_mix([0.026, 0.029, 0.034], dist, fog_rgb)
    sof_c1 = fog_mix([0.054, 0.058, 0.066], dist, fog_rgb)
    fill_3d_quad_grad(
        ctx, cam,
        [np.array([X_WALL_L, y0, Z_ROOF_BOT]), np.array([X_WALL_R, y0, Z_ROOF_BOT]),
         np.array([X_WALL_R, y1, Z_ROOF_BOT]), np.array([X_WALL_L, y1, Z_ROOF_BOT])],
        sof_c0, sof_c1, 1.0,
    )

    face_c0 = fog_mix([0.056, 0.060, 0.068], dist, fog_rgb)
    face_c1 = fog_mix([0.125, 0.132, 0.146], dist, fog_rgb)
    fill_3d_quad_grad(
        ctx, cam,
        [np.array([X_WALL_L, y0, Z_ROOF_BOT]), np.array([X_WALL_R, y0, Z_ROOF_BOT]),
         np.array([X_WALL_R, y0, Z_ROOF_TOP]), np.array([X_WALL_L, y0, Z_ROOF_TOP])],
        face_c0, face_c1, 1.0,
    )

    edge_seg = cam.project_polyline(np.array([[X_WALL_L, y0, Z_ROOF_BOT], [X_WALL_R, y0, Z_ROOF_BOT]]))
    if len(edge_seg) >= 2:
        ec = fog_mix([0.21, 0.22, 0.24], dist, fog_rgb)
        stroke_catmull_rom(ctx, [(p[0], p[1]) for p in edge_seg], 1.6, dist, (*ec, 0.65), rng, taper=False, tremor=0.08)


def draw_recessed_beacon_plinth(
    ctx: cairo.Context,
    cam: Camera3D,
    fog_rgb: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """
    Monolithic brutalist concrete water-edge pier cleanly seated in the reflecting
    canal at Y = 12.65..13.45 m, housing a single recessed acid-yellow (#d9d84a)
    architectural beacon luminaire with crisp waterline grounding.
    """
    x0, x1 = BEACON_X0, BEACON_X1
    y0, y1 = BEACON_Y0, BEACON_Y1
    z0, z1 = Z_WATER, 1.08
    dist = float(np.linalg.norm(BEACON_POS - cam.eye))

    # Front face, water-facing inner face, and honed top cap
    fill_3d_quad_grad(
        ctx, cam,
        [np.array([x0, y0, z0]), np.array([x1, y0, z0]), np.array([x1, y0, z1]), np.array([x0, y0, z1])],
        fog_mix([0.026, 0.028, 0.031], dist, fog_rgb),
        fog_mix([0.064, 0.066, 0.070], dist, fog_rgb),
    )
    fill_3d_quad_grad(
        ctx, cam,
        [np.array([x0, y0, z0]), np.array([x0, y1, z0]), np.array([x0, y1, z1]), np.array([x0, y0, z1])],
        fog_mix(np.array([0.044, 0.046, 0.048]) + ACCENT_YELLOW * 0.24, dist, fog_rgb),
        fog_mix(np.array([0.090, 0.092, 0.095]) + ACCENT_YELLOW * 0.20, dist, fog_rgb),
    )
    fill_3d_quad_grad(
        ctx, cam,
        [np.array([x0, y0, z1]), np.array([x1, y0, z1]), np.array([x1, y1, z1]), np.array([x0, y1, z1])],
        fog_mix([0.120, 0.126, 0.136], dist, fog_rgb),
        fog_mix([0.152, 0.160, 0.172], dist, fog_rgb),
    )

    # Crisp waterline shadow meniscus right where concrete base meets dark water!
    wl_seg = cam.project_polyline(np.array([[x0, y1, z0], [x0, y0, z0], [x1, y0, z0]]))
    if len(wl_seg) >= 2:
        stroke_catmull_rom(ctx, [(p[0], p[1]) for p in wl_seg], 2.4, dist, (0.010, 0.011, 0.014, 0.92), rng, taper=False, tremor=0.05)

    # Recessed luminous acid-yellow (#d9d84a) beacon aperture inset into both water & front faces
    b_pts = [
        np.array([x0 - 0.004, 12.84, 0.26]),
        np.array([x0 - 0.004, 13.24, 0.26]),
        np.array([x0 - 0.004, 13.24, 0.90]),
        np.array([x0 - 0.004, 12.84, 0.90]),
    ]
    bf_pts = [
        np.array([x0 + 0.08, y0 - 0.004, 0.30]),
        np.array([x0 + 0.34, y0 - 0.004, 0.30]),
        np.array([x0 + 0.34, y0 - 0.004, 0.88]),
        np.array([x0 + 0.08, y0 - 0.004, 0.88]),
    ]

    ub, vb, zb = cam.project(BEACON_POS)
    if zb > Z_NEAR:
        g_h = cairo.RadialGradient(float(ub), float(vb), 6.0, float(ub), float(vb), 155.0)
        g_h.add_color_stop_rgba(0.0, float(ACCENT_CORE[0]), float(ACCENT_CORE[1]), float(ACCENT_CORE[2]), 0.44)
        g_h.add_color_stop_rgba(0.35, float(ACCENT_YELLOW[0]), float(ACCENT_YELLOW[1]), float(ACCENT_YELLOW[2]), 0.20)
        g_h.add_color_stop_rgba(1.0, float(ACCENT_WARM[0]), float(ACCENT_WARM[1]), float(ACCENT_WARM[2]), 0.0)
        ctx.set_source(g_h)
        ctx.arc(float(ub), float(vb), 155.0, 0, math.tau)
        ctx.fill()

    fill_3d_quad_grad(ctx, cam, b_pts, tuple(ACCENT_YELLOW), tuple(ACCENT_CORE), 1.0)
    fill_3d_quad_grad(ctx, cam, bf_pts, tuple(ACCENT_YELLOW), tuple(ACCENT_CORE), 0.98)

    for zl in (0.48, 0.68):
        lseg = cam.project_polyline(np.array([[x0 + 0.07, y0 - 0.008, zl], [x0 + 0.35, y0 - 0.008, zl]]))
        if len(lseg) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in lseg], 2.0, dist, (0.024, 0.024, 0.016, 0.92), rng, taper=False, tremor=0.04)


def draw_promenade_curb_and_bench(
    ctx: cairo.Context,
    cam: Camera3D,
    fog_rgb: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Crisp vertical concrete canal coping curb + minimalist monolithic stone bench on the walkway."""
    curb_pts = [
        np.array([X_CANAL_L, 1.8, Z_WATER - 0.02]),
        np.array([X_CANAL_L, Y_OPEN_END, Z_WATER - 0.02]),
        np.array([X_CANAL_L, Y_OPEN_END, Z_WALK]),
        np.array([X_CANAL_L, 1.8, Z_WALK]),
    ]
    fill_3d_quad_grad(
        ctx, cam, curb_pts,
        fog_mix([0.022, 0.024, 0.028], 12.0, fog_rgb),
        fog_mix([0.054, 0.058, 0.066], 12.0, fog_rgb),
    )
    lip_seg = cam.project_polyline(np.array([[X_CANAL_L, 2.0, Z_WALK], [X_CANAL_L, Y_OPEN_END, Z_WALK]]))
    if len(lip_seg) >= 2:
        stroke_catmull_rom(ctx, [(p[0], p[1]) for p in lip_seg], 2.2, 8.0, (0.36, 0.38, 0.42, 0.74), rng, taper=False, tremor=0.08)

    # Minimalist cantilevered brutalist stone bench nestled cleanly between Pillar 0 and Pillar 1
    bx0, bx1 = -2.52, -1.92
    by0, by1 = 6.25, 7.80
    bz_leg, bz_top = Z_WALK, 0.50
    bdist = float(np.linalg.norm(np.array([bx1, 7.0, bz_top]) - cam.eye))
    fill_3d_quad_grad(
        ctx, cam,
        [np.array([bx0 + 0.12, by0 + 0.15, bz_leg]), np.array([bx1 - 0.08, by0 + 0.15, bz_leg]),
         np.array([bx1 - 0.08, by0 + 0.15, bz_top - 0.09]), np.array([bx0 + 0.12, by0 + 0.15, bz_top - 0.09])],
        fog_mix([0.018, 0.020, 0.024], bdist, fog_rgb),
        fog_mix([0.028, 0.030, 0.034], bdist, fog_rgb),
    )
    fill_3d_quad_grad(
        ctx, cam,
        [np.array([bx0, by0, bz_top - 0.10]), np.array([bx1, by0, bz_top - 0.10]),
         np.array([bx1, by0, bz_top]), np.array([bx0, by0, bz_top])],
        fog_mix([0.044, 0.047, 0.052], bdist, fog_rgb),
        fog_mix([0.072, 0.076, 0.084], bdist, fog_rgb),
    )
    fill_3d_quad_grad(
        ctx, cam,
        [np.array([bx1, by0, bz_top - 0.10]), np.array([bx1, by1, bz_top - 0.10]),
         np.array([bx1, by1, bz_top]), np.array([bx1, by0, bz_top])],
        fog_mix([0.066, 0.070, 0.078], bdist, fog_rgb),
        fog_mix([0.112, 0.118, 0.128], bdist, fog_rgb),
    )
    fill_3d_quad_grad(
        ctx, cam,
        [np.array([bx0, by0, bz_top]), np.array([bx1, by0, bz_top]),
         np.array([bx1, by1, bz_top]), np.array([bx0, by1, bz_top])],
        fog_mix([0.122, 0.128, 0.138], bdist, fog_rgb),
        fog_mix([0.205, 0.215, 0.230], bdist, fog_rgb),
    )
    b_hi = cam.project_polyline(np.array([[bx1, by0, bz_top], [bx1, by1, bz_top]]))
    if len(b_hi) >= 2:
        stroke_catmull_rom(ctx, [(p[0], p[1]) for p in b_hi], 1.8, bdist, (0.36, 0.38, 0.42, 0.74), rng, taper=False, tremor=0.06)


# =============================================================================
# Stage 4 — Foreground Circular Disc Bokeh Concrete Louvre & 35mm Optics
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


def paint_foreground_bokeh_louvre(rng: np.random.Generator) -> np.ndarray:
    """
    Close out-of-focus brutalist architectural concrete louvre fins framing the
    upper-left corner, convolved in premultiplied RGBA with circular aperture disc
    bokeh kernel (r = 7.8) + ~1 px opposite red/blue lateral shift.
    """
    surf, ctx = new_cairo_2x()

    header = [(-30, -30), (660, -30), (605, 58), (240, 134), (-30, 190)]
    ctx.set_source_rgba(0.013, 0.014, 0.017, 0.98)
    ctx.move_to(*header[0])
    for p in header[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    ctx.fill()

    slats = [
        [(-25, 152), (425, 48), (446, 110), (-25, 238)],
        [(-25, 276), (292, 194), (308, 246), (-25, 350)],
        [(-25, 388), (172, 334), (184, 374), (-25, 442)],
    ]
    for s_poly in slats:
        ctx.set_source_rgba(0.014, 0.015, 0.019, 0.97)
        ctx.move_to(*s_poly[0])
        for p in s_poly[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.fill()
        # Subtle twilight silver specular rim grazing the chamfered bevel of each louvre fin
        ctx.set_source_rgba(0.30, 0.32, 0.37, 0.62)
        ctx.set_line_width(3.8)
        ctx.move_to(*s_poly[0])
        ctx.line_to(*s_poly[1])
        ctx.stroke()

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
    tint = np.array([1.05, 1.04, 0.82], dtype=np.float32)
    hal = (
        0.25 * gaussian_blur_2d(hl_src, 6.5)
        + 0.19 * gaussian_blur_2d(hl_src, 24.0)
        + 0.13 * gaussian_blur_2d(hl_src, 72.0)
    ) * tint[None, None, :]
    img = 1.0 - (1.0 - np.clip(img, 0.0, 1.0)) * (1.0 - np.clip(hal, 0.0, 0.85))

    # 2. Radial corner CA + gentle optical vignette
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (xx - W * 0.5) / (W * 0.5)
    ny = (yy - H * 0.5) / (H * 0.5)
    r2 = nx * nx + ny * ny
    ca_scale = 0.0010
    img[:, :, 0] = remap_bilinear(img[:, :, 0], xx + nx * r2 * (W * ca_scale), yy + ny * r2 * (H * ca_scale))
    img[:, :, 2] = remap_bilinear(img[:, :, 2], xx - nx * r2 * (W * ca_scale), yy - ny * r2 * (H * ca_scale))
    img *= np.clip(1.0 - 0.22 * (r2 ** 1.16), 0.68, 1.0)[..., None]

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
    img = np.clip(img + (0.74 * mono_g + 0.26 * grain) * env[..., None], 0.0, 1.0)

    # 5. Faint analog transfer patina: horizontal YIQ chroma softening (1.0 px) + reflect101
    #    + slight luma ring + sub-pixel per-line horizontal tape jitter + <=0.8% scanlines
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

    line_shift = gaussian_blur_1d(rng.normal(0.0, 0.15, size=(H, 1)).astype(np.float32), 1.2, axis=0)
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
    audit_dir: pathlib.Path | None = None,
    webp_outputs: list[pathlib.Path] | None = None,
) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # Stage 1: Off-axis Camera3D (VP ~58.6% width, horizon ~47.6% height)
    cam = Camera3D(W, H, eye=(0.48, 0.0, 1.68), target=(-0.64, 12.0, 1.50), fov_x_deg=57.0)
    print(f"[mira-02] Camera VP: ({cam.vp_x:.1f}, {cam.vp_y:.1f}) -> {cam.vp_x / W * 100:.1f}% width")

    # Stage 1 & 2: Deep twilight monochrome horizon, soft mountain layers & calm outer sea
    canvas, fog_rgb = paint_twilight_horizon_backdrop(cam, rng)

    # Left & right monolithic board-formed concrete walls (right wall holds the luminous acid-yellow slit window!)
    left_wall_uv = paint_board_concrete_wall_uv(rng, X_WALL_L, +1.0, (1.6, Y_OPEN_END), (Z_WALK, Z_ROOF_TOP + 0.1), has_slit_window=False)
    right_wall_uv = paint_board_concrete_wall_uv(rng, X_WALL_R, -1.0, (1.6, Y_OPEN_END), (Z_WATER - 0.05, Z_ROOF_TOP + 0.1), has_slit_window=True)

    canvas = cam.warp_plane_over(
        canvas, left_wall_uv,
        p0=np.array([X_WALL_L, 1.6, Z_WALK]),
        e1=np.array([0.0, Y_OPEN_END - 1.6, 0.0]),
        e2=np.array([0.0, 0.0, Z_ROOF_TOP + 0.1 - Z_WALK]),
        fog_rgb=fog_rgb, fog_start=8.0, fog_density=0.042,
    )
    canvas = cam.warp_plane_over(
        canvas, right_wall_uv,
        p0=np.array([X_WALL_R, 1.6, Z_WATER - 0.05]),
        e1=np.array([0.0, Y_OPEN_END - 1.6, 0.0]),
        e2=np.array([0.0, 0.0, Z_ROOF_TOP + 0.15]),
        fog_rgb=fog_rgb, fog_start=8.0, fog_density=0.042,
    )

    # High-res wet Voronoi basalt stone promenade along the left colonnade
    paving_uv = paint_voronoi_paving_uv(rng, (X_WALL_L, X_CANAL_L), (1.4, Y_OPEN_END))
    canvas = cam.warp_plane_over(
        canvas, paving_uv,
        p0=np.array([X_WALL_L, 1.4, Z_WALK]),
        e1=np.array([X_CANAL_L - X_WALL_L, 0.0, 0.0]),
        e2=np.array([0.0, Y_OPEN_END - 1.4, 0.0]),
        fog_rgb=fog_rgb, fog_start=8.0, fog_density=0.042,
    )

    # Build exact screen-space water canal footprint extending seamlessly out to infinity horizon,
    # while cleanly excluding the recessed beacon plinth's base footprint!
    c_th, c_tw = 256, 128
    cyy, cxx = np.mgrid[0:c_th, 0:c_tw].astype(np.float64)
    c_wx = X_CANAL_L + (cxx / (c_tw - 1)) * (X_WALL_R - X_CANAL_L)
    c_wy = 1.2 + (cyy / (c_th - 1)) * (Y_OPEN_END - 1.2)
    in_plinth_base = (c_wx >= BEACON_X0 - 0.02) & (c_wy >= BEACON_Y0 - 0.02) & (c_wy <= BEACON_Y1 + 0.02)
    canal_uv = np.ones((c_th, c_tw, 4), dtype=np.float32)
    canal_uv[in_plinth_base, :] = 0.0

    canal_mask_screen = cam.warp_plane_over(
        np.zeros((H, W, 3), dtype=np.float32), canal_uv,
        p0=np.array([X_CANAL_L, 1.2, Z_WATER]),
        e1=np.array([X_WALL_R - X_CANAL_L, 0.0, 0.0]),
        e2=np.array([0.0, Y_OPEN_END - 1.2, 0.0]),
        fog_rgb=np.zeros(3, dtype=np.float32), fog_start=999.0, fog_density=0.0,
    )

    # Stage 3 Band 0 (Far Open-End Frame, Far Skylight Beams & Far Colonnade Pillars, Y = 20..32 m)
    s_far, c_far = new_cairo_2x()
    draw_skylight_beam(c_far, cam, Y_OPEN_END - 0.35, fog_rgb, rng)
    for yb in np.arange(30.2, 19.5, -BEAM_PITCH):
        draw_skylight_beam(c_far, cam, float(yb), fog_rgb, rng)
    for yp in (28.55, 24.65, 20.75):
        draw_monolith_pillar(c_far, cam, yp, fog_rgb, rng)
    canvas = over_premul(canvas, finish_cairo_2x(s_far))

    # Stage 3 Band 1 (Midground Colonnade Pillars & Skylight Beams, Y = 11..20 m)
    s_mid, c_mid = new_cairo_2x()
    for yb in np.arange(19.1, 11.2, -BEAM_PITCH):
        draw_skylight_beam(c_mid, cam, float(yb), fog_rgb, rng)
    for yp in (16.85, 12.95):
        draw_monolith_pillar(c_mid, cam, yp, fog_rgb, rng)
    mid_1x = finish_cairo_2x(s_mid)
    m_mot = fbm_gouache(H, W, rng)
    m_dabs = flat_brush_dabs(H, W, rng, count=82)
    ma = mid_1x[:, :, 3:4]
    mid_1x[:, :, :3] = np.clip(mid_1x[:, :, :3] + ma * (m_mot * 0.008 + m_dabs * 0.005)[..., None], 0.0, 1.35)
    canvas = over_premul(canvas, mid_1x)

    # Back pass of crisp geometric slatted twilight skylight shafts aligned with SUN_DIR
    shafts_back = paint_volumetric_skylight_shafts(cam, rng, intensity=0.48)
    canvas = 1.0 - (1.0 - np.clip(canvas, 0.0, 1.0)) * (1.0 - np.clip(shafts_back, 0.0, 0.80))

    # Stage 3 Band 2 (Near-Mid Colonnade Pillar, Promenade Bench, Canal Curb & Near Skylight Beams)
    s_nmid, c_nmid = new_cairo_2x()
    for yb in np.arange(11.1, 6.2, -BEAM_PITCH):
        draw_skylight_beam(c_nmid, cam, float(yb), fog_rgb, rng)
    draw_monolith_pillar(c_nmid, cam, 9.05, fog_rgb, rng)
    draw_promenade_curb_and_bench(c_nmid, cam, fog_rgb, rng)
    nmid_1x = finish_cairo_2x(s_nmid)
    na = nmid_1x[:, :, 3:4]
    nmid_1x[:, :, :3] = np.clip(nmid_1x[:, :, :3] + na * (m_mot * 0.008 + m_dabs * 0.005)[..., None], 0.0, 1.3)
    canvas = over_premul(canvas, nmid_1x)

    # Stage 2: Composite Still Indoor Reflecting Water Canal (Fresnel ripples + 3D mirrored colonnade & #d9d84a beacon/slit)
    canvas = composite_reflecting_water_canal(canvas, cam, canal_mask_screen, rng)

    # Stage 3 Band 3 (Recessed #d9d84a Beacon Plinth cleanly seated at waterline + Foreground Colonnade Pillar & Near Beams)
    s_near, c_near = new_cairo_2x()
    draw_recessed_beacon_plinth(c_near, cam, fog_rgb, rng)
    draw_skylight_beam(c_near, cam, 5.55, fog_rgb, rng)
    draw_skylight_beam(c_near, cam, 3.70, fog_rgb, rng)
    draw_monolith_pillar(c_near, cam, 5.15, fog_rgb, rng)
    near_1x = finish_cairo_2x(s_near)
    n2a = near_1x[:, :, 3:4]
    near_1x[:, :, :3] = np.clip(near_1x[:, :, :3] + n2a * (m_mot * 0.007 + m_dabs * 0.004)[..., None], 0.0, 1.4)
    canvas = over_premul(canvas, near_1x)

    # Subtle volumetric acid-yellow mist bloom around the recessed slit window & beacon
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    us, vs, _ = cam.project(SLIT_POS)
    ub, vb, _ = cam.project(BEACON_POS)
    d_slit = np.hypot((xx - float(us)) / 180.0, (yy - float(vs)) / 270.0)
    d_beac = np.hypot((xx - float(ub)) / 165.0, (yy - float(vb)) / 145.0)
    acc_halo = (0.20 * np.exp(-d_slit * d_slit * 1.2) + 0.24 * np.exp(-d_beac * d_beac * 1.25)) * (0.88 + 0.12 * m_mot)
    canvas = 1.0 - (1.0 - np.clip(canvas, 0.0, 1.0)) * (1.0 - np.clip(acc_halo[..., None] * ACCENT_YELLOW[None, None, :], 0.0, 0.72))

    # Gentle front volumetric skylight haze crossing the near colonnade & water air
    shafts_front = paint_volumetric_skylight_shafts(cam, rng, intensity=0.25)
    canvas = 1.0 - (1.0 - np.clip(canvas, 0.0, 1.0)) * (1.0 - np.clip(shafts_front, 0.0, 0.65))

    # Stage 4: Upper-left foreground circular disc bokeh brutalist concrete louvre silhouette + 35mm optics
    fg_bokeh = paint_foreground_bokeh_louvre(rng)
    canvas = over_premul(canvas, fg_bokeh)

    final_rgb = apply_35mm_finish(canvas, rng)
    elapsed = time.perf_counter() - t0
    print(f"[mira-02] Rendered {W}x{H} in {elapsed:.2f}s")

    u8 = np.round(np.clip(final_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    full_im = Image.fromarray(u8, mode="RGB")

    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        full_im.save(audit_dir / "w02_full.png", optimize=True)
        crops = {
            "w02_crop_aperture.png": (int(us) - 310, int(vs) - 180),
            "w02_crop_midground.png": (int(ub) - 310, H - 450),
            "w02_crop_foreground.png": (220, H - 450),
            "w02_crop_bokeh.png": (0, 0),
        }
        for cname, (cx0, cy0) in crops.items():
            x0 = max(0, min(W - 600, cx0))
            y0 = max(0, min(H - 450, cy0))
            full_im.crop((x0, y0, x0 + 600, y0 + 450)).save(audit_dir / cname)

    if webp_outputs:
        for wp in webp_outputs:
            wp.parent.mkdir(parents=True, exist_ok=True)
            full_im.save(wp, format="WEBP", quality=90, method=6)
            print(f"[mira-02] Saved {wp} ({wp.stat().st_size / 1024:.1f} KiB)")

    return final_rgb, {"elapsed": elapsed, "vp_ratio": cam.vp_x / W}


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Render Mira 02 — Monolith Reflecting Atrium (2048x1152)")
    ap.add_argument("--seed", type=int, default=202602)
    ap.add_argument("--audit-dir", type=pathlib.Path, default=None)
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    outputs = None if args.no_export else [
        repo_root / "session" / "themes" / "mira" / "walls" / "wall-02.webp",
        repo_root / "docs" / "assets" / "walls" / "mira-02.webp",
    ]
    render_scene(seed=args.seed, audit_dir=args.audit_dir, webp_outputs=outputs)


if __name__ == "__main__":
    main()

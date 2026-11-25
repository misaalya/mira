#!/usr/bin/env python3
"""
Mira — Pure-Python 3D-Grounded 1990s Underwater Anime Scene Pipeline (2048 x 1152)

Five-stage deterministic painting pipeline (zero image models, zero external assets):
  Stage 1: Metric right-handed Camera3D looking slightly upward through deep water
           (off-axis vanishing point ~60% width), exact 3x3 UV-to-screen plane
           homographies masked by ray-plane depth z_cam > 0.05 with exponential
           underwater depth fog, near-plane 3D line clipping, and Voronoi (d2 - d1)
           fractured submerged concrete & masonry crevices.
  Stage 2: Single normalized diagonal SUN_DIR piercing the sunlit rippling water
           surface overhead (Snell's window + Fresnel ripples), casting aligned
           volumetric god rays and caustic highlights onto submerged stonework
           and the drifting girl.
  Stage 3: 4 depth-interleaved 2x supersampled PyCairo bands (Lanczos-downsampled)
           featuring tilted sunken concrete pillars, crumbling archways, exposed
           bent rebar, drifting rubble, rising air bubbles, and a 1990s cel-style
           young girl drifting horizontally face-up in graceful side profile with
           peacefully closed eyes, arms floating upward, legs trailing left, and
           weightless billowing hair & pale dress, broken up by isotropic
           fbm_gouache mottling, flat-brush dabs, paper tooth, and shadow bounce.
  Stage 4: Foreground lower-right broken concrete & rebar silhouette convolved in
           premultiplied RGBA with a circular aperture disc bokeh kernel (r = 7.8)
           plus 1 px opposite R/B lateral shift, followed by 35mm multi-scale
           halation, radial CA, optical vignette, shadow black lift + C1 tanh
           highlight shoulder, luminance-weighted silver-halide grain, and faint
           analog YIQ transfer patina.
  Stage 5: Deterministic render (<60s CPU) exporting full 2048x1152 frame, three
           1:1 (600x450) macro audit crops, and a clean web WebP asset.
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

# Stage 2: Unified diagonal 3D sunlight vector piercing surface -> abyss (z < 0)
_raw_sun = np.array([-0.38, 0.14, -0.914], dtype=np.float64)
SUN_DIR = _raw_sun / np.linalg.norm(_raw_sun)

# World vertical bounds (meters)
Z_FLOOR = -1.65
Z_SURF = 5.55


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


def flat_brush_dabs(h: int, w: int, rng: np.random.Generator, count: int = 90) -> np.ndarray:
    qh, qw = max(16, h // 4), max(16, w // 4)
    arr_sum = np.zeros((qh, qw), dtype=np.float32)
    for _ in range(count):
        cx = float(rng.uniform(0, qw))
        cy = float(rng.uniform(0, qh))
        ang = float(rng.uniform(0, math.pi))
        rx = float(rng.uniform(qw * 0.035, qw * 0.11))
        ry = float(rng.uniform(rx * 0.28, rx * 0.62))
        n_pts = int(rng.integers(5, 8))
        strength = float(rng.uniform(0.18, 0.44)) * (1.0 if rng.random() > 0.5 else -1.0)
        s_one = cairo.ImageSurface(cairo.FORMAT_A8, qw, qh)
        c1 = cairo.Context(s_one)
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
        s_one.flush()
        stride = s_one.get_stride()
        buf = np.frombuffer(s_one.get_data(), dtype=np.uint8).reshape(qh, stride)[:, :qw].astype(np.float32) / 255.0
        arr_sum += strength * buf
    arr_sum = gaussian_blur_2d(arr_sum, 3.2)
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
        fog_start: float = 4.0,
        fog_density: float = 0.085,
        depth_fog_grad: bool = True,
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

        if depth_fog_grad:
            # Blend toward vertical underwater ambient fog (brighter cyan higher, deeper Prussian blue lower)
            vy = (yy / self.h).astype(np.float32)
            top_fog = np.array([0.16, 0.48, 0.54], dtype=np.float32)
            bot_fog = np.array([0.014, 0.056, 0.118], dtype=np.float32)
            loc_fog = top_fog[None, None, :] * (1.0 - vy[..., None]) + bot_fog[None, None, :] * vy[..., None]
            loc_fog = 0.55 * loc_fog + 0.45 * fog_rgb[None, None, :]
        else:
            loc_fog = fog_rgb[None, None, :]

        fg_rgb = rgb_valid * (1.0 - fog_t[..., None]) + (loc_fog * alpha) * fog_t[..., None]
        patch = np.concatenate([fg_rgb, alpha], axis=-1)
        dst_rgb[y0:y1, x0:x1] = over_premul(dst_rgb[y0:y1, x0:x1], patch)
        return dst_rgb


# =============================================================================
# Stage 2 — Unified 3D Underwater Caustics & Overhead Surface Tracing
# =============================================================================

def caustic_field_at_surface(sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """Analytical multi-wave interference caustic intensity at the sunlit water surface."""
    w1 = np.sin(sx * 1.9 + sy * 1.15) * np.cos(sx * 1.1 - sy * 1.85)
    w2 = np.sin(sx * 3.4 - sy * 2.3 + 0.8) * np.cos(sx * 2.5 + sy * 3.1 - 0.5)
    w3 = np.sin(sx * 5.7 + sy * 4.2 - 1.2)
    ridge = 1.0 - np.abs(w1 * 0.55 + w2 * 0.32 + w3 * 0.13)
    return np.power(np.clip(ridge, 0.0, 1.0), 4.2).astype(np.float32)


def trace_surface_caustics(wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> np.ndarray:
    """Trace any 3D submerged point backward along -SUN_DIR to z = Z_SURF (6 steps with wave height)."""
    inv_sz = 1.0 / (-SUN_DIR[2])
    z_surf = np.full_like(wx, Z_SURF, dtype=np.float64)
    for _ in range(6):
        t = np.maximum(0.0, (z_surf - wz) * inv_sz)
        sx = wx - SUN_DIR[0] * t
        sy = wy - SUN_DIR[1] * t
        z_surf = Z_SURF + 0.08 * np.sin(sx * 1.4 + sy * 0.9) + 0.04 * np.cos(sx * 2.7 - sy * 1.8)
    t = np.maximum(0.0, (z_surf - wz) * inv_sz)
    sx = wx - SUN_DIR[0] * t
    sy = wy - SUN_DIR[1] * t
    caust = caustic_field_at_surface(sx, sy)
    depth_att = np.exp(-np.maximum(0.0, Z_SURF - wz) * 0.24).astype(np.float32)
    # Shadow of overhead broken archway lintel at y ~ 9.4, x < -1.8, z ~ 4.4
    t_arch = np.maximum(0.0, (4.4 - wz) * inv_sz)
    ax = wx - SUN_DIR[0] * t_arch
    ay = wy - SUN_DIR[1] * t_arch
    in_arch_sh = (wz < 4.4) & (ax > -5.8) & (ax < -1.6) & (np.abs(ay - 9.4) < 0.72)
    shadow = np.where(in_arch_sh, 0.32, 1.0).astype(np.float32)
    return caust * depth_att * shadow


# =============================================================================
# Stage 1 & 2 — Deep Underwater Backdrop, Rippling Sunlit Surface & Concrete Planes
# =============================================================================

def paint_underwater_backdrop_and_surface(cam: Camera3D, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Paint the abyssal-to-seafoam underwater gradient + exact 3D sunlit rippling
    water surface overhead (Snell's window + Fresnel capillary reflections).
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = yy / H
    u = xx / W

    mottle = fbm_gouache(H, W, rng)
    dabs = flat_brush_dabs(H, W, rng, count=92)
    tooth = paper_tooth(H, W, rng)

    # Melancholic 1990s underwater film palette: deep Prussian blue -> dark teal -> seafoam cyan
    c_abyss = np.array([0.012, 0.042, 0.098], dtype=np.float32)
    c_mid = np.array([0.046, 0.175, 0.255], dtype=np.float32)
    c_upper = np.array([0.145, 0.445, 0.505], dtype=np.float32)
    c_cream = np.array([0.84, 0.94, 0.89], dtype=np.float32)

    t1 = np.clip((v - 0.14) / 0.42, 0.0, 1.0)
    t2 = np.clip((v - 0.48) / 0.52, 0.0, 1.0)
    water = (
        c_upper[None, None, :] * (1.0 - t1[..., None])
        + c_mid[None, None, :] * (t1 * (1.0 - t2))[..., None]
        + c_abyss[None, None, :] * (t1 * t2)[..., None]
    )

    # Soft sunlit upper-right glow aligned with SUN_DIR entering the water
    du = u - 0.57
    dv = v - 0.09
    sun_bloom = np.exp(-(du * du * 3.6 + dv * dv * 11.5)).astype(np.float32)
    water += sun_bloom[..., None] * np.array([0.28, 0.40, 0.36], dtype=np.float32)
    water += (mottle * 0.014 + dabs * 0.010 + tooth * 0.005)[..., None]

    # Exact 3D ray-plane intersection with the overhead rippling water surface z = Z_SURF
    rx_c = (xx.astype(np.float64) - cam.K[0, 2]) / cam.K[0, 0]
    ry_c = (yy.astype(np.float64) - cam.K[1, 2]) / cam.K[1, 1]
    RT = cam.R.T
    dx_w = RT[0, 0] * rx_c + RT[0, 1] * ry_c + RT[0, 2]
    dy_w = RT[1, 0] * rx_c + RT[1, 1] * ry_c + RT[1, 2]
    dz_w = RT[2, 0] * rx_c + RT[2, 1] * ry_c + RT[2, 2]

    # Horizontal capillary wave octaves (ONLY used on water surface ripples!)
    rip1 = resize_bicubic(rng.standard_normal((H // 6, W // 52)).astype(np.float32), W, H)
    rip2 = resize_bicubic(rng.standard_normal((H // 3, W // 24)).astype(np.float32), W, H)
    rip = 0.62 * rip1 + 0.38 * rip2

    hit_surf = dz_w > 0.035
    t_surf = np.where(hit_surf, (Z_SURF - cam.eye[2]) / np.where(hit_surf, dz_w, 1.0), 1e5)
    sx = cam.eye[0] + dx_w * t_surf
    sy = cam.eye[1] + dy_w * t_surf

    # Elevation angle & Fresnel Snell's window transmission (strongest near top edge v < 0.28)
    horiz_dist = np.hypot(dx_w, dy_w)
    elev = np.arctan2(dz_w, np.maximum(1e-6, horiz_dist))
    # Perturb apparent elevation with capillary wave slopes -> bright undulating light ribbons!
    wave_h = (
        0.55 * np.sin(sx * 1.85 + sy * 0.95 + rip * 1.35)
        + 0.30 * np.cos(sx * 3.55 - sy * 2.15 + rip * 1.9)
        + 0.15 * np.sin(sx * 6.8 + sy * 4.4)
    ).astype(np.float32)

    snell = np.clip((elev.astype(np.float32) + wave_h * 0.038 - 0.175) / 0.24, 0.0, 1.0)
    snell *= np.exp(-np.maximum(0.0, t_surf.astype(np.float32) - 8.0) * 0.12)
    snell *= np.clip((0.36 - v) / 0.22, 0.0, 1.0)  # cleanly confined to upper water surface band!

    # Fresnel mirror underside reflection + bright sunlit wave troughs & foam shimmer
    dist_t = np.clip(v / 0.34, 0.0, 1.0)
    fresnel_w = 0.22 + (0.86 - 0.22) * np.power(1.0 - dist_t, 1.4)
    caust_s = caustic_field_at_surface(sx, sy)
    sun_spot = np.exp(-((u - 0.58) ** 2 * 4.2 + (v - 0.03) ** 2 * 24.0)).astype(np.float32)

    # Underside dark teal mirror waves contrasted against luminous seafoam & warm cream sky openings
    surf_dark = np.array([0.055, 0.23, 0.30], dtype=np.float32)[None, None, :]
    surf_cyan = np.array([0.46, 0.84, 0.84], dtype=np.float32)[None, None, :]
    crest_glow = np.clip(0.45 + 0.55 * wave_h, 0.0, 1.0)
    sky_break = np.clip(snell * (0.45 + 0.65 * crest_glow + 0.55 * caust_s + 0.72 * sun_spot), 0.0, 1.25)

    surf_col = surf_dark * (1.0 - fresnel_w[..., None] * 0.6) + surf_cyan * (sky_break * 0.72)[..., None]
    surf_col += np.power(np.clip(sky_break - 0.35, 0.0, 1.0), 1.4)[..., None] * (c_cream[None, None, :] * 0.78)
    # Crisp bright sunlit wave crest sparkles near the top Snell aperture
    sparkle = np.clip((wave_h + caust_s * 0.7 - 0.72) * 3.2, 0.0, 1.0) * snell * (0.45 + 0.85 * sun_spot)
    surf_col += sparkle[..., None] * np.array([0.72, 0.76, 0.64], dtype=np.float32)

    surf_alpha = np.clip(snell * 1.15, 0.0, 0.96)
    canvas = water * (1.0 - surf_alpha[..., None]) + surf_col * surf_alpha[..., None]
    fog_rgb = np.array([0.042, 0.165, 0.245], dtype=np.float32)
    return np.clip(canvas, 0.0, 1.4), fog_rgb


def paint_voronoi_concrete_uv(
    rng: np.random.Generator,
    x_span: tuple[float, float],
    y_span: tuple[float, float],
    is_floor: bool,
    tw: int = 896,
    th: int = 896,
) -> np.ndarray:
    """Submerged fractured concrete masonry with irregular Voronoi crevices & traced caustics."""
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    a_m = x_span[0] + u * (x_span[1] - x_span[0])
    b_m = y_span[0] + v * (y_span[1] - y_span[0])

    nx = resize_bicubic(rng.standard_normal((th // 28, tw // 28)).astype(np.float32), tw, th) * 0.065
    ny = resize_bicubic(rng.standard_normal((th // 28, tw // 28)).astype(np.float32), tw, th) * 0.065
    px, py = a_m + nx, b_m + ny

    cell_a, cell_b = (1.35, 1.45) if is_floor else (1.55, 1.20)
    ix = np.floor(px / cell_a).astype(np.int32)
    iy = np.floor(py / cell_b).astype(np.int32)

    jx_tab = rng.uniform(0.18, 0.82, size=256)
    jy_tab = rng.uniform(0.18, 0.82, size=256)
    tone_tab = rng.uniform(-0.04, 0.05, size=256)

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

    crack = np.clip(((d2 - d1).astype(np.float32) - 0.015) / 0.032, 0.0, 1.0)
    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=84)
    tooth = paper_tooth(th, tw, rng)
    sv = tone_tab[best_id].astype(np.float32)

    # Underwater weathered algae-tinted concrete
    base = np.zeros((th, tw, 3), dtype=np.float32)
    base[:, :, 0] = 0.048 + sv * 0.25 + mottle * 0.015 + dabs * 0.012
    base[:, :, 1] = 0.118 + sv * 0.32 + mottle * 0.020 + dabs * 0.015
    base[:, :, 2] = 0.155 + sv * 0.34 + mottle * 0.022 + dabs * 0.016

    crevice_col = np.array([0.012, 0.036, 0.062], dtype=np.float32)
    rgb = crack[..., None] * base + (1.0 - crack[..., None]) * crevice_col

    # Compute exact 3D world coords to apply SUN_DIR traced caustics
    if is_floor:
        wx, wy, wz = a_m, b_m, np.full_like(a_m, Z_FLOOR)
        caust = trace_surface_caustics(wx, wy, wz)
        rgb += (caust * crack)[..., None] * np.array([0.14, 0.36, 0.38], dtype=np.float32)
        alpha = np.ones((th, tw), dtype=np.float32)
    else:
        # Left broken wall: wx = -5.2, wy = a_m, wz = b_m (crumbled top silhouette!)
        wx = np.full_like(a_m, -5.2)
        wy, wz = a_m, b_m
        caust = trace_surface_caustics(wx, wy, wz)
        rgb += (caust * crack)[..., None] * np.array([0.22, 0.52, 0.54], dtype=np.float32)
        # Organic crumbled top ruin edge
        top_edge = 3.55 + 0.95 * np.sin(wy * 0.42 + 0.5) + 0.45 * np.cos(wy * 1.15 - 0.3) + 0.28 * mottle
        alpha = np.clip((top_edge - wz) / 0.14, 0.0, 1.0).astype(np.float32)

    # Warm bounce in deep shadows + cold-press paper tooth
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    rgb += np.clip((0.28 - lum) / 0.28, 0.0, 1.0)[..., None] * np.array([0.012, 0.014, 0.010], dtype=np.float32)
    rgb += tooth[..., None] * 0.007

    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.1) * alpha[..., None]
    rgba[:, :, 3] = alpha
    return rgba


# =============================================================================
# Stage 2 & 3 — Volumetric Diagonal God Rays Aligned With SUN_DIR
# =============================================================================

def paint_volumetric_god_rays(
    cam: Camera3D,
    rng: np.random.Generator,
    intensity: float = 1.0,
    front_pass: bool = False,
) -> np.ndarray:
    """Render additive/screen diagonal volumetric shafts strictly parallel to projected SUN_DIR."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # Project SUN_DIR direction in screen space around the hero depth (y = 7 m)
    p_ref = np.array([0.0, 7.0, 2.5], dtype=np.float64)
    u0, v0, _ = cam.project(p_ref)
    u1, v1, _ = cam.project(p_ref + SUN_DIR * 3.0)
    dx, dy = float(u1 - u0), float(v1 - v0)
    dlen = math.hypot(dx, dy) + 1e-6
    rx, ry = dx / dlen, dy / dlen       # unit ray direction in screen pixels (down-left!)
    nx, ny = -ry, rx                    # perpendicular coordinate across shafts

    # Coordinate along shaft (s) and perpendicular across shaft (p)
    s_coord = (xx - W * 0.56) * rx + (yy - 0.0) * ry
    p_coord = (xx - W * 0.56) * nx + (yy - 0.0) * ny

    acc = np.zeros((H, W), dtype=np.float32)
    # Seeded asymmetric shafts piercing from the sunlit surface down across the girl & left ruins
    shafts = (
        [(-280, 95, 0.34), (-95, 78, 0.52), (55, 118, 0.68), (225, 86, 0.56), (410, 110, 0.36)]
        if not front_pass
        else [(-160, 62, 0.24), (25, 84, 0.38), (175, 58, 0.32), (345, 70, 0.20)]
    )
    for center_p, width_p, strength in shafts:
        # Shaft widens softly with depth s_coord
        w_eff = width_p * (1.0 + np.maximum(0.0, s_coord) / (H * 1.45))
        beam = np.exp(-0.5 * ((p_coord - center_p) / w_eff) ** 2)
        # Fade gently from bright surface entry downward into abyssal depths
        vert = np.clip((s_coord + 220.0) / 260.0, 0.0, 1.0) * np.exp(-np.maximum(0.0, s_coord) / (H * 0.92))
        acc += beam * vert * strength

    # Subtle gouache turbidity breaks along rays
    turb = 0.82 + 0.18 * fbm_gouache(H // 4, W // 4, rng)
    turb_full = resize_bicubic(turb, W, H)
    acc = np.clip(acc * turb_full * intensity, 0.0, 1.0)

    ray_col = np.array([0.48, 0.86, 0.84], dtype=np.float32) if not front_pass else np.array([0.62, 0.92, 0.86], dtype=np.float32)
    return acc[..., None] * ray_col[None, None, :]


# =============================================================================
# Stage 3 — 2x Supersampled PyCairo Helpers, Sunken Ruins, Bubbles & Hero Girl
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
    fade_dist: float = 22.0,
    taper: bool = True,
    tremor: float = 0.35,
) -> None:
    if len(pts) < 2:
        return
    decay = float(np.clip(1.0 / (1.0 + z_depth / (fade_dist * 0.45)), 0.25, 1.0))
    w_eff = base_w * decay
    ext = [pts[0]] + pts + [pts[-1]]
    dense: list[tuple[float, float]] = []
    steps = 8
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


def draw_tilted_pillar(
    ctx: cairo.Context,
    cam: Camera3D,
    base_pos: np.ndarray,
    half_w: float,
    height_m: float,
    tilt_x: float,
    tilt_y: float,
    broken_top: bool,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    fog_start: float = 4.0,
    fog_density: float = 0.088,
) -> None:
    """Draw a massive tilted sunken concrete pillar with chipped top & exposed bent rebar."""
    dist = float(np.linalg.norm(base_pos - cam.eye))
    fog_t = 1.0 - math.exp(-max(0.0, dist - fog_start) * fog_density)

    def fog_c(c3):
        c = np.asarray(c3, dtype=np.float64) * (1.0 - fog_t) + fog_rgb * fog_t
        return (float(c[0]), float(c[1]), float(c[2]))

    axis = np.array([tilt_x, tilt_y, 1.0], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    # Build orthonormal cross-section frame
    ref = np.array([1.0, 0.0, 0.0])
    u_vec = ref - np.dot(ref, axis) * axis
    u_vec /= np.linalg.norm(u_vec)
    v_vec = np.cross(axis, u_vec)

    # Slight vertical height variation per corner for fractured concrete top
    h_offs = [float(rng.uniform(-0.38, 0.35)) if broken_top else 0.0 for _ in range(4)]
    c_signs = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    bot_pts = [base_pos + (sx * half_w) * u_vec + (sy * half_w) * v_vec for sx, sy in c_signs]
    top_pts = [bot_pts[i] + (height_m + h_offs[i]) * axis for i in range(4)]

    # Draw visible front (-v) and sunlit right (+u) faces
    faces = [
        ([bot_pts[0], bot_pts[1], top_pts[1], top_pts[0]], [0.040, 0.105, 0.155]),  # front shadow face
        ([bot_pts[1], bot_pts[2], top_pts[2], top_pts[1]], [0.115, 0.275, 0.335]),  # right sunlit face (faces -SUN_DIR!)
    ]
    if base_pos[0] > 0.8:
        # Pillar on right shows its left (-u) shaded face instead
        faces[1] = ([bot_pts[3], bot_pts[0], top_pts[0], top_pts[3]], [0.028, 0.082, 0.130])

    for pts3, base_col in faces:
        proj = [cam.project(p) for p in pts3]
        if any(z < Z_NEAR for _, _, z in proj):
            continue
        c_top = fog_c(np.asarray(base_col) * 1.45 + np.array([0.04, 0.12, 0.12]))
        c_bot = fog_c(np.asarray(base_col) * 0.62)
        g = cairo.LinearGradient(float(proj[2][0]), float(proj[2][1]), float(proj[0][0]), float(proj[0][1]))
        g.add_color_stop_rgba(0.0, *c_top, 1.0)
        g.add_color_stop_rgba(1.0, *c_bot, 1.0)
        ctx.set_source(g)
        ctx.move_to(float(proj[0][0]), float(proj[0][1]))
        for pu, pv, _ in proj[1:]:
            ctx.line_to(float(pu), float(pv))
        ctx.close_path()
        ctx.fill()

        # Subtle horizontal concrete pour seams & weathering streaks
        seam_col = fog_c([0.022, 0.062, 0.098])
        for k_s in range(1, 5):
            fr = k_s / 5.0
            pa = pts3[0] * (1 - fr) + pts3[3] * fr
            pb = pts3[1] * (1 - fr) + pts3[2] * fr
            seg = cam.project_polyline(np.array([pa, pb]))
            if len(seg) >= 2:
                stroke_catmull_rom(ctx, [(p[0], p[1]) for p in seg], base_w=1.6, z_depth=dist,
                                   rgba=(*seam_col, 0.55), rng=rng, taper=False, tremor=0.25)

    # Sunlit edge rim highlight along corner facing SUN_DIR
    edge_seg = cam.project_polyline(np.array([bot_pts[1], top_pts[1]]))
    if len(edge_seg) >= 2:
        rim_c = fog_c([0.36, 0.72, 0.74])
        stroke_catmull_rom(ctx, [(p[0], p[1]) for p in edge_seg], base_w=2.4, z_depth=dist,
                           rgba=(*rim_c, 0.62), rng=rng, taper=False, tremor=0.20)

    # Exposed bent steel rebar rods emerging from fractured concrete top!
    if broken_top:
        rebar_col = fog_c([0.025, 0.055, 0.088])
        for _ in range(int(rng.integers(4, 7))):
            ox = float(rng.uniform(-half_w * 0.72, half_w * 0.72))
            oy = float(rng.uniform(-half_w * 0.72, half_w * 0.72))
            p_start = base_pos + ox * u_vec + oy * v_vec + (height_m - 0.12) * axis
            bend_x = float(rng.uniform(-0.45, 0.55))
            bend_y = float(rng.uniform(-0.35, 0.35))
            ln = float(rng.uniform(0.55, 1.35))
            pts_r = np.array([
                p_start,
                p_start + axis * (ln * 0.45) + u_vec * (bend_x * 0.25),
                p_start + axis * (ln * 0.85) + u_vec * bend_x + v_vec * bend_y,
            ])
            rseg = cam.project_polyline(pts_r)
            if len(rseg) >= 2:
                stroke_catmull_rom(ctx, [(p[0], p[1]) for p in rseg], base_w=3.2, z_depth=dist,
                                   rgba=(*rebar_col, 0.92), rng=rng, taper=False, tremor=0.18)


def draw_sunken_archway(
    ctx: cairo.Context,
    cam: Camera3D,
    x_left: float,
    x_right: float,
    y_plane: float,
    z_spring: float,
    z_top: float,
    thickness: float,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    fog_density: float = 0.088,
) -> None:
    """Draw a crumbling submerged concrete archway spanning between ruins."""
    center = np.array([(x_left + x_right) * 0.5, y_plane, z_spring], dtype=np.float64)
    dist = float(np.linalg.norm(center - cam.eye))
    fog_t = 1.0 - math.exp(-max(0.0, dist - 4.0) * fog_density)

    def fog_c(c3):
        c = np.asarray(c3, dtype=np.float64) * (1.0 - fog_t) + fog_rgb * fog_t
        return (float(c[0]), float(c[1]), float(c[2]))

    cx = (x_left + x_right) * 0.5
    rad = (x_right - x_left) * 0.5
    n_arc = 18
    # Outer broken top contour + inner arch curve
    outer_pts = []
    for i in range(n_arc + 1):
        fr = i / n_arc
        x = x_left + (x_right - x_left) * fr
        # Crumbling dip near middle-right so arch looks authentically fractured
        crumble = -0.35 * math.exp(-((fr - 0.68) / 0.16) ** 2) + float(rng.uniform(-0.07, 0.07))
        outer_pts.append(np.array([x, y_plane, z_top + crumble]))

    inner_pts = []
    for i in range(n_arc, -1, -1):
        ang = math.pi * (i / n_arc)
        x = cx - math.cos(ang) * rad * 0.78
        z = z_spring + math.sin(ang) * rad * 0.72
        inner_pts.append(np.array([x, y_plane, z]))

    # Draw soffit thickness (back-to-front extrusion along +Y)
    soffit_col = fog_c([0.025, 0.072, 0.115])
    for i in range(len(inner_pts) - 1):
        q = [
            inner_pts[i],
            inner_pts[i + 1],
            inner_pts[i + 1] + np.array([0.0, thickness, 0.0]),
            inner_pts[i] + np.array([0.0, thickness, 0.0]),
        ]
        pr = [cam.project(p) for p in q]
        if any(z < Z_NEAR for _, _, z in pr):
            continue
        ctx.set_source_rgba(*soffit_col, 0.96)
        ctx.move_to(float(pr[0][0]), float(pr[0][1]))
        for pu, pv, _ in pr[1:]:
            ctx.line_to(float(pu), float(pv))
        ctx.close_path()
        ctx.fill()

    # Front concrete spandrel face
    poly = outer_pts + inner_pts
    proj = [cam.project(p) for p in poly]
    if not any(z < Z_NEAR for _, _, z in proj):
        c_top = fog_c([0.11, 0.27, 0.34])
        c_bot = fog_c([0.036, 0.098, 0.148])
        g = cairo.LinearGradient(float(proj[0][0]), float(proj[0][1]), float(proj[-1][0]), float(proj[-1][1]))
        g.add_color_stop_rgba(0.0, *c_top, 0.98)
        g.add_color_stop_rgba(1.0, *c_bot, 0.98)
        ctx.set_source(g)
        ctx.move_to(float(proj[0][0]), float(proj[0][1]))
        for pu, pv, _ in proj[1:]:
            ctx.line_to(float(pu), float(pv))
        ctx.close_path()
        ctx.fill()


def draw_drifting_rubble(
    ctx: cairo.Context,
    cam: Camera3D,
    count: int,
    y_range: tuple[float, float],
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
) -> None:
    """Scatter delicate tumbling fractured concrete shards & plaster flakes near ruins."""
    for _ in range(count):
        wy = float(rng.uniform(*y_range))
        # Concentrate debris organically near left ruins and deeper abyss, keeping hero center airy
        wx = float(rng.uniform(-4.8, 3.8))
        wz = float(rng.uniform(-1.1, 3.8))
        if abs(wx) < 2.2 and 0.8 < wz < 4.5:
            continue
        ux, uy, uz = cam.project(np.array([wx, wy, wz]))
        if uz < 1.8 or ux < -30 or ux > W + 30 or uy < 190 or uy > H + 30:
            continue
        dist = float(np.linalg.norm(np.array([wx, wy, wz]) - cam.eye))
        fog_t = 1.0 - math.exp(-max(0.0, dist - 4.0) * 0.095)
        px_m = cam.K[0, 0] / float(uz)
        rad = float(rng.uniform(0.018, 0.068)) * px_m
        n_v = int(rng.integers(5, 9))
        ang0 = float(rng.uniform(0, math.tau))
        pts = []
        for k in range(n_v):
            a = ang0 + (math.tau * k) / n_v + float(rng.uniform(-0.28, 0.28))
            r = rad * float(rng.uniform(0.45, 1.18))
            pts.append((float(ux) + math.cos(a) * r, float(uy) + math.sin(a) * r * float(rng.uniform(0.55, 0.95))))

        base_c = np.array([0.036, 0.105, 0.155]) * (1.0 - fog_t) + fog_rgb * fog_t
        lit_c = np.array([0.18, 0.44, 0.48]) * (1.0 - fog_t) + fog_rgb * fog_t
        g = cairo.LinearGradient(float(ux) + rad * 0.6, float(uy) - rad * 0.6, float(ux) - rad * 0.6, float(uy) + rad * 0.6)
        g.add_color_stop_rgba(0.0, float(lit_c[0]), float(lit_c[1]), float(lit_c[2]), 0.86)
        g.add_color_stop_rgba(1.0, float(base_c[0]), float(base_c[1]), float(base_c[2]), 0.86)
        ctx.set_source(g)
        ctx.move_to(*pts[0])
        for p in pts[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.fill()


def draw_bubble(ctx: cairo.Context, x: float, y: float, r: float, alpha: float = 0.85) -> None:
    """Delicate 1990s anime underwater air bubble: dark teal Fresnel rim + crisp cyan-cream glint."""
    ctx.save()
    ctx.translate(x, y)
    # Slight buoyancy flattening on larger bubbles
    sy = 0.86 if r > 4.5 else 0.94
    ctx.scale(1.0, sy)

    # Translucent glowing interior
    g = cairo.RadialGradient(-r * 0.25, -r * 0.28, r * 0.1, 0.0, 0.0, r)
    g.add_color_stop_rgba(0.0, 0.85, 0.98, 0.96, 0.26 * alpha)
    g.add_color_stop_rgba(0.72, 0.22, 0.66, 0.74, 0.18 * alpha)
    g.add_color_stop_rgba(1.0, 0.62, 0.94, 0.94, 0.65 * alpha)
    ctx.set_source(g)
    ctx.arc(0, 0, r, 0, math.tau)
    ctx.fill()

    # Crisp thin rim outline
    ctx.set_source_rgba(0.76, 0.96, 0.96, 0.68 * alpha)
    ctx.set_line_width(max(0.65, r * 0.13))
    ctx.arc(0, 0, r, 0, math.tau)
    ctx.stroke()

    # Upper-right sunlit specular crescent (aligned with SUN_DIR!)
    ctx.set_source_rgba(0.98, 1.0, 0.96, 0.94 * alpha)
    ctx.arc(r * 0.30, -r * 0.32, max(0.8, r * 0.24), 0, math.tau)
    ctx.fill()
    if r > 3.2:
        ctx.set_source_rgba(0.62, 0.94, 0.92, 0.55 * alpha)
        ctx.arc(-r * 0.26, r * 0.34, r * 0.14, 0, math.tau)
        ctx.fill()
    ctx.restore()


def draw_bubble_streams(ctx: cairo.Context, rng: np.random.Generator, hero_near: bool = False) -> None:
    """Draw organic rising trails of air bubbles & tiny suspended underwater dust specks."""
    if hero_near:
        # Delicate bubble trails curling gently upward from near the girl's lips, fingertips & hem!
        origins = [
            (1165, 558, 16, 1.6, 5.2, 290),   # rising from parted lips/nose
            (1082, 490, 10, 1.4, 4.0, 240),   # rising from far floating fingertips
            (1008, 508, 11, 1.3, 4.4, 260),   # rising from near floating hand
            (895, 608, 12, 1.5, 5.0, 330),    # rising from billowing dress hem
            (1265, 520, 10, 1.2, 3.8, 270),   # rising through trailing hair
        ]
    else:
        origins = [
            (460, 960, 18, 2.0, 8.5, 620),
            (680, 1020, 15, 1.6, 6.2, 580),
            (1420, 920, 22, 1.8, 7.6, 720),
            (1640, 860, 16, 1.6, 6.5, 640),
            (1310, 1040, 14, 1.5, 5.5, 540),
        ]

    for ox, oy, count, r_min, r_max, rise_h in origins:
        for _ in range(count):
            t = float(rng.uniform(0.0, 1.0))
            bx = ox + math.sin(t * 7.5 + ox * 0.02) * (10.0 + 26.0 * t) + float(rng.normal(0, 12.0 * t + 3.0))
            by = oy - t * rise_h
            if by < 10 or by > H - 10:
                continue
            r = float(rng.uniform(r_min, r_max)) * (0.65 + 0.55 * t)
            draw_bubble(ctx, bx, by, r, alpha=float(rng.uniform(0.52, 0.90)) * (1.0 - 0.25 * t))

    # Fine drifting underwater plankton / light-catching dust specks
    n_specks = 140 if hero_near else 180
    for _ in range(n_specks):
        sx = float(rng.uniform(30, W - 30))
        sy = float(rng.uniform(90, H - 40))
        sr = float(rng.uniform(0.7, 2.1))
        a = float(rng.uniform(0.14, 0.48)) * (1.15 - 0.65 * (sy / H))
        ctx.set_source_rgba(0.76, 0.96, 0.94, a)
        ctx.arc(sx, sy, sr, 0, math.tau)
        ctx.fill()


# =============================================================================
# Stage 3 HERO — 1990s Anime Cel Girl Drifting Horizontally Face-Up
# =============================================================================

def _draw_wave_hair_ribbon(
    ctx: cairo.Context,
    ctrl_pts: list[tuple[float, float]],
    w_start: float,
    w_mid: float,
    w_end: float,
    fill_rgba: tuple[float, float, float, float],
    rim_rgba: tuple[float, float, float, float] | None = None,
) -> None:
    """Paint a lush, sinusoidal S-curved 1990s underwater hair ribbon via Catmull-Rom spine."""
    if len(ctrl_pts) < 3:
        return
    ext = [ctrl_pts[0]] + ctrl_pts + [ctrl_pts[-1]]
    spine: list[tuple[float, float]] = []
    steps = 10
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for s in range(steps):
            t = s / steps
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            spine.append((x, y))
    spine.append(ctrl_pts[-1])

    n = len(spine) - 1
    left_edge = []
    right_edge = []
    for i, (x, y) in enumerate(spine):
        prev_p = spine[max(0, i - 1)]
        next_p = spine[min(n, i + 1)]
        tx, ty = next_p[0] - prev_p[0], next_p[1] - prev_p[1]
        tl = math.hypot(tx, ty) + 1e-6
        nx, ny = -ty / tl, tx / tl
        t = i / max(1, n)
        # Organic swelling ribbon width that tapers gracefully into a soft curling tip
        hw = (1 - t) ** 2 * w_start + 2 * (1 - t) * t * w_mid + t**2 * w_end
        left_edge.append((x + nx * hw, y + ny * hw))
        right_edge.append((x - nx * hw, y - ny * hw))

    ctx.set_source_rgba(*fill_rgba)
    ctx.move_to(*left_edge[0])
    for p in left_edge[1:]:
        ctx.line_to(*p)
    for p in reversed(right_edge):
        ctx.line_to(*p)
    ctx.close_path()
    ctx.fill()

    if rim_rgba is not None:
        # Soft backlit cyan rim grazing the upper sunlit side of the flowing ribbon
        ctx.set_source_rgba(*rim_rgba)
        ctx.set_line_width(1.25)
        ctx.move_to(*left_edge[2])
        for p in left_edge[3:-2]:
            ctx.line_to(*p)
        ctx.stroke()


def _stroke_hair_strand(
    ctx: cairo.Context,
    ctrl_pts: list[tuple[float, float]],
    width: float,
    rgba: tuple[float, float, float, float],
) -> None:
    """Delicate tapering stray hair wisp drifting organically in the water current."""
    ext = [ctrl_pts[0]] + ctrl_pts + [ctrl_pts[-1]]
    dense = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for s in range(8):
            t = s / 8.0
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            dense.append((x, y))
    dense.append(ctrl_pts[-1])
    ctx.set_source_rgba(*rgba)
    n = len(dense) - 1
    for i in range(n):
        t = i / max(1, n)
        ctx.set_line_width(max(0.45, width * (1.0 - 0.78 * t)))
        ctx.move_to(*dense[i])
        ctx.line_to(*dense[i + 1])
        ctx.stroke()


def draw_drifting_anime_girl(ctx: cairo.Context, rng: np.random.Generator) -> None:
    """
    1990s melancholic anime cel-style young girl drifting suspended horizontally
    near frame center, face-up in graceful supine side profile:
      - Proportional rounded anime head (~76 px) on the RIGHT tilted softly back
        toward the sunlit surface, clean profile with peacefully closed eye.
      - Cohesive, flowing SILK-VEIL long dark hair drifting gently upward-right
        with translucent fading tips (unified soft underwater current, NEVER
        separated spiky tentacles!).
      - Clear slender human torso with soft dipping concave waist and airy,
        draped pale sundress whose translucent hem billows gently upward.
      - Asymmetric, softly curved weightless arms with relaxed drooping wrists.
      - Naturally bent, gently parted legs trailing gracefully toward the LEFT.
    """
    INK = (0.065, 0.155, 0.235, 0.84)        # soft deep-teal 1990s cel ink (delicate, never harsh)
    SKIN_HI = (0.94, 0.97, 0.95, 1.0)        # sunlit pale warm cel skin
    SKIN_MID = (0.76, 0.87, 0.89, 1.0)
    SKIN_SH = (0.38, 0.60, 0.71, 1.0)        # cool cyan-teal underwater cel shadow
    DRESS_HI = (0.96, 0.99, 0.98, 0.96)
    DRESS_MID = (0.78, 0.90, 0.93, 0.92)
    DRESS_SH = (0.34, 0.58, 0.70, 0.90)

    ctx.save()
    # Centered smoothly below the sunlit surface aperture (fits cleanly beneath MIRA+ on hero too!)
    ctx.translate(0.0, 48.0)

    # Gentle underwater sunlight scatter aura behind her pale form
    aura = cairo.RadialGradient(1045, 595, 30, 1045, 595, 320)
    aura.add_color_stop_rgba(0.0, 0.54, 0.90, 0.90, 0.14)
    aura.add_color_stop_rgba(0.55, 0.28, 0.68, 0.76, 0.05)
    aura.add_color_stop_rgba(1.0, 0.10, 0.35, 0.48, 0.0)
    ctx.set_source(aura)
    ctx.arc(1045, 595, 320, 0, math.tau)
    ctx.fill()

    # Pull head + hair veil closer to shoulders (-26, +24) for natural slender anime neck (~22 px!)
    HEAD_DX, HEAD_DY = -26.0, 24.0

    # =========================================================================
    # 1. COHESIVE FLOWING SILK-VEIL LONG HAIR (UNIFIED CURTAIN, ZERO TENTACLES!)
    #    Soft rich midnight-teal-indigo silk veil drifting smoothly upward-right
    #    with gentle overlapping wave folds & translucent fading tips.
    # =========================================================================
    ctx.save()
    ctx.translate(HEAD_DX, HEAD_DY)
    # Layer 1A: Outer sheer billowing hair veil (soft translucent fade into turquoise water!)
    hv_outer = cairo.LinearGradient(1178, 565, 1435, 455)
    hv_outer.add_color_stop_rgba(0.0, 0.068, 0.155, 0.245, 0.90)
    hv_outer.add_color_stop_rgba(0.52, 0.088, 0.205, 0.305, 0.74)
    hv_outer.add_color_stop_rgba(0.84, 0.115, 0.265, 0.365, 0.42)
    hv_outer.add_color_stop_rgba(1.0, 0.155, 0.345, 0.435, 0.08)
    ctx.set_source(hv_outer)
    ctx.move_to(1178, 524)                             # upper forehead/crown hairline
    ctx.curve_to(1224, 488, 1286, 446, 1348, 434)      # smooth soaring upper silk wave crest
    ctx.curve_to(1388, 426, 1420, 440, 1434, 464)      # soft rounded curling upper tip
    ctx.curve_to(1408, 466, 1384, 480, 1372, 496)      # delicate gentle wave scallop
    ctx.curve_to(1396, 498, 1418, 512, 1424, 534)      # soft secondary trailing curl
    ctx.curve_to(1386, 548, 1346, 570, 1296, 588)      # sweeping silky underside
    ctx.curve_to(1248, 606, 1202, 604, 1166, 584)      # soft tuck beneath nape
    ctx.close_path()
    ctx.fill()

    # Layer 1B: Main rich mid-indigo silk mass (cohesive overlapping S-wave body)
    hv_main = cairo.LinearGradient(1182, 558, 1396, 476)
    hv_main.add_color_stop_rgba(0.0, 0.048, 0.112, 0.192, 0.96)
    hv_main.add_color_stop_rgba(0.58, 0.068, 0.158, 0.248, 0.88)
    hv_main.add_color_stop_rgba(1.0, 0.105, 0.238, 0.335, 0.36)
    ctx.set_source(hv_main)
    ctx.move_to(1186, 526)
    ctx.curve_to(1234, 502, 1288, 470, 1344, 458)
    ctx.curve_to(1378, 452, 1398, 462, 1406, 480)
    ctx.curve_to(1374, 488, 1348, 504, 1326, 520)
    ctx.curve_to(1354, 520, 1378, 528, 1388, 544)
    ctx.curve_to(1350, 560, 1304, 578, 1254, 586)
    ctx.curve_to(1214, 592, 1184, 586, 1164, 574)
    ctx.close_path()
    ctx.fill()

    # Layer 1C: Deep shadow under-fold beneath her nape for smooth 3D hair volume
    hv_deep = cairo.LinearGradient(1174, 570, 1340, 545)
    hv_deep.add_color_stop_rgba(0.0, 0.036, 0.086, 0.156, 0.94)
    hv_deep.add_color_stop_rgba(0.65, 0.054, 0.125, 0.205, 0.78)
    hv_deep.add_color_stop_rgba(1.0, 0.080, 0.185, 0.275, 0.18)
    ctx.set_source(hv_deep)
    ctx.move_to(1168, 570)
    ctx.curve_to(1212, 578, 1266, 570, 1318, 548)
    ctx.curve_to(1346, 536, 1368, 536, 1378, 548)
    ctx.curve_to(1342, 572, 1288, 594, 1236, 596)
    ctx.curve_to(1196, 598, 1172, 588, 1158, 576)
    ctx.close_path()
    ctx.fill()

    # Layer 1D: Subtle glowing cyan-seafoam silk sheen folds INSIDE the unified hair veil
    for pts, w_val, a_val in (
        ([(1194, 530), (1244, 502), (1298, 472), (1358, 456), (1408, 458)], 4.8, 0.34),
        ([(1196, 542), (1252, 522), (1312, 496), (1366, 488)], 3.6, 0.28),
        ([(1188, 556), (1248, 546), (1312, 528), (1372, 526)], 3.2, 0.22),
        ([(1184, 522), (1228, 488), (1284, 454), (1346, 438)], 2.2, 0.52),
    ):
        _stroke_hair_strand(ctx, pts, w_val, (0.48, 0.86, 0.88, a_val))

    # Only 3 soft delicate drifting hair ribbons curling organically at the upper sunlit border
    for pts, w0, wm, we, fcol, rcol in (
        ([(1190, 524), (1228, 488), (1274, 444), (1324, 418)], 5.5, 4.0, 0.5, (0.075, 0.175, 0.268, 0.72), (0.56, 0.92, 0.92, 0.52)),
        ([(1196, 536), (1246, 506), (1302, 474), (1362, 454), (1412, 448)], 5.0, 3.6, 0.5, (0.062, 0.145, 0.232, 0.78), (0.52, 0.90, 0.90, 0.45)),
        ([(1180, 566), (1232, 586), (1292, 590), (1350, 574)], 5.0, 3.4, 0.5, (0.048, 0.112, 0.188, 0.74), None),
    ):
        _draw_wave_hair_ribbon(ctx, pts, w0, wm, we, fcol, rcol)
    ctx.restore()

    # =========================================================================
    # 2. FAR ARM (LEFT ARM) — GRACEFULLY BENT, FLOATING SOFTLY UP-RIGHT
    #    Anchored naturally behind far chest/shoulder (not poking out of throat!)
    # =========================================================================
    ctx.save()
    ctx.translate(-12.0, 4.0)
    g_far_arm = cairo.LinearGradient(1122, 452, 1118, 592)
    g_far_arm.add_color_stop_rgba(0.0, 0.68, 0.84, 0.87, 1.0)
    g_far_arm.add_color_stop_rgba(0.55, 0.48, 0.70, 0.78, 1.0)
    g_far_arm.add_color_stop_rgba(1.0, 0.28, 0.48, 0.60, 1.0)
    ctx.set_source(g_far_arm)
    ctx.move_to(1126, 588)                             # far shoulder base
    ctx.curve_to(1140, 558, 1144, 526, 1134, 502)      # soft upper arm curving gently out
    ctx.curve_to(1126, 484, 1112, 472, 1098, 462)      # relaxed bent forearm drifting up-left
    # Graceful drooping relaxed anime hand & wrist
    ctx.curve_to(1090, 452, 1080, 444, 1072, 444)      # arched back of hand -> delicate index tip
    ctx.curve_to(1067, 445, 1068, 451, 1074, 454)
    ctx.curve_to(1069, 455, 1068, 460, 1075, 462)      # gently trailing soft fingers
    ctx.curve_to(1078, 466, 1086, 468, 1092, 472)      # slender wrist
    ctx.curve_to(1104, 484, 1116, 498, 1120, 512)      # inner elbow bend
    ctx.curve_to(1122, 534, 1114, 562, 1106, 588)      # inner upper arm
    ctx.close_path()
    ctx.fill_preserve()
    ctx.set_source_rgba(*INK)
    ctx.set_line_width(1.25)
    ctx.stroke()
    ctx.restore()

    # =========================================================================
    # 3. GRACEFULLY BENT, GENTLY PARTED LEGS TRAILING TO THE LEFT
    #    Natural relaxed knee flexion (NOT ramrod-straight pencils!)
    # =========================================================================
    # Far (lower) leg: deeper relaxed knee bend, trailing softly lower
    g_fleg = cairo.LinearGradient(845, 622, 845, 706)
    g_fleg.add_color_stop_rgba(0.0, 0.56, 0.74, 0.80, 1.0)
    g_fleg.add_color_stop_rgba(0.55, 0.36, 0.58, 0.68, 1.0)
    g_fleg.add_color_stop_rgba(1.0, 0.22, 0.42, 0.56, 1.0)
    ctx.set_source(g_fleg)
    ctx.move_to(984, 636)                              # hip/thigh under skirt
    ctx.curve_to(936, 634, 892, 640, 862, 650)         # thigh curving into relaxed knee
    ctx.curve_to(826, 662, 788, 676, 764, 686)         # slender shin -> ankle
    ctx.curve_to(746, 692, 732, 704, 722, 712)         # relaxed pointed instep -> toes
    ctx.curve_to(720, 716, 726, 718, 736, 714)         # soft sole -> heel
    ctx.curve_to(750, 706, 762, 700, 774, 696)
    ctx.curve_to(802, 692, 838, 680, 870, 668)         # graceful feminine calf -> back of knee
    ctx.curve_to(908, 664, 948, 664, 984, 660)         # underside thigh
    ctx.close_path()
    ctx.fill_preserve()
    ctx.set_source_rgba(*INK)
    ctx.set_line_width(1.30)
    ctx.stroke()

    # Near (upper) leg: gentle serene floating arch with soft knee bend & sunlit cyan rim!
    g_nleg = cairo.LinearGradient(855, 598, 855, 668)
    g_nleg.add_color_stop_rgba(0.0, *SKIN_HI)
    g_nleg.add_color_stop_rgba(0.54, *SKIN_MID)
    g_nleg.add_color_stop_rgba(1.0, *SKIN_SH)
    ctx.set_source(g_nleg)
    ctx.move_to(992, 616)                              # upper thigh under skirt
    ctx.curve_to(942, 608, 896, 604, 864, 606)         # thigh -> soft slightly lifted knee
    ctx.curve_to(826, 608, 788, 618, 762, 624)         # graceful dipping shin -> slim ankle
    ctx.curve_to(744, 628, 728, 636, 716, 642)         # delicate relaxed bare foot instep -> toes
    ctx.curve_to(714, 647, 720, 650, 732, 648)         # soft sole
    ctx.curve_to(748, 644, 760, 638, 772, 636)         # ankle/heel
    ctx.curve_to(802, 638, 838, 632, 870, 626)         # shapely slender calf -> back of knee
    ctx.curve_to(912, 632, 952, 642, 992, 646)         # lower thigh
    ctx.close_path()
    ctx.fill_preserve()
    ctx.set_source_rgba(*INK)
    ctx.set_line_width(1.35)
    ctx.stroke()

    # Sunlit highlight along upper thigh/knee/shin facing SUN_DIR
    ctx.set_source_rgba(0.98, 1.0, 0.96, 0.72)
    ctx.set_line_width(1.4)
    ctx.move_to(916, 607)
    ctx.curve_to(874, 605, 822, 610, 766, 623)
    ctx.stroke()

    # =========================================================================
    # 4. NATURAL SHORT SLENDER NECK & PROPORTIONAL 1990s ANIME PROFILE HEAD
    #    Clear, serene, classic anime girl proportions (~22 px neck, zero stretch!)
    # =========================================================================
    neck_g = cairo.LinearGradient(1118, 568, 1128, 608)
    neck_g.add_color_stop_rgba(0.0, *SKIN_HI)
    neck_g.add_color_stop_rgba(0.62, *SKIN_MID)
    neck_g.add_color_stop_rgba(1.0, *SKIN_SH)
    ctx.set_source(neck_g)
    ctx.move_to(1106, 586)                             # soft collarbone base
    ctx.curve_to(1114, 578, 1120, 570, 1124, 562)      # short natural upturned throat (~22 px!)
    ctx.line_to(1152, 582)                             # tucked softly under jaw/nape
    ctx.curve_to(1142, 594, 1134, 602, 1124, 608)      # upper shoulder/back join
    ctx.close_path()
    ctx.fill_preserve()
    ctx.set_source_rgba(*INK)
    ctx.set_line_width(1.25)
    ctx.stroke()

    # Larger, clearly readable anime head (1.16x scale) nestled naturally at short neck!
    ctx.save()
    ctx.translate(HEAD_DX, HEAD_DY)
    ctx.translate(1178.0, 548.0)
    ctx.scale(1.16, 1.16)
    ctx.translate(-1178.0, -548.0)

    head_g = cairo.LinearGradient(1176, 506, 1184, 576)
    head_g.add_color_stop_rgba(0.0, 0.96, 0.98, 0.96, 1.0)
    head_g.add_color_stop_rgba(0.55, 0.80, 0.90, 0.91, 1.0)
    head_g.add_color_stop_rgba(1.0, 0.44, 0.65, 0.75, 1.0)
    ctx.set_source(head_g)
    ctx.move_to(1209, 540)                             # crown
    ctx.curve_to(1203, 525, 1192, 517, 1181, 517)      # smooth rounded anime forehead
    ctx.curve_to(1176, 517, 1173, 518, 1171, 516)      # delicate brow dip
    ctx.curve_to(1168, 512, 1164, 507, 1162, 508)      # tiny cute upturned anime nose
    ctx.curve_to(1161, 509, 1162, 514, 1159, 516)      # philtrum
    ctx.curve_to(1156, 515, 1154, 516, 1153, 518)      # soft upper lip
    ctx.curve_to(1152, 519, 1152, 520, 1150, 520)      # peaceful lip seam
    ctx.curve_to(1148, 520, 1146, 521, 1145, 523)      # soft lower lip
    ctx.curve_to(1142, 524, 1139, 527, 1139, 531)      # delicate small chin
    ctx.curve_to(1141, 539, 1148, 550, 1162, 559)      # graceful soft jawline
    ctx.curve_to(1178, 569, 1198, 572, 1210, 562)      # back of skull
    ctx.curve_to(1218, 555, 1216, 546, 1209, 540)
    ctx.close_path()
    ctx.fill_preserve()
    ctx.set_source_rgba(*INK)
    ctx.set_line_width(1.25)
    ctx.stroke()

    # Soft cheek shading & delicate blush-cyan shadow plane
    ctx.set_source_rgba(0.42, 0.65, 0.76, 0.24)
    ctx.move_to(1144, 534)
    ctx.curve_to(1150, 546, 1160, 554, 1174, 558)
    ctx.curve_to(1172, 546, 1162, 538, 1148, 532)
    ctx.close_path()
    ctx.fill()

    # Ear tucked softly behind temple hair
    ctx.set_source_rgba(*SKIN_MID)
    ctx.move_to(1174, 550)
    ctx.curve_to(1182, 558, 1190, 552, 1185, 544)
    ctx.curve_to(1182, 540, 1176, 542, 1174, 546)
    ctx.fill_preserve()
    ctx.set_source_rgba(*INK)
    ctx.set_line_width(0.95)
    ctx.stroke()

    # PEACEFULLY CLOSED EYE: serene curved lash arc + gentle brow
    ctx.set_source_rgba(0.05, 0.12, 0.20, 0.92)
    ctx.move_to(1180.5, 531.0)
    ctx.curve_to(1175.5, 532.6, 1170.5, 530.8, 1167.0, 527.0)
    ctx.curve_to(1170.8, 528.8, 1175.8, 530.4, 1181.2, 529.0)
    ctx.close_path()
    ctx.fill()
    ctx.set_line_width(0.85)
    ctx.move_to(1180.5, 530.4)
    ctx.line_to(1183.2, 532.0)
    ctx.stroke()
    ctx.set_source_rgba(0.06, 0.14, 0.23, 0.72)
    ctx.set_line_width(1.0)
    ctx.move_to(1184.5, 524.2)
    ctx.curve_to(1179.8, 522.2, 1175.5, 521.2, 1172.0, 521.0)
    ctx.stroke()

    # Sunlit nose/forehead highlight
    ctx.set_source_rgba(0.99, 1.0, 0.96, 0.80)
    ctx.set_line_width(1.2)
    ctx.move_to(1194, 519)
    ctx.curve_to(1184, 517, 1174, 516, 1163, 509)
    ctx.stroke()

    # Soft rounded hair crown & delicate framing bangs hugging her head naturally!
    crown_g = cairo.LinearGradient(1186, 514, 1218, 574)
    crown_g.add_color_stop_rgba(0.0, 0.095, 0.210, 0.305, 0.96)
    crown_g.add_color_stop_rgba(0.55, 0.052, 0.122, 0.205, 0.98)
    crown_g.add_color_stop_rgba(1.0, 0.036, 0.086, 0.155, 0.98)
    ctx.set_source(crown_g)
    ctx.move_to(1192, 516)
    ctx.curve_to(1212, 516, 1226, 532, 1224, 552)
    ctx.curve_to(1220, 570, 1198, 582, 1174, 576)
    ctx.curve_to(1164, 574, 1162, 566, 1170, 558)
    ctx.curve_to(1184, 556, 1194, 546, 1194, 534)
    ctx.curve_to(1194, 525, 1188, 520, 1182, 518)
    ctx.curve_to(1186, 516, 1190, 516, 1192, 516)
    ctx.close_path()
    ctx.fill()

    # Soft curved bangs resting gently across her upper forehead (no upward horn!)
    ctx.set_source_rgba(0.068, 0.160, 0.250, 0.88)
    ctx.move_to(1198, 517)
    ctx.curve_to(1188, 514, 1179, 515, 1174, 519)
    ctx.curve_to(1182, 519, 1190, 521, 1196, 526)
    ctx.close_path()
    ctx.fill()
    # Soft side lock curving gently past ear & jaw
    ctx.set_source_rgba(0.055, 0.130, 0.215, 0.92)
    ctx.move_to(1192, 536)
    ctx.curve_to(1182, 548, 1180, 562, 1192, 576)
    ctx.curve_to(1188, 564, 1190, 550, 1200, 540)
    ctx.close_path()
    ctx.fill()
    ctx.restore()

    # =========================================================================
    # 5. SLENDER DRAPED PALE SUNDRESS — CLEAR FEMININE WAIST + AIRY SKIRT DRAPE
    #    Natural slim human figure (NOT a bloated round turtle shell!)
    # =========================================================================
    # Soft sheer trailing fabric veil floating gently above/behind thighs
    veil_g = cairo.LinearGradient(960, 562, 960, 668)
    veil_g.add_color_stop_rgba(0.0, 0.84, 0.96, 0.96, 0.56)
    veil_g.add_color_stop_rgba(0.55, 0.46, 0.72, 0.80, 0.48)
    veil_g.add_color_stop_rgba(1.0, 0.24, 0.48, 0.62, 0.50)
    ctx.set_source(veil_g)
    ctx.move_to(1078, 588)
    ctx.curve_to(1028, 598, 976, 566, 916, 568)
    ctx.curve_to(884, 570, 862, 588, 856, 616)
    ctx.curve_to(884, 644, 938, 664, 1008, 660)
    ctx.curve_to(1062, 656, 1102, 634, 1126, 608)
    ctx.close_path()
    ctx.fill()

    # Main sleeveless pale sundress hugging chest & concave dipping waist, billowing lightly at skirt hem!
    dress_g = cairo.LinearGradient(1026, 576, 1012, 668)
    dress_g.add_color_stop_rgba(0.0, *DRESS_HI)
    dress_g.add_color_stop_rgba(0.46, *DRESS_MID)
    dress_g.add_color_stop_rgba(1.0, *DRESS_SH)
    ctx.set_source(dress_g)
    ctx.move_to(1110, 584)                             # soft scooped neckline at collarbone
    ctx.curve_to(1092, 579, 1072, 585, 1054, 602)      # gentle feminine bust curve
    ctx.curve_to(1038, 616, 1020, 622, 1000, 614)      # clear slender DIPPING CONCAVE WAIST!
    ctx.curve_to(968, 594, 932, 578, 892, 584)         # airy loose skirt lifting lightly upward
    ctx.curve_to(870, 588, 854, 602, 850, 618)         # flutter wave 1
    ctx.curve_to(866, 624, 872, 634, 864, 646)         # delicate hem wave indentation
    ctx.curve_to(882, 658, 912, 658, 944, 654)         # trailing lower skirt edge
    ctx.curve_to(988, 664, 1036, 664, 1076, 646)       # relaxed concave back/waist underside
    ctx.curve_to(1106, 632, 1122, 618, 1130, 604)      # back shoulder
    ctx.curve_to(1122, 600, 1114, 593, 1110, 584)      # gentle curved sundress neckline (no stiff flat cut!)
    ctx.close_path()
    ctx.fill_preserve()
    ctx.set_source_rgba(*INK)
    ctx.set_line_width(1.35)
    ctx.stroke()

    # Subtle cyan cel shadow pleats along slender waist & fluttering skirt
    for fpts in (
        [(1072, 626), (1024, 638), (962, 632), (888, 638), (924, 652), (1022, 658), (1078, 640)],
        [(1018, 618), (968, 610), (918, 602), (866, 616), (912, 624), (970, 624), (1024, 628)],
    ):
        ctx.set_source_rgba(0.30, 0.54, 0.66, 0.36)
        ctx.move_to(*fpts[0])
        for p in fpts[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.fill()

    # Delicate tapered skirt fold lines
    for crease in (
        [(1026, 620), (968, 610), (912, 600), (862, 614)],
        [(1016, 632), (964, 634), (904, 632), (870, 644)],
    ):
        stroke_catmull_rom(ctx, crease, base_w=1.15, z_depth=4.0, rgba=(0.08, 0.22, 0.34, 0.55), rng=rng, taper=True, tremor=0.18)

    # Sunlit warm-cream rim along bust, dipping waist & billowing skirt crest
    ctx.set_source_rgba(0.98, 1.0, 0.97, 0.85)
    ctx.set_line_width(1.65)
    ctx.move_to(1114, 584)
    ctx.curve_to(1092, 578, 1072, 586, 1052, 603)
    ctx.curve_to(1036, 616, 1018, 621, 998, 613)
    ctx.curve_to(966, 594, 930, 578, 892, 585)
    ctx.stroke()

    # =========================================================================
    # 6. FRONT ARM (RIGHT ARM) — SOFT RELAXED S-CURVE DRIFTING UP-LEFT
    #    Clean bare shoulder/sleeveless drape (NO weird oval shoulder balloon!)
    # =========================================================================
    g_near_arm = cairo.LinearGradient(996, 488, 1106, 618)
    g_near_arm.add_color_stop_rgba(0.0, *SKIN_HI)
    g_near_arm.add_color_stop_rgba(0.56, *SKIN_MID)
    g_near_arm.add_color_stop_rgba(1.0, *SKIN_SH)
    ctx.set_source(g_near_arm)
    ctx.move_to(1112, 598)                             # natural slender shoulder
    ctx.curve_to(1088, 574, 1062, 558, 1040, 540)      # graceful upper arm drifting up-left
    ctx.curve_to(1022, 524, 1012, 502, 1004, 484)      # relaxed bent forearm rising softly
    # Graceful drooping side-profile relaxed hand & tapered fingers!
    ctx.curve_to(1002, 472, 994, 462, 986, 460)        # arched wrist -> slender curving index/middle
    ctx.curve_to(981, 460, 982, 466, 988, 470)
    ctx.curve_to(983, 471, 982, 476, 988, 479)         # gently trailing soft ring/pinky
    ctx.curve_to(992, 484, 996, 488, 998, 494)         # slender inner wrist
    ctx.curve_to(1004, 514, 1014, 536, 1030, 554)      # soft inner elbow curve
    ctx.curve_to(1048, 574, 1072, 596, 1094, 614)      # inner upper arm -> smooth underarm
    ctx.curve_to(1106, 616, 1116, 608, 1112, 598)      # soft rounded shoulder cap
    ctx.close_path()
    ctx.fill_preserve()
    ctx.set_source_rgba(*INK)
    ctx.set_line_width(1.30)
    ctx.stroke()

    # Delicate thin sundress shoulder strap & sunlit rim along upper arm
    ctx.set_source_rgba(0.95, 0.99, 0.98, 0.92)
    ctx.set_line_width(3.4)
    ctx.move_to(1114, 584)
    ctx.curve_to(1112, 594, 1106, 606, 1096, 616)
    ctx.stroke()

    ctx.set_source_rgba(0.98, 1.0, 0.96, 0.70)
    ctx.set_line_width(1.30)
    ctx.move_to(1106, 593)
    ctx.curve_to(1086, 574, 1062, 558, 1040, 540)
    ctx.curve_to(1022, 524, 1012, 502, 1004, 485)
    ctx.stroke()

    ctx.restore()


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


def paint_foreground_bokeh_ruin(rng: np.random.Generator) -> np.ndarray:
    """Close out-of-focus broken concrete & bent rebar framing the lower-right corner."""
    surf, ctx = new_cairo_2x()

    # Jagged broken concrete slab mass rising from lower-right corner
    slab = [
        (W + 30, H + 30),
        (W - 510, H + 30),
        (W - 465, H - 68),
        (W - 395, H - 122),
        (W - 332, H - 140),
        (W - 285, H - 205),
        (W - 210, H - 228),
        (W - 148, H - 295),
        (W - 65, H - 325),
        (W + 30, H - 365),
    ]
    ctx.set_source_rgba(0.010, 0.032, 0.068, 0.97)
    ctx.move_to(*slab[0])
    for p in slab[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    ctx.fill()

    # Subtle cyan backlight grazing the fractured upper silhouette edge
    ctx.set_source_rgba(0.14, 0.36, 0.44, 0.55)
    ctx.set_line_width(4.0)
    ctx.move_to(*slab[2])
    for p in slab[3:-1]:
        ctx.line_to(*p)
    ctx.stroke()

    # Protruding twisted steel rebar rods reaching softly into the lower-right frame
    rebars = [
        [(W - 365, H - 120), (W - 435, H - 205), (W - 498, H - 245)],
        [(W - 265, H - 205), (W - 315, H - 310), (W - 348, H - 385)],
        [(W - 185, H - 250), (W - 238, H - 348), (W - 215, H - 438)],
        [(W - 108, H - 310), (W - 178, H - 392), (W - 242, H - 425)],
    ]
    for rpts in rebars:
        stroke_catmull_rom(ctx, rpts, base_w=8.5, z_depth=1.0, rgba=(0.010, 0.030, 0.064, 0.96), rng=rng, taper=False, tremor=0.5)

    fg_1x = finish_cairo_2x(surf)
    blurred = convolve_fft_2d(fg_1x, make_disc_kernel(radius=7.8))
    # Separate ~1 px opposite red/blue lateral chromatic shift
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
    tint = np.array([1.02, 1.06, 0.94], dtype=np.float32)  # warm cream-seafoam 90s anime bloom
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
    ca_scale = 0.0011
    img[:, :, 0] = remap_bilinear(img[:, :, 0], xx + nx * r2 * (W * ca_scale), yy + ny * r2 * (H * ca_scale))
    img[:, :, 2] = remap_bilinear(img[:, :, 2], xx - nx * r2 * (W * ca_scale), yy - ny * r2 * (H * ca_scale))
    img *= np.clip(1.0 - 0.22 * (r2 ** 1.15), 0.70, 1.0)[..., None]

    # 3. Linear shadow black lift + C1-continuous tanh highlight shoulder (knee=0.80, ceiling=0.965)
    img = 0.014 + img * 0.986
    knee, ceiling = 0.80, 0.965
    span = ceiling - knee
    img = np.where(img > knee, knee + span * np.tanh((img - knee) / span), img)

    # 4. Luminance-weighted silver-halide grain
    lum2 = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    env = np.clip(1.12 - 0.95 * np.power(np.clip(lum2, 0.0, 1.0), 1.55), 0.12, 1.12)
    grain = rng.normal(0.0, 0.011, size=(H, W, 3)).astype(np.float32)
    mono_g = rng.normal(0.0, 0.009, size=(H, W, 1)).astype(np.float32)
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

    line_shift = gaussian_blur_1d(rng.normal(0.0, 0.18, size=(H, 1)).astype(np.float32), 1.2, axis=0)
    jx = np.broadcast_to(np.arange(W, dtype=np.float32)[None, :], (H, W)) + line_shift
    jy = np.broadcast_to(np.arange(H, dtype=np.float32)[:, None], (H, W))
    out = remap_bilinear(out, jx, jy)
    out *= (1.0 - 0.007 * (0.5 - 0.5 * np.cos(np.arange(H, dtype=np.float32) * math.pi)))[:, None, None]
    return np.clip(out, 0.0, 1.0)


# =============================================================================
# Full Scene Assembly & Deterministic Export
# =============================================================================

def render_scene(
    seed: int = 199507,
    out_dir: pathlib.Path | None = None,
    web_webp: pathlib.Path | None = None,
    with_character: bool = False,
) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # Stage 1: Off-axis Camera3D looking slightly upward through deep water
    # target.x < eye.x shifts +Y vanishing point to ~59% width; target.z > eye.z tilts upward toward sunlit surface
    cam = Camera3D(W, H, eye=(0.35, 0.0, 2.20), target=(-1.05, 12.0, 3.72), fov_x_deg=58.0)
    print(f"Camera VP: ({cam.vp_x:.1f}, {cam.vp_y:.1f}) -> {cam.vp_x / W * 100:.1f}% width")

    # Stage 1 & 2: Deep underwater gradient + sunlit rippling water surface overhead
    canvas, fog_rgb = paint_underwater_backdrop_and_surface(cam, rng)

    # Planar submerged Voronoi concrete floor & fractured left ruin wall
    floor_uv = paint_voronoi_concrete_uv(rng, (-6.2, 6.2), (3.5, 24.0), is_floor=True)
    wall_uv = paint_voronoi_concrete_uv(rng, (3.8, 22.0), (Z_FLOOR, 5.1), is_floor=False)

    canvas = cam.warp_plane_over(
        canvas, floor_uv,
        p0=np.array([-6.2, 3.5, Z_FLOOR]), e1=np.array([12.4, 0.0, 0.0]), e2=np.array([0.0, 20.5, 0.0]),
        fog_rgb=fog_rgb, fog_start=4.0, fog_density=0.095,
    )
    canvas = cam.warp_plane_over(
        canvas, wall_uv,
        p0=np.array([-5.2, 3.8, Z_FLOOR]), e1=np.array([0.0, 18.2, 0.0]), e2=np.array([0.0, 0.0, 5.1 - Z_FLOOR]),
        fog_rgb=fog_rgb, fog_start=4.0, fog_density=0.088,
    )

    # Stage 3 Band 0 (Deep Abyssal Far Ruins, y = 14..22 m)
    s_far, c_far = new_cairo_2x()
    draw_sunken_archway(c_far, cam, -5.2, -0.4, 18.5, 2.0, 4.7, 0.9, rng, fog_rgb)
    draw_tilted_pillar(c_far, cam, np.array([-3.4, 17.5, Z_FLOOR]), 0.42, 6.8, 0.08, -0.04, True, rng, fog_rgb)
    draw_tilted_pillar(c_far, cam, np.array([3.9, 16.8, Z_FLOOR]), 0.38, 6.2, -0.11, 0.03, True, rng, fog_rgb)
    draw_drifting_rubble(c_far, cam, 18, (12.0, 21.0), rng, fog_rgb)
    canvas = over_premul(canvas, finish_cairo_2x(s_far))

    # Back volumetric diagonal god rays aligned with SUN_DIR
    rays_back = paint_volumetric_god_rays(cam, rng, intensity=0.58, front_pass=False)
    canvas = 1.0 - (1.0 - np.clip(canvas, 0.0, 1.0)) * (1.0 - np.clip(rays_back, 0.0, 0.9))

    # Stage 3 Band 1 (Midground Asymmetric Sunken Ruins — heavy mass on left!)
    s_mid, c_mid = new_cairo_2x()
    draw_sunken_archway(c_mid, cam, -5.2, -1.65, 9.4, 2.35, 4.95, 0.85, rng, fog_rgb)
    draw_tilted_pillar(c_mid, cam, np.array([-4.05, 11.8, Z_FLOOR]), 0.44, 6.6, -0.06, 0.05, True, rng, fog_rgb)
    draw_tilted_pillar(c_mid, cam, np.array([-2.35, 8.8, Z_FLOOR]), 0.40, 6.2, 0.09, -0.03, True, rng, fog_rgb)
    draw_drifting_rubble(c_mid, cam, 18, (6.5, 12.5), rng, fog_rgb)
    draw_bubble_streams(c_mid, rng, hero_near=False)
    mid_1x = finish_cairo_2x(s_mid)
    # Break up pillar/arch flat fills with gouache mottling + soft caustic shimmer!
    m_mot = fbm_gouache(H, W, rng)
    m_dabs = flat_brush_dabs(H, W, rng, count=82)
    ma = mid_1x[:, :, 3:4]
    mid_1x[:, :, :3] = np.clip(mid_1x[:, :, :3] + ma * (m_mot * 0.022 + m_dabs * 0.014)[..., None], 0.0, 1.2)
    canvas = over_premul(canvas, mid_1x)

    # Stage 3 Band 2 (Optional Hero Character — disabled by default for pure architectural ruins)
    if with_character:
        s_hero, c_hero = new_cairo_2x()
        draw_drifting_anime_girl(c_hero, rng)
        hero_1x = finish_cairo_2x(s_hero)
        # Break up flat cel fills subtly with isotropic gouache mottling + cold-press paper tooth!
        h_mottle = fbm_gouache(H, W, rng, scales=((64, 60), (24, 26), (10, 9), (4.5, 4.5)))
        h_tooth = paper_tooth(H, W, rng)
        ha = hero_1x[:, :, 3:4]
        hero_1x[:, :, :3] = np.clip(hero_1x[:, :, :3] + ha * (h_mottle * 0.016 + h_tooth * 0.008)[..., None], 0.0, 1.2)
        canvas = over_premul(canvas, hero_1x)

    # Front volumetric god rays piercing down through the sunken atrium & stonework!
    rays_front = paint_volumetric_god_rays(cam, rng, intensity=0.40, front_pass=True)
    canvas = 1.0 - (1.0 - np.clip(canvas, 0.0, 1.0)) * (1.0 - np.clip(rays_front, 0.0, 0.85))

    # Stage 3 Band 3 (Near Left Ruined Pillar, Rising Ambient Bubbles & Suspended Motes)
    s_near, c_near = new_cairo_2x()
    draw_tilted_pillar(c_near, cam, np.array([-2.68, 4.85, Z_FLOOR]), 0.46, 5.9, -0.10, 0.04, True, rng, fog_rgb)
    draw_drifting_rubble(c_near, cam, 12, (3.2, 6.2), rng, fog_rgb)
    draw_bubble_streams(c_near, rng, hero_near=with_character)
    near_1x = finish_cairo_2x(s_near)
    na = near_1x[:, :, 3:4]
    near_1x[:, :, :3] = np.clip(near_1x[:, :, :3] + na * (m_mot * 0.020 + m_dabs * 0.012)[..., None], 0.0, 1.2)
    canvas = over_premul(canvas, near_1x)

    # Stage 4: Lower-right foreground broken concrete & rebar circular disc bokeh + 35mm finish
    fg_bokeh = paint_foreground_bokeh_ruin(rng)
    canvas = over_premul(canvas, fg_bokeh)

    final_rgb = apply_35mm_finish(canvas, rng)
    elapsed = time.perf_counter() - t0
    print(f"Rendered {W}x{H} in {elapsed:.2f}s")

    u8 = np.round(np.clip(final_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    full_im = Image.fromarray(u8, mode="RGB")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        full_im.save(out_dir / "mira-underwater-full.png", optimize=True)
        crops = {
            "crop_aperture.png": (int(W * 0.42), 0),                      # Sunlit rippling surface + god rays
            "crop_midground.png": (int(W * 0.36), int(H * 0.34)),         # Hero drifting girl 1:1 macro audit
            "crop_foreground.png": (W - 600, H - 450),                    # Lower-right disc bokeh concrete/rebar
        }
        for cname, (cx0, cy0) in crops.items():
            x0 = max(0, min(W - 600, cx0))
            y0 = max(0, min(H - 450, cy0))
            full_im.crop((x0, y0, x0 + 600, y0 + 450)).save(out_dir / cname)

    if web_webp is not None:
        web_webp.parent.mkdir(parents=True, exist_ok=True)
        # Clean 1280x720 source asset for the live ordered-Bayer pixel-motion engine in docs/
        im_web = full_im.resize((1280, 720), Image.Resampling.LANCZOS)
        im_web.save(web_webp, format="WEBP", quality=90, method=6)
        print(f"Saved web painting source: {web_webp} ({web_webp.stat().st_size / 1024:.1f} KiB)")

    return final_rgb, {"elapsed": elapsed, "vp_ratio": cam.vp_x / W}


def main() -> None:
    ap = argparse.ArgumentParser(description="Render Mira 2048x1152 1990s underwater anime scene")
    ap.add_argument("--seed", type=int, default=199507)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--web-webp", type=pathlib.Path, default=None)
    ap.add_argument("--with-character", action="store_true", help="Include the drifting underwater character")
    args = ap.parse_args()
    render_scene(seed=args.seed, out_dir=args.out_dir, web_webp=args.web_webp, with_character=args.with_character)


if __name__ == "__main__":
    main()

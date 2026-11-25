#!/usr/bin/env python3
"""
Matcha 01 — "Arashiyama Tea Veranda" (2048 x 1152)

Pure-Python 3D-grounded 5-stage painting pipeline (zero image models, zero external assets):
  Stage 1: Metric right-handed Camera3D (off-axis vanishing point ~61.4% width),
           exact 3x3 UV-to-screen plane homographies masked by UV bounds and
           ray-plane depth z_cam > 0.05 with exponential misty distance fog,
           3D near-plane segment clipping, and vectorized 3x3 Voronoi (d2 - d1)
           mossy Kyoto garden stones.
  Stage 2: Single normalized diagonal morning SUN_DIR (SUN_DIR[2] < 0) traced
           backward with 6 height-refinement steps onto the pitched slatted cedar
           eaves & veranda pillars so sunlit bars on timber planks, stone shadows,
           volumetric sunbeams, and bench shadows share exact 3D alignment; plus
           mirror-calm pond Fresnel reflections combining capillary-rippled
           horizon reflections with 3D plane-mirrored quads.
  Stage 3: 4 depth-interleaved 2x supersampled PyCairo bands (Lanczos-downsampled)
           featuring misty Arashiyama cedar/pine mountain ridges, lush mossy pond
           shoreline with snow-viewing stone lantern (yukimi-doro), per-seed
           arching bamboo stalks with dense feathery Bezier leaf cascades, cedar
           veranda joinery, and a hero low cedar tea bench with handmade ceramic
           matcha bowl (chawan), whisk (chasen), tea scoop (chashaku), and soft
           misty steam wisps, enriched by isotropic fbm_gouache, flat-brush dabs,
           cold-press paper tooth, and warm shadow bounce.
  Stage 4: Upper-right foreground Kyoto maple (Acer palmatum) branch convolved in
           premultiplied RGBA with a circular aperture disc kernel (r = 7.8,
           spherical-aberration rim) + 1 px opposite R/B lateral shift, followed
           by 35mm multi-scale halation, radial CA, optical vignette, shadow
           black lift + C1 tanh highlight shoulder, luminance-weighted silver-
           halide grain, and faint YIQ analog transfer patina.
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

# Stage 2: Single locked normalized 3D morning light travel vector (upper-right -> lower-left)
_raw_sun = np.array([-0.54, -0.28, -0.794], dtype=np.float64)
SUN_DIR = _raw_sun / np.linalg.norm(_raw_sun)

# World architectural constants (meters)
Z_POND = -0.02
Z_GARDEN = 0.02
Z_DECK = 0.44
X_WALL = -3.45
X_VERANDA_EDGE = -0.65

PILLAR_YS = (5.2, 8.6, 12.0, 15.4)
SLAT_PITCH = 0.56   # natural architectural cedar rafter spacing (meters)
SLAT_OPEN = 0.62    # warm morning sunlight aperture fraction


def eave_height_at_x(wx: np.ndarray | float) -> np.ndarray | float:
    """Pitched cedar eave plane sloping gently downward toward the garden."""
    return 3.08 - 0.165 * (wx - X_WALL)


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


def apply_shadow_bounce(rgb: np.ndarray, bounce_rgb=(0.022, 0.026, 0.014)) -> np.ndarray:
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
        fog_start: float = 5.5,
        fog_density: float = 0.055,
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
# Stage 2 — Unified 3D Morning Sunlight Solver (6-Step Pitched Slatted Eaves)
# =============================================================================

def trace_veranda_sunlight(wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> np.ndarray:
    """
    Trace any 3D deck/garden/wall point backward along -SUN_DIR onto the pitched
    slatted cedar eaves (6 fixed height-refinement steps), veranda pillars, and
    overhead bamboo/maple foliage dappling so light & shadow align in 3D space.
    """
    inv_sz = 1.0 / (-SUN_DIR[2])
    z_roof = np.full_like(wx, 2.75, dtype=np.float64)
    for _ in range(6):
        t = np.maximum(0.0, (z_roof - wz) * inv_sz)
        sx = wx - SUN_DIR[0] * t
        z_roof = eave_height_at_x(sx)

    t = np.maximum(0.0, (z_roof - wz) * inv_sz)
    sx = wx - SUN_DIR[0] * t
    sy = wy - SUN_DIR[1] * t

    # Solid inner roof covers x <= -1.35; open slatted cedar eave pergola covers [-1.35, +0.42]
    solid_roof = (sx <= -1.35) & (sy >= 0.2) & (sy <= 16.5)
    in_slats = (sx > -1.35) & (sx <= 0.42) & (sy >= 0.2) & (sy <= 16.5)
    phase = np.mod(sy / SLAT_PITCH, 1.0)
    # Soft natural morning penumbra edges on slatted cedar beams (not harsh binary stripes!)
    slat_open = np.clip((phase - 0.05) / 0.14, 0.0, 1.0) * np.clip((SLAT_OPEN - phase) / 0.14, 0.0, 1.0)
    # Cross purlin beam at x ~ -0.46 + heavier tie-beams above pillars
    purlin = np.exp(-((sx + 0.46) / 0.085) ** 2)
    slat_trans = np.clip(slat_open * (1.0 - 0.80 * purlin), 0.0, 1.0)

    # Organic maple/bamboo foliage dappling softly breaking up the slat light bars!
    fol_dapple = (
        0.58
        + 0.24 * np.sin(sx * 1.65 + sy * 0.72 + 0.4)
        + 0.18 * np.cos(sx * 2.85 - sy * 1.35 + 1.1)
        + 0.12 * np.sin(sy * 0.34 - sx * 0.9)
    )
    fol_mod = np.clip(fol_dapple, 0.36, 1.08)

    sun = np.where(solid_roof, 0.0, np.where(in_slats, slat_trans * fol_mod, np.clip(fol_mod * 0.95, 0.48, 1.0))).astype(np.float32)

    # Soft veranda pillar cast shadows along SUN_DIR
    px_c = X_VERANDA_EDGE - 0.06
    for py_c in PILLAR_YS:
        t_col = (wx - px_c) / SUN_DIR[0]
        hit_z = wz - t_col * SUN_DIR[2]
        hit_y = wy - t_col * SUN_DIR[1]
        in_h = (t_col > 0.0) & (hit_z >= Z_DECK) & (hit_z <= eave_height_at_x(px_c))
        d_col = np.abs(hit_y - py_c)
        col_mask = np.where(in_h, np.clip((d_col - 0.075) / 0.075, 0.0, 1.0), 1.0)
        sun *= col_mask.astype(np.float32)

    return np.clip(sun, 0.0, 1.0)


def mirror_pts_across_plane(pts: list[np.ndarray], z_plane: float = Z_POND) -> list[np.ndarray]:
    """Exact 3D plane mirror: pts - 2 * dot(pts - p0, n) * n for horizontal pond plane z = z_plane."""
    p0 = np.array([0.0, 0.0, z_plane], dtype=np.float64)
    n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return [p - 2.0 * float(np.dot(p - p0, n)) * n for p in pts]


# =============================================================================
# Stage 1 & 3 — Misty Arashiyama Sky, Painterly Pine Ridges & Planar UVs
# =============================================================================

def paint_sky_and_mountain_ridges(cam: Camera3D, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Paint warm morning olive-cream sky and three painterly misty Arashiyama cedar/pine ridges."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = xx / W
    v = yy / H

    mottle = fbm_gouache(H, W, rng)
    dabs = flat_brush_dabs(H, W, rng, count=92)
    tooth = paper_tooth(H, W, rng)

    # Matcha morning sky: muted Kyoto sage zenith -> soft olive-cream -> glowing morning sun mist
    c_zenith = np.array([0.28, 0.40, 0.35], dtype=np.float32)
    c_mid = np.array([0.64, 0.75, 0.62], dtype=np.float32)
    c_horiz = np.array([0.94, 0.94, 0.82], dtype=np.float32)

    hv = float(cam.horizon_y / H)
    t_sky = np.clip(v / max(0.25, hv + 0.06), 0.0, 1.0)
    t1 = np.clip(t_sky / 0.56, 0.0, 1.0)
    t2 = np.clip((t_sky - 0.38) / 0.62, 0.0, 1.0)

    sky = (
        c_zenith[None, None, :] * (1.0 - t1[..., None])
        + c_mid[None, None, :] * (t1 * (1.0 - t2))[..., None]
        + c_horiz[None, None, :] * (t1 * t2)[..., None]
    )

    # Morning sun glow in upper-right sky aligned with -SUN_DIR
    du = u - 0.74
    dv = v - 0.22
    sun_halo = np.exp(-(du * du * 4.2 + dv * dv * 11.5)).astype(np.float32)
    sun_core = np.exp(-(du * du * 22.0 + dv * dv * 56.0)).astype(np.float32)
    sky += sun_halo[..., None] * np.array([0.26, 0.26, 0.14], dtype=np.float32)
    sky += sun_core[..., None] * np.array([0.34, 0.32, 0.20], dtype=np.float32)

    # Soft morning mist clouds broken up by isotropic gouache
    cloud_n = fbm_gouache(H // 2, W // 2, rng, scales=((48, 44), (20, 18), (8, 7.5), (3.5, 3.5)))
    cloud_n = resize_bicubic(cloud_n, W, H)
    cloud_m = np.clip((cloud_n + 0.35) * 0.52, 0.0, 1.0) * np.exp(-((v - 0.24) / 0.15) ** 2)
    sky = sky * (1.0 - 0.22 * cloud_m[..., None]) + np.array([0.97, 0.96, 0.86], dtype=np.float32) * (0.22 * cloud_m[..., None])
    sky += (mottle * 0.015 + dabs * 0.011 + tooth * 0.005)[..., None]

    # Three painterly Arashiyama cedar/pine ridges with irregular organic conifer profiles (ZERO sine-wave teeth!)
    canvas = np.clip(sky, 0.0, 1.35)
    xs = np.linspace(0.0, 1.0, W, dtype=np.float32)
    ridge_specs = [
        (0.30, 0.062, [0.34, 0.45, 0.38], [0.60, 0.69, 0.57], [0.84, 0.87, 0.75], 0.010, 52),
        (0.37, 0.052, [0.17, 0.28, 0.22], [0.38, 0.49, 0.39], [0.70, 0.77, 0.64], 0.016, 78),
        (0.43, 0.042, [0.078, 0.165, 0.118], [0.16, 0.27, 0.19], [0.46, 0.56, 0.43], 0.022, 110),
    ]
    for idx, (bv, amp, c_top, c_bot, c_mist, t_amp, n_trees) in enumerate(ridge_specs):
        # Smooth organic mountain contour from low-freq bicubic noise + gentle valley saddle
        macro = resize_bicubic(rng.standard_normal((4, 9)).astype(np.float32), W, 4)[2]
        saddle = 0.032 * np.exp(-((xs - 0.62) / 0.24) ** 2)  # calm Arashiyama river valley dip!
        ridge_v = bv + saddle - amp * (0.35 * macro + 0.45 * np.sin(xs * (3.2 + idx) + idx * 1.7))

        # Irregular seeded conifer cedar/pine tree crowns stamped along the ridge (varied height & width!)
        crown_prof = np.zeros(W, dtype=np.float32)
        for _ in range(n_trees):
            tc = float(rng.uniform(-0.02, 1.02))
            tw = float(rng.uniform(0.004, 0.014)) * (1.2 - 0.25 * idx)
            th = t_amp * float(rng.uniform(0.45, 1.45))
            # Pointed cedar spire or rounded Kyoto pine umbrella crown
            if rng.random() < 0.62:
                shape = np.maximum(0.0, 1.0 - np.abs(xs - tc) / tw) ** 1.35
            else:
                shape = np.exp(-((xs - tc) / (tw * 1.15)) ** 2)
            crown_prof = np.maximum(crown_prof, th * shape)

        line_v = ridge_v - crown_prof
        mask = np.clip((v - line_v[None, :]) * H / 2.4, 0.0, 1.0)
        rel_y = np.clip((v - line_v[None, :]) / np.maximum(0.06, hv + 0.08 - line_v[None, :]), 0.0, 1.0)
        col_top = np.array(c_top, dtype=np.float32)
        col_bot = np.array(c_bot, dtype=np.float32)
        col_mist = np.array(c_mist, dtype=np.float32)
        r_rgb = col_top[None, None, :] * (1.0 - rel_y[..., None]) + col_bot[None, None, :] * rel_y[..., None]
        # Warm golden-olive morning sunlight grazing treetop crowns facing upper-right sun
        sun_rim = np.exp(-((v - line_v[None, :]) / 0.018) ** 2) * np.clip(0.35 + 0.85 * u, 0.0, 1.0)
        r_rgb += sun_rim[..., None] * np.array([0.16, 0.19, 0.08], dtype=np.float32) * (0.65 if idx < 2 else 1.0)
        # Valley mist veil pooling gracefully in the lower hollows between ridges
        mist_band = np.power(rel_y, 1.65) * (0.70 - 0.14 * idx) * (0.72 + 0.28 * np.clip(mottle, -1.0, 1.0))
        r_rgb = r_rgb * (1.0 - mist_band[..., None]) + col_mist[None, None, :] * mist_band[..., None]
        r_rgb += (mottle * 0.018 + dabs * 0.014 + tooth * 0.006)[..., None]
        canvas = canvas * (1.0 - mask[..., None]) + r_rgb * mask[..., None]

    # Soft morning mist ribbon hovering right along the distant shoreline
    mist_rib = np.exp(-((v - (hv + 0.012)) / 0.052) ** 2) * (0.42 + 0.24 * np.clip(mottle, -1.0, 1.0))
    canvas = canvas * (1.0 - mist_rib[..., None]) + np.array([0.85, 0.88, 0.76], dtype=np.float32) * mist_rib[..., None]

    fog_rgb = np.array([0.66, 0.73, 0.60], dtype=np.float32)
    return np.clip(canvas, 0.0, 1.35), fog_rgb, mottle, dabs


def paint_shoji_wall_uv(rng: np.random.Generator, tw: int = 1024, th: int = 512) -> np.ndarray:
    """Left teahouse wall: aged cedar wainscoting, sliding wooden sill, glowing washi shoji screens."""
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wy = 0.5 + u * 15.5
    wz = Z_DECK + v * (3.12 - Z_DECK)

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=84)
    tooth = paper_tooth(th, tw, rng)

    # Warm translucent parchment washi paper with subtle vertical warm interior bounce
    paper = np.zeros((th, tw, 3), dtype=np.float32)
    paper[:, :, 0] = 0.44 + 0.09 * (1.0 - v) + mottle * 0.024 + dabs * 0.018
    paper[:, :, 1] = 0.46 + 0.09 * (1.0 - v) + mottle * 0.025 + dabs * 0.018
    paper[:, :, 2] = 0.36 + 0.06 * (1.0 - v) + mottle * 0.018 + dabs * 0.014

    # Faint dreamy outside maple/bamboo shadow silhouette projected onto washi paper!
    leaf_sh = np.clip(np.sin(wy * 1.15 + wz * 2.2 + mottle * 2.2) * np.cos(wy * 0.72 - wz * 1.6), 0.0, 1.0)
    paper -= leaf_sh[..., None] * np.array([0.055, 0.050, 0.045], dtype=np.float32)

    # Delicate kumiko wooden shoji lattice muntins
    cedar_dark = np.array([0.105, 0.078, 0.058], dtype=np.float32)
    cedar_mid = np.array([0.185, 0.135, 0.092], dtype=np.float32)
    v_grid = np.abs(np.mod(wy, 0.34) - 0.17) > 0.156
    h_grid = np.abs(np.mod(wz - Z_DECK, 0.24) - 0.12) > 0.108
    frame_v = np.abs(np.mod(wy, 1.70) - 0.85) > 0.79
    lattice = v_grid | h_grid

    rgb = np.where(lattice[..., None], cedar_mid[None, None, :], paper)
    rgb = np.where(frame_v[..., None], cedar_dark[None, None, :], rgb)

    # Lower cedar wainscoting (koshi-ita) + sliding sill track (shikii)
    wainscot = wz < 1.02
    sill = (wz >= 1.00) & (wz <= 1.05)
    plank_seam = np.abs(np.mod(wy, 0.22) - 0.11) > 0.102
    wood_col = np.zeros((th, tw, 3), dtype=np.float32)
    wood_col[:, :, 0] = 0.135 + mottle * 0.022 + dabs * 0.016
    wood_col[:, :, 1] = 0.102 + mottle * 0.018 + dabs * 0.013
    wood_col[:, :, 2] = 0.074 + mottle * 0.014 + dabs * 0.010
    wood_col = np.where(plank_seam[..., None], wood_col * 0.56, wood_col)
    rgb = np.where(wainscot[..., None], wood_col, rgb)
    rgb = np.where(sill[..., None], cedar_dark[None, None, :] * 1.15, rgb)

    # Upper Kyoto clay-plaster transom (juraku-kabe) & dark lintel beam (kamoi)
    lintel = (wz >= 2.42) & (wz <= 2.54)
    plaster = wz > 2.54
    plaster_c = np.array([0.19, 0.23, 0.17], dtype=np.float32)[None, None, :] + (mottle * 0.022 + dabs * 0.016)[..., None]
    rgb = np.where(plaster[..., None], plaster_c, rgb)
    rgb = np.where(lintel[..., None], cedar_dark[None, None, :], rgb)

    sun = trace_veranda_sunlight(np.full_like(wy, X_WALL + 0.05, dtype=np.float64), wy.astype(np.float64), wz.astype(np.float64))
    rgb += sun[..., None] * np.array([0.24, 0.25, 0.15], dtype=np.float32)
    rgb = apply_shadow_bounce(rgb)
    rgb += tooth[..., None] * 0.007

    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.2)
    return rgba


def paint_veranda_deck_uv(rng: np.random.Generator, tw: int = 896, th: int = 1280) -> np.ndarray:
    """
    Worn Kyoto cedar engawa deck planks with rain-damp outer edge sheen and
    exact 6-step -SUN_DIR slatted cedar eave + foliage-dappled + bench shadows.
    """
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wx = X_WALL + u * (X_VERANDA_EDGE - X_WALL)
    wy = 0.5 + v * 15.5
    wz = np.full_like(wx, Z_DECK)

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=88)
    tooth = paper_tooth(th, tw, rng)

    plank_w = 0.215
    p_idx = np.floor((wx - X_WALL) / plank_w).astype(np.int32)
    p_frac = np.mod((wx - X_WALL) / plank_w, 1.0)
    tone_tab = rng.uniform(-0.045, 0.045, size=64)
    p_tone = tone_tab[p_idx & 63].astype(np.float32)

    seam_x = (p_frac < 0.030) | (p_frac > 0.970)
    butt_off = (p_idx * 1.73) % 3.8
    butt_y = np.abs(np.mod(wy + butt_off, 3.8) - 1.9) > 1.885
    seam = seam_x | butt_y

    grain_wave = np.sin(wx * 38.0 + wy * 0.65 + mottle * 3.8).astype(np.float32) * 0.5 + 0.5

    rgb = np.zeros((th, tw, 3), dtype=np.float32)
    rgb[:, :, 0] = 0.205 + p_tone * 0.55 + grain_wave * 0.028 + mottle * 0.022 + dabs * 0.016
    rgb[:, :, 1] = 0.160 + p_tone * 0.45 + grain_wave * 0.020 + mottle * 0.020 + dabs * 0.014
    rgb[:, :, 2] = 0.115 + p_tone * 0.34 + grain_wave * 0.014 + mottle * 0.015 + dabs * 0.011

    outer_damp = np.clip((wx - (X_VERANDA_EDGE - 0.58)) / 0.58, 0.0, 1.0).astype(np.float32)
    moss_edge = np.clip((outer_damp - 0.35) * 1.4 + mottle * 0.28, 0.0, 1.0)
    rgb = rgb * (1.0 - 0.28 * moss_edge[..., None]) + np.array([0.13, 0.22, 0.12], dtype=np.float32) * (0.28 * moss_edge[..., None])
    rgb = np.where(seam[..., None], rgb * 0.34, rgb)

    # Stage 2: Exact 6-step pitched slatted eave + foliage dappling + pillar shadows
    sun = trace_veranda_sunlight(wx, wy, wz)

    # Soft cast shadow of hero low cedar tea bench at (-1.38, 4.25), top z = 0.74
    dz_bench = 0.30
    t_b = dz_bench / (-SUN_DIR[2])
    bx = wx - SUN_DIR[0] * t_b
    by = wy - SUN_DIR[1] * t_b
    b_sh_x = np.clip((0.34 - np.abs(bx + 1.38)) / 0.07, 0.0, 1.0)
    b_sh_y = np.clip((0.52 - np.abs(by - 4.25)) / 0.07, 0.0, 1.0)
    sun *= (1.0 - 0.84 * b_sh_x * b_sh_y).astype(np.float32)
    ao_b = np.exp(-(((wx + 1.38) / 0.38) ** 2 + ((wy - 4.25) / 0.58) ** 2)).astype(np.float32)
    rgb *= (1.0 - 0.48 * ao_b)[..., None]

    # Golden-cream (#f2f0d4 / #c5d988) morning sunlit bars across worn timber planks
    sun_col = np.array([0.52, 0.51, 0.31], dtype=np.float32)
    rgb += (sun * (1.0 - 0.65 * seam.astype(np.float32)))[..., None] * sun_col[None, None, :]

    # Post-rain wet sheen along outer rain-mist edge
    wet_puddle = np.clip((outer_damp * 0.90 + mottle * 0.25 - 0.20) * 1.5, 0.0, 1.0)
    sky_sheen = np.array([0.54, 0.65, 0.52], dtype=np.float32)
    rgb = rgb * (1.0 - 0.28 * wet_puddle[..., None]) + sky_sheen[None, None, :] * (0.28 * wet_puddle[..., None])
    wet_glint = wet_puddle * sun * np.clip(0.55 + 0.45 * grain_wave, 0.0, 1.0)
    rgb += wet_glint[..., None] * np.array([0.24, 0.25, 0.15], dtype=np.float32)

    rgb = apply_shadow_bounce(rgb)
    rgb += tooth[..., None] * 0.007

    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.3)
    return rgba


def paint_veranda_fascia_uv(rng: np.random.Generator, tw: int = 896, th: int = 128) -> np.ndarray:
    """Dark shadowed timber fascia & sub-deck recess between engawa edge and mossy garden."""
    mottle = fbm_gouache(th, tw, rng)
    tooth = paper_tooth(th, tw, rng)
    yy, _ = np.mgrid[0:th, 0:tw].astype(np.float32)
    v = yy / (th - 1)
    beam = v > 0.70
    rgb = np.zeros((th, tw, 3), dtype=np.float32)
    rgb[:, :, 0] = np.where(beam, 0.155, 0.038) + mottle * 0.014
    rgb[:, :, 1] = np.where(beam, 0.122, 0.052) + mottle * 0.015
    rgb[:, :, 2] = np.where(beam, 0.088, 0.038) + mottle * 0.011
    rgb = apply_shadow_bounce(rgb)
    rgb += tooth[..., None] * 0.005
    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.0)
    return rgba


def paint_slatted_eaves_uv(rng: np.random.Generator, tw: int = 768, th: int = 1024) -> np.ndarray:
    """Overhead pitched cedar eaves matching the exact SLAT_PITCH & SLAT_OPEN traced in Stage 2."""
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wx = (X_WALL - 0.1) + u * (0.45 - (X_WALL - 0.1))
    wy = 0.2 + v * 16.3

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=82)
    tooth = paper_tooth(th, tw, rng)

    solid = wx <= -1.35
    phase = np.mod(wy / SLAT_PITCH, 1.0)
    rafter_bar = (phase <= 0.08) | (phase >= SLAT_OPEN)
    purlin = (np.abs(wx + 0.46) < 0.072) | (wx > 0.34)
    # Heavier cedar tie-beams aligned with each veranda pillar
    tie_beam = np.zeros_like(solid)
    for py_c in PILLAR_YS:
        tie_beam |= np.abs(wy - py_c) < 0.085
    beam_mask = solid | rafter_bar | purlin | tie_beam

    rgb = np.zeros((th, tw, 3), dtype=np.float32)
    rgb[:, :, 0] = 0.105 + mottle * 0.018 + dabs * 0.013
    rgb[:, :, 1] = 0.084 + mottle * 0.016 + dabs * 0.011
    rgb[:, :, 2] = 0.062 + mottle * 0.012 + dabs * 0.009

    outer_rim = np.clip((wx + 1.35) / 1.80, 0.0, 1.0).astype(np.float32)
    edge_glow = np.exp(-((phase - SLAT_OPEN) / 0.045) ** 2).astype(np.float32)
    rgb += (outer_rim * (0.28 + 0.72 * edge_glow))[..., None] * np.array([0.18, 0.19, 0.09], dtype=np.float32)

    soffit_rib = (np.sin(wy * (math.tau / SLAT_PITCH)) * 0.5 + 0.5).astype(np.float32)
    rgb *= np.where(solid[..., None], 0.70 + 0.30 * soffit_rib[..., None], 1.0)
    rgb = apply_shadow_bounce(rgb)
    rgb += tooth[..., None] * 0.006

    alpha = gaussian_blur_2d(np.where(beam_mask, 1.0, 0.0).astype(np.float32), 0.75)
    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.2) * alpha[..., None]
    rgba[:, :, 3] = alpha
    return rgba


def paint_voronoi_mossy_stones_uv(rng: np.random.Generator, tw: int = 896, th: int = 1280) -> np.ndarray:
    """
    Stage 1 & 2: Irregular Voronoi (d2 - d1) mossy Kyoto garden stones with smooth
    organic shoreline & far-mist fade (zero sharp back-edge cutoff!).
    """
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)   # x in [-0.72, +2.85]
    v = yy / (th - 1)   # y in [1.0, 22.0]
    wx = -0.72 + u * (2.85 - (-0.72))
    wy = 1.0 + v * 21.0
    wz = np.full_like(wx, Z_GARDEN)

    nx = resize_bicubic(rng.standard_normal((th // 28, tw // 28)).astype(np.float32), tw, th) * 0.065
    ny = resize_bicubic(rng.standard_normal((th // 28, tw // 28)).astype(np.float32), tw, th) * 0.065
    px, py = wx + nx, wy + ny

    cell_x, cell_y = 0.54, 0.60
    ix = np.floor(px / cell_x).astype(np.int32)
    iy = np.floor(py / cell_y).astype(np.int32)

    jx_tab = rng.uniform(0.18, 0.82, size=256)
    jy_tab = rng.uniform(0.18, 0.82, size=256)
    tone_tab = rng.uniform(-0.05, 0.06, size=256)
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
    stone_mask = np.clip((edge_dist - 0.032) / 0.042, 0.0, 1.0)
    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=90)
    tooth = paper_tooth(th, tw, rng)

    sv = tone_tab[best_id].astype(np.float32)
    stone = np.zeros((th, tw, 3), dtype=np.float32)
    stone[:, :, 0] = 0.205 + sv * 0.45 + mottle * 0.025 + dabs * 0.018
    stone[:, :, 1] = 0.238 + sv * 0.48 + mottle * 0.028 + dabs * 0.020
    stone[:, :, 2] = 0.200 + sv * 0.42 + mottle * 0.022 + dabs * 0.016

    moss = np.zeros((th, tw, 3), dtype=np.float32)
    moss[:, :, 0] = 0.115 + mottle * 0.034 + dabs * 0.025
    moss[:, :, 1] = 0.228 + mottle * 0.052 + dabs * 0.036
    moss[:, :, 2] = 0.096 + mottle * 0.022 + dabs * 0.016

    # Organic curving pond shoreline that gently narrows toward the distant misty pond cove!
    shore_x = 1.66 + 0.42 * np.sin(wy * 0.40 + 0.4) + 0.24 * np.cos(wy * 0.92 - 0.7) + 0.18 * mottle - 0.06 * np.maximum(0.0, wy - 13.0)
    near_shore = np.clip((wx - (shore_x - 0.68)) / 0.68, 0.0, 1.0).astype(np.float32)
    cell_is_moss = (moss_tab[best_id] < (0.28 + 0.56 * near_shore)).astype(np.float32)
    effective_stone = stone_mask * (1.0 - 0.86 * cell_is_moss)

    crevice_dark = np.exp(-((edge_dist - 0.028) / 0.018) ** 2) * (1.0 - cell_is_moss)
    rgb = effective_stone[..., None] * stone + (1.0 - effective_stone[..., None]) * moss
    rgb *= (1.0 - 0.44 * crevice_dark)[..., None]

    sun = trace_veranda_sunlight(wx, wy, wz)
    sun_stone = np.array([0.40, 0.42, 0.26], dtype=np.float32)
    sun_moss = np.array([0.33, 0.44, 0.16], dtype=np.float32)
    rgb += sun[..., None] * (effective_stone[..., None] * sun_stone + (1.0 - effective_stone[..., None]) * sun_moss)

    wet = effective_stone * np.clip(0.45 + 0.35 * mottle, 0.0, 1.0)
    rgb = rgb * (1.0 - 0.20 * wet[..., None]) + np.array([0.62, 0.70, 0.58], dtype=np.float32) * (0.20 * wet[..., None])

    wet_bank = np.clip((wx - (shore_x - 0.20)) / 0.20, 0.0, 1.0).astype(np.float32)
    rgb *= (1.0 - 0.45 * wet_bank)[..., None]

    rgb = apply_shadow_bounce(rgb)
    rgb += tooth[..., None] * 0.008

    # Smooth organic shoreline AND gentle far-edge dissolution into mist/bank (ZERO straight cutoff!)
    alpha_shore = np.clip((shore_x - wx) / 0.16, 0.0, 1.0).astype(np.float32)
    far_edge = 17.8 + 1.4 * np.sin(wx * 1.1 + 0.5) + 0.65 * mottle
    alpha_far = np.clip((far_edge - wy) / 2.2, 0.0, 1.0).astype(np.float32)
    alpha = alpha_shore * alpha_far

    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.3) * alpha[..., None]
    rgba[:, :, 3] = alpha
    return rgba


# =============================================================================
# Stage 3 — 2x Supersampled PyCairo Helpers & Organic Kyoto Garden Assets
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
    fade_dist: float = 22.0,
    taper: bool = True,
    tremor: float = 0.30,
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
        g = cairo.LinearGradient(float(proj[0][0]), float(proj[0][1]), float(proj[-1][0]), float(proj[-1][1]))
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


def fog_mix(c3: Iterable[float], dist: float, fog_rgb: np.ndarray, fog_start: float = 5.5, fog_density: float = 0.055) -> tuple[float, float, float]:
    ft = 1.0 - math.exp(-max(0.0, dist - fog_start) * fog_density)
    c = np.asarray(c3, dtype=np.float64) * (1.0 - ft) + fog_rgb * ft
    return (float(c[0]), float(c[1]), float(c[2]))


def draw_far_mossy_pond_bank(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    mirrored: bool = False,
) -> None:
    """Natural Arashiyama Tenryu-ji style far mossy shoreline with layered shrubs, sedge reeds & weathered bank rocks."""
    alpha_m = 0.62 if mirrored else 0.98
    sign = -1.0 if mirrored else 1.0

    # Subtle low irregular bank spine (NOT big smooth balloons!)
    bank_mounds = [
        (0.6, 17.6, 3.0, 0.38, [0.09, 0.17, 0.11], [0.18, 0.31, 0.16], [0.38, 0.50, 0.25]),
        (3.8, 19.0, 4.4, 0.44, [0.08, 0.16, 0.10], [0.17, 0.30, 0.15], [0.36, 0.48, 0.24]),
        (7.8, 17.4, 4.8, 0.52, [0.07, 0.15, 0.10], [0.16, 0.29, 0.15], [0.35, 0.47, 0.23]),
        (6.4, 13.6, 2.6, 0.36, [0.08, 0.15, 0.10], [0.17, 0.31, 0.15], [0.38, 0.50, 0.24]),
    ]
    for idx_m, (cx, cy, rx, rz, cd, cm, cl) in enumerate(bank_mounds):
        srng = np.random.default_rng(4200 + idx_m * 17)
        p_c = np.array([cx, cy, Z_POND], dtype=np.float64)
        dist = float(np.linalg.norm(p_c - cam.eye))
        ux, uy, uz = cam.project(p_c)
        if uz < Z_NEAR:
            continue
        px_m = cam.K[0, 0] / float(uz)
        w_px, h_px = rx * px_m, rz * px_m
        c1 = fog_mix(cd, dist, fog_rgb)
        c2 = fog_mix(cm, dist, fog_rgb)
        c3 = fog_mix(cl, dist, fog_rgb)

        # Irregular sculpted earth bank base
        g = cairo.LinearGradient(float(ux), float(uy) - sign * h_px, float(ux), float(uy) + sign * h_px * 0.15)
        g.add_color_stop_rgba(0.0, *c3, alpha_m)
        g.add_color_stop_rgba(0.55, *c2, alpha_m)
        g.add_color_stop_rgba(1.0, *c1, alpha_m)
        ctx.set_source(g)
        n_b = 14
        ctx.move_to(float(ux) - w_px, float(uy) + sign * 2.5)
        for k in range(n_b + 1):
            fr = k / n_b
            bx = float(ux) - w_px + 2.0 * w_px * fr
            env = math.sin(fr * math.pi) ** 0.75
            und = 0.72 + 0.28 * math.sin(fr * 9.5 + idx_m * 1.9) + float(srng.uniform(-0.14, 0.14))
            by = float(uy) - sign * h_px * env * und
            ctx.line_to(bx, by)
        ctx.line_to(float(ux) + w_px, float(uy) + sign * 2.5)
        ctx.close_path()
        ctx.fill()

        # Layered Kyoto garden azalea/moss shrub cushions & scalloped foliage tufts breaking up the shoreline!
        n_shrubs = 11
        for k_s in range(n_shrubs):
            fr = 0.08 + 0.84 * (k_s / max(1, n_shrubs - 1)) + float(srng.uniform(-0.04, 0.04))
            sx = float(ux) - w_px * 0.92 + 1.84 * w_px * fr
            env = math.sin(np.clip(fr, 0.05, 0.95) * math.pi)
            sy = float(uy) - sign * h_px * env * float(srng.uniform(0.35, 0.85))
            srx = h_px * float(srng.uniform(0.55, 1.15))
            sry = srx * float(srng.uniform(0.42, 0.65))
            lit_v = float(srng.uniform(0.15, 0.88))
            col = tuple(c1[j] * (1 - lit_v) + c3[j] * lit_v for j in range(3))
            ctx.save()
            ctx.translate(sx, sy)
            ctx.scale(1.0, sry / max(1.0, srx))
            ctx.set_source_rgba(*col, 0.90 * alpha_m)
            n_lb = 7
            for lb in range(n_lb):
                a0 = (math.tau * lb) / n_lb
                a1 = (math.tau * (lb + 0.5)) / n_lb
                a2 = (math.tau * (lb + 1.0)) / n_lb
                ro = srx * float(srng.uniform(0.88, 1.18))
                ri = srx * 0.72
                if lb == 0:
                    ctx.move_to(math.cos(a0) * ri, math.sin(a0) * ri)
                ctx.curve_to(math.cos(a1) * ro, math.sin(a1) * ro, math.cos(a1 + 0.14) * ro, math.sin(a1 + 0.14) * ro, math.cos(a2) * ri, math.sin(a2) * ri)
            ctx.close_path()
            ctx.fill()
            ctx.restore()

        # Delicate lakeside Japanese sedge & iris grass blades rising cleanly along the bank water edge!
        if not mirrored:
            for k_g in range(18):
                gx = float(ux) + float(srng.uniform(-w_px * 0.88, w_px * 0.88))
                gh = h_px * float(srng.uniform(0.45, 1.05))
                lean = float(srng.uniform(-gh * 0.35, gh * 0.35))
                ctx.set_source_rgba(*(c1 if k_g % 2 == 0 else c2), 0.78)
                ctx.set_line_width(max(0.8, h_px * 0.032))
                ctx.move_to(gx, float(uy) + 2.0)
                ctx.curve_to(gx + lean * 0.3, float(uy) - gh * 0.5, gx + lean * 0.7, float(uy) - gh * 0.85, gx + lean, float(uy) - gh)
                ctx.stroke()

    # Craggy rocky promontory island supporting the snow-viewing stone lantern at (4.05, 12.9)
    draw_mossy_boulder(ctx, cam, np.array([4.05, 12.95, Z_POND + 0.01]), 0.96, 0.34, rng, fog_rgb, mirrored=mirrored)
    draw_mossy_boulder(ctx, cam, np.array([3.32, 13.10, Z_POND + 0.01]), 0.56, 0.24, rng, fog_rgb, mirrored=mirrored)
    # Lush garden shrub cushion softening the far end of the Voronoi stone walkway behind pillar #4
    draw_mossy_boulder(ctx, cam, np.array([0.45, 16.6, Z_GARDEN]), 0.82, 0.44, rng, fog_rgb, mirrored=mirrored)


def draw_bamboo_leaf_spray(
    ctx: cairo.Context,
    x: float,
    y: float,
    scale: float,
    base_ang: float,
    seed_rng: np.random.Generator,
    dark_c: tuple[float, float, float],
    lit_c: tuple[float, float, float],
    alpha: float = 0.94,
    mirrored: bool = False,
) -> None:
    """Lush per-seed cubic Bezier Japanese bamboo leaf cascade varying length, droop, and nicks."""
    n_leaves = int(seed_rng.integers(5, 9))
    sign = -1.0 if mirrored else 1.0
    for i in range(n_leaves):
        spread = (i - (n_leaves - 1) * 0.5) * float(seed_rng.uniform(0.20, 0.32))
        ang = base_ang + spread + float(seed_rng.uniform(-0.14, 0.14))
        ln = scale * float(seed_rng.uniform(0.75, 1.38))
        wid = ln * float(seed_rng.uniform(0.115, 0.165))
        droop = sign * ln * float(seed_rng.uniform(0.18, 0.38))
        nick = float(seed_rng.uniform(-0.08, 0.08)) * wid

        ca, sa = math.cos(ang), math.sin(ang)
        px, py = -sa, ca
        tip_x = x + ca * ln
        tip_y = y + sa * ln + droop
        c1_x = x + ca * (ln * 0.36) + px * wid
        c1_y = y + sa * (ln * 0.36) + py * wid + droop * 0.18
        c2_x = x + ca * (ln * 0.72) + px * (wid * 0.45 + nick)
        c2_y = y + sa * (ln * 0.72) + py * (wid * 0.45) + droop * 0.58
        c3_x = x + ca * (ln * 0.70) - px * (wid * 0.42)
        c3_y = y + sa * (ln * 0.70) - py * (wid * 0.42) + droop * 0.62
        c4_x = x + ca * (ln * 0.34) - px * (wid * 0.92 - nick)
        c4_y = y + sa * (ln * 0.34) - py * (wid * 0.92) + droop * 0.20

        lit_f = float(seed_rng.uniform(0.12, 0.96))
        col = tuple(dark_c[k] * (1.0 - lit_f) + lit_c[k] * lit_f for k in range(3))
        ctx.set_source_rgba(*col, alpha)
        ctx.move_to(x, y)
        ctx.curve_to(c1_x, c1_y, c2_x, c2_y, tip_x, tip_y)
        ctx.curve_to(c3_x, c3_y, c4_x, c4_y, x, y)
        ctx.close_path()
        ctx.fill()


def draw_bamboo_stalk(
    ctx: cairo.Context,
    cam: Camera3D,
    base_pos: np.ndarray,
    height: float,
    radius: float,
    arch_x: float,
    arch_y: float,
    seed: int,
    fog_rgb: np.ndarray,
    mirrored: bool = False,
) -> None:
    """Per-seed gracefully arching Kyoto bamboo culm with internode rings & rich feathery leaf cascades."""
    srng = np.random.default_rng(seed)
    dist = float(np.linalg.norm(base_pos - cam.eye))
    n_seg = 12
    pts3 = []
    for i in range(n_seg + 1):
        t = i / n_seg
        bend = t * t * (0.55 + 0.45 * t)
        p = base_pos + np.array([arch_x * bend, arch_y * bend, height * t], dtype=np.float64)
        pts3.append(p)
    if mirrored:
        pts3 = mirror_pts_across_plane(pts3, Z_POND)

    dark_culm = fog_mix([0.10, 0.20, 0.11], dist, fog_rgb)
    mid_culm = fog_mix([0.23, 0.38, 0.18], dist, fog_rgb)
    lit_culm = fog_mix([0.64, 0.75, 0.36], dist, fog_rgb)
    node_col = fog_mix([0.06, 0.13, 0.07], dist, fog_rgb)
    leaf_dark = fog_mix([0.08, 0.18, 0.10], dist, fog_rgb)
    leaf_lit = fog_mix([0.52, 0.66, 0.30], dist, fog_rgb)

    alpha_m = 0.60 if mirrored else 0.98
    for i in range(n_seg):
        p0, p1 = pts3[i], pts3[i + 1]
        cl = cam.clip_segment_3d(p0, p1)
        if cl is None:
            continue
        a, b = cl
        ua, va, za = cam.project(a)
        ub, vb, zb = cam.project(b)
        t_mid = (i + 0.5) / n_seg
        r_m = radius * (1.0 - 0.55 * t_mid)
        w0 = max(1.2, r_m * 2.0 * cam.K[0, 0] / max(0.8, float(za)))
        w1 = max(1.0, r_m * 2.0 * cam.K[0, 0] / max(0.8, float(zb)))

        dx, dy = float(ub - ua), float(vb - va)
        dl = math.hypot(dx, dy) + 1e-6
        nx, ny = -dy / dl, dx / dl
        # Slight overlap (+0.8 px) along segment direction prevents any subpixel crack!
        ox, oy = (dx / dl) * 0.8, (dy / dl) * 0.8
        q = [
            (float(ua) - nx * w0 * 0.5 - ox, float(va) - ny * w0 * 0.5 - oy),
            (float(ua) + nx * w0 * 0.5 - ox, float(va) + ny * w0 * 0.5 - oy),
            (float(ub) + nx * w1 * 0.5 + ox, float(vb) + ny * w1 * 0.5 + oy),
            (float(ub) - nx * w1 * 0.5 + ox, float(vb) - ny * w1 * 0.5 + oy),
        ]
        g = cairo.LinearGradient(q[0][0], q[0][1], q[1][0], q[1][1])
        g.add_color_stop_rgba(0.0, *dark_culm, alpha_m)
        g.add_color_stop_rgba(0.54, *mid_culm, alpha_m)
        g.add_color_stop_rgba(0.86, *lit_culm, alpha_m)
        g.add_color_stop_rgba(1.0, *mid_culm, alpha_m)
        ctx.set_source(g)
        ctx.move_to(*q[0])
        for qp in q[1:]:
            ctx.line_to(*qp)
        ctx.close_path()
        ctx.fill()

        if i > 0:
            ctx.set_source_rgba(*node_col, 0.78 * alpha_m)
            ctx.set_line_width(max(0.8, w0 * 0.16))
            ctx.move_to(float(ua) - nx * w0 * 0.56, float(va) - ny * w0 * 0.56)
            ctx.line_to(float(ua) + nx * w0 * 0.56, float(va) + ny * w0 * 0.56)
            ctx.stroke()

        # Lush feathery pairs of twigs + multi-cluster Bezier bamboo leaf cascades on upper two-thirds!
        if i >= 3:
            n_twigs = 2 if i >= 5 else 1
            for t_k in range(n_twigs):
                side = -1.0 if ((i + t_k) % 2 == 0) else 1.0
                twig_len = float(srng.uniform(34.0, 76.0)) * (10.5 / max(3.5, float(za)))
                t_ang = (-0.25 if side > 0 else math.pi + 0.25) + float(srng.uniform(-0.42, 0.42))
                if mirrored:
                    t_ang = -t_ang
                tx = float(ub) + math.cos(t_ang) * twig_len
                ty = float(vb) + math.sin(t_ang) * twig_len
                ctx.set_source_rgba(*node_col, 0.72 * alpha_m)
                ctx.set_line_width(max(0.65, w1 * 0.20))
                ctx.move_to(float(ub), float(vb))
                ctx.line_to(tx, ty)
                ctx.stroke()
                # Mid-twig + tip leaf sprays for genuine airy Kyoto bamboo canopy!
                for frac_t, sc_m in ((0.58, 0.85), (1.0, 1.12)):
                    lx = float(ub) + (tx - float(ub)) * frac_t
                    ly = float(vb) + (ty - float(vb)) * frac_t
                    draw_bamboo_leaf_spray(
                        ctx,
                        lx,
                        ly,
                        scale=twig_len * sc_m * float(srng.uniform(0.75, 1.15)),
                        base_ang=t_ang + (0.28 if not mirrored else -0.28),
                        seed_rng=srng,
                        dark_c=leaf_dark,
                        lit_c=leaf_lit,
                        alpha=0.88 * alpha_m,
                        mirrored=mirrored,
                    )


def draw_stone_garden_lantern(
    ctx: cairo.Context,
    cam: Camera3D,
    base_pos: np.ndarray,
    scale: float,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    mirrored: bool = False,
) -> None:
    """Traditional Kyoto Yukimi-doro (snow-viewing stone lantern) resting on a mossy rock island."""
    dist = float(np.linalg.norm(base_pos - cam.eye))
    alpha_m = 0.64 if mirrored else 0.99
    c_sh = fog_mix([0.13, 0.17, 0.13], dist, fog_rgb)
    c_mid = fog_mix([0.28, 0.33, 0.27], dist, fog_rgb)
    c_lit = fog_mix([0.62, 0.67, 0.46], dist, fog_rgb)
    c_moss = fog_mix([0.20, 0.37, 0.16], dist, fog_rgb)

    def proj_pt(dx: float, dz: float) -> tuple[float, float, float]:
        p = base_pos + np.array([dx * scale, 0.0, dz * scale], dtype=np.float64)
        if mirrored:
            p = mirror_pts_across_plane([p], Z_POND)[0]
        u, v, z = cam.project(p)
        return float(u), float(v), float(z)

    def poly_rel(rel_pts: list[tuple[float, float]], col_l, col_r, col_hi=None):
        pr = [proj_pt(dx, dz) for dx, dz in rel_pts]
        if any(z < Z_NEAR for _, _, z in pr):
            return
        xs = [p[0] for p in pr]
        ys = [p[1] for p in pr]
        g = cairo.LinearGradient(min(xs), min(ys), max(xs), min(ys))
        g.add_color_stop_rgba(0.0, *col_l, alpha_m)
        g.add_color_stop_rgba(0.65, *col_r, alpha_m)
        g.add_color_stop_rgba(1.0, *(col_hi if col_hi is not None else c_lit), alpha_m)
        ctx.set_source(g)
        ctx.move_to(pr[0][0], pr[0][1])
        for p in pr[1:]:
            ctx.line_to(p[0], p[1])
        ctx.close_path()
        ctx.fill()

    # Arched stone legs rising cleanly above the mossy rock island!
    poly_rel([(-0.32, 0.0), (-0.18, 0.44), (-0.07, 0.44), (-0.19, 0.0)], c_sh, c_mid)
    poly_rel([(0.19, 0.0), (0.07, 0.44), (0.18, 0.44), (0.32, 0.0)], c_sh, c_mid)
    poly_rel([(-0.04, 0.04), (-0.04, 0.44), (0.04, 0.44), (0.04, 0.04)], c_sh, c_sh, c_mid)
    # Platform & carved stone firebox
    poly_rel([(-0.36, 0.42), (-0.31, 0.54), (0.31, 0.54), (0.36, 0.42)], c_sh, c_mid)
    poly_rel([(-0.22, 0.54), (-0.22, 0.88), (0.22, 0.88), (0.22, 0.54)], c_sh, c_mid)

    # Pure warm cream-amber candle window glow (NO dark gradient stop on the flame!)
    cu, cv, cz = proj_pt(0.0, 0.71)
    if cz > Z_NEAR:
        rad = max(3.5, 0.16 * scale * cam.K[0, 0] / cz)
        glow = cairo.RadialGradient(cu, cv, rad * 0.1, cu, cv, rad * 2.5)
        glow.add_color_stop_rgba(0.0, 1.0, 0.96, 0.78, 0.94 * alpha_m)
        glow.add_color_stop_rgba(0.45, 0.92, 0.82, 0.46, 0.48 * alpha_m)
        glow.add_color_stop_rgba(1.0, 0.82, 0.72, 0.34, 0.0)
        ctx.set_source(glow)
        ctx.arc(cu, cv, rad * 2.5, 0, math.tau)
        ctx.fill()
        poly_rel(
            [(-0.09, 0.60), (-0.09, 0.82), (0.09, 0.82), (0.09, 0.60)],
            (0.98, 0.90, 0.62),
            (1.0, 0.97, 0.82),
            (0.98, 0.92, 0.66),
        )

    # Wide snow-viewing curved umbrella stone cap + lotus finial + velvet moss blanket
    poly_rel(
        [(-0.60, 0.86), (-0.42, 1.03), (-0.14, 1.14), (0.14, 1.14), (0.42, 1.03), (0.60, 0.86), (0.48, 0.82), (-0.48, 0.82)],
        c_sh,
        c_mid,
    )
    poly_rel([(-0.09, 1.13), (-0.05, 1.26), (0.0, 1.33), (0.05, 1.26), (0.09, 1.13)], c_mid, c_lit)
    poly_rel([(-0.44, 0.95), (-0.20, 1.11), (0.12, 1.13), (0.36, 1.02), (0.14, 0.98), (-0.26, 0.94)], c_moss, c_moss, c_lit)


def draw_veranda_pillar(
    ctx: cairo.Context,
    cam: Camera3D,
    y_pos: float,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    mirrored: bool = False,
) -> None:
    """Weathered square Kyoto cedar veranda post with top joinery bracket and sunlit bevel."""
    px = X_VERANDA_EDGE - 0.06
    hw = 0.068
    z0 = Z_DECK
    z1 = float(eave_height_at_x(px))
    dist = float(np.linalg.norm(np.array([px, y_pos, 1.5]) - cam.eye))
    alpha_m = 0.58 if mirrored else 0.99

    c_front = fog_mix([0.13, 0.095, 0.068], dist, fog_rgb)
    c_right = fog_mix([0.34, 0.27, 0.17], dist, fog_rgb)
    c_rim = fog_mix([0.74, 0.72, 0.44], dist, fog_rgb)

    q_front = [
        np.array([px - hw, y_pos - hw, z0]),
        np.array([px + hw, y_pos - hw, z0]),
        np.array([px + hw, y_pos - hw, z1]),
        np.array([px - hw, y_pos - hw, z1]),
    ]
    q_right = [
        np.array([px + hw, y_pos - hw, z0]),
        np.array([px + hw, y_pos + hw, z0]),
        np.array([px + hw, y_pos + hw, z1]),
        np.array([px + hw, y_pos - hw, z1]),
    ]
    q_br = [
        np.array([px - hw * 1.4, y_pos - hw * 0.9, z1 - 0.14]),
        np.array([px + hw * 2.8, y_pos - hw * 0.9, z1 - 0.14]),
        np.array([px + hw * 2.8, y_pos - hw * 0.9, z1]),
        np.array([px - hw * 1.4, y_pos - hw * 0.9, z1]),
    ]
    if mirrored:
        q_front = mirror_pts_across_plane(q_front, Z_POND)
        q_right = mirror_pts_across_plane(q_right, Z_POND)
        q_br = mirror_pts_across_plane(q_br, Z_POND)

    fill_3d_quad(ctx, cam, q_front, (*c_front, alpha_m), (*(v * 0.78 for v in c_front), alpha_m))
    fill_3d_quad(ctx, cam, q_right, (*c_right, alpha_m), (*(v * 0.85 for v in c_right), alpha_m))
    fill_3d_quad(ctx, cam, q_br, (*c_front, alpha_m))

    if not mirrored:
        for a3, b3, col, lw in (
            (q_front[0], q_front[3], (0.04, 0.03, 0.02, 0.82), 1.6),
            (q_front[1], q_front[2], (*c_rim, 0.78), 2.0),
        ):
            seg = cam.project_polyline(np.array([a3, b3]))
            if len(seg) >= 2:
                stroke_catmull_rom(ctx, [(p[0], p[1]) for p in seg], base_w=lw, z_depth=dist, rgba=col, rng=rng, taper=False, tremor=0.22)


def draw_hero_bench_and_matcha_bowl(ctx: cairo.Context, cam: Camera3D, rng: np.random.Generator, fog_rgb: np.ndarray) -> None:
    """
    Hero prop on the wooden engawa veranda:
    Low Kyoto cedar tea bench with mortised joinery & wood grain, folded linen
    chakin cloth, rustic hand-thrown celadon-charcoal ceramic matcha bowl (chawan)
    with frothy jade matcha tea, carved bamboo tea whisk (chasen) & scoop (chashaku).
    (Delicate steam mist is rendered in a soft Gaussian-blurred pass afterward!)
    """
    cx, cy, z_deck = -1.38, 4.25, Z_DECK
    hx, hy = 0.28, 0.48
    h_bench = 0.30
    zt = z_deck + h_bench
    dist = float(np.linalg.norm(np.array([cx, cy, zt]) - cam.eye))

    c_top_lit = fog_mix([0.46, 0.36, 0.23], dist, fog_rgb)
    c_top_sh = fog_mix([0.24, 0.18, 0.11], dist, fog_rgb)
    c_side = fog_mix([0.18, 0.13, 0.085], dist, fog_rgb)
    c_dark = fog_mix([0.095, 0.070, 0.048], dist, fog_rgb)
    ink = (0.05, 0.038, 0.028, 0.85)

    # Trestle legs with traditional arched Japanese cutout & horizontal stretcher rail
    for ly in (cy + hy - 0.10, cy - hy + 0.10):
        leg_f = [
            np.array([cx - hx + 0.05, ly - 0.03, z_deck]),
            np.array([cx + hx - 0.05, ly - 0.03, z_deck]),
            np.array([cx + hx - 0.05, ly - 0.03, zt - 0.04]),
            np.array([cx - hx + 0.05, ly - 0.03, zt - 0.04]),
        ]
        leg_r = [
            np.array([cx + hx - 0.05, ly - 0.03, z_deck]),
            np.array([cx + hx - 0.05, ly + 0.03, z_deck]),
            np.array([cx + hx - 0.05, ly + 0.03, zt - 0.04]),
            np.array([cx + hx - 0.05, ly - 0.03, zt - 0.04]),
        ]
        fill_3d_quad(ctx, cam, leg_f, (*c_dark, 0.99))
        fill_3d_quad(ctx, cam, leg_r, (*c_side, 0.99))

    # Stretcher bar linking front & back trestle legs
    stretcher = [
        np.array([cx + 0.02, cy - hy + 0.10, z_deck + 0.10]),
        np.array([cx + 0.02, cy + hy - 0.10, z_deck + 0.10]),
        np.array([cx + 0.02, cy + hy - 0.10, z_deck + 0.145]),
        np.array([cx + 0.02, cy - hy + 0.10, z_deck + 0.145]),
    ]
    fill_3d_quad(ctx, cam, stretcher, (*c_side, 0.95), (*c_dark, 0.95))

    q_edge_r = [
        np.array([cx + hx, cy - hy, zt - 0.045]),
        np.array([cx + hx, cy + hy, zt - 0.045]),
        np.array([cx + hx, cy + hy, zt]),
        np.array([cx + hx, cy - hy, zt]),
    ]
    q_edge_f = [
        np.array([cx - hx, cy - hy, zt - 0.045]),
        np.array([cx + hx, cy - hy, zt - 0.045]),
        np.array([cx + hx, cy - hy, zt]),
        np.array([cx - hx, cy - hy, zt]),
    ]
    q_top = [
        np.array([cx - hx, cy - hy, zt]),
        np.array([cx + hx, cy - hy, zt]),
        np.array([cx + hx, cy + hy, zt]),
        np.array([cx - hx, cy + hy, zt]),
    ]
    fill_3d_quad(ctx, cam, q_edge_f, (*c_dark, 0.99))
    fill_3d_quad(ctx, cam, q_edge_r, (*c_side, 0.99))
    fill_3d_quad(ctx, cam, q_top, (*c_top_sh, 0.99), (*c_top_lit, 0.99))

    # Wood plank seam down the middle of the cedar bench top + sunlit golden slat bar!
    seam_b = cam.project_polyline(np.array([np.array([cx, cy - hy, zt + 0.001]), np.array([cx, cy + hy, zt + 0.001])]))
    if len(seam_b) >= 2:
        stroke_catmull_rom(ctx, [(p[0], p[1]) for p in seam_b], base_w=1.2, z_depth=dist, rgba=(0.06, 0.045, 0.03, 0.55), rng=rng, taper=False, tremor=0.18)

    q_sunbar = [
        np.array([cx - hx + 0.03, cy - 0.14, zt + 0.001]),
        np.array([cx + hx - 0.01, cy - 0.14, zt + 0.001]),
        np.array([cx + hx - 0.01, cy + 0.28, zt + 0.001]),
        np.array([cx - hx + 0.03, cy + 0.28, zt + 0.001]),
    ]
    fill_3d_quad(ctx, cam, q_sunbar, (0.85, 0.87, 0.55, 0.30), (0.95, 0.94, 0.76, 0.42))

    # Crisp hand-inked bench edges + sunlit right top bevel
    for i in range(4):
        seg = cam.project_polyline(np.array([q_top[i], q_top[(i + 1) % 4]]))
        if len(seg) >= 2:
            col_e = (0.76, 0.72, 0.46, 0.55) if i == 1 else ink
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in seg], base_w=1.6, z_depth=dist, rgba=col_e, rng=rng, taper=False, tremor=0.20)

    # Folded warm parchment-cream linen tea cloth (chakin) beneath the bowl
    q_cloth = [
        np.array([cx - 0.15, cy - 0.18, zt + 0.003]),
        np.array([cx + 0.17, cy - 0.16, zt + 0.003]),
        np.array([cx + 0.16, cy + 0.20, zt + 0.003]),
        np.array([cx - 0.16, cy + 0.18, zt + 0.003]),
    ]
    fill_3d_quad(ctx, cam, q_cloth, (0.76, 0.75, 0.62, 0.96), (0.92, 0.91, 0.79, 0.96))

    # --- Handmade Kyoto Ceramic Matcha Bowl (Chawan) ---
    bu, bv, bz = cam.project(np.array([cx + 0.01, cy + 0.03, zt + 0.004]))
    px_m = cam.K[0, 0] / float(bz)
    bw = 0.094 * px_m
    bh = 0.108 * px_m

    # Soft cast shadow of bowl extending down-left along SUN_DIR
    ctx.save()
    ctx.translate(float(bu) - bw * 0.44, float(bv) + bh * 0.08)
    ctx.scale(1.28, 0.45)
    g_sh = cairo.RadialGradient(0, 0, bw * 0.1, 0, 0, bw * 1.18)
    g_sh.add_color_stop_rgba(0.0, 0.05, 0.04, 0.03, 0.64)
    g_sh.add_color_stop_rgba(1.0, 0.05, 0.04, 0.03, 0.0)
    ctx.set_source(g_sh)
    ctx.arc(0, 0, bw * 1.18, 0, math.tau)
    ctx.fill()
    ctx.restore()

    # Foot ring (kodai) of raw dark stoneware clay
    ctx.set_source_rgba(0.14, 0.11, 0.085, 0.98)
    ctx.move_to(float(bu) - bw * 0.56, float(bv) - bh * 0.06)
    ctx.curve_to(float(bu) - bw * 0.54, float(bv) + bh * 0.12, float(bu) + bw * 0.54, float(bv) + bh * 0.12, float(bu) + bw * 0.56, float(bv) - bh * 0.06)
    ctx.close_path()
    ctx.fill()

    # Hand-thrown ceramic bowl body
    g_bowl = cairo.LinearGradient(float(bu) - bw, float(bv) - bh * 0.5, float(bu) + bw, float(bv) - bh * 0.5)
    g_bowl.add_color_stop_rgba(0.0, 0.10, 0.12, 0.10, 1.0)
    g_bowl.add_color_stop_rgba(0.42, 0.21, 0.25, 0.19, 1.0)
    g_bowl.add_color_stop_rgba(0.78, 0.48, 0.54, 0.38, 1.0)
    g_bowl.add_color_stop_rgba(1.0, 0.26, 0.31, 0.22, 1.0)
    ctx.set_source(g_bowl)
    ctx.move_to(float(bu) - bw, float(bv) - bh * 0.84)
    ctx.curve_to(float(bu) - bw * 1.04, float(bv) - bh * 0.24, float(bu) - bw * 0.78, float(bv) + bh * 0.02, float(bu) - bw * 0.52, float(bv) + bh * 0.03)
    ctx.curve_to(float(bu) - bw * 0.15, float(bv) + bh * 0.06, float(bu) + bw * 0.15, float(bv) + bh * 0.06, float(bu) + bw * 0.52, float(bv) + bh * 0.03)
    ctx.curve_to(float(bu) + bw * 0.78, float(bv) + bh * 0.02, float(bu) + bw * 1.04, float(bv) - bh * 0.24, float(bu) + bw, float(bv) - bh * 0.84)
    ctx.close_path()
    ctx.fill_preserve()
    ctx.set_source_rgba(0.055, 0.065, 0.048, 0.82)
    ctx.set_line_width(1.35)
    ctx.stroke()

    # Thick artisanal celadon-cream glaze drip band
    g_glaze = cairo.LinearGradient(float(bu) - bw, float(bv) - bh * 0.7, float(bu) + bw, float(bv) - bh * 0.7)
    g_glaze.add_color_stop_rgba(0.0, 0.32, 0.38, 0.29, 0.84)
    g_glaze.add_color_stop_rgba(0.60, 0.65, 0.70, 0.53, 0.88)
    g_glaze.add_color_stop_rgba(0.86, 0.90, 0.92, 0.76, 0.94)
    g_glaze.add_color_stop_rgba(1.0, 0.52, 0.58, 0.44, 0.84)
    ctx.set_source(g_glaze)
    ctx.move_to(float(bu) - bw * 0.99, float(bv) - bh * 0.84)
    ctx.line_to(float(bu) + bw * 0.99, float(bv) - bh * 0.84)
    ctx.curve_to(float(bu) + bw * 0.96, float(bv) - bh * 0.42, float(bu) + bw * 0.62, float(bv) - bh * 0.26, float(bu) + bw * 0.34, float(bv) - bh * 0.34)
    ctx.curve_to(float(bu) + bw * 0.12, float(bv) - bh * 0.42, float(bu) - bw * 0.06, float(bv) - bh * 0.16, float(bu) - bw * 0.30, float(bv) - bh * 0.32)
    ctx.curve_to(float(bu) - bw * 0.62, float(bv) - bh * 0.46, float(bu) - bw * 0.92, float(bv) - bh * 0.38, float(bu) - bw * 0.99, float(bv) - bh * 0.84)
    ctx.close_path()
    ctx.fill()

    # Bowl rim & vibrant frothy ceremonial matcha tea
    ctx.save()
    ctx.translate(float(bu), float(bv) - bh * 0.84)
    ctx.scale(1.0, 0.35)
    ctx.set_source_rgba(0.76, 0.78, 0.64, 0.98)
    ctx.arc(0, 0, bw * 1.0, 0, math.tau)
    ctx.fill()
    ctx.set_source_rgba(0.15, 0.20, 0.13, 0.98)
    ctx.arc(0, 2.0, bw * 0.88, 0, math.tau)
    ctx.fill()
    g_tea = cairo.RadialGradient(bw * 0.24, -bw * 0.18, bw * 0.08, 0, bw * 0.12, bw * 0.86)
    g_tea.add_color_stop_rgba(0.0, 0.84, 0.92, 0.56, 1.0)
    g_tea.add_color_stop_rgba(0.45, 0.48, 0.68, 0.28, 1.0)
    g_tea.add_color_stop_rgba(1.0, 0.19, 0.35, 0.15, 1.0)
    ctx.set_source(g_tea)
    ctx.arc(0, bw * 0.14, bw * 0.82, 0, math.tau)
    ctx.fill()
    # Delicate frothy matcha micro-foam ring
    ctx.set_source_rgba(0.88, 0.95, 0.68, 0.46)
    ctx.set_line_width(2.2)
    ctx.arc(bw * 0.06, bw * 0.12, bw * 0.64, -1.2, 0.6)
    ctx.stroke()
    ctx.set_source_rgba(0.98, 0.98, 0.90, 0.92)
    ctx.set_line_width(3.0)
    ctx.arc(0, 0, bw * 0.95, -0.85, 0.18)
    ctx.stroke()
    ctx.restore()

    # Small bamboo matcha whisk (chasen) standing neatly beside the bowl on the far corner
    wu, wv, wz = cam.project(np.array([cx + 0.12, cy + 0.24, zt + 0.005]))
    wh = 0.095 * (cam.K[0, 0] / float(wz))
    ww = 0.030 * (cam.K[0, 0] / float(wz))
    ctx.set_source_rgba(0.56, 0.50, 0.30, 0.96)
    ctx.move_to(float(wu) - ww * 0.52, float(wv))
    ctx.line_to(float(wu) + ww * 0.52, float(wv))
    ctx.curve_to(float(wu) + ww * 1.05, float(wv) - wh * 0.48, float(wu) + ww * 0.92, float(wv) - wh * 0.86, float(wu) + ww * 0.55, float(wv) - wh)
    ctx.line_to(float(wu) - ww * 0.55, float(wv) - wh)
    ctx.curve_to(float(wu) - ww * 0.92, float(wv) - wh * 0.86, float(wu) - ww * 1.05, float(wv) - wh * 0.48, float(wu) - ww * 0.52, float(wv))
    ctx.close_path()
    ctx.fill()
    # Delicate vertical whisk tines & dark woven silk band
    ctx.set_source_rgba(0.22, 0.17, 0.10, 0.72)
    ctx.set_line_width(1.1)
    for k_t in (-0.55, -0.20, 0.20, 0.55):
        ctx.move_to(float(wu) + ww * k_t * 0.6, float(wv) - wh * 0.25)
        ctx.line_to(float(wu) + ww * k_t * 0.8, float(wv) - wh * 0.94)
        ctx.stroke()

    # Carved bamboo tea scoop (chashaku) resting cleanly along the front edge of the cloth
    s0 = cam.project(np.array([cx - 0.12, cy - 0.10, zt + 0.008]))
    s1 = cam.project(np.array([cx + 0.14, cy - 0.16, zt + 0.011]))
    stroke_catmull_rom(
        ctx,
        [(float(s0[0]), float(s0[1])), (float(s0[0] * 0.5 + s1[0] * 0.5), float(s0[1] * 0.5 + s1[1] * 0.5) - 2.0), (float(s1[0]), float(s1[1]))],
        base_w=4.0,
        z_depth=dist,
        rgba=(0.68, 0.62, 0.38, 0.95),
        rng=rng,
        taper=False,
        tremor=0.14,
    )


def paint_soft_matcha_steam(cam: Camera3D, rng: np.random.Generator) -> np.ndarray:
    """Delicate soft Gaussian-blurred sunlit morning steam curling organically above the matcha bowl."""
    surf, ctx = new_cairo_2x()
    bu, bv, bz = cam.project(np.array([-1.37, 4.28, Z_DECK + 0.38]))
    for off_x, h_st, ph, w_st, al in ((-12.0, 96.0, 0.2, 16.0, 0.24), (2.0, 128.0, 1.5, 22.0, 0.30), (15.0, 84.0, 2.8, 14.0, 0.20)):
        spts = []
        for k in range(8):
            t = k / 7.0
            sx = float(bu) + off_x + math.sin(t * 4.2 + ph) * (10.0 + 16.0 * t) - 18.0 * t
            sy = float(bv) - 12.0 - t * h_st
            spts.append((sx, sy))
        stroke_catmull_rom(ctx, spts, base_w=w_st, z_depth=3.0, rgba=(0.98, 0.98, 0.90, al), rng=rng, taper=True, tremor=0.22)
    steam_1x = finish_cairo_2x(surf)
    return gaussian_blur_2d(steam_1x, 5.5)


def draw_mossy_boulder(
    ctx: cairo.Context,
    cam: Camera3D,
    center: np.ndarray,
    rx: float,
    rz: float,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    mirrored: bool = False,
) -> None:
    """Asymmetric craggy Kyoto granite garden rock (niwa-ishi) with angular facets, irregular moss tufts & reeds."""
    p = mirror_pts_across_plane([center], Z_POND)[0] if mirrored else center
    ux, uy, uz = cam.project(p)
    if uz < Z_NEAR:
        return
    dist = float(np.linalg.norm(center - cam.eye))
    px_m = cam.K[0, 0] / float(uz)
    w_px, h_px = rx * px_m, rz * px_m
    alpha_m = 0.62 if mirrored else 0.99

    # Deterministic per-rock seed from world position so normal & mirrored rocks match 100%!
    srng = np.random.default_rng(int(abs(center[0] * 731 + center[1] * 1319)) & 0xFFFF)

    c_dark = fog_mix([0.085, 0.120, 0.095], dist, fog_rgb)
    c_mid = fog_mix([0.21, 0.25, 0.21], dist, fog_rgb)
    c_hi = fog_mix([0.38, 0.42, 0.34], dist, fog_rgb)
    c_moss_sh = fog_mix([0.14, 0.28, 0.13], dist, fog_rgb)
    c_moss_hi = fog_mix([0.48, 0.62, 0.27], dist, fog_rgb)

    sign = -1.0 if mirrored else 1.0
    x, y = float(ux), float(uy)

    if not mirrored:
        ctx.save()
        ctx.translate(x, y + h_px * 0.12)
        ctx.scale(1.0, 0.22)
        ctx.set_source_rgba(0.025, 0.068, 0.045, 0.44)
        ctx.arc(0, 0, w_px * 1.08, 0, math.tau)
        ctx.fill()
        ctx.restore()

    # Asymmetric craggy Kyoto granite silhouette (shifted off-center crest + weathered facet shoulders!)
    peak_x = float(srng.uniform(-0.28, 0.22)) * w_px
    r_pts = [
        (x - w_px * 0.96, y + sign * h_px * 0.08),
        (x - w_px * 0.84, y - sign * h_px * float(srng.uniform(0.42, 0.58))),
        (x - w_px * 0.48, y - sign * h_px * float(srng.uniform(0.76, 0.94))),
        (x + peak_x, y - sign * h_px * float(srng.uniform(0.95, 1.08))),
        (x + w_px * 0.44, y - sign * h_px * float(srng.uniform(0.72, 0.88))),
        (x + w_px * 0.80, y - sign * h_px * float(srng.uniform(0.44, 0.62))),
        (x + w_px * 0.98, y + sign * h_px * 0.05),
        (x + w_px * 0.42, y + sign * h_px * 0.18),
        (x - w_px * 0.46, y + sign * h_px * 0.18),
    ]

    # Base granite body clipped to organic rock contour
    ctx.save()
    ctx.move_to(*r_pts[0])
    for i in range(1, len(r_pts)):
        p_prev = r_pts[i - 1]
        p_cur = r_pts[i]
        mx, my = 0.5 * (p_prev[0] + p_cur[0]), 0.5 * (p_prev[1] + p_cur[1])
        ctx.curve_to(p_prev[0], p_prev[1], mx, my, p_cur[0], p_cur[1])
    ctx.close_path()

    g = cairo.LinearGradient(x - w_px * 0.75, y + sign * h_px * 0.1, x + w_px * 0.75, y - sign * h_px)
    g.add_color_stop_rgba(0.0, *c_dark, alpha_m)
    g.add_color_stop_rgba(0.48, *c_mid, alpha_m)
    g.add_color_stop_rgba(0.86, *c_hi, alpha_m)
    g.add_color_stop_rgba(1.0, *c_mid, alpha_m)
    ctx.set_source(g)
    ctx.fill_preserve()
    ctx.clip()

    # Angular sculpted sunlit right rock facet & shadowed left crevice plane!
    ctx.set_source_rgba(*c_dark, 0.48 * alpha_m)
    ctx.move_to(r_pts[0][0], r_pts[0][1])
    ctx.line_to(r_pts[2][0], r_pts[2][1])
    ctx.line_to(x + peak_x * 0.4 - w_px * 0.08, y - sign * h_px * 0.28)
    ctx.line_to(x - w_px * 0.25, y + sign * h_px * 0.22)
    ctx.close_path()
    ctx.fill()

    ctx.set_source_rgba(*c_hi, 0.36 * alpha_m)
    ctx.move_to(r_pts[3][0], r_pts[3][1])
    ctx.line_to(r_pts[5][0], r_pts[5][1])
    ctx.line_to(x + w_px * 0.35, y - sign * h_px * 0.12)
    ctx.line_to(x + peak_x * 0.3, y - sign * h_px * 0.36)
    ctx.close_path()
    ctx.fill()

    # Broken-up irregular mottled moss tufts clinging organically along upper ridges & cracks (NO swim-cap!)
    for _ in range(9):
        mx = x + float(srng.uniform(-0.68, 0.64)) * w_px
        my = y - sign * float(srng.uniform(0.32, 0.94)) * h_px
        mrx = w_px * float(srng.uniform(0.14, 0.30))
        mry = h_px * float(srng.uniform(0.12, 0.24))
        lit_m = float(np.clip(0.5 + (mx - x) / max(1.0, w_px) * 0.6, 0.1, 0.95))
        cm = tuple(c_moss_sh[j] * (1 - lit_m) + c_moss_hi[j] * lit_m for j in range(3))
        gm = cairo.RadialGradient(mrx * 0.15, -sign * mry * 0.15, mrx * 0.08, 0.0, 0.0, mrx)
        gm.add_color_stop_rgba(0.0, *cm, 0.90 * alpha_m)
        gm.add_color_stop_rgba(0.65, *c_moss_sh, 0.72 * alpha_m)
        gm.add_color_stop_rgba(1.0, *c_moss_sh, 0.0)
        ctx.save()
        ctx.translate(mx, my)
        ctx.scale(1.0, mry / max(1.0, mrx))
        ctx.set_source(gm)
        ctx.arc(0, 0, mrx, 0, math.tau)
        ctx.fill()
        ctx.restore()
    ctx.restore()

    # Delicate streamside reed & fern blades tucking organically around boulder base
    if not mirrored:
        for _ in range(7):
            bx = x + float(srng.uniform(-w_px * 0.95, -w_px * 0.25))
            by = y + h_px * 0.12
            bl = h_px * float(srng.uniform(0.55, 1.15))
            lean = float(srng.uniform(-bl * 0.55, bl * 0.25))
            ctx.set_source_rgba(*(c_moss_sh if srng.random() < 0.6 else c_moss_hi), 0.82)
            ctx.set_line_width(max(0.9, h_px * 0.032))
            ctx.move_to(bx, by)
            ctx.curve_to(bx + lean * 0.3, by - bl * 0.5, bx + lean * 0.7, by - bl * 0.82, bx + lean, by - bl)
            ctx.stroke()


# =============================================================================
# Stage 2 — Mirror-Calm Garden Pond with Fresnel & 3D Plane-Mirrored Reflections
# =============================================================================

def paint_pond_fresnel_water(
    canvas: np.ndarray,
    ref_backdrop: np.ndarray,
    mirrored_3d_rgba: np.ndarray,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    mottle_full: np.ndarray,
) -> np.ndarray:
    """Mirror-calm Kyoto garden pond with capillary-rippled horizon + 3D mirrored Fresnel reflections."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    rx_c = (xx - cam.K[0, 2]) / cam.K[0, 0]
    ry_c = (yy - cam.K[1, 2]) / cam.K[1, 1]
    RT = cam.R.T
    dx_w = RT[0, 0] * rx_c + RT[0, 1] * ry_c + RT[0, 2]
    dy_w = RT[1, 0] * rx_c + RT[1, 1] * ry_c + RT[1, 2]
    dz_w = RT[2, 0] * rx_c + RT[2, 1] * ry_c + RT[2, 2]

    hit = dz_w < -0.004
    t_w = np.where(hit, (Z_POND - cam.eye[2]) / np.where(hit, dz_w, -1.0), -1.0)
    wx = cam.eye[0] + dx_w * t_w
    wy = cam.eye[1] + dy_w * t_w
    valid = hit & (t_w > Z_NEAR) & (wy > 1.2) & (wy < 30.0) & (wx > 0.35)

    left_shore = 1.40 + 0.42 * np.sin(wy * 0.40 + 0.4) + 0.24 * np.cos(wy * 0.92 - 0.7) + 0.12 * mottle_full
    far_shore = 21.5 + 2.8 * np.sin(wx * 0.38 + 0.5) + 1.2 * mottle_full
    pond_mask = (
        valid.astype(np.float32)
        * np.clip((wx - left_shore) / 0.25, 0.0, 1.0).astype(np.float32)
        * np.clip((far_shore - wy) / 2.2, 0.0, 1.0).astype(np.float32)
    )
    if not np.any(pond_mask > 1e-4):
        return canvas

    # Horizontal capillary water ripples (ONLY used on water!)
    rip1 = resize_bicubic(rng.standard_normal((H // 5, W // 48)).astype(np.float32), W, H)
    rip2 = resize_bicubic(rng.standard_normal((H // 3, W // 22)).astype(np.float32), W, H)
    rip = 0.62 * rip1 + 0.38 * rip2

    ring_amp = np.zeros((H, W), dtype=np.float32)
    for rx0, ry0, r_rad in ((2.3, 6.2, 0.65), (3.4, 9.8, 0.82), (2.7, 13.5, 0.75)):
        d_r = np.hypot(wx - rx0, wy - ry0).astype(np.float32)
        ring_amp += np.sin(d_r * 28.0) * np.exp(-d_r / r_rad) * 0.65

    dist_t = np.clip((yy.astype(np.float32) - float(cam.horizon_y)) / max(1.0, H - float(cam.horizon_y)), 0.0, 1.0)
    rip_scale = (1.5 + 6.2 * dist_t).astype(np.float32)
    map_x = xx.astype(np.float32) + (rip * 0.85 + ring_amp * 0.45) * rip_scale
    map_y = (2.0 * float(cam.horizon_y) - yy.astype(np.float32)) + (rip * 1.15 + ring_amp * 0.75) * rip_scale

    refl_bg = remap_bilinear(ref_backdrop, map_x, np.clip(map_y, 0.0, H - 1.001))
    mx_3d = xx.astype(np.float32) + (rip * 0.75 + ring_amp * 0.45) * rip_scale
    my_3d = yy.astype(np.float32) + (rip * 0.65 + ring_amp * 0.35) * rip_scale
    refl_3d = remap_bilinear(mirrored_3d_rgba, mx_3d, my_3d)
    refl_combined = over_premul(refl_bg, refl_3d) * np.array([0.82, 0.90, 0.82], dtype=np.float32)[None, None, :]

    bed_col = np.array([0.046, 0.115, 0.086], dtype=np.float32)[None, None, :] + (mottle_full * 0.018)[..., None] * np.array(
        [0.45, 0.85, 0.60], dtype=np.float32
    )
    fresnel = 0.28 + (0.88 - 0.28) * np.power(1.0 - dist_t, 1.4)
    water_rgb = bed_col * (1.0 - fresnel[..., None]) + refl_combined * fresnel[..., None]

    glint = np.clip((rip + ring_amp * 0.8 - 1.02) * 2.2, 0.0, 1.0) * np.exp(-((xx / W - 0.74) / 0.22) ** 2).astype(np.float32)
    water_rgb += glint[..., None] * np.array([0.44, 0.46, 0.26], dtype=np.float32)
    ring_hi = np.clip(ring_amp - 0.18, 0.0, 1.0) * 0.16
    water_rgb += ring_hi[..., None] * np.array([0.78, 0.84, 0.68], dtype=np.float32)

    fog_t = 1.0 - np.exp(-np.maximum(0.0, t_w.astype(np.float32) - 6.5) * 0.048)
    water_rgb = water_rgb * (1.0 - fog_t[..., None]) + fog_rgb[None, None, :] * fog_t[..., None]
    return canvas * (1.0 - pond_mask[..., None]) + water_rgb * pond_mask[..., None]


# =============================================================================
# Stage 2 & 3 — Diagonal Morning Volumetric Sunbeams Aligned With SUN_DIR
# =============================================================================

def paint_morning_sunbeams(cam: Camera3D, rng: np.random.Generator, intensity: float = 0.48, front: bool = False) -> np.ndarray:
    """Golden-cream diagonal morning sunbeams strictly parallel to projected 3D SUN_DIR."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    p_ref = np.array([-0.6, 5.5, 1.8], dtype=np.float64)
    u0, v0, _ = cam.project(p_ref)
    u1, v1, _ = cam.project(p_ref + SUN_DIR * 3.0)
    dx, dy = float(u1 - u0), float(v1 - v0)
    dl = math.hypot(dx, dy) + 1e-6
    rx, ry = dx / dl, dy / dl
    nx, ny = -ry, rx

    s_coord = (xx - W * 0.62) * rx + (yy - H * 0.10) * ry
    p_coord = (xx - W * 0.62) * nx + (yy - H * 0.10) * ny

    acc = np.zeros((H, W), dtype=np.float32)
    shafts = (
        [(-360, 62, 0.46), (-190, 56, 0.62), (-40, 74, 0.72), (115, 64, 0.58), (275, 82, 0.40)]
        if not front
        else [(-260, 42, 0.40), (-90, 48, 0.56), (60, 40, 0.48), (195, 52, 0.34)]
    )
    for cp, wp, st in shafts:
        w_eff = wp * (1.0 + np.maximum(0.0, s_coord) / (H * 1.45))
        beam = np.exp(-0.5 * ((p_coord - cp) / w_eff) ** 2)
        vert = np.clip((s_coord + 340.0) / 360.0, 0.0, 1.0) * np.exp(-np.maximum(0.0, s_coord) / (H * 1.12))
        acc += beam * vert * st

    turb = 0.80 + 0.20 * resize_bicubic(fbm_gouache(H // 4, W // 4, rng), W, H)
    acc = np.clip(acc * turb * intensity, 0.0, 1.0)
    beam_col = np.array([0.96, 0.94, 0.70], dtype=np.float32) if front else np.array([0.85, 0.90, 0.58], dtype=np.float32)
    return acc[..., None] * beam_col[None, None, :]


# =============================================================================
# Stage 4 — Foreground Circular-Bokeh Kyoto Maple Branch & 35mm Optics
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


def draw_acer_palmatum_leaf(
    ctx: cairo.Context,
    cx: float,
    cy: float,
    radius: float,
    rot: float,
    srng: np.random.Generator,
    backlit: float,
) -> None:
    """Authentic Kyoto Acer palmatum (iroha-momiji) leaf: slender long upper lobes, tiny basal lobes."""
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(rot)

    # Authentic Acer palmatum proportions: 5 graceful slender main fingers + 2 tiny backward spurs
    lobe_angles = [-2.05, -1.22, -0.58, 0.0, 0.58, 1.22, 2.05]
    lobe_scales = [0.34, 0.66, 0.92, 1.12, 0.92, 0.66, 0.34]

    # Deep mossy forest olive transitioning to backlit warm chartreuse-gold (#c5d988) tips!
    c_base = (0.048 + 0.14 * backlit, 0.125 + 0.24 * backlit, 0.058 + 0.07 * backlit, 0.97)
    c_mid = (0.14 + 0.34 * backlit, 0.28 + 0.36 * backlit, 0.09 + 0.14 * backlit, 0.96)
    c_tip = (0.34 + 0.48 * backlit, 0.48 + 0.40 * backlit, 0.14 + 0.24 * backlit, 0.94)
    g = cairo.RadialGradient(0, 0, radius * 0.08, radius * 0.22, 0, radius * 1.15)
    g.add_color_stop_rgba(0.0, *c_base)
    g.add_color_stop_rgba(0.55, *c_mid)
    g.add_color_stop_rgba(1.0, *c_tip)
    ctx.set_source(g)

    ctx.move_to(0.0, 0.0)
    for idx_l, (a0, sc) in enumerate(zip(lobe_angles, lobe_scales)):
        a = a0 + float(srng.uniform(-0.05, 0.05))
        ln = radius * sc * float(srng.uniform(0.90, 1.10))
        w_lobe = ln * (0.145 if sc > 0.6 else 0.19)
        ca, sa = math.cos(a), math.sin(a)
        px, py = -sa, ca
        tipx, tipy = ca * ln, sa * ln
        # Deep graceful sinus cut nearly back to palm center (classic Japanese maple silhouette!)
        sinus_r = radius * 0.18
        next_a = lobe_angles[idx_l + 1] if idx_l + 1 < len(lobe_angles) else a0 + 0.6
        mid_a = 0.5 * (a + next_a)
        ctx.curve_to(ca * ln * 0.38 - px * w_lobe * 1.15, sa * ln * 0.38 - py * w_lobe * 1.15, ca * ln * 0.74 - px * w_lobe * 0.65, sa * ln * 0.74 - py * w_lobe * 0.65, tipx, tipy)
        ctx.curve_to(ca * ln * 0.74 + px * w_lobe * 0.65, sa * ln * 0.74 + py * w_lobe * 0.65, ca * ln * 0.38 + px * w_lobe * 1.15, sa * ln * 0.38 + py * w_lobe * 1.15, math.cos(mid_a) * sinus_r, math.sin(mid_a) * sinus_r)
    ctx.close_path()
    ctx.fill()

    # Delicate backlit central veins
    ctx.set_source_rgba(0.78, 0.88, 0.48, 0.32 + 0.32 * backlit)
    ctx.set_line_width(max(0.9, radius * 0.020))
    for a0, sc in zip(lobe_angles[1:-1], lobe_scales[1:-1]):
        ctx.move_to(0, 0)
        ctx.line_to(math.cos(a0) * radius * sc * 0.78, math.sin(a0) * radius * sc * 0.78)
        ctx.stroke()
    ctx.restore()


def paint_foreground_bokeh_maple_branch(rng: np.random.Generator) -> np.ndarray:
    """Stage 4: Graceful near-camera Kyoto maple branch framing upper-right corner with r=7.8 disc bokeh."""
    surf, ctx = new_cairo_2x()

    main_bough = [(W + 40, 105), (W - 135, 145), (W - 295, 205), (W - 465, 222), (W - 585, 265)]
    sub_bough_1 = [(W - 175, 158), (W - 255, 82), (W - 375, 48), (W - 475, 65)]
    sub_bough_2 = [(W - 305, 205), (W - 375, 295), (W - 445, 365)]
    sub_bough_3 = [(W - 65, 130), (W - 115, 268), (W - 185, 390), (W - 230, 478)]

    for pts, bw in ((main_bough, 14.5), (sub_bough_1, 9.5), (sub_bough_2, 7.8), (sub_bough_3, 10.5)):
        stroke_catmull_rom(ctx, pts, base_w=bw, z_depth=0.9, rgba=(0.072, 0.058, 0.044, 0.98), rng=rng, taper=True, tremor=0.50)

    # Naturally paired drooping Kyoto maple sprays hanging downward/inward from twigs
    leaf_clusters = [
        (W - 575, 268, 72, 2.25, 0.86),
        (W - 515, 225, 82, 2.65, 0.74),
        (W - 425, 200, 76, 1.85, 0.58),
        (W - 440, 355, 74, 1.70, 0.88),
        (W - 365, 292, 82, 2.08, 0.54),
        (W - 465, 68, 68, 2.55, 0.78),
        (W - 355, 55, 78, 2.85, 0.46),
        (W - 245, 95, 86, 2.20, 0.34),
        (W - 280, 210, 88, 1.60, 0.30),
        (W - 150, 160, 92, 2.35, 0.24),
        (W - 135, 278, 84, 1.50, 0.42),
        (W - 195, 392, 76, 1.82, 0.68),
        (W - 228, 478, 66, 1.42, 0.84),
        (W - 65, 225, 92, 1.75, 0.26),
        (W - 52, 358, 86, 1.32, 0.38),
        (W - 88, 62, 92, 2.60, 0.22),
        (W - 615, 220, 56, 2.75, 0.92),
        (W - 318, 138, 64, 2.00, 0.66),
    ]
    for i, (lx, ly, lr, lrot, bl) in enumerate(leaf_clusters):
        srng = np.random.default_rng(71000 + i * 31)
        ctx.set_source_rgba(0.11, 0.12, 0.07, 0.88)
        ctx.set_line_width(2.0)
        ctx.move_to(lx - math.cos(lrot) * 20, ly - math.sin(lrot) * 20)
        ctx.line_to(lx, ly)
        ctx.stroke()
        draw_acer_palmatum_leaf(ctx, lx, ly, lr, lrot, srng, bl)

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
    tint = np.array([1.05, 1.04, 0.86], dtype=np.float32)
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
    img *= np.clip(1.0 - 0.21 * (r2 ** 1.15), 0.72, 1.0)[..., None]

    # 3. Linear shadow black lift + C1-continuous tanh highlight shoulder (knee=0.80, ceiling=0.965)
    img = 0.015 + img * 0.985
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
    seed: int = 202601,
    audit_dir: pathlib.Path | None = None,
    export_webps: list[pathlib.Path] | None = None,
) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # Stage 1: Off-axis Camera3D (VP ~61.4% width, engawa mass on left, misty pond on right)
    cam = Camera3D(W, H, eye=(0.35, 0.0, 1.54), target=(-1.10, 11.5, 1.24), fov_x_deg=58.0)
    print(f"[matcha-01] Camera VP: ({cam.vp_x:.1f}, {cam.vp_y:.1f}) -> {cam.vp_x / W * 100:.1f}% width")

    # Sky & three painterly misty Arashiyama cedar/pine ridges
    canvas, fog_rgb, m_global, d_global = paint_sky_and_mountain_ridges(cam, rng)

    # Bamboo grove rooted firmly on the mossy far & right pond banks!
    far_bamboos = [
        (np.array([2.2, 19.8, Z_GARDEN + 0.22]), 6.8, 0.062, 0.85, -0.15, 101),
        (np.array([3.6, 19.2, Z_GARDEN + 0.30]), 7.4, 0.066, -1.15, 0.15, 102),
        (np.array([4.8, 18.4, Z_GARDEN + 0.32]), 7.6, 0.068, -1.45, -0.20, 103),
        (np.array([5.9, 17.8, Z_GARDEN + 0.34]), 7.1, 0.064, -1.10, 0.12, 104),
        (np.array([7.1, 17.2, Z_GARDEN + 0.38]), 7.8, 0.070, -1.65, -0.18, 105),
        (np.array([8.4, 16.8, Z_GARDEN + 0.36]), 7.2, 0.064, -1.30, 0.10, 106),
        (np.array([9.6, 16.5, Z_GARDEN + 0.35]), 7.5, 0.068, -1.55, -0.12, 107),
    ]
    mid_bamboos = [
        (np.array([6.0, 13.8, Z_GARDEN + 0.28]), 7.2, 0.068, -1.45, -0.15, 201),
        (np.array([6.8, 12.8, Z_GARDEN + 0.26]), 7.6, 0.072, -1.75, 0.18, 202),
        (np.array([5.3, 14.2, Z_GARDEN + 0.24]), 6.6, 0.062, -1.05, -0.10, 203),
        (np.array([7.6, 11.8, Z_GARDEN + 0.24]), 7.8, 0.074, -1.85, -0.22, 204),
        (np.array([6.4, 11.0, Z_GARDEN + 0.20]), 6.8, 0.064, -1.25, 0.14, 205),
    ]
    boulders = [
        (np.array([1.95, 11.8, Z_POND + 0.02]), 0.58, 0.38),
        (np.array([2.38, 11.2, Z_POND + 0.02]), 0.42, 0.26),
        (np.array([1.68, 7.6, Z_POND + 0.02]), 0.46, 0.28),
    ]
    # Stone snow-viewing lantern perched prominently on the mossy rock island at (4.05, 12.9)
    lantern_pos = np.array([4.05, 12.9, Z_POND + 0.26], dtype=np.float64)

    # Stage 3 Band 0 (Far Pond Bank, Mossy Embankment, Distant Bamboo Grove & Snow-Viewing Lantern)
    s_far, c_far = new_cairo_2x()
    draw_far_mossy_pond_bank(c_far, cam, rng, fog_rgb, mirrored=False)
    for bp, bh, br, ax, ay, bseed in far_bamboos:
        draw_bamboo_stalk(c_far, cam, bp, bh, br, ax, ay, bseed, fog_rgb, mirrored=False)
    draw_stone_garden_lantern(c_far, cam, lantern_pos, 0.90, rng, fog_rgb, mirrored=False)
    canvas = over_premul(canvas, finish_cairo_2x(s_far))

    # Snapshot horizon reflection source BEFORE painting water
    ref_backdrop = canvas.copy()

    # Dedicated 3D plane-mirrored quads (mossy bank, lantern, bamboo grove, boulders, veranda pillars & eaves)
    s_ref, c_ref = new_cairo_2x()
    draw_far_mossy_pond_bank(c_ref, cam, rng, fog_rgb, mirrored=True)
    for bp, bh, br, ax, ay, bseed in far_bamboos + mid_bamboos:
        draw_bamboo_stalk(c_ref, cam, bp, bh, br, ax, ay, bseed, fog_rgb, mirrored=True)
    for bpos, brx, brz in boulders:
        draw_mossy_boulder(c_ref, cam, bpos, brx, brz, rng, fog_rgb, mirrored=True)
    draw_stone_garden_lantern(c_ref, cam, lantern_pos, 0.90, rng, fog_rgb, mirrored=True)
    for py_c in PILLAR_YS:
        draw_veranda_pillar(c_ref, cam, py_c, rng, fog_rgb, mirrored=True)
    eave_q = [
        np.array([X_VERANDA_EDGE - 0.5, 2.5, float(eave_height_at_x(X_VERANDA_EDGE - 0.5))]),
        np.array([0.45, 2.5, float(eave_height_at_x(0.45))]),
        np.array([0.45, 16.5, float(eave_height_at_x(0.45))]),
        np.array([X_VERANDA_EDGE - 0.5, 16.5, float(eave_height_at_x(X_VERANDA_EDGE - 0.5))]),
    ]
    fill_3d_quad(ctx=c_ref, cam=cam, pts3=mirror_pts_across_plane(eave_q, Z_POND), rgba=(0.08, 0.11, 0.08, 0.52))
    mirrored_3d = finish_cairo_2x(s_ref)

    # Stage 2: Mirror-calm pond with capillary-rippled Fresnel reflections
    canvas = paint_pond_fresnel_water(canvas, ref_backdrop, mirrored_3d, cam, rng, fog_rgb, m_global)

    # Stage 1: Planar UV homography surfaces (Voronoi mossy stones, shoji wall, fascia, deck planks)
    stones_uv = paint_voronoi_mossy_stones_uv(rng)
    canvas = cam.warp_plane_over(
        canvas,
        stones_uv,
        p0=np.array([-0.72, 1.0, Z_GARDEN]),
        e1=np.array([2.85 - (-0.72), 0.0, 0.0]),
        e2=np.array([0.0, 21.0, 0.0]),
        fog_rgb=fog_rgb,
        fog_start=6.0,
        fog_density=0.052,
    )

    shoji_uv = paint_shoji_wall_uv(rng)
    canvas = cam.warp_plane_over(
        canvas,
        shoji_uv,
        p0=np.array([X_WALL, 0.5, Z_DECK]),
        e1=np.array([0.0, 15.5, 0.0]),
        e2=np.array([0.0, 0.0, 3.12 - Z_DECK]),
        fog_rgb=fog_rgb,
        fog_start=6.5,
        fog_density=0.048,
    )

    fascia_uv = paint_veranda_fascia_uv(rng)
    canvas = cam.warp_plane_over(
        canvas,
        fascia_uv,
        p0=np.array([X_VERANDA_EDGE, 0.5, Z_GARDEN]),
        e1=np.array([0.0, 15.5, 0.0]),
        e2=np.array([0.0, 0.0, Z_DECK - Z_GARDEN]),
        fog_rgb=fog_rgb,
        fog_start=6.5,
        fog_density=0.048,
    )

    deck_uv = paint_veranda_deck_uv(rng)
    canvas = cam.warp_plane_over(
        canvas,
        deck_uv,
        p0=np.array([X_WALL, 0.5, Z_DECK]),
        e1=np.array([X_VERANDA_EDGE - X_WALL, 0.0, 0.0]),
        e2=np.array([0.0, 15.5, 0.0]),
        fog_rgb=fog_rgb,
        fog_start=6.5,
        fog_density=0.046,
    )

    # Stage 3 Band 1 (Midground Arching Bamboo Grove, Smooth Mossy Pond Boulders & Far Veranda Pillars)
    s_mid, c_mid = new_cairo_2x()
    for bp, bh, br, ax, ay, bseed in mid_bamboos:
        draw_bamboo_stalk(c_mid, cam, bp, bh, br, ax, ay, bseed, fog_rgb, mirrored=False)
    for bpos, brx, brz in boulders:
        draw_mossy_boulder(c_mid, cam, bpos, brx, brz, rng, fog_rgb, mirrored=False)
    for py_c in reversed(PILLAR_YS[1:]):
        draw_veranda_pillar(c_mid, cam, py_c, rng, fog_rgb, mirrored=False)
    mid_1x = finish_cairo_2x(s_mid)
    ma = mid_1x[:, :, 3:4]
    mid_1x[:, :, :3] = np.clip(mid_1x[:, :, :3] + ma * (m_global * 0.018 + d_global * 0.012)[..., None], 0.0, 1.25)
    canvas = over_premul(canvas, mid_1x)

    # Overhead pitched slatted cedar eaves warped cleanly over pillars
    eaves_uv = paint_slatted_eaves_uv(rng)
    x0_e = X_WALL - 0.1
    x1_e = 0.45
    canvas = cam.warp_plane_over(
        canvas,
        eaves_uv,
        p0=np.array([x0_e, 0.2, float(eave_height_at_x(x0_e))]),
        e1=np.array([x1_e - x0_e, 0.0, float(eave_height_at_x(x1_e) - eave_height_at_x(x0_e))]),
        e2=np.array([0.0, 16.3, 0.0]),
        fog_rgb=fog_rgb,
        fog_start=7.5,
        fog_density=0.042,
    )

    # Back diagonal golden-cream volumetric morning sunbeams aligned with SUN_DIR
    rays_back = paint_morning_sunbeams(cam, rng, intensity=0.46, front=False)
    canvas = 1.0 - (1.0 - np.clip(canvas, 0.0, 1.0)) * (1.0 - np.clip(rays_back, 0.0, 0.85))

    # Stage 3 Band 2 (Near Cedar Pillar, Hero Low Cedar Bench & Ceramic Matcha Bowl + Floating Fallen Leaves)
    s_hero, c_hero = new_cairo_2x()
    draw_veranda_pillar(c_hero, cam, PILLAR_YS[0], rng, fog_rgb, mirrored=False)
    draw_hero_bench_and_matcha_bowl(c_hero, cam, rng, fog_rgb)

    for fx, fy, fz, frad, frot in (
        (2.15, 5.8, Z_POND + 0.005, 0.055, 0.4),
        (2.85, 7.1, Z_POND + 0.005, 0.050, 1.8),
        (3.55, 8.9, Z_POND + 0.005, 0.052, -0.7),
        (2.45, 9.6, Z_POND + 0.005, 0.048, 2.3),
        (-0.85, 5.1, Z_DECK + 0.005, 0.046, 0.9),
        (-0.92, 6.4, Z_DECK + 0.005, 0.044, -1.2),
    ):
        fu, fv, fz_c = cam.project(np.array([fx, fy, fz]))
        if fz_c > Z_NEAR:
            r_px = max(3.5, frad * cam.K[0, 0] / float(fz_c))
            c_hero.save()
            c_hero.translate(float(fu), float(fv))
            c_hero.scale(1.0, 0.36)
            draw_acer_palmatum_leaf(c_hero, 0.0, 0.0, r_px, frot, np.random.default_rng(int(fy * 1000)), backlit=0.78)
            c_hero.restore()

    hero_1x = finish_cairo_2x(s_hero)
    h_tooth = paper_tooth(H, W, rng)
    ha = hero_1x[:, :, 3:4]
    hero_1x[:, :, :3] = np.clip(hero_1x[:, :, :3] + ha * (m_global * 0.014 + h_tooth * 0.006)[..., None], 0.0, 1.25)
    canvas = over_premul(canvas, hero_1x)

    # Soft Gaussian-blurred translucent steam rising organically from the hot ceramic matcha bowl!
    steam_rgba = paint_soft_matcha_steam(cam, rng)
    canvas = over_premul(canvas, steam_rgba)

    # Front subtle morning sunbeams & sunlit drifting forest motes
    rays_front = paint_morning_sunbeams(cam, rng, intensity=0.30, front=True)
    canvas = 1.0 - (1.0 - np.clip(canvas, 0.0, 1.0)) * (1.0 - np.clip(rays_front, 0.0, 0.80))

    s_motes, c_motes = new_cairo_2x()
    for _ in range(95):
        mx = float(rng.uniform(120, W - 120))
        my = float(rng.uniform(140, H - 120))
        mr = float(rng.uniform(0.9, 2.2))
        ma_v = float(rng.uniform(0.18, 0.54)) * (0.55 + 0.45 * math.exp(-((mx / W - 0.55) / 0.35) ** 2))
        c_motes.set_source_rgba(0.96, 0.96, 0.78, ma_v)
        c_motes.arc(mx, my, mr, 0, math.tau)
        c_motes.fill()
    canvas = over_premul(canvas, finish_cairo_2x(s_motes))

    # Stage 4: Foreground upper-right circular-bokeh Kyoto maple branch + 35mm analog finish
    fg_bokeh = paint_foreground_bokeh_maple_branch(rng)
    canvas = over_premul(canvas, fg_bokeh)

    final_rgb = apply_35mm_finish(canvas, rng)
    elapsed = time.perf_counter() - t0
    print(f"[matcha-01] Rendered {W}x{H} in {elapsed:.2f}s")

    u8 = np.round(np.clip(final_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    full_im = Image.fromarray(u8, mode="RGB")

    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        full_im.save(audit_dir / "w01_full.png", optimize=True)
        crops = {
            "w01_crop_aperture.png": (int(W * 0.52), int(H * 0.05)),       # Slatted cedar eaves, Arashiyama pine ridge & upper-right maple bokeh
            "w01_crop_midground.png": (int(W * 0.52), int(H * 0.44)),      # Mossy garden stones, boulders, snow-viewing lantern & Fresnel pond
            "w01_crop_foreground.png": (int(W * 0.14), H - 450),           # Hero cedar bench, ceramic matcha bowl, soft steam & sunlit planks
        }
        for cname, (cx0, cy0) in crops.items():
            x0 = max(0, min(W - 600, cx0))
            y0 = max(0, min(H - 450, cy0))
            full_im.crop((x0, y0, x0 + 600, y0 + 450)).save(audit_dir / cname)

    if export_webps:
        for wp in export_webps:
            wp.parent.mkdir(parents=True, exist_ok=True)
            full_im.save(wp, format="WEBP", quality=90, method=6)
            print(f"[matcha-01] Exported {wp} ({wp.stat().st_size / 1024:.1f} KiB)")

    return final_rgb, {"elapsed": elapsed, "vp_ratio": cam.vp_x / W}


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    default_webps = [
        repo_root / "session" / "themes" / "matcha" / "walls" / "wall-01.webp",
        repo_root / "docs" / "assets" / "walls" / "matcha-01.webp",
    ]
    ap = argparse.ArgumentParser(description='Render Matcha 01 — "Arashiyama Tea Veranda" (2048x1152)')
    ap.add_argument("--seed", type=int, default=202601)
    ap.add_argument("--audit-dir", type=pathlib.Path, default=None)
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()
    render_scene(seed=args.seed, audit_dir=args.audit_dir, export_webps=None if args.no_export else default_webps)


if __name__ == "__main__":
    main()

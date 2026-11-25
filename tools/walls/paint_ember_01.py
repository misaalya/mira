#!/usr/bin/env python3
"""
Ember 01 ("Crimson Harbor Pier") — Pure-Python 3D-Grounded 1990s Anime Scene Pipeline (2048 x 1152)

Five-stage deterministic painting pipeline (zero image models, zero external assets):
  Stage 1: Metric right-handed Camera3D along a rain-slicked harbor pier (off-axis
           vanishing point ~59% width), exact 3x3 UV-to-screen plane homographies
           masked by valid UV and exact ray-plane depth z_cam > 0.05 with exponential
           atmospheric haze, near-plane 3D line clipping, and irregular Voronoi
           (d2 - d1) wet basalt seawall & quay stones alongside weathered timber planks.
  Stage 2: Single locked normalized low-sun SUN_DIR (z < 0) tracing pier planks,
           basalt quay, and warehouse wall backward along -SUN_DIR (6 height-refinement
           steps for pitched eaves) for unified cast shadows, volumetric golden-hour
           haze rays, and Fresnel capillary-rippled molten-copper bay reflections
           combining mirrored horizon sky/headland/lighthouse with 3D plane-mirrored
           seawall, pilings, and moored 3/4 Japanese wooden coastal boat.
  Stage 3: 4 depth-interleaved 2x supersampled PyCairo bands (Lanczos-downsampled)
           featuring distant headland lighthouse, 3/4 moored wooden harbor boat,
           camera-aligned cylindrical timber pilings, charred-cedar wooden harbor
           warehouse eaves & glowing amber windows/lanterns, cast-iron mooring
           bollards with chunky braided hemp ropes, slatted crates, and distant
           gliding gulls — broken up by isotropic fbm_gouache, flat-brush dabs,
           cold-press paper tooth, and warm shadow bounce (lum < 0.28).
  Stage 4: Foreground lower-right wooden mooring post & wrapped rope silhouette
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

# Stage 2: Single locked normalized low-angle sunset travel vector (z < 0)
_raw_sun = np.array([-0.34, -0.905, -0.255], dtype=np.float64)
SUN_DIR = _raw_sun / np.linalg.norm(_raw_sun)

# World vertical geometry (meters)
Z_WATER = 0.0
Z_PIER = 1.16

# Left wooden harbor warehouse geometry (shared by Stage 2 shadow solver & Stage 3 eaves)
WAREHOUSE_X_WALL = -2.55
WAREHOUSE_X_EAVE = -1.05
WAREHOUSE_Y0 = 1.4
WAREHOUSE_Y1 = 22.0
WAREHOUSE_Z_EAVE = 3.98
WAREHOUSE_ROOF_SLOPE = 0.44

# Cast-iron mooring bollards along the basalt quay curb (x, y)
BOLLARD_COORDS = [
    (0.50, 4.75),
    (0.50, 9.20),
    (0.50, 14.30),
    (0.50, 20.10),
    (0.50, 26.40),
]

# Warm hanging amber bulkhead lamps mounted cleanly between windows on warehouse wall
LANTERN_LIGHTS = [
    (-2.08, 4.45, 3.25, 1.30),
    (-2.08, 9.55, 3.25, 1.20),
    (-2.08, 15.10, 3.25, 1.10),
]

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_THEME_WEBP = REPO_ROOT / "session" / "themes" / "ember" / "walls" / "wall-01.webp"
DEFAULT_DOCS_WEBP = REPO_ROOT / "docs" / "assets" / "walls" / "ember-01.webp"


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


def apply_warm_shadow_bounce(rgb: np.ndarray, strength: float = 0.020) -> np.ndarray:
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    mask = np.clip((0.28 - lum) / 0.28, 0.0, 1.0)[..., None]
    bounce = np.array([1.28, 0.54, 0.30], dtype=np.float32) * strength
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
        fog_start: float = 7.5,
        fog_density: float = 0.036,
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


def mirror_pts_plane(pts: np.ndarray, p0: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Exact 3D plane reflection: pts - 2 * dot(pts - p0, n) * n."""
    d = np.sum((pts - p0) * n, axis=-1, keepdims=True)
    return pts - 2.0 * d * n


# =============================================================================
# Stage 2 — Unified 3D Sunset Shadow Solver & Lantern Illumination
# =============================================================================

def trace_eave_shadow(wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> np.ndarray:
    """
    Trace any 3D point backward along -SUN_DIR onto the pitched overhanging warehouse
    eave roof using 6 fixed height-refinement steps.
    """
    inv_sz = np.float32(1.0 / (-SUN_DIR[2]))
    sx = np.float32(SUN_DIR[0])
    sy = np.float32(SUN_DIR[1])
    z_roof = np.full_like(wx, WAREHOUSE_Z_EAVE, dtype=np.float32)
    for _ in range(6):
        t = np.maximum(0.0, (z_roof - wz) * inv_sz)
        rx = wx - sx * t
        z_roof = WAREHOUSE_Z_EAVE + np.maximum(0.0, (WAREHOUSE_X_EAVE - rx)) * np.float32(WAREHOUSE_ROOF_SLOPE)
    t = np.maximum(0.0, (z_roof - wz) * inv_sz)
    rx = wx - sx * t
    ry = wy - sy * t
    under_roof = (wz < z_roof - 0.02) & (rx >= -5.6) & (rx <= WAREHOUSE_X_EAVE) & (ry >= WAREHOUSE_Y0) & (ry <= WAREHOUSE_Y1)
    edge_soft = np.clip((rx - (WAREHOUSE_X_EAVE - 0.18)) / 0.18, 0.0, 1.0)
    return np.where(under_roof, 0.22 + 0.78 * edge_soft, 1.0).astype(np.float32)


def trace_pier_shadows(wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> np.ndarray:
    """Unified cast shadows on pier planks & basalt quay from warehouse eaves + mooring bollards."""
    vis = trace_eave_shadow(wx, wy, wz)
    sx = np.float32(-SUN_DIR[0])
    sy = np.float32(-SUN_DIR[1])
    sz = np.float32(-SUN_DIR[2])
    s2 = sx * sx + sy * sy
    inv_s = np.float32(1.0 / math.sqrt(float(s2)))
    for bx, by in BOLLARD_COORDS:
        dx = np.float32(bx) - wx
        dy = np.float32(by) - wy
        t_h = (dx * sx + dy * sy) / s2
        perp = np.abs(dx * sy - dy * sx) * inv_s
        z_hit = wz + t_h * sz
        in_bol = (t_h > 0.0) & (z_hit >= Z_PIER - 0.02) & (z_hit <= Z_PIER + 0.56) & (perp < 0.21)
        soft = np.clip((perp - 0.12) / 0.09, 0.0, 1.0)
        vis = np.minimum(vis, np.where(in_bol, 0.26 + 0.74 * soft, 1.0).astype(np.float32))
    wall_ao = np.clip((wx - np.float32(WAREHOUSE_X_WALL)) / 0.65, 0.0, 1.0)
    return vis * (0.42 + 0.58 * wall_ao)


def eval_lantern_glow(wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> np.ndarray:
    acc = np.zeros_like(wx, dtype=np.float32)
    for lx, ly, lz, inten in LANTERN_LIGHTS:
        d2 = (wx - lx) ** 2 + (wy - ly) ** 2 + (wz - lz) ** 2
        acc += np.float32(inten) / (1.0 + 0.34 * d2)
    return np.clip(acc, 0.0, 1.6)


# =============================================================================
# Stage 1 & 2 — Crimson Golden-Hour Sky, Painterly Clouds, Volcanic Headland & Copper Bay
# =============================================================================

def paint_sky_headland_and_copper_bay(
    cam: Camera3D,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Paint a lush 1990s anime golden-hour sunset sky with sweeping painterly backlit
    stratocumulus cloud decks, rugged volcanic headland & distant lighthouse beacon,
    and Fresnel capillary-rippled molten-copper harbor bay water.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = xx / W
    v = yy / H
    hy = float(cam.horizon_y)
    hv = hy / H

    mottle = fbm_gouache(H, W, rng)
    dabs = flat_brush_dabs(H, W, rng, count=90)
    tooth = paper_tooth(H, W, rng)

    # Low sun disk aligned with -SUN_DIR projected onto open right-center sky
    sun_p = cam.K @ (cam.R @ (-SUN_DIR))
    sun_u = float(np.clip(sun_p[0] / sun_p[2] / W, 0.61, 0.70))
    sun_v = float(np.clip(sun_p[1] / sun_p[2] / H, hv - 0.105, hv - 0.055))

    # Rich Ember sunset gradient: deep obsidian-plum zenith -> crimson -> vermilion -> molten amber horizon
    t_sky = np.clip(v / max(0.05, hv), 0.0, 1.0)
    c_zenith = np.array([0.058, 0.022, 0.044], dtype=np.float32)
    c_plum = np.array([0.210, 0.055, 0.088], dtype=np.float32)
    c_vermil = np.array([0.745, 0.240, 0.155], dtype=np.float32)
    c_amber = np.array([0.990, 0.585, 0.245], dtype=np.float32)
    c_horiz = np.array([1.070, 0.815, 0.475], dtype=np.float32)

    s1 = np.clip(t_sky / 0.36, 0.0, 1.0)[..., None]
    s2 = np.clip((t_sky - 0.32) / 0.34, 0.0, 1.0)[..., None]
    s3 = np.clip((t_sky - 0.64) / 0.24, 0.0, 1.0)[..., None]
    s4 = np.clip((t_sky - 0.85) / 0.15, 0.0, 1.0)[..., None]

    sky = c_zenith * (1.0 - s1) + c_plum * s1
    sky = sky * (1.0 - s2) + c_vermil * s2
    sky = sky * (1.0 - s3) + c_amber * s3
    sky = sky * (1.0 - s4) + c_horiz * s4

    # Blazing sunset solar halo & core
    du = (u - sun_u) * (W / H)
    dv = v - sun_v
    d_sun = np.hypot(du, dv)
    sun_halo_w = np.exp(-(d_sun / 0.44) ** 1.55).astype(np.float32)
    sun_halo_m = np.exp(-(d_sun / 0.14) ** 1.75).astype(np.float32)
    sun_core = np.clip((0.027 - d_sun) / 0.010, 0.0, 1.0).astype(np.float32)

    sky += sun_halo_w[..., None] * np.array([0.34, 0.145, 0.055], dtype=np.float32)
    sky += sun_halo_m[..., None] * np.array([0.52, 0.35, 0.14], dtype=np.float32)
    sky += sun_core[..., None] * np.array([0.70, 0.60, 0.44], dtype=np.float32)

    # Sweeping 1990s anime hand-painted stratocumulus sunset cloud banks (rich multi-scale gouache!)
    c_billow_big = resize_bicubic(rng.standard_normal((16, 26)).astype(np.float32), W, H)
    c_billow_mid = resize_bicubic(rng.standard_normal((36, 58)).astype(np.float32), W, H)
    c_billow_sm = resize_bicubic(rng.standard_normal((76, 124)).astype(np.float32), W, H)
    c_billow_fine = resize_bicubic(rng.standard_normal((148, 240)).astype(np.float32), W, H)
    cloud_field = 0.48 * c_billow_big + 0.28 * c_billow_mid + 0.16 * c_billow_sm + 0.08 * c_billow_fine
    cloud_field = (cloud_field - cloud_field.mean()) / (float(cloud_field.std()) + 1e-6)

    # Three graceful diagonal perspective cloud decks fanning softly away from the low sunset
    persp_ang = (u - sun_u) * 0.15
    deck1 = np.exp(-((t_sky - 0.22 - persp_ang * 0.85 + 0.052 * c_billow_big) / 0.145) ** 2)
    deck2 = np.exp(-((t_sky - 0.48 - persp_ang * 0.60 + 0.036 * c_billow_big) / 0.110) ** 2)
    deck3 = np.exp(-((t_sky - 0.72 - persp_ang * 0.30 + 0.020 * c_billow_big) / 0.062) ** 2)

    raw_c = (deck1 * 0.88 + deck2 * 0.94 + deck3 * 0.74) * np.clip(0.58 + 0.54 * cloud_field, 0.0, 1.38)
    cloud_a = np.clip((raw_c - 0.22) / 0.56, 0.0, 1.0) * np.clip((hv - 0.015 - v) / 0.048, 0.0, 1.0)
    # Soft natural bottom-edge solar rim lighting (gentle gradient rather than bright outline rings)
    c_shift = np.roll(np.roll(cloud_a, -18, axis=0), -int(round((sun_u - 0.5) * 22)), axis=1)
    cloud_rim = np.clip((cloud_a - c_shift * 0.82) * 1.65, 0.0, 1.0) * (0.40 + 0.90 * sun_halo_w)
    c_body_dark = np.array([0.115, 0.038, 0.060], dtype=np.float32)[None, None, :]
    c_body_mid = np.array([0.50, 0.155, 0.125], dtype=np.float32)[None, None, :]
    c_rim_gold = np.array([1.06, 0.64, 0.29], dtype=np.float32)[None, None, :]

    prox = np.clip(0.22 + 0.88 * sun_halo_w, 0.0, 1.0)[..., None]
    cloud_col = c_body_dark * (1.0 - prox * 0.56) + c_body_mid * (prox * 0.64) + cloud_rim[..., None] * (c_rim_gold * 0.86)
    cloud_col += (mottle * 0.018 + dabs * 0.016)[..., None]

    sky = sky * (1.0 - cloud_a[..., None] * 0.84) + cloud_col * (cloud_a[..., None] * 0.84)
    sky += (mottle * 0.011 + dabs * 0.010 + tooth * 0.005)[..., None]

    # Rugged Volcanic Headland Silhouette (multi-octave 1D fBm ridge profile — NO hard cutoff seam!)
    u_row = np.linspace(0.0, 1.0, W, dtype=np.float32)

    def fbm_ridge_1d(seed_r: np.random.Generator) -> np.ndarray:
        r = np.zeros(W, dtype=np.float32)
        for n_k, amp in ((14, 0.52), (32, 0.28), (72, 0.13), (150, 0.07)):
            v_k = seed_r.standard_normal(n_k).astype(np.float32)
            r += amp * np.interp(np.linspace(0, n_k - 1, W), np.arange(n_k), v_k)
        return r

    r_far_n = fbm_ridge_1d(rng)
    r_near_n = fbm_ridge_1d(rng)

    # Distant volcanic island mountain chain across center-left into mist
    far_env = np.clip((u_row - 0.34) / 0.18, 0.0, 1.0) * (0.45 + 0.55 * np.exp(-((u_row - 0.62) / 0.24) ** 2))
    ridge_far = hv - (0.022 + 0.036 * np.exp(-((u_row - 0.54) / 0.15) ** 2) + 0.012 * np.abs(r_far_n)) * far_env

    # Closer stepped volcanic basalt headland jutting dramatically from right edge down to the bay cape tip at u = 0.665
    cape_ramp = np.clip((u_row - 0.662) / 0.338, 0.0, 1.0)
    # Steep sea-cliff shoulder at u ~ 0.745 for the lighthouse, rising to rugged volcanic crag at u ~ 0.86..1.0
    cape_h = (
        0.044 * np.power(cape_ramp, 0.52)
        + 0.056 * np.exp(-((u_row - 0.88) / 0.105) ** 2)
        + 0.024 * np.exp(-((u_row - 0.752) / 0.048) ** 2)
        + 0.014 * np.abs(r_near_n) * np.clip((u_row - 0.675) / 0.12, 0.0, 1.0)
    ) * np.clip((u_row - 0.658) / 0.022, 0.0, 1.0)
    ridge_near = hv - cape_h

    # Anti-aliased soft-edge masks
    a_far = np.clip((v - ridge_far[None, :]) / 0.0022, 0.0, 1.0) * (v <= hv + 0.002).astype(np.float32)
    a_near = np.clip((v - ridge_near[None, :]) / 0.0018, 0.0, 1.0) * (v <= hv + 0.002).astype(np.float32)

    c_far_r = np.array([0.30, 0.115, 0.108], dtype=np.float32)[None, None, :] + sun_halo_w[..., None] * np.array([0.24, 0.105, 0.052], dtype=np.float32)
    c_near_r = np.array([0.115, 0.044, 0.052], dtype=np.float32)[None, None, :] + sun_halo_w[..., None] * np.array([0.12, 0.052, 0.028], dtype=np.float32)
    c_near_r += (mottle * 0.011 + dabs * 0.009)[..., None]

    # Smooth feathered sunset rim light kissing ONLY the sunlit crest slope smoothly (no vertical seam!)
    crest_fade = np.clip((u_row[None, :] - 0.662) / 0.06, 0.0, 1.0) * np.exp(-((u_row[None, :] - 0.75) / 0.18) ** 2)
    crest_rim = np.exp(-((v - ridge_near[None, :]) / 0.0038) ** 2) * crest_fade * (0.45 + 0.85 * sun_halo_w)

    sky = sky * (1.0 - a_far[..., None]) + c_far_r * a_far[..., None]
    sky = sky * (1.0 - a_near[..., None]) + c_near_r * a_near[..., None]
    sky += crest_rim[..., None] * np.array([0.52, 0.24, 0.10], dtype=np.float32)

    # Crisp Distant Lighthouse Silhouette perched on the volcanic headland shoulder (u ~ 0.752)
    lh_u = 0.752
    lh_x = int(round(lh_u * W))
    lh_base_y = int(round(float(ridge_near[lh_x]) * H)) + 4
    lh_h = 52
    for dy_l in range(lh_h):
        py = lh_base_y - dy_l
        if py < 0 or py >= H:
            continue
        frac = dy_l / float(lh_h)
        hw = 5.6 * (1.0 - 0.42 * frac)
        if 0.75 <= frac <= 0.81:
            hw = 6.0  # gallery
        elif 0.81 < frac <= 0.92:
            hw = 3.4  # glass lantern room
        elif frac > 0.92:
            hw = 3.9 * (1.0 - (frac - 0.92) / 0.09)
        x_lo = max(0, int(lh_x - hw))
        x_hi = min(W, int(lh_x + hw) + 1)
        sky[py, x_lo:x_hi] = np.array([0.095, 0.036, 0.044], dtype=np.float32)
        if x_lo < W:
            sky[py, x_lo : min(W, x_lo + 2)] += np.array([0.36, 0.17, 0.08], dtype=np.float32)
    # Glowing warm lighthouse beacon + delicate sweeping twilight haze beam!
    beacon_y = lh_base_y - int(lh_h * 0.86)
    d_b = np.hypot(xx - lh_x, yy - beacon_y)
    sky += np.exp(-(d_b / 3.2) ** 2).astype(np.float32)[..., None] * np.array([1.28, 0.96, 0.56], dtype=np.float32)
    sky += np.exp(-(d_b / 20.0) ** 1.6).astype(np.float32)[..., None] * np.array([0.42, 0.22, 0.08], dtype=np.float32)
    # Subtle slanted lighthouse beam sweeping softly left into the dusk air
    bx_d = xx - lh_x
    by_d = yy - beacon_y
    beam_ang = np.arctan2(by_d, bx_d)
    beam_cone = np.exp(-0.5 * ((beam_ang + 2.94) / 0.075) ** 2) * np.exp(-d_b / 260.0) * (bx_d < 0).astype(np.float32)
    sky += beam_cone[..., None] * np.array([0.28, 0.18, 0.08], dtype=np.float32)

    # =========================================================================
    # Stage 2 — Fresnel Capillary-Rippled Molten-Copper Harbor Bay Water (v > hv)
    # =========================================================================
    rip_coarse = resize_bicubic(rng.standard_normal((H // 5, W // 48)).astype(np.float32), W, H)
    rip_mid = resize_bicubic(rng.standard_normal((H // 3, W // 26)).astype(np.float32), W, H)
    rip_fine = resize_bicubic(rng.standard_normal((H // 2, W // 14)).astype(np.float32), W, H)
    ripple = 0.52 * rip_coarse + 0.32 * rip_mid + 0.16 * rip_fine

    water_t = np.clip((yy - hy) / max(1.0, H - hy), 0.0, 1.0)
    dist_t = np.power(water_t, 0.72)

    disp_y = ripple * (3.2 + 32.0 * dist_t)
    disp_x = (rip_mid * 0.6 + rip_fine * 0.4) * (2.2 + 16.0 * dist_t)
    y_refl = np.clip(2.0 * hy - yy + disp_y, 0.0, hy - 1.0)
    x_refl = np.clip(xx + disp_x, 0.0, W - 1.0)
    sky_refl = remap_bilinear(sky, x_refl, y_refl)

    fresnel = 0.24 + (0.90 - 0.24) * np.power(1.0 - dist_t, 1.4)

    c_water_deep = np.array([0.042, 0.018, 0.024], dtype=np.float32)[None, None, :]
    c_water_mid = np.array([0.135, 0.052, 0.048], dtype=np.float32)[None, None, :]
    water_body = c_water_deep * dist_t[..., None] + c_water_mid * (1.0 - dist_t[..., None])
    water_col = water_body * (1.0 - fresnel[..., None]) + sky_refl * (fresnel[..., None] * 0.94)

    # Shimmering molten-copper sun reflection path across capillary wave crests
    sun_col_w = 0.028 + 0.125 * dist_t
    sun_path = np.exp(-0.5 * ((u - sun_u) / sun_col_w) ** 2)
    wave_slope = np.clip(0.5 + 0.55 * ripple + 0.35 * np.sin(water_t * 145.0 + rip_coarse * 4.2), 0.0, 1.0)
    glitter = np.power(wave_slope, 4.6) * sun_path * (0.55 + 0.85 * (1.0 - dist_t * 0.5))
    shimmer_band = np.power(wave_slope, 2.0) * sun_path * 0.42
    water_col += shimmer_band[..., None] * np.array([0.84, 0.36, 0.15], dtype=np.float32)
    water_col += glitter[..., None] * np.array([1.25, 0.84, 0.42], dtype=np.float32)

    trough = np.clip(-ripple * 0.45, 0.0, 0.65) * (0.28 + 0.72 * dist_t)
    water_col *= (1.0 - 0.28 * trough[..., None])

    horiz_blend = np.clip((yy - hy + 2.0) / 5.0, 0.0, 1.0)[..., None]
    canvas = sky * (1.0 - horiz_blend) + water_col * horiz_blend
    haze_line = np.exp(-((v - hv) / 0.036) ** 2).astype(np.float32)
    canvas += haze_line[..., None] * np.array([0.22, 0.10, 0.052], dtype=np.float32) * (0.5 + 0.8 * sun_halo_w[..., None])

    fog_rgb = np.array([0.45, 0.19, 0.130], dtype=np.float32)
    return np.clip(canvas, 0.0, 1.5), fog_rgb, ripple


# =============================================================================
# Stage 1 & 2 — Planar UV Textures: Wet Voronoi Basalt Seawall, Timber Pier, Warehouse
# =============================================================================

def paint_voronoi_basalt_uv(
    rng: np.random.Generator,
    u_span: tuple[float, float],
    v_span: tuple[float, float],
    plane_kind: str,  # "seawall_vert" | "quay_cap"
    tw: int = 896,
    th: int = 768,
) -> np.ndarray:
    """
    Irregular wet volcanic Voronoi basalt seawall & quay stones in physical meters
    with gentle chiseled perturbation and vectorized 3x3 cell neighborhood (d2 - d1).
    """
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    a_m = np.float32(u_span[0]) + u * np.float32(u_span[1] - u_span[0])
    b_m = np.float32(v_span[0]) + v * np.float32(v_span[1] - v_span[0])

    # Crisp hand-chiseled stone perturbation (subtle, never wiggly cracked mud!)
    nx = resize_bicubic(rng.standard_normal((max(4, th // 22), max(4, tw // 22))).astype(np.float32), tw, th) * np.float32(0.018)
    ny = resize_bicubic(rng.standard_normal((max(4, th // 22), max(4, tw // 22))).astype(np.float32), tw, th) * np.float32(0.018)
    px, py = a_m + nx, b_m + ny

    cell_a, cell_b = (np.float32(0.64), np.float32(0.36)) if plane_kind == "seawall_vert" else (np.float32(0.52), np.float32(0.68))
    ix = np.floor(px / cell_a).astype(np.int32)
    iy = np.floor(py / cell_b).astype(np.int32)

    jx_tab = rng.uniform(0.20, 0.80, size=256).astype(np.float32)
    jy_tab = rng.uniform(0.20, 0.80, size=256).astype(np.float32)
    tone_tab = rng.uniform(-0.048, 0.056, size=256).astype(np.float32)
    warm_tab = rng.uniform(-0.015, 0.025, size=256).astype(np.float32)

    d1 = np.full((th, tw), 1e9, dtype=np.float32)
    d2 = np.full((th, tw), 1e9, dtype=np.float32)
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

    mortar = np.clip(((d2 - d1) - 0.008) / 0.016, 0.0, 1.0)
    edge_bevel = np.clip(((d2 - d1) - 0.016) / 0.038, 0.0, 1.0)
    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=84)
    tooth = paper_tooth(th, tw, rng)
    sv = tone_tab[best_id]
    sw = warm_tab[best_id]

    # Deep volcanic charcoal-basalt stone tones (high contrast against molten sunset sheen!)
    stone = np.zeros((th, tw, 3), dtype=np.float32)
    stone[:, :, 0] = 0.115 + sv * 0.52 + sw * 0.62 + mottle * 0.020 + dabs * 0.016
    stone[:, :, 1] = 0.076 + sv * 0.38 + sw * 0.24 + mottle * 0.015 + dabs * 0.012
    stone[:, :, 2] = 0.074 + sv * 0.34 + mottle * 0.013 + dabs * 0.010

    crevice_col = np.array([0.022, 0.011, 0.012], dtype=np.float32)
    rgb = mortar[..., None] * stone + (1.0 - mortar[..., None]) * crevice_col

    if plane_kind == "seawall_vert":
        wy, wz = a_m, b_m
        tide_wet = np.clip((0.46 - wz) / 0.46, 0.0, 1.0)
        moss = np.clip((0.52 - wz + 0.18 * mottle) / 0.34, 0.0, 1.0) * (0.42 + 0.58 * (1.0 - mortar))
        rgb *= (1.0 - 0.52 * tide_wet[..., None])
        rgb = rgb * (1.0 - 0.46 * moss[..., None]) + np.array([0.055, 0.072, 0.032], dtype=np.float32) * (0.46 * moss[..., None])

        # Grazing molten-vermilion sunset & copper water caustic bounce on the +X seawall face
        sun_graze = (0.45 + 0.55 * edge_bevel) * mortar * (1.0 - 0.38 * tide_wet)
        rgb += sun_graze[..., None] * np.array([0.28, 0.11, 0.048], dtype=np.float32)
        water_up = np.exp(-wz / 0.44) * (0.5 + 0.5 * np.sin(wy * 4.5 + mottle * 2.8))
        rgb += (water_up * mortar)[..., None] * np.array([0.24, 0.10, 0.040], dtype=np.float32)
        coping_sh = np.exp(-np.maximum(0.0, np.float32(Z_PIER) - wz) / 0.09)
        rgb *= (1.0 - 0.48 * coping_sh[..., None])
    else:
        wx, wy = a_m, b_m
        wz = np.full_like(wx, Z_PIER)
        vis = trace_pier_shadows(wx, wy, wz)
        lantern = eval_lantern_glow(wx, wy, wz)

        puddle = np.clip(0.28 + 0.52 * mottle + 0.25 * dabs, 0.0, 1.0)
        sky_sheen = (0.30 + 0.70 * puddle) * (0.30 + 0.70 * vis) * mortar
        rgb *= (0.42 + 0.58 * vis[..., None])
        rgb += sky_sheen[..., None] * np.array([0.42, 0.18, 0.09], dtype=np.float32)
        rim_spec = np.power(np.clip(1.0 - edge_bevel, 0.0, 1.0) * mortar, 1.2) * vis
        rgb += rim_spec[..., None] * np.array([0.50, 0.26, 0.11], dtype=np.float32)
        rgb += (lantern * (0.40 + 0.60 * puddle))[..., None] * np.array([0.42, 0.20, 0.075], dtype=np.float32)

    rgb = apply_warm_shadow_bounce(rgb, strength=0.018)
    rgb += tooth[..., None] * 0.007

    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.35)
    return rgba


def paint_timber_pier_uv(
    rng: np.random.Generator,
    x_span: tuple[float, float],
    y_span: tuple[float, float],
    tw: int = 768,
    th: int = 1024,
) -> np.ndarray:
    """
    Rain-slicked weathered dark cedar dock planks with organic staggered butt joints,
    wet glossy puddles reflecting the vermilion-amber sky, and exact Stage 2 shadows.
    """
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    wx = np.float32(x_span[0]) + (xx / (tw - 1)) * np.float32(x_span[1] - x_span[0])
    wy = np.float32(y_span[0]) + (yy / (th - 1)) * np.float32(y_span[1] - y_span[0])
    wz = np.full_like(wx, Z_PIER)

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=88)
    tooth = paper_tooth(th, tw, rng)

    plank_w = np.float32(0.245)
    plank_id = np.floor(wy / plank_w).astype(np.int32)
    plank_f = (wy / plank_w) - plank_id
    pf_w = np.clip(plank_f + 0.015 * mottle, 0.0, 1.0)
    seam_mask = np.clip(np.minimum(pf_w, 1.0 - pf_w) / 0.055, 0.0, 1.0)

    butt_off = (((plank_id * 1103515245 + 12345) & 255).astype(np.float32) / 255.0) * 1.9
    bx_f = np.mod((wx + butt_off) / 1.9, 1.0)
    butt_mask = np.clip(np.minimum(bx_f, 1.0 - bx_f) / 0.012, 0.0, 1.0)
    plank_solid = seam_mask * butt_mask

    p_hash = ((plank_id * 73856093) ^ (np.floor((wx + butt_off) / 1.9).astype(np.int32) * 19349663)) & 255
    tone_tab = rng.uniform(-0.044, 0.048, size=256).astype(np.float32)
    warm_tab = rng.uniform(-0.016, 0.024, size=256).astype(np.float32)
    pt = tone_tab[p_hash]
    pw = warm_tab[p_hash]

    grain_wave = np.sin(wy * 40.0 + mottle * 4.2 + wx * 1.6) * 0.012

    # Deeper charred-cedar weathered dock planks so wet puddles & golden sunset glints shine!
    wood = np.zeros((th, tw, 3), dtype=np.float32)
    wood[:, :, 0] = 0.135 + pt * 0.65 + pw * 0.78 + mottle * 0.020 + dabs * 0.016 + grain_wave
    wood[:, :, 1] = 0.068 + pt * 0.40 + pw * 0.30 + mottle * 0.013 + dabs * 0.010 + grain_wave * 0.6
    wood[:, :, 2] = 0.056 + pt * 0.28 + mottle * 0.011 + dabs * 0.008 + grain_wave * 0.4

    gap_col = np.array([0.016, 0.008, 0.009], dtype=np.float32)
    rgb = plank_solid[..., None] * wood + (1.0 - plank_solid[..., None]) * gap_col

    vis = trace_pier_shadows(wx, wy, wz)
    lantern = eval_lantern_glow(wx, wy, wz)

    rgb *= (0.36 + 0.64 * vis[..., None])
    rgb += (vis * plank_solid)[..., None] * np.array([0.16, 0.072, 0.030], dtype=np.float32)

    # Organic rain puddles pooling along the worn pier planks reflecting molten sunset sky & lanterns!
    puddle_n = resize_bicubic(fbm_gouache(th // 2, tw // 2, rng, scales=((30, 30), (13, 14), (6, 6), (3, 3))), tw, th)
    puddle_mask = np.clip((puddle_n + 0.06 + 0.24 * np.sin(wy * 0.52)) * 1.55, 0.0, 1.0) * plank_solid

    y_norm = np.clip((wy - y_span[0]) / (y_span[1] - y_span[0]), 0.0, 1.0)
    refl_near = np.array([0.46, 0.165, 0.105], dtype=np.float32)[None, None, :]
    refl_far = np.array([0.96, 0.54, 0.22], dtype=np.float32)[None, None, :]
    puddle_sky = refl_near * (1.0 - y_norm[..., None]) + refl_far * y_norm[..., None]
    puddle_col = puddle_sky * (0.30 + 0.70 * vis[..., None]) + lantern[..., None] * np.array([0.92, 0.48, 0.15], dtype=np.float32)

    wet_f = (0.14 + 0.60 * puddle_mask * (0.55 + 0.45 * y_norm))[..., None]
    rgb = rgb * (1.0 - wet_f) + puddle_col * wet_f
    rgb += (lantern * (0.50 + 0.50 * plank_solid))[..., None] * np.array([0.38, 0.18, 0.068], dtype=np.float32)

    bevel_hi = np.clip((0.14 - np.abs(pf_w - 0.08)) / 0.10, 0.0, 1.0) * vis * plank_solid
    rgb += bevel_hi[..., None] * np.array([0.22, 0.10, 0.042], dtype=np.float32)

    rgb = apply_warm_shadow_bounce(rgb, strength=0.018)
    rgb += tooth[..., None] * 0.007

    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.45)
    return rgba


def paint_warehouse_wall_uv(
    rng: np.random.Generator,
    y_span: tuple[float, float],
    z_span: tuple[float, float],
    tw: int = 896,
    th: int = 448,
) -> np.ndarray:
    """
    Dark charred-cedar (yakisugi) Japanese harbor warehouse timber siding with warm
    amber-gold paned windows (clean golden glass!), structural timber posts, and exact
    Stage 2 pitched-eave shadow band traced along -SUN_DIR.
    """
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    wy = np.float32(y_span[0]) + (xx / (tw - 1)) * np.float32(y_span[1] - y_span[0])
    wz = np.float32(z_span[0]) + (yy / (th - 1)) * np.float32(z_span[1] - z_span[0])
    wx = np.full_like(wy, WAREHOUSE_X_WALL)

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=82)
    tooth = paper_tooth(th, tw, rng)

    b_w = np.float32(0.22)
    bid = np.floor(wy / b_w).astype(np.int32)
    bf = (wy / b_w) - bid
    b_seam = np.clip(np.minimum(bf, 1.0 - bf) / 0.055, 0.0, 1.0)
    btab = rng.uniform(-0.020, 0.026, size=256).astype(np.float32)
    bt = btab[bid & 255]

    # Rich dark charred-cedar lacquer-brown warehouse siding (keeps warm windows & lamps glowing!)
    wall = np.zeros((th, tw, 3), dtype=np.float32)
    wall[:, :, 0] = 0.095 + bt * 0.60 + mottle * 0.016 + dabs * 0.013
    wall[:, :, 1] = 0.044 + bt * 0.34 + mottle * 0.010 + dabs * 0.008
    wall[:, :, 2] = 0.040 + bt * 0.28 + mottle * 0.008 + dabs * 0.007
    wall = b_seam[..., None] * wall + (1.0 - b_seam[..., None]) * np.array([0.016, 0.007, 0.009], dtype=np.float32)

    # Structural dark stone plinth + timber dado rail + vertical timber structural posts
    plinth = (wz < 1.50).astype(np.float32)
    dado = (np.abs(wz - 2.24) < 0.065).astype(np.float32)
    upper_beam = (np.abs(wz - 3.68) < 0.075).astype(np.float32)
    posts = np.zeros_like(wy)
    for py_post in (3.2, 8.2, 13.6, 19.0):
        posts = np.maximum(posts, (np.abs(wy - py_post) < 0.11).astype(np.float32))

    wall = wall * (1.0 - plinth[..., None]) + np.array([0.062, 0.042, 0.042], dtype=np.float32) * plinth[..., None]
    trim = np.clip(dado + upper_beam + posts, 0.0, 1.0)[..., None]
    wall = wall * (1.0 - trim * 0.55) + np.array([0.036, 0.015, 0.016], dtype=np.float32) * (trim * 0.55)

    # Exact Stage 2 pitched overhanging eave shadow along -SUN_DIR!
    eave_vis = trace_eave_shadow(wx, wy, wz)
    wall *= (0.34 + 0.66 * eave_vis[..., None])
    wall += (eave_vis * b_seam)[..., None] * np.array([0.16, 0.065, 0.028], dtype=np.float32)

    # Warm amber lantern glow pools spilled onto charred cedar boards
    lantern = eval_lantern_glow(wx, wy, wz)
    wall += lantern[..., None] * np.array([0.46, 0.21, 0.065], dtype=np.float32)

    # Warm glowing golden-amber paned harbor windows (clean warm palette, zero green cast!)
    windows = [
        (5.1, 7.0, 2.34, 3.38),
        (10.5, 12.4, 2.34, 3.38),
        (16.0, 17.8, 2.34, 3.38),
    ]
    for wy0, wy1, wz0, wz1 in windows:
        in_frame = (wy >= wy0 - 0.10) & (wy <= wy1 + 0.10) & (wz >= wz0 - 0.10) & (wz <= wz1 + 0.10)
        in_glass = (wy >= wy0) & (wy <= wy1) & (wz >= wz0) & (wz <= wz1)
        fu = (wy - wy0) / (wy1 - wy0)
        fv = (wz - wz0) / (wz1 - wz0)
        mullion = (np.abs(fu - 0.333) < 0.026) | (np.abs(fu - 0.667) < 0.026) | (np.abs(fv - 0.50) < 0.032)
        frame_c = np.array([0.030, 0.012, 0.014], dtype=np.float32)
        wall = np.where(in_frame[..., None], frame_c, wall)
        # Warm glowing lamp center inside the cozy harbor warehouse room
        r_lamp = np.hypot((fu - 0.52) * 1.2, (fv - 0.58) * 1.1)
        lamp_g = np.exp(-(r_lamp / 0.65) ** 2)
        pane_c = (
            np.array([0.82, 0.36, 0.12], dtype=np.float32)[None, None, :]
            + lamp_g[..., None] * np.array([0.42, 0.44, 0.24], dtype=np.float32)[None, None, :]
            + np.clip(mottle, -0.5, 0.5)[..., None] * np.array([0.06, 0.03, 0.01], dtype=np.float32)[None, None, :]
        )
        pane_mask = in_glass & (~mullion)
        wall = np.where(pane_mask[..., None], pane_c, wall)

    wall = apply_warm_shadow_bounce(wall, strength=0.018)
    wall += tooth[..., None] * 0.007

    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(wall, 0.0, 1.45)
    return rgba


# =============================================================================
# Stage 2 & 3 — Volumetric Golden-Hour Sunbeams Aligned With SUN_DIR
# =============================================================================

def paint_golden_sunbeams(cam: Camera3D, rng: np.random.Generator, intensity: float = 0.36) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sun_p = cam.K @ (cam.R @ (-SUN_DIR))
    su = float(np.clip(sun_p[0] / sun_p[2], W * 0.61, W * 0.70))
    sv = float(np.clip(sun_p[1] / sun_p[2], cam.horizon_y - H * 0.105, cam.horizon_y - H * 0.055))

    dx = xx - su
    dy = yy - sv
    dist = np.hypot(dx, dy) + 1e-4
    ang = np.arctan2(dy, dx)

    acc = np.zeros((H, W), dtype=np.float32)
    for a0, aw, st in [
        (2.44, 0.14, 0.40),
        (2.74, 0.12, 0.54),
        (3.02, 0.15, 0.46),
        (-2.86, 0.13, 0.44),
        (-2.48, 0.16, 0.34),
    ]:
        da = np.arctan2(np.sin(ang - a0), np.cos(ang - a0))
        beam = np.exp(-0.5 * (da / aw) ** 2)
        radial = np.exp(-dist / (W * 0.56)) * np.clip(dist / 85.0, 0.0, 1.0)
        acc += beam * radial * st

    turb = 0.85 + 0.15 * resize_bicubic(fbm_gouache(H // 4, W // 4, rng), W, H)
    acc = np.clip(acc * turb * intensity, 0.0, 0.72)
    return acc[..., None] * np.array([1.05, 0.54, 0.24], dtype=np.float32)[None, None, :]


# =============================================================================
# Stage 3 — Fast Tight-BBox 2x Supersampled PyCairo Helpers & Hero Harbor Props
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
    """
    Lanczos-downsample premultiplied 2x Cairo surface cleanly over its tight non-zero
    bounding box (exact premultiplied linear filtering -> clean 1x unpremultiply/repremultiply).
    """
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

    # Cairo ARGB32 little-endian memory layout is [B, G, R, A] premultiplied
    a1 = np.asarray(Image.fromarray(sub[:, :, 3], mode="L").resize((bw, bh), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    a1 = np.clip(a1, 0.0, 1.0)
    out[y0:y1, x0:x1, 3] = a1
    for out_c, src_c in ((0, 2), (1, 1), (2, 0)):
        c1 = np.asarray(Image.fromarray(sub[:, :, src_c], mode="L").resize((bw, bh), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
        # Cleanly unpremultiply & clamp straight color to [0, 1], then keep premultiplied for over_premul
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


def fog_mix(c3: Iterable[float], dist: float, fog_rgb: np.ndarray, fog_start: float = 7.5, fog_density: float = 0.036) -> tuple[float, float, float]:
    ft = 1.0 - math.exp(-max(0.0, dist - fog_start) * fog_density)
    c = np.asarray(c3, dtype=np.float64) * (1.0 - ft) + fog_rgb * ft
    return (float(c[0]), float(c[1]), float(c[2]))


def draw_timber_piling(
    ctx: cairo.Context,
    cam: Camera3D,
    x: float,
    y: float,
    z_top: float,
    rad: float,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    mirror: bool = False,
) -> None:
    """
    Camera-aligned cylindrical weathered wooden harbor fender piling seated against the
    basalt seawall (uses cam.right_xy so cylinder silhouette is always true metric width!).
    """
    p_bot = np.array([x, y, Z_WATER], dtype=np.float64)
    p_top = np.array([x, y, z_top], dtype=np.float64)
    if mirror:
        p0_w = np.array([0.0, 0.0, Z_WATER])
        n_w = np.array([0.0, 0.0, 1.0])
        p_bot = mirror_pts_plane(p_bot, p0_w, n_w)
        p_top = mirror_pts_plane(p_top, p0_w, n_w)

    dist = float(np.linalg.norm(p_top - cam.eye))
    rvec = cam.right_xy * rad
    quad = np.array([p_bot - rvec, p_bot + rvec, p_top + rvec, p_top - rvec])
    poly = project_quad_3d(cam, quad)
    if poly is None:
        return

    alpha = 0.56 if mirror else 0.99
    c_sh = fog_mix([0.048, 0.020, 0.022], dist, fog_rgb)
    c_mid = fog_mix([0.145, 0.062, 0.046], dist, fog_rgb)
    c_hi = fog_mix([0.52, 0.21, 0.095], dist, fog_rgb)
    xs = [p[0] for p in poly]
    g = cairo.LinearGradient(min(xs), poly[0][1], max(xs), poly[0][1])
    g.add_color_stop_rgba(0.0, *c_sh, alpha)
    g.add_color_stop_rgba(0.65, *c_mid, alpha)
    g.add_color_stop_rgba(1.0, *c_hi, alpha)
    ctx.set_source(g)
    fill_poly(ctx, poly)

    if not mirror:
        # Exact-width weathered timber top cap + dark iron strap band around top
        u_l, v_l, z_l = cam.project(p_top - rvec)
        u_r, v_r, z_r = cam.project(p_top + rvec)
        if z_l > Z_NEAR and z_r > Z_NEAR:
            cx_t = 0.5 * (float(u_l) + float(u_r))
            cy_t = 0.5 * (float(v_l) + float(v_r))
            rx = max(1.8, 0.5 * abs(float(u_r) - float(u_l)))
            ry = rx * 0.34
            ctx.save()
            ctx.translate(cx_t, cy_t)
            ctx.scale(rx, ry)
            g_c = cairo.LinearGradient(-1.0, 0.0, 1.0, 0.0)
            g_c.add_color_stop_rgba(0.0, *fog_mix([0.11, 0.048, 0.038], dist, fog_rgb), 1.0)
            g_c.add_color_stop_rgba(1.0, *fog_mix([0.58, 0.26, 0.12], dist, fog_rgb), 1.0)
            ctx.set_source(g_c)
            ctx.arc(0, 0, 1.0, 0, math.tau)
            ctx.fill()
            ctx.restore()

            # Iron band near top of piling
            p_b1 = p_top - np.array([0.0, 0.0, 0.14])
            p_b2 = p_top - np.array([0.0, 0.0, 0.07])
            band_q = project_quad_3d(cam, np.array([p_b1 - rvec * 1.02, p_b1 + rvec * 1.02, p_b2 + rvec * 1.02, p_b2 - rvec * 1.02]))
            ctx.set_source_rgba(*fog_mix([0.032, 0.014, 0.016], dist, fog_rgb), 0.94)
            fill_poly(ctx, band_q)

            # Warm sunset rim along right silhouette edge
            seg = cam.project_polyline(np.array([p_bot + rvec, p_top + rvec]))
            if len(seg) >= 2:
                stroke_catmull_rom(ctx, [(p[0], p[1]) for p in seg], base_w=1.8, z_depth=dist,
                                   rgba=(*fog_mix([0.85, 0.42, 0.18], dist, fog_rgb), 0.72), rng=rng, taper=False, tremor=0.18)


def draw_moored_wooden_boat(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    mirror: bool = False,
) -> None:
    """
    Nostalgic 1990s anime 3/4-angled Japanese coastal wooden fishing/harbor boat (wasen)
    moored alongside the pier (bow pointing gracefully forward-right toward the copper bay,
    starboard flank + aft transom facing the viewer cleanly with zero face-order seams!),
    curved upturned prow, weathered wooden plank hull, vermilion gunwale stripe, cozy glowing
    wheelhouse cabin, offset mast standing cleanly left of the lighthouse, and exact 3D
    plane-mirrored reflection across Z_WATER.
    """
    p0_w = np.array([0.0, 0.0, Z_WATER], dtype=np.float64)
    n_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # Boat center at (3.40, 13.0, 0.0), yawed +36 deg so bow points forward-right into the bay
    # and BOTH the aft transom (-L) and full starboard hull flank (+B) face cleanly toward the camera!
    center = np.array([3.40, 13.0, 0.0], dtype=np.float64)
    yaw = math.radians(36.0)
    fwd_b = np.array([math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float64)
    right_b = np.array([math.cos(yaw), -math.sin(yaw), 0.0], dtype=np.float64)  # starboard (+B, camera-facing!)
    up_b = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def to_world(l_arr, b_arr, z_arr) -> np.ndarray:
        l_a = np.asarray(l_arr, dtype=np.float64)
        b_a = np.asarray(b_arr, dtype=np.float64)
        z_a = np.asarray(z_arr, dtype=np.float64)
        pts = center + l_a[..., None] * fwd_b + b_a[..., None] * right_b + z_a[..., None] * up_b
        return mirror_pts_plane(pts, p0_w, n_w) if mirror else pts

    dist = float(np.linalg.norm(center + np.array([0, 0, 0.6]) - cam.eye))
    alpha = 0.58 if mirror else 1.0

    # Smooth hull station profile from stern (l = -2.65) to curved pointed Japanese prow (l = +3.10)
    n_s = 16
    t = np.linspace(0.0, 1.0, n_s)
    l_s = -2.65 + t * 5.75
    beam = 0.96 * np.power(1.0 - t, 0.60) * (0.80 + 0.32 * np.sin(t * math.pi * 0.85))
    beam[-1] = 0.02
    z_top = 0.62 + 0.44 * np.power(t, 2.3) + 0.06 * np.power(1.0 - t, 2.0)
    z_bot = 0.02 + 0.14 * np.power(np.maximum(0.0, t - 0.72) / 0.28, 1.8)

    # 1. Open wooden cockpit deck interior (drawn first, behind camera-facing starboard gunwale & transom)
    port_top = to_world(l_s, -beam, z_top)
    stbd_top = to_world(l_s, beam, z_top)
    poly_in = project_quad_3d(cam, np.vstack([port_top, stbd_top[::-1]]))
    if poly_in is not None:
        ctx.set_source_rgba(*fog_mix([0.15, 0.062, 0.044], dist, fog_rgb), alpha)
        fill_poly(ctx, poly_in)

    # 2. Wooden Aft Stern Transom (camera-facing at l = -2.65, flush with starboard waterline!)
    transom = to_world(
        [-2.65, -2.65, -2.65, -2.65],
        [-beam[0] * 0.78, beam[0] * 0.78, beam[0], -beam[0]],
        [z_bot[0], z_bot[0], z_top[0], z_top[0]],
    )
    ctx.set_source_rgba(*fog_mix([0.064, 0.026, 0.028], dist, fog_rgb), alpha)
    fill_poly(ctx, project_quad_3d(cam, transom))

    # 3. Main Camera-Facing Starboard Wooden Hull (+beam, meeting transom seamlessly at l = -2.65!)
    stbd_bot = to_world(l_s, beam * 0.78, z_bot)
    poly_hull = project_quad_3d(cam, np.vstack([stbd_bot, stbd_top[::-1]]))
    if poly_hull is not None:
        ys_p = [p[1] for p in poly_hull]
        g_h = cairo.LinearGradient(poly_hull[0][0], min(ys_p), poly_hull[0][0], max(ys_p))
        g_h.add_color_stop_rgba(0.0, *fog_mix([0.24, 0.095, 0.064], dist, fog_rgb), alpha)
        g_h.add_color_stop_rgba(0.55, *fog_mix([0.115, 0.046, 0.040], dist, fog_rgb), alpha)
        g_h.add_color_stop_rgba(1.0, *fog_mix([0.048, 0.020, 0.024], dist, fog_rgb), alpha)
        ctx.set_source(g_h)
        fill_poly(ctx, poly_hull)

    # 4. Vermilion lacquer gunwale trim band along starboard sheer & stern transom top
    stripe_bot = to_world(l_s, beam * 0.96 + 0.01, z_top - 0.14)
    stripe_top = to_world(l_s, beam + 0.01, z_top)
    poly_st = project_quad_3d(cam, np.vstack([stripe_bot, stripe_top[::-1]]))
    if poly_st is not None:
        ctx.set_source_rgba(*fog_mix([0.80, 0.25, 0.13], dist, fog_rgb), alpha)
        fill_poly(ctx, poly_st)
    stern_st = to_world(
        [-2.66, -2.66, -2.66, -2.66],
        [-beam[0], beam[0], beam[0] * 0.96, -beam[0] * 0.96],
        [z_top[0], z_top[0], z_top[0] - 0.14, z_top[0] - 0.14],
    )
    ctx.set_source_rgba(*fog_mix([0.56, 0.16, 0.095], dist, fog_rgb), alpha)
    fill_poly(ctx, project_quad_3d(cam, stern_st))

    # Raised dark-timber Japanese bow stempost (miyoshi) at prow tip
    prow_q = to_world(
        [3.00, 3.20, 3.34, 3.10],
        [0.02, 0.02, 0.04, 0.04],
        [z_top[-1] - 0.14, z_top[-1] - 0.04, z_top[-1] + 0.34, z_top[-1] + 0.28],
    )
    ctx.set_source_rgba(*fog_mix([0.072, 0.028, 0.028], dist, fog_rgb), alpha)
    fill_poly(ctx, project_quad_3d(cam, prow_q))

    # 4b. Draw pier-facing mooring lines & far-side port bow rim BEFORE the wheelhouse cabin so nothing crosses cabin glass!
    if not mirror:
        rim_seg = cam.project_polyline(port_top[10:])
        if len(rim_seg) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in rim_seg], base_w=2.2, z_depth=dist,
                               rgba=(*fog_mix([1.0, 0.60, 0.26], dist, fog_rgb), 0.82), rng=rng, taper=True, tremor=0.14)
        for p_a, p_b, sag_h in (
            (np.array([0.50, 9.20, Z_PIER + 0.22]), to_world(-2.55, -beam[0] * 0.9, z_top[0] + 0.02), 0.40),
            (np.array([0.50, 14.30, Z_PIER + 0.22]), to_world(1.55, -beam[10], z_top[10] + 0.02), 0.34),
        ):
            rpts = []
            for k in range(13):
                fr = k / 12.0
                rpts.append(p_a * (1.0 - fr) + p_b * fr - np.array([0.0, 0.0, sag_h * math.sin(fr * math.pi)]))
            rseg = cam.project_polyline(np.array(rpts))
            if len(rseg) >= 2:
                stroke_catmull_rom(ctx, [(p[0], p[1]) for p in rseg], base_w=3.2, z_depth=dist * 0.85,
                                   rgba=(*fog_mix([0.12, 0.055, 0.040], dist * 0.85, fog_rgb), 0.94), rng=rng, taper=False, tremor=0.16)

    # 5. Cozy weathered wooden wheelhouse cabin (rear bulkhead + camera-facing starboard wall + sloped roof!)
    cab_aft = to_world([-1.25, -1.25, -1.18, -1.18], [-0.58, 0.58, 0.52, -0.52], [0.62, 0.62, 1.56, 1.56])
    ctx.set_source_rgba(*fog_mix([0.062, 0.024, 0.026], dist, fog_rgb), alpha)
    fill_poly(ctx, project_quad_3d(cam, cab_aft))

    cab_stbd = to_world([-1.25, 0.55, 0.45, -1.18], [0.58, 0.58, 0.52, 0.52], [0.62, 0.64, 1.58, 1.56])
    ctx.set_source_rgba(*fog_mix([0.115, 0.046, 0.042], dist, fog_rgb), alpha)
    fill_poly(ctx, project_quad_3d(cam, cab_stbd))

    # Glowing warm amber double-paned cabin windows on starboard side + rear cabin porthole!
    for l0_w, l1_w in ((-0.95, -0.38), (-0.22, 0.32)):
        wq = to_world([l0_w, l1_w, l1_w - 0.03, l0_w + 0.02], [0.56, 0.56, 0.53, 0.53], [1.02, 1.03, 1.40, 1.39])
        ctx.set_source_rgba(*fog_mix([1.06, 0.62, 0.20], dist, fog_rgb), alpha)
        fill_poly(ctx, project_quad_3d(cam, wq))
    wq_aft = to_world([-1.22, -1.22, -1.19, -1.19], [-0.28, 0.24, 0.22, -0.26], [1.02, 1.02, 1.38, 1.38])
    ctx.set_source_rgba(*fog_mix([0.96, 0.50, 0.16], dist, fog_rgb), alpha)
    fill_poly(ctx, project_quad_3d(cam, wq_aft))

    # Overhanging cabin roof catching glowing golden-vermilion sunset sky!
    cab_roof = to_world([-1.36, 0.68, 0.68, -1.36], [-0.68, -0.68, 0.68, 0.68], [1.56, 1.62, 1.62, 1.56])
    ctx.set_source_rgba(*fog_mix([0.52, 0.22, 0.115], dist, fog_rgb), alpha)
    fill_poly(ctx, project_quad_3d(cam, cab_roof))

    # Short slender wooden mast offset cleanly LEFT of the distant lighthouse (at l = -0.35, h = 2.65)
    mast_pts = to_world([-0.32, -0.45], [0.0, 0.0], [1.58, 2.76])
    mseg = cam.project_polyline(mast_pts)
    if len(mseg) >= 2:
        stroke_catmull_rom(ctx, [(p[0], p[1]) for p in mseg], base_w=2.8, z_depth=dist,
                           rgba=(*fog_mix([0.11, 0.044, 0.042], dist, fog_rgb), alpha * 0.92), rng=rng, taper=False, tremor=0.14)
    stay_pts = to_world([-0.44, 3.18], [0.0, 0.0], [2.68, z_top[-1] + 0.22])
    sseg = cam.project_polyline(stay_pts)
    if len(sseg) >= 2:
        stroke_catmull_rom(ctx, [(p[0], p[1]) for p in sseg], base_w=1.2, z_depth=dist,
                           rgba=(*fog_mix([0.14, 0.055, 0.046], dist, fog_rgb), alpha * 0.72), rng=rng, taper=False, tremor=0.10)

    if not mirror:
        # Subtle crisp waterline copper-foam lap hugging the hull waterline
        wl_seg = cam.project_polyline(stbd_bot)
        if len(wl_seg) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in wl_seg], base_w=2.2, z_depth=dist,
                               rgba=(*fog_mix([0.96, 0.56, 0.24], dist, fog_rgb), 0.62), rng=rng, taper=True, tremor=0.22)
        # Hull sheer ink contour + mid strake seam
        sheer_seg = cam.project_polyline(stbd_top)
        if len(sheer_seg) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in sheer_seg], base_w=2.0, z_depth=dist,
                               rgba=(*fog_mix([0.030, 0.012, 0.014], dist, fog_rgb), 0.88), rng=rng, taper=False, tremor=0.16)
        mid_strake = to_world(l_s, beam * 0.88, 0.5 * (z_bot + z_top))
        ms_seg = cam.project_polyline(mid_strake)
        if len(ms_seg) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in ms_seg], base_w=1.3, z_depth=dist,
                               rgba=(*fog_mix([0.036, 0.014, 0.016], dist, fog_rgb), 0.65), rng=rng, taper=False, tremor=0.16)


def draw_iron_bollard(
    ctx: cairo.Context,
    cam: Camera3D,
    bx: float,
    by: float,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    with_coil: bool = False,
) -> None:
    """Heavy weathered cast-iron mooring bollard with chunky braided hemp wraps seated snugly."""
    p_base = np.array([bx, by, Z_PIER], dtype=np.float64)
    p_neck = np.array([bx, by, Z_PIER + 0.34], dtype=np.float64)
    p_cap = np.array([bx, by, Z_PIER + 0.48], dtype=np.float64)
    dist = float(np.linalg.norm(p_base - cam.eye))
    u0, v0, z0 = cam.project(p_base)
    u1, v1, z1 = cam.project(p_neck)
    u2, v2, z2 = cam.project(p_cap)
    if z0 < 1.2 or z2 < 1.2:
        return

    px_m = float(cam.K[0, 0] / z0)
    r_base = 0.22 * px_m
    r_shaft = 0.138 * px_m
    r_cap = 0.195 * px_m

    # Soft cast contact shadow beneath bollard
    ctx.save()
    ctx.translate(float(u0) - r_base * 0.32, float(v0) + r_base * 0.08)
    ctx.scale(r_base * 1.48, r_base * 0.40)
    ctx.set_source_rgba(0.022, 0.009, 0.011, 0.62)
    ctx.arc(0, 0, 1.0, 0, math.tau)
    ctx.fill()
    ctx.restore()

    # Flanged cast-iron baseplate
    ctx.save()
    ctx.translate(float(u0), float(v0))
    ctx.scale(r_base, r_base * 0.36)
    ctx.set_source_rgba(*fog_mix([0.052, 0.024, 0.025], dist, fog_rgb), 0.99)
    ctx.arc(0, 0, 1.0, 0, math.tau)
    ctx.fill()
    ctx.restore()

    # Cylindrical cast-iron waist
    g_body = cairo.LinearGradient(float(u0) - r_shaft, float(v1), float(u0) + r_shaft, float(v1))
    g_body.add_color_stop_rgba(0.0, *fog_mix([0.042, 0.018, 0.020], dist, fog_rgb), 1.0)
    g_body.add_color_stop_rgba(0.60, *fog_mix([0.135, 0.058, 0.050], dist, fog_rgb), 1.0)
    g_body.add_color_stop_rgba(1.0, *fog_mix([0.64, 0.28, 0.13], dist, fog_rgb), 1.0)
    ctx.set_source(g_body)
    ctx.move_to(float(u0) - r_shaft * 1.08, float(v0))
    ctx.line_to(float(u1) - r_shaft * 0.92, float(v1))
    ctx.line_to(float(u1) + r_shaft * 0.92, float(v1))
    ctx.line_to(float(u0) + r_shaft * 1.08, float(v0))
    ctx.close_path()
    ctx.fill()

    # Chunky snugly wrapped hemp hawser loops hugging ONLY the bollard waist!
    if with_coil:
        rope_sh = fog_mix([0.095, 0.044, 0.032], dist, fog_rgb)
        rope_mid = fog_mix([0.24, 0.115, 0.068], dist, fog_rgb)
        rope_hi = fog_mix([0.74, 0.38, 0.17], dist, fog_rgb)
        rope_w = max(2.2, r_shaft * 0.28)
        for k_wrap in range(3):
            vy = float(v0) - (float(v0) - float(v1)) * (0.24 + k_wrap * 0.25)
            hw = r_shaft * 1.04
            # Solid thick braided rope band curving snugly across front of cylinder
            ctx.set_line_width(rope_w)
            ctx.set_source_rgba(*rope_sh, 0.98)
            ctx.move_to(float(u0) - hw, vy - rope_w * 0.25)
            ctx.curve_to(float(u0) - hw * 0.4, vy + rope_w * 0.55, float(u0) + hw * 0.4, vy + rope_w * 0.55, float(u0) + hw, vy - rope_w * 0.15)
            ctx.stroke()
            ctx.set_line_width(rope_w * 0.62)
            ctx.set_source_rgba(*rope_mid, 0.95)
            ctx.move_to(float(u0) - hw * 0.92, vy - rope_w * 0.30)
            ctx.curve_to(float(u0) - hw * 0.35, vy + rope_w * 0.42, float(u0) + hw * 0.38, vy + rope_w * 0.42, float(u0) + hw * 0.95, vy - rope_w * 0.22)
            ctx.stroke()
            # Warm sunlit specular highlight on right sun-facing curve of rope
            ctx.set_line_width(rope_w * 0.32)
            ctx.set_source_rgba(*rope_hi, 0.85)
            ctx.move_to(float(u0) + hw * 0.15, vy + rope_w * 0.22)
            ctx.curve_to(float(u0) + hw * 0.55, vy + rope_w * 0.16, float(u0) + hw * 0.82, vy - rope_w * 0.05, float(u0) + hw * 0.96, vy - rope_w * 0.24)
            ctx.stroke()

    # Flared mushroom cap of bollard
    g_cap = cairo.LinearGradient(float(u2) - r_cap, float(v2), float(u2) + r_cap, float(v2))
    g_cap.add_color_stop_rgba(0.0, *fog_mix([0.058, 0.025, 0.026], dist, fog_rgb), 1.0)
    g_cap.add_color_stop_rgba(0.55, *fog_mix([0.19, 0.082, 0.060], dist, fog_rgb), 1.0)
    g_cap.add_color_stop_rgba(1.0, *fog_mix([0.86, 0.42, 0.18], dist, fog_rgb), 1.0)
    ctx.set_source(g_cap)
    ctx.move_to(float(u1) - r_shaft * 0.95, float(v1))
    ctx.curve_to(float(u1) - r_cap * 1.05, float(v1) - r_cap * 0.18, float(u2) - r_cap, float(v2) + r_cap * 0.15, float(u2) - r_cap * 0.95, float(v2))
    ctx.line_to(float(u2) + r_cap * 0.95, float(v2))
    ctx.curve_to(float(u2) + r_cap, float(v2) + r_cap * 0.15, float(u1) + r_cap * 1.05, float(v1) - r_cap * 0.18, float(u1) + r_shaft * 0.95, float(v1))
    ctx.close_path()
    ctx.fill()

    # Wet glistening crown oval catching vermilion-gold sunset!
    ctx.save()
    ctx.translate(float(u2), float(v2))
    ctx.scale(r_cap * 0.96, r_cap * 0.34)
    g_top = cairo.LinearGradient(-1.0, 0.0, 1.0, 0.0)
    g_top.add_color_stop_rgba(0.0, *fog_mix([0.24, 0.10, 0.068], dist, fog_rgb), 1.0)
    g_top.add_color_stop_rgba(1.0, *fog_mix([0.96, 0.56, 0.25], dist, fog_rgb), 1.0)
    ctx.set_source(g_top)
    ctx.arc(0, 0, 1.0, 0, math.tau)
    ctx.fill()
    ctx.restore()

    # Crisp 1990s rim ink stroke on sun-facing flank
    stroke_catmull_rom(
        ctx,
        [(float(u0) + r_shaft * 1.04, float(v0)), (float(u1) + r_shaft * 0.92, float(v1)), (float(u2) + r_cap * 0.92, float(v2))],
        base_w=2.0,
        z_depth=dist,
        rgba=(*fog_mix([1.0, 0.62, 0.28], dist, fog_rgb), 0.78),
        rng=rng,
        taper=False,
        tremor=0.16,
    )


def draw_warehouse_eaves_and_props(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
) -> None:
    """
    Substantial Japanese wooden harbor warehouse overhanging timber eaves, layered rafters,
    chunky knee-braces positioned cleanly BETWEEN windows, glowing amber bulkhead lanterns,
    and slatted wooden harbor crates with wood-plank detail & rim highlights.
    """
    # 1. Under-eave shadowed timber soffit
    soffit_pts = np.array([
        [WAREHOUSE_X_WALL, WAREHOUSE_Y0, WAREHOUSE_Z_EAVE + 0.66],
        [WAREHOUSE_X_EAVE, WAREHOUSE_Y0, WAREHOUSE_Z_EAVE],
        [WAREHOUSE_X_EAVE, WAREHOUSE_Y1, WAREHOUSE_Z_EAVE],
        [WAREHOUSE_X_WALL, WAREHOUSE_Y1, WAREHOUSE_Z_EAVE + 0.66],
    ])
    sp = project_quad_3d(cam, soffit_pts)
    if sp is not None:
        g_s = cairo.LinearGradient(sp[0][0], sp[0][1], sp[2][0], sp[2][1])
        g_s.add_color_stop_rgba(0.0, 0.036, 0.014, 0.016, 0.99)
        g_s.add_color_stop_rgba(1.0, *fog_mix([0.080, 0.032, 0.032], 18.0, fog_rgb), 0.99)
        ctx.set_source(g_s)
        fill_poly(ctx, sp)

    # Exposed underside wooden rafters spaced cleanly along the eave soffit
    for ry in np.arange(2.0, 21.6, 0.78):
        dist = float(np.linalg.norm(np.array([WAREHOUSE_X_EAVE, ry, WAREHOUSE_Z_EAVE]) - cam.eye))
        seg = cam.project_polyline(np.array([
            [WAREHOUSE_X_WALL, ry, WAREHOUSE_Z_EAVE + 0.60],
            [WAREHOUSE_X_EAVE, ry, WAREHOUSE_Z_EAVE - 0.04],
        ]))
        if len(seg) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in seg], base_w=4.2, z_depth=dist,
                               rgba=(*fog_mix([0.022, 0.009, 0.011], dist, fog_rgb), 0.95), rng=rng, taper=False, tremor=0.14)

    # Substantial wooden knee-braces seated cleanly BELOW the eave soffit (never poking above roof!)
    for by in (8.2, 13.6, 19.0):
        dist = float(np.linalg.norm(np.array([WAREHOUSE_X_WALL, by, 3.6]) - cam.eye))
        # Exact soffit height at x = WAREHOUSE_X_EAVE - 0.16 is WAREHOUSE_Z_EAVE + 0.16 * 0.44
        x_end = WAREHOUSE_X_EAVE - 0.14
        z_sof = WAREHOUSE_Z_EAVE + (WAREHOUSE_X_EAVE - x_end) * WAREHOUSE_ROOF_SLOPE
        brace_q = np.array([
            [WAREHOUSE_X_WALL, by - 0.05, WAREHOUSE_Z_EAVE - 0.62],
            [WAREHOUSE_X_WALL, by - 0.05, WAREHOUSE_Z_EAVE - 0.44],
            [x_end, by - 0.05, z_sof - 0.02],
            [x_end - 0.14, by - 0.05, z_sof - 0.14],
        ])
        ctx.set_source_rgba(*fog_mix([0.052, 0.020, 0.020], dist, fog_rgb), 0.98)
        fill_poly(ctx, project_quad_3d(cam, brace_q))

    # Thick timber eave fascia board + dark roof gutter lip catching grazing sunset rim!
    fascia_pts = np.array([
        [WAREHOUSE_X_EAVE, WAREHOUSE_Y0, WAREHOUSE_Z_EAVE - 0.18],
        [WAREHOUSE_X_EAVE, WAREHOUSE_Y1, WAREHOUSE_Z_EAVE - 0.18],
        [WAREHOUSE_X_EAVE, WAREHOUSE_Y1, WAREHOUSE_Z_EAVE + 0.12],
        [WAREHOUSE_X_EAVE, WAREHOUSE_Y0, WAREHOUSE_Z_EAVE + 0.12],
    ])
    fp = project_quad_3d(cam, fascia_pts)
    if fp is not None:
        g_f = cairo.LinearGradient(fp[0][0], fp[0][1], fp[1][0], fp[1][1])
        g_f.add_color_stop_rgba(0.0, 0.14, 0.052, 0.040, 1.0)
        g_f.add_color_stop_rgba(1.0, *fog_mix([0.42, 0.18, 0.10], 20.0, fog_rgb), 1.0)
        ctx.set_source(g_f)
        fill_poly(ctx, fp)

    # Sloped corrugated weathered harbor roof above fascia
    roof_pts = np.array([
        [WAREHOUSE_X_EAVE - 0.02, WAREHOUSE_Y0, WAREHOUSE_Z_EAVE + 0.10],
        [WAREHOUSE_X_EAVE - 0.02, WAREHOUSE_Y1, WAREHOUSE_Z_EAVE + 0.10],
        [WAREHOUSE_X_WALL - 1.8, WAREHOUSE_Y1, WAREHOUSE_Z_EAVE + 1.55],
        [WAREHOUSE_X_WALL - 1.8, WAREHOUSE_Y0, WAREHOUSE_Z_EAVE + 1.55],
    ])
    ctx.set_source_rgba(0.044, 0.018, 0.022, 0.99)
    fill_poly(ctx, project_quad_3d(cam, roof_pts))

    # Grazing warm sunset rim stroke along outer roof drip edge
    eseg = cam.project_polyline(np.array([[WAREHOUSE_X_EAVE, WAREHOUSE_Y0 + 0.4, WAREHOUSE_Z_EAVE + 0.08],
                                          [WAREHOUSE_X_EAVE, WAREHOUSE_Y1, WAREHOUSE_Z_EAVE + 0.08]]))
    if len(eseg) >= 2:
        stroke_catmull_rom(ctx, [(p[0], p[1]) for p in eseg], base_w=2.4, z_depth=8.0,
                           rgba=(0.94, 0.50, 0.22, 0.72), rng=rng, taper=False, tremor=0.14)

    # Warm glowing amber harbor bulkhead lamps mounted cleanly between windows
    for lx, ly, lz, _ in LANTERN_LIGHTS:
        dist = float(np.linalg.norm(np.array([lx, ly, lz]) - cam.eye))
        bseg = cam.project_polyline(np.array([[WAREHOUSE_X_WALL, ly, lz + 0.24], [lx, ly, lz + 0.24], [lx, ly, lz + 0.08]]))
        if len(bseg) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in bseg], base_w=3.2, z_depth=dist,
                               rgba=(*fog_mix([0.036, 0.015, 0.016], dist, fog_rgb), 0.96), rng=rng, taper=False, tremor=0.10)
        ul, vl, zl = cam.project(np.array([lx, ly, lz]))
        if zl < 1.2:
            continue
        r_l = max(3.5, float(cam.K[0, 0] * 0.11 / zl))
        g_l = cairo.RadialGradient(float(ul), float(vl), r_l * 0.2, float(ul), float(vl), r_l * 7.8)
        g_l.add_color_stop_rgba(0.0, 1.0, 0.86, 0.46, 0.74)
        g_l.add_color_stop_rgba(0.32, 0.98, 0.50, 0.16, 0.30)
        g_l.add_color_stop_rgba(1.0, 0.88, 0.26, 0.08, 0.0)
        ctx.set_source(g_l)
        ctx.arc(float(ul), float(vl), r_l * 7.8, 0, math.tau)
        ctx.fill()
        ctx.set_source_rgba(*fog_mix([0.044, 0.018, 0.020], dist, fog_rgb), 0.98)
        ctx.move_to(float(ul) - r_l * 1.6, float(vl) - r_l * 0.35)
        ctx.line_to(float(ul), float(vl) - r_l * 1.25)
        ctx.line_to(float(ul) + r_l * 1.6, float(vl) - r_l * 0.35)
        ctx.close_path()
        ctx.fill()
        ctx.set_source_rgba(1.0, 0.93, 0.70, 0.98)
        ctx.arc(float(ul), float(vl) + r_l * 0.15, r_l * 0.62, 0, math.tau)
        ctx.fill()

    # Detailed slatted weathered wooden fish crates beside the warehouse wall!
    crates = [
        (-2.46, -1.78, 5.75, 6.65, Z_PIER, Z_PIER + 0.56),
        (-2.42, -1.86, 5.88, 6.52, Z_PIER + 0.56, Z_PIER + 1.02),
        (-2.44, -1.80, 12.9, 13.8, Z_PIER, Z_PIER + 0.54),
    ]
    for x0, x1, y0, y1, z0, z1 in crates:
        dist = float(np.linalg.norm(np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5, z0]) - cam.eye))
        faces = [
            (np.array([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]), [0.095, 0.042, 0.034]),
            (np.array([[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]]), [0.34, 0.15, 0.078]),
            (np.array([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]), [0.26, 0.11, 0.062]),
        ]
        for fi, (q, col) in enumerate(faces):
            qp = project_quad_3d(cam, q)
            if qp is not None:
                ctx.set_source_rgba(*fog_mix(col, dist, fog_rgb), 0.99)
                fill_poly(ctx, qp)
                # Horizontal wood slats on crate sides
                if fi < 2:
                    for fr in (0.28, 0.54, 0.78):
                        seg = cam.project_polyline(np.array([q[0] * (1 - fr) + q[3] * fr, q[1] * (1 - fr) + q[2] * fr]))
                        if len(seg) >= 2:
                            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in seg], base_w=1.8, z_depth=dist,
                                               rgba=(0.025, 0.010, 0.012, 0.80), rng=rng, taper=False, tremor=0.14)
        # Sunlit vertical corner edge highlight on right-front corner of crate
        cseg = cam.project_polyline(np.array([[x1, y0, z0], [x1, y0, z1]]))
        if len(cseg) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in cseg], base_w=1.8, z_depth=dist,
                               rgba=(*fog_mix([0.78, 0.38, 0.16], dist, fog_rgb), 0.68), rng=rng, taper=False, tremor=0.12)


def draw_seabirds_and_motes(ctx: cairo.Context, rng: np.random.Generator) -> None:
    """Delicate distant gliding harbor gulls in sunset silhouette + warm floating motes."""
    gulls = [
        (1140, 290, 15.5, -0.14),
        (1215, 260, 12.0, -0.08),
        (1310, 318, 13.5, 0.10),
        (1455, 236, 10.0, -0.16),
        (1535, 282, 9.0, 0.05),
        (1035, 336, 11.0, -0.06),
    ]
    for gx, gy, span, tilt in gulls:
        ctx.save()
        ctx.translate(gx, gy)
        ctx.rotate(tilt)
        ctx.set_source_rgba(0.13, 0.042, 0.052, 0.78)
        ctx.set_line_width(max(1.1, span * 0.13))
        ctx.move_to(-span, span * 0.18)
        ctx.curve_to(-span * 0.55, -span * 0.34, -span * 0.15, -span * 0.16, 0.0, span * 0.06)
        ctx.curve_to(span * 0.15, -span * 0.16, span * 0.55, -span * 0.34, span, span * 0.18)
        ctx.stroke()
        ctx.restore()

    for _ in range(85):
        mx = float(rng.uniform(80, W - 80))
        my = float(rng.uniform(120, H - 80))
        mr = float(rng.uniform(0.8, 2.2))
        ctx.set_source_rgba(1.0, 0.72, 0.38, float(rng.uniform(0.14, 0.40)))
        ctx.arc(mx, my, mr, 0, math.tau)
        ctx.fill()


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


def paint_foreground_bokeh_post(rng: np.random.Generator) -> np.ndarray:
    """
    Close out-of-focus foreground wooden mooring post & snug wrapped hawser rope
    framing the lower-right corner, convolved in premultiplied RGBA with a circular
    aperture disc kernel (r = 7.8) + ~1 px opposite R/B lateral shift.
    """
    surf, ctx = new_cairo_2x()

    post_poly = [
        (W - 210, H + 40),
        (W - 236, H - 345),
        (W - 188, H - 380),
        (W - 92, H - 384),
        (W - 42, H - 344),
        (W - 18, H + 40),
    ]
    g_p = cairo.LinearGradient(W - 242, H - 200, W - 26, H - 200)
    g_p.add_color_stop_rgba(0.0, 0.22, 0.082, 0.048, 0.98)
    g_p.add_color_stop_rgba(0.24, 0.040, 0.016, 0.018, 0.99)
    g_p.add_color_stop_rgba(1.0, 0.018, 0.008, 0.010, 0.99)
    ctx.set_source(g_p)
    ctx.move_to(*post_poly[0])
    for p in post_poly[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    ctx.fill()

    # Warm vermilion sunset edge grazing the left silhouette of the foreground post
    ctx.set_source_rgba(0.84, 0.34, 0.14, 0.66)
    ctx.set_line_width(5.5)
    ctx.move_to(W - 212, H + 10)
    ctx.line_to(W - 235, H - 342)
    ctx.line_to(W - 188, H - 376)
    ctx.stroke()

    # Snugly wrapped thick hemp rope coils hugging the post (no overhanging ribs!)
    for k in range(4):
        y_r = H - 268 + k * 42
        pts = [
            (W - 234 + k * 2, y_r - 8),
            (W - 172, y_r + 16),
            (W - 96, y_r + 14),
            (W - 38 + k * 2, y_r - 6),
        ]
        stroke_catmull_rom(ctx, pts, base_w=26.0, z_depth=0.5, rgba=(0.032, 0.014, 0.014, 0.98), rng=rng, taper=False, tremor=0.45)
        stroke_catmull_rom(ctx, pts[:3], base_w=6.5, z_depth=0.5, rgba=(0.64, 0.28, 0.12, 0.56), rng=rng, taper=True, tremor=0.35)

    # Heavy trailing catenary rope sagging gracefully toward lower-right frame bottom
    loop_pts = [
        (W - 232, H - 144),
        (W - 325, H - 56),
        (W - 410, H + 30),
    ]
    stroke_catmull_rom(ctx, loop_pts, base_w=24.0, z_depth=0.5, rgba=(0.025, 0.011, 0.012, 0.97), rng=rng, taper=False, tremor=0.40)
    stroke_catmull_rom(ctx, loop_pts, base_w=5.5, z_depth=0.5, rgba=(0.68, 0.30, 0.13, 0.58), rng=rng, taper=True, tremor=0.32)

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
    tint = np.array([1.14, 0.94, 0.78], dtype=np.float32)
    hal = (
        0.26 * gaussian_blur_2d(hl_src, 6.5)
        + 0.18 * gaussian_blur_2d(hl_src, 24.0)
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
    img *= np.clip(1.0 - 0.21 * (r2 ** 1.15), 0.72, 1.0)[..., None]

    # 3. Linear shadow black lift + C1-continuous tanh highlight shoulder (knee=0.80, ceiling=0.965)
    img = 0.012 + img * 0.988
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
    seed: int = 199608,
    audit_dir: pathlib.Path | None = None,
    webp_targets: list[pathlib.Path] | None = None,
) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # Stage 1: Off-axis pinhole Camera3D (VP ~59.4% width, horizon ~45.0% height)
    cam = Camera3D(W, H, eye=(1.55, 0.0, 2.52), target=(-0.12, 16.0, 2.02), fov_x_deg=58.0)
    print(f"[ember-01] Camera VP: ({cam.vp_x:.1f}, {cam.vp_y:.1f}) -> {cam.vp_x / W * 100:.1f}% width, horizon {cam.horizon_y / H * 100:.1f}% height")

    # Stage 1 & 2: Sunset sky, painterly clouds, rugged headland, lighthouse & copper Fresnel bay
    canvas, fog_rgb, water_ripple = paint_sky_headland_and_copper_bay(cam, rng)

    # Stage 2: 3D Plane-Mirrored Water Reflections (pilings & 3/4 moored boat) warped by capillary waves!
    s_refl, c_refl = new_cairo_2x()
    for py_p in (2.8, 6.2, 10.0, 14.5, 19.6, 25.4):
        draw_timber_piling(c_refl, cam, 1.07, py_p, Z_PIER + 0.34, 0.14, rng, fog_rgb, mirror=True)
    draw_moored_wooden_boat(c_refl, cam, rng, fog_rgb, mirror=True)
    refl_1x = finish_cairo_2x(s_refl)

    yy_w, xx_w = np.mgrid[0:H, 0:W].astype(np.float32)
    hy = float(cam.horizon_y)
    dist_t = np.power(np.clip((yy_w - hy) / max(1.0, H - hy), 0.0, 1.0), 0.72)
    fresnel_m = (0.34 + (0.86 - 0.34) * np.power(1.0 - dist_t, 1.4)) * (yy_w >= hy).astype(np.float32)
    rx_map = np.clip(xx_w + water_ripple * (3.0 + 22.0 * dist_t), 0.0, W - 1.0)
    ry_map = np.clip(yy_w + water_ripple * (1.2 + 7.5 * dist_t), 0.0, H - 1.0)
    refl_rippled = remap_bilinear(refl_1x, rx_map, ry_map) * fresnel_m[..., None]
    canvas = over_premul(canvas, refl_rippled)

    # Stage 1 & 2: Exact 3x3 Plane-Homography Architectural Surfaces
    seawall_uv = paint_voronoi_basalt_uv(rng, (1.2, 34.0), (Z_WATER, Z_PIER), plane_kind="seawall_vert", tw=896, th=320)
    canvas = cam.warp_plane_over(
        canvas, seawall_uv,
        p0=np.array([0.94, 1.2, Z_WATER]),
        e1=np.array([0.0, 32.8, 0.0]),
        e2=np.array([0.0, 0.0, Z_PIER - Z_WATER]),
        fog_rgb=fog_rgb, fog_start=7.5, fog_density=0.036,
    )

    pier_uv = paint_timber_pier_uv(rng, (WAREHOUSE_X_WALL, -0.18), (1.2, 34.0), tw=768, th=1024)
    canvas = cam.warp_plane_over(
        canvas, pier_uv,
        p0=np.array([WAREHOUSE_X_WALL, 1.2, Z_PIER]),
        e1=np.array([-0.18 - WAREHOUSE_X_WALL, 0.0, 0.0]),
        e2=np.array([0.0, 32.8, 0.0]),
        fog_rgb=fog_rgb, fog_start=7.5, fog_density=0.036,
    )

    quay_uv = paint_voronoi_basalt_uv(rng, (-0.18, 0.96), (1.2, 34.0), plane_kind="quay_cap", tw=448, th=1024)
    canvas = cam.warp_plane_over(
        canvas, quay_uv,
        p0=np.array([-0.18, 1.2, Z_PIER + 0.004]),
        e1=np.array([1.14, 0.0, 0.0]),
        e2=np.array([0.0, 32.8, 0.0]),
        fog_rgb=fog_rgb, fog_start=7.5, fog_density=0.036,
    )

    wall_uv = paint_warehouse_wall_uv(rng, (WAREHOUSE_Y0, WAREHOUSE_Y1), (Z_PIER, 4.65), tw=896, th=448)
    canvas = cam.warp_plane_over(
        canvas, wall_uv,
        p0=np.array([WAREHOUSE_X_WALL, WAREHOUSE_Y0, Z_PIER]),
        e1=np.array([0.0, WAREHOUSE_Y1 - WAREHOUSE_Y0, 0.0]),
        e2=np.array([0.0, 0.0, 4.65 - Z_PIER]),
        fog_rgb=fog_rgb, fog_start=7.5, fog_density=0.036,
    )

    # Shared isotropic gouache + flat-brush texture pass for Stage 3 PyCairo fills
    g_mottle = fbm_gouache(H, W, rng)
    g_dabs = flat_brush_dabs(H, W, rng, count=88)
    g_tooth = paper_tooth(H, W, rng)

    # Stage 3 Band 0 (Far Harbor Pier End, Far Pilings, Distant Gulls & Far Bollards)
    s_far, c_far = new_cairo_2x()
    draw_seabirds_and_motes(c_far, rng)
    for py_p in (25.4, 19.6):
        draw_timber_piling(c_far, cam, 1.07, py_p, Z_PIER + 0.34, 0.14, rng, fog_rgb, mirror=False)
    for bx, by in BOLLARD_COORDS[3:]:
        draw_iron_bollard(c_far, cam, bx, by, rng, fog_rgb, with_coil=False)
    canvas = over_premul(canvas, finish_cairo_2x(s_far))

    # Volumetric golden-hour sunbeams streaming from low sun across headland & harbor
    beams = paint_golden_sunbeams(cam, rng, intensity=0.36)
    canvas = 1.0 - (1.0 - np.clip(canvas, 0.0, 1.0)) * (1.0 - np.clip(beams, 0.0, 0.85))

    # Stage 3 Band 1 (Midground 3/4 Moored Wooden Boat, Mid Pilings & Mid Bollard)
    s_mid, c_mid = new_cairo_2x()
    draw_moored_wooden_boat(c_mid, cam, rng, fog_rgb, mirror=False)
    for py_p in (14.5, 10.0):
        draw_timber_piling(c_mid, cam, 1.07, py_p, Z_PIER + 0.36, 0.145, rng, fog_rgb, mirror=False)
    draw_iron_bollard(c_mid, cam, *BOLLARD_COORDS[2], rng, fog_rgb, with_coil=True)
    mid_1x = finish_cairo_2x(s_mid)
    ma = mid_1x[:, :, 3:4]
    mid_1x[:, :, :3] = np.clip(mid_1x[:, :, :3] + ma * (g_mottle * 0.018 + g_dabs * 0.012 + g_tooth * 0.006)[..., None], 0.0, 1.3)
    canvas = over_premul(canvas, mid_1x)

    # Stage 3 Band 2 (Left Wooden Warehouse Overhanging Eaves, Lanterns, Crates & Near Bollards/Pilings)
    s_near, c_near = new_cairo_2x()
    draw_warehouse_eaves_and_props(c_near, cam, rng, fog_rgb)
    for py_p in (6.2, 2.8):
        draw_timber_piling(c_near, cam, 1.07, py_p, Z_PIER + 0.38, 0.15, rng, fog_rgb, mirror=False)
    draw_iron_bollard(c_near, cam, *BOLLARD_COORDS[1], rng, fog_rgb, with_coil=True)
    draw_iron_bollard(c_near, cam, *BOLLARD_COORDS[0], rng, fog_rgb, with_coil=True)
    near_1x = finish_cairo_2x(s_near)
    na = near_1x[:, :, 3:4]
    near_1x[:, :, :3] = np.clip(near_1x[:, :, :3] + na * (g_mottle * 0.018 + g_dabs * 0.013 + g_tooth * 0.006)[..., None], 0.0, 1.3)
    canvas = over_premul(canvas, near_1x)

    # Stage 4: Lower-right foreground circular-bokeh mooring post & wrapped rope + 35mm analog finish
    fg_bokeh = paint_foreground_bokeh_post(rng)
    canvas = over_premul(canvas, fg_bokeh)

    final_rgb = apply_35mm_finish(canvas, rng)
    elapsed = time.perf_counter() - t0
    print(f"[ember-01] Rendered {W}x{H} in {elapsed:.2f}s")

    u8 = np.round(np.clip(final_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    full_im = Image.fromarray(u8, mode="RGB")

    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        full_im.save(audit_dir / "w01_full.png", optimize=True)
        crops = {
            "w01_crop_aperture.png": (int(W * 0.50), int(H * 0.22)),      # Vermilion sunset, volcanic headland & lighthouse
            "w01_crop_midground.png": (int(W * 0.45), int(H * 0.50)),     # 3/4 Moored wooden boat, pilings, copper ripples & basalt seawall
            "w01_crop_foreground.png": (W - 700, H - 450),                # Near bollard/wet pier & lower-right disc-bokeh mooring post
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
        print(f"[ember-01] Saved {W}x{H} WebP: {first} ({first.stat().st_size / 1024:.1f} KiB)")
        for extra in targets[1:]:
            extra.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(first, extra)
            print(f"[ember-01] Copied {W}x{H} WebP: {extra} ({extra.stat().st_size / 1024:.1f} KiB)")

    return final_rgb, {"elapsed": elapsed, "vp_ratio": cam.vp_x / W}


def main() -> None:
    ap = argparse.ArgumentParser(description="Render Ember 01 (Crimson Harbor Pier) 2048x1152 wallpaper")
    ap.add_argument("--seed", type=int, default=199608)
    ap.add_argument("--audit-dir", type=pathlib.Path, default=None)
    ap.add_argument("--no-webp", action="store_true")
    args = ap.parse_args()
    targets = [] if args.no_webp else [DEFAULT_THEME_WEBP, DEFAULT_DOCS_WEBP]
    render_scene(seed=args.seed, audit_dir=args.audit_dir, webp_targets=targets)


if __name__ == "__main__":
    main()

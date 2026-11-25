#!/usr/bin/env python3
"""
Matcha 01 — "Sunlit Botanical Conservatory" (2048 x 1152)

1990s Studio Ghibli (Arrietty / Whisper of the Heart) hand-painted Meiji/Victorian
arched wrought-iron & sage-painted botanical glasshouse interior at golden morning
light, synthesized via a 5-stage Pure-Python 3D-grounded pipeline:

  Stage 1: Metric right-handed Camera3D (off-axis vanishing point ~58.1% width),
           near-plane-safe (z_cam > 0.05) exact 3x3 UV-to-screen homographies for
           all planar conservatory surfaces (wet Voronoi mossy slate flagstones
           with broad shallow watering puddles, left & right mossy brick/stone
           kneewalls + tall arched condensation-misted glass bays, arched rear
           glass gable wall with wide open garden doorway & swung French doors,
           right raised mossy stone planter bed, left weathered cedar potting
           workbench & stepped plant shelves), exponential morning conservatory
           mist haze, and 3D near-plane clipping.
  Stage 2: Single locked normalized morning SUN_DIR (SUN_DIR[2] < 0, slanting
           diagonally down-right from upper-left arched glass vault across the
           potting bench & wet aisle) traced backward with 6 height-refinement
           steps across the elliptical vaulted glass roof ribs, mullions, left
           arched windows, and potting bench slats for 100% geometric alignment
           of crisp arched light patches, cast shadows, and volumetric golden-jade
           sunbeams; plus broad floor watering puddles combining cv2.remap
           micro-rippled reflections with 3D plane-mirrored quads weighted by
           exact Fresnel reflection.
  Stage 3: Depth-interleaved 2x supersampled PyCairo bands (Lanczos-downsampled)
           with seeded cubic-Bezier botanical curves (zero stamped polygons!):
             Band 0 — Luminous pale sage-cream misty morning garden visible
                      through the condensation-misted glass & open arched doorway
             Band 1 — Soaring vaulted sage wrought-iron conservatory ribs, pierced
                      filigree gussets, curved glass skylight sheen, hanging iron/
                      macrame fern baskets on taut cables & vintage brass pendant lamp
             Band 2 — Weathered cedar potting workbench, tiered wooden plant racks,
                      varied hand-thrown terracotta pots, matcha-glazed ceramic
                      planters, gleaming copper long-spout watering can, glass
                      propagation jars with submerged cuttings, wooden seed trays
                      & forged garden tools — all grounded with feathered contact
                      AO + unified SUN_DIR cast shadows
             Band 3 — Rich organic botanical specimens via seeded cubic Beziers:
                      fenestrated monstera leaves, cascading Boston/sword fern
                      fronds, climbing English ivy vines on wrought-iron columns,
                      broad backlit banana/bird-of-paradise leaves on the right
                      raised stone planter bed & graceful misty palms near doorway
           Enriched by isotropic square-scale fbm_gouache, angled flat-brush dabs,
           cold-press paper tooth, and warm golden-olive shadow bounce (lum < 0.28).
  Stage 4: Near-camera foreground wet monstera & fern fronds convolved in
           premultiplied RGBA with circular disc bokeh kernel (r = 7.8) + separate
           cv2.warpAffine ~1px opposite R/B lateral shift; followed by float32
           35mm optical finish (3-scale cream-sage halation, radial corner CA,
           optical vignette, shadow black lift + C1 tanh shoulder, silver-halide
           grain, faint YIQ chroma softening with BORDER_REFLECT_101, sub-pixel
           tape jitter, <= 0.75% scanlines).
  Stage 5: Deterministic CPU render (<60s), exact Camera3D UV readout for GLSL
           shader calibration, 2048x1152 WebP export (quality=94), and 1:1 macro
           audit crops.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import time
from typing import Iterable

import cairo
import cv2
import numpy as np
from PIL import Image

W, H = 2048, 1152
Z_NEAR = 0.05

# =============================================================================
# Stage 1 & 2 World Metric Constants (Right-handed meters: +X right, +Y fwd, +Z up)
# =============================================================================

# Single locked normalized morning SUN_DIR: high upper-left -> down-right (SUN_DIR[2] < 0)
_raw_sun = np.array([0.68, -0.22, -0.699], dtype=np.float64)
SUN_DIR = _raw_sun / np.linalg.norm(_raw_sun)

# Conservatory structural dimensions (meters)
X_LEFT = -2.65
X_RIGHT = 2.65
HALF_SPAN = 2.65
Y_FRONT = 0.40
Y_REAR = 14.50

Z_FLOOR = 0.00
Z_KNEE = 0.90       # Top of mossy brick/stone kneewall coping
Z_SPRING = 2.92     # Height where vertical arched windows spring into barrel vault
ARCH_RISE = 1.66    # Elliptical vault arch rise above spring line
Z_PEAK = Z_SPRING + ARCH_RISE  # 4.58 m ridge peak

# Structural wrought-iron bay rib Y positions along the nave
BAY_YS = (1.70, 3.80, 5.90, 8.00, 10.10, 12.20, 14.30)

# Open arched conservatory garden doorway on rear gable wall (y = Y_REAR)
DOOR_HW = 1.04      # Half-width -> 2.08 m wide open double archway
DOOR_SPRING = 1.98  # Doorway vertical jamb height
DOOR_TOP = 2.86     # Semicircular arch crown

# Left weathered cedar potting workbench (exact 3D bounds)
BENCH_X0 = -2.54
BENCH_X1 = -1.24
BENCH_Y0 = 2.35
BENCH_Y1 = 9.55
BENCH_Z = 0.86      # Tabletop surface height
BENCH_SHELF_Z = 0.34
BENCH_THICK = 0.055

# Right raised mossy stone planter bed (exact 3D bounds)
PLANTER_X0 = 1.46
PLANTER_X1 = 2.62
PLANTER_Y0 = 2.10
PLANTER_Y1 = 13.90
PLANTER_Z = 0.56


def vault_z(wx: np.ndarray | float) -> np.ndarray | float:
    """Smooth elliptical Meiji/Victorian conservatory barrel-vault roof profile."""
    u = np.clip(np.asarray(wx, dtype=np.float64) / HALF_SPAN, -0.9999, 0.9999)
    # Slight graceful pointed Victorian lift near crown
    ell = np.sqrt(np.maximum(0.0, 1.0 - u * u))
    lift = 0.10 * np.exp(-((u / 0.24) ** 2))
    out = Z_SPRING + ARCH_RISE * ell + lift
    return float(out) if np.ndim(wx) == 0 else out


def puddle_depth_field(wx: np.ndarray, wy: np.ndarray) -> np.ndarray:
    """
    Deterministic organic broad shallow watering-puddle depth field (0..1) on the
    conservatory slate floor (z = 0). Forms one graceful connected S-curved sheet
    of shallow rainwater/watering runoff down the center-right flagstone aisle plus
    a delicate sunlit spill near the potting bench — zero hard internal seams!
    """
    # Centerline meanders smoothly down the center-right flagstone aisle
    cx_aisle = 0.10 + 0.16 * np.sin(wy * 0.48 - 0.85) + 0.08 * np.cos(wy * 0.92 + 0.4)
    # Organic variable half-width tapering gracefully near foreground and doorway
    hw_aisle = (
        0.84
        + 0.16 * np.sin(wy * 0.72 + 0.6)
        - 0.24 * np.exp(-(((wy - 8.2) / 1.15) ** 2))  # natural narrow waist
    )
    wob = (
        0.13 * np.sin(wy * 1.18 + wx * 1.70 + 0.35)
        + 0.09 * np.cos(wx * 2.55 - wy * 0.88 + 1.15)
        + 0.05 * np.sin(wy * 2.60 + wx * 1.35)
    )
    dx_main = (wx - cx_aisle) / np.maximum(0.32, hw_aisle)
    dy_main = (wy - 7.65) / 5.55
    p_main = 1.0 - (dx_main * dx_main + np.power(np.abs(dy_main), 2.4) + wob)

    # Secondary sunlit watering splash near the foreground potting bench edge
    dx_sp = (wx + 0.44) / 0.56
    dy_sp = (wy - 4.15) / 1.35
    p_spill = 0.92 - (dx_sp * dx_sp + dy_sp * dy_sp + 0.11 * np.sin(wy * 2.1 + wx * 2.5))

    # Smooth union (soft max) so puddle lobes flow seamlessly together without an internal seam
    k_sm = 0.28
    h_sm = np.clip(0.5 + 0.5 * (p_spill - p_main) / k_sm, 0.0, 1.0)
    raw = p_main * (1.0 - h_sm) + p_spill * h_sm + k_sm * h_sm * (1.0 - h_sm)

    # Smoothly keep puddles strictly off the bench legs & right raised stone planter curb
    margin_l = np.clip((wx - (BENCH_X1 + 0.08)) / 0.26, 0.0, 1.0)
    margin_r = np.clip(((PLANTER_X0 - 0.10) - wx) / 0.24, 0.0, 1.0)
    margin_f = np.clip((wy - 2.25) / 0.60, 0.0, 1.0)
    margin_b = np.clip((13.85 - wy) / 0.75, 0.0, 1.0)
    bounds = margin_l * margin_r * margin_f * margin_b
    return np.clip(raw * bounds, 0.0, 1.0).astype(np.float32)


# =============================================================================
# Core Vectorized Image & Painterly Gouache Primitives
# =============================================================================

def resize_bicubic(arr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    return cv2.resize(arr.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_CUBIC)


def gaussian_blur_2d(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.08:
        return arr.copy()
    if sigma <= 10.0:
        return cv2.GaussianBlur(
            arr.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101
        )
    h, w = arr.shape[:2]
    factor = 4 if sigma < 36.0 else 8
    sw, sh = max(8, w // factor), max(8, h // factor)
    small = cv2.resize(arr.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    s_sig = sigma / factor
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=s_sig, sigmaY=s_sig, borderType=cv2.BORDER_REFLECT_101)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def gaussian_blur_1d_x(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.05:
        return arr.copy()
    ksize = max(3, int(math.ceil(sigma * 3.0)) * 2 + 1)
    return cv2.GaussianBlur(
        arr.astype(np.float32), (ksize, 1), sigmaX=sigma, sigmaY=0.0, borderType=cv2.BORDER_REFLECT_101
    )


def remap_reflect(src: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        src.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def fbm_gouache(
    h: int,
    w: int,
    rng: np.random.Generator,
    scales=((72, 70), (27, 29), (10.5, 10.0), (4.6, 4.8)),
) -> np.ndarray:
    """Isotropic square-scale multi-octave gouache mottling (zero row/col stretching!)."""
    acc = np.zeros((h, w), dtype=np.float32)
    weights = (0.46, 0.28, 0.17, 0.09)
    for (sy, sx), wt in zip(scales, weights):
        gh = max(4, int(round(h / sy)))
        gw = max(4, int(round(w / sx)))
        acc += wt * resize_bicubic(rng.standard_normal((gh, gw)).astype(np.float32), w, h)
    acc -= acc.mean()
    return acc / (float(acc.std()) + 1e-6)


def paper_tooth(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    coarse = resize_bicubic(rng.standard_normal((max(4, h // 3), max(4, w // 3))).astype(np.float32), w, h)
    fine = rng.standard_normal((h, w)).astype(np.float32)
    t = 0.56 * coarse + 0.44 * fine
    t -= t.mean()
    return t / (float(t.std()) + 1e-6)


def flat_brush_dabs(h: int, w: int, rng: np.random.Generator, count: int = 92) -> np.ndarray:
    """85-100 angled flat-brush gouache dabs rendered at 1/4 res and bicubic upsampled."""
    qh, qw = max(16, h // 4), max(16, w // 4)
    arr_sum = np.zeros((qh, qw), dtype=np.float32)
    s_one = cairo.ImageSurface(cairo.FORMAT_A8, qw, qh)
    c1 = cairo.Context(s_one)
    stride = s_one.get_stride()
    for _ in range(count):
        cx = float(rng.uniform(0, qw))
        cy = float(rng.uniform(0, qh))
        ang = float(rng.uniform(0, math.pi))
        rx = float(rng.uniform(qw * 0.032, qw * 0.096))
        ry = float(rng.uniform(rx * 0.26, rx * 0.58))
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
            a = (math.tau * k) / n_pts + float(rng.uniform(-0.20, 0.20))
            rj = float(rng.uniform(0.76, 1.18))
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
    arr_sum = gaussian_blur_2d(arr_sum, 2.8)
    arr_sum -= arr_sum.mean()
    arr_sum /= float(arr_sum.std()) + 1e-6
    return resize_bicubic(arr_sum, w, h)


def voronoi_metric_grid(
    wx: np.ndarray, wy: np.ndarray, cell_size: float, jitter: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized metric 3x3 jittered Voronoi returning (d1, edge_dist = d2 - d1, cell_hash).
    Used for physical-meter mossy slate conservatory flagstones.
    """
    gx = wx / cell_size
    gy = wy / cell_size
    ix0 = np.floor(gx).astype(np.int32)
    iy0 = np.floor(gy).astype(np.int32)

    # Deterministic hash table for 3x3 neighbor cell sites
    tab_x = rng.uniform(0.5 - jitter, 0.5 + jitter, size=(64, 64)).astype(np.float64)
    tab_y = rng.uniform(0.5 - jitter, 0.5 + jitter, size=(64, 64)).astype(np.float64)
    tab_id = rng.uniform(0.0, 1.0, size=(64, 64)).astype(np.float32)

    d1 = np.full_like(wx, 1e9, dtype=np.float64)
    d2 = np.full_like(wx, 1e9, dtype=np.float64)
    best_id = np.zeros_like(wx, dtype=np.float32)

    for ox in (-1, 0, 1):
        nx = ix0 + ox
        hx = nx & 63
        for oy in (-1, 0, 1):
            ny = iy0 + oy
            hy = ny & 63
            sx = nx + tab_x[hx, hy]
            sy = ny + tab_y[hx, hy]
            dist = np.hypot(gx - sx, gy - sy) * cell_size
            closer1 = dist < d1
            d2 = np.where(closer1, d1, np.minimum(d2, dist))
            d1 = np.where(closer1, dist, d1)
            best_id = np.where(closer1, tab_id[hx, hy], best_id)

    return d1.astype(np.float32), (d2 - d1).astype(np.float32), best_id


def over_premul(dst_rgb: np.ndarray, src_premul_rgba: np.ndarray) -> np.ndarray:
    a = np.clip(src_premul_rgba[:, :, 3:4], 0.0, 1.0)
    return src_premul_rgba[:, :, :3] + dst_rgb * (1.0 - a)


def apply_shadow_bounce(rgb: np.ndarray, bounce_rgb=(0.025, 0.029, 0.017)) -> np.ndarray:
    """Warm golden-olive greenhouse shadow bounce in deep shadows (lum < 0.28)."""
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    w_sh = np.clip((0.28 - lum) / 0.28, 0.0, 1.0)
    return rgb + (w_sh * w_sh)[..., None] * np.array(bounce_rgb, dtype=np.float32)


def fog_mix(
    c3: Iterable[float],
    dist: float,
    fog_rgb: np.ndarray,
    fog_start: float = 4.5,
    fog_density: float = 0.046,
) -> tuple[float, float, float]:
    ft = 1.0 - math.exp(-max(0.0, dist - fog_start) * fog_density)
    c = np.asarray(c3, dtype=np.float64) * (1.0 - ft) + fog_rgb * ft
    return (float(c[0]), float(c[1]), float(c[2]))


# =============================================================================
# Stage 1 — Metric Right-Handed Camera3D & Near-Plane-Safe 3x3 Homographies
# =============================================================================

class Camera3D:
    def __init__(self, w: int, h: int, eye: Iterable[float], target: Iterable[float], fov_x_deg: float = 61.5):
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

    def project_uv(self, pt: np.ndarray) -> tuple[float, float, float]:
        u, v, z = self.project(pt)
        return float(u) / self.w, float(v) / self.h, float(z)

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
        fog_start: float = 4.5,
        fog_density: float = 0.046,
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

        u_cl = np.clip(u_px, 0.0, tw - 1.001).astype(np.float32)
        v_cl = np.clip(v_px, 0.0, th - 1.001).astype(np.float32)
        sampled = cv2.remap(
            tex_premul_rgba.astype(np.float32),
            u_cl,
            v_cl,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        v_f = valid[..., None].astype(np.float32)
        alpha = sampled[:, :, 3:4] * v_f
        rgb_valid = sampled[:, :, :3] * v_f

        fg_rgb = rgb_valid * (1.0 - fog_t[..., None]) + (fog_rgb[None, None, :] * alpha) * fog_t[..., None]
        patch = np.concatenate([fg_rgb, alpha], axis=-1)
        dst_rgb[y0:y1, x0:x1] = over_premul(dst_rgb[y0:y1, x0:x1], patch)
        return dst_rgb


def mirror_pts_across_plane(pts: list[np.ndarray], z_plane: float = Z_FLOOR) -> list[np.ndarray]:
    """Exact 3D reflection across horizontal water plane z = z_plane."""
    p0 = np.array([0.0, 0.0, z_plane], dtype=np.float64)
    n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return [p - 2.0 * float(np.dot(p - p0, n)) * n for p in pts]


# =============================================================================
# Stage 2 — Unified 3D Morning Sunlight Solver (6-Step Vault & Mullion Trace)
# =============================================================================

def outdoor_sun_dapple(
    ex: np.ndarray, ey: np.ndarray, ez: np.ndarray
) -> np.ndarray:
    """
    Deterministic outdoor botanical garden tree-canopy & trellis transmission field
    on the conservatory glass envelope (shared 100% identically by surface sun-hits
    and 3D volumetric sunbeam shafts so rays & surface light pools lock seamlessly!).
    Creates authentic Studio Ghibli rhythm of blazing morning window shafts and
    softly dappled foliage troughs across Bays 2..5 & the upper-left vaulted skylight.
    """
    # Organic outdoor garden canopy modulation across the glass panes (sculpted shafts + branch shadows)
    wave = (
        0.50 * np.sin(ey * 0.88 + ez * 1.12 - ex * 0.65 + 0.55)
        + 0.32 * np.cos(ey * 1.72 - ez * 1.38 + ex * 1.05 + 1.15)
        + 0.18 * np.sin(ey * 2.65 + ez * 2.05 + 0.25)
    )
    dapple = np.clip(0.76 + 0.42 * wave, 0.26, 1.14)

    # Distinct sculptural hero-bay morning openings in Bays 2, 3, 4 & 5 so each window
    # bay pours an unmistakable golden sunbeam shaft across the conservatory!
    hero_bays = (
        0.34 * np.exp(-(((ey - 4.82) / 0.82) ** 2))
        + 0.38 * np.exp(-(((ey - 6.92) / 0.82) ** 2))
        + 0.28 * np.exp(-(((ey - 9.05) / 0.86) ** 2))
        + 0.20 * np.exp(-(((ey - 11.15) / 0.90) ** 2))
    )
    return np.clip(dapple + hero_bays, 0.24, 1.28)


def trace_conservatory_sunlight(
    wx: np.ndarray,
    wy: np.ndarray,
    wz: np.ndarray,
    include_bench: bool = True,
) -> np.ndarray:
    """
    Trace any 3D conservatory floor, potting bench, wall, or planter coordinate backward
    along -SUN_DIR (6 height-refinement steps across the arched wrought-iron glass
    vault, left arched glass windows, mullions, bay ribs, and potting bench slats) so
    bright arched window patches, rib shadows, and volumetric sunbeams align in exact 3D!
    """
    inv_sz = 1.0 / (-SUN_DIR[2])
    # 6-step height refinement against the curved elliptical barrel vault z = vault_z(x)
    z_roof = np.full_like(wx, 3.85, dtype=np.float64)
    for _ in range(6):
        t_r = np.maximum(0.0, (z_roof - wz) * inv_sz)
        sx_r = np.clip(wx - SUN_DIR[0] * t_r, X_LEFT, X_RIGHT)
        z_roof = vault_z(sx_r)

    t_r = np.maximum(0.0, (z_roof - wz) * inv_sz)
    sx_r = wx - SUN_DIR[0] * t_r
    sy_r = wy - SUN_DIR[1] * t_r

    # Intersection with the tall left arched glass wall (x = X_LEFT)
    t_w = (wx - X_LEFT) / SUN_DIR[0]
    zw = wz - SUN_DIR[2] * t_w
    yw = wy - SUN_DIR[1] * t_w
    hits_left_wall = (t_w > 0.0) & (zw <= Z_SPRING)

    # Effective exit point (either through left side wall or arched glass vault)
    ey = np.where(hits_left_wall, yw, sy_r)
    ex = np.where(hits_left_wall, X_LEFT, sx_r)
    ez = np.where(hits_left_wall, zw, z_roof)

    # 1. Bay wrought-iron structural ribs & columns at BAY_YS (pitch = 2.10 m) — natural solar penumbra!
    bay_phase = np.mod(ey - BAY_YS[0], 2.10)
    dist_rib = np.minimum(bay_phase, 2.10 - bay_phase)
    rib_shadow = np.clip((dist_rib - 0.056) / 0.072, 0.0, 1.0)

    # 2. Delicate secondary glazing mullions (3 vertical panes per 2.10 m bay -> 0.70 m spacing)
    mull_phase = np.mod(ey - BAY_YS[0], 0.70)
    dist_mull = np.minimum(mull_phase, 0.70 - mull_phase)
    mull_shadow = np.clip((dist_mull - 0.015) / 0.044, 0.0, 1.0)

    # 3. Left wall arched window head, horizontal transoms & solid mossy brick kneewall
    above_knee = np.clip((zw - (Z_KNEE + 0.02)) / 0.068, 0.0, 1.0)
    # Semicircular Victorian arch spandrel inside top of each 2.10 m bay (z in [2.32, 2.92])
    bay_u = (bay_phase - 1.05) / 0.96
    arch_z_top = 2.34 + 0.56 * np.sqrt(np.maximum(0.0, 1.0 - np.clip(bay_u, -1.0, 1.0) ** 2))
    under_arch = np.clip((arch_z_top - zw) / 0.072, 0.0, 1.0)
    # Horizontal window transoms at z = 1.68 and z = 2.34
    trans_d = np.minimum(np.abs(zw - 1.68), np.abs(zw - 2.34))
    trans_wall = np.clip((trans_d - 0.020) / 0.048, 0.0, 1.0)
    wall_trans = above_knee * under_arch * trans_wall

    # 4. Vaulted glass roof longitudinal iron purlins along x-arc
    purlin_x = np.array([-2.22, -1.58, -0.82, 0.0, 0.82, 1.58, 2.22], dtype=np.float64)
    min_px = np.full_like(ex, 10.0)
    for px_v in purlin_x:
        min_px = np.minimum(min_px, np.abs(ex - px_v))
    roof_purlin = np.clip((min_px - 0.028) / 0.052, 0.0, 1.0)
    # Spring eave cornice beam shadow right at z = Z_SPRING
    spring_sh = np.clip((np.abs(zw - Z_SPRING) - 0.068) / 0.078, 0.0, 1.0)
    roof_trans = roof_purlin * np.where(sx_r < X_LEFT + 0.14, spring_sh, 1.0)

    # Combine wall vs vault aperture cleanly
    glass_open = np.where(hits_left_wall, wall_trans, roof_trans)
    sun = rib_shadow * (0.28 + 0.72 * mull_shadow) * glass_open

    # Shared outdoor garden foliage dapple shining through conservatory glass
    sun *= outdoor_sun_dapple(ex, ey, ez)

    # Rear gable wall blocks rays coming from beyond y > Y_REAR except through open doorway
    past_rear = ey > Y_REAR
    sun = np.where(past_rear, sun * 0.32, sun)

    # 5. Cast shadow of the left weathered cedar potting workbench onto floor (wz < BENCH_Z - 0.02)
    if include_bench:
        t_b = (BENCH_Z - wz) * inv_sz
        bx = wx - SUN_DIR[0] * t_b
        by = wy - SUN_DIR[1] * t_b
        in_bench_top = (
            (wz < BENCH_Z - 0.02)
            & (t_b > 0.0)
            & (bx >= BENCH_X0)
            & (bx <= BENCH_X1)
            & (by >= BENCH_Y0)
            & (by <= BENCH_Y1)
        )
        # Soft penumbra around bench top edge
        dx_b = np.minimum(bx - BENCH_X0, BENCH_X1 - bx)
        dy_b = np.minimum(by - BENCH_Y0, BENCH_Y1 - by)
        d_in = np.minimum(dx_b, dy_b)
        bench_sh = np.where(in_bench_top, np.clip(1.0 - np.clip(d_in / 0.075, 0.0, 1.0) * 0.94, 0.06, 1.0), 1.0)
        sun *= bench_sh

        # Lower slatted shelf subtle secondary cast slats
        t_s = (BENCH_SHELF_Z - wz) * inv_sz
        sx_s = wx - SUN_DIR[0] * t_s
        sy_s = wy - SUN_DIR[1] * t_s
        in_shelf = (
            (wz < BENCH_SHELF_Z - 0.02)
            & (t_s > 0.0)
            & (sx_s >= BENCH_X0 + 0.08)
            & (sx_s <= BENCH_X1 - 0.06)
            & (sy_s >= BENCH_Y0 + 0.06)
            & (sy_s <= BENCH_Y1 - 0.06)
        )
        slat_gap = np.clip((np.mod(sx_s, 0.14) - 0.095) / 0.035, 0.0, 1.0)
        shelf_sh = np.where(in_shelf, 0.22 + 0.56 * slat_gap, 1.0)
        sun *= shelf_sh

    return np.clip(sun, 0.0, 1.15).astype(np.float32)


# =============================================================================
# Stage 3 Helpers — 2x Supersampled PyCairo & Seeded Cubic-Bezier Primitives
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
    """Lanczos-downsample 2x PyCairo surface to 1x with clean unpremultiply."""
    surf.flush()
    buf = np.frombuffer(surf.get_data(), dtype=np.uint8).reshape(H * 2, W * 2, 4)
    rgba_2x = np.stack([buf[:, :, 2], buf[:, :, 1], buf[:, :, 0], buf[:, :, 3]], axis=-1).astype(np.float32) / 255.0
    a2 = rgba_2x[:, :, 3:4]
    rgb_unpremul = np.where(a2 > 1e-4, rgba_2x[:, :, :3] / np.maximum(a2, 1e-4), 0.0)
    rgb_1x = np.clip(cv2.resize(rgb_unpremul, (W, H), interpolation=cv2.INTER_LANCZOS4), 0.0, 1.0)
    a_1x = np.clip(cv2.resize(rgba_2x[:, :, 3], (W, H), interpolation=cv2.INTER_LANCZOS4), 0.0, 1.0)
    out = np.empty((H, W, 4), dtype=np.float32)
    out[:, :, :3] = rgb_1x * a_1x[..., None]
    out[:, :, 3] = a_1x
    return out


def fill_3d_quad(
    ctx: cairo.Context,
    cam: Camera3D,
    pts3: list[np.ndarray],
    rgba: tuple[float, float, float, float],
    grad_rgba: tuple[float, float, float, float] | None = None,
    grad_idx: tuple[int, int] = (0, -1),
) -> None:
    cl = cam.clip_polygon_3d(pts3)
    if len(cl) < 3:
        return
    proj = [cam.project(p) for p in cl]
    if grad_rgba is not None:
        i0, i1 = grad_idx
        g = cairo.LinearGradient(float(proj[i0][0]), float(proj[i0][1]), float(proj[i1][0]), float(proj[i1][1]))
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


def stroke_3d_segment(
    ctx: cairo.Context,
    cam: Camera3D,
    p0: np.ndarray,
    p1: np.ndarray,
    world_w: float,
    rgba: tuple[float, float, float, float],
    min_px: float = 0.7,
) -> None:
    cl = cam.clip_segment_3d(p0, p1)
    if cl is None:
        return
    a, b = cl
    ua, va, za = cam.project(a)
    ub, vb, zb = cam.project(b)
    zm = max(0.5, 0.5 * (float(za) + float(zb)))
    w_px = max(min_px, world_w * cam.K[0, 0] / zm)
    ctx.set_source_rgba(*rgba)
    ctx.set_line_width(w_px)
    ctx.move_to(float(ua), float(va))
    ctx.line_to(float(ub), float(vb))
    ctx.stroke()


def stroke_3d_ribbon(
    ctx: cairo.Context,
    cam: Camera3D,
    pts3: list[np.ndarray],
    world_w: float,
    dark_c: tuple[float, float, float],
    mid_c: tuple[float, float, float],
    lit_c: tuple[float, float, float],
    alpha: float = 1.0,
) -> None:
    """Draw a perspective-tapered shaded 3D wrought-iron structural curve/rib."""
    for i in range(len(pts3) - 1):
        cl = cam.clip_segment_3d(pts3[i], pts3[i + 1])
        if cl is None:
            continue
        a, b = cl
        ua, va, za = cam.project(a)
        ub, vb, zb = cam.project(b)
        wa = max(1.1, world_w * cam.K[0, 0] / max(0.6, float(za)))
        wb = max(1.1, world_w * cam.K[0, 0] / max(0.6, float(zb)))
        dx, dy = float(ub - ua), float(vb - va)
        dl = math.hypot(dx, dy) + 1e-6
        nx, ny = -dy / dl, dx / dl
        ox, oy = (dx / dl) * 0.65, (dy / dl) * 0.65

        q = [
            (float(ua) - nx * wa * 0.5 - ox, float(va) - ny * wa * 0.5 - oy),
            (float(ua) + nx * wa * 0.5 - ox, float(va) + ny * wa * 0.5 - oy),
            (float(ub) + nx * wb * 0.5 + ox, float(vb) + ny * wb * 0.5 + oy),
            (float(ub) - nx * wb * 0.5 + ox, float(vb) - ny * wb * 0.5 + oy),
        ]
        # Left/upper edge catches golden sunlight from upper-left SUN_DIR
        g = cairo.LinearGradient(q[0][0], q[0][1], q[1][0], q[1][1])
        g.add_color_stop_rgba(0.0, *lit_c, alpha)
        g.add_color_stop_rgba(0.35, *mid_c, alpha)
        g.add_color_stop_rgba(1.0, *dark_c, alpha)
        ctx.set_source(g)
        ctx.move_to(*q[0])
        for qp in q[1:]:
            ctx.line_to(*qp)
        ctx.close_path()
        ctx.fill()


# =============================================================================
# Stage 3 Band 0 — Luminous Pale Sage-Cream Misty Morning Garden Backdrop
# =============================================================================

def paint_misty_garden_backdrop(
    cam: Camera3D, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Band 0: Luminous parchment-cream misty morning skylight & soft rounded
    botanical garden canopies seen outside the conservatory glass & open arched
    doorway. ZERO geometric triangles — purely soft rounded hand-painted crowns!
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = xx / W
    v = yy / H

    mottle = fbm_gouache(H, W, rng)
    dabs = flat_brush_dabs(H, W, rng, count=94)
    tooth = paper_tooth(H, W, rng)

    mist_flow = gaussian_blur_2d(
        fbm_gouache(H // 2, W // 2, rng, scales=((88, 84), (38, 36), (18, 17), (8.5, 8.2))),
        5.5,
    )
    mist_flow = resize_bicubic(mist_flow, W, H)

    # Morning botanical conservatory sky: rich cool-sage zenith -> warm sunlit upper-left morning glow
    # (Deep enough misty garden tone so golden diagonal sunbeams burst dramatically out of the windows!)
    c_zenith = np.array([0.29, 0.41, 0.34], dtype=np.float32)
    c_mid = np.array([0.55, 0.66, 0.54], dtype=np.float32)
    c_low = np.array([0.72, 0.78, 0.67], dtype=np.float32)
    c_cream = np.array([0.85, 0.85, 0.73], dtype=np.float32)

    hv = float(cam.horizon_y / H)
    t_sky = np.clip(v / max(0.32, hv + 0.10), 0.0, 1.0)
    w0 = np.clip(1.0 - t_sky / 0.34, 0.0, 1.0)
    w1 = np.clip(1.0 - np.abs(t_sky - 0.34) / 0.34, 0.0, 1.0)
    w2 = np.clip(1.0 - np.abs(t_sky - 0.68) / 0.34, 0.0, 1.0)
    w3 = np.clip((t_sky - 0.66) / 0.34, 0.0, 1.0)
    w_sum = np.maximum(1e-5, w0 + w1 + w2 + w3)
    sky = (
        c_zenith[None, None, :] * w0[..., None]
        + c_mid[None, None, :] * w1[..., None]
        + c_low[None, None, :] * w2[..., None]
        + c_cream[None, None, :] * w3[..., None]
    ) / w_sum[..., None]

    # Concentrated golden morning sun halo strictly upper-left aligned with -SUN_DIR!
    du = u - 0.10
    dv = v - 0.12
    sun_halo = np.exp(-(du * du * 3.8 + dv * dv * 6.6)).astype(np.float32)
    sun_core = np.exp(-(du * du * 18.0 + dv * dv * 30.0)).astype(np.float32)
    sky += sun_halo[..., None] * np.array([0.30, 0.26, 0.11], dtype=np.float32)
    sky += sun_core[..., None] * np.array([0.34, 0.29, 0.13], dtype=np.float32)

    # Warm dreamy sunlit glow right behind the open arched rear conservatory doorway (~u=0.58, v=0.45)
    du_d = u - float(cam.vp_x / W)
    dv_d = v - (hv - 0.04)
    door_bloom = np.exp(-(du_d * du_d * 22.0 + dv_d * dv_d * 34.0)).astype(np.float32)
    sky += door_bloom[..., None] * np.array([0.16, 0.17, 0.10], dtype=np.float32)

    # Soft wispy morning clouds visible through upper arched glass vault
    cloud_m = np.clip((mist_flow + 0.32) * 0.46, 0.0, 1.0) * np.exp(-((v - 0.16) / 0.15) ** 2)
    sky = sky * (1.0 - 0.24 * cloud_m[..., None]) + np.array([0.97, 0.96, 0.88], dtype=np.float32) * (0.24 * cloud_m[..., None])
    sky += (mottle * 0.014 + dabs * 0.011 + tooth * 0.005)[..., None]

    canvas = np.clip(sky, 0.0, 1.32)

    # Soft painterly outdoor botanical garden canopies visible beyond the glass walls & open doorway!
    s_bg, c_bg = new_cairo_2x()
    # Tier A: Far pale misty botanical grove canopies across horizon (soft rounded Ghibli foliage puffs)
    for idx in range(72):
        cx = float(rng.uniform(-0.04, 1.04)) * W
        base_v = hv + 0.02 + 0.04 * math.sin(cx / W * 5.2 + 0.5)
        cy = float(rng.uniform(hv - 0.26, base_v)) * H
        rad = float(rng.uniform(44.0, 128.0))
        # Closer to sun (left) -> warmer pale olive-gold; right -> cooler misty sage
        warm = max(0.0, 1.0 - cx / (W * 0.85))
        c_core = (
            0.44 + 0.16 * warm + float(rng.uniform(-0.04, 0.04)),
            0.58 + 0.13 * warm + float(rng.uniform(-0.04, 0.04)),
            0.46 + 0.05 * warm + float(rng.uniform(-0.03, 0.03)),
        )
        g = cairo.RadialGradient(cx - rad * 0.22, cy - rad * 0.24, rad * 0.12, cx, cy, rad)
        g.add_color_stop_rgba(0.0, min(1.0, c_core[0] + 0.12), min(1.0, c_core[1] + 0.12), min(1.0, c_core[2] + 0.06), 0.34)
        g.add_color_stop_rgba(0.62, *c_core, 0.24)
        g.add_color_stop_rgba(1.0, *c_core, 0.0)
        c_bg.set_source(g)
        c_bg.arc(cx, cy, rad, 0.0, math.tau)
        c_bg.fill()

    # Tier B: Mid-distant lush garden palms, sunlit curving stone path & misty shrubs outside open doorway!
    vp_u = float(cam.vp_x / W)
    # Smooth mossy garden lawn fading organically upward into golden morning mist (zero hard horizon line!)
    g_lawn = cairo.LinearGradient(vp_u * W, (hv - 0.04) * H, vp_u * W, (hv + 0.22) * H)
    g_lawn.add_color_stop_rgba(0.0, 0.86, 0.90, 0.78, 0.0)
    g_lawn.add_color_stop_rgba(0.26, 0.72, 0.80, 0.66, 0.62)
    g_lawn.add_color_stop_rgba(0.55, 0.44, 0.58, 0.41, 0.92)
    g_lawn.add_color_stop_rgba(1.0, 0.24, 0.37, 0.25, 0.98)
    c_bg.set_source(g_lawn)
    c_bg.rectangle(0.0, (hv - 0.04) * H, W, H - (hv - 0.04) * H)
    c_bg.fill()

    # Curving sunlit wet garden stone walkway leading out from the open conservatory doorway!
    path_g = cairo.LinearGradient(vp_u * W, (hv + 0.01) * H, vp_u * W, (hv + 0.17) * H)
    path_g.add_color_stop_rgba(0.0, 0.88, 0.90, 0.79, 0.35)
    path_g.add_color_stop_rgba(0.45, 0.76, 0.80, 0.68, 0.78)
    path_g.add_color_stop_rgba(1.0, 0.54, 0.60, 0.50, 0.92)
    c_bg.set_source(path_g)
    c_bg.move_to((vp_u - 0.020) * W, (hv + 0.008) * H)
    c_bg.curve_to((vp_u - 0.035) * W, (hv + 0.065) * H, (vp_u - 0.065) * W, (hv + 0.12) * H, (vp_u - 0.095) * W, (hv + 0.18) * H)
    c_bg.line_to((vp_u + 0.075) * W, (hv + 0.18) * H)
    c_bg.curve_to((vp_u + 0.048) * W, (hv + 0.12) * H, (vp_u + 0.025) * W, (hv + 0.065) * H, (vp_u + 0.016) * W, (hv + 0.008) * H)
    c_bg.close_path()
    c_bg.fill()

    # Soft misty sunlit outdoor garden trees, palms & flowering hedgerows framing the open doorway!
    outdoor_palms = [
        (vp_u - 0.076, hv + 0.055, 195, -0.16, 0.62),
        (vp_u + 0.072, hv + 0.048, 205, 0.20, 0.58),
        (vp_u - 0.036, hv + 0.025, 148, -0.10, 0.46),
        (vp_u + 0.042, hv + 0.028, 155, 0.13, 0.48),
        (0.14, hv + 0.02, 235, 0.16, 0.44),
        (0.28, hv + 0.03, 215, -0.12, 0.40),
        (0.84, hv + 0.02, 230, -0.18, 0.46),
        (0.95, hv + 0.03, 245, -0.24, 0.50),
    ]
    for idx_p, (pu, pv, ph, lean, alpha_p) in enumerate(outdoor_palms):
        prng = np.random.default_rng(1800 + idx_p * 41)
        bx, by = pu * W, pv * H
        tx, ty = bx + lean * ph, by - ph
        c_bg.set_source_rgba(0.28, 0.39, 0.30, alpha_p * 0.78)
        c_bg.set_line_width(max(3.5, ph * 0.052))
        c_bg.move_to(bx, by + 40)
        c_bg.curve_to(bx + lean * ph * 0.25, by - ph * 0.35, bx + lean * ph * 0.72, by - ph * 0.72, tx, ty)
        c_bg.stroke()
        for f_k in range(9):
            ang = -math.pi * 0.92 + (f_k / 8.0) * math.pi * 1.12 + float(prng.uniform(-0.08, 0.08))
            flen = ph * float(prng.uniform(0.50, 0.86))
            ex = tx + math.cos(ang) * flen
            ey = ty + math.sin(ang) * flen + flen * 0.32
            mx = tx + math.cos(ang) * flen * 0.52
            my = ty + math.sin(ang) * flen * 0.42 - flen * 0.08
            lit = 0.5 + 0.5 * max(0.0, -math.cos(ang))
            c_fr = (0.34 + 0.28 * lit, 0.49 + 0.25 * lit, 0.33 + 0.14 * lit)
            c_bg.set_source_rgba(*c_fr, alpha_p * 0.86)
            c_bg.set_line_width(max(2.2, ph * 0.024))
            c_bg.move_to(tx, ty)
            c_bg.curve_to(mx, my, (mx + ex) * 0.5, my + flen * 0.08, ex, ey)
            c_bg.stroke()
            for s in np.linspace(0.22, 0.95, 7):
                fx = (1 - s) ** 2 * tx + 2 * (1 - s) * s * mx + s * s * ex
                fy = (1 - s) ** 2 * ty + 2 * (1 - s) * s * my + s * s * ey
                rx_f = ph * 0.065 * math.sin(s * math.pi) + 4.0
                c_bg.set_source_rgba(*c_fr, alpha_p * 0.48)
                c_bg.arc(fx, fy + rx_f * 0.35, rx_f, 0, math.tau)
                c_bg.fill()

    # Lush layered outdoor garden shrub cushions & pale cream-sage hydrangea blooms flanking the garden path!
    for idx_s in range(56):
        sx = float(rng.uniform(-0.02, 1.02)) * W
        sy = (hv + float(rng.uniform(-0.015, 0.155))) * H
        # Keep center of garden walkway gently open near doorway
        if abs(sx / W - vp_u) < 0.026 and sy > (hv + 0.03) * H:
            sx += (0.042 if sx > vp_u * W else -0.042) * W
        sr = float(rng.uniform(26.0, 82.0))
        lit = float(rng.uniform(0.15, 1.0))
        col_s = (0.20 + 0.24 * lit, 0.36 + 0.26 * lit, 0.22 + 0.14 * lit)
        g = cairo.RadialGradient(sx - sr * 0.25, sy - sr * 0.25, sr * 0.1, sx, sy, sr)
        g.add_color_stop_rgba(0.0, *col_s, 0.56)
        g.add_color_stop_rgba(1.0, *col_s, 0.0)
        c_bg.set_source(g)
        c_bg.arc(sx, sy, sr, 0, math.tau)
        c_bg.fill()
        # Occasional delicate pale parchment-cream garden blossoms glowing in the mist!
        if idx_s % 5 == 0 and abs(sx / W - vp_u) < 0.16:
            for _ in range(4):
                bx = sx + float(rng.uniform(-sr * 0.45, sr * 0.45))
                by = sy + float(rng.uniform(-sr * 0.38, sr * 0.22))
                c_bg.set_source_rgba(0.95, 0.94, 0.82, 0.46)
                c_bg.arc(bx, by, float(rng.uniform(3.2, 6.5)), 0, math.tau)
                c_bg.fill()

    bg_1x = finish_cairo_2x(s_bg)
    canvas = over_premul(canvas, bg_1x)

    # Silky pale sage-cream morning garden mist scarves drifting gently outside the conservatory
    for mv_c, mv_s, mv_a in ((hv - 0.015, 0.038, 0.34), (hv + 0.035, 0.048, 0.44)):
        wave_v = mv_c + 0.012 * np.sin(u * 4.8 + mv_c * 16.0) + 0.009 * mist_flow
        rib = np.exp(-((v - wave_v) / mv_s) ** 2) * mv_a * np.clip(0.78 + 0.28 * mist_flow, 0.45, 1.1)
        canvas = canvas * (1.0 - rib[..., None]) + np.array([0.88, 0.91, 0.81], dtype=np.float32) * rib[..., None]

    fog_rgb = np.array([0.74, 0.80, 0.69], dtype=np.float32)
    return np.clip(canvas, 0.0, 1.32), fog_rgb, mottle, dabs


# =============================================================================
# Stage 1 — Planar UV Surface Textures (Walls, Gable, Floor, Planter, Bench)
# =============================================================================

def paint_side_glass_wall_uv(
    rng: np.random.Generator, side: str = "left", tw: int = 1600, th: int = 720
) -> np.ndarray:
    """
    Tall Victorian/Meiji conservatory side wall (left or right):
      - Lower mossy weathered brick & limestone kneewall (z in [0, Z_KNEE])
      - Cut-stone coping sill catching morning light (z in [0.84, 0.94])
      - Tall arched glass window bays (z in [0.94, Z_SPRING]) with sage-patina
        wrought-iron mullions, semicircular fanlight arches, translucent misty
        glass condensation sheen, fine water droplets & outdoor silhouette haze!
    """
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wy = Y_FRONT + u * (Y_REAR - Y_FRONT)
    wz = Z_FLOOR + v * (Z_SPRING - Z_FLOOR)

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=88)
    tooth = paper_tooth(th, tw, rng)

    is_left = side == "left"

    # --- 1. Translucent condensation-misted glass panes ---
    # Subtle glass tint + condensation mist thicker near edges and bottom sill
    bay_phase = np.mod(wy - BAY_YS[0], 2.10)
    bay_u = (bay_phase - 1.05) / 0.96
    mull_phase = np.mod(wy - BAY_YS[0], 0.70)
    dist_mull = np.minimum(mull_phase, 0.70 - mull_phase)
    dist_rib = np.minimum(bay_phase, 2.10 - bay_phase)

    pane_edge = np.exp(-((dist_mull / 0.085) ** 2)) + 0.75 * np.exp(-((np.abs(wz - 1.68) / 0.075) ** 2))
    cond_sill = np.exp(-((np.maximum(0.0, wz - Z_KNEE) / 0.28) ** 2))
    # Diagonal specular glass sheen catching warm morning skylight
    diag_sheen = np.exp(-(((np.mod(wy * 0.48 + wz * 0.55, 1.8) - 0.85) / 0.28) ** 2))

    # Fine condensation mist droplets on cool glass
    drop_noise = resize_bicubic(rng.standard_normal((th // 3, tw // 3)).astype(np.float32), tw, th)
    droplets = np.clip((drop_noise - 1.55) * 2.8, 0.0, 1.0) * (0.35 + 0.65 * cond_sill)

    glass_col = np.zeros((th, tw, 3), dtype=np.float32)
    if is_left:
        # Sunlit left glass contrasts richer misty sage-shaded panes against blazing warm golden sun apertures!
        sun_win = outdoor_sun_dapple(
            np.full_like(wy, X_LEFT, dtype=np.float64),
            wy.astype(np.float64),
            wz.astype(np.float64),
        ).astype(np.float32)
        upper_flare = np.clip((wz - 1.10) / 1.60, 0.0, 1.0) * np.clip((12.0 - wy) / 8.0, 0.28, 1.0)
        win_glow = np.power(np.clip((sun_win - 0.34) / 0.80, 0.0, 1.30), 1.45) * (0.48 + 0.52 * upper_flare)
        glass_col[:, :, 0] = 0.50 + 0.54 * win_glow + 0.10 * diag_sheen + 0.04 * cond_sill + mottle * 0.022
        glass_col[:, :, 1] = 0.59 + 0.42 * win_glow + 0.08 * diag_sheen + 0.04 * cond_sill + mottle * 0.018
        glass_col[:, :, 2] = 0.49 + 0.19 * win_glow + 0.05 * diag_sheen + 0.03 * cond_sill + mottle * 0.015
        glass_alpha = np.clip(
            0.22 + 0.26 * win_glow + 0.14 * cond_sill + 0.10 * pane_edge + 0.10 * diag_sheen + 0.07 * droplets,
            0.18,
            0.74,
        )
    else:
        # Shaded right glass carries cool sage conservatory reflection + gentle mist
        glass_col[:, :, 0] = 0.34 + 0.13 * diag_sheen + 0.08 * cond_sill + mottle * 0.022
        glass_col[:, :, 1] = 0.44 + 0.12 * diag_sheen + 0.08 * cond_sill + mottle * 0.022
        glass_col[:, :, 2] = 0.39 + 0.09 * diag_sheen + 0.07 * cond_sill + mottle * 0.018
        glass_alpha = np.clip(0.32 + 0.18 * cond_sill + 0.12 * pane_edge + 0.11 * diag_sheen + 0.08 * droplets, 0.22, 0.68)

    glass_col += droplets[..., None] * np.array([0.24, 0.24, 0.16], dtype=np.float32)

    # --- 2. Sage-patina wrought-iron bay columns, mullions, transoms & semicircular arches ---
    arch_z_top = 2.34 + 0.55 * np.sqrt(np.maximum(0.0, 1.0 - np.clip(bay_u, -1.0, 1.0) ** 2))
    # Smoothly anti-aliased arch spandrel & circular filigree medallions (zero pixel stair-stepping at 1:1!)
    spandrel_cov = np.clip((wz - arch_z_top) / 0.012, 0.0, 1.0) * (wz <= Z_SPRING).astype(np.float32)
    is_spandrel = (wz > arch_z_top) & (wz <= Z_SPRING)
    is_arch_rim = np.abs(wz - arch_z_top) < 0.038
    is_rib = dist_rib < 0.068
    is_mull = (dist_mull < 0.022) & (wz < arch_z_top + 0.012)
    is_transom = (np.abs(wz - 1.68) < 0.024) | (np.abs(wz - 2.34) < 0.030) | (wz > Z_SPRING - 0.075)
    frame_cov = np.maximum.reduce([
        spandrel_cov,
        np.clip((0.038 - np.abs(wz - arch_z_top)) / 0.010, 0.0, 1.0),
        np.clip((0.068 - dist_rib) / 0.010, 0.0, 1.0),
        np.clip((0.022 - dist_mull) / 0.008, 0.0, 1.0) * (wz < arch_z_top + 0.012),
        is_transom.astype(np.float32),
    ])

    # Crisp backlit sage wrought-iron silhouette against golden morning windows (#17211c -> #35493d -> #748e76)
    iron_base = np.array([0.175, 0.252, 0.205] if is_left else [0.148, 0.218, 0.180], dtype=np.float32)
    iron_hi = np.array([0.540, 0.625, 0.460] if is_left else [0.310, 0.405, 0.335], dtype=np.float32)
    iron_dark = np.array([0.076, 0.118, 0.095], dtype=np.float32)

    iron_col = iron_base[None, None, :] + (mottle * 0.022 + dabs * 0.016)[..., None]
    # Warm backlit rim bevel kissing top edges of transoms & arches
    bevel_top = ((wz - 1.68) > 0.008) & ((wz - 1.68) < 0.024) | ((wz - 2.34) > 0.010) & ((wz - 2.34) < 0.030)
    iron_col = np.where(bevel_top[..., None], iron_hi[None, None, :], iron_col)
    iron_col = np.where((is_spandrel & (~is_arch_rim))[..., None], iron_dark[None, None, :] * 1.15 + mottle[..., None] * 0.014, iron_col)
    # Anti-aliased ornamental pierced circular medallions inside upper spandrel corners
    med_x = np.abs(bay_phase - 1.05) - 0.78
    med_z = wz - 2.74
    med_r = np.hypot(med_x, med_z)
    med_hole_cov = np.clip((0.075 - med_r) / 0.010, 0.0, 1.0) * spandrel_cov
    med_ring = is_spandrel & (med_r >= 0.072) & (med_r < 0.102)
    iron_col = np.where(med_ring[..., None], iron_hi[None, None, :] * 0.85, iron_col)
    frame_cov = np.clip(frame_cov * (1.0 - med_hole_cov), 0.0, 1.0)

    # --- 3. Lower mossy weathered brick & stone kneewall (z in [0.0, Z_KNEE]) ---
    is_knee = wz <= Z_KNEE
    is_coping = (wz > Z_KNEE - 0.085) & (wz <= Z_KNEE + 0.025)
    brick_row = np.floor(wz / 0.11).astype(np.int32)
    brick_off = (brick_row & 1) * 0.145
    by_frac = np.mod(wy + brick_off, 0.29) / 0.29
    bz_frac = np.mod(wz, 0.11) / 0.11
    mortar = (by_frac < 0.065) | (by_frac > 0.935) | (bz_frac < 0.09) | (bz_frac > 0.91)

    b_idx_y = np.floor((wy + brick_off) / 0.29).astype(np.int32)
    tone_tab = rng.uniform(-0.045, 0.045, size=(64, 16)).astype(np.float32)
    b_tone = tone_tab[b_idx_y & 63, brick_row & 15]

    # Deep rich shaded mossy brick kneewall anchoring the backlit golden windows
    brick_c = np.zeros((th, tw, 3), dtype=np.float32)
    if is_left:
        brick_c[:, :, 0] = 0.185 + b_tone + mottle * 0.024 + dabs * 0.018
        brick_c[:, :, 1] = 0.178 + b_tone * 0.85 + mottle * 0.022 + dabs * 0.016
        brick_c[:, :, 2] = 0.142 + b_tone * 0.72 + mottle * 0.018 + dabs * 0.012
    else:
        brick_c[:, :, 0] = 0.180 + b_tone + mottle * 0.024 + dabs * 0.016
        brick_c[:, :, 1] = 0.200 + b_tone * 0.85 + mottle * 0.024 + dabs * 0.016
        brick_c[:, :, 2] = 0.168 + b_tone * 0.72 + mottle * 0.018 + dabs * 0.012

    mortar_c = np.array([0.088, 0.116, 0.088], dtype=np.float32)
    brick_c = np.where(mortar[..., None], mortar_c[None, None, :], brick_c)

    # Damp velvety emerald moss creeping up from floor & tucked into mortar joints
    moss_floor = np.clip((0.36 - wz + 0.14 * mottle) / 0.34, 0.0, 1.0)
    moss_joint = np.where(mortar, np.clip(0.35 + 0.55 * mottle, 0.0, 1.0), 0.0) * 0.65
    moss_amt = np.clip(moss_floor * 0.78 + moss_joint, 0.0, 0.88)
    moss_col = np.array([0.132, 0.238, 0.125], dtype=np.float32)[None, None, :] + mottle[..., None] * 0.028
    brick_c = brick_c * (1.0 - moss_amt[..., None]) + moss_col * moss_amt[..., None]

    # Cut limestone coping sill at top of kneewall
    coping_c = np.array([0.320, 0.355, 0.290] if is_left else [0.245, 0.295, 0.245], dtype=np.float32)[None, None, :]
    coping_c = coping_c + (mottle * 0.024 + dabs * 0.016)[..., None]
    coping_hi = wz > (Z_KNEE - 0.018)
    coping_c = np.where(coping_hi[..., None], coping_c * 1.32, coping_c)

    # Compose kneewall, coping, smoothly anti-aliased wrought-iron frame, and translucent misted glass
    is_solid = (is_knee | is_coping | (frame_cov > 0.5))
    rgb = glass_col * (1.0 - frame_cov[..., None]) + iron_col * frame_cov[..., None]
    alpha = glass_alpha * (1.0 - frame_cov) + 0.995 * frame_cov
    rgb = np.where(is_knee[..., None], brick_c, rgb)
    alpha = np.where(is_knee, 1.0, alpha)
    rgb = np.where(is_coping[..., None], coping_c, rgb)
    alpha = np.where(is_coping, 1.0, alpha)

    # Stage 2: Exact unified 3D morning sunlight on the right wall & interior sill bevels!
    if not is_left:
        sun_r = trace_conservatory_sunlight(
            np.full_like(wy, X_RIGHT - 0.04, dtype=np.float64),
            wy.astype(np.float64),
            wz.astype(np.float64),
            include_bench=False,
        )
        rgb += np.where(is_solid, sun_r, sun_r * 0.32)[..., None] * np.array([0.38, 0.36, 0.19], dtype=np.float32)
    else:
        # Warm golden rim light kissing left sill & window frame edges from outdoor sun
        sill_rim = np.exp(-((wz - Z_KNEE) / 0.048) ** 2)
        rgb += sill_rim[..., None] * np.array([0.34, 0.33, 0.17], dtype=np.float32)

    # Floor contact AO at base of kneewall
    base_ao = np.exp(-((wz - Z_FLOOR) / 0.14) ** 2)
    rgb *= (1.0 - 0.36 * base_ao)[..., None]

    rgb = apply_shadow_bounce(rgb)
    rgb += tooth[..., None] * 0.006

    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.28) * alpha[..., None]
    rgba[:, :, 3] = alpha
    return rgba


def paint_rear_gable_uv(rng: np.random.Generator, tw: int = 860, th: int = 760) -> np.ndarray:
    """
    Soaring arched rear conservatory glass gable wall (y = Y_REAR):
      - Clipped cleanly to the elliptical vault silhouette z <= vault_z(x)
      - Wide open arched conservatory garden doorway (alpha = 0!) revealing the
        sunlit misty botanical garden outside
      - Ornate Victorian radial sunburst fanlight ironwork above the doorway
      - Translucent condensation-misted flanking glass panels & mossy stone kneewall.
    """
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wx = X_LEFT + u * (X_RIGHT - X_LEFT)
    wz = Z_FLOOR + v * (Z_PEAK + 0.06 - Z_FLOOR)

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=86)
    tooth = paper_tooth(th, tw, rng)

    vz = vault_z(wx).astype(np.float32)
    inside_vault = wz <= (vz + 0.03)

    # Open arched garden doorway (x in [-DOOR_HW, +DOOR_HW], arched top above DOOR_SPRING)
    d_arch_z = DOOR_SPRING + (DOOR_TOP - DOOR_SPRING) * np.sqrt(
        np.maximum(0.0, 1.0 - np.clip(wx / DOOR_HW, -1.0, 1.0) ** 2)
    )
    in_doorway = (np.abs(wx) < DOOR_HW) & (wz < d_arch_z)

    # Translucent glowing morning glass panes flanking & above doorway
    dist_center = np.hypot(wx / 2.2, (wz - 2.6) / 1.8)
    glass_col = np.zeros((th, tw, 3), dtype=np.float32)
    glass_col[:, :, 0] = 0.74 + 0.14 * np.exp(-dist_center * 1.4) + mottle * 0.024
    glass_col[:, :, 1] = 0.80 + 0.12 * np.exp(-dist_center * 1.4) + mottle * 0.022
    glass_col[:, :, 2] = 0.69 + 0.08 * np.exp(-dist_center * 1.4) + mottle * 0.018
    glass_alpha = np.clip(0.24 + 0.12 * np.exp(-((wz - Z_KNEE) / 0.35) ** 2) + 0.08 * np.abs(mottle), 0.16, 0.44)

    # Ornate Victorian sage wrought-iron gable framing & sunburst radial fanlight!
    iron_dark = np.array([0.135, 0.195, 0.158], dtype=np.float32)
    iron_mid = np.array([0.245, 0.335, 0.278], dtype=np.float32)
    iron_hi = np.array([0.455, 0.565, 0.465], dtype=np.float32)

    # Doorway arch casing & heavy jamb pilasters
    d_arch_outer = DOOR_SPRING + (DOOR_TOP - DOOR_SPRING + 0.095) * np.sqrt(
        np.maximum(0.0, 1.0 - np.clip(wx / (DOOR_HW + 0.095), -1.0, 1.0) ** 2)
    )
    door_jamb = (np.abs(wx) >= DOOR_HW) & (np.abs(wx) <= DOOR_HW + 0.095) & (wz <= DOOR_SPRING + 0.04)
    door_arch_frame = (~in_doorway) & (np.abs(wx) <= DOOR_HW + 0.095) & (wz >= DOOR_SPRING - 0.02) & (wz <= d_arch_outer)

    # Main horizontal transom beam at z = Z_SPRING and side window mullions
    h_beam = (np.abs(wz - Z_SPRING) < 0.055) | (np.abs(wz - 1.75) < 0.024) | (np.abs(wz - 2.38) < 0.022)
    v_mulls = (
        (np.abs(np.abs(wx) - 1.58) < 0.024)
        | (np.abs(np.abs(wx) - 2.12) < 0.024)
        | (np.abs(np.abs(wx) - 2.58) < 0.065)
    )
    # Vault outer rim rib
    vault_rim = np.abs(wz - vz) < 0.085

    # Upper gable radial sunburst spokes & concentric fanlight ring centered at (0, DOOR_SPRING)
    rx = wx
    rz = wz - DOOR_SPRING
    rr = np.hypot(rx, rz)
    ang = np.arctan2(rz, rx)
    above_door = (wz > d_arch_outer - 0.01) & (wz <= vz)
    # 7 elegant sunburst spokes radiating into the vaulted glass gable
    spoke_d = np.abs(np.sin((ang - math.pi * 0.5) * 6.0)) * rr
    is_spoke = above_door & (spoke_d < 0.026) & (rr > 0.85)
    is_ring = above_door & ((np.abs(rr - 1.55) < 0.028) | (np.abs(rr - 2.18) < 0.026))

    is_frame = (door_jamb | door_arch_frame | h_beam | v_mulls | vault_rim | is_spoke | is_ring) & (~in_doorway)
    is_knee = (wz <= Z_KNEE) & (~in_doorway) & (np.abs(wx) >= DOOR_HW)

    iron_c = iron_mid[None, None, :] + (mottle * 0.022 + dabs * 0.016)[..., None]
    iron_c = np.where((door_arch_frame | vault_rim)[..., None], iron_hi[None, None, :] * 0.72 + iron_dark * 0.28, iron_c)

    knee_c = np.array([0.215, 0.245, 0.195], dtype=np.float32)[None, None, :] + (mottle * 0.026 + dabs * 0.018)[..., None]
    coping = (wz > Z_KNEE - 0.075) & is_knee
    knee_c = np.where(coping[..., None], np.array([0.34, 0.39, 0.32], dtype=np.float32), knee_c)

    rgb = glass_col.copy()
    alpha = glass_alpha.copy()
    rgb = np.where(is_frame[..., None], iron_c, rgb)
    alpha = np.where(is_frame, 0.995, alpha)
    rgb = np.where(is_knee[..., None], knee_c, rgb)
    alpha = np.where(is_knee, 1.0, alpha)

    # Zero alpha cleanly inside open arched doorway & outside curved vault arch!
    alpha = np.where(in_doorway | (~inside_vault), 0.0, alpha)

    rgb = apply_shadow_bounce(rgb)
    rgb += tooth[..., None] * 0.005

    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.25) * alpha[..., None]
    rgba[:, :, 3] = alpha
    return rgba


def paint_slate_floor_uv(rng: np.random.Generator, tw: int = 1024, th: int = 1360) -> tuple[np.ndarray, np.ndarray]:
    """
    Wet conservatory slate flagstone floor (z = 0) with:
      - Metric jittered 3x3 Voronoi (d2 - d1) irregular slate flagstones & lush mossy crevices
      - Wet-stone rim darkening around broad shallow watering puddles
      - Exact Stage 2 6-step -SUN_DIR arched window sunlight patches, wrought-iron rib
        shadows, and grounded cedar potting workbench cast shadow!
    Returns (floor_premul_rgba, puddle_uv_mask).
    """
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wx = X_LEFT + u * (X_RIGHT - X_LEFT)
    wy = Y_FRONT + v * (Y_REAR + 0.90 - Y_FRONT)
    wz = np.full_like(wx, Z_FLOOR)

    mottle = fbm_gouache(th, tw, rng)
    mottle_fine = fbm_gouache(th, tw, rng, scales=((28, 28), (11, 11), (5.0, 5.0), (2.5, 2.5)))
    dabs = flat_brush_dabs(th, tw, rng, count=92)
    tooth = paper_tooth(th, tw, rng)

    # Metric 3x3 Voronoi slate flagstones (0.62 m cell size, organic irregular natural flags)
    _, edge_d, cell_id = voronoi_metric_grid(wx, wy, cell_size=0.62, jitter=0.34, rng=rng)
    # Slightly wobble joint line with low-freq noise for organic hand-laid weathered stone
    edge_wob = edge_d + mottle_fine * 0.007
    crevice = np.clip((0.036 - edge_wob) / 0.026, 0.0, 1.0)
    bevel_edge = np.clip((0.068 - edge_wob) / 0.034, 0.0, 1.0) * (1.0 - crevice)

    # Per-flagstone rich cool slate-green shadow base (#1d2621 range so golden sun pools POP!)
    st_var = (cell_id - 0.5) * 0.088
    slate = np.zeros((th, tw, 3), dtype=np.float32)
    slate[:, :, 0] = 0.118 + st_var * 0.88 + mottle * 0.024 + dabs * 0.018 + mottle_fine * 0.014
    slate[:, :, 1] = 0.156 + st_var * 0.96 + mottle * 0.026 + dabs * 0.020 + mottle_fine * 0.015
    slate[:, :, 2] = 0.148 + st_var * 0.90 + mottle * 0.022 + dabs * 0.016 + mottle_fine * 0.013

    # Subtle sunlit stone edge bevel glint facing upper-left (-X)
    ed_dx = np.gradient(edge_wob, axis=1)
    bevel_lit = np.clip(ed_dx * 24.0, 0.0, 1.0) * bevel_edge
    slate += bevel_lit[..., None] * np.array([0.075, 0.082, 0.062], dtype=np.float32)

    # Velvety emerald-olive conservatory moss growing richly in flagstone crevices!
    # Moss is lusher near the kneewalls, under the potting bench & near the right planter bed
    wall_damp = np.clip(1.0 - np.minimum(wx - X_LEFT, X_RIGHT - wx) / 1.45, 0.0, 1.0).astype(np.float32)
    moss_spread = np.clip(crevice + np.clip((0.085 - edge_wob) / 0.065, 0.0, 1.0) * (0.25 + 0.55 * wall_damp) * np.clip(0.45 + 0.55 * mottle, 0.0, 1.0), 0.0, 1.0)
    moss_col = np.zeros((th, tw, 3), dtype=np.float32)
    moss_col[:, :, 0] = 0.112 + mottle_fine * 0.028 + dabs * 0.018
    moss_col[:, :, 1] = 0.228 + mottle_fine * 0.038 + dabs * 0.024
    moss_col[:, :, 2] = 0.112 + mottle_fine * 0.020 + dabs * 0.012

    dark_grout = np.array([0.038, 0.056, 0.044], dtype=np.float32)
    deep_seam = np.clip((0.015 - edge_wob) / 0.015, 0.0, 1.0)
    rgb = slate * (1.0 - moss_spread[..., None]) + moss_col * moss_spread[..., None]
    rgb = rgb * (1.0 - 0.58 * deep_seam[..., None]) + dark_grout[None, None, :] * (0.58 * deep_seam[..., None])

    # Damp spilled potting soil near the left workbench base
    soil_spill = (
        np.exp(-(((wx - (BENCH_X1 - 0.15)) / 0.42) ** 2))
        * np.exp(-(((wy - 4.4) / 2.6) ** 2))
        * np.clip(0.35 + 0.65 * mottle_fine, 0.0, 1.0)
    ).astype(np.float32)
    rgb = rgb * (1.0 - 0.44 * soil_spill[..., None]) + np.array([0.100, 0.076, 0.054], dtype=np.float32) * (0.44 * soil_spill[..., None])

    # Broad shallow watering puddles & realistic wet darkened damp-stone rim!
    p_depth = puddle_depth_field(wx, wy)
    # Slightly follow flagstone depressions so puddle edge looks natural & organic
    p_organic = np.clip(p_depth + (0.045 * mottle + 0.035 * crevice) * (p_depth > 0.02), 0.0, 1.0)
    puddle_mask = np.clip((p_organic - 0.14) / 0.12, 0.0, 1.0)
    puddle_mask = puddle_mask * puddle_mask * (3.0 - 2.0 * puddle_mask)
    wet_rim = np.clip((p_organic - 0.02) / 0.16, 0.0, 1.0)
    # Wet slate darkens & saturates around puddles (physically accurate!)
    rgb *= (1.0 - 0.38 * wet_rim)[..., None]

    # Stage 2: Exact unified 3D morning sunlight & arched window patches + bench cast shadow!
    sun = trace_conservatory_sunlight(wx, wy, wz, include_bench=True)
    sun_crisp = np.power(np.clip(sun, 0.0, 1.2), 1.12).astype(np.float32)

    # Rich luminous golden-cream morning sun pools where exact window shafts land on damp flagstones & moss!
    sun_slate = np.array([0.56, 0.54, 0.29], dtype=np.float32)
    sun_moss = np.array([0.34, 0.43, 0.14], dtype=np.float32)
    rgb += sun_crisp[..., None] * (moss_spread[..., None] * sun_moss + (1.0 - moss_spread[..., None]) * sun_slate)

    # Subtle glossy wet-flagstone morning skylight sheen even outside main puddles (stronger in sun pools!)
    damp_sheen = (0.08 + 0.14 * wet_rim + 0.10 * sun_crisp * wet_rim) * np.clip(0.55 + 0.45 * mottle, 0.0, 1.0) * (1.0 - moss_spread)
    sheen_col = np.array([0.44, 0.52, 0.44], dtype=np.float32) + sun_crisp[..., None] * np.array([0.36, 0.32, 0.16], dtype=np.float32)
    rgb = rgb * (1.0 - damp_sheen[..., None]) + sheen_col * damp_sheen[..., None]

    # Contact AO tucked cleanly along left/right walls, right planter curb & bench footprint
    ao_left = np.exp(-((wx - X_LEFT) / 0.24) ** 2)
    ao_planter = np.where(
        (wy >= PLANTER_Y0 - 0.1) & (wy <= PLANTER_Y1 + 0.1),
        np.exp(-((np.maximum(0.0, PLANTER_X0 - wx)) / 0.22) ** 2),
        0.0,
    ).astype(np.float32)
    ao_bench = (
        np.clip((BENCH_X1 + 0.18 - wx) / 0.38, 0.0, 1.0)
        * np.clip((wy - (BENCH_Y0 - 0.15)) / 0.25, 0.0, 1.0)
        * np.clip((BENCH_Y1 + 0.15 - wy) / 0.25, 0.0, 1.0)
    ).astype(np.float32)
    rgb *= (1.0 - 0.44 * ao_left - 0.44 * ao_planter - 0.38 * ao_bench)[..., None]

    rgb = apply_shadow_bounce(rgb)
    rgb += tooth[..., None] * 0.007

    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.28)
    return rgba, puddle_mask.astype(np.float32)


def paint_planter_curb_uv(rng: np.random.Generator, tw: int = 960, th: int = 240) -> np.ndarray:
    """Right raised mossy limestone planter bed front wall (x = PLANTER_X0, z in [0, PLANTER_Z])."""
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wy = PLANTER_Y0 + u * (PLANTER_Y1 - PLANTER_Y0)
    wz = Z_FLOOR + v * PLANTER_Z

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=86)
    tooth = paper_tooth(th, tw, rng)

    row = np.floor(wz / 0.16).astype(np.int32)
    off = (row & 1) * 0.24
    by_f = np.mod(wy + off, 0.48) / 0.48
    bz_f = np.mod(wz, 0.16) / 0.16
    seam = (by_f < 0.045) | (by_f > 0.955) | (bz_f < 0.075) | (bz_f > 0.925)
    b_idx = np.floor((wy + off) / 0.48).astype(np.int32)
    tab = rng.uniform(-0.042, 0.042, size=(64, 8)).astype(np.float32)
    bt = tab[b_idx & 63, row & 7]

    stone = np.zeros((th, tw, 3), dtype=np.float32)
    stone[:, :, 0] = 0.200 + bt + mottle * 0.025 + dabs * 0.018
    stone[:, :, 1] = 0.225 + bt * 0.92 + mottle * 0.025 + dabs * 0.018
    stone[:, :, 2] = 0.190 + bt * 0.85 + mottle * 0.020 + dabs * 0.014
    stone = np.where(seam[..., None], np.array([0.072, 0.102, 0.074], dtype=np.float32), stone)

    # Top cut-stone coping slab & creeping velvety emerald moss overhanging the edge!
    is_cap = wz > (PLANTER_Z - 0.09)
    cap_c = np.array([0.295, 0.328, 0.265], dtype=np.float32)[None, None, :] + (mottle * 0.022 + dabs * 0.015)[..., None]
    stone = np.where(is_cap[..., None], cap_c, stone)

    moss_top = np.clip((wz - (PLANTER_Z - 0.22) + 0.08 * mottle) / 0.20, 0.0, 1.0) * np.clip(0.32 + 0.68 * mottle, 0.0, 1.0)
    moss_bot = np.clip((0.18 - wz + 0.06 * mottle) / 0.18, 0.0, 1.0)
    moss_m = np.clip(moss_top * 0.72 + moss_bot * 0.68, 0.0, 0.86)
    moss_c = np.array([0.135, 0.248, 0.120], dtype=np.float32)[None, None, :] + mottle[..., None] * 0.028
    rgb = stone * (1.0 - moss_m[..., None]) + moss_c * moss_m[..., None]

    # Exact Stage 2 unified morning sunlight hitting the left-facing stone planter face!
    sun = trace_conservatory_sunlight(
        np.full_like(wy, PLANTER_X0 - 0.02, dtype=np.float64),
        wy,
        wz,
        include_bench=True,
    )
    rgb += sun[..., None] * np.array([0.44, 0.42, 0.22], dtype=np.float32)

    # Contact shadow under coping lip and at floor joint
    lip_sh = np.exp(-(((wz - (PLANTER_Z - 0.10)) / 0.035) ** 2))
    floor_ao = np.exp(-((wz / 0.09) ** 2))
    rgb *= (1.0 - 0.32 * lip_sh - 0.34 * floor_ao)[..., None]

    rgb = apply_shadow_bounce(rgb)
    rgb += tooth[..., None] * 0.006
    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.28)
    return rgba


def paint_planter_soil_uv(rng: np.random.Generator, tw: int = 420, th: int = 960) -> np.ndarray:
    """Rich dark damp greenhouse potting soil bed & moss cushions on top of right raised planter."""
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wx = PLANTER_X0 + u * (PLANTER_X1 - PLANTER_X0)
    wy = PLANTER_Y0 + v * (PLANTER_Y1 - PLANTER_Y0)
    wz = np.full_like(wx, PLANTER_Z)

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=86)
    tooth = paper_tooth(th, tw, rng)

    # Stone coping rim along inner aisle edge (x in [PLANTER_X0, PLANTER_X0 + 0.14])
    is_rim = wx < (PLANTER_X0 + 0.14)
    soil = np.zeros((th, tw, 3), dtype=np.float32)
    soil[:, :, 0] = 0.135 + mottle * 0.026 + dabs * 0.018
    soil[:, :, 1] = 0.118 + mottle * 0.024 + dabs * 0.016
    soil[:, :, 2] = 0.088 + mottle * 0.018 + dabs * 0.012

    moss_cush = np.clip(mottle * 0.70 + 0.32, 0.0, 1.0)
    soil = soil * (1.0 - 0.65 * moss_cush[..., None]) + np.array([0.155, 0.275, 0.135], dtype=np.float32) * (0.65 * moss_cush[..., None])

    rim_c = np.array([0.375, 0.415, 0.340], dtype=np.float32)[None, None, :] + (mottle * 0.024 + dabs * 0.018)[..., None]
    rgb = np.where(is_rim[..., None], rim_c, soil)

    sun = trace_conservatory_sunlight(wx, wy, wz, include_bench=True)
    rgb += sun[..., None] * np.array([0.28, 0.30, 0.16], dtype=np.float32)
    rgb = apply_shadow_bounce(rgb)
    rgb += tooth[..., None] * 0.006

    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.28)
    return rgba


def paint_cedar_bench_top_uv(
    rng: np.random.Generator,
    pot_footprints: list[tuple[float, float, float]],
    tw: int = 640,
    th: int = 1200,
) -> np.ndarray:
    """
    Weathered warm cedar potting workbench tabletop (z = BENCH_Z):
      - Longitudinal weathered cedar planks (#563d2b -> #8f6b4e) with fine woodgrain
      - Spilled dark potting compost patches & faint damp terracotta water rings
      - Exact Stage 2 6-step -SUN_DIR arched morning window light patches
      - Grounded feathered contact AO + directional SUN_DIR cast shadows under EVERY pot/prop!
    """
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wx = BENCH_X0 + u * (BENCH_X1 - BENCH_X0)
    wy = BENCH_Y0 + v * (BENCH_Y1 - BENCH_Y0)
    wz = np.full_like(wx, BENCH_Z)

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=90)
    tooth = paper_tooth(th, tw, rng)

    plank_w = (BENCH_X1 - BENCH_X0) / 6.0
    p_idx = np.floor((wx - BENCH_X0) / plank_w).astype(np.int32)
    p_frac = np.mod((wx - BENCH_X0) / plank_w, 1.0)
    tone_tab = rng.uniform(-0.048, 0.048, size=16).astype(np.float32)
    pt = tone_tab[p_idx & 15]

    seam = ((p_frac < 0.032) | (p_frac > 0.968)).astype(np.float32)
    grain = (0.5 + 0.5 * np.sin(wx * 56.0 + wy * 1.4 + mottle * 4.2)).astype(np.float32)

    # Warm weathered honey-cedar shadow base so golden window sun pools POP across the planks!
    wood = np.zeros((th, tw, 3), dtype=np.float32)
    wood[:, :, 0] = 0.295 + pt + grain * 0.038 + mottle * 0.028 + dabs * 0.020
    wood[:, :, 1] = 0.215 + pt * 0.85 + grain * 0.026 + mottle * 0.022 + dabs * 0.016
    wood[:, :, 2] = 0.148 + pt * 0.70 + grain * 0.018 + mottle * 0.016 + dabs * 0.012

    wood *= (1.0 - 0.56 * seam)[..., None]

    # Subtle weathered patina & scattered dark damp potting soil crumbs on the potting bench!
    soil_patch = (
        np.exp(-(((wx + 1.62) / 0.24) ** 2 + ((wy - 3.05) / 0.34) ** 2))
        + 0.75 * np.exp(-(((wx + 1.92) / 0.22) ** 2 + ((wy - 4.85) / 0.32) ** 2))
    ).astype(np.float32)
    soil_m = np.clip(soil_patch * np.clip(0.45 + 0.75 * mottle, 0.0, 1.0), 0.0, 0.78)
    wood = wood * (1.0 - soil_m[..., None]) + np.array([0.110, 0.082, 0.058], dtype=np.float32) * soil_m[..., None]

    # Stage 2: Exact 6-step -SUN_DIR arched morning sunlight across the cedar bench!
    sun = trace_conservatory_sunlight(wx, wy, wz, include_bench=False)

    # Exact grounded feathered contact AO + directional SUN_DIR cast shadow for every pot & prop on bench!
    prop_ao = np.zeros((th, tw), dtype=np.float32)
    prop_cast = np.zeros((th, tw), dtype=np.float32)
    s_dx = float(SUN_DIR[0] / (-SUN_DIR[2]))
    s_dy = float(SUN_DIR[1] / (-SUN_DIR[2]))
    for px, py, prad in pot_footprints:
        d0 = np.hypot(wx - px, wy - py)
        ao = np.exp(-((np.maximum(0.0, d0 - prad * 0.82) / (prad * 0.42)) ** 2)).astype(np.float32)
        prop_ao = np.maximum(prop_ao, ao * 0.76)
        # Cast shadow stretched diagonally down-right along SUN_DIR!
        sh_cx = px + s_dx * prad * 0.95
        sh_cy = py + s_dy * prad * 0.95
        d_sh = np.hypot((wx - sh_cx) / (prad * 1.48), (wy - sh_cy) / (prad * 1.08))
        csh = np.clip((1.15 - d_sh) / 0.45, 0.0, 1.0).astype(np.float32)
        prop_cast = np.maximum(prop_cast, csh * 0.82)

    sun_eff = np.power(np.clip(sun * (1.0 - prop_cast * 0.88), 0.0, 1.2), 1.06).astype(np.float32)
    rgb = wood + sun_eff[..., None] * np.array([0.56, 0.48, 0.24], dtype=np.float32)
    # Crisp warm golden sunlight wood-bevel glint along sunlit aisle edge
    edge_hi = np.clip((wx - (BENCH_X1 - 0.05)) / 0.05, 0.0, 1.0).astype(np.float32) * (0.45 + 0.55 * sun_eff)
    rgb += edge_hi[..., None] * np.array([0.11, 0.095, 0.05], dtype=np.float32)
    rgb *= (1.0 - prop_ao * 0.70)[..., None]

    rgb = apply_shadow_bounce(rgb)
    rgb += tooth[..., None] * 0.006

    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.32)
    return rgba


def paint_cedar_bench_shelf_uv(rng: np.random.Generator, tw: int = 440, th: int = 960) -> np.ndarray:
    """Lower slatted cedar storage shelf under the potting bench (z = BENCH_SHELF_Z)."""
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wx = BENCH_X0 + 0.06 + u * (BENCH_X1 - BENCH_X0 - 0.12)
    wy = BENCH_Y0 + 0.06 + v * (BENCH_Y1 - BENCH_Y0 - 0.12)

    mottle = fbm_gouache(th, tw, rng)
    tooth = paper_tooth(th, tw, rng)
    slat_f = np.mod(wx - BENCH_X0, 0.145) / 0.145
    is_slat = (slat_f < 0.78).astype(np.float32)

    rgb = np.zeros((th, tw, 3), dtype=np.float32)
    rgb[:, :, 0] = 0.162 + mottle * 0.018
    rgb[:, :, 1] = 0.122 + mottle * 0.015
    rgb[:, :, 2] = 0.088 + mottle * 0.012
    # Warm bounce along outer edge
    outer_lit = np.clip((wx - (BENCH_X1 - 0.34)) / 0.30, 0.0, 1.0).astype(np.float32)
    rgb += outer_lit[..., None] * np.array([0.095, 0.082, 0.045], dtype=np.float32)
    rgb = apply_shadow_bounce(rgb) + tooth[..., None] * 0.005

    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.2) * is_slat[..., None]
    rgba[:, :, 3] = is_slat
    return rgba


# =============================================================================
# Stage 2 — Broad Floor Watering Puddles (Fresnel + cv2.remap + 3D Reflections)
# =============================================================================

def paint_floor_puddle_reflections(
    canvas: np.ndarray,
    ref_backdrop: np.ndarray,
    mirrored_3d: np.ndarray,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    m_global: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stage 2: Broad shallow watering puddles on the wet conservatory slate floor:
      - Raycasts each floor pixel to exact world (wx, wy, 0) to evaluate puddle_depth_field
      - Combines cv2.remap subtle micro-rippled reflections (y_src = 2 * horizon_y - y)
        with 3D plane-mirrored quads (vault ribs, open doorway, hanging ferns, brass
        lamp, bench legs) weighted by exact Fresnel:
        fresnel = fresnel_min + (fresnel_max - fresnel_min) * (1 - dist_t) ** 1.4
      - Adds crisp sunlit morning skylight glints along sunlit puddle zones!
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    # Raycast screen pixel onto horizontal floor plane z = Z_FLOOR = 0
    fx = cam.K[0, 0]
    cx, cy = cam.K[0, 2], cam.K[1, 2]
    dx_c = (xx - cx) / fx
    dy_c = (yy - cy) / fx
    # World ray direction = R^T @ [dx_c, dy_c, 1]
    R = cam.R
    rx = R[0, 0] * dx_c + R[1, 0] * dy_c + R[2, 0]
    ry = R[0, 1] * dx_c + R[1, 1] * dy_c + R[2, 1]
    rz = R[0, 2] * dx_c + R[1, 2] * dy_c + R[2, 2]

    valid_floor = rz < -1e-4
    t_f = np.where(valid_floor, (Z_FLOOR - cam.eye[2]) / np.where(valid_floor, rz, -1.0), 1e6)
    valid_floor &= (t_f > 0.2) & (t_f < 22.0)

    wx = cam.eye[0] + rx * t_f
    wy = cam.eye[1] + ry * t_f

    p_raw = np.where(valid_floor, puddle_depth_field(wx, wy), 0.0).astype(np.float32)
    p_org = np.clip(p_raw + 0.048 * m_global * (p_raw > 0.015), 0.0, 1.0)
    puddle_mask = np.clip((p_org - 0.12) / 0.16, 0.0, 1.0)
    puddle_mask = puddle_mask * puddle_mask * (3.0 - 2.0 * puddle_mask)
    puddle_mask = np.where(valid_floor & (wy >= 2.2) & (wy <= Y_REAR - 0.15) & (wx >= BENCH_X1 + 0.03) & (wx <= PLANTER_X0 - 0.04), puddle_mask, 0.0).astype(np.float32)

    # Distance factor dist_t: 0 at far horizon grazing angle, 1 near camera steep angle
    dist_t = np.clip((14.0 - t_f.astype(np.float32)) / 12.2, 0.0, 1.0)

    # Ultra-serene indoor conservatory puddle glassy micro-ripple (subtle calm liquid mirror!)
    rip_field = (
        0.52 * np.sin(wy.astype(np.float32) * 22.0 + wx.astype(np.float32) * 4.5 + m_global * 1.4)
        + 0.30 * np.cos(wx.astype(np.float32) * 14.0 - wy.astype(np.float32) * 11.0)
        + 0.18 * np.sin(wy.astype(np.float32) * 42.0 + 0.6)
    ).astype(np.float32)

    # Gentle glassy micro-rippled horizon reflection of vaulted skylight & misty garden doorway
    map_x = (xx + rip_field * (0.72 + 0.45 * dist_t)).astype(np.float32)
    map_y = (2.0 * cam.horizon_y - yy + rip_field * (0.95 + 0.65 * dist_t)).astype(np.float32)
    map_y = np.clip(map_y, 0.0, H - 1.0)
    refl_bg = remap_reflect(ref_backdrop, map_x, map_y)

    # Gentle glassy micro-rippled 3D plane-mirrored quads (vault ribs, hanging ferns, brass lamp, bench legs)
    mirr_rip = remap_reflect(mirrored_3d, map_x, (yy + rip_field * (0.90 + 0.55 * dist_t)).astype(np.float32))
    refl_combined = over_premul(refl_bg * 0.94, mirr_rip)
    # Soft painterly puddle gloss blur
    refl_combined = 0.68 * refl_combined + 0.32 * gaussian_blur_2d(refl_combined, 1.5)
    # Cool serene sage-emerald glassy tint
    refl_combined *= np.array([0.93, 0.99, 0.94], dtype=np.float32)[None, None, :]

    # Exact Stage 2 Fresnel blend: grazing far puddles reflect brightly, near puddles reveal wet slate below!
    fresnel_min, fresnel_max = 0.28, 0.85
    fresnel = fresnel_min + (fresnel_max - fresnel_min) * np.power(1.0 - dist_t, 1.4)

    # Deep rich wet slate flagstones visible beneath shallow foreground water
    bed_col = canvas * np.array([0.46, 0.52, 0.50], dtype=np.float32)[None, None, :]
    water_rgb = bed_col * (1.0 - fresnel[..., None]) + refl_combined * fresnel[..., None]

    # Luminous golden-cream morning specular glints & warm sun-pool bounce strictly on sunlit puddle zones!
    sun_p = np.where(valid_floor, trace_conservatory_sunlight(wx, wy, np.zeros_like(wx), include_bench=True), 0.0)
    water_rgb += (sun_p * (0.20 + 0.18 * (1.0 - fresnel)))[..., None] * np.array([0.52, 0.48, 0.24], dtype=np.float32)
    spec_glint = np.clip((rip_field - 0.60) * 2.6, 0.0, 1.0) * sun_p
    water_rgb += spec_glint[..., None] * np.array([0.58, 0.55, 0.30], dtype=np.float32)

    # Crisp glossy sunlit meniscus rim highlight catching morning sunbeams along puddle boundary
    rim_hl = np.exp(-(((puddle_mask - 0.50) / 0.20) ** 2)) * (0.16 + 0.84 * sun_p)
    water_rgb += rim_hl[..., None] * np.array([0.22, 0.24, 0.13], dtype=np.float32)

    fog_t = 1.0 - np.exp(-np.maximum(0.0, t_f.astype(np.float32) - 5.5) * 0.038)
    water_rgb = water_rgb * (1.0 - fog_t[..., None]) + fog_rgb[None, None, :] * fog_t[..., None]

    out = canvas * (1.0 - puddle_mask[..., None]) + water_rgb * puddle_mask[..., None]
    return out, puddle_mask


# =============================================================================
# Stage 3 Band 1 — Vaulted Wrought-Iron Ribs, Filigree, Hanging Ferns & Brass Lamp
# =============================================================================

def draw_glass_vault_sheen(ctx: cairo.Context, cam: Camera3D, fog_rgb: np.ndarray) -> None:
    """Translucent curved glass conservatory roof panels with warm golden-morning skylight sheen."""
    xs_grid = np.linspace(X_LEFT, X_RIGHT, 14)
    for bi in range(len(BAY_YS) - 1):
        y0, y1 = BAY_YS[bi], BAY_YS[bi + 1]
        ym = 0.5 * (y0 + y1)
        dist = float(np.linalg.norm(np.array([0.0, ym, Z_PEAK]) - cam.eye))
        for xi in range(len(xs_grid) - 1):
            xa, xb = float(xs_grid[xi]), float(xs_grid[xi + 1])
            za, zb = float(vault_z(xa)), float(vault_z(xb))
            # Sunlit left curved roof panels glow warmer golden parchment-cream; right panels cooler sage glass
            left_f = np.clip((0.55 - 0.5 * (xa + xb)) / 3.2, 0.0, 1.0)
            c_pane = fog_mix(
                [0.68 + 0.32 * left_f, 0.76 + 0.22 * left_f, 0.66 + 0.10 * left_f],
                dist,
                fog_rgb,
                fog_start=5.2,
                fog_density=0.034,
            )
            a_pane = 0.14 + 0.14 * left_f if (xi + bi) % 2 == 0 else 0.09 + 0.10 * left_f
            pts = [
                np.array([xa, y0, za]),
                np.array([xb, y0, zb]),
                np.array([xb, y1, zb]),
                np.array([xa, y1, za]),
            ]
            fill_3d_quad(ctx, cam, pts, (*c_pane, a_pane))


def draw_vaulted_wrought_iron(
    ctx: cairo.Context,
    cam: Camera3D,
    fog_rgb: np.ndarray,
    mirrored: bool = False,
) -> None:
    """
    Soaring Meiji/Victorian arched sage-painted wrought-iron conservatory architecture:
      - Longitudinal glass purlin ribs & raised ridge clerestory spine beam
      - Double-chord arched barrel-vault ribs at BAY_YS with pierced filigree spandrel
        gusset brackets at every column spring joint & slender horizontal tie-rods!
      - Half-open sage-painted iron-and-glass French conservatory doors at Y_REAR!
    """
    alpha_m = 0.54 if mirrored else 0.995
    xs_arch = np.linspace(X_LEFT, X_RIGHT, 34)

    # 1. Longitudinal wrought-iron purlins running the full length of the curved glass vault
    purlin_xs = (-2.65, -2.22, -1.58, -0.82, -0.34, 0.0, 0.34, 0.82, 1.58, 2.22, 2.65)
    for px_v in purlin_xs:
        pz = float(vault_z(px_v))
        is_ridge = abs(px_v) < 0.05
        is_eave = abs(abs(px_v) - HALF_SPAN) < 0.05
        w_m = 0.078 if (is_ridge or is_eave) else 0.034
        ys = np.linspace(Y_FRONT, Y_REAR, 18)
        pts = [np.array([px_v, float(yv), pz]) for yv in ys]
        if mirrored:
            pts = mirror_pts_across_plane(pts, Z_FLOOR)
        c_d = (0.12, 0.18, 0.145)
        c_m = (0.24, 0.34, 0.275)
        c_l = (0.48, 0.59, 0.485)
        stroke_3d_ribbon(ctx, cam, pts, w_m, c_d, c_m, c_l, alpha=alpha_m)

    # 2. Half-open arched French conservatory doors swung gracefully inward at Y_REAR!
    for side_s, ang_deg in ((-1.0, 38.0), (1.0, -44.0)):
        hx = side_s * DOOR_HW
        ang = math.radians(ang_deg)
        dw = 0.94
        # Swing inward (-Y) toward the aisle!
        ex = hx - side_s * math.cos(ang) * dw
        ey = Y_REAR - abs(math.sin(ang)) * dw
        dist_d = float(np.linalg.norm(np.array([0.5 * (hx + ex), 0.5 * (Y_REAR + ey), 1.2]) - cam.eye))
        c_frame = fog_mix([0.23, 0.33, 0.27], dist_d, fog_rgb)
        c_hi = fog_mix([0.46, 0.57, 0.47], dist_d, fog_rgb)
        c_glass = fog_mix([0.78, 0.85, 0.74], dist_d, fog_rgb)

        p_bl = np.array([hx, Y_REAR, Z_FLOOR + 0.03])
        p_br = np.array([ex, ey, Z_FLOOR + 0.03])
        p_tr = np.array([ex, ey, DOOR_SPRING + 0.16])
        p_tl = np.array([hx, Y_REAR, DOOR_SPRING + 0.16])
        p_ml = np.array([hx, Y_REAR, 0.62])
        p_mr = np.array([ex, ey, 0.62])
        if mirrored:
            p_bl, p_br, p_tr, p_tl, p_ml, p_mr = mirror_pts_across_plane([p_bl, p_br, p_tr, p_tl, p_ml, p_mr], Z_FLOOR)

        # Translucent door glass pane
        fill_3d_quad(ctx, cam, [p_ml, p_mr, p_tr, p_tl], (*c_glass, 0.26 * alpha_m))
        # Solid lower kick panel
        fill_3d_quad(ctx, cam, [p_bl, p_br, p_mr, p_ml], (*c_frame, 0.96 * alpha_m), (*c_hi, 0.96 * alpha_m))
        # Door perimeter rails & glass muntins
        for a, b, ww in (
            (p_bl, p_tl, 0.055),
            (p_br, p_tr, 0.055),
            (p_tl, p_tr, 0.055),
            (p_ml, p_mr, 0.048),
            (0.5 * (p_ml + p_tl), 0.5 * (p_mr + p_tr), 0.025),
            (0.5 * (p_ml + p_mr), 0.5 * (p_tl + p_tr), 0.025),
        ):
            stroke_3d_segment(ctx, cam, a, b, ww, (*c_frame, alpha_m))

    # 3. Arched wrought-iron ribs, side bay columns, pierced filigree gussets & tie-rods (back-to-front!)
    for yb in reversed(BAY_YS):
        dist = float(np.linalg.norm(np.array([0.0, yb, 2.8]) - cam.eye))
        c_d = fog_mix([0.105, 0.158, 0.130], dist, fog_rgb, fog_start=4.8, fog_density=0.042)
        c_m = fog_mix([0.245, 0.345, 0.285], dist, fog_rgb, fog_start=4.8, fog_density=0.042)
        c_l = fog_mix([0.495, 0.605, 0.495], dist, fog_rgb, fog_start=4.8, fog_density=0.042)

        # Main outer curved vault rib
        rib_outer = [np.array([float(xv), yb, float(vault_z(xv))]) for xv in xs_arch]
        # Delicate parallel inner truss chord (0.14 m inside) for authentic Victorian ironwork depth!
        rib_inner = [np.array([float(xv) * 0.965, yb, float(vault_z(xv)) - 0.14]) for xv in xs_arch[1:-1]]

        if mirrored:
            rib_outer = mirror_pts_across_plane(rib_outer, Z_FLOOR)
            rib_inner = mirror_pts_across_plane(rib_inner, Z_FLOOR)

        stroke_3d_ribbon(ctx, cam, rib_outer, 0.088, c_d, c_m, c_l, alpha=alpha_m)
        stroke_3d_ribbon(ctx, cam, rib_inner, 0.036, c_d, c_m, c_l, alpha=alpha_m)

        # Pierced circular lattice rings nestled between outer & inner arch chords!
        for k in range(2, len(xs_arch) - 2, 3):
            xv = float(xs_arch[k])
            pc = np.array([xv * 0.982, yb, float(vault_z(xv)) - 0.072])
            if mirrored:
                pc = mirror_pts_across_plane([pc], Z_FLOOR)[0]
            uc, vc, zc = cam.project(pc)
            if float(zc) > Z_NEAR:
                rad_px = max(1.4, 0.056 * cam.K[0, 0] / float(zc))
                ctx.set_source_rgba(*c_m, alpha_m * 0.92)
                ctx.set_line_width(max(0.8, rad_px * 0.22))
                ctx.arc(float(uc), float(vc), rad_px, 0.0, math.tau)
                ctx.stroke()

        # Left & right vertical fluted cast-iron columns supporting each vault arch
        for sx_c in (X_LEFT + 0.04, X_RIGHT - 0.04):
            p_bot = np.array([sx_c, yb, Z_FLOOR])
            p_top = np.array([sx_c, yb, Z_SPRING + 0.04])
            if mirrored:
                p_bot, p_top = mirror_pts_across_plane([p_bot, p_top], Z_FLOOR)
            stroke_3d_ribbon(ctx, cam, [p_bot, p_top], 0.115, c_d, c_m, c_l, alpha=alpha_m)

            # Ornamental curved filigree knee-brace / spandrel bracket at column spring joint!
            inward = -1.0 if sx_c > 0 else 1.0
            brace_pts = []
            for t in np.linspace(0.0, 1.0, 10):
                bx = sx_c + inward * (0.04 + 0.58 * (t ** 1.25))
                bz = (Z_SPRING - 0.56) + 0.78 * (1.0 - (1.0 - t) ** 1.7)
                brace_pts.append(np.array([bx, yb, bz]))
            if mirrored:
                brace_pts = mirror_pts_across_plane(brace_pts, Z_FLOOR)
            stroke_3d_ribbon(ctx, cam, brace_pts, 0.032, c_d, c_m, c_l, alpha=alpha_m)

            # Decorative filigree scrollwork circle tucked inside the knee brace
            sp_c = np.array([sx_c + inward * 0.20, yb, Z_SPRING - 0.14])
            if mirrored:
                sp_c = mirror_pts_across_plane([sp_c], Z_FLOOR)[0]
            us, vs, zs = cam.project(sp_c)
            if float(zs) > Z_NEAR:
                rr = max(1.6, 0.105 * cam.K[0, 0] / float(zs))
                ctx.set_source_rgba(*c_l, alpha_m * 0.85)
                ctx.set_line_width(max(0.9, rr * 0.18))
                ctx.arc(float(us), float(vs), rr, 0.0, math.tau)
                ctx.stroke()

        # Slender horizontal wrought-iron tension tie-rod spanning high across the nave (z = 3.22 m)
        if yb < Y_REAR - 0.5:
            x_tie = HALF_SPAN * math.sqrt(max(0.0, 1.0 - ((3.22 - Z_SPRING) / ARCH_RISE) ** 2))
            p_tl = np.array([-x_tie, yb, 3.22])
            p_tr = np.array([x_tie, yb, 3.22])
            p_mid = np.array([0.0, yb, 3.22])
            p_king = np.array([0.0, yb, Z_PEAK - 0.12])
            if mirrored:
                p_tl, p_tr, p_mid, p_king = mirror_pts_across_plane([p_tl, p_tr, p_mid, p_king], Z_FLOOR)
            stroke_3d_segment(ctx, cam, p_tl, p_tr, 0.022, (*c_m, alpha_m * 0.90))
            stroke_3d_segment(ctx, cam, p_mid, p_king, 0.020, (*c_m, alpha_m * 0.88))


# =============================================================================
# Stage 3 Botanical Curves (Seeded Cubic Beziers — Zero Stamped Polygons!)
# =============================================================================

def draw_fern_frond_bezier(
    ctx: cairo.Context,
    x0: float,
    y0: float,
    length: float,
    ang: float,
    droop: float,
    dark_c: tuple[float, float, float],
    mid_c: tuple[float, float, float],
    lit_c: tuple[float, float, float],
    rng: np.random.Generator,
    alpha: float = 0.95,
    n_pairs: int = 16,
    mirrored: bool = False,
) -> None:
    """Graceful arching Boston / sword fern frond with curved cubic-Bezier rachis & pinnae."""
    sign = -1.0 if mirrored else 1.0
    ca, sa = math.cos(ang), math.sin(ang)
    p1x = x0 + ca * length * 0.36
    p1y = y0 + sa * length * 0.36 - sign * length * 0.06
    p2x = x0 + ca * length * 0.72
    p2y = y0 + sa * length * 0.72 + sign * droop * 0.45
    p3x = x0 + ca * length
    p3y = y0 + sa * length + sign * droop

    # Slender curving rachis stem
    ctx.set_source_rgba(*dark_c, alpha * 0.88)
    ctx.set_line_width(max(0.8, length * 0.018))
    ctx.move_to(x0, y0)
    ctx.curve_to(p1x, p1y, p2x, p2y, p3x, p3y)
    ctx.stroke()

    # Alternating tapered cubic-Bezier pinnae leaflets along the frond
    for i in range(1, n_pairs + 1):
        t = i / (n_pairs + 1.15)
        om = 1.0 - t
        bx = om**3 * x0 + 3 * om**2 * t * p1x + 3 * om * t**2 * p2x + t**3 * p3x
        by = om**3 * y0 + 3 * om**2 * t * p1y + 3 * om * t**2 * p2y + t**3 * p3y
        tx = 3 * om**2 * (p1x - x0) + 6 * om * t * (p2x - p1x) + 3 * t**2 * (p3x - p2x)
        ty = 3 * om**2 * (p1y - y0) + 6 * om * t * (p2y - p1y) + 3 * t**2 * (p3y - p2y)
        tl = math.hypot(tx, ty) + 1e-6
        ux, uy = tx / tl, ty / tl
        nx, ny = -uy, ux

        env = math.sin(t * math.pi) ** 0.72
        plen = length * (0.14 + 0.04 * float(rng.uniform(-0.4, 0.4))) * (0.28 + 0.72 * env)
        pw = plen * 0.26

        for side in (-1.0, 1.0):
            # Angle pinna slightly forward along the frond
            dx = (nx * side * 0.82 + ux * 0.56) * plen
            dy = (ny * side * 0.82 + uy * 0.56) * plen + sign * plen * 0.14
            ex, ey = bx + dx, by + dy
            px, py = -dy / (plen + 1e-6) * pw, dx / (plen + 1e-6) * pw
            # Sunlit upper pinnae & tips glow warm matcha-gold
            lit_t = np.clip(0.25 + 0.58 * t + (0.22 if dy < 0 else -0.08) + float(rng.uniform(-0.10, 0.12)), 0.0, 1.0)
            if lit_t < 0.5:
                f = lit_t * 2.0
                col = tuple(dark_c[k] * (1 - f) + mid_c[k] * f for k in range(3))
            else:
                f = (lit_t - 0.5) * 2.0
                col = tuple(mid_c[k] * (1 - f) + lit_c[k] * f for k in range(3))
            ctx.set_source_rgba(*col, alpha)
            ctx.move_to(bx, by)
            ctx.curve_to(bx + dx * 0.35 + px, by + dy * 0.35 + py, bx + dx * 0.75 + px * 0.4, by + dy * 0.75 + py * 0.4, ex, ey)
            ctx.curve_to(bx + dx * 0.75 - px * 0.4, by + dy * 0.75 - py * 0.4, bx + dx * 0.35 - px, by + dy * 0.35 - py, bx, by)
            ctx.close_path()
            ctx.fill()


def draw_monstera_leaf_bezier(
    ctx: cairo.Context,
    x: float,
    y: float,
    size: float,
    ang: float,
    dark_c: tuple[float, float, float],
    mid_c: tuple[float, float, float],
    lit_c: tuple[float, float, float],
    vein_c: tuple[float, float, float],
    alpha: float = 0.98,
    backlit: float = 0.45,
) -> None:
    """
    Lush hand-painted fenestrated Monstera deliciosa leaf built purely from smooth
    cubic Bezier lobes, tapered curved veins routed cleanly BETWEEN fenestrations
    (NEVER crossing a hole!), elongated almond-shaped botanical fenestrations, deep
    sinus shading, and luminous wet-leaf morning specular highlights!
    """
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(ang)

    w = size * 0.84
    h = size * 1.12

    def _trace_monstera_path() -> None:
        ctx.new_path()
        # Start at heart base notch
        ctx.move_to(0.0, h * 0.08)
        # Right side sweeping upward with 3 organic curved Bezier fenestration splits
        ctx.curve_to(w * 0.36, -h * 0.08, w * 0.58, h * 0.14, w * 0.54, h * 0.34)
        ctx.curve_to(w * 0.46, h * 0.36, w * 0.24, h * 0.38, w * 0.22, h * 0.41)
        ctx.curve_to(w * 0.24, h * 0.44, w * 0.48, h * 0.45, w * 0.53, h * 0.49)
        ctx.curve_to(w * 0.54, h * 0.60, w * 0.46, h * 0.68, w * 0.44, h * 0.70)
        ctx.curve_to(w * 0.36, h * 0.70, w * 0.19, h * 0.69, w * 0.18, h * 0.72)
        ctx.curve_to(w * 0.20, h * 0.75, w * 0.36, h * 0.76, w * 0.40, h * 0.80)
        ctx.curve_to(w * 0.34, h * 0.94, w * 0.16, h * 1.04, 0.0, h * 1.10)
        # Asymmetric left side
        ctx.curve_to(-w * 0.16, h * 1.03, -w * 0.35, h * 0.93, -w * 0.41, h * 0.78)
        ctx.curve_to(-w * 0.36, h * 0.74, -w * 0.19, h * 0.73, -w * 0.17, h * 0.70)
        ctx.curve_to(-w * 0.19, h * 0.67, -w * 0.39, h * 0.67, -w * 0.46, h * 0.66)
        ctx.curve_to(-w * 0.52, h * 0.58, -w * 0.55, h * 0.49, -w * 0.52, h * 0.45)
        ctx.curve_to(-w * 0.44, h * 0.42, -w * 0.22, h * 0.41, -w * 0.21, h * 0.38)
        ctx.curve_to(-w * 0.23, h * 0.35, -w * 0.46, h * 0.33, -w * 0.53, h * 0.28)
        ctx.curve_to(-w * 0.56, h * 0.10, -w * 0.34, -h * 0.09, 0.0, h * 0.08)
        ctx.close_path()

        # Elongated botanical almond-shaped fenestration holes aligned along the lateral veins!
        for fx, fy, rx_h, ry_h, rot_h in (
            (w * 0.20, h * 0.24, w * 0.085, h * 0.028, 0.22),
            (-w * 0.19, h * 0.54, w * 0.088, h * 0.028, -0.24),
            (w * 0.17, h * 0.57, w * 0.072, h * 0.024, 0.20),
        ):
            ctx.new_sub_path()
            ctx.save()
            ctx.translate(fx, fy)
            ctx.rotate(rot_h)
            ctx.scale(rx_h, ry_h)
            ctx.arc(0.0, 0.0, 1.0, 0.0, math.tau)
            ctx.restore()

    # Clip everything (gradient, curved lobe shadows, veins, glossy highlights) strictly to the fenestrated blade!
    ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    _trace_monstera_path()
    ctx.clip_preserve()

    # Deep rich botanical gouache gradient (warm backlit matcha-emerald on left/upper half -> deep forest shade)
    c_hi = tuple(mid_c[k] * (1 - backlit) + lit_c[k] * backlit for k in range(3))
    g = cairo.LinearGradient(-w * 0.48, h * 0.05, w * 0.48, h * 0.98)
    g.add_color_stop_rgba(0.0, *c_hi, alpha)
    g.add_color_stop_rgba(0.38, *mid_c, alpha)
    g.add_color_stop_rgba(0.82, *dark_c, alpha)
    g.add_color_stop_rgba(1.0, *(dark_c[k] * 0.78 for k in range(3)), alpha)
    ctx.set_source(g)
    ctx.fill()

    # Subtle shaded right half across midrib fold for genuine 3D leaf camber
    ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
    ctx.set_source_rgba(*dark_c, alpha * 0.24)
    ctx.rectangle(0.0, -h * 0.1, w * 0.65, h * 1.25)
    ctx.fill()

    # Soft wet-leaf morning specular highlights kissing individual curved lobes!
    for hx, hy, hr in ((-w * 0.26, h * 0.22, size * 0.28), (-w * 0.24, h * 0.58, size * 0.22), (w * 0.22, h * 0.34, size * 0.20)):
        g_sp = cairo.RadialGradient(hx, hy, hr * 0.06, hx, hy, hr)
        g_sp.add_color_stop_rgba(0.0, 0.88, 0.94, 0.68, alpha * 0.24 * (0.45 + backlit))
        g_sp.add_color_stop_rgba(1.0, 0.88, 0.94, 0.68, 0.0)
        ctx.set_source(g_sp)
        ctx.arc(hx, hy, hr, 0.0, math.tau)
        ctx.fill()

    # Delicate tapered pale-chartreuse central midrib & curved lateral veins (clipped strictly off holes!)
    ctx.set_source_rgba(*vein_c, alpha * 0.62)
    ctx.set_line_width(max(1.0, size * 0.022))
    ctx.move_to(0.0, h * 0.07)
    ctx.curve_to(w * 0.02, h * 0.40, -w * 0.01, h * 0.72, 0.0, h * 1.03)
    ctx.stroke()

    ctx.set_line_width(max(0.65, size * 0.011))
    for vy_f, side_s, reach in (
        (0.16, 1.0, 0.44),
        (0.17, -1.0, 0.42),
        (0.47, 1.0, 0.42),
        (0.45, -1.0, 0.42),
        (0.76, 1.0, 0.30),
        (0.75, -1.0, 0.29),
    ):
        ctx.move_to(0.0, h * vy_f)
        ctx.curve_to(side_s * w * reach * 0.45, h * (vy_f + 0.02), side_s * w * reach * 0.85, h * (vy_f + 0.07), side_s * w * reach, h * (vy_f + 0.12))
        ctx.stroke()

    # Crisp soft dark botanical rim contour along shaded leaf margin for painterly Ghibli readability
    ctx.reset_clip()
    ctx.set_source_rgba(*(dark_c[k] * 0.75 for k in range(3)), alpha * 0.42)
    ctx.set_line_width(max(0.75, size * 0.008))
    _trace_monstera_path()
    ctx.stroke()
    ctx.restore()


def draw_banana_leaf_bezier(
    ctx: cairo.Context,
    x0: float,
    y0: float,
    length: float,
    width: float,
    ang: float,
    droop: float,
    dark_c: tuple[float, float, float],
    mid_c: tuple[float, float, float],
    lit_c: tuple[float, float, float],
    rng: np.random.Generator,
    alpha: float = 0.98,
    backlit: float = 0.60,
) -> None:
    """
    Sweeping broad tropical banana / bird-of-paradise blade built via cubic Beziers
    with translucent backlit golden-matcha lamina, pale midrib, and fine parallel veins!
    """
    ca, sa = math.cos(ang), math.sin(ang)
    nx, ny = -sa, ca
    # Curved centerline Bezier
    p1x, p1y = x0 + ca * length * 0.38, y0 + sa * length * 0.38 - length * 0.06
    p2x, p2y = x0 + ca * length * 0.74, y0 + sa * length * 0.74 + droop * 0.42
    p3x, p3y = x0 + ca * length, y0 + sa * length + droop

    # Broad tapering leaf blade silhouette (left & right curving margins)
    g = cairo.LinearGradient(x0, y0, p3x, p3y)
    c_tip = tuple(mid_c[k] * (1.0 - backlit) + lit_c[k] * backlit for k in range(3))
    g.add_color_stop_rgba(0.0, *dark_c, alpha)
    g.add_color_stop_rgba(0.45, *mid_c, alpha)
    g.add_color_stop_rgba(0.88, *c_tip, alpha)
    g.add_color_stop_rgba(1.0, *lit_c, alpha * 0.96)
    ctx.set_source(g)

    ctx.move_to(x0, y0)
    ctx.curve_to(
        p1x + nx * width * 0.56,
        p1y + ny * width * 0.56,
        p2x + nx * width * 0.48,
        p2y + ny * width * 0.48,
        p3x,
        p3y,
    )
    ctx.curve_to(
        p2x - nx * width * 0.46,
        p2y - ny * width * 0.46,
        p1x - nx * width * 0.54,
        p1y - ny * width * 0.54,
        x0,
        y0,
    )
    ctx.close_path()
    ctx.fill()

    # Shaded lower half of blade (subtle folded V-channel across central midrib!)
    ctx.set_source_rgba(*dark_c, alpha * 0.24)
    ctx.move_to(x0, y0)
    ctx.curve_to(p1x, p1y, p2x, p2y, p3x, p3y)
    ctx.curve_to(
        p2x - nx * width * 0.44,
        p2y - ny * width * 0.44,
        p1x - nx * width * 0.50,
        p1y - ny * width * 0.50,
        x0,
        y0,
    )
    ctx.close_path()
    ctx.fill()

    # Tapered luminous translucent chartreuse central midrib (thick stem base -> hairline apex tip!)
    n_mr = 12
    for k in range(n_mr):
        t0, t1 = k / n_mr, (k + 1) / n_mr
        om0, om1 = 1.0 - t0, 1.0 - t1
        mx0 = om0**3 * x0 + 3 * om0**2 * t0 * p1x + 3 * om0 * t0**2 * p2x + t0**3 * p3x
        my0 = om0**3 * y0 + 3 * om0**2 * t0 * p1y + 3 * om0 * t0**2 * p2y + t0**3 * p3y
        mx1 = om1**3 * x0 + 3 * om1**2 * t1 * p1x + 3 * om1 * t1**2 * p2x + t1**3 * p3x
        my1 = om1**3 * y0 + 3 * om1**2 * t1 * p1y + 3 * om1 * t1**2 * p2y + t1**3 * p3y
        ctx.set_source_rgba(*(lit_c[j] * 0.85 + 0.15 for j in range(3)), alpha * (0.86 - 0.28 * t0))
        ctx.set_line_width(max(0.75, width * 0.056 * (1.0 - 0.82 * t0)))
        ctx.move_to(mx0, my0)
        ctx.line_to(mx1, my1)
        ctx.stroke()

    # Fine diagonal feather veins & occasional delicate botanical wind-split shading along blade
    ctx.set_source_rgba(*lit_c, alpha * 0.28)
    ctx.set_line_width(max(0.65, width * 0.014))
    for k in range(1, 13):
        t = k / 13.5
        om = 1.0 - t
        bx = om**3 * x0 + 3 * om**2 * t * p1x + 3 * om * t**2 * p2x + t**3 * p3x
        by = om**3 * y0 + 3 * om**2 * t * p1y + 3 * om * t**2 * p2y + t**3 * p3y
        vw = width * 0.42 * math.sin(t * math.pi)
        for s in (-1.0, 1.0):
            ctx.move_to(bx, by)
            ctx.line_to(bx + (nx * s * 0.78 + ca * 0.42) * vw, by + (ny * s * 0.78 + sa * 0.42) * vw)
            ctx.stroke()
            # Occasional subtle dark botanical wind-notch crease on outer blade margin
            if k in (4, 9) and s > 0:
                ctx.set_source_rgba(*(dark_c[j] * 0.78 for j in range(3)), alpha * 0.45)
                ctx.set_line_width(max(0.9, width * 0.018))
                ex_n = bx + (nx * s * 0.92 + ca * 0.36) * (vw * 1.18)
                ey_n = by + (ny * s * 0.92 + sa * 0.36) * (vw * 1.18)
                ctx.move_to(bx + (nx * s * 0.36 + ca * 0.18) * vw, by + (ny * s * 0.36 + sa * 0.18) * vw)
                ctx.line_to(ex_n, ey_n)
                ctx.stroke()
                ctx.set_source_rgba(*lit_c, alpha * 0.28)
                ctx.set_line_width(max(0.65, width * 0.014))

    # Soft wet-leaf morning specular highlight sheen kissing upper sunlit margin
    hx_b = 0.52 * p1x + 0.48 * p2x + nx * width * 0.22
    hy_b = 0.52 * p1y + 0.48 * p2y + ny * width * 0.22
    g_sp = cairo.RadialGradient(hx_b, hy_b, width * 0.06, hx_b, hy_b, width * 0.65)
    g_sp.add_color_stop_rgba(0.0, 0.90, 0.95, 0.70, alpha * 0.22 * backlit)
    g_sp.add_color_stop_rgba(1.0, 0.90, 0.95, 0.70, 0.0)
    ctx.set_source(g_sp)
    ctx.arc(hx_b, hy_b, width * 0.65, 0.0, math.tau)
    ctx.fill()

    # Soft dark painterly botanical contour along shaded lower blade edge
    ctx.set_source_rgba(*(dark_c[k] * 0.72 for k in range(3)), alpha * 0.38)
    ctx.set_line_width(max(0.85, width * 0.016))
    ctx.move_to(p3x, p3y)
    ctx.curve_to(
        p2x - nx * width * 0.46,
        p2y - ny * width * 0.46,
        p1x - nx * width * 0.54,
        p1y - ny * width * 0.54,
        x0,
        y0,
    )
    ctx.stroke()


def draw_ivy_leaf_bezier(
    ctx: cairo.Context,
    x: float,
    y: float,
    size: float,
    ang: float,
    col: tuple[float, float, float],
    vein_c: tuple[float, float, float],
    alpha: float = 0.96,
) -> None:
    """Classic 3-lobed English ivy leaf painted with smooth cubic Beziers."""
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(ang)
    s = size
    ctx.set_source_rgba(*col, alpha)
    ctx.move_to(0.0, 0.0)
    ctx.curve_to(s * 0.48, -s * 0.16, s * 0.62, s * 0.22, s * 0.44, s * 0.42)
    ctx.curve_to(s * 0.34, s * 0.48, s * 0.28, s * 0.74, 0.0, s * 0.98)
    ctx.curve_to(-s * 0.28, s * 0.74, -s * 0.34, s * 0.48, -s * 0.44, s * 0.42)
    ctx.curve_to(-s * 0.62, s * 0.22, -s * 0.48, -s * 0.16, 0.0, 0.0)
    ctx.close_path()
    ctx.fill()

    ctx.set_source_rgba(*vein_c, alpha * 0.55)
    ctx.set_line_width(max(0.55, s * 0.06))
    ctx.move_to(0.0, s * 0.08)
    ctx.line_to(0.0, s * 0.82)
    ctx.move_to(0.0, s * 0.25)
    ctx.line_to(s * 0.30, s * 0.36)
    ctx.move_to(0.0, s * 0.25)
    ctx.line_to(-s * 0.30, s * 0.36)
    ctx.stroke()
    ctx.restore()


def draw_climbing_ivy_on_column(
    ctx: cairo.Context,
    cam: Camera3D,
    col_x: float,
    col_y: float,
    z_start: float,
    z_end: float,
    seed: int,
    fog_rgb: np.ndarray,
    mirrored: bool = False,
) -> None:
    """Climbing English ivy vine spiraling organically up a wrought-iron conservatory column."""
    srng = np.random.default_rng(seed)
    dist = float(np.linalg.norm(np.array([col_x, col_y, 0.5 * (z_start + z_end)]) - cam.eye))
    alpha_m = 0.55 if mirrored else 0.98
    c_stem = fog_mix([0.14, 0.22, 0.12], dist, fog_rgb)
    c_dark = fog_mix([0.10, 0.22, 0.12], dist, fog_rgb)
    c_mid = fog_mix([0.22, 0.39, 0.20], dist, fog_rgb)
    c_lit = fog_mix([0.52, 0.67, 0.34], dist, fog_rgb)
    c_vein = fog_mix([0.74, 0.84, 0.52], dist, fog_rgb)

    inward = 1.0 if col_x < 0 else -1.0
    n_pts = 22
    vine_pts3 = []
    for k in range(n_pts):
        f = k / (n_pts - 1)
        zv = z_start + f * (z_end - z_start)
        ang = f * 7.8 + float(srng.uniform(-0.2, 0.2))
        vx = col_x + inward * (0.065 + 0.055 * math.cos(ang))
        vy = col_y + 0.075 * math.sin(ang)
        vine_pts3.append(np.array([vx, vy, zv]))
    if mirrored:
        vine_pts3 = mirror_pts_across_plane(vine_pts3, Z_FLOOR)

    for k in range(len(vine_pts3) - 1):
        stroke_3d_segment(ctx, cam, vine_pts3[k], vine_pts3[k + 1], 0.015, (*c_stem, alpha_m * 0.90))

    # Rich cluster of 3-lobed English ivy leaves clinging & draping off the vine
    for k in range(1, len(vine_pts3)):
        u, v, zc = cam.project(vine_pts3[k])
        if float(zc) < Z_NEAR:
            continue
        px_m = cam.K[0, 0] / float(zc)
        for _ in range(3):
            ox = float(srng.uniform(-0.08, 0.10)) * px_m
            oy = float(srng.uniform(-0.05, 0.06)) * px_m
            ls = float(srng.uniform(0.055, 0.105)) * px_m
            lang = float(srng.uniform(-1.3, 1.3)) + (math.pi if mirrored else 0.0)
            t_l = float(srng.uniform(0.0, 1.0))
            col = c_lit if t_l > 0.66 else (c_mid if t_l > 0.26 else c_dark)
            draw_ivy_leaf_bezier(ctx, float(u) + ox, float(v) + oy, ls, lang, col, c_vein, alpha=alpha_m)


def draw_hanging_fern_basket(
    ctx: cairo.Context,
    cam: Camera3D,
    pos3: np.ndarray,
    anchor_z: float,
    radius: float,
    seed: int,
    fog_rgb: np.ndarray,
    mirrored: bool = False,
) -> tuple[float, float, float]:
    """
    Hanging wrought-iron & coconut-fiber fern basket suspended cleanly on 3 taut chains
    up to a single iron hook anchored right on the vault rib/tie-rod at anchor_z!
    Overflows with lush cascading cubic-Bezier Boston fern fronds & trailing ivy!
    Returns projected screen UV (u, v, radius_u).
    """
    srng = np.random.default_rng(seed)
    dist = float(np.linalg.norm(pos3 - cam.eye))
    alpha_m = 0.56 if mirrored else 0.99

    p_c = mirror_pts_across_plane([pos3], Z_FLOOR)[0] if mirrored else pos3
    p_hook = np.array([pos3[0], pos3[1], anchor_z], dtype=np.float64)
    p_mid_ring = np.array([pos3[0], pos3[1], pos3[2] + 0.58], dtype=np.float64)
    if mirrored:
        p_hook, p_mid_ring = mirror_pts_across_plane([p_hook, p_mid_ring], Z_FLOOR)

    c_iron = fog_mix([0.15, 0.22, 0.18], dist, fog_rgb)
    c_basket_d = fog_mix([0.24, 0.17, 0.11], dist, fog_rgb)
    c_basket_l = fog_mix([0.46, 0.34, 0.22], dist, fog_rgb)
    f_dark = fog_mix([0.09, 0.21, 0.12], dist, fog_rgb)
    f_mid = fog_mix([0.22, 0.40, 0.21], dist, fog_rgb)
    f_lit = fog_mix([0.62, 0.76, 0.38], dist, fog_rgb)

    # Taut suspension rod + 3 triangular basket chains (zero floating!)
    stroke_3d_segment(ctx, cam, p_hook, p_mid_ring, 0.014, (*c_iron, alpha_m))
    for ang_c in (0.3, 2.4, 4.5):
        rim_pt = np.array([pos3[0] + math.cos(ang_c) * radius, pos3[1] + math.sin(ang_c) * radius, pos3[2] + 0.04])
        if mirrored:
            rim_pt = mirror_pts_across_plane([rim_pt], Z_FLOOR)[0]
        stroke_3d_segment(ctx, cam, p_mid_ring, rim_pt, 0.010, (*c_iron, alpha_m * 0.92))

    uc, vc, zc = cam.project(p_c)
    if float(zc) < Z_NEAR:
        return (0.5, 0.2, 0.03)
    px_m = cam.K[0, 0] / float(zc)
    rx = radius * px_m
    ry = rx * 0.72
    sign = -1.0 if mirrored else 1.0

    # Rich deep-emerald foliage crown cushion nestled inside basket top for authentic lush Ghibli volume!
    for k_c in range(9):
        ox_c = float(srng.uniform(-rx * 0.78, rx * 0.78))
        oy_c = -sign * float(srng.uniform(ry * 0.05, ry * 0.72))
        rc = rx * float(srng.uniform(0.32, 0.56))
        col_c = f_dark if k_c < 4 else (f_mid if k_c < 7 else f_lit)
        g_c = cairo.RadialGradient(float(uc) + ox_c - rc * 0.2, float(vc) + oy_c - sign * rc * 0.2, rc * 0.1, float(uc) + ox_c, float(vc) + oy_c, rc)
        g_c.add_color_stop_rgba(0.0, *col_c, alpha_m * 0.88)
        g_c.add_color_stop_rgba(1.0, *col_c, 0.0)
        ctx.set_source(g_c)
        ctx.arc(float(uc) + ox_c, float(vc) + oy_c, rc, 0.0, math.tau)
        ctx.fill()

    # Back fern fronds arching upward & outward behind basket rim
    for k in range(11):
        ang = -math.pi * 0.98 + (k / 10.0) * math.pi * 0.96 + float(srng.uniform(-0.12, 0.12))
        if mirrored:
            ang = -ang
        flen = rx * float(srng.uniform(1.60, 2.55))
        draw_fern_frond_bezier(
            ctx,
            float(uc) + float(srng.uniform(-rx * 0.58, rx * 0.58)),
            float(vc) - sign * ry * 0.14,
            flen,
            ang,
            droop=flen * float(srng.uniform(0.34, 0.62)),
            dark_c=f_dark,
            mid_c=f_mid,
            lit_c=f_lit,
            rng=srng,
            alpha=alpha_m * 0.94,
            n_pairs=15,
            mirrored=mirrored,
        )

    # Rounded woven iron & mossy coconut-coir hanging basket bowl
    g_b = cairo.LinearGradient(float(uc) - rx, float(vc), float(uc) + rx, float(vc) + sign * ry)
    g_b.add_color_stop_rgba(0.0, *c_basket_l, alpha_m)
    g_b.add_color_stop_rgba(0.55, *c_basket_d, alpha_m)
    g_b.add_color_stop_rgba(1.0, *(c_basket_d[k] * 0.58 for k in range(3)), alpha_m)
    ctx.set_source(g_b)
    ctx.move_to(float(uc) - rx, float(vc))
    ctx.curve_to(float(uc) - rx * 0.86, float(vc) + sign * ry * 1.18, float(uc) + rx * 0.86, float(vc) + sign * ry * 1.18, float(uc) + rx, float(vc))
    ctx.close_path()
    ctx.fill()

    # Ornamental curved iron basket ribs + rim band
    ctx.set_source_rgba(*c_iron, alpha_m * 0.92)
    ctx.set_line_width(max(1.2, rx * 0.065))
    ctx.move_to(float(uc) - rx * 1.04, float(vc))
    ctx.line_to(float(uc) + rx * 1.04, float(vc))
    ctx.stroke()
    ctx.set_line_width(max(0.85, rx * 0.04))
    for rib_f in (-0.55, 0.0, 0.55):
        ctx.move_to(float(uc) + rx * rib_f, float(vc))
        ctx.curve_to(float(uc) + rx * rib_f * 0.85, float(vc) + sign * ry * 0.62, float(uc) + rx * rib_f * 0.35, float(vc) + sign * ry * 0.92, float(uc), float(vc) + sign * ry * 0.92)
        ctx.stroke()

    # Lush front cascading Boston fern fronds + delicate trailing ivy strands spilling over rim!
    for k in range(14):
        frac = k / 13.0
        fx = float(uc) + (frac - 0.5) * rx * 1.72
        ang = (math.pi * 0.92 - frac * math.pi * 0.84 + float(srng.uniform(-0.14, 0.14))) * (1.0 if not mirrored else -1.0)
        flen = rx * float(srng.uniform(1.55, 2.75))
        # Alternate shaded inner layer & sunlit outer cascading fronds for painterly depth!
        is_inner = k % 3 == 0
        draw_fern_frond_bezier(
            ctx,
            fx,
            float(vc) + sign * ry * float(srng.uniform(-0.08, 0.14)),
            flen,
            ang,
            droop=flen * float(srng.uniform(0.38, 0.72)),
            dark_c=f_dark,
            mid_c=f_dark if is_inner else f_mid,
            lit_c=f_mid if is_inner else f_lit,
            rng=srng,
            alpha=alpha_m,
            n_pairs=16,
            mirrored=mirrored,
        )

    return (float(uc) / W, float(vc) / H, float(rx * 2.1) / W)


def draw_vintage_brass_lamp(
    ctx: cairo.Context,
    cam: Camera3D,
    pos3: np.ndarray,
    anchor_z: float,
    fog_rgb: np.ndarray,
    mirrored: bool = False,
) -> tuple[float, float]:
    """
    Vintage Meiji/Victorian conservatory pendant lamp suspended from the central ridge:
    taut brass suspension rod, scalloped sage-enameled metal reflector shade, antique
    brass socket gallery & warm glowing glass hurricane globe.
    """
    dist = float(np.linalg.norm(pos3 - cam.eye))
    alpha_m = 0.56 if mirrored else 0.995
    p_top = np.array([pos3[0], pos3[1], anchor_z], dtype=np.float64)
    p_c = pos3.copy()
    if mirrored:
        p_top, p_c = mirror_pts_across_plane([p_top, p_c], Z_FLOOR)

    c_brass_d = fog_mix([0.34, 0.25, 0.12], dist, fog_rgb)
    c_brass_l = fog_mix([0.82, 0.68, 0.36], dist, fog_rgb)
    c_shade_d = fog_mix([0.14, 0.22, 0.17], dist, fog_rgb)
    c_shade_l = fog_mix([0.36, 0.49, 0.39], dist, fog_rgb)

    stroke_3d_segment(ctx, cam, p_top, p_c, 0.018, (*c_brass_d, alpha_m))

    uc, vc, zc = cam.project(p_c)
    if float(zc) < Z_NEAR:
        return (0.5, 0.2)
    px_m = cam.K[0, 0] / float(zc)
    rw = 0.22 * px_m
    rh = 0.11 * px_m
    sign = -1.0 if mirrored else 1.0

    # Soft warm morning bulb glow inside glass globe
    gy = float(vc) + sign * rh * 1.18
    g_glow = cairo.RadialGradient(float(uc), gy, rw * 0.08, float(uc), gy, rw * 1.25)
    g_glow.add_color_stop_rgba(0.0, 1.00, 0.95, 0.72, 0.68 * alpha_m)
    g_glow.add_color_stop_rgba(0.38, 0.96, 0.86, 0.54, 0.28 * alpha_m)
    g_glow.add_color_stop_rgba(1.0, 0.90, 0.80, 0.46, 0.0)
    ctx.set_source(g_glow)
    ctx.arc(float(uc), gy, rw * 1.25, 0.0, math.tau)
    ctx.fill()

    # Glass hurricane bulb globe
    ctx.save()
    ctx.translate(float(uc), gy)
    ctx.scale(rw * 0.36, rh * 0.78)
    ctx.set_source_rgba(0.98, 0.95, 0.78, 0.82 * alpha_m)
    ctx.arc(0.0, 0.0, 1.0, 0.0, math.tau)
    ctx.fill()
    ctx.restore()

    # Warm cream-enameled underside glow inside vintage reflector shade
    ctx.save()
    ctx.translate(float(uc), float(vc) + sign * rh * 0.88)
    ctx.scale(rw * 0.94, rh * 0.24)
    ctx.set_source_rgba(0.96, 0.92, 0.72, 0.84 * alpha_m)
    ctx.arc(0.0, 0.0, 1.0, 0.0, math.tau)
    ctx.fill()
    ctx.restore()

    # Flared vintage sage-enameled metal reflector shade with sunlit morning sheen!
    g_sh = cairo.LinearGradient(float(uc) - rw, float(vc), float(uc) + rw, float(vc) + sign * rh)
    g_sh.add_color_stop_rgba(0.0, *c_shade_l, alpha_m)
    g_sh.add_color_stop_rgba(0.28, *(min(1.0, c_shade_l[k] * 1.24) for k in range(3)), alpha_m)
    g_sh.add_color_stop_rgba(0.54, *c_shade_d, alpha_m)
    g_sh.add_color_stop_rgba(1.0, *(c_shade_d[k] * 0.58 for k in range(3)), alpha_m)
    ctx.set_source(g_sh)
    ctx.move_to(float(uc) - rw * 0.26, float(vc) - sign * rh * 0.42)
    ctx.curve_to(float(uc) - rw * 0.52, float(vc) + sign * rh * 0.20, float(uc) - rw * 0.98, float(vc) + sign * rh * 0.55, float(uc) - rw, float(vc) + sign * rh * 0.86)
    ctx.curve_to(float(uc), float(vc) + sign * rh * 1.08, float(uc), float(vc) + sign * rh * 1.08, float(uc) + rw, float(vc) + sign * rh * 0.86)
    ctx.curve_to(float(uc) + rw * 0.98, float(vc) + sign * rh * 0.55, float(uc) + rw * 0.52, float(vc) + sign * rh * 0.20, float(uc) + rw * 0.26, float(vc) - sign * rh * 0.42)
    ctx.close_path()
    ctx.fill()

    # Delicate brass lip trim around outer edge of enameled shade
    ctx.set_source_rgba(*c_brass_l, 0.78 * alpha_m)
    ctx.set_line_width(max(1.0, rw * 0.04))
    ctx.move_to(float(uc) - rw, float(vc) + sign * rh * 0.86)
    ctx.curve_to(float(uc), float(vc) + sign * rh * 1.08, float(uc), float(vc) + sign * rh * 1.08, float(uc) + rw, float(vc) + sign * rh * 0.86)
    ctx.stroke()

    # Warm antique brass collar gallery atop shade
    g_br = cairo.LinearGradient(float(uc) - rw * 0.28, float(vc), float(uc) + rw * 0.28, float(vc))
    g_br.add_color_stop_rgba(0.0, *c_brass_l, alpha_m)
    g_br.add_color_stop_rgba(0.6, *c_brass_d, alpha_m)
    g_br.add_color_stop_rgba(1.0, *(c_brass_d[k] * 0.65 for k in range(3)), alpha_m)
    ctx.set_source(g_br)
    ctx.rectangle(float(uc) - rw * 0.24, float(vc) - sign * rh * 0.88 - (0.0 if not mirrored else rh * 0.52), rw * 0.48, rh * 0.52)
    ctx.fill()

    return (float(uc) / W, float(gy) / H)


# =============================================================================
# Stage 3 Band 2 — Cedar Bench Frame, Terracotta Pots, Copper Watering Can & Props
# =============================================================================

def draw_cedar_bench_structure(
    ctx: cairo.Context, cam: Camera3D, fog_rgb: np.ndarray, mirrored: bool = False
) -> None:
    """Grounded stout weathered cedar potting workbench legs, cross-rails & front apron fascia."""
    alpha_m = 0.54 if mirrored else 0.995
    leg_ys = (BENCH_Y0 + 0.14, 4.75, 7.15, BENCH_Y1 - 0.14)
    leg_xs = (BENCH_X0 + 0.10, BENCH_X1 - 0.10)

    # Legs (back-to-front)
    for ly in reversed(leg_ys):
        for lx in leg_xs:
            dist = float(np.linalg.norm(np.array([lx, ly, 0.45]) - cam.eye))
            c_lit = fog_mix([0.46, 0.34, 0.23], dist, fog_rgb)
            c_mid = fog_mix([0.30, 0.22, 0.15], dist, fog_rgb)
            c_dk = fog_mix([0.16, 0.115, 0.08], dist, fog_rgb)
            hw = 0.044
            # Front (-Y) face & aisle-facing (+X) face
            q_side = [
                np.array([lx + hw, ly - hw, Z_FLOOR]),
                np.array([lx + hw, ly + hw, Z_FLOOR]),
                np.array([lx + hw, ly + hw, BENCH_Z - 0.02]),
                np.array([lx + hw, ly - hw, BENCH_Z - 0.02]),
            ]
            q_front = [
                np.array([lx - hw, ly - hw, Z_FLOOR]),
                np.array([lx + hw, ly - hw, Z_FLOOR]),
                np.array([lx + hw, ly - hw, BENCH_Z - 0.02]),
                np.array([lx - hw, ly - hw, BENCH_Z - 0.02]),
            ]
            if mirrored:
                q_side = mirror_pts_across_plane(q_side, Z_FLOOR)
                q_front = mirror_pts_across_plane(q_front, Z_FLOOR)
            fill_3d_quad(ctx, cam, q_side, (*c_mid, alpha_m), (*c_dk, alpha_m))
            fill_3d_quad(ctx, cam, q_front, (*c_lit, alpha_m), (*c_mid, alpha_m))

            # Feathered dark contact shadow pad under each leg on wet floor
            if not mirrored:
                ub, vb, zb = cam.project(np.array([lx, ly, Z_FLOOR + 0.005]))
                if float(zb) > Z_NEAR:
                    rr = 0.095 * cam.K[0, 0] / float(zb)
                    g_ao = cairo.RadialGradient(float(ub), float(vb), rr * 0.1, float(ub), float(vb), rr)
                    g_ao.add_color_stop_rgba(0.0, 0.04, 0.06, 0.05, 0.62)
                    g_ao.add_color_stop_rgba(1.0, 0.04, 0.06, 0.05, 0.0)
                    ctx.set_source(g_ao)
                    ctx.save()
                    ctx.translate(float(ub), float(vb))
                    ctx.scale(1.0, 0.42)
                    ctx.arc(0, 0, rr, 0, math.tau)
                    ctx.fill()
                    ctx.restore()

    # Longitudinal aisle-facing tabletop apron fascia & lower shelf stretcher rail
    for z_top, h_rail, shade_m in ((BENCH_Z, 0.095, 1.0), (BENCH_SHELF_Z + 0.02, 0.055, 0.72)):
        pts_a = [
            np.array([BENCH_X1, BENCH_Y0, z_top - h_rail]),
            np.array([BENCH_X1, BENCH_Y1, z_top - h_rail]),
            np.array([BENCH_X1, BENCH_Y1, z_top]),
            np.array([BENCH_X1, BENCH_Y0, z_top]),
        ]
        if mirrored:
            pts_a = mirror_pts_across_plane(pts_a, Z_FLOOR)
        fill_3d_quad(
            ctx,
            cam,
            pts_a,
            (0.38 * shade_m, 0.28 * shade_m, 0.19 * shade_m, alpha_m),
            (0.24 * shade_m, 0.175 * shade_m, 0.12 * shade_m, alpha_m),
        )


def draw_tiered_plant_rack(ctx: cairo.Context, cam: Camera3D, rng: np.random.Generator, fog_rgb: np.ndarray) -> None:
    """3-tier stepped weathered cedar plant stand receding along left glass wall (y in [9.9, 13.7])."""
    y0, y1 = 9.95, 13.65
    tiers = [
        (-1.60, -1.24, 0.44),
        (-2.06, -1.62, 0.92),
        (-2.52, -2.08, 1.40),
    ]
    dist = float(np.linalg.norm(np.array([-1.9, 11.8, 0.9]) - cam.eye))
    c_top = fog_mix([0.44, 0.34, 0.23], dist, fog_rgb)
    c_side = fog_mix([0.28, 0.21, 0.14], dist, fog_rgb)
    c_dk = fog_mix([0.16, 0.12, 0.08], dist, fog_rgb)

    # Vertical cedar end frames
    for yp in (y0 + 0.08, 11.80, y1 - 0.08):
        for x0, x1, zt in tiers:
            stroke_3d_segment(ctx, cam, np.array([x1 - 0.03, yp, Z_FLOOR]), np.array([x1 - 0.03, yp, zt]), 0.055, (*c_dk, 0.98))

    # Draw shelves highest-to-lowest (back-to-front relative to camera) & populate with lush potted plants!
    for idx_t, (x0, x1, zt) in enumerate(reversed(tiers)):
        q_top = [
            np.array([x0, y0, zt]),
            np.array([x1, y0, zt]),
            np.array([x1, y1, zt]),
            np.array([x0, y1, zt]),
        ]
        q_face = [
            np.array([x1, y0, zt - 0.06]),
            np.array([x1, y1, zt - 0.06]),
            np.array([x1, y1, zt]),
            np.array([x1, y0, zt]),
        ]
        fill_3d_quad(ctx, cam, q_top, (*c_top, 0.99), (*c_side, 0.99))
        fill_3d_quad(ctx, cam, q_face, (*c_side, 0.99), (*c_dk, 0.99))

        # Populate each tier with terracotta, matcha & cream stoneware pots overflowing with diverse botanicals
        xc = 0.5 * (x0 + x1)
        tier_kinds = ("fern", "calathea", "bushy", "trailing", "sansevieria", "fern")
        tier_styles = ("terracotta", "matcha", "cream", "terracotta", "stone", "matcha")
        for pk, py in enumerate(np.linspace(y1 - 0.28, y0 + 0.28, 6)):
            style = tier_styles[(pk + idx_t * 2) % len(tier_styles)]
            rad = 0.115 + 0.025 * ((pk + idx_t) % 2)
            ht = 0.18 + 0.045 * (pk % 2)
            draw_potted_plant(
                ctx,
                cam,
                np.array([xc + float(rng.uniform(-0.03, 0.03)), float(py), zt]),
                rad=rad,
                height=ht,
                pot_style=style,
                plant_kind=tier_kinds[(pk + idx_t) % len(tier_kinds)],
                seed=5100 + idx_t * 100 + pk * 17,
                fog_rgb=fog_rgb,
            )


def draw_terracotta_pot(
    ctx: cairo.Context,
    cam: Camera3D,
    base_pos: np.ndarray,
    rad: float,
    height: float,
    style: str = "terracotta",
    fog_rgb: np.ndarray | None = None,
    with_saucer: bool = True,
) -> tuple[float, float, float]:
    """
    Perspective-grounded hand-thrown unglazed terracotta clay pot or glossy speckled
    matcha-glazed ceramic planter with rolled collar rim, weathered patina, saucer,
    and warm sunlit highlight facing upper-left SUN_DIR!
    Returns top rim screen center (u_top, v_top, rx_top_px).
    """
    dist = float(np.linalg.norm(base_pos - cam.eye))
    ub, vb, zb = cam.project(base_pos)
    ut, vt, zt = cam.project(base_pos + np.array([0.0, 0.0, height]))
    if float(zb) < Z_NEAR or float(zt) < Z_NEAR:
        return (0.0, 0.0, 0.0)

    px_b = cam.K[0, 0] / float(zb)
    px_t = cam.K[0, 0] / float(zt)
    rx_b = rad * 0.72 * px_b
    rx_t = rad * px_t
    # True perspective foreshortening ratio for horizontal circles at base_pos.z
    elev_b = abs(float(cam.eye[2] - base_pos[2])) / max(0.6, float(zb))
    elev_t = abs(float(cam.eye[2] - (base_pos[2] + height))) / max(0.6, float(zt))
    ry_b = max(2.0, rx_b * np.clip(elev_b * 1.15, 0.16, 0.46))
    ry_t = max(2.0, rx_t * np.clip(elev_t * 1.15, 0.14, 0.44))

    if style == "terracotta":
        c_lit = [0.78, 0.49, 0.34]      # Warm sunlit clay (#c77d57)
        c_mid = [0.61, 0.38, 0.27]      # Classic unglazed terracotta (#9c6348)
        c_dk = [0.30, 0.17, 0.12]       # Deep umber shadow side
        c_rim = [0.84, 0.57, 0.40]
    elif style == "matcha":
        c_lit = [0.58, 0.71, 0.52]      # Glossy celadon-matcha glaze
        c_mid = [0.34, 0.48, 0.34]
        c_dk = [0.14, 0.22, 0.16]
        c_rim = [0.72, 0.80, 0.62]
    elif style == "cream":
        c_lit = [0.84, 0.81, 0.69]      # Warm hand-glazed cream ceramic
        c_mid = [0.64, 0.61, 0.51]
        c_dk = [0.32, 0.31, 0.25]
        c_rim = [0.90, 0.88, 0.76]
    else:  # aged mossy stone/slate stoneware
        c_lit = [0.68, 0.67, 0.56]
        c_mid = [0.46, 0.48, 0.40]
        c_dk = [0.22, 0.25, 0.21]
        c_rim = [0.76, 0.75, 0.64]

    if fog_rgb is not None:
        c_lit = list(fog_mix(c_lit, dist, fog_rgb))
        c_mid = list(fog_mix(c_mid, dist, fog_rgb))
        c_dk = list(fog_mix(c_dk, dist, fog_rgb))
        c_rim = list(fog_mix(c_rim, dist, fog_rgb))

    # Grounded soft feathered contact AO directly beneath pot base (natural wide horizontal shadow)
    g_ao = cairo.RadialGradient(float(ub) + rx_b * 0.22, float(vb) + ry_b * 0.10, rx_b * 0.18, float(ub) + rx_b * 0.22, float(vb) + ry_b * 0.10, rx_b * 1.58)
    g_ao.add_color_stop_rgba(0.0, 0.06, 0.06, 0.05, 0.46)
    g_ao.add_color_stop_rgba(0.62, 0.06, 0.06, 0.05, 0.18)
    g_ao.add_color_stop_rgba(1.0, 0.06, 0.06, 0.05, 0.0)
    ctx.set_source(g_ao)
    ctx.save()
    ctx.translate(float(ub) + rx_b * 0.14, float(vb) + ry_b * 0.12)
    ctx.scale(1.0, 0.30)
    ctx.arc(0.0, 0.0, rx_b * 1.52, 0.0, math.tau)
    ctx.fill()
    ctx.restore()

    # Optional shallow terracotta drip saucer
    if with_saucer:
        srx, sry = rx_b * 1.24, ry_b * 1.22
        sh = max(2.0, (float(vb) - float(vt)) * 0.11)
        g_s = cairo.LinearGradient(float(ub) - srx, float(vb), float(ub) + srx, float(vb))
        g_s.add_color_stop_rgba(0.0, *c_lit, 0.99)
        g_s.add_color_stop_rgba(0.55, *c_mid, 0.99)
        g_s.add_color_stop_rgba(1.0, *c_dk, 0.99)
        ctx.set_source(g_s)
        ctx.move_to(float(ub) - srx, float(vb) - sh)
        ctx.line_to(float(ub) - srx * 0.88, float(vb))
        ctx.curve_to(float(ub) - srx * 0.5, float(vb) + sry, float(ub) + srx * 0.5, float(vb) + sry, float(ub) + srx * 0.88, float(vb))
        ctx.line_to(float(ub) + srx, float(vb) - sh)
        ctx.close_path()
        ctx.fill()

    # Main tapered pot body
    g_body = cairo.LinearGradient(float(ut) - rx_t, float(vt), float(ut) + rx_t, float(vb))
    g_body.add_color_stop_rgba(0.0, *c_lit, 1.0)
    g_body.add_color_stop_rgba(0.28, *c_lit, 1.0)
    g_body.add_color_stop_rgba(0.58, *c_mid, 1.0)
    g_body.add_color_stop_rgba(0.92, *c_dk, 1.0)
    g_body.add_color_stop_rgba(1.0, *(c_mid[k] * 0.72 for k in range(3)), 1.0)  # subtle warm bounce rim
    ctx.set_source(g_body)
    ctx.move_to(float(ut) - rx_t * 0.94, float(vt))
    ctx.line_to(float(ub) - rx_b, float(vb))
    ctx.curve_to(float(ub) - rx_b * 0.55, float(vb) + ry_b * 1.05, float(ub) + rx_b * 0.55, float(vb) + ry_b * 1.05, float(ub) + rx_b, float(vb))
    ctx.line_to(float(ut) + rx_t * 0.94, float(vt))
    ctx.close_path()
    ctx.fill()

    # Subtle weathered mineral efflorescence / mossy patina near pot base
    g_pat = cairo.LinearGradient(float(ub), float(vb) - (float(vb) - float(vt)) * 0.35, float(ub), float(vb) + ry_b)
    g_pat.add_color_stop_rgba(0.0, 0.24, 0.34, 0.20, 0.0)
    g_pat.add_color_stop_rgba(1.0, 0.22, 0.32, 0.18, 0.34)
    ctx.set_source(g_pat)
    ctx.move_to(float(ut) - rx_t * 0.94, float(vt))
    ctx.line_to(float(ub) - rx_b, float(vb))
    ctx.curve_to(float(ub) - rx_b * 0.55, float(vb) + ry_b, float(ub) + rx_b * 0.55, float(vb) + ry_b, float(ub) + rx_b, float(vb))
    ctx.line_to(float(ut) + rx_t * 0.94, float(vt))
    ctx.close_path()
    ctx.fill()

    # Thickened rolled terracotta collar lip
    collar_h = max(3.0, (float(vb) - float(vt)) * 0.21)
    # Shadow tucked cleanly under collar lip
    ctx.set_source_rgba(0.12, 0.08, 0.06, 0.38)
    ctx.set_line_width(max(1.2, collar_h * 0.26))
    ctx.move_to(float(ut) - rx_t * 0.96, float(vt) + collar_h + 1.0)
    ctx.curve_to(float(ut) - rx_t * 0.45, float(vt) + collar_h + ry_t * 0.95, float(ut) + rx_t * 0.45, float(vt) + collar_h + ry_t * 0.95, float(ut) + rx_t * 0.96, float(vt) + collar_h + 1.0)
    ctx.stroke()

    g_col = cairo.LinearGradient(float(ut) - rx_t * 1.06, float(vt), float(ut) + rx_t * 1.06, float(vt))
    g_col.add_color_stop_rgba(0.0, *c_rim, 1.0)
    g_col.add_color_stop_rgba(0.32, *c_lit, 1.0)
    g_col.add_color_stop_rgba(0.66, *c_mid, 1.0)
    g_col.add_color_stop_rgba(1.0, *c_dk, 1.0)
    ctx.set_source(g_col)
    ctx.move_to(float(ut) - rx_t * 1.06, float(vt))
    ctx.line_to(float(ut) - rx_t * 1.03, float(vt) + collar_h)
    ctx.curve_to(float(ut) - rx_t * 0.50, float(vt) + collar_h + ry_t * 1.02, float(ut) + rx_t * 0.50, float(vt) + collar_h + ry_t * 1.02, float(ut) + rx_t * 1.03, float(vt) + collar_h)
    ctx.line_to(float(ut) + rx_t * 1.06, float(vt))
    ctx.curve_to(float(ut) + rx_t * 0.55, float(vt) - ry_t * 0.96, float(ut) - rx_t * 0.55, float(vt) - ry_t * 0.96, float(ut) - rx_t * 1.06, float(vt))
    ctx.close_path()
    ctx.fill()

    # Glossy ceramic specular strip for matcha-glazed planters
    if style == "matcha":
        ctx.set_source_rgba(0.92, 0.96, 0.84, 0.36)
        ctx.set_line_width(max(1.5, rx_t * 0.14))
        ctx.move_to(float(ut) - rx_t * 0.56, float(vt) + collar_h * 0.6)
        ctx.line_to(float(ub) - rx_b * 0.56, float(vb) - ry_b * 0.4)
        ctx.stroke()

    # Top ellipse opening: dark moist potting soil if planted, warm shaded hollow clay if empty spare pot!
    ctx.save()
    ctx.translate(float(ut), float(vt))
    ctx.scale(rx_t * 0.90, ry_t * 0.84)
    if with_saucer:
        ctx.set_source_rgba(0.14, 0.10, 0.07, 1.0)
    else:
        g_in = cairo.LinearGradient(-1.0, 0.0, 1.0, 0.0)
        g_in.add_color_stop_rgba(0.0, *(c_dk[k] * 0.82 for k in range(3)), 1.0)
        g_in.add_color_stop_rgba(0.65, *(c_mid[k] * 0.65 for k in range(3)), 1.0)
        g_in.add_color_stop_rgba(1.0, *(c_lit[k] * 0.72 for k in range(3)), 1.0)
        ctx.set_source(g_in)
    ctx.arc(0.0, 0.0, 1.0, 0.0, math.tau)
    ctx.fill()
    ctx.restore()

    # Sunlit top rim lip highlight
    ctx.save()
    ctx.translate(float(ut), float(vt))
    ctx.scale(rx_t * 1.01, ry_t * 0.96)
    ctx.set_source_rgba(*c_rim, 0.72)
    ctx.set_line_width(max(0.06, 1.5 / max(8.0, rx_t)))
    ctx.arc(0.0, 0.0, 1.0, math.pi * 0.85, math.pi * 1.75)
    ctx.stroke()
    ctx.restore()

    return (float(ut), float(vt), float(rx_t))


def draw_potted_plant(
    ctx: cairo.Context,
    cam: Camera3D,
    base_pos: np.ndarray,
    rad: float,
    height: float,
    pot_style: str,
    plant_kind: str,
    seed: int,
    fog_rgb: np.ndarray,
) -> None:
    """Perspective-grounded terracotta/matcha pot filled with organic cubic-Bezier botanical foliage."""
    srng = np.random.default_rng(seed)
    dist = float(np.linalg.norm(base_pos - cam.eye))
    ut, vt, rx = draw_terracotta_pot(ctx, cam, base_pos, rad, height, style=pot_style, fog_rgb=fog_rgb)
    if rx < 2.0:
        return

    c_dark = fog_mix([0.085, 0.205, 0.115], dist, fog_rgb)
    c_mid = fog_mix([0.220, 0.395, 0.195], dist, fog_rgb)
    c_lit = fog_mix([0.620, 0.770, 0.375], dist, fog_rgb)
    c_vein = fog_mix([0.820, 0.890, 0.560], dist, fog_rgb)

    if plant_kind == "fern":
        for k in range(11):
            frac = k / 10.0
            ang = -math.pi * 0.95 + frac * math.pi * 0.90 + float(srng.uniform(-0.10, 0.10))
            flen = rx * float(srng.uniform(2.0, 3.4))
            draw_fern_frond_bezier(
                ctx,
                ut + (frac - 0.5) * rx * 1.1,
                vt - rx * 0.05,
                flen,
                ang,
                droop=flen * float(srng.uniform(0.32, 0.62)),
                dark_c=c_dark,
                mid_c=c_mid,
                lit_c=c_lit,
                rng=srng,
                alpha=0.98,
                n_pairs=13,
            )
    elif plant_kind == "monstera":
        # Iconic fenestrated Monstera deliciosa stems & sculpted curved leaves
        stems = [
            (-0.9, -2.6, 1.55, -0.45, 0.78),
            (0.2, -3.2, 1.72, -0.10, 0.72),
            (1.2, -2.5, 1.50, 0.42, 0.52),
            (-1.5, -1.6, 1.42, -0.82, 0.85),
            (1.6, -1.5, 1.38, 0.78, 0.38),
            (-0.4, -1.9, 1.60, -0.25, 0.65),
            (0.7, -1.7, 1.48, 0.25, 0.48),
        ]
        for sx_f, sy_f, lsc, lang, bl in stems:
            ex = ut + sx_f * rx * 1.25
            ey = vt + sy_f * rx * 1.05
            ctx.set_source_rgba(*c_dark, 0.95)
            ctx.set_line_width(max(1.5, rx * 0.10))
            ctx.move_to(ut + sx_f * rx * 0.22, vt)
            ctx.curve_to(ut + sx_f * rx * 0.5, vt + sy_f * rx * 0.45, ex - sx_f * rx * 0.2, ey + rx * 0.4, ex, ey)
            ctx.stroke()
            draw_monstera_leaf_bezier(
                ctx,
                ex,
                ey,
                size=rx * lsc * 1.25,
                ang=lang + math.pi + float(srng.uniform(-0.14, 0.14)),
                dark_c=c_dark,
                mid_c=c_mid,
                lit_c=c_lit,
                vein_c=c_vein,
                alpha=0.98,
                backlit=bl,
            )
    elif plant_kind == "trailing":
        # Cascading glossy pothos / trailing ivy spilling generously down the side of the pot!
        for k in range(8):
            frac = k / 7.0
            sx = ut + (frac - 0.5) * rx * 1.5
            sy = vt
            len_t = rx * float(srng.uniform(2.2, 3.8))
            ang = math.pi * 0.5 + (frac - 0.5) * 1.05
            ex = sx + math.cos(ang) * len_t
            ey = sy + math.sin(ang) * len_t
            mx = 0.5 * (sx + ex) + float(srng.uniform(-rx * 0.4, rx * 0.4))
            my = 0.5 * (sy + ey)
            ctx.set_source_rgba(*c_dark, 0.90)
            ctx.set_line_width(max(1.0, rx * 0.055))
            ctx.move_to(sx, sy)
            ctx.curve_to(mx, sy + len_t * 0.25, mx, ey - len_t * 0.25, ex, ey)
            ctx.stroke()
            for t_l in np.linspace(0.15, 0.98, 6):
                lx = (1 - t_l) * sx + t_l * ex + (mx - 0.5 * (sx + ex)) * 2 * t_l * (1 - t_l)
                ly = (1 - t_l) * sy + t_l * ey
                col = c_lit if (lx < ut + rx * 0.2 and srng.random() > 0.35) else (c_mid if srng.random() > 0.3 else c_dark)
                draw_ivy_leaf_bezier(ctx, lx, ly, rx * float(srng.uniform(0.45, 0.72)), ang - math.pi * 0.5 + float(srng.uniform(-0.5, 0.5)), col, c_vein)
    elif plant_kind == "sansevieria":
        # Architectural upright variegated snake-plant sword blades with bright chartreuse margins!
        n_blades = 9
        for bk in range(n_blades):
            frac = bk / (n_blades - 1)
            bx = ut + (frac - 0.5) * rx * 1.25
            bh = rx * float(srng.uniform(2.4, 4.1)) * (0.78 + 0.36 * math.sin(frac * math.pi))
            bw = rx * float(srng.uniform(0.22, 0.34))
            lean = (frac - 0.5) * rx * 0.75 + float(srng.uniform(-rx * 0.18, rx * 0.18))
            tip_x, tip_y = bx + lean, vt - bh
            c_blade = c_mid if bk % 2 == 0 else c_dark
            # Outer sunlit chartreuse-gold leaf margin
            ctx.set_source_rgba(*c_lit, 0.96)
            ctx.move_to(bx - bw * 0.72, vt + rx * 0.04)
            ctx.curve_to(bx - bw * 0.95 + lean * 0.35, vt - bh * 0.48, tip_x - bw * 0.40, tip_y + bh * 0.18, tip_x, tip_y)
            ctx.curve_to(tip_x + bw * 0.40, tip_y + bh * 0.18, bx + bw * 0.95 + lean * 0.35, vt - bh * 0.48, bx + bw * 0.72, vt + rx * 0.04)
            ctx.close_path()
            ctx.fill()
            # Rich dark-emerald striped inner blade core
            ctx.set_source_rgba(*c_blade, 0.96)
            ctx.move_to(bx - bw * 0.48, vt + rx * 0.04)
            ctx.curve_to(bx - bw * 0.64 + lean * 0.35, vt - bh * 0.48, tip_x - bw * 0.24, tip_y + bh * 0.20, tip_x, tip_y + bh * 0.05)
            ctx.curve_to(tip_x + bw * 0.24, tip_y + bh * 0.20, bx + bw * 0.64 + lean * 0.35, vt - bh * 0.48, bx + bw * 0.48, vt + rx * 0.04)
            ctx.close_path()
            ctx.fill()
    elif plant_kind == "calathea":
        # Broad painterly striped oval conservatory calathea / prayer-plant foliage cluster!
        stems_c = [
            (-0.85, -1.75, 1.05, -0.58, 0.78),
            (0.05, -2.15, 1.18, -0.06, 0.70),
            (0.92, -1.72, 1.02, 0.52, 0.52),
            (-1.25, -1.10, 0.96, -0.96, 0.84),
            (1.24, -1.05, 0.92, 0.92, 0.42),
            (-0.42, -1.35, 1.08, -0.32, 0.64),
            (0.46, -1.30, 1.04, 0.30, 0.48),
        ]
        for sx_f, sy_f, lsc, lang, bl in stems_c:
            ex = ut + sx_f * rx * 1.05
            ey = vt + sy_f * rx * 0.95
            ctx.set_source_rgba(*c_dark, 0.92)
            ctx.set_line_width(max(1.2, rx * 0.075))
            ctx.move_to(ut + sx_f * rx * 0.20, vt)
            ctx.curve_to(ut + sx_f * rx * 0.45, vt + sy_f * rx * 0.40, ex - sx_f * rx * 0.15, ey + rx * 0.35, ex, ey)
            ctx.stroke()
            llen = rx * lsc * 1.18
            lwid = llen * 0.52
            ctx.save()
            ctx.translate(ex, ey)
            ctx.rotate(lang)
            g_cal = cairo.LinearGradient(-lwid, -llen * 0.4, lwid, llen * 0.4)
            g_cal.add_color_stop_rgba(0.0, *(c_mid[j] * (1.0 - bl * 0.55) + c_lit[j] * (bl * 0.55) for j in range(3)), 0.98)
            g_cal.add_color_stop_rgba(0.55, *c_mid, 0.98)
            g_cal.add_color_stop_rgba(1.0, *c_dark, 0.98)
            ctx.set_source(g_cal)
            ctx.move_to(0.0, llen * 0.42)
            ctx.curve_to(-lwid, llen * 0.18, -lwid * 0.96, -llen * 0.34, 0.0, -llen * 0.64)
            ctx.curve_to(lwid * 0.96, -llen * 0.34, lwid, llen * 0.18, 0.0, llen * 0.42)
            ctx.close_path()
            ctx.fill()
            # Delicate pale-sage feather-vein markings across calathea blade
            ctx.set_source_rgba(*c_vein, 0.58)
            ctx.set_line_width(max(0.7, llen * 0.038))
            ctx.move_to(0.0, llen * 0.36)
            ctx.line_to(0.0, -llen * 0.54)
            ctx.stroke()
            for v_f in (-0.28, -0.06, 0.14):
                vy = llen * v_f
                ctx.move_to(0.0, vy)
                ctx.line_to(-lwid * 0.58, vy - llen * 0.14)
                ctx.move_to(0.0, vy)
                ctx.line_to(lwid * 0.58, vy - llen * 0.14)
                ctx.stroke()
            ctx.restore()
    elif plant_kind == "succulent":
        # Wide sculpted fleshy echeveria rosette splayed radially outward from the pot center!
        for ring_i, (n_pet, r_base, p_len, p_wid, col_p) in enumerate((
            (11, 0.24, 0.74, 0.34, c_dark),
            (9,  0.14, 0.54, 0.28, c_mid),
            (7,  0.05, 0.34, 0.22, c_lit),
        )):
            for pk in range(n_pet):
                pa = (pk / n_pet) * math.tau + ring_i * 0.38 + float(srng.uniform(-0.07, 0.07))
                dx_p = math.cos(pa)
                dy_p = math.sin(pa) * 0.44 - 0.18 * (ring_i + 1) * 0.35
                # Petal grows radially OUTWARD from pot center (ut, vt - lift)
                bx = ut + dx_p * rx * r_base
                by = vt - rx * (0.04 + ring_i * 0.08) + math.sin(pa) * rx * r_base * 0.36
                ex = bx + dx_p * rx * p_len
                ey = by + dy_p * rx * p_len
                ln_p = math.hypot(ex - bx, ey - by) + 1e-4
                nx_p, ny_p = -(ey - by) / ln_p, (ex - bx) / ln_p
                pw = rx * p_wid
                ctx.set_source_rgba(*col_p, 0.97)
                ctx.move_to(bx, by)
                ctx.curve_to(bx + (ex - bx) * 0.42 + nx_p * pw, by + (ey - by) * 0.42 + ny_p * pw,
                             ex - (ex - bx) * 0.14 + nx_p * pw * 0.65, ey - (ey - by) * 0.14 + ny_p * pw * 0.65,
                             ex, ey)
                ctx.curve_to(ex - (ex - bx) * 0.14 - nx_p * pw * 0.65, ey - (ey - by) * 0.14 - ny_p * pw * 0.65,
                             bx + (ex - bx) * 0.42 - nx_p * pw, by + (ey - by) * 0.42 - ny_p * pw,
                             bx, by)
                ctx.close_path()
                ctx.fill()
        # Cute pale-jade rosette heart bud right at the center
        ctx.set_source_rgba(*c_vein, 0.88)
        ctx.arc(ut, vt - rx * 0.18, max(2.0, rx * 0.16), 0.0, math.tau)
        ctx.fill()
    else:  # Lush organic Studio Ghibli conservatory herb / camellia / maidenhair foliage mound!
        # Soft shaded foliage base volume so soil blends seamlessly into leafy crown
        for kc in range(6):
            ox = float(srng.uniform(-rx * 0.65, rx * 0.65))
            oy = -float(srng.uniform(rx * 0.20, rx * 0.85))
            rc = rx * float(srng.uniform(0.38, 0.62))
            g_b = cairo.RadialGradient(ut + ox, vt + oy, rc * 0.1, ut + ox, vt + oy, rc)
            g_b.add_color_stop_rgba(0.0, *c_dark, 0.86)
            g_b.add_color_stop_rgba(1.0, *c_dark, 0.0)
            ctx.set_source(g_b)
            ctx.arc(ut + ox, vt + oy, rc, 0.0, math.tau)
            ctx.fill()

        # Graceful curving woody stems bearing clustered rounded-oval botanical leaves & tiny cream buds!
        for st_i in range(9):
            frac = st_i / 8.0
            sx0 = ut + (frac - 0.5) * rx * 1.15
            sy0 = vt
            ang_s = -math.pi * 0.88 + frac * math.pi * 0.76 + float(srng.uniform(-0.14, 0.14))
            slen = rx * float(srng.uniform(1.35, 2.35))
            ex = sx0 + math.cos(ang_s) * slen
            ey = sy0 + math.sin(ang_s) * slen
            mx = 0.5 * (sx0 + ex) + float(srng.uniform(-rx * 0.25, rx * 0.25))
            my = 0.5 * (sy0 + ey) - rx * 0.14
            ctx.set_source_rgba(*c_dark, 0.90)
            ctx.set_line_width(max(1.0, rx * 0.06))
            ctx.move_to(sx0, sy0)
            ctx.curve_to(mx, my, mx, my, ex, ey)
            ctx.stroke()

            for lf_t in (0.35, 0.62, 0.86, 1.0):
                lx = (1 - lf_t) ** 2 * sx0 + 2 * (1 - lf_t) * lf_t * mx + lf_t * lf_t * ex
                ly = (1 - lf_t) ** 2 * sy0 + 2 * (1 - lf_t) * lf_t * my + lf_t * lf_t * ey
                for side_l in (-1.0, 1.0):
                    lang = ang_s + side_l * float(srng.uniform(0.32, 0.82))
                    llen = rx * float(srng.uniform(0.42, 0.72)) * (0.75 + 0.30 * lf_t)
                    lwid = llen * 0.44
                    tx = lx + math.cos(lang) * llen
                    ty = ly + math.sin(lang) * llen
                    nx_l, ny_l = -math.sin(lang), math.cos(lang)
                    lit_v = np.clip(0.32 + 0.48 * lf_t - 0.30 * math.cos(lang) + float(srng.uniform(-0.12, 0.16)), 0.0, 1.0)
                    col_l = (
                        tuple(c_mid[j] * (1 - (lit_v - 0.45) / 0.55) + c_lit[j] * ((lit_v - 0.45) / 0.55) for j in range(3))
                        if lit_v > 0.45
                        else tuple(c_dark[j] * (1 - lit_v / 0.45) + c_mid[j] * (lit_v / 0.45) for j in range(3))
                    )
                    ctx.set_source_rgba(*col_l, 0.96)
                    ctx.move_to(lx, ly)
                    ctx.curve_to(lx + math.cos(lang) * llen * 0.45 + nx_l * lwid, ly + math.sin(lang) * llen * 0.45 + ny_l * lwid, tx - math.cos(lang) * llen * 0.2 + nx_l * lwid * 0.5, ty - math.sin(lang) * llen * 0.2 + ny_l * lwid * 0.5, tx, ty)
                    ctx.curve_to(tx - math.cos(lang) * llen * 0.2 - nx_l * lwid * 0.5, ty - math.sin(lang) * llen * 0.2 - ny_l * lwid * 0.5, lx + math.cos(lang) * llen * 0.45 - nx_l * lwid, ly + math.sin(lang) * llen * 0.45 - ny_l * lwid, lx, ly)
                    ctx.close_path()
                    ctx.fill()
            # Delicate tiny warm cream botanical blossom bud at occasional sunlit stem tips
            if st_i in (2, 5, 7):
                ctx.set_source_rgba(0.95, 0.94, 0.80, 0.92)
                ctx.arc(ex, ey - rx * 0.06, max(1.8, rx * 0.13), 0.0, math.tau)
                ctx.fill()


def draw_copper_watering_can(
    ctx: cairo.Context, cam: Camera3D, base_pos: np.ndarray, fog_rgb: np.ndarray
) -> None:
    """
    Hero vintage long-spout burnished copper & brass conservatory watering can
    resting cleanly on the sunlit cedar potting workbench with crisp contact AO,
    tapered riveted body, sweeping gooseneck spout, perforated brass rose sprinkler
    head, and graceful arching rear/top handle!
    """
    dist = float(np.linalg.norm(base_pos - cam.eye))
    ub, vb, zb = cam.project(base_pos)
    if float(zb) < Z_NEAR:
        return
    px_m = cam.K[0, 0] / float(zb)
    bw = 0.125 * px_m
    bh = 0.245 * px_m
    ry = bw * 0.34

    c_hi = fog_mix([0.96, 0.76, 0.48], dist, fog_rgb)
    c_mid = fog_mix([0.74, 0.46, 0.26], dist, fog_rgb)
    c_dk = fog_mix([0.32, 0.18, 0.11], dist, fog_rgb)
    c_brass = fog_mix([0.88, 0.74, 0.38], dist, fog_rgb)

    # Soft grounded contact AO beneath watering can
    g_ao = cairo.RadialGradient(float(ub) + bw * 0.22, float(vb) + ry * 0.2, bw * 0.2, float(ub) + bw * 0.22, float(vb) + ry * 0.2, bw * 1.65)
    g_ao.add_color_stop_rgba(0.0, 0.05, 0.04, 0.03, 0.62)
    g_ao.add_color_stop_rgba(1.0, 0.05, 0.04, 0.03, 0.0)
    ctx.set_source(g_ao)
    ctx.save()
    ctx.translate(float(ub) + bw * 0.16, float(vb) + ry * 0.2)
    ctx.scale(1.0, 0.38)
    ctx.arc(0, 0, bw * 1.6, 0, math.tau)
    ctx.fill()
    ctx.restore()

    # Sweeping arching copper handle (left-rear loop arching over top!)
    ctx.set_source_rgba(*c_dk, 0.98)
    ctx.set_line_width(max(3.0, bw * 0.16))
    ctx.move_to(float(ub) - bw * 0.78, float(vb) - bh * 0.22)
    ctx.curve_to(float(ub) - bw * 1.82, float(vb) - bh * 0.35, float(ub) - bw * 1.68, float(vb) - bh * 1.48, float(ub) - bw * 0.22, float(vb) - bh * 1.24)
    ctx.curve_to(float(ub) + bw * 0.35, float(vb) - bh * 1.15, float(ub) + bw * 0.58, float(vb) - bh * 1.02, float(ub) + bw * 0.55, float(vb) - bh * 0.88)
    ctx.stroke()

    ctx.set_source_rgba(*c_hi, 0.85)
    ctx.set_line_width(max(1.3, bw * 0.06))
    ctx.move_to(float(ub) - bw * 0.82, float(vb) - bh * 0.25)
    ctx.curve_to(float(ub) - bw * 1.78, float(vb) - bh * 0.38, float(ub) - bw * 1.64, float(vb) - bh * 1.44, float(ub) - bw * 0.22, float(vb) - bh * 1.26)
    ctx.stroke()

    # Long slender tapering gooseneck watering spout angling toward the right aisle (+X)
    sp_x0, sp_y0 = float(ub) + bw * 0.65, float(vb) - bh * 0.18
    sp_x1, sp_y1 = float(ub) + bw * 2.55, float(vb) - bh * 1.12
    g_sp = cairo.LinearGradient(sp_x0, sp_y0 - bw * 0.2, sp_x0, sp_y0 + bw * 0.2)
    g_sp.add_color_stop_rgba(0.0, *c_hi, 1.0)
    g_sp.add_color_stop_rgba(0.55, *c_mid, 1.0)
    g_sp.add_color_stop_rgba(1.0, *c_dk, 1.0)
    ctx.set_source(g_sp)
    ctx.move_to(sp_x0, sp_y0 - bw * 0.24)
    ctx.curve_to(sp_x0 + bw * 0.72, sp_y0 - bh * 0.36, sp_x1 - bw * 0.55, sp_y1 + bh * 0.18, sp_x1, sp_y1 - bw * 0.08)
    ctx.line_to(sp_x1 + bw * 0.06, sp_y1 + bw * 0.10)
    ctx.curve_to(sp_x1 - bw * 0.48, sp_y1 + bh * 0.30, sp_x0 + bw * 0.78, sp_y0 - bh * 0.16, sp_x0, sp_y0 + bw * 0.22)
    ctx.close_path()
    ctx.fill()

    # Brass strut brace connecting neck to spout
    ctx.set_source_rgba(*c_brass, 0.95)
    ctx.set_line_width(max(1.5, bw * 0.07))
    ctx.move_to(float(ub) + bw * 0.64, float(vb) - bh * 0.85)
    ctx.line_to(float(ub) + bw * 1.62, float(vb) - bh * 0.74)
    ctx.stroke()

    # Oval brass sprinkler rose head at tip of spout (with fine metallic rim & sprinkler face!)
    ctx.save()
    ctx.translate(sp_x1 + bw * 0.14, sp_y1 - bw * 0.04)
    ctx.rotate(-0.48)
    g_rose = cairo.LinearGradient(-bw * 0.22, -bw * 0.34, bw * 0.22, bw * 0.34)
    g_rose.add_color_stop_rgba(0.0, *c_hi, 1.0)
    g_rose.add_color_stop_rgba(0.45, *c_brass, 1.0)
    g_rose.add_color_stop_rgba(1.0, *c_dk, 1.0)
    ctx.set_source(g_rose)
    ctx.scale(bw * 0.24, bw * 0.38)
    ctx.arc(0.0, 0.0, 1.0, 0.0, math.tau)
    ctx.fill()
    # Shaded perforated rose sprinkler face rim
    ctx.set_source_rgba(*c_dk, 0.65)
    ctx.set_line_width(0.10)
    ctx.arc(0.18, 0.0, 0.80, 0.0, math.tau)
    ctx.stroke()
    ctx.restore()

    # Main tapered cylindrical burnished copper body catching direct Bay 2 morning sunbeam!
    g_c = cairo.LinearGradient(float(ub) - bw, float(vb), float(ub) + bw, float(vb))
    g_c.add_color_stop_rgba(0.0, *c_mid, 1.0)
    g_c.add_color_stop_rgba(0.20, *c_hi, 1.0)
    g_c.add_color_stop_rgba(0.35, 1.00, 0.96, 0.78, 1.0)  # Blazing sunlit copper morning specular strip!
    g_c.add_color_stop_rgba(0.56, *c_mid, 1.0)
    g_c.add_color_stop_rgba(0.88, *c_dk, 1.0)
    g_c.add_color_stop_rgba(1.0, *c_mid, 1.0)
    ctx.set_source(g_c)
    ctx.move_to(float(ub) - bw * 0.84, float(vb) - bh)
    ctx.line_to(float(ub) - bw, float(vb))
    ctx.curve_to(float(ub) - bw * 0.55, float(vb) + ry * 1.15, float(ub) + bw * 0.55, float(vb) + ry * 1.15, float(ub) + bw, float(vb))
    ctx.line_to(float(ub) + bw * 0.84, float(vb) - bh)
    ctx.close_path()
    ctx.fill()

    # Warm golden morning sun-hit shoulder specular glint facing upper-left -SUN_DIR!
    sh_x, sh_y = float(ub) - bw * 0.30, float(vb) - bh * 0.76
    g_sh = cairo.RadialGradient(sh_x, sh_y, bw * 0.04, sh_x, sh_y, bw * 0.58)
    g_sh.add_color_stop_rgba(0.0, 1.00, 0.98, 0.84, 0.68)
    g_sh.add_color_stop_rgba(0.45, 1.00, 0.86, 0.56, 0.28)
    g_sh.add_color_stop_rgba(1.0, 1.00, 0.82, 0.48, 0.0)
    ctx.set_source(g_sh)
    ctx.arc(sh_x, sh_y, bw * 0.58, 0.0, math.tau)
    ctx.fill()

    # Crisp hot glint on the sunlit brass sprinkler rose rim
    g_rg = cairo.RadialGradient(sp_x1 + bw * 0.06, sp_y1 - bw * 0.16, bw * 0.02, sp_x1 + bw * 0.06, sp_y1 - bw * 0.16, bw * 0.30)
    g_rg.add_color_stop_rgba(0.0, 1.00, 0.98, 0.84, 0.72)
    g_rg.add_color_stop_rgba(1.0, 1.00, 0.90, 0.56, 0.0)
    ctx.set_source(g_rg)
    ctx.arc(sp_x1 + bw * 0.06, sp_y1 - bw * 0.16, bw * 0.30, 0.0, math.tau)
    ctx.fill()

    # Partial brass splash-guard top collar & dark water fill opening
    ctx.save()
    ctx.translate(float(ub), float(vb) - bh)
    ctx.scale(bw * 0.84, ry * 0.84)
    ctx.set_source_rgba(*c_brass, 1.0)
    ctx.arc(0.0, 0.0, 1.0, 0.0, math.tau)
    ctx.fill()
    ctx.set_source_rgba(0.14, 0.09, 0.06, 0.95)
    ctx.arc(-0.22, 0.0, 0.58, 0.0, math.tau)
    ctx.fill()
    ctx.restore()

    # Brass rim band near base
    ctx.set_source_rgba(*c_brass, 0.86)
    ctx.set_line_width(max(1.5, bw * 0.08))
    ctx.move_to(float(ub) - bw * 0.98, float(vb) - bh * 0.14)
    ctx.curve_to(float(ub) - bw * 0.45, float(vb) - bh * 0.14 + ry * 0.9, float(ub) + bw * 0.45, float(vb) - bh * 0.14 + ry * 0.9, float(ub) + bw * 0.98, float(vb) - bh * 0.14)
    ctx.stroke()


def draw_glass_propagation_jar(
    ctx: cairo.Context, cam: Camera3D, base_pos: np.ndarray, seed: int, fog_rgb: np.ndarray
) -> None:
    """Apothecary clear glass propagation jar filled with sunlit water, roots, caustic glow & pothos cutting."""
    srng = np.random.default_rng(seed)
    dist = float(np.linalg.norm(base_pos - cam.eye))
    ub, vb, zb = cam.project(base_pos)
    if float(zb) < Z_NEAR:
        return
    px_m = cam.K[0, 0] / float(zb)
    rw = 0.056 * px_m
    rh = 0.155 * px_m
    ry = rw * 0.34

    # Bright warm refracted sunlight caustic lens glow projected rightward (+X along SUN_DIR) onto cedar bench!
    g_cau = cairo.RadialGradient(float(ub) + rw * 1.50, float(vb) + ry * 0.22, rw * 0.12, float(ub) + rw * 1.50, float(vb) + ry * 0.22, rw * 1.65)
    g_cau.add_color_stop_rgba(0.0, 1.00, 0.97, 0.74, 0.56)
    g_cau.add_color_stop_rgba(0.42, 0.98, 0.94, 0.66, 0.30)
    g_cau.add_color_stop_rgba(1.0, 0.98, 0.94, 0.66, 0.0)
    ctx.set_source(g_cau)
    ctx.save()
    ctx.translate(float(ub) + rw * 1.50, float(vb) + ry * 0.22)
    ctx.scale(1.35, 0.44)
    ctx.arc(0, 0, rw * 1.45, 0, math.tau)
    ctx.fill()
    ctx.restore()
    # Concentrated golden focal caustic ribbon right inside the caustic pool
    ctx.save()
    ctx.translate(float(ub) + rw * 1.62, float(vb) + ry * 0.18)
    ctx.scale(1.4, 0.32)
    ctx.set_source_rgba(1.00, 0.98, 0.80, 0.44)
    ctx.arc(0, 0, rw * 0.55, 0, math.tau)
    ctx.fill()
    ctx.restore()

    # Translucent glass body with luminous sage-cream refracted water inside
    g_w = cairo.LinearGradient(float(ub) - rw, float(vb) - rh * 0.68, float(ub) + rw, float(vb))
    g_w.add_color_stop_rgba(0.0, 0.86, 0.94, 0.80, 0.54)
    g_w.add_color_stop_rgba(0.5, 0.56, 0.72, 0.58, 0.44)
    g_w.add_color_stop_rgba(1.0, 0.28, 0.42, 0.34, 0.56)
    ctx.set_source(g_w)
    ctx.move_to(float(ub) - rw, float(vb) - rh * 0.68)
    ctx.line_to(float(ub) - rw, float(vb))
    ctx.curve_to(float(ub) - rw * 0.5, float(vb) + ry, float(ub) + rw * 0.5, float(vb) + ry, float(ub) + rw, float(vb))
    ctx.line_to(float(ub) + rw, float(vb) - rh * 0.68)
    ctx.close_path()
    ctx.fill()

    # Luminous curved water meniscus ellipse at water surface
    ctx.save()
    ctx.translate(float(ub), float(vb) - rh * 0.68)
    ctx.scale(rw * 0.96, ry * 0.82)
    ctx.set_source_rgba(0.95, 0.98, 0.88, 0.58)
    ctx.set_line_width(max(0.07, 1.2 / max(8.0, rw)))
    ctx.arc(0.0, 0.0, 1.0, 0.0, math.tau)
    ctx.stroke()
    ctx.restore()

    # Delicate submerged cutting stem & wispy translucent roots
    ctx.set_source_rgba(0.24, 0.46, 0.22, 0.88)
    ctx.set_line_width(max(1.2, rw * 0.12))
    ctx.move_to(float(ub) - rw * 0.15, float(vb) - rh * 0.16)
    ctx.curve_to(float(ub), float(vb) - rh * 0.72, float(ub) + rw * 0.10, float(vb) - rh * 1.15, float(ub) + rw * 0.22, float(vb) - rh * 1.50)
    ctx.stroke()
    ctx.set_source_rgba(0.90, 0.88, 0.74, 0.72)
    ctx.set_line_width(max(0.75, rw * 0.05))
    for _ in range(4):
        ctx.move_to(float(ub) - rw * 0.15, float(vb) - rh * 0.24)
        ctx.curve_to(
            float(ub) + float(srng.uniform(-rw * 0.6, rw * 0.5)),
            float(vb) - rh * 0.12,
            float(ub) + float(srng.uniform(-rw * 0.7, rw * 0.6)),
            float(vb) - rh * 0.04,
            float(ub) + float(srng.uniform(-rw * 0.6, rw * 0.6)),
            float(vb) + ry * 0.2,
        )
        ctx.stroke()

    # Glass jar silhouette + rolled mouth rim + bright sunlit vertical glass reflection highlight
    ctx.set_source_rgba(0.92, 0.96, 0.86, 0.66)
    ctx.set_line_width(max(1.0, rw * 0.08))
    ctx.move_to(float(ub) - rw, float(vb))
    ctx.line_to(float(ub) - rw, float(vb) - rh * 0.74)
    ctx.curve_to(float(ub) - rw, float(vb) - rh * 0.90, float(ub) - rw * 0.62, float(vb) - rh * 0.92, float(ub) - rw * 0.62, float(vb) - rh)
    ctx.line_to(float(ub) + rw * 0.62, float(vb) - rh)
    ctx.curve_to(float(ub) + rw * 0.62, float(vb) - rh * 0.92, float(ub) + rw, float(vb) - rh * 0.90, float(ub) + rw, float(vb) - rh * 0.74)
    ctx.line_to(float(ub) + rw, float(vb))
    ctx.stroke()

    ctx.save()
    ctx.translate(float(ub), float(vb) - rh)
    ctx.scale(rw * 0.64, ry * 0.58)
    ctx.set_source_rgba(0.96, 0.98, 0.90, 0.78)
    ctx.set_line_width(max(0.08, 1.3 / max(8.0, rw)))
    ctx.arc(0.0, 0.0, 1.0, 0.0, math.tau)
    ctx.stroke()
    ctx.restore()

    ctx.set_source_rgba(0.99, 0.99, 0.92, 0.82)
    ctx.set_line_width(max(1.2, rw * 0.14))
    ctx.move_to(float(ub) - rw * 0.68, float(vb) - rh * 0.70)
    ctx.line_to(float(ub) - rw * 0.68, float(vb) - rh * 0.10)
    ctx.stroke()

    # Soft heart-shaped glowing pothos cutting leaves emerging gracefully from the glass jar!
    c_lt = fog_mix([0.52, 0.72, 0.30], dist, fog_rgb)
    c_md = fog_mix([0.24, 0.45, 0.20], dist, fog_rgb)
    c_vn = fog_mix([0.86, 0.92, 0.58], dist, fog_rgb)
    for lx_j, ly_j, lsz, lang, col_j in (
        (float(ub) + rw * 0.22, float(vb) - rh * 1.48, rw * 1.35, -2.55, c_lt),
        (float(ub) - rw * 0.08, float(vb) - rh * 1.24, rw * 1.15, 2.38, c_md),
    ):
        ctx.save()
        ctx.translate(lx_j, ly_j)
        ctx.rotate(lang)
        ctx.set_source_rgba(*col_j, 0.97)
        ctx.move_to(0.0, 0.0)
        ctx.curve_to(lsz * 0.58, -lsz * 0.12, lsz * 0.62, lsz * 0.56, 0.0, lsz * 1.06)
        ctx.curve_to(-lsz * 0.62, lsz * 0.56, -lsz * 0.58, -lsz * 0.12, 0.0, 0.0)
        ctx.close_path()
        ctx.fill()
        ctx.set_source_rgba(*c_vn, 0.65)
        ctx.set_line_width(max(0.6, lsz * 0.05))
        ctx.move_to(0.0, lsz * 0.08)
        ctx.line_to(0.0, lsz * 0.88)
        ctx.stroke()
        ctx.restore()


def draw_wooden_seed_tray(
    ctx: cairo.Context, cam: Camera3D, x0: float, y0: float, w_m: float, l_m: float, fog_rgb: np.ndarray
) -> None:
    """Weathered wooden conservatory seed-starting tray with rows of glowing matcha seedlings."""
    h_t = 0.068
    dist = float(np.linalg.norm(np.array([x0 + 0.5 * w_m, y0 + 0.5 * l_m, BENCH_Z]) - cam.eye))
    c_wd = fog_mix([0.46, 0.35, 0.24], dist, fog_rgb)
    c_wd_d = fog_mix([0.24, 0.17, 0.11], dist, fog_rgb)
    c_soil = fog_mix([0.14, 0.11, 0.08], dist, fog_rgb)
    c_spr = fog_mix([0.62, 0.78, 0.32], dist, fog_rgb)

    p00 = np.array([x0, y0, BENCH_Z])
    p10 = np.array([x0 + w_m, y0, BENCH_Z])
    p11 = np.array([x0 + w_m, y0 + l_m, BENCH_Z])
    p01 = np.array([x0, y0 + l_m, BENCH_Z])
    q00 = p00 + np.array([0, 0, h_t])
    q10 = p10 + np.array([0, 0, h_t])
    q11 = p11 + np.array([0, 0, h_t])
    q01 = p01 + np.array([0, 0, h_t])

    # Top dark soil bed inside wooden rim
    fill_3d_quad(ctx, cam, [q00, q10, q11, q01], (*c_soil, 0.99))
    # Front (-Y) and aisle (+X) weathered wood sides
    fill_3d_quad(ctx, cam, [p00, p10, q10, q00], (*c_wd, 1.0), (*c_wd_d, 1.0))
    fill_3d_quad(ctx, cam, [p10, p11, q11, q10], (*c_wd_d, 1.0))
    # Wooden tray top rim lip
    for a, b in ((q00, q10), (q10, q11), (q11, q01), (q01, q00)):
        stroke_3d_segment(ctx, cam, a, b, 0.016, (*c_wd, 0.98))

    # Rows of bright delicate tender green seedlings catching the morning sunbeams!
    for rx in np.linspace(x0 + 0.06, x0 + w_m - 0.06, 4):
        for ry in np.linspace(y0 + 0.08, y0 + l_m - 0.08, 6):
            us, vs, zs = cam.project(np.array([float(rx), float(ry), BENCH_Z + h_t]))
            if float(zs) > Z_NEAR:
                sc = 0.022 * cam.K[0, 0] / float(zs)
                ctx.set_source_rgba(*c_spr, 0.96)
                ctx.set_line_width(max(0.8, sc * 0.14))
                ctx.move_to(float(us), float(vs))
                ctx.line_to(float(us), float(vs) - sc * 1.1)
                ctx.stroke()
                # Twin cotyledon leaves
                for side in (-1.0, 1.0):
                    ctx.save()
                    ctx.translate(float(us) + side * sc * 0.48, float(vs) - sc * 1.15)
                    ctx.rotate(side * 0.45)
                    ctx.scale(sc * 0.52, sc * 0.28)
                    ctx.arc(0.0, 0.0, 1.0, 0.0, math.tau)
                    ctx.fill()
                    ctx.restore()


def draw_garden_trowel_and_shears(ctx: cairo.Context, cam: Camera3D, fog_rgb: np.ndarray) -> None:
    """Turned-wood hand trowel & forged steel pruning shears resting casually on the bench."""
    # Hand trowel at (-1.52, 2.86, BENCH_Z + 0.006)
    p_h0 = np.array([-1.72, 2.78, BENCH_Z + 0.012])
    p_h1 = np.array([-1.58, 2.85, BENCH_Z + 0.012])
    p_b1 = np.array([-1.39, 2.94, BENCH_Z + 0.008])
    stroke_3d_segment(ctx, cam, p_h0, p_h1, 0.032, (0.48, 0.30, 0.17, 0.99))
    stroke_3d_segment(ctx, cam, p_h1, p_b1, 0.014, (0.52, 0.56, 0.54, 0.99))
    # Trowel scoop blade
    blade = [
        np.array([-1.54, 2.84, BENCH_Z + 0.008]),
        np.array([-1.46, 2.82, BENCH_Z + 0.008]),
        np.array([-1.36, 2.96, BENCH_Z + 0.006]),
        np.array([-1.46, 2.94, BENCH_Z + 0.008]),
    ]
    fill_3d_quad(ctx, cam, blade, (0.64, 0.68, 0.65, 0.98), (0.28, 0.32, 0.30, 0.98))

    # Forged garden shears resting beside small pots at (-1.45, 4.18)
    s_piv = np.array([-1.46, 4.18, BENCH_Z + 0.008])
    stroke_3d_segment(ctx, cam, s_piv + np.array([-0.10, -0.04, 0]), s_piv + np.array([0.08, 0.03, 0]), 0.015, (0.24, 0.34, 0.26, 0.98))
    stroke_3d_segment(ctx, cam, s_piv + np.array([-0.10, 0.04, 0]), s_piv + np.array([0.08, -0.02, 0]), 0.015, (0.62, 0.66, 0.62, 0.98))


# =============================================================================
# Stage 3 Band 3 — Lush Right Raised Stone Planter Bed Tropical Foliage & Palms
# =============================================================================

def draw_right_planter_botanicals(
    ctx: cairo.Context, cam: Camera3D, rng: np.random.Generator, fog_rgb: np.ndarray, mirrored: bool = False
) -> None:
    """
    Lush tropical conservatory jungle rising from the right raised mossy stone planter bed
    (x in [PLANTER_X0, PLANTER_X1], z = PLANTER_Z):
      - Graceful misty parlor/kentia palms framing the rear doorway
      - Sweeping broad sunlit banana & bird-of-paradise leaves with translucent chartreuse midribs
      - Fenestrated monstera deliciosa clusters spilling slightly over the mossy stone curb
      - Arching cascading Boston & sword fern fronds glowing emerald-gold!
    """
    alpha_m = 0.52 if mirrored else 0.99

    # Draw back-to-front along Y (13.4 m -> 2.8 m) for authentic painter's depth overlap!
    # 1. Rear & mid-right arching Kentia / Parlor Palms (tucked cleanly toward right glass wall!)
    palm_specs = [
        (np.array([2.10, 13.1, PLANTER_Z]), 2.40, -0.45, 8101),
        (np.array([2.18, 10.6, PLANTER_Z]), 2.55, -0.52, 8102),
        (np.array([2.15, 7.8, PLANTER_Z]), 2.35, -0.48, 8103),
    ]
    for p_base, p_h, lean_x, pseed in palm_specs:
        srng = np.random.default_rng(pseed)
        dist = float(np.linalg.norm(p_base - cam.eye))
        c_dk = fog_mix([0.085, 0.200, 0.115], dist, fog_rgb)
        c_md = fog_mix([0.215, 0.385, 0.195], dist, fog_rgb)
        c_lt = fog_mix([0.580, 0.720, 0.365], dist, fog_rgb)

        p_top = p_base + np.array([lean_x, -0.15, p_h])
        p_b_use = mirror_pts_across_plane([p_base], Z_FLOOR)[0] if mirrored else p_base
        p_t_use = mirror_pts_across_plane([p_top], Z_FLOOR)[0] if mirrored else p_top
        stroke_3d_segment(ctx, cam, p_b_use, p_t_use, 0.068, (*c_dk, alpha_m * 0.92))

        ut, vt, zt = cam.project(p_t_use)
        if float(zt) > Z_NEAR:
            px_m = cam.K[0, 0] / float(zt)
            for fk in range(8):
                ang = -math.pi * 1.05 + (fk / 7.0) * math.pi * 1.25 + float(srng.uniform(-0.10, 0.10))
                if mirrored:
                    ang = -ang
                flen = float(srng.uniform(0.95, 1.55)) * px_m
                draw_fern_frond_bezier(
                    ctx,
                    float(ut),
                    float(vt),
                    flen,
                    ang,
                    droop=flen * float(srng.uniform(0.32, 0.56)),
                    dark_c=c_dk,
                    mid_c=c_md,
                    lit_c=c_lt,
                    rng=srng,
                    alpha=alpha_m,
                    n_pairs=16,
                    mirrored=mirrored,
                )

    # 2. Broad sweeping banana & bird-of-paradise leaves + fern clumps along right planter (back-to-front!)
    clump_ys = (12.2, 10.8, 9.2, 7.6, 6.2, 4.8, 3.4)
    for idx_c, cy in enumerate(clump_ys):
        srng = np.random.default_rng(8400 + idx_c * 53)
        base3 = np.array([1.78 + float(srng.uniform(-0.14, 0.24)), cy, PLANTER_Z + 0.04])
        dist = float(np.linalg.norm(base3 - cam.eye))
        p_use = mirror_pts_across_plane([base3], Z_FLOOR)[0] if mirrored else base3
        ub, vb, zb = cam.project(p_use)
        if float(zb) < Z_NEAR:
            continue
        px_m = cam.K[0, 0] / float(zb)

        c_dk = fog_mix([0.078, 0.195, 0.110], dist, fog_rgb)
        c_md = fog_mix([0.210, 0.390, 0.190], dist, fog_rgb)
        c_lt = fog_mix([0.640, 0.785, 0.370], dist, fog_rgb)
        c_vein = fog_mix([0.840, 0.900, 0.560], dist, fog_rgb)

        # Tall sweeping broad banana / bird-of-paradise blades (angled gracefully inward & up)
        if not mirrored:
            for bk in range(4 if cy < 8.5 else 3):
                ang = -math.pi * 0.55 - float(srng.uniform(0.05, 0.62)) + (bk - 1.5) * 0.24
                blen = float(srng.uniform(0.95, 1.52)) * px_m
                bwid = blen * float(srng.uniform(0.24, 0.34))
                # Petiole stem
                sx = float(ub) + float(srng.uniform(-0.18, 0.18)) * px_m
                sy = float(vb)
                lx = sx + math.cos(ang) * blen * 0.24
                ly = sy + math.sin(ang) * blen * 0.24
                ctx.set_source_rgba(*c_dk, alpha_m * 0.92)
                ctx.set_line_width(max(1.6, 0.024 * px_m))
                ctx.move_to(sx, sy)
                ctx.line_to(lx, ly)
                ctx.stroke()
                draw_banana_leaf_bezier(
                    ctx,
                    lx,
                    ly,
                    length=blen * 0.85,
                    width=bwid,
                    ang=ang,
                    droop=blen * float(srng.uniform(0.18, 0.38)),
                    dark_c=c_dk,
                    mid_c=c_md,
                    lit_c=c_lt,
                    rng=srng,
                    alpha=alpha_m,
                    backlit=float(srng.uniform(0.45, 0.82)),
                )

            # Iconic fenestrated Monstera leaves near mid-foreground planter curb!
            if cy in (6.2, 4.8, 3.4):
                for mk in range(3):
                    mx = float(ub) + float(srng.uniform(-0.34, 0.16)) * px_m
                    my = float(vb) - float(srng.uniform(0.22, 0.68)) * px_m
                    msz = float(srng.uniform(0.28, 0.44)) * px_m
                    mang = math.pi + float(srng.uniform(-0.78, 0.52))
                    ctx.set_source_rgba(*c_dk, 0.94)
                    ctx.set_line_width(max(1.5, 0.016 * px_m))
                    ctx.move_to(float(ub), float(vb))
                    ctx.curve_to(0.5 * (float(ub) + mx), float(vb) - 0.15 * px_m, mx, my + 0.12 * px_m, mx, my)
                    ctx.stroke()
                    draw_monstera_leaf_bezier(ctx, mx, my, msz, mang, c_dk, c_md, c_lt, c_vein, alpha=0.98, backlit=0.68)

        # Cascading sword/Boston ferns spilling delicately over the mossy limestone planter curb!
        for fk in range(5):
            ang = math.pi * 0.82 + float(srng.uniform(-0.38, 0.42))
            if mirrored:
                ang = -ang
            flen = float(srng.uniform(0.52, 0.88)) * px_m
            draw_fern_frond_bezier(
                ctx,
                float(ub) - 0.22 * px_m + float(srng.uniform(-0.10, 0.14)) * px_m,
                float(vb) + (0.04 * px_m if not mirrored else -0.04 * px_m),
                flen,
                ang,
                droop=flen * float(srng.uniform(0.38, 0.68)),
                dark_c=c_dk,
                mid_c=c_md,
                lit_c=c_lt,
                rng=srng,
                alpha=alpha_m,
                n_pairs=13,
                mirrored=mirrored,
            )


# =============================================================================
# Stage 2 & 3 — True 3D Ray-Marched Window-Aperture Volumetric Morning Sunbeams
# =============================================================================

def _ray_aabb_hit(
    eye: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    bmin: tuple[float, float, float],
    bmax: tuple[float, float, float],
) -> np.ndarray:
    """Vectorized ray-AABB nearest positive hit distance (returns 1e6 if missed)."""
    inv_x = np.where(np.abs(dx) > 1e-7, 1.0 / np.where(np.abs(dx) > 1e-7, dx, 1.0), 1e9)
    inv_y = np.where(np.abs(dy) > 1e-7, 1.0 / np.where(np.abs(dy) > 1e-7, dy, 1.0), 1e9)
    inv_z = np.where(np.abs(dz) > 1e-7, 1.0 / np.where(np.abs(dz) > 1e-7, dz, 1.0), 1e9)

    tx0 = (bmin[0] - eye[0]) * inv_x
    tx1 = (bmax[0] - eye[0]) * inv_x
    ty0 = (bmin[1] - eye[1]) * inv_y
    ty1 = (bmax[1] - eye[1]) * inv_y
    tz0 = (bmin[2] - eye[2]) * inv_z
    tz1 = (bmax[2] - eye[2]) * inv_z

    tmin = np.maximum(np.maximum(np.minimum(tx0, tx1), np.minimum(ty0, ty1)), np.minimum(tz0, tz1))
    tmax = np.minimum(np.minimum(np.maximum(tx0, tx1), np.maximum(ty0, ty1)), np.maximum(tz0, tz1))
    hit = (tmax >= np.maximum(tmin, 0.05)) & (tmin > 0.05)
    return np.where(hit, tmin, 1e6)


def _trace_volumetric_aperture(
    wx: np.ndarray,
    wy: np.ndarray,
    wz: np.ndarray,
    basket_centers: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    High-contrast 3D collimated-sunlight aperture evaluator for volumetric god-rays!
    Traces each 3D air sample (wx, wy, wz) backward along -SUN_DIR to the exact
    tall left-side arched glass window bay OR upper-left vaulted glass skylight bay,
    carving crisp, soft-feathered dark shadow corridors for every wrought-iron column,
    climbing ivy vine, arch spandrel, horizontal transom, glazing mullion, eave beam,
    vault purlin, hanging fern basket, mossy kneewall, and opaque cedar potting bench!
    Returns (aperture_trans, ray_dist_from_glass).
    """
    inv_sz = 1.0 / (-SUN_DIR[2])
    # 5-step Newton/fixed-point refinement against elliptical barrel vault z = vault_z(x)
    z_roof = np.full_like(wx, 3.85, dtype=np.float64)
    for _ in range(5):
        t_r = np.maximum(0.0, (z_roof - wz) * inv_sz)
        sx_r = np.clip(wx - SUN_DIR[0] * t_r, X_LEFT, X_RIGHT)
        z_roof = vault_z(sx_r)
    t_r = np.maximum(0.0, (z_roof - wz) * inv_sz)
    sx_r = wx - SUN_DIR[0] * t_r
    sy_r = wy - SUN_DIR[1] * t_r

    # Intersection with tall left arched glass wall (x = X_LEFT)
    t_w = (wx - X_LEFT) / SUN_DIR[0]
    zw = wz - SUN_DIR[2] * t_w
    yw = wy - SUN_DIR[1] * t_w
    hits_left_wall = (t_w > 0.0) & (zw <= Z_SPRING)

    d_ray = np.where(hits_left_wall, t_w, t_r)
    ey = np.where(hits_left_wall, yw, sy_r)
    ex = np.where(hits_left_wall, X_LEFT, sx_r)
    ez = np.where(hits_left_wall, zw, z_roof)

    # 1. Structural bay columns & vaulted wrought-iron ribs at BAY_YS (pitch = 2.10 m)
    #    Bold volumetric shadow gap (widened organically where climbing ivy wraps left columns!)
    bay_phase = np.mod(ey - BAY_YS[0], 2.10)
    dist_rib = np.minimum(bay_phase, 2.10 - bay_phase)
    ivy_col = hits_left_wall & (ey > 3.2) & (ey < 10.7)
    rib_hw = np.where(ivy_col, 0.275 + 0.055 * np.sin(ez * 4.5 + ey * 1.2), 0.185)
    rib_open = np.clip((dist_rib - rib_hw) / 0.085, 0.0, 1.0)

    # 2. Secondary vertical glazing mullions (3 panes per 2.10 m bay -> 0.70 m pitch)
    mull_phase = np.mod(ey - BAY_YS[0], 0.70)
    dist_mull = np.minimum(mull_phase, 0.70 - mull_phase)
    mull_open = 0.03 + 0.97 * np.clip((dist_mull - 0.064) / 0.048, 0.0, 1.0)

    # 3. Tall left arched window opening: solid kneewall below Z_KNEE, semicircular arch spandrel above,
    #    and TWO bold Y-running horizontal transoms (z = 1.68, z = 2.34) that carve unmistakable
    #    diagonal dark shadow corridors straight through the entire conservatory air!
    above_knee = np.clip((zw - (Z_KNEE + 0.08)) / 0.078, 0.0, 1.0)
    bay_u = (bay_phase - 1.05) / 0.90
    arch_z_top = 2.34 + 0.52 * np.sqrt(np.maximum(0.0, 1.0 - np.clip(bay_u, -1.0, 1.0) ** 2))
    under_arch = np.clip((arch_z_top - zw) / 0.082, 0.0, 1.0)
    trans_d = np.minimum(np.abs(zw - 1.68), np.abs(zw - 2.34))
    trans_open = 0.015 + 0.985 * np.clip((trans_d - 0.098) / 0.058, 0.0, 1.0)
    # Sculpted vertical tier emphasis: tall left arched windows pour bold, unmistakable golden shafts
    # right across the climbing ivy, monstera, copper watering can, cedar bench & wet slate floor puddles!
    wall_tier_w = np.where(zw > 2.34, 1.66, np.where(zw > 1.68, 1.56, 1.14))
    wall_open = above_knee * under_arch * trans_open * wall_tier_w

    # 4. Upper-left vaulted glass skylight bays separated by bold longitudinal purlins & solid eave beam!
    purlin_x = (-2.65, -2.22, -1.58, -0.82, -0.34, 0.0)
    min_px = np.full_like(ex, 10.0)
    for px_v in purlin_x:
        min_px = np.minimum(min_px, np.abs(ex - px_v))
    roof_purlin = 0.015 + 0.985 * np.clip((min_px - 0.105) / 0.068, 0.0, 1.0)
    # Solid eave cornice separator right around x = X_LEFT / z = Z_SPRING
    eave_gap = np.clip((ex - (X_LEFT + 0.24)) / 0.16, 0.0, 1.0)
    # Dramatic vaulted skylight shafts cleanly balanced against left-wall arched window shafts!
    left_vault_env = np.clip((0.20 - ex) / 0.62, 0.0, 1.0)
    roof_open = roof_purlin * eave_gap * left_vault_env * 1.10

    aperture = rib_open * mull_open * np.where(hits_left_wall, wall_open, roof_open)

    # 5. Shared outdoor botanical garden canopy dapple (syncs shafts 100% with floor/bench sun pools!)
    dapple = outdoor_sun_dapple(ex, ey, ez)
    # Crisp shaft-vs-foliage-shadow separation so individual window shafts gleam cleanly against dark gaps!
    dapple_sharp = np.power(np.clip((dapple - 0.32) / 0.86, 0.0, 1.35), 1.62)
    aperture *= dapple_sharp

    # 6. Rear gable wall blocks rays from beyond Y_REAR & allow tall near-left arched window bays (y >= 1.95)
    #    to pour balanced, unmistakable diagonal shafts cleanly across the bench, potted plants & wet stone aisle!
    in_nave_y = np.clip((Y_REAR - 0.35 - ey) / 0.85, 0.0, 1.0) * np.clip((ey - 1.95) / 0.75, 0.0, 1.0)
    aperture *= in_nave_y

    # 7. 3D Hanging fern baskets cast real organic volumetric shadow streaks through upper skylight shafts!
    for bc in basket_centers:
        rx_b = wx - bc[0]
        ry_b = wy - bc[1]
        rz_b = wz - bc[2]
        proj = rx_b * SUN_DIR[0] + ry_b * SUN_DIR[1] + rz_b * SUN_DIR[2]
        px_b = rx_b - SUN_DIR[0] * proj
        py_b = ry_b - SUN_DIR[1] * proj
        pz_b = rz_b - SUN_DIR[2] * proj
        d2_b = (px_b * px_b + py_b * py_b) / (0.46 * 0.46) + (pz_b * pz_b) / (0.38 * 0.38)
        downstream = np.clip(proj / 0.35, 0.0, 1.0)
        aperture *= 1.0 - 0.78 * downstream * np.exp(-d2_b * 1.55)

    # 8. Opaque cedar potting workbench blocks rays underneath tabletop (dark shaded cavity under bench!)
    t_b = (BENCH_Z - wz) * inv_sz
    bx = wx - SUN_DIR[0] * t_b
    by = wy - SUN_DIR[1] * t_b
    under_bench_hit = (
        (wz < BENCH_Z - 0.01)
        & (t_b > 0.0)
        & (bx >= BENCH_X0)
        & (bx <= BENCH_X1)
        & (by >= BENCH_Y0)
        & (by <= BENCH_Y1)
    )
    dx_b = np.minimum(bx - BENCH_X0, BENCH_X1 - bx)
    dy_b = np.minimum(by - BENCH_Y0, BENCH_Y1 - by)
    d_in = np.minimum(dx_b, dy_b)
    bench_open = np.where(under_bench_hit, np.clip(1.0 - np.clip(d_in / 0.06, 0.0, 1.0) * 0.98, 0.02, 1.0), 1.0)
    aperture *= bench_open

    return np.clip(aperture, 0.0, 1.45).astype(np.float32), np.maximum(0.0, d_ray).astype(np.float32)


def paint_conservatory_volumetric_sunbeams(
    cam: Camera3D,
    rng: np.random.Generator,
    baskets_spec: list[tuple[np.ndarray, float, float, int]],
    m_global: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """
    Stage 2 & 3: True 3D ray-marched volumetric morning sunbeams (god-rays) pouring
    diagonally through the tall left-side arched glass window bays & upper-left vaulted
    glass skylight ribs along exact locked 3D SUN_DIR!

    Features:
      - Exact 3D perspective ray marching through conservatory air with R2 low-discrepancy
        stratified jitter (zero banding / zero striping!)
      - Exact 3D scene termination against floor, walls, right raised stone planter bed,
        and cedar potting workbench tabletop
      - Crisp dark architectural shadow bands (columns, climbing ivy, arched spandrels,
        horizontal transoms, glazing mullions, eave cornice, roof purlins, hanging ferns)
      - Depth-split into `rays_back` (behind hero potting bench plants for dreamy backlit
        silhouette separation) and `rays_front` (sweeping diagonally across the copper
        watering can, monstera, cedar bench & wet slate aisle puddles, while staying
        strictly behind Stage 4 foreground bokeh leaves!)
      - Returns (rays_back_rgb, rays_front_rgb, beam_vis_2d, sun_dir_2d).
    """
    p_ref = np.array([-1.2, 5.2, 2.2], dtype=np.float64)
    u0, v0, _ = cam.project(p_ref)
    u1, v1, _ = cam.project(p_ref + SUN_DIR * 3.0)
    dx2, dy2 = float(u1 - u0), float(v1 - v0)
    dl2 = math.hypot(dx2, dy2) + 1e-6
    beam_dir_2d = (dx2 / dl2, dy2 / dl2)

    # 1/3-res ray-march grid (682 x 384) -> silky optical penumbra & <1.8s deterministic CPU speed!
    vw, vh = 682, 384
    yy, xx = np.mgrid[0:vh, 0:vw].astype(np.float64)
    sx = (xx + 0.5) * (W / vw)
    sy = (yy + 0.5) * (H / vh)

    fx = cam.K[0, 0]
    cx, cy = cam.K[0, 2], cam.K[1, 2]
    dx_c = (sx - cx) / fx
    dy_c = (sy - cy) / fx
    R = cam.R
    rdx = R[0, 0] * dx_c + R[1, 0] * dy_c + R[2, 0]
    rdy = R[0, 1] * dx_c + R[1, 1] * dy_c + R[2, 1]
    rdz = R[0, 2] * dx_c + R[1, 2] * dy_c + R[2, 2]
    rn = np.sqrt(rdx * rdx + rdy * rdy + rdz * rdz)
    rdx /= rn
    rdy /= rn
    rdz /= rn

    # Exact 3D scene exit distance t_max (floor, left/right walls, rear gable, stone planter, bench slab)
    t_max = np.full((vh, vw), 16.5, dtype=np.float64)
    t_floor = np.where(rdz < -1e-5, (Z_FLOOR - cam.eye[2]) / np.where(rdz < -1e-5, rdz, -1.0), 1e6)
    t_left = np.where(rdx < -1e-5, (X_LEFT - cam.eye[0]) / np.where(rdx < -1e-5, rdx, -1.0), 1e6)
    t_right = np.where(rdx > 1e-5, (X_RIGHT - cam.eye[0]) / np.where(rdx > 1e-5, rdx, 1.0), 1e6)
    t_rear = np.where(rdy > 1e-5, (Y_REAR - cam.eye[1]) / np.where(rdy > 1e-5, rdy, 1.0), 1e6)
    t_planter = _ray_aabb_hit(cam.eye, rdx, rdy, rdz, (PLANTER_X0, PLANTER_Y0, Z_FLOOR), (PLANTER_X1, PLANTER_Y1, PLANTER_Z))
    # Clamp rays to the full cedar potting bench volume [Z_FLOOR, BENCH_Z] so rays terminate cleanly
    # on the tabletop & shaded front apron (zero bright ray bleed across dark lower wooden slats!)
    t_bench = _ray_aabb_hit(cam.eye, rdx, rdy, rdz, (BENCH_X0, BENCH_Y0, Z_FLOOR), (BENCH_X1, BENCH_Y1, BENCH_Z))

    t_max = np.minimum(t_max, np.where(t_floor > 0.05, t_floor, 1e6))
    t_max = np.minimum(t_max, np.where(t_left > 0.05, t_left, 1e6))
    t_max = np.minimum(t_max, np.where(t_right > 0.05, t_right, 1e6))
    t_max = np.minimum(t_max, np.where(t_rear > 0.05, t_rear, 1e6))
    t_max = np.minimum(t_max, t_planter)
    t_max = np.minimum(t_max, t_bench)

    t_min = 0.68
    span_t = np.maximum(0.0, t_max - t_min)

    # Forward-scatter Mie/Henyey-Greenstein phase boost toward upper-left morning sun (-SUN_DIR)
    cos_sun = -(rdx * SUN_DIR[0] + rdy * SUN_DIR[1] + rdz * SUN_DIR[2])
    phase = (0.58 + 0.72 * np.power(np.clip(0.5 + 0.5 * cos_sun, 0.0, 1.0), 2.1)).astype(np.float32)

    # R2 low-discrepancy stratified jitter per pixel (silky smooth, zero noise, zero step stripes!)
    jitter = (xx * 0.7548776662466927 + yy * 0.5698402909980532) % 1.0

    # Hanging fern basket 3D canopy centers for authentic foliage shadows carved in skylight rays
    basket_centers = [bp + np.array([0.0, 0.0, -0.10], dtype=np.float64) for bp, _, _, _ in baskets_spec]

    acc_back_warm = np.zeros((vh, vw), dtype=np.float32)
    acc_back_sage = np.zeros((vh, vw), dtype=np.float32)
    acc_front_warm = np.zeros((vh, vw), dtype=np.float32)
    acc_front_sage = np.zeros((vh, vw), dtype=np.float32)
    # Peak-preserving soft-max buffers prevent multi-bay Y-sightline smearing so individual
    # arched window shafts & skylight god-rays stay razor-separated & dramatic in 3D!
    peak_back_warm = np.zeros((vh, vw), dtype=np.float32)
    peak_back_sage = np.zeros((vh, vw), dtype=np.float32)
    peak_front_warm = np.zeros((vh, vw), dtype=np.float32)
    peak_front_sage = np.zeros((vh, vw), dtype=np.float32)
    acc_win_bloom = np.zeros((vh, vw), dtype=np.float32)

    n_steps = 46
    dt = (span_t / n_steps).astype(np.float32)

    for s in range(n_steps):
        frac = (s + jitter) / n_steps
        t = t_min + frac * span_t
        wx = cam.eye[0] + rdx * t
        wy = cam.eye[1] + rdy * t
        wz = cam.eye[2] + rdz * t

        # Strictly inside the arched conservatory interior volume!
        z_v = vault_z(np.clip(wx, X_LEFT, X_RIGHT))
        inside = (
            (span_t > 0.02)
            & (wx >= X_LEFT + 0.005)
            & (wx <= X_RIGHT - 0.005)
            & (wy >= Y_FRONT)
            & (wy <= Y_REAR - 0.01)
            & (wz >= Z_FLOOR + 0.005)
            & (wz <= z_v - 0.005)
        )
        if not np.any(inside):
            continue

        ap, d_ray = _trace_volumetric_aperture(wx, wy, wz, basket_centers)
        # Non-linear aperture sharpening so shadow corridors between mullions/transoms/arches stay deeply dark!
        ap_hi = np.power(np.clip(ap, 0.0, 1.5), 1.92).astype(np.float32)

        # Humid conservatory 3D drifting mist-density wisps (gives painterly organic texture inside shafts!)
        mist3d = (
            0.78
            + 0.15 * np.sin(wx * 1.18 + wy * 0.52 - wz * 0.88 + 0.4)
            + 0.10 * np.cos(wy * 1.25 - wx * 0.92 + wz * 1.05 + 1.1)
        ).astype(np.float32)

        # Extended physical attenuation along each sunbeam so golden shafts visibly sweep all the way
        # from the left arched windows & vault skylight diagonally across the bench, monstera & wet aisle puddles!
        near_glass = np.exp(-d_ray / 2.15).astype(np.float32)
        beam_atten = np.exp(-d_ray * 0.125).astype(np.float32) * (1.0 + 0.68 * near_glass)

        # Near-to-mid conservatory (Bays 2..5, y = 3.0..12.2) stays cohesive & luminous; gently taper very near lens
        near_cam_fade = np.clip((t - 1.15) / 1.10, 0.0, 1.0).astype(np.float32)
        depth_w = (0.38 + 0.62 * np.exp(-np.maximum(0.0, wy - 4.8) / 7.2)).astype(np.float32) * near_cam_fade
        cam_ext = np.exp(-np.maximum(0.0, t - 2.6) * 0.038).astype(np.float32)
        humid_z = (0.82 + 0.24 * np.exp(-((wz - 1.45) / 1.65) ** 2)).astype(np.float32)

        inst_dens = (
            inside.astype(np.float32)
            * ap_hi
            * beam_atten
            * mist3d
            * depth_w
            * cam_ext
            * humid_z
        )
        contrib = inst_dens * dt

        # Split warm golden-cream aperture core vs delicate golden-sage mid-air tail
        w_warm = np.clip(0.42 + 0.58 * np.exp(-d_ray / 3.20), 0.0, 1.0).astype(np.float32)
        c_warm = contrib * w_warm
        c_sage = contrib * (1.0 - w_warm)
        p_warm = inst_dens * w_warm
        p_sage = inst_dens * (1.0 - w_warm)

        # 3D Depth Split relative to hero potting bench props (`hero_1x` occupies x < -1.24, y in [2.1, 9.0]):
        # - Air tight against left glass behind bench plants (wx < -2.35) -> rays_back (dreamy backlit rim!)
        # - Air sweeping out from tall left arched windows across monstera, watering can, cedar bench & aisle
        #   (above benchtop or aisle-side wx > -1.85) -> rays_front (unmistakable connected shafts from windows!)
        above_bench = np.clip((wz - (BENCH_Z + 0.04)) / 0.26, 0.0, 1.0).astype(np.float32)
        f_x_hi = np.clip((wx - (-2.42)) / 0.52, 0.0, 1.0).astype(np.float32) * above_bench
        f_x_lo = np.clip((wx - (-1.92)) / 0.50, 0.0, 1.0).astype(np.float32)
        f_y = np.clip((3.55 - wy) / 0.85, 0.0, 1.0).astype(np.float32) * above_bench
        w_front = np.maximum(np.maximum(f_x_hi, f_x_lo), f_y)
        w_back = 1.0 - w_front

        acc_front_warm += c_warm * w_front
        acc_front_sage += c_sage * w_front
        acc_back_warm += c_warm * w_back
        acc_back_sage += c_sage * w_back

        peak_front_warm = np.maximum(peak_front_warm, p_warm * w_front)
        peak_front_sage = np.maximum(peak_front_sage, p_sage * w_front)
        peak_back_warm = np.maximum(peak_back_warm, p_warm * w_back)
        peak_back_sage = np.maximum(peak_back_sage, p_sage * w_back)

        # Tighter crisp aperture glow strictly inside bright sunlit window openings (zero milky mullion wash!)
        acc_win_bloom += inside.astype(np.float32) * ap_hi * np.exp(-d_ray / 0.48).astype(np.float32) * depth_w * dt

    # Combine true 3D volumetric integral + peak-preserving shaft ridge (eliminates multi-bay washout!)
    # Wide low-freq floor (sigma=48) preserves full arched-window shaft width while keeping dark mullion corridors CRISP!
    def _shape_shaft(
        warm_int: np.ndarray,
        sage_int: np.ndarray,
        warm_pk: np.ndarray,
        sage_pk: np.ndarray,
        gain_int: float,
        gain_pk: float,
        max_out: float,
    ) -> np.ndarray:
        warm_raw = (warm_int * gain_int + warm_pk * gain_pk) * phase
        sage_raw = (sage_int * gain_int + sage_pk * gain_pk) * phase
        # Gentle 1.0px denoise at 1/3 res -> silky painterly optical shafts, ZERO noise, ZERO striping!
        warm_s = gaussian_blur_2d(warm_raw, 1.0)
        sage_s = gaussian_blur_2d(sage_raw, 1.0)
        tot_s = np.maximum(1e-5, warm_s + sage_s)

        local_floor = gaussian_blur_2d(tot_s, 42.0)
        ridge = np.maximum(0.0, tot_s - 0.42 * local_floor)
        combined = 0.22 * tot_s + 1.38 * ridge
        shaped = np.power(np.clip(combined, 0.0, 4.5), 1.15)
        shaped = (shaped / (1.0 + 0.92 * shaped)) * max_out

        scale = shaped / tot_s
        w_f = warm_s * scale
        s_f = sage_s * scale
        # Theatrical 1990s Studio Ghibli golden-morning cream (#ffefaa) -> sunlit golden-sage (#e7ee92)
        col_warm = np.array([1.00, 0.94, 0.62], dtype=np.float32)
        col_sage = np.array([0.93, 0.95, 0.58], dtype=np.float32)
        return w_f[..., None] * col_warm[None, None, :] + s_f[..., None] * col_sage[None, None, :]

    back_lo = _shape_shaft(
        acc_back_warm, acc_back_sage, peak_back_warm, peak_back_sage,
        gain_int=0.58, gain_pk=0.92, max_out=0.38,
    )
    front_lo = _shape_shaft(
        acc_front_warm, acc_front_sage, peak_front_warm, peak_front_sage,
        gain_int=0.64, gain_pk=1.08, max_out=0.44,
    )

    # Balanced warm pane glow strictly inside open window apertures (preserves iron mullions & deep botanical greens!)
    bloom_s = gaussian_blur_2d(acc_win_bloom * phase * 0.24, 1.6)
    bloom_s = bloom_s / (1.0 + 1.25 * bloom_s)
    back_lo += bloom_s[..., None] * np.array([1.00, 0.94, 0.62], dtype=np.float32) * 0.14

    # Upsample smoothly to full 2048x1152 and infuse subtle gouache mottling so rays feel hand-painted!
    rays_back = np.clip(resize_bicubic(back_lo, W, H), 0.0, 0.44)
    rays_front = np.clip(resize_bicubic(front_lo, W, H), 0.0, 0.48)

    gouache_tex = np.clip(1.0 + 0.095 * m_global, 0.82, 1.18)[..., None]
    rays_back *= gouache_tex
    rays_front *= gouache_tex

    # Combined scalar sunbeam visibility map for lighting suspended golden pollen & mist motes!
    beam_vis_2d = np.clip(
        0.299 * (rays_back[:, :, 0] + rays_front[:, :, 0])
        + 0.587 * (rays_back[:, :, 1] + rays_front[:, :, 1])
        + 0.114 * (rays_back[:, :, 2] + rays_front[:, :, 2]),
        0.0,
        1.0,
    ).astype(np.float32)

    return rays_back, rays_front, beam_vis_2d, beam_dir_2d


# =============================================================================
# Stage 4 — Foreground Disc Bokeh Monstera/Fern Fronds & 35mm Optical Finish
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


def paint_foreground_bokeh_botanicals(rng: np.random.Generator) -> np.ndarray:
    """
    Stage 4: Near-camera foreground wet monstera leaf & arching fern fronds framing
    the lower-right & upper-right corners (leaving the open garden doorway, hero
    watering puddles & potting workbench cleanly unobstructed!), convolved in
    premultiplied RGBA with make_disc_kernel(radius=7.8) + separate cv2.warpAffine
    ~1px opposite R/B lateral chromatic shift!
    """
    surf, ctx = new_cairo_2x()
    c_dk = (0.048, 0.128, 0.072)
    c_md = (0.145, 0.295, 0.145)
    c_lt = (0.460, 0.620, 0.265)
    c_vn = (0.700, 0.800, 0.440)

    # Lower-right near-camera wet fenestrated Monstera deliciosa leaves curving softly in!
    draw_monstera_leaf_bezier(ctx, W - 95, H - 155, size=245.0, ang=-2.55, dark_c=c_dk, mid_c=c_md, lit_c=c_lt, vein_c=c_vn, alpha=0.96, backlit=0.62)
    draw_monstera_leaf_bezier(ctx, W - 260, H - 30, size=210.0, ang=-2.20, dark_c=c_dk, mid_c=c_md, lit_c=c_lt, vein_c=c_vn, alpha=0.94, backlit=0.72)

    # Lower-right graceful arching foreground fern frond tips kissing the corner frame
    for idx_f, (fx, fy, flen, fang, droop) in enumerate(
        (
            (W + 20, H - 240, 310, 2.75, 85),
            (W + 10, H - 90, 340, 2.95, 55),
            (W - 80, H + 30, 290, -2.18, 60),
        )
    ):
        draw_fern_frond_bezier(
            ctx, fx, fy, flen, fang, droop, c_dk, c_md, c_lt, np.random.default_rng(9500 + idx_f * 23), alpha=0.92, n_pairs=15
        )

    # Delicate upper-right trailing hanging fern frond tip entering softly from top-right edge
    for idx_u, (ux, uy, ulen, uang, udroop) in enumerate(
        (
            (W + 15, 45, 320, 2.62, 115),
            (W - 95, -15, 295, 2.25, 130),
        )
    ):
        draw_fern_frond_bezier(
            ctx, ux, uy, ulen, uang, udroop, c_dk, c_md, c_lt, np.random.default_rng(9600 + idx_u * 29), alpha=0.88, n_pairs=14
        )

    fg_1x = finish_cairo_2x(surf)
    blurred = convolve_fft_2d_bbox(fg_1x, make_disc_kernel(radius=7.8))

    M_r = np.array([[1.0, 0.0, 1.05], [0.0, 1.0, 0.0]], dtype=np.float32)
    M_b = np.array([[1.0, 0.0, -1.05], [0.0, 1.0, 0.0]], dtype=np.float32)
    out = blurred.copy()
    out[:, :, 0] = cv2.warpAffine(blurred[:, :, 0], M_r, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    out[:, :, 2] = cv2.warpAffine(blurred[:, :, 2], M_b, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return np.clip(out, 0.0, 1.0)


def apply_35mm_finish(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Stage 4 float32 35mm theatrical anime optical finish:
      1. 3-scale warm cream-sage highlight halation (sigma = 6.5, 24, 72 px above lum > 0.72)
      2. Radial corner CA + subtle optical vignette
      3. Shadow black lift + C1-continuous tanh highlight shoulder (knee = 0.80, ceiling = 0.965)
      4. Luminance-weighted silver-halide film grain
      5. Faint YIQ tape patina (1.0 px horizontal chroma softening with BORDER_REFLECT_101,
         slight luma ring, sub-pixel line jitter, <= 0.70% scanlines).
    """
    img = rgb.astype(np.float32).copy()

    # 1. 3-scale warm cream-sage highlight halation
    lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    hl_mask = np.clip((lum - 0.72) / 0.28, 0.0, 1.5)
    hl_src = img * hl_mask[..., None]
    tint = np.array([1.05, 1.04, 0.88], dtype=np.float32)
    hal = (
        0.22 * gaussian_blur_2d(hl_src, 6.5)
        + 0.16 * gaussian_blur_2d(hl_src, 24.0)
        + 0.10 * gaussian_blur_2d(hl_src, 72.0)
    ) * tint[None, None, :]
    img = 1.0 - (1.0 - np.clip(img, 0.0, 1.0)) * (1.0 - np.clip(hal, 0.0, 0.80))

    # 2. Radial corner chromatic aberration + gentle optical vignette
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (xx - W * 0.5) / (W * 0.5)
    ny = (yy - H * 0.5) / (H * 0.5)
    r2 = nx * nx + ny * ny
    ca_scale = 0.0010
    img[:, :, 0] = remap_reflect(img[:, :, 0], xx + nx * r2 * (W * ca_scale), yy + ny * r2 * (H * ca_scale))
    img[:, :, 2] = remap_reflect(img[:, :, 2], xx - nx * r2 * (W * ca_scale), yy - ny * r2 * (H * ca_scale))
    img *= np.clip(1.0 - 0.19 * np.power(r2, 1.15), 0.75, 1.0)[..., None]

    # 3. Shadow black lift + C1-continuous tanh highlight shoulder (knee = 0.80, ceiling = 0.965)
    img = 0.014 + img * 0.986
    knee, ceiling = 0.80, 0.965
    span = ceiling - knee
    img = np.where(img > knee, knee + span * np.tanh((img - knee) / span), img)

    # 4. Luminance-weighted silver-halide grain
    lum2 = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    env = np.clip(1.12 - 0.95 * np.power(np.clip(lum2, 0.0, 1.0), 1.55), 0.12, 1.12)
    grain = rng.normal(0.0, 0.0094, size=(H, W, 3)).astype(np.float32)
    mono_g = rng.normal(0.0, 0.0078, size=(H, W, 1)).astype(np.float32)
    img = np.clip(img + (0.65 * mono_g + 0.35 * grain) * env[..., None], 0.0, 1.0)

    # 5. Faint YIQ tape patina (1.0 px horizontal chroma softening, BORDER_REFLECT_101, <= 0.70% scanlines)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    I = 0.596 * R - 0.274 * G - 0.322 * B
    Q = 0.211 * R - 0.523 * G + 0.312 * B

    I_soft = gaussian_blur_1d_x(I, 1.0)
    Q_soft = gaussian_blur_1d_x(Q, 1.0)
    Y_ring = Y + 0.036 * (Y - gaussian_blur_1d_x(Y, 1.1))

    out = np.empty_like(img)
    out[:, :, 0] = Y_ring + 0.956 * I_soft + 0.621 * Q_soft
    out[:, :, 1] = Y_ring - 0.272 * I_soft - 0.647 * Q_soft
    out[:, :, 2] = Y_ring - 1.106 * I_soft + 1.703 * Q_soft

    line_shift = cv2.GaussianBlur(
        rng.normal(0.0, 0.14, size=(H, 1)).astype(np.float32),
        (1, 7),
        sigmaX=0.0,
        sigmaY=1.2,
        borderType=cv2.BORDER_REFLECT_101,
    )
    jx = np.broadcast_to(np.arange(W, dtype=np.float32)[None, :], (H, W)) + line_shift
    jy = np.broadcast_to(np.arange(H, dtype=np.float32)[:, None], (H, W))
    out = remap_reflect(out, jx, jy)
    out *= (1.0 - 0.0068 * (0.5 - 0.5 * np.cos(np.arange(H, dtype=np.float32) * math.pi)))[:, None, None]
    return np.clip(out, 0.0, 1.0)


# =============================================================================
# Stage 5 — Deterministic Scene Assembly, Camera3D UV Readout & Export
# =============================================================================

def render_scene(
    seed: int = 20261125,
    audit_dir: pathlib.Path | None = None,
    export_webps: list[pathlib.Path] | None = None,
) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # Stage 1: Off-axis Camera3D (VP ~58.1% width, cleanly inside 56%-60%!)
    cam = Camera3D(W, H, eye=(0.52, 0.20, 1.46), target=(-0.62, 12.0, 1.36), fov_x_deg=61.5)
    print(f"[matcha-01] Camera VP: ({cam.vp_x:.1f}, {cam.vp_y:.1f}) -> {cam.vp_x / W * 100:.2f}% width, horizon_v={cam.horizon_y / H:.4f}")

    # Hanging fern baskets (anchored cleanly on vaulted roof ribs/tie-rods — zero collision with palms!)
    baskets_spec = [
        (np.array([-1.06, 4.35, 2.78]), float(vault_z(-1.06)), 0.20, 6101),
        (np.array([0.46, 5.45, 2.96]), float(vault_z(0.46)), 0.20, 6102),
        (np.array([-0.92, 8.00, 2.86]), float(vault_z(-0.92)), 0.20, 6103),
        (np.array([0.62, 10.10, 2.90]), float(vault_z(0.62)), 0.19, 6104),
        (np.array([-0.68, 12.20, 2.92]), float(vault_z(-0.68)), 0.18, 6105),
    ]
    lamp_pos = np.array([-0.04, 6.75, 3.06], dtype=np.float64)
    lamp_anchor = float(vault_z(-0.04))

    # Potting workbench prop layout (exact 3D coordinates on tabletop z = BENCH_Z)
    # Lush, brimming botanical nursery collection showcasing rich pot & foliage diversity
    # while keeping the copper watering can, apothecary jars, wooden seed tray & tools clear!
    bench_pots = [
        # (wx, wy, rad, height, style, kind, seed) — ordered back-to-front!
        (-2.12, 9.10, 0.16, 0.24, "terracotta", "sansevieria", 7200),
        (-1.58, 8.75, 0.135, 0.21, "cream", "bushy", 7201),
        (-2.06, 8.35, 0.16, 0.24, "terracotta", "fern", 7211),
        (-1.54, 7.95, 0.135, 0.21, "matcha", "trailing", 7202),
        (-2.14, 7.42, 0.175, 0.27, "terracotta", "monstera", 7203),
        (-1.56, 7.12, 0.125, 0.19, "cream", "calathea", 7212),
        (-2.04, 6.72, 0.14, 0.21, "stone", "bushy", 7204),
        (-1.52, 6.52, 0.115, 0.17, "terracotta", "succulent", 7213),
        (-2.16, 5.95, 0.165, 0.25, "terracotta", "fern", 7205),
        (-1.50, 5.42, 0.125, 0.19, "matcha", "trailing", 7206),
        (-2.18, 5.05, 0.155, 0.24, "cream", "calathea", 7214),
        (-2.16, 4.46, 0.15, 0.23, "terracotta", "bushy", 7207),
        (-2.20, 3.88, 0.165, 0.25, "matcha", "fern", 7208),
        (-1.82, 3.30, 0.11, 0.16, "cream", "succulent", 7215),
        (-2.22, 3.08, 0.15, 0.23, "terracotta", "sansevieria", 7209),
        (-1.42, 3.04, 0.082, 0.125, "terracotta", "succulent", 7216),
        (-1.90, 2.64, 0.125, 0.19, "matcha", "trailing", 7210),
        (-1.38, 2.58, 0.095, 0.145, "cream", "bushy", 7217),
    ]
    watering_can_pos = np.array([-1.60, 3.55, BENCH_Z], dtype=np.float64)
    jar_positions = [
        (np.array([-1.50, 4.42, BENCH_Z]), 7301),
        (np.array([-1.75, 4.74, BENCH_Z]), 7302),
    ]
    pot_footprints = [(p[0], p[1], p[2]) for p in bench_pots] + [
        (float(watering_can_pos[0]), float(watering_can_pos[1]), 0.16),
        (-1.50, 4.42, 0.075),
        (-1.75, 4.74, 0.075),
        (-1.68, 6.08, 0.24),  # seed tray
    ]

    # Stage 3 Band 0: Luminous misty morning botanical garden backdrop outside the glasshouse
    canvas, fog_rgb, m_global, d_global = paint_misty_garden_backdrop(cam, rng)

    # Stage 1 Planar Architecture: Rear Arched Glass Gable Wall + Left & Right Arched Glass Side Walls
    rear_uv = paint_rear_gable_uv(rng)
    canvas = cam.warp_plane_over(
        canvas,
        rear_uv,
        p0=np.array([X_LEFT, Y_REAR, Z_FLOOR]),
        e1=np.array([X_RIGHT - X_LEFT, 0.0, 0.0]),
        e2=np.array([0.0, 0.0, Z_PEAK + 0.06 - Z_FLOOR]),
        fog_rgb=fog_rgb,
        fog_start=5.5,
        fog_density=0.042,
    )

    left_wall_uv = paint_side_glass_wall_uv(rng, side="left")
    canvas = cam.warp_plane_over(
        canvas,
        left_wall_uv,
        p0=np.array([X_LEFT, Y_FRONT, Z_FLOOR]),
        e1=np.array([0.0, Y_REAR - Y_FRONT, 0.0]),
        e2=np.array([0.0, 0.0, Z_SPRING - Z_FLOOR]),
        fog_rgb=fog_rgb,
        fog_start=5.0,
        fog_density=0.042,
    )

    right_wall_uv = paint_side_glass_wall_uv(rng, side="right")
    canvas = cam.warp_plane_over(
        canvas,
        right_wall_uv,
        p0=np.array([X_RIGHT, Y_FRONT, Z_FLOOR]),
        e1=np.array([0.0, Y_REAR - Y_FRONT, 0.0]),
        e2=np.array([0.0, 0.0, Z_SPRING - Z_FLOOR]),
        fog_rgb=fog_rgb,
        fog_start=5.0,
        fog_density=0.042,
    )

    # Stage 3 Band 1: Curved Glass Vault Sheen, Soaring Arched Wrought-Iron Ribs, Filigree,
    # Open French Doors, Hanging Fern Baskets & Vintage Brass Pendant Lamp!
    s_vault, c_vault = new_cairo_2x()
    draw_glass_vault_sheen(c_vault, cam, fog_rgb)
    draw_vaulted_wrought_iron(c_vault, cam, fog_rgb, mirrored=False)
    for yb_ivy, sd_ivy in ((10.10, 6401), (8.00, 6402), (5.90, 6403), (3.80, 6404)):
        draw_climbing_ivy_on_column(c_vault, cam, X_LEFT + 0.04, yb_ivy, Z_KNEE + 0.05, Z_SPRING + 0.35, sd_ivy, fog_rgb, mirrored=False)
    draw_climbing_ivy_on_column(c_vault, cam, X_RIGHT - 0.04, 8.00, PLANTER_Z + 0.1, Z_SPRING + 0.25, 6409, fog_rgb, mirrored=False)

    baskets_uv = []
    for bp, bz_a, brad, bseed in reversed(baskets_spec):
        buv = draw_hanging_fern_basket(c_vault, cam, bp, bz_a, brad, bseed, fog_rgb, mirrored=False)
        baskets_uv.append(buv)
    lamp_uv = draw_vintage_brass_lamp(c_vault, cam, lamp_pos, lamp_anchor, fog_rgb, mirrored=False)

    vault_1x = finish_cairo_2x(s_vault)
    va = vault_1x[:, :, 3:4]
    vault_1x[:, :, :3] = np.clip(vault_1x[:, :, :3] + va * (m_global * 0.015 + d_global * 0.010)[..., None], 0.0, 1.25)
    canvas = over_premul(canvas, vault_1x)

    # True 3D ray-marched volumetric window-aperture & skylight morning sunbeams!
    # Computed once in exact 3D and cleanly split into back (behind bench plants) & front (across bench/aisle)
    rays_back, rays_front, beam_vis_2d, beam_dir_2d = paint_conservatory_volumetric_sunbeams(
        cam, rng, baskets_spec, m_global
    )

    # Snapshot conservatory upper architecture & skylight backdrop BEFORE floor puddles
    # (infusing a whisper of back sunbeam glow so wet aisle puddles mirror the glowing morning shafts!)
    ref_backdrop = 1.0 - (1.0 - np.clip(canvas, 0.0, 1.0)) * (1.0 - np.clip(rays_back * 0.45, 0.0, 0.55))

    # Dedicated 3D plane-mirrored quads across Z_FLOOR = 0 for exact geometric watering-puddle reflections!
    s_ref, c_ref = new_cairo_2x()
    draw_vaulted_wrought_iron(c_ref, cam, fog_rgb, mirrored=True)
    for bp, bz_a, brad, bseed in reversed(baskets_spec):
        draw_hanging_fern_basket(c_ref, cam, bp, bz_a, brad, bseed, fog_rgb, mirrored=True)
    draw_vintage_brass_lamp(c_ref, cam, lamp_pos, lamp_anchor, fog_rgb, mirrored=True)
    draw_cedar_bench_structure(c_ref, cam, fog_rgb, mirrored=True)
    draw_right_planter_botanicals(c_ref, cam, rng, fog_rgb, mirrored=True)
    mirrored_3d = finish_cairo_2x(s_ref)

    # Stage 1 Wet Voronoi Mossy Slate Flagstone Floor + Stage 2 Fresnel Watering Puddles!
    floor_uv, _ = paint_slate_floor_uv(rng)
    canvas = cam.warp_plane_over(
        canvas,
        floor_uv,
        p0=np.array([X_LEFT, Y_FRONT, Z_FLOOR]),
        e1=np.array([X_RIGHT - X_LEFT, 0.0, 0.0]),
        e2=np.array([0.0, Y_REAR + 0.90 - Y_FRONT, 0.0]),
        fog_rgb=fog_rgb,
        fog_start=5.6,
        fog_density=0.036,
    )
    canvas, puddle_mask = paint_floor_puddle_reflections(canvas, ref_backdrop, mirrored_3d, cam, rng, fog_rgb, m_global)

    # Stage 1 Planar Surfaces: Right Raised Mossy Stone Planter Bed (Soil Top + Vertical Stone Curb)
    planter_soil_uv = paint_planter_soil_uv(rng)
    canvas = cam.warp_plane_over(
        canvas,
        planter_soil_uv,
        p0=np.array([PLANTER_X0, PLANTER_Y0, PLANTER_Z]),
        e1=np.array([PLANTER_X1 - PLANTER_X0, 0.0, 0.0]),
        e2=np.array([0.0, PLANTER_Y1 - PLANTER_Y0, 0.0]),
        fog_rgb=fog_rgb,
        fog_start=5.5,
        fog_density=0.036,
    )
    planter_curb_uv = paint_planter_curb_uv(rng)
    canvas = cam.warp_plane_over(
        canvas,
        planter_curb_uv,
        p0=np.array([PLANTER_X0, PLANTER_Y0, Z_FLOOR]),
        e1=np.array([0.0, PLANTER_Y1 - PLANTER_Y0, 0.0]),
        e2=np.array([0.0, 0.0, PLANTER_Z - Z_FLOOR]),
        fog_rgb=fog_rgb,
        fog_start=5.5,
        fog_density=0.036,
    )

    # Stage 3 Band 2/3 Midground: Stepped Tiered Plant Rack, Lush Aisle Floor Planters, Right Planter Jungle & Bench
    s_mid, c_mid = new_cairo_2x()
    draw_tiered_plant_rack(c_mid, cam, rng, fog_rgb)
    # Lush conservatory potted floor planters lining BOTH sides of the wet slate aisle (leaving center puddles open!)
    floor_mid_pots = [
        # (wx, wy, rad, height, style, kind, seed) — strictly back-to-front!
        (-1.28, 13.75, 0.24, 0.42, "terracotta", "fern", 7901),
        (1.24, 13.65, 0.23, 0.40, "stone", "fern", 7902),
        (1.06, 12.25, 0.19, 0.32, "terracotta", "calathea", 7903),
        (-1.16, 11.45, 0.20, 0.34, "cream", "sansevieria", 7904),
        (1.08, 10.55, 0.21, 0.35, "matcha", "bushy", 7905),
        (-1.14, 9.65, 0.19, 0.31, "terracotta", "calathea", 7906),
        (1.06, 8.75, 0.22, 0.36, "terracotta", "fern", 7907),
        (1.08, 7.15, 0.20, 0.33, "cream", "sansevieria", 7908),
        (1.04, 5.65, 0.21, 0.34, "matcha", "calathea", 7909),
    ]
    for fx, fy, frad, fht, fsty, fkind, fsd in floor_mid_pots:
        draw_potted_plant(c_mid, cam, np.array([fx, fy, Z_FLOOR]), frad, fht, fsty, fkind, fsd, fog_rgb)
    draw_right_planter_botanicals(c_mid, cam, rng, fog_rgb, mirrored=False)
    draw_cedar_bench_structure(c_mid, cam, fog_rgb, mirrored=False)
    mid_1x = finish_cairo_2x(s_mid)
    ma = mid_1x[:, :, 3:4]
    mid_1x[:, :, :3] = np.clip(mid_1x[:, :, :3] + ma * (m_global * 0.025 + d_global * 0.017)[..., None], 0.0, 1.25)
    canvas = over_premul(canvas, mid_1x)

    # Lower slatted storage shelf stocked generously with potted shade ferns, trailing vines & ceramic planters!
    shelf_uv = paint_cedar_bench_shelf_uv(rng)
    canvas = cam.warp_plane_over(
        canvas,
        shelf_uv,
        p0=np.array([BENCH_X0 + 0.06, BENCH_Y0 + 0.06, BENCH_SHELF_Z]),
        e1=np.array([BENCH_X1 - BENCH_X0 - 0.12, 0.0, 0.0]),
        e2=np.array([0.0, BENCH_Y1 - BENCH_Y0 - 0.12, 0.0]),
        fog_rgb=fog_rgb,
        fog_start=5.8,
        fog_density=0.036,
    )
    s_under, c_under = new_cairo_2x()
    for sx, sy, srad, sh, ssty, skind, ssd in (
        (-1.74, 8.35, 0.145, 0.21, "terracotta", "fern", 7801),
        (-1.54, 7.65, 0.135, 0.19, "cream", "bushy", 7802),
        (-1.72, 6.95, 0.15, 0.22, "matcha", "calathea", 7803),
        (-1.55, 6.25, 0.135, 0.20, "terracotta", "trailing", 7804),
        (-1.74, 5.55, 0.145, 0.21, "terracotta", "fern", 7805),
        (-1.52, 4.92, 0.135, 0.19, "cream", "succulent", 7806),
        (-1.76, 4.30, 0.15, 0.22, "matcha", "calathea", 7807),
        (-1.56, 3.78, 0.14, 0.20, "terracotta", "bushy", 7808),
        (-1.74, 3.18, 0.145, 0.21, "stone", "fern", 7809),
        (-1.54, 2.75, 0.13, 0.18, "terracotta", "trailing", 7810),
    ):
        draw_potted_plant(c_under, cam, np.array([sx, sy, BENCH_SHELF_Z]), srad, sh, ssty, skind, ssd, fog_rgb)
    canvas = over_premul(canvas, finish_cairo_2x(s_under))

    # Stage 1 Planar Weathered Cedar Potting Workbench Tabletop (with exact 6-step sunlight + pot shadows!)
    bench_top_uv = paint_cedar_bench_top_uv(rng, pot_footprints)
    canvas = cam.warp_plane_over(
        canvas,
        bench_top_uv,
        p0=np.array([BENCH_X0, BENCH_Y0, BENCH_Z]),
        e1=np.array([BENCH_X1 - BENCH_X0, 0.0, 0.0]),
        e2=np.array([0.0, BENCH_Y1 - BENCH_Y0, 0.0]),
        fog_rgb=fog_rgb,
        fog_start=5.8,
        fog_density=0.034,
    )

    # Balanced highlight-protected morning sunbeams behind bench plants (clearly visible shafts, zero white washout!)
    lum_b = 0.299 * canvas[:, :, 0] + 0.587 * canvas[:, :, 1] + 0.114 * canvas[:, :, 2]
    hl_guard_b = (1.0 - 0.52 * np.clip((lum_b - 0.72) / 0.24, 0.0, 1.0))[..., None]
    canvas = np.clip(
        canvas + rays_back * (0.48 + 0.52 * (1.0 - np.clip(canvas, 0.0, 1.0))) * hl_guard_b,
        0.0,
        1.18,
    )

    # Stage 3 Band 2 & 3 Hero Foreground: Potting Bench Terracotta Planters, Copper Watering Can,
    # Apothecary Propagation Jars, Wooden Seed Tray, Garden Tools & Grounded Aisle Floor Planters!
    s_hero, c_hero = new_cairo_2x()
    seed_drawn = False
    jars_drawn = False
    can_drawn = False
    for bx, by, brad, bht, bsty, bkind, bsd in bench_pots:
        # Insert seed tray, propagation jars & copper watering can in exact back-to-front Y depth order!
        if (not seed_drawn) and by <= 6.00:
            draw_wooden_seed_tray(c_hero, cam, -1.86, 5.72, 0.50, 0.72, fog_rgb)
            seed_drawn = True
        if (not jars_drawn) and by <= 4.78:
            for jp, jsd in reversed(jar_positions):
                draw_glass_propagation_jar(c_hero, cam, jp, jsd, fog_rgb)
            jars_drawn = True
        if (not can_drawn) and by <= 3.60:
            draw_copper_watering_can(c_hero, cam, watering_can_pos, fog_rgb)
            can_drawn = True
        draw_potted_plant(c_hero, cam, np.array([bx, by, BENCH_Z]), brad, bht, bsty, bkind, bsd, fog_rgb)

    draw_garden_trowel_and_shears(c_hero, cam, fog_rgb)

    # Additional grounded floor planters tucked neatly beside the left bench legs & foreground edge!
    for fx, fy, frad, fht, fsty, fkind, fsd in (
        (-1.12, 7.35, 0.185, 0.30, "terracotta", "fern", 7941),
        (-1.08, 5.15, 0.18, 0.29, "matcha", "calathea", 7942),
        (-1.06, 3.62, 0.17, 0.28, "cream", "bushy", 7943),
        (-1.14, 2.38, 0.175, 0.29, "terracotta", "fern", 7950),
        (-2.26, 2.15, 0.15, 0.25, "terracotta", "monstera", 7955),
    ):
        fz = Z_FLOOR + 0.42 if fy < 2.20 else Z_FLOOR
        draw_potted_plant(c_hero, cam, np.array([fx, fy, fz]), frad, fht, fsty, fkind, fsd, fog_rgb)

    hero_1x = finish_cairo_2x(s_hero)
    h_tooth = paper_tooth(H, W, rng)
    ha = hero_1x[:, :, 3:4]
    hero_1x[:, :, :3] = np.clip(hero_1x[:, :, :3] + ha * (m_global * 0.024 + d_global * 0.016 + h_tooth * 0.008)[..., None], 0.0, 1.25)
    canvas = over_premul(canvas, hero_1x)

    # Balanced front morning sunbeam shafts sweeping poetically across bench, potted plants & wet stone aisle!
    lum_f = 0.299 * canvas[:, :, 0] + 0.587 * canvas[:, :, 1] + 0.114 * canvas[:, :, 2]
    hl_guard_f = (1.0 - 0.55 * np.clip((lum_f - 0.72) / 0.24, 0.0, 1.0))[..., None]
    canvas = np.clip(
        canvas + rays_front * (0.50 + 0.50 * (1.0 - np.clip(canvas, 0.0, 1.0))) * hl_guard_f,
        0.0,
        1.18,
    )

    # Delicate suspended golden pollen motes catching morning light inside the active sunbeams!
    s_motes, c_motes = new_cairo_2x()
    for idx_m in range(195):
        if idx_m < 145:
            mx = float(rng.uniform(55, W * 0.76))
            my = float(rng.uniform(45, H * 0.88))
        else:
            mx = float(rng.uniform(90, W - 110))
            my = float(rng.uniform(80, H - 100))
        bv = float(beam_vis_2d[int(np.clip(my, 0, H - 1)), int(np.clip(mx, 0, W - 1))])
        if bv < 0.08 and idx_m < 135:
            continue
        in_beam = float(np.clip((bv - 0.06) / 0.28, 0.0, 1.0))
        mr = float(rng.uniform(0.85, 2.0)) * (0.72 + 0.48 * in_beam)
        ma_v = min(0.70, float(rng.uniform(0.16, 0.40)) * (0.22 + 1.25 * in_beam))
        if in_beam > 0.28:
            g_m = cairo.RadialGradient(mx, my, mr * 0.25, mx, my, mr * 2.8)
            g_m.add_color_stop_rgba(0.0, 1.00, 0.97, 0.74, ma_v * 0.45)
            g_m.add_color_stop_rgba(1.0, 1.00, 0.94, 0.62, 0.0)
            c_motes.set_source(g_m)
            c_motes.arc(mx, my, mr * 2.8, 0, math.tau)
            c_motes.fill()
        c_motes.set_source_rgba(1.00, 0.98, 0.78, ma_v)
        c_motes.arc(mx, my, mr, 0, math.tau)
        c_motes.fill()
    canvas = over_premul(canvas, finish_cairo_2x(s_motes))

    # Stage 4: Foreground disc-bokeh wet monstera/fern fronds (r=7.8 + R/B shift) & 35mm optical finish
    fg_bokeh = paint_foreground_bokeh_botanicals(rng)
    canvas = over_premul(canvas, fg_bokeh)

    final_rgb = apply_35mm_finish(canvas, rng)
    elapsed = time.perf_counter() - t0
    print(f"[matcha-01] Rendered {W}x{H} in {elapsed:.2f}s")

    # =========================================================================
    # Stage 5 — Exact Camera3D UV Coordinate Calibration for wall-01.frag
    # =========================================================================
    py_idx, px_idx = np.where(puddle_mask > 0.35)
    pud_u_min = float(px_idx.min()) / W if len(px_idx) else 0.42
    pud_u_max = float(px_idx.max()) / W if len(px_idx) else 0.78
    pud_v_min = float(py_idx.min()) / H if len(py_idx) else 0.58
    pud_v_max = float(py_idx.max()) / H if len(py_idx) else 0.98

    puddle_rows = {}
    for vv in (0.60, 0.66, 0.72, 0.78, 0.84, 0.90, 0.95):
        row = int(vv * H)
        xs_r = np.where(puddle_mask[row] > 0.35)[0]
        if len(xs_r):
            puddle_rows[f"v={vv:.2f}"] = (round(float(xs_r.min()) / W, 4), round(float(xs_r.max()) / W, 4))

    # Rear open arched garden doorway screen UV box
    dl_u, dl_v, _ = cam.project_uv(np.array([-DOOR_HW, Y_REAR, DOOR_TOP]))
    dr_u, dr_v, _ = cam.project_uv(np.array([DOOR_HW, Y_REAR, Z_FLOOR]))
    door_uv = (round(dl_u, 4), round(dr_u, 4), round(dl_v, 4), round(dr_v, 4))

    uv_info = {
        "vp_ratio": round(cam.vp_x / W, 4),
        "horizon_v": round(cam.horizon_y / H, 4),
        "sun_beam_dir_2d": (round(beam_dir_2d[0], 4), round(beam_dir_2d[1], 4)),
        "puddle_u_range": (round(pud_u_min, 4), round(pud_u_max, 4)),
        "puddle_v_range": (round(pud_v_min, 4), round(pud_v_max, 4)),
        "puddle_rows_u_by_v": puddle_rows,
        "doorway_uv_u0_u1_v0_v1": door_uv,
        "hanging_baskets_uv": [(round(b[0], 4), round(b[1], 4), round(b[2], 4)) for b in baskets_uv],
        "brass_lamp_uv": (round(lamp_uv[0], 4), round(lamp_uv[1], 4)),
        "elapsed": round(elapsed, 2),
    }
    print(f"[matcha-01] UV calibration: {uv_info}")

    u8 = np.round(np.clip(final_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    full_im = Image.fromarray(u8, mode="RGB")

    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        full_im.save(audit_dir / "w01_full.png", optimize=True)
        crops = {
            # 1: Left arched glass window ray origin, wrought-iron mullions, ivy & upper skylight shafts
            "w01_crop_left_window_rays.png": (0, int(H * 0.05)),
            # 2: Cedar potting bench sun-hit patches, copper watering can, propagation jars & monstera
            "w01_crop_bench_watering_can.png": (int(W * 0.04), int(H * 0.48)),
            # 3: Aisle wet Voronoi slate puddles, sunbeam landings & hanging ferns
            "w01_crop_aisle_puddles_ferns.png": (int(W * 0.34), int(H * 0.44)),
            # 4: Upper vaulted glass skylight ribs, hanging fern baskets & brass pendant lamp
            "w01_crop_vault_skylight_ferns.png": (int(W * 0.26), int(H * 0.02)),
        }
        for cname, (cx0, cy0) in crops.items():
            x0 = max(0, min(W - 600, cx0))
            y0 = max(0, min(H - 450, cy0))
            full_im.crop((x0, y0, x0 + 600, y0 + 450)).save(audit_dir / cname)
            print(f"[matcha-01] Wrote audit crop {audit_dir / cname} ({x0},{y0})")

    if export_webps:
        for wp in export_webps:
            wp.parent.mkdir(parents=True, exist_ok=True)
            full_im.save(wp, format="WEBP", quality=94, method=6)
            print(f"[matcha-01] Exported {wp} ({wp.stat().st_size / 1024:.1f} KiB)")

    return final_rgb, uv_info


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    default_webps = [
        repo_root / "session" / "themes" / "matcha" / "walls" / "wall-01.webp",
        repo_root / "docs" / "assets" / "walls" / "matcha-01.webp",
    ]
    ap = argparse.ArgumentParser(description='Render Matcha 01 — "Sunlit Botanical Conservatory" (2048x1152)')
    ap.add_argument("--seed", type=int, default=20261125)
    ap.add_argument("--audit-dir", type=pathlib.Path, default=None)
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()
    render_scene(seed=args.seed, audit_dir=args.audit_dir, export_webps=None if args.no_export else default_webps)


if __name__ == "__main__":
    main()

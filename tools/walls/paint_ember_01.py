#!/usr/bin/env python3
"""
Ember 01 ("Crimson Viaduct & Seaside Tram Crossing") — 1990s Theatrical Cel-Anime Pipeline (2048 x 1152)

Five-stage deterministic pure-Python painting pipeline (zero image models, zero external assets):
  Stage 1: Metric right-handed Camera3D (off-axis vanishing point ~60.1% width,
           horizon ~45.7% height), exact 3x3 UV-to-screen plane homographies masked
           by valid UV and near-plane depth z_cam > 0.05 with exponential twilight
           haze, 3D segment clipping (z_cam >= 0.05), metric jittered 3x3 Voronoi
           (d2 - d1) wet cobblestone lane, stone sidewalk curb, weathered stone
           coastal parapet & waterfront overlook balustrade, left ochre-plaster &
           smoked-cedar corner bakery / railway signal house with pitched kawara
           tile eaves, and a monumental angled multi-arch ashlar stone railway viaduct.
  Stage 2: Single locked normalized low-sunset SUN_DIR (z < 0) unified across the
           blazing setting sun inside Arch[1], 6-step height-refined pitched eave &
           viaduct shadow tracing, crossing-gate/pole/parapet cast shadows,
           volumetric golden sunbeams cutting under the stone viaduct arches, and dual
           Fresnel capillary-rippled reflections (open vermilion sunset sea + broad
           wet cobblestone rainwater puddle) combining cv2.remap mirrored sky/clouds
           with exact 3D plane-mirrored viaduct piers, tram glow, crossing lamps,
           utility pole, and shopfront light.
  Stage 3: 4 depth-interleaved 2x supersampled PyCairo bands (Lanczos-downsampled,
           cleanly unpremultiplied):
             - Band 0: Fiery dusk sky, 6-step Beer-Lambert optical-marched backlit
               billowing cumulus & high cirrus clouds, distant volcanic headland
               silhouette, and molten-copper sunset ocean bay.
             - Band 1: Monumental multi-arch stone viaduct with recessed 3D barrel
               vaults, radiating keystone voussoirs, catenary steel masts, sweeping
               cubic-Bezier power wires, and a richly detailed retro cream-and-vermilion
               Japanese coastal tram car (Enoden style) with warmly glowing interior
               windows, commuter silhouettes, roof vents, curved cab & pantograph.
             - Band 2: Left corner bakery/signal-box 3D craftsmanship, 2x PyCairo
               Japanese signage, detailed vintage mamachari bicycle with woven basket
               & fenders, slatted wooden bakery crates, lush terracotta herb pots,
               iconic yellow-and-black striped Japanese fumikiri railway crossing post
               with X-crossbuck, train-direction arrow, hooded twin glowing red-amber
               warning lamps & counterweighted barrier boom, flush steel rails, dark
               timber utility pole with cooling-fin transformer, climbing pegs, porcelain
               insulators & sweeping catenary wires, and rusted coastal guardrail.
             - Band 3: Organic wind-swept Japanese coastal black pine (kuromatsu)
               tiered boughs & natural trailing autumn ivy vines rendered purely with
               seeded cubic Bezier curves, finished with balanced isotropic fbm_gouache,
               flat-brush dabs, cold-press paper tooth, and warm shadow bounce (lum < 0.28).
  Stage 4: Near-camera foreground translucent autumn vine & rusted railway guardrail
           silhouette convolved in premultiplied RGBA with circular disc bokeh
           (radius = 7.8) + separate cv2.warpAffine ~1 px opposite R/B lateral shift,
           followed by 35mm 3-scale highlight halation (sigma = 6.5, 24, 72), radial
           corner CA, optical vignette, shadow black lift + C1 tanh highlight shoulder,
           luminance-weighted silver-halide grain, and faint YIQ tape patina.
  Stage 5: Deterministic render (<60s CPU), byte-identical WebP quality=94 exports,
           1:1 actual-pixel audit crops, and printed Camera3D UV calibration map for
           session/themes/ember/walls/wall-01.frag.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import shutil
import time
from typing import Iterable

import cairo
import cv2
import numpy as np
from PIL import Image

W, H = 2048, 1152
Z_NEAR = 0.05

# =============================================================================
# Scene Metric World Constants (Right-handed meters: +X right, +Y forward, +Z up)
# =============================================================================
Z_LANE = 0.0
Z_CURB = 0.14
Z_SEA = -1.80

# Lane bounds & coastal parapet + far waterfront overlook terrace wall
LANE_X0 = -3.15
CURB_X1 = -2.02
PARAPET_X0 = 2.66
PARAPET_X1 = 3.16
PARAPET_Z = 0.86
LANE_Y0 = 3.2
LANE_Y1 = 33.4

# Left-side corner seaside bakery-shop & railway signal box
SHOP_X_WALL = -3.15
SHOP_X_EAVE = -1.96
SHOP_Y0 = 3.80
SHOP_Y1 = 15.60
SHOP_Z_EAVE = 4.55
SHOP_ROOF_SLOPE = 0.48
SHOP_AWNING_Z0 = 2.66
SHOP_AWNING_Z1 = 2.34
SHOP_AWNING_X1 = -2.26

# Stepped rear signal-room annex linking toward viaduct embankment
ANNEX_X_WALL = -3.45
ANNEX_X_EAVE = -2.42
ANNEX_Y0 = 15.60
ANNEX_Y1 = 20.60
ANNEX_Z_EAVE = 3.55
ANNEX_ROOF_SLOPE = 0.42

# Elevated Multi-Arch Stone Railway Viaduct (angled 14 deg across upper midground)
VD_ANGLE = math.radians(14.0)
VD_DIR = np.array([math.cos(VD_ANGLE), math.sin(VD_ANGLE), 0.0], dtype=np.float64)
VD_NORM = np.array([-math.sin(VD_ANGLE), math.cos(VD_ANGLE), 0.0], dtype=np.float64)
VD_ORIG = np.array([-4.0, 35.5, 0.0], dtype=np.float64)
VD_THICK = 3.40
VD_SPAN = 9.20
VD_PIER_W = 2.15
VD_ARCH_R = 0.5 * (VD_SPAN - VD_PIER_W)
VD_Z_SPRING = 1.45
VD_Z_CROWN = VD_Z_SPRING + VD_ARCH_R * 0.96
VD_Z_DECK = 6.00
VD_Z_TRACK = 5.65
VD_TRAM_Z0 = 5.92
VD_TRAM_Z1 = 8.82
VD_WIRE_Z = 10.12

# Seaside Tram Crossing (Fumikiri) in midground lane
FUMI_X = -1.82
FUMI_Y = 14.55
FUMI_LAMP_Z = 2.48
FUMI_LAMP_L = (FUMI_X - 0.22, FUMI_Y, FUMI_LAMP_Z)
FUMI_LAMP_R = (FUMI_X + 0.22, FUMI_Y, FUMI_LAMP_Z)
FUMI_BARRIER_Z = 0.96
FUMI_BARRIER_X1 = 1.66

# Flush seaside spur tram rails crossing the cobbles behind the crossing barrier
RAIL_Y_L = 16.05
RAIL_Y_R = 17.55
RAIL_GAUGE = 1.067  # Japanese Cape Gauge (Enoden)

# Tall dark timber utility pole on left curb
POLE_X = -2.24
POLE_Y = 10.65
POLE_Z_TOP = 6.85

# Warm light sources (x, y, z, intensity) for unified ground/facade illumination
SHOP_LIGHTS = [
    (-2.95, 7.80, 1.85, 1.38),
    (-2.95, 11.45, 1.65, 1.60),
    (-2.95, 13.75, 1.85, 1.25),
    (-2.95, 9.60, 3.65, 0.85),
    (-2.54, 11.45, 2.14, 1.28),
]

# Single locked normalized low-sunset SUN_DIR (z < 0):
# Projects cleanly inside Viaduct Arch[1] at u ~ 0.789, v ~ 0.401
_raw_sun = np.array([-0.190, -0.9812, -0.032], dtype=np.float64)
SUN_DIR = _raw_sun / np.linalg.norm(_raw_sun)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_THEME_WEBP = REPO_ROOT / "session" / "themes" / "ember" / "walls" / "wall-01.webp"
DEFAULT_DOCS_WEBP = REPO_ROOT / "docs" / "assets" / "walls" / "ember-01.webp"


# =============================================================================
# Vectorized Core Painting & DSP Primitives
# =============================================================================

def resize_bicubic(arr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    return cv2.resize(arr.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_CUBIC)


def blur_2d(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.08:
        return arr.copy()
    if sigma <= 14.0:
        return cv2.GaussianBlur(arr.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma,
                                borderType=cv2.BORDER_REFLECT_101)
    h, w = arr.shape[:2]
    factor = 4 if sigma < 42.0 else 8
    sw, sh = max(8, w // factor), max(8, h // factor)
    small = cv2.resize(arr.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma / factor, sigmaY=sigma / factor,
                             borderType=cv2.BORDER_REFLECT_101)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def blur_1d_x(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.05:
        return arr.copy()
    k = max(3, int(math.ceil(sigma * 3.0)) * 2 + 1)
    return cv2.GaussianBlur(arr.astype(np.float32), (k, 1), sigmaX=sigma, sigmaY=0.0,
                            borderType=cv2.BORDER_REFLECT_101)


def blur_1d_y(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.05:
        return arr.copy()
    k = max(3, int(math.ceil(sigma * 3.0)) * 2 + 1)
    return cv2.GaussianBlur(arr.astype(np.float32), (1, k), sigmaX=0.0, sigmaY=sigma,
                            borderType=cv2.BORDER_REFLECT_101)


def fbm_gouache(
    h: int,
    w: int,
    rng: np.random.Generator,
    scales=((72, 68), (26, 29), (10.5, 9.5), (4.5, 4.8)),
) -> np.ndarray:
    """Isotropic square-scale gouache pigment mottle (zero directional streaks!)."""
    acc = np.zeros((h, w), dtype=np.float32)
    weights = (0.46, 0.28, 0.17, 0.09)
    for (sy, sx), wt in zip(scales, weights):
        gh = max(4, int(round(h / sy)))
        gw = max(4, int(round(w / sx)))
        acc += wt * resize_bicubic(rng.standard_normal((gh, gw)).astype(np.float32), w, h)
    acc -= float(acc.mean())
    return acc / (float(acc.std()) + 1e-6)


def paper_tooth(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Fine isotropic cold-press watercolor paper tooth."""
    coarse = resize_bicubic(rng.standard_normal((max(4, h // 3), max(4, w // 3))).astype(np.float32), w, h)
    fine = rng.standard_normal((h, w)).astype(np.float32)
    t = 0.56 * coarse + 0.44 * fine
    t -= float(t.mean())
    return t / (float(t.std()) + 1e-6)


def flat_brush_dabs(h: int, w: int, rng: np.random.Generator, count: int = 94) -> np.ndarray:
    """85-100 angled soft convex-polygon flat-brush gouache dabs at 1/4 resolution."""
    qh, qw = max(16, h // 4), max(16, w // 4)
    arr_sum = np.zeros((qh, qw), dtype=np.float32)
    s_one = cairo.ImageSurface(cairo.FORMAT_A8, qw, qh)
    c1 = cairo.Context(s_one)
    stride = s_one.get_stride()
    for _ in range(count):
        cx = float(rng.uniform(0, qw))
        cy = float(rng.uniform(0, qh))
        ang = float(rng.uniform(0, math.pi))
        rx = float(rng.uniform(qw * 0.026, qw * 0.092))
        ry = float(rng.uniform(rx * 0.30, rx * 0.64))
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
            a = (2.0 * math.pi * k) / n_pts + float(rng.uniform(-0.20, 0.20))
            rj = float(rng.uniform(0.78, 1.15))
            pts.append((math.cos(a) * rx * rj, math.sin(a) * ry * rj))
        c1.move_to(*pts[0])
        for p in pts[1:]:
            c1.line_to(*p)
        c1.close_path()
        c1.set_source_rgba(1, 1, 1, 1)
        c1.fill()
        c1.restore()
        s_one.flush()
        buf = np.frombuffer(s_one.get_data(), dtype=np.uint8).reshape(qh, stride)[:, :qw].astype(np.float32) * (1.0 / 255.0)
        arr_sum += strength * buf
    arr_sum = blur_2d(arr_sum, 3.0)
    arr_sum -= float(arr_sum.mean())
    arr_sum /= float(arr_sum.std()) + 1e-6
    return resize_bicubic(arr_sum, w, h)


def over_premul(dst_rgb: np.ndarray, src_premul_rgba: np.ndarray) -> np.ndarray:
    a = np.clip(src_premul_rgba[:, :, 3:4], 0.0, 1.0)
    return src_premul_rgba[:, :, :3] + dst_rgb * (1.0 - a)


def apply_warm_shadow_bounce(rgb: np.ndarray, strength: float = 0.022) -> np.ndarray:
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    mask = np.clip((0.28 - lum) / 0.28, 0.0, 1.0)[..., None]
    bounce = np.array([1.28, 0.54, 0.28], dtype=np.float32) * strength
    return rgb + mask * bounce


def new_cairo_2x() -> tuple[cairo.ImageSurface, cairo.Context]:
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W * 2, H * 2)
    ctx = cairo.Context(surf)
    ctx.scale(2.0, 2.0)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    return surf, ctx


def finish_cairo_2x(surf: cairo.ImageSurface) -> np.ndarray:
    """Convert 2x Cairo premul BGRA -> 1x premul float32 RGBA via Lanczos downsample."""
    surf.flush()
    buf = np.frombuffer(surf.get_data(), dtype=np.uint8).reshape(H * 2, W * 2, 4)
    rgba_2x = np.empty((H * 2, W * 2, 4), dtype=np.float32)
    rgba_2x[:, :, 0] = buf[:, :, 2] * (1.0 / 255.0)
    rgba_2x[:, :, 1] = buf[:, :, 1] * (1.0 / 255.0)
    rgba_2x[:, :, 2] = buf[:, :, 0] * (1.0 / 255.0)
    rgba_2x[:, :, 3] = buf[:, :, 3] * (1.0 / 255.0)
    out = cv2.resize(rgba_2x, (W, H), interpolation=cv2.INTER_LANCZOS4)
    out[:, :, 3] = np.clip(out[:, :, 3], 0.0, 1.0)
    out[:, :, :3] = np.clip(out[:, :, :3], 0.0, out[:, :, 3:4])
    return out


def fog_mix(rgb: tuple[float, float, float], fog_rgb: np.ndarray, z_dist: float,
            fog_start: float = 8.5, fog_density: float = 0.023) -> tuple[float, float, float]:
    ft = 1.0 - math.exp(-max(0.0, z_dist - fog_start) * fog_density)
    return (
        rgb[0] * (1.0 - ft) + float(fog_rgb[0]) * ft,
        rgb[1] * (1.0 - ft) + float(fog_rgb[1]) * ft,
        rgb[2] * (1.0 - ft) + float(fog_rgb[2]) * ft,
    )


def stroke_catmull_rom(
    ctx: cairo.Context,
    pts: list[tuple[float, float]],
    base_w: float,
    rgba: tuple[float, float, float, float],
    rng: np.random.Generator,
    taper: bool = False,
    tremor: float = 0.25,
) -> None:
    if len(pts) < 2:
        return
    p = [pts[0]] + list(pts) + [pts[-1]]
    ctx.set_source_rgba(*rgba)
    n_seg = len(pts) - 1
    for i in range(n_seg):
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        t_frac = (i + 0.5) / max(1, n_seg)
        w_cur = base_w * (1.0 - 0.68 * t_frac) if taper else base_w
        jit = float(rng.uniform(-tremor, tremor))
        c1x = p1[0] + (p2[0] - p0[0]) / 6.0 + jit
        c1y = p1[1] + (p2[1] - p0[1]) / 6.0 + jit
        c2x = p2[0] - (p3[0] - p1[0]) / 6.0 - jit
        c2y = p2[1] - (p3[1] - p1[1]) / 6.0 - jit
        ctx.set_line_width(max(0.45, w_cur))
        ctx.move_to(p1[0], p1[1])
        ctx.curve_to(c1x, c1y, c2x, c2y, p2[0], p2[1])
        ctx.stroke()


# =============================================================================
# Stage 1 — Metric Right-Handed Camera3D & Exact 3x3 Plane Homographies
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

    def project_pt(self, pt3: Iterable[float]) -> tuple[float, float, float]:
        u, v, z = self.project(np.asarray(pt3, dtype=np.float64))
        return float(u), float(v), float(z)

    def project_uv_norm(self, pt3: Iterable[float]) -> tuple[float, float, float]:
        u, v, z = self.project_pt(pt3)
        return u / self.w, v / self.h, z

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

    def clip_poly_3d(self, pts: Iterable[Iterable[float]], z_near: float = Z_NEAR) -> list[tuple[float, float]]:
        poly = [np.asarray(p, dtype=np.float64) for p in pts]
        if len(poly) < 3:
            return []
        out: list[np.ndarray] = []
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            za = float((self.R @ (a - self.eye))[2])
            zb = float((self.R @ (b - self.eye))[2])
            in_a, in_b = za >= z_near, zb >= z_near
            if in_a and in_b:
                out.append(b)
            elif in_a and not in_b:
                t = (z_near - za) / (zb - za)
                out.append(a + t * (b - a))
            elif not in_a and in_b:
                t = (z_near - za) / (zb - za)
                out.append(a + t * (b - a))
                out.append(b)
        if len(out) < 3:
            return []
        res = []
        for p in out:
            u, v, _ = self.project_pt(p)
            res.append((u, v))
        return res

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
        fog_start: float = 8.5,
        fog_density: float = 0.023,
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
                sx, sy, _ = self.project_pt(pt)
                xs.append(sx)
                ys.append(sy)
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

        u_cl = np.clip(u_px, 0.0, tw - 1.001).astype(np.float32)
        v_cl = np.clip(v_px, 0.0, th - 1.001).astype(np.float32)
        sampled = cv2.remap(tex_premul_rgba, u_cl, v_cl, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)

        v_f = valid[..., None].astype(np.float32)
        alpha = sampled[:, :, 3:4] * v_f
        rgb_valid = sampled[:, :, :3] * v_f

        fg_rgb = rgb_valid * (1.0 - fog_t[..., None]) + (fog_rgb[None, None, :] * alpha) * fog_t[..., None]
        patch = np.concatenate([fg_rgb, alpha], axis=-1)
        dst_rgb[y0:y1, x0:x1] = over_premul(dst_rgb[y0:y1, x0:x1], patch)
        return dst_rgb


def vd_point(s: float, d: float = 0.0, z: float = 0.0) -> np.ndarray:
    return VD_ORIG + s * VD_DIR + d * VD_NORM + np.array([0.0, 0.0, z], dtype=np.float64)


def mirror_z_pt(pt: Iterable[float], z_plane: float) -> np.ndarray:
    p = np.asarray(pt, dtype=np.float64).copy()
    p[2] = 2.0 * z_plane - p[2]
    return p


def fill_3d_poly(ctx: cairo.Context, cam: Camera3D, pts3d: Iterable[Iterable[float]],
                 rgba: tuple[float, float, float, float], mirror_z: float | None = None) -> list[tuple[float, float]]:
    if mirror_z is not None:
        pts3d = [mirror_z_pt(p, mirror_z) for p in pts3d]
    poly2 = cam.clip_poly_3d(pts3d)
    if len(poly2) < 3:
        return []
    ctx.set_source_rgba(*rgba)
    ctx.move_to(*poly2[0])
    for p in poly2[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    ctx.fill()
    return poly2


def fill_3d_poly_grad(
    ctx: cairo.Context,
    cam: Camera3D,
    pts3d: Iterable[Iterable[float]],
    p_g0: Iterable[float],
    p_g1: Iterable[float],
    stops: list[tuple[float, float, float, float, float]],
    mirror_z: float | None = None,
) -> list[tuple[float, float]]:
    if mirror_z is not None:
        pts3d = [mirror_z_pt(p, mirror_z) for p in pts3d]
        p_g0 = mirror_z_pt(p_g0, mirror_z)
        p_g1 = mirror_z_pt(p_g1, mirror_z)
    poly2 = cam.clip_poly_3d(pts3d)
    if len(poly2) < 3:
        return []
    u0, v0, _ = cam.project_pt(p_g0)
    u1, v1, _ = cam.project_pt(p_g1)
    g = cairo.LinearGradient(u0, v0, u1, v1)
    for st in stops:
        g.add_color_stop_rgba(*st)
    ctx.set_source(g)
    ctx.move_to(*poly2[0])
    for p in poly2[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    ctx.fill()
    return poly2


def stroke_3d_line(ctx: cairo.Context, cam: Camera3D, p0: Iterable[float], p1: Iterable[float],
                   width_m: float, rgba: tuple[float, float, float, float],
                   mirror_z: float | None = None, min_px: float = 0.8) -> None:
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    if mirror_z is not None:
        a = mirror_z_pt(a, mirror_z)
        b = mirror_z_pt(b, mirror_z)
    cl = cam.clip_segment_3d(a, b)
    if cl is None:
        return
    ua, va, za = cam.project_pt(cl[0])
    ub, vb, zb = cam.project_pt(cl[1])
    z_mid = max(0.5, 0.5 * (za + zb))
    px_w = max(min_px, float(cam.K[0, 0] * width_m / z_mid))
    ctx.set_source_rgba(*rgba)
    ctx.set_line_width(px_w)
    ctx.move_to(ua, va)
    ctx.line_to(ub, vb)
    ctx.stroke()


# =============================================================================
# Stage 2 — Unified 3D SUN_DIR Shadow Solver & Shop/Lamp Illumination
# =============================================================================

def trace_eave_and_awning_shadow(wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> np.ndarray:
    """6-step height-refined ray trace backward along -SUN_DIR against pitched roofs & awning."""
    inv_sz = np.float32(1.0 / (-SUN_DIR[2]))
    sx = np.float32(SUN_DIR[0])
    sy = np.float32(SUN_DIR[1])

    z_roof = np.full_like(wx, SHOP_Z_EAVE, dtype=np.float32)
    for _ in range(6):
        t = np.maximum(0.0, (z_roof - wz) * inv_sz)
        rx = wx - sx * t
        z_roof = SHOP_Z_EAVE + np.maximum(0.0, (SHOP_X_EAVE - rx)) * np.float32(SHOP_ROOF_SLOPE)
    t = np.maximum(0.0, (z_roof - wz) * inv_sz)
    rx = wx - sx * t
    ry = wy - sy * t
    under_main = (wz < z_roof - 0.02) & (rx >= -7.8) & (rx <= SHOP_X_EAVE) & (ry >= SHOP_Y0) & (ry <= SHOP_Y1)
    edge_main = np.clip((rx - (SHOP_X_EAVE - 0.22)) / 0.22, 0.0, 1.0)
    vis = np.where(under_main, 0.24 + 0.76 * edge_main, 1.0).astype(np.float32)

    z_anx = np.full_like(wx, ANNEX_Z_EAVE, dtype=np.float32)
    for _ in range(6):
        ta = np.maximum(0.0, (z_anx - wz) * inv_sz)
        rxa = wx - sx * ta
        z_anx = ANNEX_Z_EAVE + np.maximum(0.0, (ANNEX_X_EAVE - rxa)) * np.float32(ANNEX_ROOF_SLOPE)
    ta = np.maximum(0.0, (z_anx - wz) * inv_sz)
    rxa = wx - sx * ta
    rya = wy - sy * ta
    under_anx = (wz < z_anx - 0.02) & (rxa >= -7.8) & (rxa <= ANNEX_X_EAVE) & (rya >= ANNEX_Y0) & (rya <= ANNEX_Y1)
    edge_anx = np.clip((rxa - (ANNEX_X_EAVE - 0.18)) / 0.18, 0.0, 1.0)
    vis = np.minimum(vis, np.where(under_anx, 0.26 + 0.74 * edge_anx, 1.0).astype(np.float32))

    aw_slope = (SHOP_AWNING_Z0 - SHOP_AWNING_Z1) / (SHOP_AWNING_X1 - SHOP_X_WALL)
    z_aw = np.full_like(wx, SHOP_AWNING_Z1, dtype=np.float32)
    for _ in range(4):
        tw = np.maximum(0.0, (z_aw - wz) * inv_sz)
        rxw = wx - sx * tw
        z_aw = SHOP_AWNING_Z1 + np.maximum(0.0, (SHOP_AWNING_X1 - rxw)) * np.float32(aw_slope)
    tw = np.maximum(0.0, (z_aw - wz) * inv_sz)
    rxw = wx - sx * tw
    ryw = wy - sy * tw
    under_aw = (wz < z_aw - 0.015) & (rxw >= SHOP_X_WALL - 0.05) & (rxw <= SHOP_AWNING_X1) & (ryw >= 4.4) & (ryw <= 14.8)
    edge_aw = np.clip((rxw - (SHOP_AWNING_X1 - 0.14)) / 0.14, 0.0, 1.0)
    vis = np.minimum(vis, np.where(under_aw, 0.24 + 0.76 * edge_aw, 1.0).astype(np.float32))
    return vis


def trace_viaduct_shadow(wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> np.ndarray:
    """Trace backward along -SUN_DIR through multi-arch stone viaduct piers & arches."""
    nx, ny = np.float32(VD_NORM[0]), np.float32(VD_NORM[1])
    dx_v, dy_v = np.float32(VD_DIR[0]), np.float32(VD_DIR[1])
    rx_d = np.float32(-SUN_DIR[0])
    ry_d = np.float32(-SUN_DIR[1])
    rz_d = np.float32(-SUN_DIR[2])
    denom = rx_d * nx + ry_d * ny
    if abs(float(denom)) < 1e-4:
        return np.ones_like(wx, dtype=np.float32)

    px = wx - np.float32(VD_ORIG[0])
    py = wy - np.float32(VD_ORIG[1])
    dist_n = px * nx + py * ny
    vis = np.ones_like(wx, dtype=np.float32)

    for d_off in (0.0, VD_THICK * 0.65):
        t = (np.float32(d_off) - dist_n) / np.float32(denom)
        hx = px + rx_d * t
        hy = py + ry_d * t
        hz = wz + rz_d * t
        s = hx * dx_v + hy * dy_v

        s_mod = np.mod(s + np.float32(VD_PIER_W * 0.5), np.float32(VD_SPAN))
        s_arch = s_mod - np.float32(VD_PIER_W + VD_ARCH_R)
        rad_ratio = np.clip(s_arch / np.float32(VD_ARCH_R), -1.0, 1.0)
        z_arch = np.float32(VD_Z_SPRING) + np.float32(VD_ARCH_R * 0.96) * np.sqrt(np.maximum(0.0, 1.0 - rad_ratio * rad_ratio))

        in_pier = s_mod < np.float32(VD_PIER_W)
        in_spandrel = (~in_pier) & (hz > z_arch)
        hit_solid = (t > 0.05) & (hz >= np.float32(Z_SEA - 0.5)) & (hz <= np.float32(VD_Z_DECK + 0.15)) & (in_pier | in_spandrel)

        soft_pier = np.clip((s_mod - np.float32(VD_PIER_W - 0.25)) / 0.25, 0.0, 1.0)
        soft_arch = np.clip((z_arch - hz + 0.25) / 0.25, 0.0, 1.0)
        sh = np.where(in_pier, 0.30 + 0.70 * soft_pier, np.where(in_spandrel, 0.32 + 0.68 * (1.0 - soft_arch), 1.0))
        vis = np.minimum(vis, np.where(hit_solid, sh, 1.0).astype(np.float32))
    return vis


def trace_lane_shadows(wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> np.ndarray:
    """Unified cast shadows on wet cobblestone lane from eaves, viaduct, parapet, crossing post & pole."""
    vis = trace_eave_and_awning_shadow(wx, wy, wz)
    vis = np.minimum(vis, trace_viaduct_shadow(wx, wy, wz))

    sx = np.float32(-SUN_DIR[0])
    sy = np.float32(-SUN_DIR[1])
    sz = np.float32(-SUN_DIR[2])
    s2 = sx * sx + sy * sy
    inv_s = np.float32(1.0 / math.sqrt(float(s2)))

    t_par = (np.float32(PARAPET_X0) - wx) / np.maximum(np.float32(1e-4), sx)
    z_par = wz + t_par * sz
    y_par = wy + t_par * sy
    in_par = (t_par > 0.0) & (wx <= np.float32(PARAPET_X0 + 0.02)) & (z_par <= np.float32(PARAPET_Z)) & (y_par >= 3.0) & (y_par <= LANE_Y1)
    edge_par = np.clip((z_par - np.float32(PARAPET_Z - 0.14)) / 0.14, 0.0, 1.0)
    vis = np.minimum(vis, np.where(in_par, 0.34 + 0.66 * edge_par, 1.0).astype(np.float32))

    for px, py, ph, rad in (
        (FUMI_X, FUMI_Y, 3.45, 0.14),
        (POLE_X, POLE_Y, 6.80, 0.18),
        (2.32, 16.95, 2.45, 0.11),
    ):
        dx = np.float32(px) - wx
        dy = np.float32(py) - wy
        t_h = (dx * sx + dy * sy) / s2
        perp = np.abs(dx * sy - dy * sx) * inv_s
        z_hit = wz + t_h * sz
        in_p = (t_h > 0.0) & (z_hit >= 0.0) & (z_hit <= np.float32(ph)) & (perp < np.float32(rad * 1.45))
        soft = np.clip((perp - np.float32(rad * 0.72)) / np.float32(rad * 0.73), 0.0, 1.0)
        vis = np.minimum(vis, np.where(in_p, 0.36 + 0.64 * soft, 1.0).astype(np.float32))

    t_bar = (np.float32(FUMI_BARRIER_Z) - wz) / np.maximum(np.float32(1e-4), sz)
    bx_hit = wx + t_bar * sx
    by_hit = wy + t_bar * sy
    in_bar = (t_bar > 0.0) & (bx_hit >= FUMI_X) & (bx_hit <= FUMI_BARRIER_X1) & (np.abs(by_hit - FUMI_Y) < 0.11)
    vis = np.minimum(vis, np.where(in_bar, 0.44, 1.0).astype(np.float32))

    curb_ao = np.clip((wx - np.float32(CURB_X1)) / 0.45, 0.0, 1.0)
    par_ao = np.clip((np.float32(PARAPET_X0) - wx) / 0.45, 0.0, 1.0)
    return vis * (0.58 + 0.42 * curb_ao) * (0.64 + 0.36 * par_ao)


def eval_shop_and_crossing_glow(wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    amber = np.zeros_like(wx, dtype=np.float32)
    for lx, ly, lz, inten in SHOP_LIGHTS:
        d2 = (wx - lx) ** 2 + (wy - ly) ** 2 + (wz - lz) ** 2
        amber += np.float32(inten) / (1.0 + 0.34 * d2)
    amber = np.clip(amber, 0.0, 2.0)

    crimson = np.zeros_like(wx, dtype=np.float32)
    for lx, ly, lz in (FUMI_LAMP_L, FUMI_LAMP_R):
        d2 = (wx - lx) ** 2 + (wy - ly) ** 2 + (wz - lz) ** 2
        crimson += np.float32(0.88) / (1.0 + 0.40 * d2)
    crimson = np.clip(crimson, 0.0, 1.5)
    return amber, crimson


# =============================================================================
# Stage 3 Band 0 — Sunset Sky, Beer-Lambert Backlit Clouds, Headland & Copper Sea
# =============================================================================

def paint_sculptural_sunset_clouds(
    hv: float,
    sun_u: float,
    sun_v: float,
    mottle: np.ndarray,
    dabs: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Grand theatrical Shinkai / Ghibli sunset cumulonimbus towers & feathered
    high cirrus bands with incandescent cream-gold silver linings, luminous
    coral-vermilion bellies, and dramatic obsidian-plum crowns!
    """
    ch, cw = H // 2, W // 2
    yy_c, xx_c = np.mgrid[0:ch, 0:cw].astype(np.float32)
    uc = xx_c / cw
    vc = yy_c / ch

    wx_n = resize_bicubic(fbm_gouache(ch // 2, cw // 2, rng, scales=((44, 42), (18, 19), (8.2, 8.0), (3.8, 4.0))), cw, ch)
    wy_n = resize_bicubic(fbm_gouache(ch // 2, cw // 2, rng, scales=((44, 42), (18, 19), (8.2, 8.0), (3.8, 4.0))), cw, ch)
    fine_edge = resize_bicubic(fbm_gouache(ch // 2, cw // 2, rng, scales=((20, 20), (9.5, 9.5), (4.8, 4.8), (2.4, 2.5))), cw, ch)

    uw = uc + wx_n * 0.0170
    vw = vc + wy_n * 0.0150

    # Layer A: Sweeping high-altitude feathered cirrus streaming diagonally toward sunset
    cirrus = np.zeros((ch, cw), dtype=np.float32)
    for cu0, cv0, rx_c, ry_c, ang_c, wt_c in (
        (0.55, 0.048, 0.30, 0.020, -0.10, 0.76),
        (0.76, 0.082, 0.25, 0.019, -0.13, 0.72),
        (0.42, 0.112, 0.26, 0.023, -0.09, 0.68),
        (0.67, 0.145, 0.24, 0.020, -0.12, 0.70),
        (0.88, 0.165, 0.20, 0.018, -0.14, 0.60),
    ):
        ca, sa = math.cos(ang_c), math.sin(ang_c)
        du, dv = uw - cu0, vw - cv0
        ru = (du * ca - dv * sa) / rx_c
        rv = (du * sa + dv * ca) / ry_c
        cirrus = np.maximum(cirrus, np.exp(-(ru * ru + rv * rv)) * wt_c)
    cirrus = np.clip((cirrus + fine_edge * 0.11 - 0.16) * 1.52, 0.0, 0.90)
    cirrus *= np.clip((hv - vc - 0.04) / 0.14, 0.0, 1.0)

    # Layer B: Towering sculpted cauliflower cumulonimbus banks behind & above the tram!
    cumulus = np.zeros((ch, cw), dtype=np.float32)
    puffs = [
        # Luminous billowing main tower rising directly behind the Enoden tram so tram silhouettes dramatically!
        (0.42, 0.155, 0.110, 0.068, 1.08),
        (0.50, 0.105, 0.098, 0.072, 1.16),
        (0.58, 0.135, 0.105, 0.065, 1.12),
        (0.66, 0.168, 0.095, 0.056, 1.06),
        (0.37, 0.218, 0.112, 0.054, 0.98),
        (0.48, 0.222, 0.116, 0.052, 1.06),
        (0.60, 0.215, 0.102, 0.052, 1.02),
        (0.73, 0.218, 0.088, 0.046, 0.96),
        # Sweeping sunset bank far right framing the glowing solar corridor
        (0.84, 0.118, 0.112, 0.064, 1.06),
        (0.92, 0.168, 0.108, 0.062, 1.08),
        (0.81, 0.215, 0.088, 0.045, 0.92),
        (0.91, 0.255, 0.102, 0.048, 0.96),
        # Luminous low golden stratus streaks behind viaduct arches
        (0.56, 0.355, 0.135, 0.018, 0.72),
        (0.84, 0.348, 0.125, 0.020, 0.74),
    ]
    for cu0, cv0, rx_c, ry_c, wt_c in puffs:
        d2 = ((uw - cu0) / rx_c) ** 2 + ((vw - cv0) / ry_c) ** 2
        cumulus = np.maximum(cumulus, np.exp(-d2 * 1.10) * wt_c)
    cumulus = np.clip((cumulus + fine_edge * 0.16 + wx_n * 0.08 - 0.24) * 1.56, 0.0, 1.28)
    cumulus *= np.clip((hv - vc - 0.008) / 0.062, 0.0, 1.0)

    density = np.clip(cumulus + 0.60 * cirrus, 0.0, 1.35)

    # 6-step Beer-Lambert optical marching toward setting sun for crisp golden silver linings!
    du_s = sun_u - uc
    dv_s = sun_v - vc
    dist_s = np.hypot(du_s, dv_s) + 1e-5
    step_u = (du_s / dist_s) * 0.0112
    step_v = (dv_s / dist_s) * 0.0112
    opt_depth = np.zeros_like(density)
    for k in range(1, 7):
        su_k = np.clip((uc + step_u * k) * (cw - 1), 0.0, cw - 1.001).astype(np.float32)
        sv_k = np.clip((vc + step_v * k) * (ch - 1), 0.0, ch - 1.001).astype(np.float32)
        opt_depth += cv2.remap(density, su_k, sv_k, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) * (0.26 / (0.65 + 0.25 * k))

    trans = np.exp(-opt_depth * 1.95)
    silver_rim = np.clip(np.exp(-opt_depth * 3.8) * np.clip(density * 2.5, 0.0, 1.0) * (1.0 - np.clip(density - 0.58, 0.0, 1.0)), 0.0, 1.0)
    sun_prox = np.exp(-((uc - sun_u) / 0.44) ** 2 - ((vc - sun_v) / 0.30) ** 2)

    # Rich luminous theatrical Ghibli/Shinkai sunset cloud palette!
    col_crown = np.array([0.22, 0.102, 0.158], dtype=np.float32)
    col_mid = np.array([0.66, 0.225, 0.175], dtype=np.float32)
    col_belly = np.array([1.02, 0.52, 0.28], dtype=np.float32)
    col_rim = np.array([1.12, 0.92, 0.68], dtype=np.float32)

    alt_t = np.clip((hv - vc) / max(0.05, hv), 0.0, 1.0)
    lit = np.clip(trans * (0.50 + 0.62 * sun_prox) + (1.0 - alt_t) * 0.32, 0.0, 1.0)

    c_rgb = col_crown[None, None, :] * (1.0 - lit[..., None]) + col_mid[None, None, :] * lit[..., None]
    belly_w = np.clip((lit - 0.30) / 0.70, 0.0, 1.0) * (0.58 + 0.42 * sun_prox)
    c_rgb = c_rgb * (1.0 - belly_w[..., None]) + col_belly[None, None, :] * belly_w[..., None]
    rim_w = np.clip(silver_rim * (0.56 + 0.85 * sun_prox), 0.0, 0.98)
    c_rgb = c_rgb * (1.0 - rim_w[..., None]) + col_rim[None, None, :] * rim_w[..., None]

    alpha = np.clip(np.power(np.clip(density, 0.0, 1.0), 0.82) * 0.94, 0.0, 0.94)
    c_rgb_full = resize_bicubic(c_rgb, W, H)
    alpha_full = resize_bicubic(alpha, W, H)
    c_rgb_full = np.clip(c_rgb_full + (mottle * 0.022 + dabs * 0.015)[..., None] * alpha_full[..., None], 0.0, 1.28)
    return c_rgb_full, alpha_full


def paint_sky_headland_and_copper_sea(cam: Camera3D, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = xx / W
    v = yy / H
    hy = float(cam.horizon_y)
    hv = hy / H

    sun_u, sun_v, _ = cam.project_uv_norm(cam.eye - SUN_DIR * 500.0)

    mottle = fbm_gouache(H, W, rng)
    dabs = flat_brush_dabs(H, W, rng, count=90)
    tooth = paper_tooth(H, W, rng)

    # 1. Luminous Ember sunset sky gradient (rich plum zenith -> crimson coral -> fiery apricot -> cream-gold horizon)
    sky_t = np.clip((hv - v) / max(0.05, hv), 0.0, 1.0)
    c_hor = np.array([1.05, 0.81, 0.52], dtype=np.float32)
    c_low = np.array([0.98, 0.48, 0.26], dtype=np.float32)
    c_mid = np.array([0.58, 0.19, 0.175], dtype=np.float32)
    c_zen = np.array([0.165, 0.080, 0.132], dtype=np.float32)

    t1 = np.clip(sky_t / 0.25, 0.0, 1.0)
    t2 = np.clip((sky_t - 0.25) / 0.35, 0.0, 1.0)
    t3 = np.clip((sky_t - 0.60) / 0.40, 0.0, 1.0)
    s01 = c_hor[None, None, :] * (1.0 - t1[..., None]) + c_low[None, None, :] * t1[..., None]
    s12 = s01 * (1.0 - t2[..., None]) + c_mid[None, None, :] * t2[..., None]
    sky = s12 * (1.0 - t3[..., None]) + c_zen[None, None, :] * t3[..., None]

    du_s = (u - sun_u) * (W / H)
    dv_s = v - sun_v
    r_sun = np.hypot(du_s, dv_s)
    solar_scatter = np.exp(-r_sun * 2.5) * 0.44 + np.exp(-r_sun * 8.0) * 0.58
    sky += solar_scatter[..., None] * np.array([1.02, 0.70, 0.36], dtype=np.float32)
    sky = np.clip(sky + (mottle * 0.015 + dabs * 0.010 + tooth * 0.0045)[..., None], 0.0, 1.35)

    # 2. Sculptural optical-marched clouds
    cloud_rgb, cloud_a = paint_sculptural_sunset_clouds(hv, sun_u, sun_v, mottle, dabs, rng)
    sky = sky * (1.0 - cloud_a[..., None]) + cloud_rgb * cloud_a[..., None]

    # 3. Blazing low setting solar disk (crisp glowing disc inside Arch[1]!)
    sun_disk = np.clip((0.0245 - r_sun) / 0.0045, 0.0, 1.0)
    sun_corona = np.exp(-(r_sun / 0.064) ** 2) * 0.58 + np.exp(-(r_sun / 0.16) ** 2) * 0.24
    sky += sun_corona[..., None] * np.array([1.04, 0.70, 0.34], dtype=np.float32)
    sky = sky * (1.0 - sun_disk[..., None]) + np.array([1.18, 1.04, 0.84], dtype=np.float32)[None, None, :] * sun_disk[..., None]

    # 4. Distant volcanic coastal headland / peninsula silhouette along horizon
    surf_hl, ctx_hl = new_cairo_2x()
    ctx_hl.move_to(0, hy + 4)
    ridge_far = [
        (0.28, 0.0), (0.38, -0.025), (0.48, -0.036), (0.58, -0.022),
        (0.68, -0.012), (0.76, -0.003), (0.85, -0.018), (0.93, -0.032), (1.02, -0.042),
    ]
    for rx, ry in ridge_far:
        ctx_hl.line_to(rx * W, hy + ry * H)
    ctx_hl.line_to(W, hy + 6)
    ctx_hl.close_path()
    g_far = cairo.LinearGradient(0, hy - 50, 0, hy + 4)
    g_far.add_color_stop_rgba(0.0, 0.42, 0.175, 0.172, 0.74)
    g_far.add_color_stop_rgba(1.0, 0.66, 0.295, 0.215, 0.56)
    ctx_hl.set_source(g_far)
    ctx_hl.fill()

    cape_r = [
        (0.818, 0.002), (0.848, -0.012), (0.878, -0.026), (0.916, -0.042),
        (0.956, -0.052), (0.986, -0.058), (1.01, -0.062),
    ]
    ctx_hl.move_to(0.814 * W, hy + 5)
    for i in range(len(cape_r)):
        px, py = cape_r[i][0] * W, hy + cape_r[i][1] * H
        if i == 0:
            ctx_hl.line_to(px, py)
        else:
            p_prev = cape_r[i - 1]
            mx = 0.5 * (p_prev[0] * W + px)
            ctx_hl.curve_to(mx, hy + p_prev[1] * H, mx, py, px, py)
    ctx_hl.line_to(W + 4, hy + 6)
    ctx_hl.close_path()
    g_cape = cairo.LinearGradient(0.81 * W, hy - 68, 0.98 * W, hy + 4)
    g_cape.add_color_stop_rgba(0.0, 0.34, 0.145, 0.145, 0.94)
    g_cape.add_color_stop_rgba(0.55, 0.22, 0.095, 0.115, 0.96)
    g_cape.add_color_stop_rgba(1.0, 0.15, 0.068, 0.090, 0.98)
    ctx_hl.set_source(g_cape)
    ctx_hl.fill()
    ctx_hl.set_source_rgba(1.00, 0.64, 0.32, 0.58)
    ctx_hl.set_line_width(2.2)
    ctx_hl.move_to(0.820 * W, hy)
    ctx_hl.curve_to(0.852 * W, hy - 15, 0.882 * W, hy - 30, 0.924 * W, hy - 48)
    ctx_hl.stroke()

    # Distant coastal cape lighthouse silhouette & beacon on right headland (u ~ 0.945, clear of Pier[2]!)
    lh_x, lh_y = 0.945 * W, hy - 0.048 * H
    ctx_hl.set_source_rgba(0.20, 0.088, 0.102, 0.95)
    ctx_hl.move_to(lh_x - 3.6, lh_y + 4)
    ctx_hl.line_to(lh_x - 2.0, lh_y - 17)
    ctx_hl.line_to(lh_x + 2.0, lh_y - 17)
    ctx_hl.line_to(lh_x + 3.6, lh_y + 4)
    ctx_hl.close_path()
    ctx_hl.fill()
    g_bcn = cairo.RadialGradient(lh_x, lh_y - 18.5, 0.5, lh_x, lh_y - 18.5, 14.0)
    g_bcn.add_color_stop_rgba(0.0, 1.0, 0.95, 0.74, 0.98)
    g_bcn.add_color_stop_rgba(0.35, 1.0, 0.62, 0.26, 0.54)
    g_bcn.add_color_stop_rgba(1.0, 1.0, 0.45, 0.15, 0.0)
    ctx_hl.set_source(g_bcn)
    ctx_hl.arc(lh_x, lh_y - 18.5, 14.0, 0, math.tau)
    ctx_hl.fill()

    hl_1x = finish_cairo_2x(surf_hl)
    sky = over_premul(sky, hl_1x)

    # 5. Open Vermilion Sunset Ocean Bay (v >= hv) with Fresnel capillary-rippled sky reflection!
    below = yy >= hy
    dist_t = np.clip((yy - hy) / max(1.0, H - hy), 0.0, 1.0)

    persp_y = np.power(np.maximum(dist_t, 1e-4), 0.52) * 145.0
    w_coarse = resize_bicubic(rng.standard_normal((H // 14, W // 72)).astype(np.float32), W, H)
    w_mid = resize_bicubic(rng.standard_normal((H // 6, W // 34)).astype(np.float32), W, H)
    w_fine = resize_bicubic(rng.standard_normal((H // 3, W // 18)).astype(np.float32), W, H)
    wave_sin = np.sin(persp_y * 2.3 + u * 28.0 + w_coarse * 2.1) * 0.42 + np.sin(persp_y * 5.8 - u * 44.0 + w_mid * 1.8) * 0.28
    water_ripple = (0.35 * w_coarse + 0.34 * w_mid + 0.19 * w_fine + 0.42 * wave_sin).astype(np.float32)
    water_ripple -= float(water_ripple.mean())
    water_ripple /= float(water_ripple.std()) + 1e-6

    map_x = np.clip(xx + water_ripple * (3.2 + 28.0 * np.power(dist_t, 0.82)), 0.0, W - 1.0).astype(np.float32)
    map_y = np.clip(2.0 * hy - yy + water_ripple * (1.2 + 12.0 * np.power(dist_t, 0.85)), 0.0, hy - 1.0).astype(np.float32)
    sky_refl = cv2.remap(sky, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    sea_deep = np.array([0.075, 0.032, 0.044], dtype=np.float32)
    sea_mid = np.array([0.162, 0.060, 0.064], dtype=np.float32)
    sea_base = sea_mid[None, None, :] * (1.0 - dist_t[..., None]) + sea_deep[None, None, :] * dist_t[..., None]

    fresnel = 0.32 + (0.92 - 0.32) * np.power(1.0 - dist_t, 1.4)
    sea = sea_base * (1.0 - fresnel[..., None]) + sky_refl * fresnel[..., None] * 0.92

    # Crisp molten-copper sun column sparkles aligned strictly with sun_u
    col_w = 0.030 + 0.145 * np.power(dist_t, 0.78)
    sun_col = np.exp(-((u - sun_u) / col_w) ** 2)
    crest = np.clip((water_ripple + 0.12) / 1.32, 0.0, 1.0)
    glint_sharp = np.power(crest, 3.6) * sun_col * (0.48 + 0.78 * np.exp(-dist_t * 2.1))
    glint_broad = np.power(crest, 1.5) * np.exp(-((u - sun_u) / (col_w * 2.3)) ** 2) * 0.36
    sea += glint_sharp[..., None] * np.array([1.26, 0.90, 0.48], dtype=np.float32)
    sea += glint_broad[..., None] * np.array([0.94, 0.44, 0.18], dtype=np.float32)

    hor_mist = np.exp(-np.abs(yy - hy) / 10.0)
    canvas = np.where(below[..., None], sea, sky)
    fog_rgb = np.array([0.76, 0.38, 0.26], dtype=np.float32)
    canvas = canvas * (1.0 - hor_mist[..., None] * 0.36) + np.array([0.99, 0.68, 0.42], dtype=np.float32)[None, None, :] * (hor_mist[..., None] * 0.36)

    meta = {
        "horizon_v": float(hv),
        "sun_uv": (float(sun_u), float(sun_v)),
        "cape_beacon_uv": (float(lh_x / W), float((lh_y - 18.5) / H)),
    }
    return canvas.astype(np.float32), fog_rgb, water_ripple, dist_t, meta


# =============================================================================
# Stage 1 & 2 — Planar UV Textures (Metric 3x3 Voronoi Cobbles, Masonry, Bakery)
# =============================================================================

def compute_voronoi_3x3(nx: int, ny: int, tw: int, th: int, rng: np.random.Generator, jitter: float = 0.36) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = (np.arange(nx, dtype=np.float32)[None, :] + 0.5 + rng.uniform(-jitter, jitter, (ny, nx)).astype(np.float32))
    gy = (np.arange(ny, dtype=np.float32)[:, None] + 0.5 + rng.uniform(-jitter, jitter, (ny, nx)).astype(np.float32))
    c_hash = rng.uniform(0.0, 1.0, (ny, nx)).astype(np.float32)

    px = np.linspace(0.0, nx - 1e-4, tw, dtype=np.float32)[None, :]
    py = np.linspace(0.0, ny - 1e-4, th, dtype=np.float32)[:, None]
    ix = np.floor(px).astype(np.int32)
    iy = np.floor(py).astype(np.int32)

    d1 = np.full((th, tw), 1e9, dtype=np.float32)
    d2 = np.full((th, tw), 1e9, dtype=np.float32)
    best_val = np.zeros((th, tw), dtype=np.float32)
    best_id = np.zeros((th, tw), dtype=np.int32)

    for dy in (-1, 0, 1):
        cy = np.clip(iy + dy, 0, ny - 1)
        for dx in (-1, 0, 1):
            cx = np.clip(ix + dx, 0, nx - 1)
            sx = gx[cy, cx] + (ix + dx - cx).astype(np.float32)
            sy = gy[cy, cx] + (iy + dy - cy).astype(np.float32)
            dist = np.hypot(px - sx, py - sy)
            closer1 = dist < d1
            d2 = np.where(closer1, d1, np.minimum(d2, dist))
            d1 = np.where(closer1, dist, d1)
            best_val = np.where(closer1, c_hash[cy, cx], best_val)
            best_id = np.where(closer1, cy * nx + cx, best_id)

    return (d2 - d1), best_id, best_val


def paint_wet_cobblestone_lane_uv(
    rng: np.random.Generator,
    x_span: tuple[float, float],
    y_span: tuple[float, float],
    tw: int = 896,
    th: int = 1600,
) -> tuple[np.ndarray, np.ndarray]:
    x0, x1 = x_span
    y0, y1 = y_span
    wx = np.linspace(x0, x1, tw, dtype=np.float32)[None, :]
    wy = np.linspace(y0, y1, th, dtype=np.float32)[:, None]
    wx_2d = np.broadcast_to(wx, (th, tw)).copy()
    wy_2d = np.broadcast_to(wy, (th, tw)).copy()
    wz_2d = np.full((th, tw), Z_LANE, dtype=np.float32)

    nx = max(8, int(round((x1 - x0) / 0.25)))
    ny = max(12, int(round((y1 - y0) / 0.24)))
    edge_d, _, c_val = compute_voronoi_3x3(nx, ny, tw, th, rng, jitter=0.34)

    mottle = fbm_gouache(th, tw, rng, scales=((56, 54), (22, 24), (9, 8.5), (4, 4.2)))
    fine_stone = paper_tooth(th, tw, rng)

    stone_a = np.array([0.165, 0.112, 0.105], dtype=np.float32)
    stone_b = np.array([0.248, 0.165, 0.134], dtype=np.float32)
    stone_c = np.array([0.115, 0.080, 0.086], dtype=np.float32)
    t_col = np.clip(c_val, 0.0, 1.0)[..., None]
    rgb = stone_a[None, None, :] * (1.0 - t_col) + stone_b[None, None, :] * t_col
    cool_m = np.clip((c_val - 0.65) * 2.8, 0.0, 1.0)[..., None]
    rgb = rgb * (1.0 - cool_m * 0.45) + stone_c[None, None, :] * (cool_m * 0.45)

    crevice = np.clip((0.12 - edge_d) / 0.12, 0.0, 1.0)
    crown = np.clip((edge_d - 0.08) / 0.28, 0.0, 1.0)
    rgb = rgb * (1.0 - crevice[..., None] * 0.72) + np.array([0.040, 0.024, 0.028], dtype=np.float32)[None, None, :] * (crevice[..., None] * 0.72)
    rgb += (mottle * 0.022 + fine_stone * 0.011)[..., None]

    sun_vis = trace_lane_shadows(wx_2d, wy_2d, wz_2d)
    amber_glow, crimson_glow = eval_shop_and_crossing_glow(wx_2d, wy_2d, wz_2d)

    sun_tint = np.array([1.30, 0.80, 0.47], dtype=np.float32)
    shd_tint = np.array([0.45, 0.33, 0.42], dtype=np.float32)
    light_rgb = shd_tint[None, None, :] * (1.0 - sun_vis[..., None]) + sun_tint[None, None, :] * sun_vis[..., None]
    rgb = rgb * light_rgb

    rgb += (crown * sun_vis * 0.12)[..., None] * np.array([1.06, 0.65, 0.35], dtype=np.float32)
    rgb += (amber_glow * (0.15 + 0.15 * crown))[..., None] * np.array([1.02, 0.62, 0.24], dtype=np.float32)
    rgb += (crimson_glow * (0.12 + 0.16 * crown))[..., None] * np.array([1.06, 0.26, 0.14], dtype=np.float32)

    # Flush Japanese Cape-Gauge coastal tram crossing rails & timber sleeper bed
    t_r = (wx_2d - x0) / max(1e-4, x1 - x0)
    rail_yc = np.float32(RAIL_Y_L) + t_r * np.float32(RAIL_Y_R - RAIL_Y_L)
    dy_rail = wy_2d - rail_yc
    in_bed = np.abs(dy_rail) < 1.12
    plank_seam = np.abs(np.mod(dy_rail + 1.12, 0.28) - 0.14) > 0.128
    wood_col = np.array([0.115, 0.068, 0.056], dtype=np.float32) * light_rgb
    wood_col = np.where(plank_seam[..., None], wood_col * 0.45, wood_col)
    rgb = np.where(in_bed[..., None], rgb * 0.22 + wood_col * 0.78, rgb)

    for r_off in (-RAIL_GAUGE * 0.5, RAIL_GAUGE * 0.5):
        d_steel = np.abs(dy_rail - np.float32(r_off))
        in_groove = (d_steel >= 0.042) & (d_steel < 0.095)
        in_rail = d_steel < 0.042
        rail_sheen = np.clip(1.0 - d_steel / 0.042, 0.0, 1.0)
        steel_rgb = (
            np.array([0.32, 0.24, 0.24], dtype=np.float32) * (1.0 - rail_sheen[..., None])
            + np.array([1.14, 0.80, 0.52], dtype=np.float32) * rail_sheen[..., None] * (0.45 + 0.55 * sun_vis[..., None])
        )
        rgb = np.where(in_groove[..., None], rgb * 0.22, rgb)
        rgb = np.where(in_rail[..., None], steel_rgb, rgb)

    # Worn painted crossing stop-line stripe at y = 13.85 m
    in_stop = (np.abs(wy_2d - 13.85) < 0.12) & (wx_2d > CURB_X1 + 0.15) & (wx_2d < PARAPET_X0 - 0.15)
    wear = np.clip(0.55 + mottle * 0.28 - crevice * 0.65, 0.0, 0.80)
    stop_col = np.array([0.80, 0.58, 0.24], dtype=np.float32) * light_rgb
    rgb = np.where(in_stop[..., None], rgb * (1.0 - wear[..., None]) + stop_col * wear[..., None], rgb)

    # Organic rainwater puddle mask with cobble-crevice seepage margins!
    puddle_warp = resize_bicubic(fbm_gouache(th // 2, tw // 2, rng, scales=((28, 28), (12, 12), (6, 6), (3, 3))), tw, th)
    d_lobe1 = ((wx_2d - 0.18) / 1.62) ** 2 + ((wy_2d - 9.45) / 3.45) ** 2
    d_lobe2 = ((wx_2d + 0.72) / 1.15) ** 2 + ((wy_2d - 11.40) / 2.40) ** 2
    d_lobe3 = ((wx_2d - 0.58) / 1.25) ** 2 + ((wy_2d - 6.85) / 1.95) ** 2
    puddle_field = 1.0 - np.minimum(np.minimum(d_lobe1, d_lobe2), d_lobe3) + puddle_warp * 0.24 - (1.0 - crevice) * 0.075
    puddle_mask = np.clip((puddle_field - 0.15) / 0.22, 0.0, 1.0)
    puddle_mask *= np.clip((wx_2d - (CURB_X1 + 0.08)) / 0.24, 0.0, 1.0)
    puddle_mask *= np.clip(((PARAPET_X0 - 0.12) - wx_2d) / 0.26, 0.0, 1.0)
    puddle_mask *= np.clip((14.20 - wy_2d) / 0.60, 0.0, 1.0)
    # Allow high cobble crowns near shallow puddle margins to peek through subtly!
    puddle_mask *= (1.0 - np.clip(crown * (1.0 - puddle_mask) * 0.45, 0.0, 0.35))

    wet_ring = np.clip((puddle_field - 0.01) / 0.18, 0.0, 1.0) * (1.0 - puddle_mask)
    rgb *= (1.0 - wet_ring[..., None] * 0.36)
    rgb *= (1.0 - puddle_mask[..., None] * 0.44)

    alpha = np.ones((th, tw, 1), dtype=np.float32)
    tex = np.concatenate([np.clip(rgb, 0.0, 1.4), alpha], axis=-1)
    puddle_rgba = np.concatenate([puddle_mask[..., None], puddle_mask[..., None], puddle_mask[..., None], alpha], axis=-1)
    return tex, puddle_rgba


def paint_stone_curb_or_parapet_uv(
    rng: np.random.Generator,
    u_span: tuple[float, float],
    v_span: tuple[float, float],
    kind: str,
    tw: int = 512,
    th: int = 1024,
) -> np.ndarray:
    u0, u1 = u_span
    v0, v1 = v_span
    mottle = fbm_gouache(th, tw, rng, scales=((48, 46), (19, 20), (8, 8.5), (3.8, 4.0)))
    tooth = paper_tooth(th, tw, rng)

    if kind == "curb_top":
        ny = max(8, int(round((v1 - v0) / 0.72)))
        edge_d, _, c_val = compute_voronoi_3x3(3, ny, tw, th, rng, jitter=0.14)
        wx = np.broadcast_to(np.linspace(u0, u1, tw, dtype=np.float32)[None, :], (th, tw))
        wy = np.broadcast_to(np.linspace(v0, v1, th, dtype=np.float32)[:, None], (th, tw))
        wz = np.full((th, tw), Z_CURB, dtype=np.float32)
        sun_vis = trace_eave_and_awning_shadow(wx, wy, wz)
        amber, crimson = eval_shop_and_crossing_glow(wx, wy, wz)
        base = np.array([0.23, 0.165, 0.145], dtype=np.float32)[None, None, :] * (0.82 + 0.34 * c_val[..., None])
        crev = np.clip((0.08 - edge_d) / 0.08, 0.0, 1.0)
        base *= (1.0 - crev[..., None] * 0.62)
        lit = np.array([0.45, 0.33, 0.42], dtype=np.float32) * (1.0 - sun_vis[..., None]) + np.array([1.26, 0.76, 0.44], dtype=np.float32) * sun_vis[..., None]
        rgb = base * lit + amber[..., None] * np.array([0.28, 0.16, 0.058], dtype=np.float32) + crimson[..., None] * np.array([0.18, 0.04, 0.025], dtype=np.float32)
        w_ao = np.clip((wx - np.float32(u0)) / 0.32, 0.0, 1.0)
        rgb *= (0.52 + 0.48 * w_ao[..., None])
    elif kind == "parapet_inner":
        uu = np.linspace(0.0, (u1 - u0) / 0.68, tw, dtype=np.float32)[None, :]
        vv = np.linspace(0.0, (v1 - v0) / 0.24, th, dtype=np.float32)[:, None]
        row = np.floor(vv).astype(np.int32)
        u_shift = uu + (row % 2) * 0.5
        col = np.floor(u_shift).astype(np.int32)
        fu = u_shift - col
        fv = vv - row
        mortar = ((fu < 0.055) | (fu > 0.945) | (fv < 0.09) | (fv > 0.91)).astype(np.float32)
        h_val = np.sin(col * 12.9898 + row * 78.233) * 43758.5453
        h_val = h_val - np.floor(h_val)
        base = np.array([0.16, 0.105, 0.098], dtype=np.float32)[None, None, :] * (0.76 + 0.42 * h_val[..., None])
        moss_m = np.clip((0.35 - np.linspace(0.0, 1.0, th, dtype=np.float32)[:, None]) / 0.35 + mottle * 0.20, 0.0, 1.0)
        base = base * (1.0 - moss_m[..., None] * 0.25) + np.array([0.085, 0.092, 0.068], dtype=np.float32)[None, None, :] * (moss_m[..., None] * 0.25)
        rgb = base * (1.0 - mortar[..., None] * 0.68)
        top_rim = np.clip((np.linspace(0.0, 1.0, th, dtype=np.float32)[:, None] - 0.85) / 0.15, 0.0, 1.0)
        rgb = rgb + top_rim[..., None] * np.array([0.09, 0.042, 0.024], dtype=np.float32)
    else:
        ny = max(12, int(round((v1 - v0) / 0.85)))
        edge_d, _, c_val = compute_voronoi_3x3(2, ny, tw, th, rng, jitter=0.08)
        base = np.array([0.36, 0.22, 0.16], dtype=np.float32)[None, None, :] * (0.82 + 0.32 * c_val[..., None])
        crev = np.clip((0.06 - edge_d) / 0.06, 0.0, 1.0)
        rgb = base * (1.0 - crev[..., None] * 0.62)
        edge_r = np.clip((np.linspace(0.0, 1.0, tw, dtype=np.float32)[None, :] - 0.65) / 0.35, 0.0, 1.0)
        rgb = rgb + edge_r[..., None] * np.array([0.36, 0.18, 0.075], dtype=np.float32)

    rgb = np.clip(rgb + (mottle * 0.020 + tooth * 0.008)[..., None], 0.0, 1.35)
    return np.concatenate([rgb, np.ones((th, tw, 1), dtype=np.float32)], axis=-1)


def paint_bakery_facade_uv(
    rng: np.random.Generator,
    y_span: tuple[float, float],
    z_span: tuple[float, float],
    x_wall: float,
    is_annex: bool = False,
    tw: int = 1280,
    th: int = 640,
) -> np.ndarray:
    """
    Weathered ochre-terracotta lime plaster (shikkui) & dark smoked-cedar (yakisugi)
    corner seaside bakery & railway signal house with softly blurred warm hand-painted
    interior scenes (bread loaves, shelves, draped curtains, pendant lamps with glowing
    cores), timber rain-shutter boxes (tobukuro), upper slatted wood balcony, sliding
    koushi-do lattice door, and crisp 2x PyCairo Japanese signage!
    """
    y0, y1 = y_span
    z0, z1 = z_span
    wy = np.broadcast_to(np.linspace(y0, y1, tw, dtype=np.float32)[None, :], (th, tw))
    wz = np.broadcast_to(np.linspace(z0, z1, th, dtype=np.float32)[:, None], (th, tw))
    wx = np.full((th, tw), x_wall, dtype=np.float32)

    mottle = fbm_gouache(th, tw, rng, scales=((52, 50), (21, 22), (9, 8.5), (4, 4.2)))
    tooth = paper_tooth(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=88)

    z_split = 2.60 if not is_annex else 2.20
    plaster_col = np.array([0.49, 0.30, 0.195], dtype=np.float32)[None, None, :] + mottle[..., None] * 0.042 + dabs[..., None] * 0.028
    # Weathered vertical rain staining on aged coastal stucco
    drip_n = resize_bicubic(rng.standard_normal((th // 20, tw // 9)).astype(np.float32), tw, th)
    plaster_col = plaster_col * (1.0 - np.clip(drip_n * 0.12, 0.0, 0.22))[..., None]

    board_v = (wz - z0) / 0.19
    board_f = board_v - np.floor(board_v)
    board_id = np.floor(board_v).astype(np.int32)
    b_hash = np.sin(board_id * 17.31 + 3.1) * 43758.54
    b_hash = b_hash - np.floor(b_hash)
    cedar_col = np.array([0.145, 0.086, 0.070], dtype=np.float32)[None, None, :] * (0.80 + 0.36 * b_hash[..., None])
    cedar_col = cedar_col * (0.55 + 0.45 * np.clip(board_f / 0.18, 0.0, 1.0))[..., None] + (mottle * 0.015 + tooth * 0.006)[..., None]

    rgb = np.where((wz >= z_split)[..., None], plaster_col, cedar_col)
    rgb = np.where((wz < (z0 + 0.34))[..., None], np.array([0.165, 0.122, 0.114], dtype=np.float32) + mottle[..., None] * 0.018, rgb)

    sun_vis = trace_eave_and_awning_shadow(wx, wy, wz)
    sun_lit = np.array([1.28, 0.74, 0.42], dtype=np.float32)
    shd_lit = np.array([0.36, 0.24, 0.33], dtype=np.float32)
    rgb = rgb * (shd_lit[None, None, :] * (1.0 - sun_vis[..., None]) + sun_lit[None, None, :] * sun_vis[..., None])

    amber_g, _ = eval_shop_and_crossing_glow(wx, wy, wz)
    rgb = rgb + amber_g[..., None] * np.array([0.16, 0.088, 0.032], dtype=np.float32)

    def y_to_px(y_m: float) -> float:
        return (y_m - y0) / (y1 - y0) * tw

    def z_to_px(z_m: float) -> float:
        return (z_m - z0) / (z1 - z0) * th

    # Pass A: Softly blurred warm interior scenes rendered onto separate surface & Gaussian-blurred!
    surf_int = cairo.ImageSurface(cairo.FORMAT_ARGB32, tw, th)
    ctx_int = cairo.Context(surf_int)
    ctx_int.set_antialias(cairo.ANTIALIAS_BEST)

    # Pass B: Crisp architectural mullions, timber shutters, balcony & Japanese signboards!
    surf_fg = cairo.ImageSurface(cairo.FORMAT_ARGB32, tw, th)
    ctx_fg = cairo.Context(surf_fg)
    ctx_fg.set_antialias(cairo.ANTIALIAS_BEST)

    def draw_timber_rect(ya: float, yb: float, za: float, zb: float, col=(0.090, 0.048, 0.044, 0.99)):
        xa, xb = y_to_px(ya), y_to_px(yb)
        za_p, zb_p = z_to_px(za), z_to_px(zb)
        ctx_fg.set_source_rgba(*col)
        ctx_fg.rectangle(min(xa, xb), min(za_p, zb_p), abs(xb - xa), abs(zb_p - za_p))
        ctx_fg.fill()
        ctx_fg.set_source_rgba(0.45, 0.22, 0.12, 0.42)
        ctx_fg.set_line_width(1.6)
        ctx_fg.move_to(max(xa, xb) - 1.0, min(za_p, zb_p))
        ctx_fg.line_to(max(xa, xb) - 1.0, max(za_p, zb_p))
        ctx_fg.stroke()

    post_ys = (4.10, 6.25, 9.65, 12.60, 15.35) if not is_annex else (15.85, 18.10, 20.35)
    for py_p in post_ys:
        draw_timber_rect(py_p - 0.11, py_p + 0.11, z0, z1)
    draw_timber_rect(y0, y1, z_split - 0.11, z_split + 0.11, (0.080, 0.042, 0.040, 0.99))
    draw_timber_rect(y0, y1, z1 - 0.18, z1, (0.068, 0.036, 0.036, 0.99))

    def draw_glowing_window(
        ya: float, yb: float, za: float, zb: float,
        cols: int, rows: int, warmth: float = 1.0,
        kind: str = "bakery",
    ):
        x_a, x_b = y_to_px(ya), y_to_px(yb)
        z_a, z_b = z_to_px(za), z_to_px(zb)
        x_l, w_r = min(x_a, x_b), abs(x_b - x_a)
        y_t, h_r = min(z_a, z_b), abs(z_b - z_a)

        # ---- Blurred painterly warm interior on surf_int ----
        ctx_int.save()
        ctx_int.rectangle(x_l, y_t, w_r, h_r)
        ctx_int.clip()
        # Rich warm golden interior depth gradient (brightest near glowing overhead pendant bulb!)
        g_win = cairo.RadialGradient(x_l + w_r * 0.50, y_t + h_r * 0.66, w_r * 0.05,
                                     x_l + w_r * 0.50, y_t + h_r * 0.48, max(w_r, h_r) * 0.92)
        g_win.add_color_stop_rgba(0.0, min(1.0, 1.00 * warmth), min(1.0, 0.92 * warmth), min(1.0, 0.62 * warmth), 1.0)
        g_win.add_color_stop_rgba(0.38, min(1.0, 0.96 * warmth), min(1.0, 0.65 * warmth), min(1.0, 0.26 * warmth), 1.0)
        g_win.add_color_stop_rgba(0.82, min(1.0, 0.62 * warmth), min(1.0, 0.28 * warmth), min(1.0, 0.12 * warmth), 1.0)
        g_win.add_color_stop_rgba(1.0, min(1.0, 0.36 * warmth), min(1.0, 0.15 * warmth), min(1.0, 0.08 * warmth), 1.0)
        ctx_int.set_source(g_win)
        ctx_int.rectangle(x_l, y_t, w_r, h_r)
        ctx_int.fill()

        # Soft interior ceiling beams & warm room depth walls
        ctx_int.set_source_rgba(0.26, 0.11, 0.06, 0.42)
        ctx_int.rectangle(x_l, y_t + h_r * 0.86, w_r, h_r * 0.14)
        ctx_int.fill()

        # Enamel hanging pendant lamp shade + incandescent bulb glow!
        lx_p, ly_p = x_l + w_r * 0.50, y_t + h_r * 0.68
        ctx_int.set_source_rgba(0.20, 0.085, 0.055, 0.65)
        ctx_int.set_line_width(3.0)
        ctx_int.move_to(lx_p, y_t + h_r)
        ctx_int.line_to(lx_p, ly_p + h_r * 0.04)
        ctx_int.stroke()
        ctx_int.move_to(lx_p - w_r * 0.12, ly_p)
        ctx_int.curve_to(lx_p - w_r * 0.05, ly_p + h_r * 0.10, lx_p + w_r * 0.05, ly_p + h_r * 0.10, lx_p + w_r * 0.12, ly_p)
        ctx_int.close_path()
        ctx_int.fill()
        # Bright glowing incandescent filament pool directly underneath shade
        g_bulb = cairo.RadialGradient(lx_p, ly_p - h_r * 0.04, 2.0, lx_p, ly_p - h_r * 0.06, w_r * 0.28)
        g_bulb.add_color_stop_rgba(0.0, 1.0, 0.98, 0.84, 0.95)
        g_bulb.add_color_stop_rgba(0.5, 1.0, 0.78, 0.36, 0.45)
        g_bulb.add_color_stop_rgba(1.0, 1.0, 0.62, 0.22, 0.0)
        ctx_int.set_source(g_bulb)
        ctx_int.arc(lx_p, ly_p - h_r * 0.04, w_r * 0.28, 0, math.tau)
        ctx_int.fill()

        if kind == "bakery":
            # Soft tiered wooden bakery counter, organic bread basket silhouettes, tall coffee jars & side curtains!
            ctx_int.set_source_rgba(0.24, 0.10, 0.058, 0.62)
            ctx_int.rectangle(x_l, y_t, w_r, h_r * 0.22)
            ctx_int.fill()
            # Mid wooden bread shelf
            ctx_int.rectangle(x_l + w_r * 0.08, y_t + h_r * 0.40, w_r * 0.84, h_r * 0.045)
            ctx_int.fill()
            # Irregular organic bread loaves, wicker tray & coffee bottles (soft organic domes & bottles!)
            ctx_int.set_source_rgba(0.32, 0.135, 0.068, 0.56)
            for bx_f, bw_f, bh_f in ((0.18, 0.14, 0.08), (0.36, 0.12, 0.07), (0.56, 0.16, 0.09), (0.76, 0.11, 0.14)):
                bx0 = x_l + w_r * bx_f
                ctx_int.move_to(bx0 - w_r * bw_f * 0.5, y_t + h_r * 0.22)
                ctx_int.curve_to(bx0 - w_r * bw_f * 0.45, y_t + h_r * (0.22 + bh_f * 1.4),
                                 bx0 + w_r * bw_f * 0.45, y_t + h_r * (0.22 + bh_f * 1.4),
                                 bx0 + w_r * bw_f * 0.5, y_t + h_r * 0.22)
                ctx_int.close_path()
                ctx_int.fill()
            # Draped warm cafe linen curtains framing left & right window edges
            ctx_int.set_source_rgba(0.30, 0.12, 0.065, 0.46)
            ctx_int.move_to(x_l, y_t + h_r)
            ctx_int.curve_to(x_l + w_r * 0.14, y_t + h_r * 0.65, x_l + w_r * 0.08, y_t + h_r * 0.35, x_l + w_r * 0.15, y_t)
            ctx_int.line_to(x_l, y_t)
            ctx_int.close_path()
            ctx_int.fill()
            ctx_int.move_to(x_l + w_r, y_t + h_r)
            ctx_int.curve_to(x_l + w_r * 0.86, y_t + h_r * 0.65, x_l + w_r * 0.92, y_t + h_r * 0.35, x_l + w_r * 0.85, y_t)
            ctx_int.line_to(x_l + w_r, y_t)
            ctx_int.close_path()
            ctx_int.fill()
        elif kind == "upper":
            # Signal-box / cozy upstairs residence: softly draped lace curtains, bookshelf & desk silhouette!
            ctx_int.set_source_rgba(0.24, 0.095, 0.060, 0.54)
            ctx_int.move_to(x_l, y_t + h_r)
            ctx_int.curve_to(x_l + w_r * 0.22, y_t + h_r * 0.62, x_l + w_r * 0.14, y_t + h_r * 0.28, x_l + w_r * 0.24, y_t)
            ctx_int.line_to(x_l, y_t)
            ctx_int.close_path()
            ctx_int.fill()
            ctx_int.move_to(x_l + w_r, y_t + h_r)
            ctx_int.curve_to(x_l + w_r * 0.78, y_t + h_r * 0.62, x_l + w_r * 0.86, y_t + h_r * 0.28, x_l + w_r * 0.76, y_t)
            ctx_int.line_to(x_l + w_r, y_t)
            ctx_int.close_path()
            ctx_int.fill()
            # Bookshelf / signal-lever desk silhouette low inside
            ctx_int.rectangle(x_l + w_r * 0.22, y_t, w_r * 0.56, h_r * 0.20)
            ctx_int.fill()
        else:  # door
            # Warm welcoming bakery doorway interior counter & subtle hanging noren cloth curtain at top!
            ctx_int.set_source_rgba(0.46, 0.14, 0.09, 0.62)
            ctx_int.rectangle(x_l, y_t + h_r * 0.74, w_r * 0.46, h_r * 0.26)
            ctx_int.rectangle(x_l + w_r * 0.54, y_t + h_r * 0.74, w_r * 0.46, h_r * 0.26)
            ctx_int.fill()
        ctx_int.restore()

        # ---- Crisp outer timber casing, mullions, shutters & sunset glass reflection on surf_fg ----
        ctx_fg.set_source_rgba(0.080, 0.042, 0.038, 0.99)
        ctx_fg.set_line_width(7.0)
        ctx_fg.rectangle(x_l - 3, y_t - 3, w_r + 6, h_r + 6)
        ctx_fg.stroke()

        if kind == "door":
            # Solid smoked-cedar lower kick-panel on sliding koushi-do door
            ctx_fg.set_source_rgba(0.105, 0.055, 0.046, 0.99)
            ctx_fg.rectangle(x_l, y_t, w_r, h_r * 0.30)
            ctx_fg.fill()

        # Crisp dark timber mullions
        ctx_fg.set_source_rgba(0.086, 0.045, 0.040, 0.99)
        ctx_fg.set_line_width(3.4 if kind != "door" else 2.6)
        for c in range(1, cols):
            mx = x_l + w_r * (c / cols)
            ctx_fg.move_to(mx, y_t)
            ctx_fg.line_to(mx, y_t + h_r)
            ctx_fg.stroke()
        for r in range(1, rows):
            my = y_t + h_r * (r / rows)
            ctx_fg.move_to(x_l, my)
            ctx_fg.line_to(x_l + w_r, my)
            ctx_fg.stroke()

        # Soft diagonal sunset reflection sheen across window glass
        ctx_fg.set_source_rgba(1.0, 0.84, 0.58, 0.14)
        ctx_fg.move_to(x_l + w_r * 0.08, y_t + h_r)
        ctx_fg.line_to(x_l + w_r * 0.28, y_t + h_r)
        ctx_fg.line_to(x_l + w_r * 0.10, y_t)
        ctx_fg.line_to(x_l, y_t)
        ctx_fg.close_path()
        ctx_fg.fill()

        if kind != "door":
            # Protruding timber windowsill & subtle weathered water stain beneath sill
            ctx_fg.set_source_rgba(0.14, 0.072, 0.056, 0.99)
            ctx_fg.rectangle(x_l - 10, y_t - 9, w_r + 20, 9)
            ctx_fg.fill()
            # Traditional wooden rain-shutter box (tobukuro) beside upper windows
            if kind == "upper":
                ctx_fg.set_source_rgba(0.11, 0.058, 0.048, 0.98)
                ctx_fg.rectangle(x_l - 22, y_t - 4, 18, h_r + 8)
                ctx_fg.fill()
                # Slatted wooden Japanese low balcony rail (tesuri) across bottom 28% of upper window!
                ctx_fg.set_source_rgba(0.092, 0.048, 0.042, 0.96)
                ctx_fg.set_line_width(4.2)
                ctx_fg.move_to(x_l - 12, y_t + h_r * 0.30)
                ctx_fg.line_to(x_l + w_r + 12, y_t + h_r * 0.30)
                ctx_fg.stroke()
                ctx_fg.set_line_width(2.2)
                for b_k in range(8):
                    bx_sl = x_l + (b_k + 0.5) * (w_r / 8.0)
                    ctx_fg.move_to(bx_sl, y_t - 6)
                    ctx_fg.line_to(bx_sl, y_t + h_r * 0.30)
                    ctx_fg.stroke()

    if not is_annex:
        draw_glowing_window(6.65, 9.25, 1.05, 2.28, cols=3, rows=2, warmth=1.04, kind="bakery")
        draw_glowing_window(10.25, 12.15, 0.20, 2.28, cols=4, rows=5, warmth=1.08, kind="door")
        draw_glowing_window(12.95, 14.95, 1.08, 2.24, cols=2, rows=2, warmth=0.96, kind="bakery")
        draw_glowing_window(6.85, 9.15, 3.08, 4.16, cols=3, rows=2, warmth=0.86, kind="upper")
        draw_glowing_window(10.45, 12.25, 3.08, 4.16, cols=2, rows=2, warmth=0.80, kind="upper")
        draw_glowing_window(13.15, 14.85, 3.12, 4.12, cols=2, rows=2, warmth=0.72, kind="upper")

        # Weathered horizontal wooden signboard above awning ("潮風ベーカリー・珈琲")
        sx0, sx1 = y_to_px(7.05), y_to_px(11.65)
        sz0, sz1 = z_to_px(2.68), z_to_px(3.04)
        ctx_fg.save()
        ctx_fg.set_source_rgba(0.095, 0.046, 0.040, 0.99)
        ctx_fg.rectangle(sx0, sz0, sx1 - sx0, sz1 - sz0)
        ctx_fg.fill()
        ctx_fg.set_source_rgba(0.68, 0.42, 0.22, 0.88)
        ctx_fg.set_line_width(2.5)
        ctx_fg.rectangle(sx0 + 4, sz0 + 4, (sx1 - sx0) - 8, (sz1 - sz0) - 8)
        ctx_fg.stroke()
        ctx_fg.translate(0.5 * (sx0 + sx1), 0.5 * (sz0 + sz1))
        ctx_fg.scale(1.0, -1.0)
        ctx_fg.select_font_face("Noto Serif CJK JP", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx_fg.set_font_size(24.0)
        txt = "潮風ベーカリー・珈琲"
        ext = ctx_fg.text_extents(txt)
        ctx_fg.set_source_rgba(0.96, 0.85, 0.64, 0.96)
        ctx_fg.move_to(-ext.width * 0.5 - ext.x_bearing, -ext.height * 0.5 - ext.y_bearing)
        ctx_fg.show_text(txt)
        ctx_fg.restore()

        # Vintage vertical cream-wood signboard beside shop doorway ("潮風堂パン")
        vx0, vx1 = y_to_px(9.62), y_to_px(10.10)
        vz0, vz1 = z_to_px(0.98), z_to_px(2.24)
        ctx_fg.save()
        ctx_fg.set_source_rgba(0.84, 0.72, 0.52, 0.97)
        ctx_fg.rectangle(vx0, vz0, vx1 - vx0, vz1 - vz0)
        ctx_fg.fill()
        ctx_fg.set_source_rgba(0.18, 0.072, 0.055, 0.95)
        ctx_fg.set_line_width(2.4)
        ctx_fg.rectangle(vx0 + 3, vz0 + 3, (vx1 - vx0) - 6, (vz1 - vz0) - 6)
        ctx_fg.stroke()
        ctx_fg.select_font_face("Noto Serif CJK JP", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx_fg.set_font_size(21.0)
        for idx, ch in enumerate(["潮", "風", "堂", "パ", "ン"]):
            cy_c = vz1 - (idx + 0.55) * ((vz1 - vz0) / 5.2)
            ctx_fg.save()
            ctx_fg.translate(0.5 * (vx0 + vx1), cy_c)
            ctx_fg.scale(1.0, -1.0)
            ex = ctx_fg.text_extents(ch)
            ctx_fg.set_source_rgba(0.18, 0.068, 0.052, 0.96)
            ctx_fg.move_to(-ex.width * 0.5 - ex.x_bearing, -ex.height * 0.5 - ex.y_bearing)
            ctx_fg.show_text(ch)
            ctx_fg.restore()
        ctx_fg.restore()

        # Small red enamel railway crossing plaque ("赤橋踏切")
        px0, px1 = y_to_px(14.05), y_to_px(15.20)
        pz0, pz1 = z_to_px(2.68), z_to_px(3.00)
        ctx_fg.save()
        ctx_fg.set_source_rgba(0.74, 0.21, 0.15, 0.97)
        ctx_fg.rectangle(px0, pz0, px1 - px0, pz1 - pz0)
        ctx_fg.fill()
        ctx_fg.set_source_rgba(0.96, 0.88, 0.72, 0.92)
        ctx_fg.set_line_width(2.0)
        ctx_fg.rectangle(px0 + 3, pz0 + 3, (px1 - px0) - 6, (pz1 - pz0) - 6)
        ctx_fg.stroke()
        ctx_fg.translate(0.5 * (px0 + px1), 0.5 * (pz0 + pz1))
        ctx_fg.scale(1.0, -1.0)
        ctx_fg.select_font_face("Noto Sans CJK JP", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx_fg.set_font_size(18.0)
        ptxt = "赤橋踏切"
        ex = ctx_fg.text_extents(ptxt)
        ctx_fg.set_source_rgba(0.98, 0.92, 0.80, 0.96)
        ctx_fg.move_to(-ex.width * 0.5 - ex.x_bearing, -ex.height * 0.5 - ex.y_bearing)
        ctx_fg.show_text(ptxt)
        ctx_fg.restore()
    else:
        draw_glowing_window(16.45, 17.80, 1.35, 2.38, cols=2, rows=2, warmth=0.76, kind="upper")
        draw_glowing_window(18.55, 19.85, 1.45, 2.38, cols=2, rows=2, warmth=0.64, kind="upper")

    # Composite softly-blurred warm interior FIRST, then crisp sharp mullions & signboards OVER it!
    surf_int.flush()
    b_int = np.frombuffer(surf_int.get_data(), dtype=np.uint8).reshape(th, tw, 4).astype(np.float32) * (1.0 / 255.0)
    int_rgba = np.stack([b_int[:, :, 2], b_int[:, :, 1], b_int[:, :, 0], b_int[:, :, 3]], axis=-1)
    # Soft depth-of-field blur on interior silhouettes ONLY inside window glass!
    int_rgb_blur = blur_2d(int_rgba[:, :, :3], 4.2)
    int_soft = np.concatenate([int_rgb_blur * int_rgba[:, :, 3:4], int_rgba[:, :, 3:4]], axis=-1)
    rgb = over_premul(rgb, int_soft)

    surf_fg.flush()
    b_fg = np.frombuffer(surf_fg.get_data(), dtype=np.uint8).reshape(th, tw, 4).astype(np.float32) * (1.0 / 255.0)
    fg_rgba = np.stack([b_fg[:, :, 2], b_fg[:, :, 1], b_fg[:, :, 0], b_fg[:, :, 3]], axis=-1)
    rgb = over_premul(rgb, fg_rgba)

    rgb = np.clip(rgb + (mottle * 0.012 + tooth * 0.005)[..., None], 0.0, 1.4)
    return np.concatenate([rgb, np.ones((th, tw, 1), dtype=np.float32)], axis=-1)


def paint_kawara_roof_uv(rng: np.random.Generator, length_m: float, slope_m: float, tw: int = 1024, th: int = 384) -> np.ndarray:
    uu = np.linspace(0.0, length_m / 0.24, tw, dtype=np.float32)[None, :]
    vv = np.linspace(0.0, slope_m / 0.26, th, dtype=np.float32)[:, None]
    mottle = fbm_gouache(th, tw, rng, scales=((44, 42), (18, 18), (8, 8), (3.5, 3.8)))

    rib = 0.5 + 0.5 * np.sin(uu * math.tau)
    course = vv - np.floor(vv)
    lap = np.clip((course - 0.80) / 0.20, 0.0, 1.0)

    base = np.array([0.115, 0.075, 0.088], dtype=np.float32)[None, None, :]
    hi = np.array([0.46, 0.22, 0.16], dtype=np.float32)[None, None, :]
    rgb = base * (1.0 - rib[..., None] * 0.55) + hi * (np.power(rib, 2.2)[..., None] * 0.55)
    rgb = rgb * (1.0 - lap[..., None] * 0.58)
    eave_band = np.clip((0.10 - np.linspace(0.0, 1.0, th, dtype=np.float32)[:, None]) / 0.10, 0.0, 1.0)
    rgb = rgb + eave_band[..., None] * np.array([0.22, 0.095, 0.055], dtype=np.float32) * (0.4 + 0.6 * rib[..., None])
    rgb = np.clip(rgb + mottle[..., None] * 0.020, 0.0, 1.3)
    return np.concatenate([rgb, np.ones((th, tw, 1), dtype=np.float32)], axis=-1)


def paint_soffit_or_awning_uv(rng: np.random.Generator, length_m: float, depth_m: float, is_awning: bool = False, tw: int = 768, th: int = 224) -> np.ndarray:
    mottle = fbm_gouache(th, tw, rng, scales=((40, 38), (16, 16), (7, 7), (3.5, 3.5)))
    uu = np.linspace(0.0, length_m / (0.34 if not is_awning else 0.14), tw, dtype=np.float32)[None, :]
    rib = 0.5 + 0.5 * np.sin(uu * math.tau)
    if is_awning:
        base = np.array([0.26, 0.118, 0.092], dtype=np.float32)[None, None, :]
        hi = np.array([0.62, 0.29, 0.16], dtype=np.float32)[None, None, :]
        rgb = base * (0.72 + 0.28 * rib[..., None]) + hi * (np.power(rib, 2.4)[..., None] * 0.42)
    else:
        rafter = (np.abs(uu - np.round(uu)) < 0.14).astype(np.float32)
        rgb = np.array([0.085, 0.048, 0.046], dtype=np.float32)[None, None, :] * (1.0 - rafter[..., None] * 0.45)
        rgb = rgb + np.array([0.065, 0.032, 0.018], dtype=np.float32)[None, None, :] * (1.0 - np.linspace(0.0, 1.0, th, dtype=np.float32)[:, None, None])
    rgb = np.clip(rgb + mottle[..., None] * 0.018, 0.0, 1.3)
    return np.concatenate([rgb, np.ones((th, tw, 1), dtype=np.float32)], axis=-1)


def paint_viaduct_ashlar_uv(rng: np.random.Generator, s0: float, s1: float, z0: float, z1: float,
                            is_far_face: bool = False, tw: int = 1600, th: int = 448) -> np.ndarray:
    """
    Monumental angled multi-arch railway viaduct ashlar masonry texture with exact
    transparent arch openings, dressed radiating stone voussoirs, granite cutwater
    pier courses, projecting cornice parapet, weathered lichen/water patina, and
    STRICTLY bounded pier-shaft rim lighting (stops cleanly at arch springing!).
    """
    ss = np.broadcast_to(np.linspace(s0, s1, tw, dtype=np.float32)[None, :], (th, tw))
    zz = np.broadcast_to(np.linspace(z0, z1, th, dtype=np.float32)[:, None], (th, tw))
    mottle = fbm_gouache(th, tw, rng, scales=((52, 50), (20, 20), (8.5, 8.5), (3.8, 4.0)))
    dabs = flat_brush_dabs(th, tw, rng, count=85)
    tooth = paper_tooth(th, tw, rng)

    s_mod = np.mod(ss + np.float32(VD_PIER_W * 0.5), np.float32(VD_SPAN))
    in_pier = s_mod < np.float32(VD_PIER_W)
    s_arch = s_mod - np.float32(VD_PIER_W + VD_ARCH_R)
    rad_ratio = np.clip(s_arch / np.float32(VD_ARCH_R), -1.0, 1.0)
    z_arch = np.float32(VD_Z_SPRING) + np.float32(VD_ARCH_R * 0.96) * np.sqrt(np.maximum(0.0, 1.0 - rad_ratio * rad_ratio))

    dist_above_arch = zz - z_arch
    arch_alpha = np.where(in_pier, 1.0, np.clip((dist_above_arch + 0.03) / 0.06, 0.0, 1.0)).astype(np.float32)

    # Richly varied dressed ashlar masonry courses (subtle warm terracotta-red & smoked basalt granite blocks!)
    row_f = (zz - z0) / 0.42
    row = np.floor(row_f).astype(np.int32)
    col_f = (ss / 0.88) + (row % 2) * 0.5
    col = np.floor(col_f).astype(np.int32)
    fu = col_f - col
    fv = row_f - row
    mortar = ((fu < 0.044) | (fu > 0.956) | (fv < 0.068) | (fv > 0.932)).astype(np.float32)
    b_hash = np.sin(col * 19.13 + row * 91.71) * 43758.54
    b_hash = b_hash - np.floor(b_hash)
    b_hash2 = np.sin(col * 41.7 + row * 23.9) * 19381.12
    b_hash2 = b_hash2 - np.floor(b_hash2)

    if not is_far_face:
        stone_lo = np.array([0.130, 0.072, 0.084], dtype=np.float32)
        stone_hi = np.array([0.265, 0.138, 0.118], dtype=np.float32)
        stone_warm = np.array([0.320, 0.152, 0.108], dtype=np.float32)
    else:
        stone_lo = np.array([0.090, 0.050, 0.064], dtype=np.float32)
        stone_hi = np.array([0.160, 0.085, 0.086], dtype=np.float32)
        stone_warm = np.array([0.190, 0.095, 0.080], dtype=np.float32)
    rgb = stone_lo[None, None, :] * (1.0 - b_hash[..., None]) + stone_hi[None, None, :] * b_hash[..., None]
    rgb = rgb * (1.0 - (b_hash2 > 0.72).astype(np.float32)[..., None] * 0.38) + stone_warm[None, None, :] * ((b_hash2 > 0.72).astype(np.float32)[..., None] * 0.38)
    rgb = rgb * (1.0 - mortar[..., None] * 0.62)

    # Subtle vertical rain-stain weathering beneath stringcourse cornice
    stain = resize_bicubic(rng.standard_normal((th // 18, tw // 14)).astype(np.float32), tw, th)
    under_cornice = np.clip((5.10 - zz) / 1.4, 0.0, 1.0) * np.clip((zz - 2.6) / 1.8, 0.0, 1.0)
    rgb *= (1.0 - np.clip(stain * 0.18, 0.0, 0.25) * under_cornice)[..., None]

    # Radiating dressed stone voussoir arch ring (0.58 m thick ring + raised keystone at top!)
    dx_a = s_arch
    dz_a = (zz - np.float32(VD_Z_SPRING)) / 0.96
    r_a = np.hypot(dx_a, dz_a)
    in_voussoir = (~in_pier) & (zz >= np.float32(VD_Z_SPRING - 0.05)) & (r_a >= np.float32(VD_ARCH_R - 0.02)) & (r_a <= np.float32(VD_ARCH_R + 0.56))
    ang_v = np.arctan2(dz_a, dx_a)
    v_seg = np.mod(ang_v * 13.0 / math.pi, 1.0)
    v_idx = np.floor(ang_v * 13.0 / math.pi).astype(np.int32)
    v_h = np.sin(v_idx * 31.7 + 5.2) * 12345.6
    v_h = v_h - np.floor(v_h)
    v_joint = ((v_seg < 0.06) | (v_seg > 0.94) | (np.abs(r_a - (VD_ARCH_R + 0.56)) < 0.048)).astype(np.float32)
    v_col = (np.array([0.24, 0.125, 0.112], dtype=np.float32)[None, None, :] * (0.82 + 0.34 * v_h[..., None])) * (1.0 - v_joint[..., None] * 0.68)
    # Warm golden sunset light wrap strictly around inner arch curve!
    arch_rim = np.exp(-np.maximum(0.0, r_a - VD_ARCH_R) / 0.15) * (zz >= VD_Z_SPRING - 0.06).astype(np.float32)
    v_col = v_col + arch_rim[..., None] * np.array([0.66, 0.30, 0.12], dtype=np.float32)
    rgb = np.where(in_voussoir[..., None], v_col, rgb)

    # Pier impost cap molding at springing line (z ~ 1.25..1.58) & tidal waterline moss/barnacles
    in_impost = in_pier & (zz >= VD_Z_SPRING - 0.22) & (zz <= VD_Z_SPRING + 0.18)
    rgb = np.where(in_impost[..., None], rgb * 1.28 + np.array([0.055, 0.025, 0.016], dtype=np.float32), rgb)
    tide_m = np.clip((-0.62 - zz) / 0.95, 0.0, 1.0)
    rgb = rgb * (1.0 - tide_m[..., None] * 0.56) + np.array([0.042, 0.036, 0.040], dtype=np.float32)[None, None, :] * (tide_m[..., None] * 0.56)

    # Projecting stone stringcourse (z ~ 5.06..5.32) + recessed shadow line under cornice + upper parapet wall
    in_under_shd = (zz >= 4.86) & (zz < 5.06)
    rgb = np.where(in_under_shd[..., None], rgb * 0.68, rgb)
    in_cornice = (zz >= 5.06) & (zz <= 5.32)
    in_parapet = zz > 5.32
    rgb = np.where(in_cornice[..., None], np.array([0.36, 0.19, 0.145], dtype=np.float32) + mottle[..., None] * 0.022, rgb)
    par_col = np.array([0.23, 0.130, 0.122], dtype=np.float32) + mottle[..., None] * 0.024 + dabs[..., None] * 0.016
    par_top = np.clip((zz - 5.82) / 0.18, 0.0, 1.0)
    par_col = par_col + par_top[..., None] * np.array([0.46, 0.22, 0.11], dtype=np.float32)
    # Subtle tram contact shadow band across parapet directly under tram car (s in 1.1..14.0)!
    under_tram = (ss >= 1.1) & (ss <= 14.0) & (zz >= 5.65)
    par_col = np.where(under_tram[..., None], par_col * 0.68, par_col)
    rgb = np.where(in_parapet[..., None], par_col, rgb)

    # FIXED: Pier sunward right edge rim highlight strictly bounded BELOW arch springing (zz < VD_Z_SPRING + 0.05)!
    pier_shaft_mask = (in_pier & (zz < np.float32(VD_Z_SPRING + 0.05))).astype(np.float32)
    pier_rim = pier_shaft_mask * np.exp(-np.maximum(0.0, np.float32(VD_PIER_W) - s_mod) / 0.14)
    rgb = rgb + pier_rim[..., None] * np.array([0.56, 0.25, 0.10], dtype=np.float32)

    rgb = np.clip(rgb + (mottle * 0.018 + dabs * 0.012 + tooth * 0.007)[..., None], 0.0, 1.35)
    return np.concatenate([rgb * arch_alpha[..., None], arch_alpha[..., None]], axis=-1)


# =============================================================================
# Stage 3 Band 1 — 3D Barrel-Vaulted Viaduct Intrados, Catenary & Hero Enoden Tram
# =============================================================================

def draw_viaduct_barrel_vaults(ctx: cairo.Context, cam: Camera3D, fog_rgb: np.ndarray,
                               mirror_z: float | None = None) -> None:
    for i in range(-1, 4):
        s_left_pier = i * VD_SPAN + VD_PIER_W * 0.5
        s_right_pier = (i + 1) * VD_SPAN - VD_PIER_W * 0.5
        s_center = 0.5 * (s_left_pier + s_right_pier)

        z_mid = float(cam.project_pt(vd_point(s_left_pier, VD_THICK * 0.5, 1.0))[2])
        c_shd = fog_mix((0.082, 0.042, 0.054), fog_rgb, z_mid)
        fill_3d_poly(ctx, cam, [
            vd_point(s_left_pier, 0.0, Z_SEA),
            vd_point(s_left_pier, VD_THICK, Z_SEA),
            vd_point(s_left_pier, VD_THICK, VD_Z_SPRING),
            vd_point(s_left_pier, 0.0, VD_Z_SPRING),
        ], (*c_shd, 0.98 if mirror_z is None else 0.50), mirror_z=mirror_z)

        c_sun = fog_mix((0.46, 0.21, 0.125), fog_rgb, z_mid)
        fill_3d_poly(ctx, cam, [
            vd_point(s_right_pier, 0.0, Z_SEA),
            vd_point(s_right_pier, VD_THICK, Z_SEA),
            vd_point(s_right_pier, VD_THICK, VD_Z_SPRING),
            vd_point(s_right_pier, 0.0, VD_Z_SPRING),
        ], (*c_sun, 0.98 if mirror_z is None else 0.50), mirror_z=mirror_z)

        n_seg = 16
        for k in range(n_seg):
            a0 = math.pi * (k / n_seg)
            a1 = math.pi * ((k + 1) / n_seg)
            am = 0.5 * (a0 + a1)
            sa0 = s_center - math.cos(a0) * VD_ARCH_R
            za0 = VD_Z_SPRING + math.sin(a0) * VD_ARCH_R * 0.96
            sa1 = s_center - math.cos(a1) * VD_ARCH_R
            za1 = VD_Z_SPRING + math.sin(a1) * VD_ARCH_R * 0.96

            lit_f = max(0.0, -math.cos(am)) * 0.72 + 0.18 * math.sin(am)
            cv = fog_mix((0.098 + 0.36 * lit_f, 0.050 + 0.165 * lit_f, 0.060 + 0.070 * lit_f), fog_rgb, z_mid)
            fill_3d_poly(ctx, cam, [
                vd_point(sa0, 0.0, za0),
                vd_point(sa0, VD_THICK, za0),
                vd_point(sa1, VD_THICK, za1),
                vd_point(sa1, 0.0, za1),
            ], (*cv, 0.98 if mirror_z is None else 0.48), mirror_z=mirror_z)


def draw_viaduct_catenary_and_tram(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    mirror_z: float | None = None,
) -> dict:
    """
    Richly hand-painted 1990s Ghibli/Shinkai hero retro Japanese coastal tram car
    (Enoden 300-series inspired!) with shaded painterly volume gradients across its
    vintage cream upper & crimson-vermilion lower body, rounded roof & front cab,
    steel bogie trucks seated cleanly on rails behind the stone parapet, warmly glowing
    layered passenger cabin interiors (ceiling glow, hanging commuter straps, subtle
    natural blurred commuter silhouettes, diagonal sunset glass reflection), destination
    headsign, glowing chrome headlamp, and crisp diamond pantograph on overhead catenary!
    """
    alpha_scale = 1.0 if mirror_z is None else 0.46
    d_track = VD_THICK * 0.44

    # 1. Steel H-beam catenary masts mounted cleanly at every pier center
    mast_s_list = [-9.2, 0.0, 9.2, 18.4, 27.6]
    mast_tops_wire: list[np.ndarray] = []
    mast_tops_msg: list[np.ndarray] = []
    for sm in mast_s_list:
        p_bot = vd_point(sm, 0.25, VD_Z_DECK - 0.15)
        p_top = vd_point(sm, 0.25, VD_WIRE_Z + 1.35)
        p_arm_in = vd_point(sm, 0.25, VD_WIRE_Z + 0.72)
        p_arm_out = vd_point(sm, d_track + 0.55, VD_WIRE_Z + 0.72)
        p_brace = vd_point(sm, 0.25, VD_WIRE_Z + 0.05)
        z_m = float(cam.project_pt(p_top)[2])
        c_steel = fog_mix((0.12, 0.065, 0.070), fog_rgb, z_m)
        c_rim = fog_mix((0.72, 0.36, 0.18), fog_rgb, z_m)

        stroke_3d_line(ctx, cam, p_bot, p_top, 0.18, (*c_steel, 0.96 * alpha_scale), mirror_z=mirror_z, min_px=1.6)
        stroke_3d_line(ctx, cam, p_bot + VD_DIR * 0.07, p_top + VD_DIR * 0.07, 0.05, (*c_rim, 0.62 * alpha_scale), mirror_z=mirror_z, min_px=0.8)
        stroke_3d_line(ctx, cam, p_arm_in, p_arm_out, 0.11, (*c_steel, 0.95 * alpha_scale), mirror_z=mirror_z, min_px=1.2)
        stroke_3d_line(ctx, cam, p_brace, p_arm_out, 0.07, (*c_steel, 0.92 * alpha_scale), mirror_z=mirror_z, min_px=1.0)
        mast_tops_wire.append(vd_point(sm, d_track, VD_WIRE_Z))
        mast_tops_msg.append(vd_point(sm, d_track, VD_WIRE_Z + 0.86))

    # 2. Sweeping overhead contact wire + catenary messenger cable + vertical droppers
    for idx in range(len(mast_s_list) - 1):
        p0w, p1w = mast_tops_wire[idx], mast_tops_wire[idx + 1]
        p0m, p1m = mast_tops_msg[idx], mast_tops_msg[idx + 1]
        z_w = float(cam.project_pt(0.5 * (p0w + p1w))[2])
        c_wire = fog_mix((0.13, 0.068, 0.072), fog_rgb, z_w)

        stroke_3d_line(ctx, cam, p0w, p1w, 0.032, (*c_wire, 0.86 * alpha_scale), mirror_z=mirror_z, min_px=1.0)
        pts_m = []
        for k in range(9):
            tf = k / 8.0
            sag = -0.44 * 4.0 * tf * (1.0 - tf)
            pm = p0m * (1.0 - tf) + p1m * tf + np.array([0.0, 0.0, sag])
            if mirror_z is not None:
                pm = mirror_z_pt(pm, mirror_z)
            u_m, v_m, z_cm = cam.project_pt(pm)
            if z_cm > Z_NEAR:
                pts_m.append((u_m, v_m))
        if len(pts_m) >= 2:
            ctx.set_source_rgba(*c_wire, 0.82 * alpha_scale)
            ctx.set_line_width(1.15)
            ctx.move_to(*pts_m[0])
            for pt in pts_m[1:]:
                ctx.line_to(*pt)
            ctx.stroke()
        if mirror_z is None:
            for k in (2, 4, 6):
                tf = k / 8.0
                sag = -0.44 * 4.0 * tf * (1.0 - tf)
                pm = p0m * (1.0 - tf) + p1m * tf + np.array([0.0, 0.0, sag])
                pw = p0w * (1.0 - tf) + p1w * tf
                stroke_3d_line(ctx, cam, pm, pw, 0.022, (*c_wire, 0.62), min_px=0.65)

    # 3. Hero Retro Japanese Coastal Tram Car (s in [1.15, 13.95], seated cleanly on rails behind viaduct parapet!)
    s_t0 = 1.15
    s_t1 = 13.95
    s_mid = 0.5 * (s_t0 + s_t1)
    d_near = d_track - 1.12
    d_far = d_track + 1.12
    z_bot = VD_TRAM_Z0
    z_belt = z_bot + 1.16
    z_top = VD_TRAM_Z1
    z_roof_c = z_top + 0.34
    z_cam_t = float(cam.project_pt(vd_point(s_mid, d_near, z_belt))[2])

    # Recessed undercarriage equipment & dual steel bogies tucked cleanly down to rail level (VD_Z_TRACK = 5.65)
    c_under = fog_mix((0.068, 0.038, 0.046), fog_rgb, z_cam_t)
    c_wheel = fog_mix((0.14, 0.075, 0.078), fog_rgb, z_cam_t)
    fill_3d_poly(ctx, cam, [
        vd_point(s_t0 + 0.35, d_near + 0.12, VD_Z_TRACK),
        vd_point(s_t1 - 0.35, d_near + 0.12, VD_Z_TRACK),
        vd_point(s_t1 - 0.35, d_near + 0.12, z_bot + 0.08),
        vd_point(s_t0 + 0.35, d_near + 0.12, z_bot + 0.08),
    ], (*c_under, 0.96 * alpha_scale), mirror_z=mirror_z)
    # Visible bogie axle boxes
    if mirror_z is None:
        for sb in (s_t0 + 2.6, s_t1 - 2.6):
            fill_3d_poly(ctx, cam, [
                vd_point(sb - 1.05, d_near + 0.05, VD_Z_TRACK + 0.02),
                vd_point(sb + 1.05, d_near + 0.05, VD_Z_TRACK + 0.02),
                vd_point(sb + 0.95, d_near + 0.05, z_bot + 0.02),
                vd_point(sb - 0.95, d_near + 0.05, z_bot + 0.02),
            ], (*c_wheel, 0.95))

    # Vertical painterly volume gradients for crimson-vermilion lower body & warm cream upper half!
    c_v_hi = fog_mix((0.68, 0.185, 0.128), fog_rgb, z_cam_t)
    c_v_md = fog_mix((0.52, 0.132, 0.108), fog_rgb, z_cam_t)
    c_v_lo = fog_mix((0.30, 0.078, 0.080), fog_rgb, z_cam_t)
    fill_3d_poly_grad(ctx, cam, [
        vd_point(s_t0, d_near, z_bot),
        vd_point(s_t1, d_near, z_bot),
        vd_point(s_t1, d_near, z_belt),
        vd_point(s_t0, d_near, z_belt),
    ], vd_point(s_mid, d_near, z_belt), vd_point(s_mid, d_near, z_bot), [
        (0.0, *c_v_hi, 0.99 * alpha_scale),
        (0.62, *c_v_md, 0.99 * alpha_scale),
        (1.0, *c_v_lo, 0.99 * alpha_scale),
    ], mirror_z=mirror_z)

    c_cr_hi = fog_mix((0.94, 0.80, 0.58), fog_rgb, z_cam_t)
    c_cr_lo = fog_mix((0.76, 0.60, 0.44), fog_rgb, z_cam_t)
    fill_3d_poly_grad(ctx, cam, [
        vd_point(s_t0 + 0.06, d_near, z_belt),
        vd_point(s_t1 - 0.06, d_near, z_belt),
        vd_point(s_t1 - 0.06, d_near, z_top),
        vd_point(s_t0 + 0.06, d_near, z_top),
    ], vd_point(s_mid, d_near, z_top), vd_point(s_mid, d_near, z_belt), [
        (0.0, *c_cr_hi, 0.99 * alpha_scale),
        (1.0, *c_cr_lo, 0.99 * alpha_scale),
    ], mirror_z=mirror_z)

    # Brass-gold waist molding + upper vermilion fascia stripe + dark drip rail
    c_gold = fog_mix((0.98, 0.74, 0.34), fog_rgb, z_cam_t)
    fill_3d_poly(ctx, cam, [
        vd_point(s_t0, d_near, z_belt - 0.04),
        vd_point(s_t1, d_near, z_belt - 0.04),
        vd_point(s_t1, d_near, z_belt + 0.065),
        vd_point(s_t0, d_near, z_belt + 0.065),
    ], (*c_gold, 0.98 * alpha_scale), mirror_z=mirror_z)
    fill_3d_poly(ctx, cam, [
        vd_point(s_t0 + 0.08, d_near, z_top - 0.19),
        vd_point(s_t1 - 0.08, d_near, z_top - 0.19),
        vd_point(s_t1 - 0.08, d_near, z_top),
        vd_point(s_t0 + 0.08, d_near, z_top),
    ], (*c_v_md, 0.98 * alpha_scale), mirror_z=mirror_z)

    # Gently curved vintage domed roof with warm sunset crest rim & clerestory AC pods
    c_roof_lo = fog_mix((0.20, 0.118, 0.128), fog_rgb, z_cam_t)
    c_roof_hi = fog_mix((0.44, 0.24, 0.19), fog_rgb, z_cam_t)
    c_roof_rim = fog_mix((1.00, 0.72, 0.38), fog_rgb, z_cam_t)
    fill_3d_poly_grad(ctx, cam, [
        vd_point(s_t0 + 0.12, d_near, z_top),
        vd_point(s_t1 - 0.12, d_near, z_top),
        vd_point(s_t1 - 0.45, d_track, z_roof_c),
        vd_point(s_t0 + 0.45, d_track, z_roof_c),
    ], vd_point(s_mid, d_track, z_roof_c), vd_point(s_mid, d_near, z_top), [
        (0.0, *c_roof_hi, 0.99 * alpha_scale),
        (1.0, *c_roof_lo, 0.99 * alpha_scale),
    ], mirror_z=mirror_z)
    stroke_3d_line(ctx, cam, vd_point(s_t0 + 0.45, d_track, z_roof_c), vd_point(s_t1 - 0.45, d_track, z_roof_c),
                   0.07, (*c_roof_rim, 0.88 * alpha_scale), mirror_z=mirror_z, min_px=1.6)
    for sv_box in (s_t0 + 3.2, s_t0 + 6.4, s_t0 + 9.6):
        fill_3d_poly(ctx, cam, [
            vd_point(sv_box - 0.72, d_track, z_roof_c - 0.02),
            vd_point(sv_box + 0.72, d_track, z_roof_c - 0.02),
            vd_point(sv_box + 0.62, d_track, z_roof_c + 0.25),
            vd_point(sv_box - 0.62, d_track, z_roof_c + 0.25),
        ], (*c_roof_lo, 0.97 * alpha_scale), mirror_z=mirror_z)
        stroke_3d_line(ctx, cam, vd_point(sv_box - 0.62, d_track, z_roof_c + 0.25),
                       vd_point(sv_box + 0.62, d_track, z_roof_c + 0.25), 0.035, (*c_roof_rim, 0.75 * alpha_scale), mirror_z=mirror_z, min_px=1.0)

    # Sunlit curved front cab nose (s = s_t1) with windshield, destination board & chrome headlamp!
    c_nose_v = fog_mix((0.80, 0.26, 0.16), fog_rgb, z_cam_t)
    c_nose_c = fog_mix((1.00, 0.86, 0.62), fog_rgb, z_cam_t)
    fill_3d_poly(ctx, cam, [
        vd_point(s_t1, d_near, z_bot),
        vd_point(s_t1 + 0.16, d_track, z_bot),
        vd_point(s_t1, d_far, z_bot),
        vd_point(s_t1, d_far, z_belt),
        vd_point(s_t1 + 0.16, d_track, z_belt),
        vd_point(s_t1, d_near, z_belt),
    ], (*c_nose_v, 0.99 * alpha_scale), mirror_z=mirror_z)
    fill_3d_poly(ctx, cam, [
        vd_point(s_t1 - 0.06, d_near, z_belt),
        vd_point(s_t1 + 0.12, d_track, z_belt),
        vd_point(s_t1 - 0.06, d_far, z_belt),
        vd_point(s_t1 - 0.06, d_far, z_top),
        vd_point(s_t1 + 0.08, d_track, z_top),
        vd_point(s_t1 - 0.06, d_near, z_top),
    ], (*c_nose_c, 0.99 * alpha_scale), mirror_z=mirror_z)
    # Cab front glowing windshield glass
    fill_3d_poly(ctx, cam, [
        vd_point(s_t1 + 0.04, d_near + 0.26, z_belt + 0.22),
        vd_point(s_t1 + 0.08, d_far - 0.26, z_belt + 0.22),
        vd_point(s_t1 + 0.05, d_far - 0.26, z_top - 0.28),
        vd_point(s_t1 + 0.01, d_near + 0.26, z_top - 0.28),
    ], (*fog_mix((0.98, 0.78, 0.42), fog_rgb, z_cam_t * 0.5), 0.95 * alpha_scale), mirror_z=mirror_z)

    # Row of 8 warmly glowing layered passenger windows & recessed sliding doors (NO chess-pawn icons!)
    win_uvs: list[tuple[float, float]] = []
    window_spans = [
        (s_t0 + 0.52, s_t0 + 1.66, False),
        (s_t0 + 2.04, s_t0 + 3.20, True),
        (s_t0 + 3.60, s_t0 + 4.82, False),
        (s_t0 + 5.14, s_t0 + 6.36, False),
        (s_t0 + 6.68, s_t0 + 7.90, False),
        (s_t0 + 8.22, s_t0 + 9.44, False),
        (s_t0 + 9.84, s_t0 + 11.00, True),
        (s_t0 + 11.34, s_t0 + 12.36, False),
    ]
    z_w0 = z_belt + 0.16
    z_w1 = z_top - 0.26
    for idx_w, (ws0, ws1, is_dr) in enumerate(window_spans):
        # Outer dark anodized aluminum window gasket frame
        fill_3d_poly(ctx, cam, [
            vd_point(ws0 - 0.05, d_near, (z_bot + 0.14 if is_dr else z_w0) - 0.05),
            vd_point(ws1 + 0.05, d_near, (z_bot + 0.14 if is_dr else z_w0) - 0.05),
            vd_point(ws1 + 0.05, d_near, z_w1 + 0.05),
            vd_point(ws0 - 0.05, d_near, z_w1 + 0.05),
        ], (*fog_mix((0.15, 0.072, 0.068), fog_rgb, z_cam_t), 0.97 * alpha_scale), mirror_z=mirror_z)

        # Recessed sliding door lower crimson panel with dark vertical center seam
        if is_dr:
            fill_3d_poly(ctx, cam, [
                vd_point(ws0, d_near, z_bot + 0.16),
                vd_point(ws1, d_near, z_bot + 0.16),
                vd_point(ws1, d_near, z_w0),
                vd_point(ws0, d_near, z_w0),
            ], (*c_v_lo, 0.98 * alpha_scale), mirror_z=mirror_z)

        # Warm glowing cabin interior gradient (bright golden ceiling fluorescent tubes -> deeper amber bench level!)
        c_wg_top = fog_mix((1.05, 0.93, 0.60), fog_rgb, z_cam_t * 0.40)
        c_wg_mid = fog_mix((1.00, 0.78, 0.36), fog_rgb, z_cam_t * 0.42)
        c_wg_bot = fog_mix((0.82, 0.48, 0.20), fog_rgb, z_cam_t * 0.48)
        poly_w = fill_3d_poly_grad(ctx, cam, [
            vd_point(ws0, d_near, z_w0),
            vd_point(ws1, d_near, z_w0),
            vd_point(ws1, d_near, z_w1),
            vd_point(ws0, d_near, z_w1),
        ], vd_point(0.5 * (ws0 + ws1), d_near, z_w1), vd_point(0.5 * (ws0 + ws1), d_near, z_w0), [
            (0.0, *c_wg_top, 0.99 * alpha_scale),
            (0.45, *c_wg_mid, 0.99 * alpha_scale),
            (1.0, *c_wg_bot, 0.99 * alpha_scale),
        ], mirror_z=mirror_z)

        if mirror_z is None and len(poly_w) >= 4:
            # Subtle interior commuter depth (overhead hanging hand-straps + soft seated window bench + subtle diagonal glass glare)
            ctx.save()
            ctx.move_to(*poly_w[0])
            for pt in poly_w[1:]:
                ctx.line_to(*pt)
            ctx.close_path()
            ctx.clip()
            # Hanging commuter straps (tsurikawa) near upper cabin!
            for st_f in (0.28, 0.56, 0.80):
                su_t, sv_t, _ = cam.project_pt(vd_point(ws0 + st_f * (ws1 - ws0), d_near, z_w1))
                su_b, sv_b, _ = cam.project_pt(vd_point(ws0 + st_f * (ws1 - ws0), d_near, z_w1 - 0.34))
                ctx.set_source_rgba(0.32, 0.14, 0.08, 0.42)
                ctx.set_line_width(0.85)
                ctx.move_to(su_t, sv_t)
                ctx.line_to(su_b, sv_b)
                ctx.stroke()
                ctx.arc(su_b, sv_b + 1.8, 1.8, 0, math.tau)
                ctx.stroke()
            # Soft painterly commuter silhouette in 2 windows (rounded natural head & slumped shoulders + overhead luggage rail!)
            if idx_w in (2, 5):
                ps = ws0 + (0.40 if idx_w == 2 else 0.58) * (ws1 - ws0)
                pu, pv, _ = cam.project_pt(vd_point(ps, d_near, z_w0 + 0.06))
                # Soft seated commuter: gentle rounded head + relaxed curved shoulders (low painterly opacity)
                ctx.set_source_rgba(0.36, 0.14, 0.075, 0.34)
                ctx.arc(pu, pv - 11.5, 4.4, 0, math.tau)
                ctx.fill()
                ctx.move_to(pu - 10.5, pv + 4.0)
                ctx.curve_to(pu - 9.5, pv - 4.5, pu - 5.5, pv - 7.0, pu, pv - 6.8)
                ctx.curve_to(pu + 5.5, pv - 7.0, pu + 9.5, pv - 4.5, pu + 10.5, pv + 4.0)
                ctx.close_path()
                ctx.fill()
            # Diagonal sunset reflection shimmer across exterior pane
            p_tl = cam.project_pt(vd_point(ws0 + 0.10 * (ws1 - ws0), d_near, z_w1))
            p_tr = cam.project_pt(vd_point(ws0 + 0.38 * (ws1 - ws0), d_near, z_w1))
            p_bl = cam.project_pt(vd_point(ws0, d_near, z_w0))
            p_br = cam.project_pt(vd_point(ws0 + 0.18 * (ws1 - ws0), d_near, z_w0))
            ctx.set_source_rgba(1.0, 0.92, 0.74, 0.18)
            ctx.move_to(p_tl[0], p_tl[1])
            ctx.line_to(p_tr[0], p_tr[1])
            ctx.line_to(p_br[0], p_br[1])
            ctx.line_to(p_bl[0], p_bl[1])
            ctx.close_path()
            ctx.fill()
            ctx.restore()

            # Horizontal sliding transom bar + vertical sliding-door center mullion
            stroke_3d_line(ctx, cam, vd_point(ws0, d_near, z_w0 + 0.72 * (z_w1 - z_w0)),
                           vd_point(ws1, d_near, z_w0 + 0.72 * (z_w1 - z_w0)), 0.042, (0.20, 0.092, 0.068, 0.88), min_px=0.9)
            if is_dr:
                sm_dr = 0.5 * (ws0 + ws1)
                stroke_3d_line(ctx, cam, vd_point(sm_dr, d_near, z_bot + 0.16),
                               vd_point(sm_dr, d_near, z_w1), 0.052, (0.14, 0.068, 0.060, 0.94), min_px=1.1)
            wu, wv, _ = cam.project_uv_norm(vd_point(0.5 * (ws0 + ws1), d_near, 0.5 * (z_w0 + z_w1)))
            win_uvs.append((wu, wv))

    # Subtle cream brass car-number plate on lower crimson side center ("305")
    if mirror_z is None:
        u_num, v_num, _ = cam.project_pt(vd_point(s_mid, d_near, z_bot + 0.62))
        ctx.save()
        ctx.select_font_face("DejaVu Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(9.5)
        ctx.set_source_rgba(0.96, 0.80, 0.44, 0.86)
        ctx.move_to(u_num - 10, v_num + 3)
        ctx.show_text("305")
        ctx.restore()

    # Warm glowing vintage headlamp on front nose brow
    hl_pt = vd_point(s_t1 + 0.06, d_track, z_top + 0.14)
    hl_u, hl_v, _ = cam.project_uv_norm(hl_pt)
    if mirror_z is None:
        hx, hy_p, _ = cam.project_pt(hl_pt)
        g_hl = cairo.RadialGradient(hx, hy_p, 1.0, hx, hy_p, 26.0)
        g_hl.add_color_stop_rgba(0.0, 1.0, 0.96, 0.80, 0.98)
        g_hl.add_color_stop_rgba(0.32, 1.0, 0.78, 0.40, 0.58)
        g_hl.add_color_stop_rgba(1.0, 1.0, 0.52, 0.18, 0.0)
        ctx.set_source(g_hl)
        ctx.arc(hx, hy_p, 26.0, 0, math.tau)
        ctx.fill()

    # Rooftop Diamond-Frame Steel Pantograph with ceramic insulators touching overhead contact wire!
    s_pan = s_t0 + 3.25
    z_p0 = z_roof_c + 0.05
    z_pm = 0.5 * (z_p0 + VD_WIRE_Z)
    z_p1 = VD_WIRE_Z
    c_pan = fog_mix((0.10, 0.052, 0.058), fog_rgb, z_cam_t)
    p_b0 = vd_point(s_pan - 0.58, d_track, z_p0)
    p_b1 = vd_point(s_pan + 0.58, d_track, z_p0)
    p_m0 = vd_point(s_pan - 1.05, d_track, z_pm)
    p_m1 = vd_point(s_pan + 1.05, d_track, z_pm)
    p_t0 = vd_point(s_pan - 0.34, d_track, z_p1)
    p_t1 = vd_point(s_pan + 0.34, d_track, z_p1)
    for pa, pb in ((p_b0, p_b1), (p_b0, p_m0), (p_b1, p_m1), (p_m0, p_t0), (p_m1, p_t1), (p_t0 - VD_DIR * 0.42, p_t1 + VD_DIR * 0.42)):
        stroke_3d_line(ctx, cam, pa, pb, 0.058, (*c_pan, 0.96 * alpha_scale), mirror_z=mirror_z, min_px=1.2)

    u_t0, v_t0, _ = cam.project_uv_norm(vd_point(s_t0, d_near, z_top))
    u_t1, v_t1, _ = cam.project_uv_norm(vd_point(s_t1, d_near, z_bot))
    return {
        "tram_u": (float(u_t0), float(u_t1)),
        "tram_v": (float(v_t0), float(v_t1)),
        "tram_win_uvs": win_uvs,
        "tram_headlamp_uv": (float(hl_u), float(hl_v)),
    }


# =============================================================================
# Stage 3 Band 2 — Fumikiri Crossing, Bakery 3D Craft/Props & Utility Pole
# =============================================================================

def draw_fumikiri_crossing(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    mirror_z: float | None = None,
) -> dict:
    """
    Authentic detailed Japanese seaside railway crossing (fumikiri):
    - Beveled steel barrier motor housing with yellow-black warning chevrons & pivot gear drum
    - Striped vertical steel mast, top cast-iron bell dome, crisp yellow-black X-crossbuck
    - Black train-direction indicator box with glowing amber arrow (←)
    - Twin hooded crimson-amber warning lamps with projecting curved black sun-visors & Fresnel glints
    - Clean authentic yellow-black striped crossing boom (遮断桿) with iron counterweight discs
      and sleek red reflective end-cap (NO clunky floating triangles!)
    - Properly striped companion far-side crossing post on right side of track
    """
    a_s = 1.0 if mirror_z is None else 0.56
    z_m = float(cam.project_pt((FUMI_X, FUMI_Y, 1.8))[2])

    c_conc = fog_mix((0.20, 0.135, 0.125), fog_rgb, z_m)
    c_box = fog_mix((0.15, 0.095, 0.092), fog_rgb, z_m)
    c_box_sun = fog_mix((0.46, 0.25, 0.15), fog_rgb, z_m)
    c_yel = fog_mix((0.92, 0.66, 0.20), fog_rgb, z_m)
    c_blk = fog_mix((0.080, 0.048, 0.054), fog_rgb, z_m)

    # 1. Concrete foundation pedestal & detailed chamfered barrier mechanism box
    fill_3d_poly(ctx, cam, [
        (FUMI_X - 0.24, FUMI_Y, Z_CURB - 0.04),
        (FUMI_X + 0.24, FUMI_Y, Z_CURB - 0.04),
        (FUMI_X + 0.21, FUMI_Y, Z_CURB + 0.34),
        (FUMI_X - 0.21, FUMI_Y, Z_CURB + 0.34),
    ], (*c_conc, 0.99 * a_s), mirror_z=mirror_z)

    bx = FUMI_X + 0.34
    fill_3d_poly_grad(ctx, cam, [
        (bx - 0.17, FUMI_Y, Z_LANE),
        (bx + 0.17, FUMI_Y, Z_LANE),
        (bx + 0.15, FUMI_Y, 1.18),
        (bx - 0.15, FUMI_Y, 1.18),
    ], (bx - 0.17, FUMI_Y, 0.6), (bx + 0.17, FUMI_Y, 0.6), [
        (0.0, *c_box, 0.99 * a_s),
        (0.75, *c_box, 0.99 * a_s),
        (1.0, *c_box_sun, 0.99 * a_s),
    ], mirror_z=mirror_z)
    # Yellow warning band & steel access door seam on motor box
    if mirror_z is None:
        fill_3d_poly(ctx, cam, [
            (bx - 0.155, FUMI_Y - 0.01, 0.14),
            (bx + 0.155, FUMI_Y - 0.01, 0.14),
            (bx + 0.155, FUMI_Y - 0.01, 0.28),
            (bx - 0.155, FUMI_Y - 0.01, 0.28),
        ], (*c_yel, 0.92))
        stroke_3d_line(ctx, cam, (bx - 0.14, FUMI_Y - 0.01, 1.18), (bx + 0.14, FUMI_Y - 0.01, 1.18), 0.04, (*c_box_sun, 0.90), min_px=1.6)

    # 2. Striped vertical steel mast (z = 0.44 .. 3.44 m)
    n_stripes = 12
    z_base_m, z_top_m = 0.44, 3.44
    dz_s = (z_top_m - z_base_m) / n_stripes
    for k in range(n_stripes):
        za = z_base_m + k * dz_s
        zb = za + dz_s
        col_s = c_yel if (k % 2 == 0) else c_blk
        fill_3d_poly(ctx, cam, [
            (FUMI_X - 0.068, FUMI_Y, za),
            (FUMI_X + 0.068, FUMI_Y, za),
            (FUMI_X + 0.068, FUMI_Y, zb),
            (FUMI_X - 0.068, FUMI_Y, zb),
        ], (*col_s, 0.99 * a_s), mirror_z=mirror_z)
    stroke_3d_line(ctx, cam, (FUMI_X + 0.068, FUMI_Y, z_base_m), (FUMI_X + 0.068, FUMI_Y, z_top_m),
                   0.022, (0.98, 0.72, 0.36, 0.56 * a_s), mirror_z=mirror_z, min_px=0.9)

    # Top electric warning bell dome
    p_bell = mirror_z_pt((FUMI_X, FUMI_Y, 3.55), mirror_z) if mirror_z is not None else np.array([FUMI_X, FUMI_Y, 3.55])
    ub, vb, zb = cam.project_pt(p_bell)
    if zb > Z_NEAR:
        r_b = max(3.0, float(cam.K[0, 0] * 0.105 / zb))
        ctx.set_source_rgba(*c_blk, 0.96 * a_s)
        ctx.arc(ub, vb, r_b, 0, math.tau)
        ctx.fill()
        ctx.set_source_rgba(0.86, 0.50, 0.24, 0.65 * a_s)
        ctx.arc(ub + r_b * 0.25, vb - r_b * 0.15, r_b * 0.45, 0, math.tau)
        ctx.fill()

    # 3. Crisp X-shaped crossbuck sign (踏切警標)
    zc_x = 3.06
    arm_dx, arm_dz = 0.42, 0.28
    for sgn in (-1.0, 1.0):
        pa = np.array([FUMI_X - arm_dx, FUMI_Y - 0.04, zc_x - sgn * arm_dz])
        pb = np.array([FUMI_X + arm_dx, FUMI_Y - 0.04, zc_x + sgn * arm_dz])
        stroke_3d_line(ctx, cam, pa, pb, 0.155, (*c_blk, 0.98 * a_s), mirror_z=mirror_z, min_px=3.0)
        stroke_3d_line(ctx, cam, pa, pb, 0.118, (*c_yel, 0.98 * a_s), mirror_z=mirror_z, min_px=2.2)
        for tf in (0.22, 0.50, 0.78):
            pm0 = pa * (1.0 - (tf - 0.06)) + pb * (tf - 0.06)
            pm1 = pa * (1.0 - (tf + 0.06)) + pb * (tf + 0.06)
            stroke_3d_line(ctx, cam, pm0, pm1, 0.118, (*c_blk, 0.94 * a_s), mirror_z=mirror_z, min_px=1.6)

    # 4. Black train-direction indicator box with glowing amber arrow right under twin lamps (z = 2.02..2.22)
    if mirror_z is None:
        fill_3d_poly(ctx, cam, [
            (FUMI_X - 0.24, FUMI_Y - 0.04, 2.04),
            (FUMI_X + 0.24, FUMI_Y - 0.04, 2.04),
            (FUMI_X + 0.24, FUMI_Y - 0.04, 2.24),
            (FUMI_X - 0.24, FUMI_Y - 0.04, 2.24),
        ], (*c_blk, 0.97))
        # Illuminated warm amber right-pointing arrow (→ matching tram direction!)
        stroke_3d_line(ctx, cam, (FUMI_X - 0.14, FUMI_Y - 0.05, 2.14), (FUMI_X + 0.14, FUMI_Y - 0.05, 2.14), 0.038, (1.0, 0.68, 0.22, 0.94), min_px=1.6)
        stroke_3d_line(ctx, cam, (FUMI_X + 0.06, FUMI_Y - 0.05, 2.20), (FUMI_X + 0.14, FUMI_Y - 0.05, 2.14), 0.034, (1.0, 0.68, 0.22, 0.94), min_px=1.5)
        stroke_3d_line(ctx, cam, (FUMI_X + 0.06, FUMI_Y - 0.05, 2.08), (FUMI_X + 0.14, FUMI_Y - 0.05, 2.14), 0.034, (1.0, 0.68, 0.22, 0.94), min_px=1.5)

    # 5. Twin Hooded Crimson-Amber Warning Lamps with curved black sun-visors!
    stroke_3d_line(ctx, cam, (FUMI_X - 0.32, FUMI_Y - 0.03, FUMI_LAMP_Z),
                   (FUMI_X + 0.32, FUMI_Y - 0.03, FUMI_LAMP_Z), 0.068, (*c_blk, 0.98 * a_s), mirror_z=mirror_z, min_px=2.0)

    lamp_uvs = []
    for idx_l, lpt in enumerate((FUMI_LAMP_L, FUMI_LAMP_R)):
        pt_use = mirror_z_pt(lpt, mirror_z) if mirror_z is not None else np.asarray(lpt, dtype=np.float64)
        ul, vl, zl = cam.project_pt(pt_use)
        if mirror_z is None:
            lamp_uvs.append((ul / W, vl / H))
        if zl <= Z_NEAR:
            continue
        r_l = max(4.0, float(cam.K[0, 0] * 0.125 / zl))
        # Cast-iron target backboard
        ctx.set_source_rgba(*c_blk, 0.98 * a_s)
        ctx.arc(ul, vl, r_l * 1.48, 0, math.tau)
        ctx.fill()
        # Alternating intensity (left bright crimson-amber, right glowing warm secondary ember!)
        inten = 1.08 if idx_l == 0 else 0.72
        g_h = cairo.RadialGradient(ul, vl, r_l * 0.15, ul, vl, r_l * (4.9 if mirror_z is None else 6.4))
        g_h.add_color_stop_rgba(0.0, 1.0, 0.90, 0.58, 0.98 * a_s * inten)
        g_h.add_color_stop_rgba(0.24, 1.0, 0.32, 0.13, 0.88 * a_s * inten)
        g_h.add_color_stop_rgba(0.56, 0.92, 0.15, 0.08, 0.36 * a_s * inten)
        g_h.add_color_stop_rgba(1.0, 0.80, 0.10, 0.06, 0.0)
        ctx.set_source(g_h)
        ctx.arc(ul, vl, r_l * (4.9 if mirror_z is None else 6.4), 0, math.tau)
        ctx.fill()
        # Glowing Fresnel glass lens
        ctx.set_source_rgba(1.0, 0.40 if idx_l == 0 else 0.26, 0.16 if idx_l == 0 else 0.11, 0.98 * a_s)
        ctx.arc(ul, vl, r_l * 0.86, 0, math.tau)
        ctx.fill()
        ctx.set_source_rgba(1.0, 0.94, 0.74, 0.92 * a_s * inten)
        ctx.arc(ul, vl, r_l * 0.42, 0, math.tau)
        ctx.fill()
        # Projecting black curved sun-visor hood over top arc of lens!
        if mirror_z is None:
            ctx.set_source_rgba(0.055, 0.032, 0.038, 0.95)
            ctx.set_line_width(max(2.0, r_l * 0.32))
            ctx.arc(ul, vl - r_l * 0.12, r_l * 1.08, math.pi * 1.08, math.pi * 1.92)
            ctx.stroke()

    # 6. Authentic Japanese Yellow-Black Striped Barrier Boom (遮断桿) with circular counterweight & sleek red tip!
    x_piv = bx
    z_piv = FUMI_BARRIER_Z
    # Pivot gear-wheel hub
    p_hub = mirror_z_pt((x_piv, FUMI_Y - 0.09, z_piv), mirror_z) if mirror_z is not None else np.array([x_piv, FUMI_Y - 0.09, z_piv])
    uh, vh, zh = cam.project_pt(p_hub)
    if zh > Z_NEAR:
        rh = max(3.5, float(cam.K[0, 0] * 0.11 / zh))
        ctx.set_source_rgba(*c_blk, 0.98 * a_s)
        ctx.arc(uh, vh, rh, 0, math.tau)
        ctx.fill()
        ctx.set_source_rgba(*c_yel, 0.92 * a_s)
        ctx.arc(uh, vh, rh * 0.55, 0, math.tau)
        ctx.fill()

    # Counterweight arm + circular cast-iron weight drum on left
    stroke_3d_line(ctx, cam, (x_piv - 0.56, FUMI_Y - 0.08, z_piv - 0.02),
                   (x_piv + 0.06, FUMI_Y - 0.08, z_piv), 0.09, (*c_blk, 0.98 * a_s), mirror_z=mirror_z, min_px=2.2)
    p_cw = mirror_z_pt((x_piv - 0.48, FUMI_Y - 0.08, z_piv - 0.02), mirror_z) if mirror_z is not None else np.array([x_piv - 0.48, FUMI_Y - 0.08, z_piv - 0.02])
    ucw, vcw, zcw = cam.project_pt(p_cw)
    if zcw > Z_NEAR:
        rcw = max(4.0, float(cam.K[0, 0] * 0.15 / zcw))
        ctx.set_source_rgba(*c_yel, 0.98 * a_s)
        ctx.arc(ucw, vcw, rcw, 0, math.tau)
        ctx.fill()
        ctx.set_source_rgba(*c_blk, 0.95 * a_s)
        ctx.set_line_width(2.0)
        ctx.arc(ucw, vcw, rcw, 0, math.tau)
        ctx.stroke()

    n_bseg = 12
    for k in range(n_bseg):
        t0 = k / n_bseg
        t1 = (k + 1) / n_bseg
        xa = x_piv + t0 * (FUMI_BARRIER_X1 - x_piv)
        xb = x_piv + t1 * (FUMI_BARRIER_X1 - x_piv)
        za = z_piv + t0 * 0.13
        zb = z_piv + t1 * 0.13
        w_a = 0.054 * (1.0 - 0.34 * t0)
        w_b = 0.054 * (1.0 - 0.34 * t1)
        col_b = c_yel if (k % 2 == 0) else c_blk
        fill_3d_poly(ctx, cam, [
            (xa, FUMI_Y - 0.08, za - w_a),
            (xb, FUMI_Y - 0.08, zb - w_b),
            (xb, FUMI_Y - 0.08, zb + w_b),
            (xa, FUMI_Y - 0.08, za + w_a),
        ], (*col_b, 0.99 * a_s), mirror_z=mirror_z)
        # Top warm golden specular edge along barrier rod
        if mirror_z is None:
            stroke_3d_line(ctx, cam, (xa, FUMI_Y - 0.08, za + w_a), (xb, FUMI_Y - 0.08, zb + w_b),
                           0.012, (1.0, 0.80, 0.42, 0.55), min_px=0.8)

    # Subtle glowing red reflective safety marker lamp at center & tip of boom (sleek authentic Japanese barrier!)
    if mirror_z is None:
        for t_m in (0.48, 0.96):
            xm = x_piv + t_m * (FUMI_BARRIER_X1 - x_piv)
            zm = z_piv + t_m * 0.13
            um, vm, zm_c = cam.project_pt((xm, FUMI_Y - 0.09, zm))
            if zm_c > Z_NEAR:
                gm = cairo.RadialGradient(um, vm, 0.5, um, vm, 9.0)
                gm.add_color_stop_rgba(0.0, 1.0, 0.55, 0.28, 0.90)
                gm.add_color_stop_rgba(1.0, 0.95, 0.18, 0.08, 0.0)
                ctx.set_source(gm)
                ctx.arc(um, vm, 9.0, 0, math.tau)
                ctx.fill()

    # 7. Companion far-side crossing post on right side of track (x = 2.32, y = 16.95) grounded with concrete plinth!
    fx, fy = 2.32, 16.95
    zf_m = float(cam.project_pt((fx, fy, 1.5))[2])
    c_yel2 = fog_mix((0.86, 0.60, 0.20), fog_rgb, zf_m)
    c_blk2 = fog_mix((0.085, 0.050, 0.056), fog_rgb, zf_m)
    c_pl2 = fog_mix((0.18, 0.12, 0.115), fog_rgb, zf_m)
    fill_3d_poly(ctx, cam, [
        (fx - 0.14, fy, Z_LANE),
        (fx + 0.14, fy, Z_LANE),
        (fx + 0.12, fy, 0.26),
        (fx - 0.12, fy, 0.26),
    ], (*c_pl2, 0.99 * a_s), mirror_z=mirror_z)
    for k in range(8):
        za = 0.24 + k * 0.265
        zb = za + 0.265
        stroke_3d_line(ctx, cam, (fx, fy, za), (fx, fy, zb), 0.105, (*(c_yel2 if k % 2 == 0 else c_blk2), 0.97 * a_s), mirror_z=mirror_z, min_px=2.0)
    for sgn in (-1.0, 1.0):
        pa = np.array([fx - 0.30, fy, 2.22 - sgn * 0.20])
        pb = np.array([fx + 0.30, fy, 2.22 + sgn * 0.20])
        stroke_3d_line(ctx, cam, pa, pb, 0.125, (*c_blk2, 0.96 * a_s), mirror_z=mirror_z, min_px=2.4)
        stroke_3d_line(ctx, cam, pa, pb, 0.092, (*c_yel2, 0.96 * a_s), mirror_z=mirror_z, min_px=1.7)
        for tf in (0.24, 0.50, 0.76):
            pm0 = pa * (1.0 - (tf - 0.06)) + pb * (tf - 0.06)
            pm1 = pa * (1.0 - (tf + 0.06)) + pb * (tf + 0.06)
            stroke_3d_line(ctx, cam, pm0, pm1, 0.092, (*c_blk2, 0.92 * a_s), mirror_z=mirror_z, min_px=1.2)

    u_post, v_post_b, _ = cam.project_uv_norm((FUMI_X, FUMI_Y, 0.0))
    _, v_post_t, _ = cam.project_uv_norm((FUMI_X, FUMI_Y, 3.6))
    return {
        "fumi_lamp_l_uv": lamp_uvs[0] if len(lamp_uvs) > 0 else (0.376, 0.421),
        "fumi_lamp_r_uv": lamp_uvs[1] if len(lamp_uvs) > 1 else (0.403, 0.421),
        "fumi_post_u": float(u_post),
        "fumi_post_v": (float(v_post_t), float(v_post_b)),
    }


def draw_utility_pole_and_wires(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    mirror_z: float | None = None,
) -> None:
    """
    Detailed weathered dark-cedar Japanese utility pole with steel climbing pegs,
    finned cylindrical pole transformer, enamel street plaque, double crossarms,
    porcelain insulators & sweeping multi-strand cubic-Bezier overhead wires!
    """
    a_s = 1.0 if mirror_z is None else 0.45
    z_m = float(cam.project_pt((POLE_X, POLE_Y, 2.6))[2])
    c_pole = fog_mix((0.092, 0.054, 0.054), fog_rgb, z_m)
    c_pole_mid = fog_mix((0.155, 0.088, 0.078), fog_rgb, z_m)
    c_rim = fog_mix((0.72, 0.36, 0.16), fog_rgb, z_m)
    c_iron = fog_mix((0.115, 0.070, 0.074), fog_rgb, z_m)
    c_ins = fog_mix((0.90, 0.84, 0.76), fog_rgb, z_m)

    # Shaded cylindrical timber pole body
    fill_3d_poly_grad(ctx, cam, [
        (POLE_X - 0.145, POLE_Y, Z_CURB - 0.02),
        (POLE_X + 0.145, POLE_Y, Z_CURB - 0.02),
        (POLE_X + 0.105, POLE_Y, POLE_Z_TOP),
        (POLE_X - 0.105, POLE_Y, POLE_Z_TOP),
    ], (POLE_X - 0.145, POLE_Y, 2.5), (POLE_X + 0.145, POLE_Y, 2.5), [
        (0.0, *c_pole, 0.99 * a_s),
        (0.68, *c_pole_mid, 0.99 * a_s),
        (1.0, *c_rim, 0.99 * a_s),
    ], mirror_z=mirror_z)

    # Climbing step bolts + authentic Japanese enamel street address plaque ("海辺通") + grounded base shadow!
    if mirror_z is None:
        up_b, vp_b, _ = cam.project_pt((POLE_X, POLE_Y, Z_CURB + 0.01))
        g_ps = cairo.RadialGradient(up_b, vp_b, 4.0, up_b, vp_b, 38.0)
        g_ps.add_color_stop_rgba(0.0, 0.03, 0.015, 0.02, 0.52)
        g_ps.add_color_stop_rgba(1.0, 0.03, 0.015, 0.02, 0.0)
        ctx.set_source(g_ps)
        ctx.save()
        ctx.translate(up_b, vp_b + 2)
        ctx.scale(1.4, 0.42)
        ctx.arc(0, 0, 34.0, 0, math.tau)
        ctx.fill()
        ctx.restore()

        for idx_p, zp in enumerate((2.15, 2.55, 2.95, 3.35, 3.75)):
            side = 1.0 if idx_p % 2 == 0 else -1.0
            stroke_3d_line(ctx, cam, (POLE_X + side * 0.12, POLE_Y, zp), (POLE_X + side * 0.24, POLE_Y, zp + 0.02),
                           0.026, (*c_iron, 0.92), min_px=1.2)
        fill_3d_poly(ctx, cam, [
            (POLE_X - 0.075, POLE_Y - 0.08, 1.52),
            (POLE_X + 0.075, POLE_Y - 0.08, 1.52),
            (POLE_X + 0.075, POLE_Y - 0.08, 1.98),
            (POLE_X - 0.075, POLE_Y - 0.08, 1.98),
        ], (0.88, 0.80, 0.64, 0.94))
        fill_3d_poly(ctx, cam, [
            (POLE_X - 0.056, POLE_Y - 0.085, 1.55),
            (POLE_X + 0.056, POLE_Y - 0.085, 1.55),
            (POLE_X + 0.056, POLE_Y - 0.085, 1.95),
            (POLE_X - 0.056, POLE_Y - 0.085, 1.95),
        ], (0.14, 0.11, 0.14, 0.92))
        ctx.save()
        ctx.select_font_face("Noto Sans CJK JP", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(11.5)
        ctx.set_source_rgba(0.95, 0.88, 0.74, 0.94)
        for idx_c, ch_s in enumerate(("海", "辺", "通")):
            uc_s, vc_s, _ = cam.project_pt((POLE_X, POLE_Y - 0.09, 1.87 - idx_c * 0.125))
            ex = ctx.text_extents(ch_s)
            ctx.move_to(uc_s - ex.width * 0.5 - ex.x_bearing, vc_s - ex.height * 0.5 - ex.y_bearing)
            ctx.show_text(ch_s)
        ctx.restore()

    # Detailed cylindrical pole transformer with vertical cooling fins & sunlit rim
    fill_3d_poly_grad(ctx, cam, [
        (POLE_X + 0.08, POLE_Y - 0.10, 4.15),
        (POLE_X + 0.56, POLE_Y - 0.10, 4.15),
        (POLE_X + 0.56, POLE_Y - 0.10, 4.92),
        (POLE_X + 0.08, POLE_Y - 0.10, 4.92),
    ], (POLE_X + 0.08, POLE_Y - 0.10, 4.5), (POLE_X + 0.56, POLE_Y - 0.10, 4.5), [
        (0.0, *c_iron, 0.99 * a_s),
        (0.76, *fog_mix((0.20, 0.118, 0.110), fog_rgb, z_m), 0.99 * a_s),
        (1.0, *c_rim, 0.99 * a_s),
    ], mirror_z=mirror_z)
    if mirror_z is None:
        for fin_x in (0.18, 0.28, 0.38, 0.48):
            stroke_3d_line(ctx, cam, (POLE_X + fin_x, POLE_Y - 0.11, 4.22), (POLE_X + fin_x, POLE_Y - 0.11, 4.86),
                           0.018, (0.05, 0.028, 0.032, 0.72), min_px=0.9)
        # Twin ceramic bushings atop transformer
        for bx_t in (0.22, 0.42):
            stroke_3d_line(ctx, cam, (POLE_X + bx_t, POLE_Y - 0.10, 4.92), (POLE_X + bx_t, POLE_Y - 0.10, 5.12),
                           0.045, (*c_ins, 0.92), min_px=1.5)

    # Double steel crossarms, diagonal braces & white porcelain pin insulators
    ins_pts_hi: list[np.ndarray] = []
    ins_pts_lo: list[np.ndarray] = []
    for z_arm, half_w, store in ((5.85, 0.88, ins_pts_hi), (5.15, 0.72, ins_pts_lo)):
        stroke_3d_line(ctx, cam, (POLE_X - half_w, POLE_Y, z_arm), (POLE_X + half_w, POLE_Y, z_arm),
                       0.09, (*c_iron, 0.98 * a_s), mirror_z=mirror_z, min_px=2.0)
        stroke_3d_line(ctx, cam, (POLE_X, POLE_Y, z_arm - 0.45), (POLE_X + half_w * 0.62, POLE_Y, z_arm),
                       0.042, (*c_iron, 0.92 * a_s), mirror_z=mirror_z, min_px=1.0)
        stroke_3d_line(ctx, cam, (POLE_X, POLE_Y, z_arm - 0.45), (POLE_X - half_w * 0.62, POLE_Y, z_arm),
                       0.042, (*c_iron, 0.92 * a_s), mirror_z=mirror_z, min_px=1.0)
        for dx_i in (-half_w * 0.85, -half_w * 0.36, half_w * 0.36, half_w * 0.85):
            pt_i = np.array([POLE_X + dx_i, POLE_Y, z_arm + 0.11])
            store.append(pt_i)
            pi_use = mirror_z_pt(pt_i, mirror_z) if mirror_z is not None else pt_i
            ui, vi, zi = cam.project_pt(pi_use)
            if zi > Z_NEAR:
                ri = max(1.8, float(cam.K[0, 0] * 0.048 / zi))
                ctx.set_source_rgba(*c_ins, 0.95 * a_s)
                ctx.arc(ui, vi, ri, 0, math.tau)
                ctx.fill()

    if mirror_z is None:
        wire_runs = [
            (ins_pts_hi[2], vd_point(0.0, 0.25, VD_WIRE_Z + 0.65), 1.35, 1.45),
            (ins_pts_hi[3], vd_point(0.0, 0.25, VD_WIRE_Z + 0.20), 1.55, 1.35),
            (ins_pts_lo[2], vd_point(0.0, 0.25, VD_WIRE_Z - 0.65), 1.70, 1.25),
            (ins_pts_hi[3], np.array([10.5, 15.5, 8.2]), 1.15, 1.40),
            (ins_pts_lo[3], np.array([10.8, 15.8, 7.3]), 1.38, 1.20),
            (ins_pts_lo[0], np.array([SHOP_X_WALL + 0.10, 8.2, 4.15]), 0.42, 1.10),
        ]
        for p0, p1, sag_m, lw in wire_runs:
            pts_w = []
            for k in range(13):
                tf = k / 12.0
                sag = -sag_m * 4.0 * tf * (1.0 - tf)
                pw = p0 * (1.0 - tf) + p1 * tf + np.array([0.0, 0.0, sag])
                uw, vw, zw = cam.project_pt(pw)
                if zw > Z_NEAR:
                    pts_w.append((uw, vw))
            if len(pts_w) >= 3:
                stroke_catmull_rom(ctx, pts_w, base_w=lw, rgba=(0.092, 0.046, 0.052, 0.78), rng=rng, taper=False, tremor=0.10)


def draw_bakery_3d_craft_and_props(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    mirror_z: float | None = None,
) -> dict:
    """
    Richly hand-crafted 3D props grounded cleanly on left curb & right coastal overlook:
    - Carved timber eave brackets, copper gutter & downspout
    - Warm glowing brass bakery porch lantern under awning
    - Slatted wooden bakery crates with recessed dark plank gaps, corner battens & warm bevels
    - Vintage Japanese mamachari bicycle with curved fenders, chain guard, thick rubber tires,
      crosshatched wicker basket holding a crusty French baguette, kickstand & feathered shadow!
    - Terracotta pots with lush coastal rosemary & crimson geranium blossoms (clear of bike!)
    - Far waterfront overlook stone balustrade grounding the lane end beneath Arch[0]
    - Right coastal stone parapet rusted tubular iron guardrail
    """
    a_s = 1.0 if mirror_z is None else 0.42
    if mirror_z is None:
        # Carved dark-timber eave brackets + copper gutter & downspout
        for py_b in (6.25, 9.65, 12.60, 15.35):
            zb_m = float(cam.project_pt((SHOP_X_WALL, py_b, SHOP_Z_EAVE))[2])
            c_br = fog_mix((0.080, 0.042, 0.040), fog_rgb, zb_m)
            stroke_3d_line(ctx, cam, (SHOP_X_WALL, py_b, SHOP_Z_EAVE - 0.55),
                           (SHOP_X_EAVE + 0.08, py_b, SHOP_Z_EAVE - 0.04), 0.11, (*c_br, 0.96), min_px=1.8)
            stroke_3d_line(ctx, cam, (SHOP_X_WALL, py_b, SHOP_AWNING_Z0 - 0.42),
                           (SHOP_AWNING_X1 + 0.05, py_b, SHOP_AWNING_Z1 - 0.02), 0.075, (*c_br, 0.94), min_px=1.4)
        c_cop = fog_mix((0.26, 0.135, 0.105), fog_rgb, 14.0)
        stroke_3d_line(ctx, cam, (SHOP_X_EAVE + 0.04, SHOP_Y0, SHOP_Z_EAVE - 0.02),
                       (SHOP_X_EAVE + 0.04, SHOP_Y1, SHOP_Z_EAVE - 0.02), 0.09, (*c_cop, 0.96), min_px=1.8)
        stroke_3d_line(ctx, cam, (SHOP_X_WALL + 0.08, 15.40, SHOP_Z_EAVE - 0.15),
                       (SHOP_X_WALL + 0.08, 15.40, Z_CURB), 0.08, (*c_cop, 0.94), min_px=1.6)

        # Far waterfront overlook railing atop ashlar stone balustrade (wall warped via exact 3x3 homography in render_scene!)
        z_end = float(cam.project_pt((0.0, LANE_Y1, 0.4))[2])
        c_end_t = fog_mix((0.48, 0.24, 0.145), fog_rgb, z_end)
        # Sunlit stone coping lip & dark ground contact line along waterfront overlook edge + slender coastal rail
        fill_3d_poly(ctx, cam, [
            (LANE_X0, LANE_Y1 - 0.10, 0.72),
            (PARAPET_X1, LANE_Y1 - 0.10, 0.72),
            (PARAPET_X1, LANE_Y1 + 0.34, 0.76),
            (LANE_X0, LANE_Y1 + 0.34, 0.76),
        ], (*c_end_t, 0.98))
        stroke_3d_line(ctx, cam, (LANE_X0, LANE_Y1 - 0.02, 0.02), (PARAPET_X0 + 0.05, LANE_Y1 - 0.02, 0.02),
                       0.09, (0.04, 0.02, 0.025, 0.70), min_px=2.0)
        stroke_3d_line(ctx, cam, (LANE_X0 + 0.2, LANE_Y1, 1.15), (PARAPET_X0 + 0.22, LANE_Y1, 1.15),
                       0.048, (*fog_mix((0.20, 0.10, 0.088), fog_rgb, z_end), 0.92), min_px=1.2)
        for bx_o in (-1.8, -0.4, 1.0, 2.3):
            stroke_3d_line(ctx, cam, (bx_o, LANE_Y1, 0.72), (bx_o, LANE_Y1, 1.18),
                           0.052, (*fog_mix((0.18, 0.092, 0.085), fog_rgb, z_end), 0.92), min_px=1.2)

    # Warm glowing brass bakery porch lantern hanging under awning
    lan_pt = np.array([-2.54, 11.45, 2.14])
    p_l_use = mirror_z_pt(lan_pt, mirror_z) if mirror_z is not None else lan_pt
    ul, vl, zl = cam.project_pt(p_l_use)
    lan_uv = (ul / W, vl / H)
    if zl > Z_NEAR:
        rl = max(4.0, float(cam.K[0, 0] * 0.13 / zl))
        stroke_3d_line(ctx, cam, (lan_pt[0], lan_pt[1], SHOP_AWNING_Z1 + 0.08), lan_pt + [0, 0, 0.14],
                       0.025, (0.09, 0.048, 0.044, 0.95 * a_s), mirror_z=mirror_z, min_px=1.0)
        g_l = cairo.RadialGradient(ul, vl, rl * 0.15, ul, vl, rl * 5.4)
        g_l.add_color_stop_rgba(0.0, 1.0, 0.95, 0.74, 0.98 * a_s)
        g_l.add_color_stop_rgba(0.28, 1.0, 0.68, 0.28, 0.74 * a_s)
        g_l.add_color_stop_rgba(0.65, 0.96, 0.42, 0.14, 0.25 * a_s)
        g_l.add_color_stop_rgba(1.0, 0.88, 0.30, 0.08, 0.0)
        ctx.set_source(g_l)
        ctx.arc(ul, vl, rl * 5.4, 0, math.tau)
        ctx.fill()
        ctx.set_source_rgba(0.14, 0.068, 0.052, 0.96 * a_s)
        ctx.arc(ul, vl - rl * 0.55, rl * 1.05, math.pi, math.tau)
        ctx.fill()

    if mirror_z is None:
        # Detailed Slatted Wooden Bakery/Milk Crates with plank gaps, corner battens & soft grounded shadow!
        def draw_crate_3d(cx: float, cy: float, cz: float, wx_s: float, wy_s: float, wz_s: float):
            zm = float(cam.project_pt((cx, cy, cz))[2])
            c_fr = fog_mix((0.32, 0.175, 0.115), fog_rgb, zm)
            c_fr_hi = fog_mix((0.46, 0.25, 0.145), fog_rgb, zm)
            c_sd = fog_mix((0.18, 0.098, 0.074), fog_rgb, zm)
            c_tp = fog_mix((0.38, 0.21, 0.130), fog_rgb, zm)
            c_gap = fog_mix((0.055, 0.028, 0.030), fog_rgb, zm)

            # Soft radial contact shadow beneath lower crate
            if abs(cz - Z_CURB) < 0.05:
                uc, vc, _ = cam.project_pt((cx + 0.05, cy, cz + 0.01))
                g_sh = cairo.RadialGradient(uc, vc, 6.0, uc, vc, 78.0)
                g_sh.add_color_stop_rgba(0.0, 0.03, 0.015, 0.020, 0.52)
                g_sh.add_color_stop_rgba(1.0, 0.03, 0.015, 0.020, 0.0)
                ctx.set_source(g_sh)
                ctx.save()
                ctx.translate(uc, vc)
                ctx.scale(1.3, 0.45)
                ctx.arc(0, 0, 72.0, 0, math.tau)
                ctx.fill()
                ctx.restore()

            # Dark interior core visible through recessed plank slots
            fill_3d_poly(ctx, cam, [
                (cx - wx_s * 0.48, cy - wy_s * 0.48, cz),
                (cx + wx_s * 0.48, cy - wy_s * 0.48, cz),
                (cx + wx_s * 0.48, cy - wy_s * 0.48, cz + wz_s),
                (cx - wx_s * 0.48, cy - wy_s * 0.48, cz + wz_s),
            ], (*c_gap, 0.98))
            fill_3d_poly(ctx, cam, [
                (cx + wx_s * 0.48, cy - wy_s * 0.48, cz),
                (cx + wx_s * 0.48, cy + wy_s * 0.48, cz),
                (cx + wx_s * 0.48, cy + wy_s * 0.48, cz + wz_s),
                (cx + wx_s * 0.48, cy - wy_s * 0.48, cz + wz_s),
            ], (*c_gap, 0.98))

            # Individual horizontal wooden slats on front & sunlit right faces!
            for s_i in range(3):
                za = cz + (s_i * 0.34 + 0.02) * wz_s
                zb = cz + (s_i * 0.34 + 0.28) * wz_s
                fill_3d_poly(ctx, cam, [
                    (cx - wx_s * 0.5, cy - wy_s * 0.5, za),
                    (cx + wx_s * 0.5, cy - wy_s * 0.5, za),
                    (cx + wx_s * 0.5, cy - wy_s * 0.5, zb),
                    (cx - wx_s * 0.5, cy - wy_s * 0.5, zb),
                ], (*c_sd, 0.99))
                fill_3d_poly_grad(ctx, cam, [
                    (cx + wx_s * 0.5, cy - wy_s * 0.5, za),
                    (cx + wx_s * 0.5, cy + wy_s * 0.5, za),
                    (cx + wx_s * 0.5, cy + wy_s * 0.5, zb),
                    (cx + wx_s * 0.5, cy - wy_s * 0.5, zb),
                ], (cx + wx_s * 0.5, cy, zb), (cx + wx_s * 0.5, cy, za), [
                    (0.0, *c_fr_hi, 0.99),
                    (1.0, *c_fr, 0.99),
                ])

            # Vertical corner wood battens reinforcing crate
            for y_bat in (cy - wy_s * 0.46, cy + wy_s * 0.46):
                stroke_3d_line(ctx, cam, (cx + wx_s * 0.505, y_bat, cz), (cx + wx_s * 0.505, y_bat, cz + wz_s),
                               0.055, (*c_fr, 0.98), min_px=2.0)
            # Top wooden rim slats
            fill_3d_poly(ctx, cam, [
                (cx - wx_s * 0.5, cy - wy_s * 0.5, cz + wz_s),
                (cx + wx_s * 0.5, cy - wy_s * 0.5, cz + wz_s),
                (cx + wx_s * 0.5, cy + wy_s * 0.5, cz + wz_s),
                (cx - wx_s * 0.5, cy + wy_s * 0.5, cz + wz_s),
            ], (*c_tp, 0.98))
            fill_3d_poly(ctx, cam, [
                (cx - wx_s * 0.36, cy - wy_s * 0.36, cz + wz_s + 0.005),
                (cx + wx_s * 0.36, cy - wy_s * 0.36, cz + wz_s + 0.005),
                (cx + wx_s * 0.36, cy + wy_s * 0.36, cz + wz_s + 0.005),
                (cx - wx_s * 0.36, cy + wy_s * 0.36, cz + wz_s + 0.005),
            ], (*c_gap, 0.72))

        draw_crate_3d(-2.76, 6.85, Z_CURB, 0.54, 0.68, 0.42)
        draw_crate_3d(-2.80, 6.92, Z_CURB + 0.42, 0.48, 0.60, 0.35)

        # Terracotta plant pots cleanly clear of bicycle (y ~ 10.32, 12.55, 12.88) with lush foliage & blossoms!
        for px_p, py_p, rad_p, h_p in ((-2.74, 10.32, 0.16, 0.29), (-2.64, 12.55, 0.16, 0.28), (-2.38, 12.86, 0.13, 0.23)):
            zm = float(cam.project_pt((px_p, py_p, Z_CURB))[2])
            c_pot_sh = fog_mix((0.34, 0.14, 0.095), fog_rgb, zm)
            c_pot_lt = fog_mix((0.64, 0.28, 0.155), fog_rgb, zm)
            c_rim_p = fog_mix((0.76, 0.36, 0.19), fog_rgb, zm)
            # Soft contact shadow under terracotta saucer
            up_s, vp_s, _ = cam.project_pt((px_p, py_p, Z_CURB))
            ctx.set_source_rgba(0.03, 0.015, 0.02, 0.42)
            ctx.save()
            ctx.translate(up_s, vp_s + 2)
            ctx.scale(1.2, 0.42)
            ctx.arc(0, 0, max(10.0, float(cam.K[0, 0] * rad_p * 0.9 / zm)), 0, math.tau)
            ctx.fill()
            ctx.restore()
            # Shaded tapered terracotta pot body + rolled lip
            fill_3d_poly_grad(ctx, cam, [
                (px_p - rad_p * 0.70, py_p, Z_CURB),
                (px_p + rad_p * 0.70, py_p, Z_CURB),
                (px_p + rad_p, py_p, Z_CURB + h_p),
                (px_p - rad_p, py_p, Z_CURB + h_p),
            ], (px_p - rad_p, py_p, Z_CURB + h_p * 0.5), (px_p + rad_p, py_p, Z_CURB + h_p * 0.5), [
                (0.0, *c_pot_sh, 0.99),
                (0.75, *c_pot_lt, 0.99),
                (1.0, *c_rim_p, 0.99),
            ])
            stroke_3d_line(ctx, cam, (px_p - rad_p * 1.08, py_p, Z_CURB + h_p),
                           (px_p + rad_p * 1.08, py_p, Z_CURB + h_p), 0.058, (*c_rim_p, 0.99), min_px=2.2)
            # Lush arching cubic-Bezier coastal rosemary/geranium stems, leaves & tiny sunset geranium blooms!
            u0_p, v0_p, _ = cam.project_pt((px_p, py_p, Z_CURB + h_p))
            sc_p = 11.5 / max(4.2, zm)
            for k in range(11):
                ang_l = -math.pi * 0.5 + (k - 5) * 0.20 + float(rng.uniform(-0.12, 0.12))
                len_l = float(rng.uniform(18.0, 38.0)) * sc_p
                ex = u0_p + math.cos(ang_l) * len_l
                ey = v0_p + math.sin(ang_l) * len_l
                mx = 0.5 * (u0_p + ex) + float(rng.uniform(-6, 6)) * sc_p
                my = 0.5 * (v0_p + ey) - 6.0 * sc_p
                c_lf = fog_mix((0.12 + 0.08 * (k % 2), 0.17, 0.095), fog_rgb, zm)
                ctx.set_source_rgba(*c_lf, 0.95)
                ctx.set_line_width(max(1.0, 1.8 * sc_p))
                ctx.move_to(u0_p, v0_p)
                ctx.curve_to(mx, my, mx, my, ex, ey)
                ctx.stroke()
                # Layered organic Bezier leaves along stem
                for frac_s in (0.58, 0.95):
                    lx_s = u0_p + (ex - u0_p) * frac_s
                    ly_s = v0_p + (ey - v0_p) * frac_s
                    draw_bezier_maple_ivy_leaf(ctx, lx_s, ly_s, ang_l + math.pi * 0.5, 9.0 * sc_p,
                                               (*c_lf, 0.94), rim_rgba=None)
                if k in (2, 5, 8):
                    # Tiny warm crimson-coral geranium blossom cluster at tip
                    c_bl = fog_mix((0.88, 0.28, 0.16), fog_rgb, zm)
                    ctx.set_source_rgba(*c_bl, 0.92)
                    for dx_b, dy_b in ((-2.5, -2.0), (2.5, -1.5), (0.0, -4.0)):
                        ctx.arc(ex + dx_b * sc_p, ey + dy_b * sc_p, max(1.6, 2.8 * sc_p), 0, math.tau)
                        ctx.fill()

        # Hero Vintage Parked Japanese Mamachari Bicycle (y = 7.95..9.15, grounded cleanly with curved fenders,
        # thick rubber tires, woven wicker basket with baguette, dynamo headlamp & soft radial tire shadows!)
        bx_c = -2.56
        y_w1, y_w2 = 7.95, 9.12
        r_wh = 0.34
        z_hub = Z_CURB + r_wh
        zm_b = float(cam.project_pt((bx_c, 8.55, z_hub))[2])
        c_tire = fog_mix((0.068, 0.040, 0.044), fog_rgb, zm_b)
        c_frame = fog_mix((0.50, 0.165, 0.130), fog_rgb, zm_b)
        c_frame_hi = fog_mix((0.86, 0.42, 0.24), fog_rgb, zm_b)
        c_chrome = fog_mix((0.86, 0.70, 0.52), fog_rgb, zm_b)
        c_bask = fog_mix((0.40, 0.23, 0.13), fog_rgb, zm_b)
        c_bask_dk = fog_mix((0.18, 0.095, 0.062), fog_rgb, zm_b)

        # Soft elliptical radial shadows under both wheels (NO hard rectangle!)
        for y_sh in (y_w1, y_w2):
            us, vs, _ = cam.project_pt((bx_c + 0.04, y_sh, Z_CURB + 0.01))
            g_ts = cairo.RadialGradient(us, vs, 3.0, us, vs, 48.0)
            g_ts.add_color_stop_rgba(0.0, 0.03, 0.015, 0.02, 0.56)
            g_ts.add_color_stop_rgba(1.0, 0.03, 0.015, 0.02, 0.0)
            ctx.set_source(g_ts)
            ctx.save()
            ctx.translate(us, vs)
            ctx.scale(1.4, 0.38)
            ctx.arc(0, 0, 44.0, 0, math.tau)
            ctx.fill()
            ctx.restore()

        for y_w in (y_w1, y_w2):
            pts_wh = []
            pts_fend = []
            for k in range(25):
                th = math.tau * k / 24.0
                wx_p = bx_c - 0.08 * math.sin(th)
                wy_p = y_w + r_wh * math.cos(th)
                wz_p = z_hub + r_wh * math.sin(th)
                u_p, v_p, _ = cam.project_pt((wx_p, wy_p, wz_p))
                pts_wh.append((u_p, v_p))
                if 0.15 * math.pi <= th <= 1.15 * math.pi:
                    uf, vf, _ = cam.project_pt((bx_c - 0.09 * math.sin(th), y_w + (r_wh + 0.045) * math.cos(th), z_hub + (r_wh + 0.045) * math.sin(th)))
                    pts_fend.append((uf, vf))
            uh, vh, _ = cam.project_pt((bx_c, y_w, z_hub))
            ctx.set_source_rgba(*c_chrome, 0.52)
            ctx.set_line_width(0.9)
            for k in range(0, 24, 3):
                ctx.move_to(uh, vh)
                ctx.line_to(*pts_wh[k])
                ctx.stroke()
            # Thick dark rubber tire + chrome rim inner ring + crimson curved mudguard fender!
            ctx.set_source_rgba(*c_tire, 0.99)
            ctx.set_line_width(6.8)
            ctx.move_to(*pts_wh[0])
            for pt in pts_wh[1:]:
                ctx.line_to(*pt)
            ctx.stroke()
            ctx.set_source_rgba(*c_chrome, 0.78)
            ctx.set_line_width(1.6)
            ctx.move_to(*pts_wh[0])
            for pt in pts_wh[1:]:
                ctx.line_to(*pt)
            ctx.stroke()
            if len(pts_fend) >= 3:
                ctx.set_source_rgba(*c_frame, 0.96)
                ctx.set_line_width(4.8)
                ctx.move_to(*pts_fend[0])
                for pt in pts_fend[1:]:
                    ctx.line_to(*pt)
                ctx.stroke()
                ctx.set_source_rgba(*c_frame_hi, 0.55)
                ctx.set_line_width(1.4)
                ctx.move_to(*pts_fend[0])
                for pt in pts_fend[1:]:
                    ctx.line_to(*pt)
                ctx.stroke()

        # Frame tubes, chain guard, kickstand, rear luggage rack, leather saddle & swept handlebars
        p_rear = np.array([bx_c, y_w1, z_hub])
        p_bb = np.array([bx_c - 0.02, y_w1 + 0.45, z_hub - 0.04])
        p_seat = np.array([bx_c - 0.08, y_w1 + 0.35, z_hub + 0.54])
        p_head_lo = np.array([bx_c - 0.05, y_w2 - 0.12, z_hub + 0.36])
        p_head_hi = np.array([bx_c - 0.09, y_w2 - 0.16, z_hub + 0.64])
        p_front = np.array([bx_c, y_w2, z_hub])
        # Kickstand grounded cleanly to curb!
        stroke_3d_line(ctx, cam, p_rear + [0.02, 0.0, -0.04], (bx_c + 0.16, y_w1 - 0.08, Z_CURB), 0.024, (*c_chrome, 0.90), min_px=1.2)
        # Rear luggage rack
        stroke_3d_line(ctx, cam, p_rear, (bx_c - 0.07, y_w1 - 0.22, z_hub + 0.42), 0.024, (*c_chrome, 0.92), min_px=1.2)
        stroke_3d_line(ctx, cam, (bx_c - 0.07, y_w1 - 0.26, z_hub + 0.42), (bx_c - 0.07, y_w1 + 0.28, z_hub + 0.42), 0.028, (*c_chrome, 0.94), min_px=1.5)
        for pa, pb in ((p_rear, p_bb), (p_rear, p_seat), (p_bb, p_seat), (p_bb, p_head_lo),
                       (p_seat - [0, 0, 0.12], p_head_hi - [0, 0, 0.08]), (p_head_lo, p_head_hi), (p_head_lo, p_front)):
            stroke_3d_line(ctx, cam, pa, pb, 0.040, (*c_frame, 0.99), min_px=2.0)
            stroke_3d_line(ctx, cam, pa + [0.012, 0, 0.008], pb + [0.012, 0, 0.008], 0.012, (*c_frame_hi, 0.52), min_px=0.8)
        # Sleek tapered chain guard, chrome chainring disc & leather saddle
        stroke_3d_line(ctx, cam, p_rear + [0, 0.06, 0.0], p_bb, 0.050, (*c_frame, 0.96), min_px=2.2)
        ubb, vbb, zbb = cam.project_pt(p_bb)
        if zbb > Z_NEAR:
            ctx.set_source_rgba(*c_chrome, 0.86)
            ctx.arc(ubb, vbb, max(4.0, float(cam.K[0, 0] * 0.065 / zbb)), 0, math.tau)
            ctx.fill()
            ctx.set_source_rgba(*c_frame, 0.96)
            ctx.arc(ubb, vbb, max(2.2, float(cam.K[0, 0] * 0.032 / zbb)), 0, math.tau)
            ctx.fill()
        stroke_3d_line(ctx, cam, p_seat + [0, -0.11, 0.05], p_seat + [0, 0.12, 0.05], 0.082, (*c_tire, 0.99), min_px=3.6)
        # Swept chrome handlebars + dark rubber grips
        stroke_3d_line(ctx, cam, p_head_hi + [-0.22, -0.12, 0.10], p_head_hi + [0.0, 0.02, 0.08], 0.034, (*c_chrome, 0.98), min_px=1.8)
        stroke_3d_line(ctx, cam, p_head_hi + [0.0, 0.02, 0.08], p_head_hi + [0.24, -0.10, 0.10], 0.034, (*c_chrome, 0.98), min_px=1.8)
        stroke_3d_line(ctx, cam, p_head_hi + [0.16, -0.06, 0.095], p_head_hi + [0.25, -0.11, 0.102], 0.044, (*c_tire, 0.98), min_px=2.4)

        # Woven wicker front bicycle basket with golden-crust French baguette & kraft paper bag peeking out!!
        b_p0 = (bx_c - 0.12, y_w2 - 0.06, z_hub + 0.36)
        b_p1 = (bx_c + 0.14, y_w2 + 0.22, z_hub + 0.36)
        b_p2 = (bx_c + 0.17, y_w2 + 0.26, z_hub + 0.64)
        b_p3 = (bx_c - 0.14, y_w2 - 0.09, z_hub + 0.64)
        # Crusty golden French baguette peeking diagonally out of the bakery basket!
        stroke_3d_line(ctx, cam, (bx_c + 0.02, y_w2 + 0.06, z_hub + 0.52),
                       (bx_c + 0.14, y_w2 + 0.20, z_hub + 0.78), 0.095, (*fog_mix((0.86, 0.56, 0.24), fog_rgb, zm_b), 0.96), min_px=4.5)
        fill_3d_poly(ctx, cam, [b_p0, b_p1, b_p2, b_p3], (*c_bask, 0.96))
        # Wicker crosshatch weave lines across basket
        for k in range(1, 4):
            f = k / 4.0
            stroke_3d_line(ctx, cam, (b_p0[0] + f * (b_p3[0] - b_p0[0]), b_p0[1] + f * (b_p3[1] - b_p0[1]), b_p0[2] + f * (b_p3[2] - b_p0[2])),
                           (b_p1[0] + f * (b_p2[0] - b_p1[0]), b_p1[1] + f * (b_p2[1] - b_p1[1]), b_p1[2] + f * (b_p2[2] - b_p1[2])),
                           0.015, (*c_bask_dk, 0.78), min_px=0.9)
        for k in range(1, 5):
            f = k / 5.0
            stroke_3d_line(ctx, cam, (b_p0[0] + f * (b_p1[0] - b_p0[0]), b_p0[1] + f * (b_p1[1] - b_p0[1]), b_p0[2]),
                           (b_p3[0] + f * (b_p2[0] - b_p3[0]), b_p3[1] + f * (b_p2[1] - b_p3[1]), b_p2[2]),
                           0.015, (*c_bask_dk, 0.78), min_px=0.9)
        stroke_3d_line(ctx, cam, b_p3, b_p2, 0.032, (*c_bask_dk, 0.95), min_px=1.8)

        # Right coastal stone parapet rusted tubular iron guardrail
        gx_r = PARAPET_X0 + 0.22
        rail_ys = [5.2, 7.6, 10.2, 13.0, 16.0, 19.2, 22.6, 26.2, 30.0]
        c_rust = fog_mix((0.18, 0.088, 0.078), fog_rgb, 11.0)
        c_rust_hi = fog_mix((0.78, 0.39, 0.18), fog_rgb, 11.0)
        for z_bar in (PARAPET_Z + 0.34, PARAPET_Z + 0.70):
            stroke_3d_line(ctx, cam, (gx_r, 4.4, z_bar), (gx_r, LANE_Y1, z_bar), 0.044, (*c_rust, 0.95), min_px=1.4)
            stroke_3d_line(ctx, cam, (gx_r + 0.02, 4.4, z_bar + 0.02), (gx_r + 0.02, LANE_Y1, z_bar + 0.02), 0.018, (*c_rust_hi, 0.62), min_px=0.8)
        for ry_p in rail_ys:
            stroke_3d_line(ctx, cam, (gx_r, ry_p, PARAPET_Z - 0.02), (gx_r, ry_p, PARAPET_Z + 0.74), 0.048, (*c_rust, 0.96), min_px=1.5)

    return {"shop_porch_lamp_uv": (float(lan_uv[0]), float(lan_uv[1]))}


# =============================================================================
# Stage 3 Band 3 — Painterly Organic Cubic-Bezier Coastal Black Pine & Autumn Ivy
# =============================================================================

def draw_bezier_maple_ivy_leaf(
    ctx: cairo.Context,
    cx: float,
    cy: float,
    ang: float,
    size: float,
    fill_rgba: tuple[float, float, float, float],
    rim_rgba: tuple[float, float, float, float] | None = None,
    translucent_tip: tuple[float, float, float] | None = None,
) -> None:
    """Organic painterly 3-lobed autumn ivy / maple leaf built purely from cubic Beziers (NO sticker outlines!)."""
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(ang)
    s = size
    ctx.move_to(0.0, 0.0)
    ctx.curve_to(-s * 0.42, -s * 0.12, -s * 0.78, -s * 0.35, -s * 0.68, -s * 0.56)
    ctx.curve_to(-s * 0.46, -s * 0.44, -s * 0.32, -s * 0.64, 0.0, -s * 1.05)
    ctx.curve_to(s * 0.32, -s * 0.64, s * 0.46, -s * 0.44, s * 0.68, -s * 0.56)
    ctx.curve_to(s * 0.78, -s * 0.35, s * 0.42, -s * 0.12, 0.0, 0.0)
    ctx.close_path()
    if translucent_tip is not None:
        g_lf = cairo.LinearGradient(0.0, 0.0, 0.0, -s * 1.05)
        g_lf.add_color_stop_rgba(0.0, *fill_rgba)
        g_lf.add_color_stop_rgba(0.65, *fill_rgba)
        g_lf.add_color_stop_rgba(1.0, *translucent_tip, fill_rgba[3])
        ctx.set_source(g_lf)
    else:
        ctx.set_source_rgba(*fill_rgba)
    ctx.fill()
    # Optional soft directional highlight ONLY on one sunward edge (never a full sticker border!)
    if rim_rgba is not None:
        ctx.set_source_rgba(*rim_rgba)
        ctx.set_line_width(max(0.7, s * 0.065))
        ctx.move_to(-s * 0.64, -s * 0.54)
        ctx.curve_to(-s * 0.42, -s * 0.44, -s * 0.28, -s * 0.64, 0.0, -s * 1.03)
        ctx.stroke()
    ctx.restore()


def draw_pine_needle_pad(
    ctx: cairo.Context,
    cx: float,
    cy: float,
    width_px: float,
    height_px: float,
    col_shadow: tuple[float, float, float],
    col_mid: tuple[float, float, float],
    col_rim: tuple[float, float, float],
    rng: np.random.Generator,
    alpha: float = 0.96,
) -> None:
    """
    Sculptural Ghibli Japanese coastal black-pine (kuromatsu) foliage cloud pad
    built 100% from dense layered cubic-Bezier needle fans (shadow under-layer,
    mid-tone body needles, and fiery sunset-backlit upper/left rim needles!).
    """
    n_fans = int(rng.integers(14, 20))
    for k in range(n_fans):
        fx = float(rng.uniform(-0.5, 0.5))
        fy = float(rng.uniform(-0.45, 0.35))
        px = cx + fx * width_px
        py = cy + fy * height_px * (1.0 - 0.45 * abs(fx))
        n_nd = int(rng.integers(9, 13))
        scale = max(9.0, height_px * float(rng.uniform(0.55, 0.95)))
        for j in range(n_nd):
            frac = (j / max(1, n_nd - 1)) - 0.5
            ang = -math.pi * 0.5 + frac * 2.15 + fx * 0.4 + float(rng.uniform(-0.10, 0.10))
            ln = scale * float(rng.uniform(0.65, 1.15))
            ex = px + math.cos(ang) * ln
            ey = py + math.sin(ang) * ln
            bend = float(rng.uniform(-0.16, 0.16)) * ln
            mx = 0.5 * (px + ex) - math.sin(ang) * bend
            my = 0.5 * (py + ey) + math.cos(ang) * bend
            if fy > 0.08:
                col = col_shadow
            elif (fx < -0.05 or fy < -0.18) and rng.random() < 0.38:
                col = col_rim
            else:
                col = col_mid
            ctx.set_source_rgba(*col, alpha)
            ctx.set_line_width(max(0.95, scale * 0.085))
            ctx.move_to(px, py)
            ctx.curve_to(mx, my, mx, my, ex, ey)
            ctx.stroke()


def draw_distant_hillside_pine(
    ctx: cairo.Context,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
) -> None:
    """Left coastal hill-embankment black-pine crown rendered BEHIND bakery roof & utility pole!"""
    z_f = 24.0
    c_bark = fog_mix((0.088, 0.044, 0.048), fog_rgb, z_f)
    c_nd_sh = fog_mix((0.055, 0.038, 0.046), fog_rgb, z_f)
    c_nd_md = fog_mix((0.100, 0.074, 0.064), fog_rgb, z_f)
    c_nd_rm = fog_mix((0.72, 0.34, 0.15), fog_rgb, z_f)
    for b_pts, bw in (
        ([(W * 0.25, H * 0.36), (W * 0.295, H * 0.312), (W * 0.340, H * 0.285), (W * 0.372, H * 0.294)], 9.5),
        ([(W * 0.28, H * 0.335), (W * 0.318, H * 0.268), (W * 0.354, H * 0.258)], 8.0),
    ):
        stroke_catmull_rom(ctx, b_pts, base_w=bw, rgba=(*c_bark, 0.96), rng=rng, taper=True, tremor=1.4)
        for i in range(1, len(b_pts)):
            px, py = b_pts[i]
            draw_pine_needle_pad(ctx, px, py - 8.0, 58.0 * (1.0 - 0.12 * i), 22.0, c_nd_sh, c_nd_md, c_nd_rm, rng, alpha=0.95)


def draw_organic_pine_and_autumn_ivy(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
) -> None:
    """
    Band 3:
    1. Lush sculptural wind-swept Japanese coastal black-pine (kuromatsu) tiered
       boughs rooted into a weathered coastal rocky bluff flank at the right edge
       (100% cubic-Bezier branches & dense layered needle pads!).
    2. Natural painterly trailing crimson-rust, olive-plum & burnt-amber autumn ivy
       cascading down the right coastal stone parapet!
    """
    # 1. Right coastal rocky bluff anchor + gnarled black-pine limbs (kept strictly right of u = 0.87!)
    z_f = 13.0
    c_rock = fog_mix((0.088, 0.048, 0.055), fog_rgb, z_f)
    c_bark = fog_mix((0.088, 0.044, 0.048), fog_rgb, z_f)
    c_bark_hi = fog_mix((0.48, 0.22, 0.115), fog_rgb, z_f)
    c_nd_sh = fog_mix((0.052, 0.036, 0.044), fog_rgb, z_f)
    c_nd_md = fog_mix((0.095, 0.072, 0.062), fog_rgb, z_f)
    c_nd_rm = fog_mix((0.68, 0.32, 0.14), fog_rgb, z_f)

    # Organic coastal cliff shoulder rooting the pine trunk cleanly at extreme right edge (u > 0.965)
    ctx.set_source_rgba(*c_rock, 0.97)
    ctx.move_to(W + 10, H * 0.28)
    ctx.curve_to(W * 0.975, H * 0.33, W * 0.968, H * 0.42, W * 0.985, H * 0.49)
    ctx.line_to(W + 10, H * 0.50)
    ctx.close_path()
    ctx.fill()

    pine_specs = [
        ([(W * 0.995, H * 0.44), (W * 0.958, H * 0.402), (W * 0.914, H * 0.365), (W * 0.878, H * 0.374)], 16.0),
        ([(W * 0.995, H * 0.325), (W * 0.964, H * 0.298), (W * 0.922, H * 0.274), (W * 0.886, H * 0.284)], 14.0),
        ([(W * 0.956, H * 0.398), (W * 0.930, H * 0.424), (W * 0.896, H * 0.432)], 10.5),
    ]
    for b_pts, base_w in pine_specs:
        stroke_catmull_rom(ctx, b_pts, base_w=base_w, rgba=(*c_bark, 0.98), rng=rng, taper=True, tremor=1.5)
        hi_pts = [(x, y - base_w * 0.25) for (x, y) in b_pts]
        stroke_catmull_rom(ctx, hi_pts, base_w=max(1.2, base_w * 0.26), rgba=(*c_bark_hi, 0.58), rng=rng, taper=True, tremor=1.0)
        for i in range(1, len(b_pts)):
            px, py = b_pts[i]
            pw = 86.0 * (1.0 - 0.13 * i)
            ph = 32.0 * (1.0 - 0.12 * i)
            draw_pine_needle_pad(ctx, px, py - ph * 0.35, pw, ph, c_nd_sh, c_nd_md, c_nd_rm, rng, alpha=0.96)

    # 2. Painterly Natural Autumn Ivy cascading organically along right coastal stone parapet!
    vine_anchors_y = [5.4, 6.8, 8.4, 10.2, 12.2, 14.6, 17.8]
    for vy_a in vine_anchors_y:
        z_v = float(cam.project_pt((PARAPET_X0, vy_a, PARAPET_Z))[2])
        scale_v = 11.0 / max(4.2, z_v)
        for t_i in range(2 if vy_a > 12.5 else 3):
            y_cur = vy_a + t_i * 0.38
            drop_m = float(rng.uniform(0.32, 0.80))
            pts3 = []
            for k in range(7):
                tf = k / 6.0
                wx_t = PARAPET_X0 + (0.14 * (1.0 - tf)) - 0.02
                wy_t = y_cur + math.sin(tf * 4.8 + t_i * 1.9) * 0.15
                wz_t = PARAPET_Z + 0.05 - tf * drop_m
                u_t, v_t, z_ct = cam.project_pt((wx_t, wy_t, wz_t))
                if z_ct > Z_NEAR:
                    pts3.append((u_t, v_t))
            if len(pts3) < 2:
                continue
            c_stem = fog_mix((0.10, 0.046, 0.044), fog_rgb, z_v)
            stroke_catmull_rom(ctx, pts3, base_w=max(1.0, 2.2 * scale_v), rgba=(*c_stem, 0.90), rng=rng, taper=True, tremor=0.5)
            # First pass: deep shadowed under-leaves (gives lush 3D volume against stone!)
            for p_idx in range(len(pts3)):
                px_l, py_l = pts3[p_idx]
                for _ in range(2):
                    ang_l = math.pi * 0.5 + float(rng.uniform(-1.35, 1.35))
                    sz_l = float(rng.uniform(6.5, 15.5)) * scale_v
                    fc_sh = fog_mix((0.14, 0.052, 0.056), fog_rgb, z_v)
                    draw_bezier_maple_ivy_leaf(ctx, px_l + float(rng.uniform(-8, 8)) * scale_v,
                                               py_l + float(rng.uniform(-5, 6)) * scale_v,
                                               ang_l, sz_l, (*fc_sh, 0.88), rim_rgba=None)
            # Second pass: varied natural autumn surface leaves (burnt crimson, muted olive-russet, warm amber tips!)
            for p_idx in range(len(pts3)):
                px_l, py_l = pts3[p_idx]
                for side in (-1.0, 1.0):
                    if rng.random() > 0.74:
                        continue
                    ang_l = math.pi * 0.5 + side * float(rng.uniform(0.28, 1.25))
                    sz_l = float(rng.uniform(5.8, 14.5)) * scale_v
                    pal = int(rng.integers(0, 4))
                    if pal == 0:
                        fc = fog_mix((0.36, 0.098, 0.088), fog_rgb, z_v)
                        tip = fog_mix((0.62, 0.19, 0.10), fog_rgb, z_v)
                    elif pal == 1:
                        fc = fog_mix((0.52, 0.175, 0.095), fog_rgb, z_v)
                        tip = fog_mix((0.88, 0.38, 0.14), fog_rgb, z_v)
                    elif pal == 2:
                        fc = fog_mix((0.22, 0.135, 0.082), fog_rgb, z_v)  # olive-russet autumn transition leaf!
                        tip = fog_mix((0.46, 0.21, 0.10), fog_rgb, z_v)
                    else:
                        fc = fog_mix((0.24, 0.078, 0.076), fog_rgb, z_v)
                        tip = None
                    # Soft directional rim glint on only ~22% of uppermost leaves (no uniform sticker look!)
                    rc = fog_mix((0.95, 0.52, 0.20), fog_rgb, z_v) if (p_idx <= 2 and rng.random() < 0.24) else None
                    draw_bezier_maple_ivy_leaf(ctx, px_l + side * float(rng.uniform(1.5, 5.5)) * scale_v,
                                               py_l + float(rng.uniform(-3, 3)) * scale_v,
                                               ang_l, sz_l, (*fc, 0.94),
                                               (*rc, 0.55) if rc is not None else None,
                                               translucent_tip=tip)


def draw_distant_gulls(ctx: cairo.Context, rng: np.random.Generator) -> None:
    gulls = [
        (0.64, 0.105, 11.5, 0.22),
        (0.69, 0.088, 9.2, -0.14),
        (0.76, 0.132, 8.4, 0.30),
        (0.83, 0.102, 7.6, -0.08),
        (0.55, 0.072, 6.8, 0.18),
    ]
    for gx, gy, span, tilt in gulls:
        cx, cy = gx * W, gy * H
        ctx.save()
        ctx.translate(cx, cy)
        ctx.rotate(tilt)
        ctx.set_source_rgba(0.18, 0.068, 0.075, 0.90)
        ctx.set_line_width(max(1.3, span * 0.16))
        ctx.move_to(-span, span * 0.12)
        ctx.curve_to(-span * 0.48, -span * 0.42, -span * 0.15, -span * 0.18, 0.0, 0.0)
        ctx.curve_to(span * 0.15, -span * 0.18, span * 0.48, -span * 0.42, span, span * 0.12)
        ctx.stroke()
        ctx.set_source_rgba(0.88, 0.48, 0.24, 0.52)
        ctx.set_line_width(max(0.8, span * 0.08))
        ctx.move_to(-span * 0.85, span * 0.16)
        ctx.curve_to(-span * 0.42, -span * 0.28, -span * 0.12, -span * 0.08, 0.0, 0.8)
        ctx.curve_to(span * 0.12, -span * 0.08, span * 0.42, -span * 0.28, span * 0.85, span * 0.16)
        ctx.stroke()
        ctx.restore()


# =============================================================================
# Stage 2 Volumetric Golden-Hour Sunbeams Cutting Under Viaduct Arches!
# =============================================================================

def paint_viaduct_sunbeams(cam: Camera3D, rng: np.random.Generator, sun_uv: tuple[float, float], intensity: float = 0.30) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = xx / W
    v = yy / H
    su, sv = sun_uv

    du = (u - su) * (W / H)
    dv = v - sv
    ang = np.arctan2(dv, du)
    dist = np.hypot(du, dv)

    ray_pat = (
        0.50 * np.power(0.5 + 0.5 * np.sin(ang * 10.0 + 0.6), 2.2)
        + 0.32 * np.power(0.5 + 0.5 * np.sin(ang * 18.0 - 1.4), 2.8)
        + 0.18 * np.power(0.5 + 0.5 * np.sin(ang * 29.0 + 2.0), 3.4)
    )
    # Dramatic golden beams cutting diagonally down-left under Arch[1] & Arch[0] onto the wet lane!
    down_left = np.clip(1.0 - np.abs(ang - 2.42) / 0.88, 0.0, 1.0)
    up_fan = np.clip(1.0 - np.abs(ang + 1.98) / 0.72, 0.0, 1.0) * 0.44
    radial_falloff = np.exp(-dist * 1.38) * np.clip((dist - 0.022) / 0.055, 0.0, 1.0)

    beam_str = ray_pat * np.maximum(down_left, up_fan) * radial_falloff * intensity
    beam_str = blur_2d(beam_str, 9.0)
    return np.clip(beam_str[..., None] * np.array([1.14, 0.66, 0.30], dtype=np.float32), 0.0, 0.70)


# =============================================================================
# Stage 4 — Foreground Circular-Disc Bokeh & 35mm Optical Finish + YIQ Patina
# =============================================================================

def make_disc_kernel(radius: float = 7.8) -> np.ndarray:
    r_int = int(math.ceil(radius + 2.0))
    yy, xx = np.mgrid[-r_int:r_int + 1, -r_int:r_int + 1].astype(np.float32)
    rr = np.hypot(xx, yy)
    aa = np.clip((radius + 0.55 - rr) / 1.10, 0.0, 1.0)
    rim = np.exp(-((rr - (radius - 0.8)) / 1.25) ** 2) * 0.24
    k = aa * (0.86 + rim)
    k /= float(k.sum()) + 1e-8
    return k.astype(np.float32)


def convolve_fft_2d_bbox(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    a = img[:, :, 3]
    nz = np.any(a > 1e-4, axis=1)
    nx = np.any(a > 1e-4, axis=0)
    if not np.any(nz) or not np.any(nx):
        return np.zeros_like(img, dtype=np.float32)
    pad = max(kh, kw) + 4
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


def paint_foreground_bokeh_elements(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """
    Soft 35mm cinema depth-of-field foreground framing elements (NO sticker outlines!):
      1. Upper-right wind-tossed translucent backlit crimson-burgundy autumn vine twig
      2. Lower-right weathered rusted coastal guardrail stanchion & iron bar with soft
         clinging autumn leaves
    Convolved in premultiplied RGBA with make_disc_kernel(radius=7.8) + separate
    cv2.warpAffine +/-1.05 px opposite R/B lateral chromatic shift!
    """
    surf, ctx = new_cairo_2x()

    # 1. Upper-right trailing backlit autumn vine twig (u ~ 0.81..1.00, v ~ 0.0..0.22)
    twig_main = [(W + 24, -10), (W - 88, 52), (W - 196, 118), (W - 295, 178), (W - 355, 206)]
    twig_sub = [(W - 135, 76), (W - 178, 150), (W - 224, 202)]
    stroke_catmull_rom(ctx, twig_main, base_w=7.5, rgba=(0.052, 0.020, 0.026, 0.94), rng=rng, taper=True, tremor=0.8)
    stroke_catmull_rom(ctx, twig_sub, base_w=4.8, rgba=(0.052, 0.020, 0.026, 0.92), rng=rng, taper=True, tremor=0.6)

    upper_leaves = [
        (W - 62, 46, 2.35, 38.0), (W - 118, 86, 2.75, 44.0), (W - 170, 112, 2.15, 40.0),
        (W - 224, 144, 2.85, 46.0), (W - 278, 172, 2.40, 40.0), (W - 332, 200, 2.65, 34.0),
        (W - 158, 140, 3.25, 36.0), (W - 204, 186, 2.95, 40.0), (W - 242, 218, 2.55, 30.0),
        (W - 40, 102, 3.05, 42.0), (W - 96, 142, 2.60, 34.0),
    ]
    for lx, ly, ang, sz in upper_leaves:
        # Rich translucent backlit crimson-plum leaf grading warmly to amber-vermilion tips (NO outline!)
        draw_bezier_maple_ivy_leaf(ctx, lx, ly, ang, sz,
                                   fill_rgba=(0.18, 0.048, 0.052, 0.92),
                                   rim_rgba=None,
                                   translucent_tip=(0.68, 0.24, 0.10))

    # 2. Lower-right sleek weathered rusted coastal guardrail corner post & diagonal bar (u ~ 0.87..1.00, v ~ 0.74..1.00)
    g_post = cairo.LinearGradient(W - 150, H - 140, W - 45, H - 140)
    g_post.add_color_stop_rgba(0.0, 0.36, 0.145, 0.075, 0.98)
    g_post.add_color_stop_rgba(0.24, 0.072, 0.030, 0.034, 0.98)
    g_post.add_color_stop_rgba(1.0, 0.022, 0.010, 0.015, 0.98)
    ctx.set_source(g_post)
    ctx.move_to(W - 134, H + 30)
    ctx.line_to(W - 148, H - 268)
    ctx.line_to(W - 130, H - 292)
    ctx.line_to(W - 72, H - 292)
    ctx.line_to(W - 56, H - 266)
    ctx.line_to(W - 42, H + 30)
    ctx.close_path()
    ctx.fill()

    ctx.set_source_rgba(0.044, 0.020, 0.024, 0.98)
    ctx.move_to(W - 245, H + 25)
    ctx.line_to(W - 138, H - 202)
    ctx.line_to(W + 30, H - 285)
    ctx.line_to(W + 30, H - 230)
    ctx.line_to(W - 114, H - 156)
    ctx.line_to(W - 182, H + 25)
    ctx.close_path()
    ctx.fill()

    # Soft golden rim catch along left edge of post
    ctx.set_source_rgba(0.88, 0.40, 0.16, 0.58)
    ctx.set_line_width(4.5)
    ctx.move_to(W - 134, H + 10)
    ctx.line_to(W - 148, H - 266)
    ctx.stroke()

    # Soft translucent clinging autumn leaves (NO bright sticker ring!)
    lower_leaves = [
        (W - 152, H - 244, -0.55, 38.0), (W - 184, H - 192, -0.85, 42.0),
        (W - 118, H - 168, 0.45, 38.0), (W - 214, H - 128, -0.65, 44.0),
        (W - 162, H - 86, 0.25, 40.0), (W - 246, H - 48, -0.95, 36.0),
    ]
    for lx, ly, ang, sz in lower_leaves:
        draw_bezier_maple_ivy_leaf(ctx, lx, ly, ang, sz,
                                   fill_rgba=(0.20, 0.052, 0.052, 0.92),
                                   rim_rgba=None,
                                   translucent_tip=(0.64, 0.22, 0.095))

    fg_1x = finish_cairo_2x(surf)
    blurred = convolve_fft_2d_bbox(fg_1x, make_disc_kernel(radius=7.8))

    m_r = np.array([[1.0, 0.0, 1.05], [0.0, 1.0, 0.0]], dtype=np.float32)
    m_b = np.array([[1.0, 0.0, -1.05], [0.0, 1.0, 0.0]], dtype=np.float32)
    out = blurred.copy()
    out[:, :, 0] = cv2.warpAffine(blurred[:, :, 0], m_r, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    out[:, :, 2] = cv2.warpAffine(blurred[:, :, 2], m_b, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    fg_meta = {
        "fg_upper_u_min": float((W - 385) / W),
        "fg_upper_v_max": float(240 / H),
        "fg_lower_u_min": float((W - 275) / W),
        "fg_lower_v_min": float((H - 315) / H),
    }
    return np.clip(out, 0.0, 1.0), fg_meta


def apply_35mm_finish(
    rgb: np.ndarray,
    rng: np.random.Generator,
    sun_uv: tuple[float, float],
    lamp_glints: list[tuple[float, float, tuple[float, float, float], float]],
) -> np.ndarray:
    img = rgb.astype(np.float32).copy()
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # 1. 3-scale highlight halation (sigma = 6.5, 24, 72 px above lum > 0.72 + sun/lamp glints)
    lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    hl_mask = np.clip((lum - 0.72) / 0.28, 0.0, 1.8)
    hl_src = img * hl_mask[..., None]

    d_sun = np.hypot((xx - sun_uv[0] * W) / (W * 0.038), (yy - sun_uv[1] * H) / (H * 0.060))
    hl_src += np.exp(-d_sun * d_sun)[..., None] * np.array([0.46, 0.30, 0.14], dtype=np.float32)
    for gu, gv, gcol, grad_px in lamp_glints:
        dg = np.hypot(xx - gu * W, yy - gv * H) / max(4.0, grad_px)
        hl_src += np.exp(-dg * dg)[..., None] * np.array(gcol, dtype=np.float32)

    tint = np.array([1.15, 0.93, 0.76], dtype=np.float32)
    hal = (
        0.26 * blur_2d(hl_src, 6.5)
        + 0.18 * blur_2d(hl_src, 24.0)
        + 0.12 * blur_2d(hl_src, 72.0)
    ) * tint[None, None, :]
    img = 1.0 - (1.0 - np.clip(img, 0.0, 1.0)) * (1.0 - np.clip(hal, 0.0, 0.84))

    # 2. Radial corner CA (cv2.remap BORDER_REFLECT_101) + gentle optical vignette
    nx = (xx - W * 0.5) / (W * 0.5)
    ny = (yy - H * 0.5) / (H * 0.5)
    r2 = nx * nx + ny * ny
    ca_scale = 0.00105
    img[:, :, 0] = cv2.remap(img[:, :, 0], (xx + nx * r2 * (W * ca_scale)).astype(np.float32),
                             (yy + ny * r2 * (H * ca_scale)).astype(np.float32),
                             interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    img[:, :, 2] = cv2.remap(img[:, :, 2], (xx - nx * r2 * (W * ca_scale)).astype(np.float32),
                             (yy - ny * r2 * (H * ca_scale)).astype(np.float32),
                             interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    img *= np.clip(1.0 - 0.18 * (r2 ** 1.15), 0.75, 1.0)[..., None]

    # 3. Linear shadow black lift + C1-continuous tanh highlight shoulder (knee = 0.80, ceil = 0.965)
    img = 0.012 + img * 0.988
    knee, ceiling = 0.80, 0.965
    span = ceiling - knee
    img = np.where(img > knee, knee + span * np.tanh((img - knee) / span), img)

    # 4. Luminance-weighted silver-halide film grain
    lum2 = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    env = np.clip(1.12 - 0.95 * np.power(np.clip(lum2, 0.0, 1.0), 1.55), 0.12, 1.12)
    grain = rng.normal(0.0, 0.0095, size=(H, W, 3)).astype(np.float32)
    mono_g = rng.normal(0.0, 0.0078, size=(H, W, 1)).astype(np.float32)
    img = np.clip(img + (0.65 * mono_g + 0.35 * grain) * env[..., None], 0.0, 1.0)

    # 5. Faint analog YIQ transfer patina (1.0 px horizontal chroma softening BORDER_REFLECT_101,
    #    slight luma ring, sub-pixel line jitter, 0.68% scanlines)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    I = 0.596 * R - 0.274 * G - 0.322 * B
    Q = 0.211 * R - 0.523 * G + 0.312 * B

    I_soft = blur_1d_x(I, 1.0)
    Q_soft = blur_1d_x(Q, 1.0)
    Y_ring = Y + 0.040 * (Y - blur_1d_x(Y, 1.1))

    out = np.empty_like(img)
    out[:, :, 0] = Y_ring + 0.956 * I_soft + 0.621 * Q_soft
    out[:, :, 1] = Y_ring - 0.272 * I_soft - 0.647 * Q_soft
    out[:, :, 2] = Y_ring - 1.106 * I_soft + 1.703 * Q_soft

    line_shift = blur_1d_y(rng.normal(0.0, 0.15, size=(H, 1)).astype(np.float32), 1.2)
    out = cv2.remap(out, (xx + line_shift).astype(np.float32), yy,
                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    out *= (1.0 - 0.0068 * (0.5 - 0.5 * np.cos(np.arange(H, dtype=np.float32) * math.pi)))[:, None, None]
    return np.clip(out, 0.0, 1.0)


# =============================================================================
# Stage 5 — Full Scene Assembly, Dual Fresnel Reflections & UV Calibration Map
# =============================================================================

def render_scene(
    seed: int = 199509,
    audit_dir: pathlib.Path | None = None,
    webp_targets: list[pathlib.Path] | None = None,
) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # Stage 1: Metric off-axis Camera3D (VP = 60.10% width, horizon = 45.70% height)
    cam = Camera3D(W, H, eye=(1.60, -0.50, 2.15), target=(-0.60, 20.0, 1.62), fov_x_deg=56.0)
    print(f"[ember-01] Camera VP: ({cam.vp_x:.1f}, {cam.vp_y:.1f}) -> {cam.vp_x / W * 100:.2f}% width, horizon {cam.horizon_y / H * 100:.2f}% height")

    # Stage 1, 2 & Band 0: Fiery sunset sky, optical-marched backlit clouds, volcanic headland & molten-copper sea
    canvas, fog_rgb, water_ripple, dist_t, sky_meta = paint_sky_headland_and_copper_sea(cam, rng)

    vd_s0, vd_s1 = -13.5, 36.0
    vd_near_uv = paint_viaduct_ashlar_uv(rng, vd_s0, vd_s1, Z_SEA, VD_Z_DECK, is_far_face=False, tw=1600, th=448)
    vd_far_uv = paint_viaduct_ashlar_uv(rng, vd_s0, vd_s1, Z_SEA, VD_Z_DECK, is_far_face=True, tw=1280, th=384)

    # =========================================================================
    # Stage 2A — 3D Plane-Mirrored Sea Reflections (Viaduct piers/arches + Tram!)
    # =========================================================================
    yy_w, xx_w = np.mgrid[0:H, 0:W].astype(np.float32)
    hy = float(cam.horizon_y)
    fresnel_sea = (0.36 + (0.90 - 0.36) * np.power(1.0 - dist_t, 1.4)) * (yy_w >= hy).astype(np.float32)
    rx_sea = np.clip(xx_w + water_ripple * (2.5 + 22.0 * np.power(dist_t, 0.78)), 0.0, W - 1.0).astype(np.float32)
    ry_sea = np.clip(yy_w + water_ripple * (1.0 + 7.5 * np.power(dist_t, 0.78)), 0.0, H - 1.0).astype(np.float32)

    vd_m_tex = vd_near_uv.copy()
    vd_m_tex[:, :, :3] *= 0.56
    vd_m_tex[:, :, 3] *= 0.56
    sea_refl_rgb = cam.warp_plane_over(
        np.zeros((H, W, 3), dtype=np.float32), vd_m_tex,
        p0=vd_point(vd_s0, 0.0, Z_SEA),
        e1=VD_DIR * (vd_s1 - vd_s0),
        e2=np.array([0.0, 0.0, -(VD_Z_DECK - Z_SEA)]),
        fog_rgb=fog_rgb, fog_start=8.5, fog_density=0.023,
    )
    vd_m_a_tex = np.concatenate([vd_m_tex[:, :, 3:4]] * 3 + [np.ones_like(vd_m_tex[:, :, 3:4])], axis=-1)
    sea_refl_a = cam.warp_plane_over(
        np.zeros((H, W, 3), dtype=np.float32), vd_m_a_tex,
        p0=vd_point(vd_s0, 0.0, Z_SEA),
        e1=VD_DIR * (vd_s1 - vd_s0),
        e2=np.array([0.0, 0.0, -(VD_Z_DECK - Z_SEA)]),
        fog_rgb=np.zeros(3, dtype=np.float32), fog_start=999.0, fog_density=0.0,
    )[:, :, 0:1]
    vd_sea_premul = np.concatenate([sea_refl_rgb, sea_refl_a], axis=-1)
    vd_sea_rippled = cv2.remap(vd_sea_premul, rx_sea, ry_sea, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0) * fresnel_sea[..., None]
    canvas = over_premul(canvas, vd_sea_rippled)

    s_sr, c_sr = new_cairo_2x()
    draw_viaduct_barrel_vaults(c_sr, cam, fog_rgb, mirror_z=Z_SEA)
    draw_viaduct_catenary_and_tram(c_sr, cam, rng, fog_rgb, mirror_z=Z_SEA)
    sr_1x = finish_cairo_2x(s_sr)
    sr_rippled = cv2.remap(sr_1x, rx_sea, ry_sea, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0) * fresnel_sea[..., None]
    canvas = over_premul(canvas, sr_rippled)

    # =========================================================================
    # Stage 1 & Band 1 — Monumental Multi-Arch Stone Viaduct + Hero Enoden Tram
    # =========================================================================
    canvas = cam.warp_plane_over(
        canvas, vd_far_uv,
        p0=vd_point(vd_s0, VD_THICK, Z_SEA),
        e1=VD_DIR * (vd_s1 - vd_s0),
        e2=np.array([0.0, 0.0, VD_Z_DECK - Z_SEA]),
        fog_rgb=fog_rgb, fog_start=8.5, fog_density=0.023,
    )
    s_bv, c_bv = new_cairo_2x()
    draw_viaduct_barrel_vaults(c_bv, cam, fog_rgb, mirror_z=None)
    canvas = over_premul(canvas, finish_cairo_2x(s_bv))

    canvas = cam.warp_plane_over(
        canvas, vd_near_uv,
        p0=vd_point(vd_s0, 0.0, Z_SEA),
        e1=VD_DIR * (vd_s1 - vd_s0),
        e2=np.array([0.0, 0.0, VD_Z_DECK - Z_SEA]),
        fog_rgb=fog_rgb, fog_start=8.5, fog_density=0.023,
    )

    g_mottle = fbm_gouache(H, W, rng)
    g_dabs = flat_brush_dabs(H, W, rng, count=94)
    g_tooth = paper_tooth(H, W, rng)

    s_b1, c_b1 = new_cairo_2x()
    draw_distant_gulls(c_b1, rng)
    draw_distant_hillside_pine(c_b1, rng, fog_rgb)
    tram_meta = draw_viaduct_catenary_and_tram(c_b1, cam, rng, fog_rgb, mirror_z=None)
    b1_1x = finish_cairo_2x(s_b1)
    a1 = b1_1x[:, :, 3:4]
    b1_1x[:, :, :3] = np.clip(b1_1x[:, :, :3] + a1 * (g_mottle * 0.015 + g_dabs * 0.011 + g_tooth * 0.005)[..., None], 0.0, 1.35)
    canvas = over_premul(canvas, b1_1x)

    # Clean upper backdrop snapshot (softly blurred & rippled like shallow rainwater!) for puddle mirror
    puddle_sky_src = blur_1d_x(blur_1d_y(canvas.copy(), 3.0), 1.6)

    # =========================================================================
    # Stage 1 — Planar Architectural Surfaces (Wet Cobbles, Curb, Parapet, Bakery)
    # =========================================================================
    par_outer_uv = paint_stone_curb_or_parapet_uv(rng, (LANE_Y0, LANE_Y1), (Z_SEA, PARAPET_Z), kind="parapet_inner", tw=1024, th=320)
    canvas = cam.warp_plane_over(
        canvas, par_outer_uv,
        p0=np.array([PARAPET_X1, LANE_Y0, Z_SEA]),
        e1=np.array([0.0, LANE_Y1 - LANE_Y0, 0.0]),
        e2=np.array([0.0, 0.0, PARAPET_Z - Z_SEA]),
        fog_rgb=fog_rgb, fog_start=8.5, fog_density=0.023,
    )

    cobble_uv, puddle_mask_uv = paint_wet_cobblestone_lane_uv(
        rng, (LANE_X0, PARAPET_X0 + 0.06), (LANE_Y0, LANE_Y1), tw=896, th=1600
    )
    canvas = cam.warp_plane_over(
        canvas, cobble_uv,
        p0=np.array([LANE_X0, LANE_Y0, Z_LANE]),
        e1=np.array([PARAPET_X0 + 0.06 - LANE_X0, 0.0, 0.0]),
        e2=np.array([0.0, LANE_Y1 - LANE_Y0, 0.0]),
        fog_rgb=fog_rgb, fog_start=8.5, fog_density=0.023,
    )

    puddle_screen = cam.warp_plane_over(
        np.zeros((H, W, 3), dtype=np.float32), puddle_mask_uv,
        p0=np.array([LANE_X0, LANE_Y0, Z_LANE]),
        e1=np.array([PARAPET_X0 + 0.06 - LANE_X0, 0.0, 0.0]),
        e2=np.array([0.0, LANE_Y1 - LANE_Y0, 0.0]),
        fog_rgb=np.zeros(3, dtype=np.float32), fog_start=999.0, fog_density=0.0,
    )[:, :, 0]
    puddle_screen = np.clip(puddle_screen, 0.0, 1.0)

    # =========================================================================
    # Stage 2B — Broad Wet Cobblestone Rainwater Puddle Dual Fresnel Reflections!
    # =========================================================================
    p_rip = resize_bicubic(rng.standard_normal((H // 6, W // 30)).astype(np.float32), W, H) * 0.55 + water_ripple * 0.45
    px_map = np.clip(xx_w + p_rip * 9.5, 0.0, W - 1.0).astype(np.float32)
    py_sky_map = np.clip(2.0 * hy - yy_w + p_rip * 4.8, 0.0, hy - 1.0).astype(np.float32)
    puddle_sky_mirror = cv2.remap(puddle_sky_src, px_map, py_sky_map, interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)

    s_pr, c_pr = new_cairo_2x()
    draw_fumikiri_crossing(c_pr, cam, rng, fog_rgb, mirror_z=Z_LANE)
    draw_utility_pole_and_wires(c_pr, cam, rng, fog_rgb, mirror_z=Z_LANE)
    draw_bakery_3d_craft_and_props(c_pr, cam, rng, fog_rgb, mirror_z=Z_LANE)
    pr_1x = finish_cairo_2x(s_pr)
    pr_1x = blur_1d_y(pr_1x, 2.0)
    pr_rippled = cv2.remap(pr_1x, px_map, np.clip(yy_w + p_rip * 3.6, 0.0, H - 1.0).astype(np.float32),
                           interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Exact Fresnel weighting: shallow puddle subtly lets submerged dark cobbles peek near edges!
    fresnel_pud = 0.40 + (0.86 - 0.40) * np.power(1.0 - dist_t, 1.4)
    puddle_w = (np.power(puddle_screen, 1.15) * fresnel_pud)[..., None]
    puddle_comp = over_premul(puddle_sky_mirror * 0.86, pr_rippled)
    canvas = canvas * (1.0 - puddle_w) + puddle_comp * puddle_w

    # Delicate wet meniscus sheen ONLY on sunward edge (gradient along +X), zero hard ring!
    gx_p = cv2.Sobel(puddle_screen, cv2.CV_32F, 1, 0, ksize=3)
    sun_meniscus = np.clip(-gx_p * 0.65, 0.0, 0.45) * np.clip(puddle_screen * 3.0, 0.0, 1.0)
    canvas += (sun_meniscus * 0.14)[..., None] * np.array([1.0, 0.68, 0.36], dtype=np.float32)

    # 2.5 Far waterfront overlook ashlar stone balustrade wall closing the lane end cleanly under Arch[0]!
    end_wall_uv = paint_stone_curb_or_parapet_uv(rng, (LANE_X0, PARAPET_X1), (Z_LANE, 0.72), kind="parapet_inner", tw=512, th=128)
    canvas = cam.warp_plane_over(
        canvas, end_wall_uv,
        p0=np.array([LANE_X0, LANE_Y1, Z_LANE]),
        e1=np.array([PARAPET_X1 - LANE_X0, 0.0, 0.0]),
        e2=np.array([0.0, 0.0, 0.72]),
        fog_rgb=fog_rgb, fog_start=8.5, fog_density=0.023,
    )

    # 3. Left Stone Sidewalk Curb
    curb_top_uv = paint_stone_curb_or_parapet_uv(rng, (LANE_X0, CURB_X1), (LANE_Y0, 21.0), kind="curb_top", tw=448, th=1024)
    canvas = cam.warp_plane_over(
        canvas, curb_top_uv,
        p0=np.array([LANE_X0, LANE_Y0, Z_CURB]),
        e1=np.array([CURB_X1 - LANE_X0, 0.0, 0.0]),
        e2=np.array([0.0, 21.0 - LANE_Y0, 0.0]),
        fog_rgb=fog_rgb, fog_start=8.5, fog_density=0.023,
    )
    curb_riser_uv = paint_stone_curb_or_parapet_uv(rng, (LANE_Y0, 21.0), (Z_LANE, Z_CURB), kind="parapet_inner", tw=768, th=64)
    canvas = cam.warp_plane_over(
        canvas, curb_riser_uv,
        p0=np.array([CURB_X1, LANE_Y0, Z_LANE]),
        e1=np.array([0.0, 21.0 - LANE_Y0, 0.0]),
        e2=np.array([0.0, 0.0, Z_CURB - Z_LANE]),
        fog_rgb=fog_rgb, fog_start=8.5, fog_density=0.023,
    )

    # 4. Right Stone Coastal Parapet
    par_in_uv = paint_stone_curb_or_parapet_uv(rng, (LANE_Y0, LANE_Y1), (Z_LANE, PARAPET_Z), kind="parapet_inner", tw=1024, th=224)
    canvas = cam.warp_plane_over(
        canvas, par_in_uv,
        p0=np.array([PARAPET_X0, LANE_Y0, Z_LANE]),
        e1=np.array([0.0, LANE_Y1 - LANE_Y0, 0.0]),
        e2=np.array([0.0, 0.0, PARAPET_Z - Z_LANE]),
        fog_rgb=fog_rgb, fog_start=8.5, fog_density=0.023,
    )
    par_top_uv = paint_stone_curb_or_parapet_uv(rng, (PARAPET_X0, PARAPET_X1), (LANE_Y0, LANE_Y1), kind="parapet_top", tw=224, th=1024)
    canvas = cam.warp_plane_over(
        canvas, par_top_uv,
        p0=np.array([PARAPET_X0 - 0.03, LANE_Y0, PARAPET_Z]),
        e1=np.array([PARAPET_X1 - PARAPET_X0 + 0.06, 0.0, 0.0]),
        e2=np.array([0.0, LANE_Y1 - LANE_Y0, 0.0]),
        fog_rgb=fog_rgb, fog_start=8.5, fog_density=0.023,
    )

    # 5. Stepped Rear Signal-Room Annex
    anx_wall_uv = paint_bakery_facade_uv(rng, (ANNEX_Y0, ANNEX_Y1), (Z_CURB, 4.10), ANNEX_X_WALL, is_annex=True, tw=512, th=384)
    canvas = cam.warp_plane_over(
        canvas, anx_wall_uv,
        p0=np.array([ANNEX_X_WALL, ANNEX_Y0, Z_CURB]),
        e1=np.array([0.0, ANNEX_Y1 - ANNEX_Y0, 0.0]),
        e2=np.array([0.0, 0.0, 4.10 - Z_CURB]),
        fog_rgb=fog_rgb, fog_start=8.5, fog_density=0.023,
    )
    anx_z_top = ANNEX_Z_EAVE + (ANNEX_X_EAVE - ANNEX_X_WALL) * ANNEX_ROOF_SLOPE
    anx_sof_uv = paint_soffit_or_awning_uv(rng, ANNEX_Y1 - ANNEX_Y0, ANNEX_X_EAVE - ANNEX_X_WALL, is_awning=False, tw=448, th=160)
    canvas = cam.warp_plane_over(
        canvas, anx_sof_uv,
        p0=np.array([ANNEX_X_WALL, ANNEX_Y0, anx_z_top]),
        e1=np.array([0.0, ANNEX_Y1 - ANNEX_Y0, 0.0]),
        e2=np.array([ANNEX_X_EAVE - ANNEX_X_WALL, 0.0, ANNEX_Z_EAVE - anx_z_top]),
        fog_rgb=fog_rgb, fog_start=8.5, fog_density=0.023,
    )
    anx_rdx = 3.2
    anx_roof_uv = paint_kawara_roof_uv(rng, ANNEX_Y1 - ANNEX_Y0 + 0.16, math.hypot(anx_rdx, anx_rdx * ANNEX_ROOF_SLOPE), tw=512, th=256)
    canvas = cam.warp_plane_over(
        canvas, anx_roof_uv,
        p0=np.array([ANNEX_X_EAVE, ANNEX_Y0, ANNEX_Z_EAVE + 0.06]),
        e1=np.array([0.0, ANNEX_Y1 - ANNEX_Y0 + 0.16, 0.0]),
        e2=np.array([-anx_rdx, 0.0, anx_rdx * ANNEX_ROOF_SLOPE]),
        fog_rgb=fog_rgb, fog_start=8.5, fog_density=0.023,
    )

    # 6. Main Corner Seaside Bakery-Shop & Signal Box
    shop_wall_uv = paint_bakery_facade_uv(rng, (SHOP_Y0, SHOP_Y1), (Z_CURB, 5.18), SHOP_X_WALL, is_annex=False, tw=1408, th=640)
    canvas = cam.warp_plane_over(
        canvas, shop_wall_uv,
        p0=np.array([SHOP_X_WALL, SHOP_Y0, Z_CURB]),
        e1=np.array([0.0, SHOP_Y1 - SHOP_Y0, 0.0]),
        e2=np.array([0.0, 0.0, 5.18 - Z_CURB]),
        fog_rgb=fog_rgb, fog_start=8.5, fog_density=0.023,
    )
    awning_uv = paint_soffit_or_awning_uv(rng, 10.6, SHOP_AWNING_X1 - SHOP_X_WALL, is_awning=True, tw=768, th=192)
    canvas = cam.warp_plane_over(
        canvas, awning_uv,
        p0=np.array([SHOP_X_WALL, 4.4, SHOP_AWNING_Z0]),
        e1=np.array([0.0, 10.6, 0.0]),
        e2=np.array([SHOP_AWNING_X1 - SHOP_X_WALL, 0.0, SHOP_AWNING_Z1 - SHOP_AWNING_Z0]),
        fog_rgb=fog_rgb, fog_start=8.5, fog_density=0.023,
    )
    shop_z_top = SHOP_Z_EAVE + (SHOP_X_EAVE - SHOP_X_WALL) * SHOP_ROOF_SLOPE
    shop_sof_uv = paint_soffit_or_awning_uv(rng, SHOP_Y1 - SHOP_Y0 + 0.18, SHOP_X_EAVE - SHOP_X_WALL, is_awning=False, tw=768, th=224)
    canvas = cam.warp_plane_over(
        canvas, shop_sof_uv,
        p0=np.array([SHOP_X_WALL, SHOP_Y0, shop_z_top]),
        e1=np.array([0.0, SHOP_Y1 - SHOP_Y0 + 0.18, 0.0]),
        e2=np.array([SHOP_X_EAVE - SHOP_X_WALL, 0.0, SHOP_Z_EAVE - shop_z_top]),
        fog_rgb=fog_rgb, fog_start=8.5, fog_density=0.023,
    )
    shop_rdx = 3.8
    shop_roof_uv = paint_kawara_roof_uv(rng, SHOP_Y1 - SHOP_Y0 + 0.18, math.hypot(shop_rdx, shop_rdx * SHOP_ROOF_SLOPE), tw=1024, th=384)
    canvas = cam.warp_plane_over(
        canvas, shop_roof_uv,
        p0=np.array([SHOP_X_EAVE, SHOP_Y0, SHOP_Z_EAVE + 0.08]),
        e1=np.array([0.0, SHOP_Y1 - SHOP_Y0 + 0.18, 0.0]),
        e2=np.array([-shop_rdx, 0.0, shop_rdx * SHOP_ROOF_SLOPE]),
        fog_rgb=fog_rgb, fog_start=8.5, fog_density=0.023,
    )

    # Volumetric golden sunset sunbeams cutting diagonally down-left under viaduct arches!
    beams = paint_viaduct_sunbeams(cam, rng, sky_meta["sun_uv"], intensity=0.29)
    canvas = 1.0 - (1.0 - np.clip(canvas, 0.0, 1.0)) * (1.0 - np.clip(beams, 0.0, 0.72))

    # =========================================================================
    # Stage 3 Band 2 — Fumikiri Crossing, Bakery 3D Craft/Props & Utility Pole
    # =========================================================================
    s_b2, c_b2 = new_cairo_2x()
    fumi_meta = draw_fumikiri_crossing(c_b2, cam, rng, fog_rgb, mirror_z=None)
    shop_meta = draw_bakery_3d_craft_and_props(c_b2, cam, rng, fog_rgb, mirror_z=None)
    draw_utility_pole_and_wires(c_b2, cam, rng, fog_rgb, mirror_z=None)
    b2_1x = finish_cairo_2x(s_b2)
    a2 = b2_1x[:, :, 3:4]
    b2_1x[:, :, :3] = np.clip(b2_1x[:, :, :3] + a2 * (g_mottle * 0.015 + g_dabs * 0.011 + g_tooth * 0.0055)[..., None], 0.0, 1.35)
    canvas = over_premul(canvas, b2_1x)

    # =========================================================================
    # Stage 3 Band 3 — Sculptural Coastal Black Pine & Natural Autumn Ivy Vines
    # =========================================================================
    s_b3, c_b3 = new_cairo_2x()
    draw_organic_pine_and_autumn_ivy(c_b3, cam, rng, fog_rgb)
    b3_1x = finish_cairo_2x(s_b3)
    a3 = b3_1x[:, :, 3:4]
    b3_1x[:, :, :3] = np.clip(b3_1x[:, :, :3] + a3 * (g_mottle * 0.016 + g_dabs * 0.012 + g_tooth * 0.005)[..., None], 0.0, 1.35)
    canvas = over_premul(canvas, b3_1x)

    # Warm ember shadow bounce (lum < 0.28)
    canvas = apply_warm_shadow_bounce(canvas, strength=0.022)

    # =========================================================================
    # Stage 4 — Foreground Disc Bokeh + 35mm Optical Finish & YIQ Tape Patina
    # =========================================================================
    fg_bokeh, fg_meta = paint_foreground_bokeh_elements(rng)
    canvas = over_premul(canvas, fg_bokeh)

    lamp_glints = [
        (fumi_meta["fumi_lamp_l_uv"][0], fumi_meta["fumi_lamp_l_uv"][1], (0.58, 0.16, 0.07), 22.0),
        (fumi_meta["fumi_lamp_r_uv"][0], fumi_meta["fumi_lamp_r_uv"][1], (0.42, 0.12, 0.05), 18.0),
        (shop_meta["shop_porch_lamp_uv"][0], shop_meta["shop_porch_lamp_uv"][1], (0.48, 0.30, 0.11), 24.0),
        (tram_meta["tram_headlamp_uv"][0], tram_meta["tram_headlamp_uv"][1], (0.44, 0.32, 0.14), 16.0),
    ]
    final_rgb = apply_35mm_finish(canvas, rng, sun_uv=sky_meta["sun_uv"], lamp_glints=lamp_glints)
    elapsed = time.perf_counter() - t0
    print(f"[ember-01] Rendered {W}x{H} in {elapsed:.2f}s")

    # =========================================================================
    # Stage 5 — Exact Camera3D UV Calibration Report for wall-01.frag
    # =========================================================================
    p_nz = np.where(puddle_screen > 0.25)
    pud_u0, pud_u1 = (float(p_nz[1].min() / W), float(p_nz[1].max() / W)) if len(p_nz[0]) else (0.25, 0.65)
    pud_v0, pud_v1 = (float(p_nz[0].min() / H), float(p_nz[0].max() / H)) if len(p_nz[0]) else (0.70, 0.99)

    print("[ember-01] === EXACT CAMERA3D UV CALIBRATION FOR wall-01.frag ===")
    print(f"  horizon_v         = {sky_meta['horizon_v']:.4f}")
    print(f"  sun_uv            = ({sky_meta['sun_uv'][0]:.4f}, {sky_meta['sun_uv'][1]:.4f})")
    print(f"  cape_beacon_uv    = ({sky_meta['cape_beacon_uv'][0]:.4f}, {sky_meta['cape_beacon_uv'][1]:.4f})")
    print(f"  fumi_lamp_l_uv    = ({fumi_meta['fumi_lamp_l_uv'][0]:.4f}, {fumi_meta['fumi_lamp_l_uv'][1]:.4f})")
    print(f"  fumi_lamp_r_uv    = ({fumi_meta['fumi_lamp_r_uv'][0]:.4f}, {fumi_meta['fumi_lamp_r_uv'][1]:.4f})")
    print(f"  shop_porch_lamp   = ({shop_meta['shop_porch_lamp_uv'][0]:.4f}, {shop_meta['shop_porch_lamp_uv'][1]:.4f})")
    print(f"  tram_u            = ({tram_meta['tram_u'][0]:.4f}, {tram_meta['tram_u'][1]:.4f})")
    print(f"  tram_v            = ({tram_meta['tram_v'][0]:.4f}, {tram_meta['tram_v'][1]:.4f})")
    print(f"  tram_headlamp_uv  = ({tram_meta['tram_headlamp_uv'][0]:.4f}, {tram_meta['tram_headlamp_uv'][1]:.4f})")
    print(f"  puddle_uv_bbox    = u:({pud_u0:.4f}, {pud_u1:.4f}) v:({pud_v0:.4f}, {pud_v1:.4f})")
    print("  parapet_outer_edge (x=PARAPET_X1):")
    for y_c in (5.0, 12.0, 20.0, 30.0, LANE_Y1):
        pu, pv, _ = cam.project_uv_norm((PARAPET_X1, y_c, PARAPET_Z))
        pu0, pv0, _ = cam.project_uv_norm((PARAPET_X1, y_c, Z_SEA))
        print(f"    y={y_c:4.1f}m -> top=({pu:.4f}, {pv:.4f}) sea_base=({pu0:.4f}, {pv0:.4f})")
    print("  far_overlook_balustrade (y=LANE_Y1):")
    for x_c in (LANE_X0, 0.0, PARAPET_X1):
        ou, ov, _ = cam.project_uv_norm((x_c, LANE_Y1, 0.76))
        print(f"    x={x_c:5.2f}m -> top=({ou:.4f}, {ov:.4f})")
    print("  viaduct_piers_u (pier edges at sea level):")
    for idx_p in (0, 1, 2):
        sp = idx_p * VD_SPAN
        u_l, v_b, _ = cam.project_uv_norm(vd_point(sp - VD_PIER_W * 0.5, 0.0, Z_SEA))
        u_r, _, _ = cam.project_uv_norm(vd_point(sp + VD_PIER_W * 0.5, 0.0, Z_SEA))
        _, v_dk, _ = cam.project_uv_norm(vd_point(sp, 0.0, VD_Z_DECK))
        print(f"    Pier[{idx_p}]: u=({u_l:.4f}..{u_r:.4f}) deck_v={v_dk:.4f} sea_v={v_b:.4f}")
    print("  shop_windows_uv:")
    for label_w, pt_w in (
        ("win_g1", (SHOP_X_WALL, 7.95, 1.66)),
        ("door",   (SHOP_X_WALL, 11.20, 1.40)),
        ("win_g2", (SHOP_X_WALL, 13.95, 1.66)),
        ("win_u1", (SHOP_X_WALL, 7.95, 3.60)),
        ("win_u2", (SHOP_X_WALL, 11.30, 3.60)),
    ):
        wu, wv, _ = cam.project_uv_norm(pt_w)
        print(f"    {label_w:6s} = ({wu:.4f}, {wv:.4f})")
    print(f"  fg_upper_uv       = u>{fg_meta['fg_upper_u_min']:.4f}, v<{fg_meta['fg_upper_v_max']:.4f}")
    print(f"  fg_lower_uv       = u>{fg_meta['fg_lower_u_min']:.4f}, v>{fg_meta['fg_lower_v_min']:.4f}")

    u8 = np.round(np.clip(final_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    full_im = Image.fromarray(u8, mode="RGB")

    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        full_im.save(audit_dir / "w01_full.png", optimize=True)
        crops = {
            "crop_01_viaduct_tram_sun.png": (int(W * 0.48), int(H * 0.14)),
            "crop_02_puddle_bakery_windows.png": (int(W * 0.05), int(H * 0.44)),
            "crop_03_crossing_parapet_bokeh.png": (int(W * 0.36), int(H * 0.46)),
        }
        for cname, (cx0, cy0) in crops.items():
            x0 = max(0, min(W - 600, cx0))
            y0 = max(0, min(H - 450, cy0))
            full_im.crop((x0, y0, x0 + 600, y0 + 450)).save(audit_dir / cname)

    targets = webp_targets if webp_targets is not None else [DEFAULT_THEME_WEBP, DEFAULT_DOCS_WEBP]
    if targets:
        first = targets[0]
        first.parent.mkdir(parents=True, exist_ok=True)
        full_im.save(first, format="WEBP", quality=94, method=6)
        print(f"[ember-01] Saved {W}x{H} WebP (q=94): {first} ({first.stat().st_size / 1024:.1f} KiB)")
        for extra in targets[1:]:
            extra.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(first, extra)
            print(f"[ember-01] Copied {W}x{H} WebP: {extra} ({extra.stat().st_size / 1024:.1f} KiB)")

    return final_rgb, {
        "elapsed": elapsed,
        "vp_ratio": cam.vp_x / W,
        **sky_meta,
        **tram_meta,
        **fumi_meta,
        **shop_meta,
        **fg_meta,
        "puddle_u": (pud_u0, pud_u1),
        "puddle_v": (pud_v0, pud_v1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Render Ember 01 (Crimson Viaduct & Seaside Tram Crossing) 2048x1152 wallpaper")
    ap.add_argument("--seed", type=int, default=199509)
    ap.add_argument("--audit-dir", type=pathlib.Path, default=None)
    ap.add_argument("--no-webp", action="store_true")
    args = ap.parse_args()
    targets = [] if args.no_webp else [DEFAULT_THEME_WEBP, DEFAULT_DOCS_WEBP]
    render_scene(seed=args.seed, audit_dir=args.audit_dir, webp_targets=targets)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Matcha 02 — "Misty Terraced Tea Valley & Cedar Watermill" (2048 x 1152)

1990s Theatrical Cel-Anime Background Painting Pipeline (Studio Ghibli *Only Yesterday* /
*My Neighbor Totoro* Uji & Wazuka highland pastoral vista — pure Python, zero image models):
  Stage 1: Metric right-handed Camera3D (off-axis vanishing point ~60.0% width, horizon ~43.5%),
           exact 3x3 UV-to-screen plane homographies masked by UV bounds and ray-plane depth
           z_cam > 0.05 with exponential silver-mint valley mist, 3D near-plane line/polygon
           clipping, physical-meter jittered 3x3 Voronoi (d2 - d1) weathered granite flagstones
           & stepped dry-stone retaining walls (ishigaki) with Sobel bevels & velvety dark moss
           crevices, and angled (+18 deg) 3/4 weathered cedar watermill walls, lime-plaster gable,
           timber porch deck & pitched mossy shingle roof.
  Stage 2: Single locked normalized golden morning SUN_DIR (SUN_DIR[2] < 0) raking from
           upper-right across the emerald tea ridges onto the cedar watermill, traced backward
           with 6 height-refinement steps against the pitched watermill roof/eaves & roadside
           hedgerow + warm hearth-window point lights; plus crystal jade mountain millrace Fresnel
           reflections combining cv2.remap flowing ripples (y_src = 2*horizon_y - y) with 3D
           plane-mirrored quads of the spoked waterwheel, mill wall, glowing window & stone slab.
  Stage 3: 4 depth-interleaved 2x supersampled PyCairo + volumetric NumPy gouache bands:
             Band 0: Luminous pale sage-cream dawn sky, sculptural sunlit morning clouds, 4 receding
                     misty blue-jade mountain ridges, and silky silver-mint unkai sea-of-clouds.
             Band 1: Sweeping volumetric sculpted contours of Uji terraced green tea plantations
                     (chabatake) with sunlit golden-matcha rounded bush crowns, deep velvety emerald
                     furrows, diagonal cloud-shadow/sunbeam valley sweeps, fine gouache leaf stipple,
                     winding harvest paths, distant wooden tea shed, and organic soft-crowned cypresses.
             Band 2: Hero 3/4 cedar watermill porch timbers, open spoked 3D wooden waterwheel (suisha)
                     with staggered timber felloes, wet mossy paddles, multi-strand kakehi water arc &
                     painterly aerated splash foam, textured granite foot-slab bridge with dry-brush
                     velvet moss, interlocking woven bamboo tea-picking baskets (chatsumi-kago)
                     brimming with fresh matcha leaves, folded burlap tea sacks & cedar bench.
             Band 3: Razor-tapered cubic-Bezier dewy meadow sedge, flowering highland tea shrubs &
                     delicate 5-petal white tea blossoms (cha-no-hana); enriched by isotropic
                     fbm_gouache, 94 angled flat-brush dabs at 1/4 res, paper tooth & warm shadow bounce.
  Stage 4: Near-camera dewy tea/camellia twig & leaves convolved in premultiplied RGBA with
           make_disc_kernel(radius=7.8) + separate cv2.warpAffine ~1px opposite R/B shift,
           followed by 3-scale cream-mint highlight halation (sigma=6.5, 24, 72 px above lum>0.72),
           radial corner CA, optical vignette, shadow black lift + C1 tanh highlight shoulder
           (knee=0.80, ceiling=0.965), luminance-weighted silver-halide grain, and faint YIQ patina.
  Stage 5: Deterministic CPU render (<60s), exact Camera3D UV telemetry for wall-02.frag,
           2048x1152 WebP export (quality=94), and 3 1:1 (600x450) macro audit crops.
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

# Stage 2: Single locked normalized golden morning light vector raking from upper-right -> lower-left
_raw_sun = np.array([-0.68, -0.25, -0.689], dtype=np.float64)
SUN_DIR = _raw_sun / np.linalg.norm(_raw_sun)

# World elevation constants (meters)
Z_STREAM = -0.22
Z_PATH = 0.00
Z_PORCH = 0.08
Z_MILL_BASE = -0.26
MILL_Z_BASE = Z_MILL_BASE

# Angled (+18 deg) 3/4 cedar watermill local orthonormal frame for heroic open spoked waterwheel view!
MILL_YAW = math.radians(18.0)
MILL_FWD = np.array([math.sin(MILL_YAW), math.cos(MILL_YAW), 0.0], dtype=np.float64)   # along mill length (+L)
MILL_RGT = np.array([math.cos(MILL_YAW), -math.sin(MILL_YAW), 0.0], dtype=np.float64)  # outward right wall normal (+U)
MILL_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# Mill front-right wall corner at ground level (u_m=0 is right wall, l_m=0 is front wall)
MILL_ORIGIN = np.array([-2.22, 8.75, 0.0], dtype=np.float64)
MILL_WIDTH = 3.80     # u_m in [-MILL_WIDTH, 0.0]
MILL_LENGTH = 6.10    # l_m in [0.0, MILL_LENGTH]
PORCH_DEPTH = 1.85    # l_m in [-PORCH_DEPTH, 0.0]
MILL_Z_EAVE = 2.60
MILL_Z_RIDGE = 4.18
ROOF_OVER_R = 0.72    # right eave overhang past u_m=0 -> u_m = +0.72
ROOF_OVER_F = 2.08    # front eave overhang sheltering timber porch -> l_m = -2.08
ROOF_OVER_B = 0.58
ROOF_Z_EAVE_R = 2.28

# Open 3D spoked wooden waterwheel (suisha) mounted cleanly BELOW the roof eave (z_top = 1.90m < 2.28m)!
WHEEL_U_INNER = 0.14
WHEEL_U_OUTER = 0.56
WHEEL_L_CENTER = 2.28
WHEEL_Z_CENTER = 0.82
WHEEL_R_OUTER = 1.08
WHEEL_R_INNER = 0.86


def mill_pt(u_m: float, l_m: float, z_m: float) -> np.ndarray:
    """Convert mill-local (u_m right, l_m forward along wall, z_m elevation) to 3D world meters."""
    return MILL_ORIGIN + u_m * MILL_RGT + l_m * MILL_FWD + z_m * MILL_UP


def world_to_mill_local(wx: np.ndarray, wy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = wx - MILL_ORIGIN[0]
    dy = wy - MILL_ORIGIN[1]
    u_m = dx * MILL_RGT[0] + dy * MILL_RGT[1]
    l_m = dx * MILL_FWD[0] + dy * MILL_FWD[1]
    return u_m, l_m


# Warm glowing tea-drying room windows (explicit 3D hearth point lights)
SIDE_WIN_CENTER = mill_pt(0.06, 4.72, 1.50)
FRONT_WIN_CENTER = mill_pt(-1.32, -0.08, 1.50)
HEARTH_LIGHTS = (
    (SIDE_WIN_CENTER, 0.95),
    (FRONT_WIN_CENTER, 0.80),
)


def millrace_left_x(wy: np.ndarray | float) -> np.ndarray | float:
    """Left stone edge of the mountain millrace hugging the angled cedar watermill wall."""
    t_m = (wy - MILL_ORIGIN[1]) / MILL_FWD[1]
    x_wall = MILL_ORIGIN[0] + t_m * MILL_FWD[0]
    fore_bend = -0.26 * np.clip((8.4 - wy) / 5.5, 0.0, 1.0)
    far_bend = 0.14 * np.sin((wy - 8.5) * 0.24) * np.clip((wy - 14.5) / 4.0, 0.0, 1.0)
    return x_wall + fore_bend + far_bend


def millrace_right_x(wy: np.ndarray | float) -> np.ndarray | float:
    """Right stone retaining edge of the millrace bordering the mossy highland stone footpath."""
    xl = millrace_left_x(wy)
    width = 2.04 - 0.24 * np.clip((wy - 6.0) / 11.0, 0.0, 1.0) + 0.14 * np.sin((wy - 4.2) * 0.22)
    return xl + width


def path_right_x(wy: np.ndarray | float) -> np.ndarray | float:
    """Valley-side curving rim of the highland footpath overlooking the sweeping emerald tea hills."""
    xr = millrace_right_x(wy)
    p_w = 2.08 - 0.62 * np.clip((wy - 5.5) / 13.5, 0.0, 1.0) + 0.22 * np.sin((wy - 3.0) * 0.22)
    return xr + p_w


def mill_roof_z_at(wx: np.ndarray, wy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact pitched watermill roof elevation in angled mill-local space for 6-step -SUN_DIR tracing."""
    u_m, l_m = world_to_mill_local(wx, wy)
    u_ridge = -0.5 * MILL_WIDTH
    slope = (MILL_Z_RIDGE - ROOF_Z_EAVE_R) / (ROOF_OVER_R - u_ridge)
    z_roof = MILL_Z_RIDGE - slope * np.abs(u_m - u_ridge)
    in_fp = (u_m >= -MILL_WIDTH - 0.55) & (u_m <= ROOF_OVER_R) & (l_m >= -ROOF_OVER_F) & (l_m <= MILL_LENGTH + ROOF_OVER_B)
    return z_roof, in_fp


def roadside_hedge_z_at(wx: np.ndarray, wy: np.ndarray) -> np.ndarray:
    """Undulating highland tea-bush border along outer right path edge for unified -SUN_DIR shadows."""
    px_r = path_right_x(wy)
    band = np.exp(-((wx - (px_r + 0.65)) / 0.80) ** 2)
    h_tea = 0.72 + 0.22 * np.sin(wy * 1.1 + wx * 0.8) + 0.12 * np.cos(wy * 2.1 - wx * 1.4)
    return band * h_tea


# =============================================================================
# Core Vectorized Primitives (NumPy + OpenCV + PyCairo)
# =============================================================================

def gaussian_blur_2d(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.1:
        return arr.copy()
    if sigma <= 9.0:
        return cv2.GaussianBlur(
            arr.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101
        )
    h, w = arr.shape[:2]
    factor = 4 if sigma < 40.0 else 8
    sw, sh = max(8, w // factor), max(8, h // factor)
    small = cv2.resize(arr.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(
        small, (0, 0), sigmaX=sigma / factor, sigmaY=sigma / factor, borderType=cv2.BORDER_REFLECT_101
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def resize_bicubic(arr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    return cv2.resize(arr.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_CUBIC)


def fbm_gouache(
    h: int,
    w: int,
    rng: np.random.Generator,
    scales=((72, 70), (27, 29), (10.5, 10.0), (4.6, 4.8)),
) -> np.ndarray:
    """Isotropic square-scale multi-octave gouache mottle (NEVER stretched row/col noise!)."""
    acc = np.zeros((h, w), dtype=np.float32)
    weights = (0.46, 0.28, 0.17, 0.09)
    for (sy, sx), wt in zip(scales, weights):
        gh = max(4, int(round(h / sy)))
        gw = max(4, int(round(w / sx)))
        acc += wt * resize_bicubic(rng.standard_normal((gh, gw)).astype(np.float32), w, h)
    acc -= float(acc.mean())
    return acc / (float(acc.std()) + 1e-6)


def paper_tooth(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    coarse = resize_bicubic(rng.standard_normal((max(4, h // 3), max(4, w // 3))).astype(np.float32), w, h)
    fine = rng.standard_normal((h, w)).astype(np.float32)
    t = 0.56 * coarse + 0.44 * fine
    t -= float(t.mean())
    return t / (float(t.std()) + 1e-6)


def flat_brush_dabs(h: int, w: int, rng: np.random.Generator, count: int = 94) -> np.ndarray:
    """85-100 angled flat-brush gouache dabs rendered at 1/4 res and softly blended."""
    qh, qw = max(16, h // 4), max(16, w // 4)
    arr_sum = np.zeros((qh, qw), dtype=np.float32)
    s_one = cairo.ImageSurface(cairo.FORMAT_A8, qw, qh)
    c1 = cairo.Context(s_one)
    stride = s_one.get_stride()
    for _ in range(count):
        cx = float(rng.uniform(0, qw))
        cy = float(rng.uniform(0, qh))
        ang = float(rng.uniform(0, math.pi))
        rx = float(rng.uniform(qw * 0.030, qw * 0.092))
        ry = float(rng.uniform(rx * 0.26, rx * 0.58))
        n_pts = int(rng.integers(5, 8))
        strength = float(rng.uniform(0.18, 0.40)) * (1.0 if rng.random() > 0.5 else -1.0)
        c1.save()
        c1.set_operator(cairo.OPERATOR_CLEAR)
        c1.paint()
        c1.set_operator(cairo.OPERATOR_OVER)
        c1.translate(cx, cy)
        c1.rotate(ang)
        pts = []
        for k in range(n_pts):
            a = (math.tau * k) / n_pts + float(rng.uniform(-0.22, 0.22))
            rj = float(rng.uniform(0.78, 1.18))
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
    arr_sum -= float(arr_sum.mean())
    arr_sum /= float(arr_sum.std()) + 1e-6
    return resize_bicubic(arr_sum, w, h)


def over_premul(dst_rgb: np.ndarray, src_premul_rgba: np.ndarray) -> np.ndarray:
    a = np.clip(src_premul_rgba[:, :, 3:4], 0.0, 1.0)
    return src_premul_rgba[:, :, :3] + dst_rgb * (1.0 - a)


def apply_shadow_bounce(rgb: np.ndarray, bounce_rgb=(0.016, 0.028, 0.015)) -> np.ndarray:
    """Warm golden-sage highland bounce into deep shadows (lum < 0.28)."""
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    w_sh = np.clip((0.28 - lum) / 0.28, 0.0, 1.0)
    return rgb + (w_sh * w_sh)[..., None] * np.array(bounce_rgb, dtype=np.float32)


def quad_to(ctx: cairo.Context, cx: float, cy: float, x2: float, y2: float) -> None:
    x0, y0 = ctx.get_current_point()
    ctx.curve_to(
        x0 + (2.0 / 3.0) * (cx - x0),
        y0 + (2.0 / 3.0) * (cy - y0),
        x2 + (2.0 / 3.0) * (cx - x2),
        y2 + (2.0 / 3.0) * (cy - y2),
        x2,
        y2,
    )


def new_cairo_2x() -> tuple[cairo.ImageSurface, cairo.Context]:
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W * 2, H * 2)
    ctx = cairo.Context(surf)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.scale(2.0, 2.0)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    return surf, ctx


def finish_cairo_2x(surf: cairo.ImageSurface) -> np.ndarray:
    """Clean unpremultiply -> Lanczos downsample 2x -> 1x -> re-premultiply (zero dark fringes)."""
    surf.flush()
    buf = np.frombuffer(surf.get_data(), dtype=np.uint8).reshape(H * 2, W * 2, 4)
    a2 = buf[:, :, 3:4].astype(np.float32) / 255.0
    inv_a = np.where(a2 > 1e-4, 1.0 / np.maximum(a2, 1e-4), 0.0)
    rgb2 = np.stack([buf[:, :, 2], buf[:, :, 1], buf[:, :, 0]], axis=-1).astype(np.float32) * (inv_a / 255.0)
    rgb2 = np.clip(rgb2, 0.0, 1.0)
    rgb1 = cv2.resize(rgb2, (W, H), interpolation=cv2.INTER_LANCZOS4)
    a1 = np.clip(cv2.resize(a2[:, :, 0], (W, H), interpolation=cv2.INTER_LANCZOS4), 0.0, 1.0)
    out = np.empty((H, W, 4), dtype=np.float32)
    out[:, :, :3] = np.clip(rgb1, 0.0, 1.0) * a1[..., None]
    out[:, :, 3] = a1
    return out


def fog_mix(
    c3: Iterable[float], dist: float, fog_rgb: np.ndarray, fog_start: float = 6.8, fog_density: float = 0.036
) -> tuple[float, float, float]:
    ft = 1.0 - math.exp(-max(0.0, dist - fog_start) * fog_density)
    c = np.asarray(c3, dtype=np.float64) * (1.0 - ft) + fog_rgb * ft
    return (float(c[0]), float(c[1]), float(c[2]))


def draw_velvet_moss_tuft(
    ctx: cairo.Context,
    cx: float,
    cy: float,
    rad: float,
    srng: np.random.Generator,
    dist: float,
    fog_rgb: np.ndarray,
    sun_w: float = 0.55,
) -> None:
    """Soft painterly dry-brush gouache velvet moss cushion (ZERO spiky triangles or flat ovals!)."""
    # Soft feathered dark moss base stain blending into stone
    ctx.save()
    ctx.translate(cx, cy + rad * 0.08)
    ctx.scale(1.45, 0.58)
    g0 = cairo.RadialGradient(0, 0, rad * 0.15, 0, 0, rad * 1.15)
    g0.add_color_stop_rgba(0.0, *fog_mix([0.08, 0.18, 0.08], dist, fog_rgb), 0.72)
    g0.add_color_stop_rgba(0.7, *fog_mix([0.11, 0.23, 0.09], dist, fog_rgb), 0.42)
    g0.add_color_stop_rgba(1.0, *fog_mix([0.11, 0.23, 0.09], dist, fog_rgb), 0.0)
    ctx.set_source(g0)
    ctx.arc(0, 0, rad * 1.15, 0, math.tau)
    ctx.fill()
    ctx.restore()

    # Soft overlapping gouache velvet moss dabs + sunlit mossy chartreuse micro-stipples
    n_micro = int(min(42, max(16, rad * 1.8)))
    for _ in range(n_micro):
        ang = float(srng.uniform(0.0, math.tau))
        rr = rad * (float(srng.uniform(0.0, 1.0)) ** 0.72)
        mx = cx + math.cos(ang) * rr * 1.32
        my = cy + math.sin(ang) * rr * 0.48
        rx_d = max(1.2, rad * float(srng.uniform(0.11, 0.24)))
        ry_d = rx_d * float(srng.uniform(0.45, 0.78))
        lit = max(0.0, min(1.0, sun_w * (0.45 + 0.55 * math.cos(ang - 0.5)) + float(srng.uniform(-0.18, 0.22))))
        c = np.array([0.09, 0.19, 0.08]) * (1.0 - lit) + np.array([0.42, 0.58, 0.21]) * lit
        ctx.save()
        ctx.translate(mx, my)
        ctx.rotate(float(srng.uniform(-0.4, 0.4)))
        ctx.scale(rx_d, ry_d)
        ctx.set_source_rgba(*fog_mix(c, dist, fog_rgb), float(srng.uniform(0.38, 0.78)))
        ctx.arc(0, 0, 1.0, 0, math.tau)
        ctx.fill()
        ctx.restore()


# =============================================================================
# Stage 1 — Metric 3D Pinhole Camera & Exact 3x3 UV-to-Screen Homographies
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
        fog_start: float = 6.8,
        fog_density: float = 0.036,
        dst_rgba_mode: bool = False,
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
        if dst_rgba_mode:
            dst_sub = dst_rgb[y0:y1, x0:x1]
            out_rgb = fg_rgb + dst_sub[:, :, :3] * (1.0 - alpha)
            out_a = alpha + dst_sub[:, :, 3:4] * (1.0 - alpha)
            dst_rgb[y0:y1, x0:x1] = np.concatenate([out_rgb, out_a], axis=-1)
        else:
            dst_rgb[y0:y1, x0:x1] = over_premul(dst_rgb[y0:y1, x0:x1], patch)
        return dst_rgb


def mirror_point_z(p: np.ndarray, z_plane: float = Z_STREAM) -> np.ndarray:
    out = p.copy()
    out[2] = 2.0 * z_plane - p[2]
    return out


# =============================================================================
# Stage 2 — Unified -SUN_DIR 6-Step Roof/Eave & Valley Shadow Solver
# =============================================================================

def trace_valley_sunlight(wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> np.ndarray:
    """
    Trace any 3D world coordinate backward along -SUN_DIR using 6 height-refinement steps
    against the pitched angled watermill roof/eaves, spoked waterwheel, stone foot-slab,
    and roadside highland tea hedgerow!
    """
    inv_sz = 1.0 / (-SUN_DIR[2])
    sun = np.ones_like(wx, dtype=np.float32)

    # 1. 6-step height refinement against pitched angled watermill roof & eaves
    z_est = np.full_like(wx, 3.05, dtype=np.float64)
    for _ in range(6):
        t = np.maximum(0.0, (z_est - wz) * inv_sz)
        sx = wx - SUN_DIR[0] * t
        sy = wy - SUN_DIR[1] * t
        z_est, _ = mill_roof_z_at(sx, sy)
        z_est = np.clip(z_est, ROOF_Z_EAVE_R, MILL_Z_RIDGE)

    t_r = np.maximum(0.0, (z_est - wz) * inv_sz)
    sx_r = wx - SUN_DIR[0] * t_r
    sy_r = wy - SUN_DIR[1] * t_r
    u_r, l_r = world_to_mill_local(sx_r, sy_r)
    mu = np.clip((u_r - (-MILL_WIDTH - 0.55)) / 0.14, 0.0, 1.0) * np.clip((ROOF_OVER_R - u_r) / 0.14, 0.0, 1.0)
    ml = np.clip((l_r - (-ROOF_OVER_F)) / 0.16, 0.0, 1.0) * np.clip((MILL_LENGTH + ROOF_OVER_B - l_r) / 0.16, 0.0, 1.0)
    roof_sh = np.where(wz < (z_est - 0.03), mu * ml, 0.0).astype(np.float32)
    sun *= (1.0 - 0.80 * roof_sh)

    # 2. Spoked waterwheel shadow cast cleanly onto the right cedar clapboard wall
    u_pt, l_pt = world_to_mill_local(wx, wy)
    sun_u = float(np.dot(SUN_DIR, MILL_RGT))
    sun_l = float(np.dot(SUN_DIR, MILL_FWD))
    t_w = (u_pt - WHEEL_U_OUTER) / sun_u
    hl_w = l_pt - t_w * sun_l
    hz_w = wz - t_w * SUN_DIR[2]
    d_w = np.hypot(hl_w - WHEEL_L_CENTER, hz_w - WHEEL_Z_CENTER)
    rim_s = np.clip((WHEEL_R_OUTER - d_w) / 0.10, 0.0, 1.0) * np.clip((d_w - (WHEEL_R_INNER - 0.08)) / 0.10, 0.0, 1.0)
    hub_s = np.clip((0.32 - d_w) / 0.10, 0.0, 1.0)
    sun *= (1.0 - np.where(t_w > 0.02, np.maximum(rim_s * 0.72, hub_s * 0.65), 0.0).astype(np.float32))

    # 3. 6-step height refinement against roadside highland tea hedgerow
    z_h = np.full_like(wx, 0.65, dtype=np.float64)
    for _ in range(6):
        th_s = np.maximum(0.0, (z_h - wz) * inv_sz)
        hx = wx - SUN_DIR[0] * th_s
        hy = wy - SUN_DIR[1] * th_s
        z_h = roadside_hedge_z_at(hx, hy)
    th_s = np.maximum(0.0, (z_h - wz) * inv_sz)
    hx = wx - SUN_DIR[0] * th_s
    hy = wy - SUN_DIR[1] * th_s
    z_hf = roadside_hedge_z_at(hx, hy)
    hedge_sh = np.clip((z_hf - wz - 0.04) / 0.30, 0.0, 1.0).astype(np.float32)
    sun *= (1.0 - 0.42 * hedge_sh)

    # 4. Crisp shadow cast underneath rustic granite foot-slab bridge (y in [6.20, 7.05])
    t_b = np.maximum(0.0, (0.04 - wz) * inv_sz)
    bx = wx - SUN_DIR[0] * t_b
    by = wy - SUN_DIR[1] * t_b
    bx0_s = float(millrace_left_x(6.62)) - 0.16
    bx1_s = float(millrace_right_x(6.62)) + 0.18
    b_m = (
        np.clip((bx - bx0_s) / 0.10, 0.0, 1.0)
        * np.clip((bx1_s - bx) / 0.10, 0.0, 1.0)
        * np.clip((by - 6.18) / 0.10, 0.0, 1.0)
        * np.clip((7.08 - by) / 0.10, 0.0, 1.0)
    )
    sun *= (1.0 - 0.78 * np.where(wz < 0.0, b_m, 0.0)).astype(np.float32)

    return np.clip(sun, 0.0, 1.0)


def eval_hearth_lights(wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> np.ndarray:
    acc = np.zeros_like(wx, dtype=np.float32)
    for lpos, strength in HEARTH_LIGHTS:
        d2 = (wx - lpos[0]) ** 2 + (wy - lpos[1]) ** 2 + (wz - lpos[2]) ** 2
        acc += (strength / (0.85 + d2 * 0.78)).astype(np.float32) * np.exp(-d2 * 0.16).astype(np.float32)
    return np.clip(acc, 0.0, 1.3)


# =============================================================================
# Stage 3 Band 0 — Dramatic Ghibli Dawn Sky, 4 Misty Mountain Ridges & Unkai Mist
# =============================================================================

def paint_dawn_valley_backdrop(cam: Camera3D, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Band 0: Theatrical Studio Ghibli morning sky (crisp highland blue-celadon zenith -> luminous
    pale silver-mint -> warm golden-cream dawn glow upper-right), sculpted sunlit morning cloud
    banks, 4 receding atmospheric blue-jade mountain ridges, and billowing unkai valley mist.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = xx / W
    v = yy / H

    mottle = fbm_gouache(H, W, rng)
    mottle_f = fbm_gouache(H, W, rng, scales=((30, 30), (13, 13), (6.0, 6.0), (2.8, 2.8)))
    dabs = flat_brush_dabs(H, W, rng, count=94)
    tooth = paper_tooth(H, W, rng)

    hv = float(cam.horizon_y / H)

    # Richer morning sky depth: cool mountain celadon-blue zenith -> silver-mint -> luminous cream dawn
    c_zenith = np.array([0.365, 0.565, 0.575], dtype=np.float32)
    c_midsky = np.array([0.665, 0.815, 0.775], dtype=np.float32)
    c_horiz = np.array([0.935, 0.962, 0.880], dtype=np.float32)

    t_sky = np.clip(v / max(0.32, hv + 0.06), 0.0, 1.0)
    t1 = np.clip(t_sky / 0.54, 0.0, 1.0)
    t2 = np.clip((t_sky - 0.34) / 0.66, 0.0, 1.0)
    bg = (
        c_zenith[None, None, :] * (1.0 - t1[..., None])
        + c_midsky[None, None, :] * (t1 * (1.0 - t2))[..., None]
        + c_horiz[None, None, :] * (t1 * t2)[..., None]
    )

    # Radiant upper-right golden morning sun bloom aligned strictly with -SUN_DIR
    du_s = (u - 0.87) * 1.15
    dv_s = (v - 0.13) * 1.55
    sun_halo = np.exp(-(du_s * du_s * 3.0 + dv_s * dv_s * 4.2)).astype(np.float32)
    sun_core = np.exp(-(du_s * du_s * 24.0 + dv_s * dv_s * 32.0)).astype(np.float32)
    bg += sun_halo[..., None] * np.array([0.24, 0.21, 0.09], dtype=np.float32)
    bg += sun_core[..., None] * np.array([0.32, 0.28, 0.12], dtype=np.float32)

    # Sculptural Ghibli morning cumulus/cirrus cloud streaks sweeping softly across the highland sky
    cloud_env = np.clip((hv - 0.05 - v) / 0.18, 0.0, 1.0) * np.clip((v - 0.02) / 0.12, 0.0, 1.0)
    shear = u * 1.0 + 0.38 * v
    cl_f = (
        0.52 * np.sin(shear * 7.8 + 0.6)
        + 0.34 * np.cos(shear * 14.5 - v * 11.0)
        + 0.56 * mottle
        + 0.28 * mottle_f
    )
    cl_mask = np.clip((cl_f - 0.16) * 1.25, 0.0, 1.0) * cloud_env * 0.80
    # Sculpted lit cloud tops vs cool sage-blue undersides
    cl_top = np.clip((cl_f - 0.32) * 1.4 + 0.35 * (u - 0.45) - 0.45 * (v - 0.16), 0.0, 1.0)
    cl_col = (
        np.array([0.64, 0.76, 0.75], dtype=np.float32)[None, None, :] * (1.0 - cl_top[..., None])
        + np.array([1.00, 0.99, 0.91], dtype=np.float32)[None, None, :] * cl_top[..., None]
    )
    bg = bg * (1.0 - cl_mask[..., None]) + cl_col * cl_mask[..., None]

    # 4 receding misty blue-jade mountain ridges with atmospheric ridge-line shading & unkai mist
    u1d = np.linspace(0.0, 1.0, W, dtype=np.float32)
    mountain_specs = [
        # (base_v, a1, f1, ph1, a2, f2, c_ridge, c_base_mist, mist_str)
        (hv - 0.115, 0.064, 4.6, 0.2, 0.025, 11.8, [0.47, 0.65, 0.63], [0.85, 0.91, 0.84], 0.84),
        (hv - 0.072, 0.056, 4.1, 2.0, 0.022, 10.5, [0.33, 0.52, 0.50], [0.82, 0.89, 0.81], 0.80),
        (hv - 0.024, 0.048, 3.6, 3.9, 0.018, 9.2,  [0.21, 0.38, 0.35], [0.78, 0.87, 0.78], 0.76),
        (hv + 0.022, 0.042, 4.3, 5.4, 0.016, 12.6, [0.14, 0.29, 0.24], [0.75, 0.85, 0.75], 0.72),
    ]
    for idx_m, (bv, a1, f1, ph1, a2, f2, c_top, c_bot, m_str) in enumerate(mountain_specs):
        saddle = 0.052 * np.exp(-((u1d - 0.61) / 0.24) ** 2) * (1.0 - 0.14 * idx_m)
        ridge_v = (
            bv
            + saddle
            - a1 * np.sin(u1d * f1 + ph1)
            - a2 * np.cos(u1d * f2 - ph1 * 0.7)
            + 0.008 * np.sin(u1d * 27.0 + idx_m * 1.9)
            + 0.004 * np.cos(u1d * 54.0 + idx_m)
        )
        dist_below = v - ridge_v[None, :]
        m_mask = np.clip(dist_below / 0.0030, 0.0, 1.0)
        depth_t = np.clip(dist_below / 0.130, 0.0, 1.0)

        slope_du = np.gradient(ridge_v)
        sun_face = np.clip(0.45 - slope_du * 135.0, 0.0, 1.0)[None, :]
        ct = np.array(c_top, dtype=np.float32)
        cb = np.array(c_bot, dtype=np.float32)
        m_col = ct[None, None, :] * (1.0 - depth_t[..., None] * m_str) + cb[None, None, :] * (depth_t[..., None] * m_str)
        m_col += (sun_face * (1.0 - depth_t * 0.72))[..., None] * np.array([0.09, 0.11, 0.055], dtype=np.float32) * (1.0 - 0.15 * idx_m)
        m_col += (mottle * 0.020 + dabs * 0.014)[..., None] * (1.0 - 0.14 * idx_m)
        bg = bg * (1.0 - m_mask[..., None]) + m_col * m_mask[..., None]

        # Silky silver-mint sea-of-clouds mist ribbons winding between the mountain ranges
        if idx_m >= 1:
            mist_y = bv + 0.062 + 0.018 * np.sin(u * 5.0 + idx_m * 1.5) + 0.018 * mottle
            mist_w = 0.032 + 0.010 * idx_m
            veil = np.exp(-((v - mist_y) / mist_w) ** 2) * (0.50 + 0.30 * np.clip(mottle, -0.8, 1.0))
            veil *= np.clip((u - 0.28) / 0.24, 0.30, 1.0)
            mist_col = np.array([0.88, 0.94, 0.86], dtype=np.float32) + sun_halo[..., None] * np.array([0.08, 0.06, 0.02], dtype=np.float32)
            bg = bg * (1.0 - veil[..., None] * 0.76) + mist_col * (veil[..., None] * 0.76)

    # Radiant billowing silver-mint sea-of-clouds pooling in the central valley basin
    sea_v = hv + 0.072 + 0.020 * np.sin(u * 4.2 - 0.5) + 0.025 * mottle + 0.012 * dabs
    sea_mask = np.exp(-((v - sea_v) / 0.064) ** 2) * np.clip((u - 0.34) / 0.22, 0.0, 1.0)
    sea_mask *= (0.68 + 0.28 * np.clip(mottle + 0.4 * mottle_f, -1.0, 1.0))
    sea_col = np.array([0.90, 0.95, 0.87], dtype=np.float32) + np.clip(u - 0.42, 0.0, 1.0)[..., None] * np.array([0.06, 0.04, 0.01], dtype=np.float32)
    bg = bg * (1.0 - sea_mask[..., None] * 0.84) + sea_col * (sea_mask[..., None] * 0.84)

    bg += (mottle * 0.014 + dabs * 0.011 + tooth * 0.005)[..., None]
    fog_rgb = np.array([0.78, 0.86, 0.78], dtype=np.float32)
    return np.clip(bg, 0.0, 1.32), fog_rgb, mottle, dabs


# =============================================================================
# Stage 3 Band 1 — Volumetric Gouache Terraced Uji Tea Hills & Organic Cypresses
# =============================================================================

def draw_painterly_cypress_tree(
    ctx: cairo.Context,
    cx: float,
    base_y: float,
    height_px: float,
    width_px: float,
    srng: np.random.Generator,
    mist_t: float,
    fog_rgb: np.ndarray,
) -> None:
    """
    Monumental true-scale Studio Ghibli Japanese cedar / sugi (Cryptomeria japonica, 20-28m tall):
      - Tall tapering weathered cedar trunk with sunlit bark edge & subtle organic lean
      - Sweeping irregular drooping conifer bough shelves with open mist-filled sky gaps &
        layered multi-tuft needle plumes so towering 500-850px foreground trees look hand-painted!
      - Grounded cast shadow & mossy hillside scrub anchoring the trunk seamlessly into terrain.
    """
    c_trunk = np.array([0.145, 0.112, 0.088], dtype=np.float64) * (1.0 - mist_t) + fog_rgb * mist_t
    c_trunk_lit = np.array([0.275, 0.215, 0.155], dtype=np.float64) * (1.0 - mist_t) + fog_rgb * mist_t
    c_deep = np.array([0.042, 0.105, 0.075], dtype=np.float64) * (1.0 - mist_t) + fog_rgb * mist_t
    c_sh = np.array([0.068, 0.158, 0.102], dtype=np.float64) * (1.0 - mist_t) + fog_rgb * mist_t
    c_mid = np.array([0.155, 0.295, 0.168], dtype=np.float64) * (1.0 - mist_t) + fog_rgb * mist_t
    c_lit = np.array([0.415, 0.585, 0.275], dtype=np.float64) * (1.0 - mist_t) + fog_rgb * mist_t
    c_rim = np.array([0.655, 0.768, 0.385], dtype=np.float64) * (1.0 - mist_t) + fog_rgb * mist_t

    lean_dir = float(srng.uniform(-0.025, 0.025))
    lean_ph = float(srng.uniform(0.0, math.tau))

    def spine_x(frac: float) -> float:
        return cx + lean_dir * height_px * frac + math.sin(frac * 2.6 + lean_ph) * width_px * 0.032

    # Soft grounded cast shadow pool anchoring tree organically onto hillside (toward lower-left!)
    ctx.save()
    ctx.translate(cx - width_px * 0.30, base_y + 3.0)
    ctx.scale(width_px * 0.90, max(4.0, width_px * 0.17))
    g_sh = cairo.RadialGradient(0.0, 0.0, 0.1, 0.0, 0.0, 1.0)
    g_sh.add_color_stop_rgba(0.0, *c_deep, 0.54 * (1.0 - 0.65 * mist_t))
    g_sh.add_color_stop_rgba(1.0, *c_deep, 0.0)
    ctx.set_source(g_sh)
    ctx.arc(0.0, 0.0, 1.0, 0.0, math.tau)
    ctx.fill()
    ctx.restore()

    # Tall tapering straight Japanese sugi trunk with sunlit right bark edge!
    tw = max(2.2, width_px * 0.095)
    top_x = spine_x(0.84)
    ctx.set_source_rgba(*c_trunk, 0.98)
    ctx.move_to(cx - tw * 0.88, base_y + 6.0)
    ctx.line_to(top_x - tw * 0.18, base_y - height_px * 0.84)
    ctx.line_to(top_x + tw * 0.18, base_y - height_px * 0.84)
    ctx.line_to(cx + tw * 0.92, base_y + 6.0)
    ctx.close_path()
    ctx.fill()

    if width_px > 55.0:
        ctx.set_source_rgba(*c_trunk_lit, 0.72)
        ctx.move_to(cx + tw * 0.18, base_y + 4.0)
        ctx.line_to(top_x + tw * 0.04, base_y - height_px * 0.78)
        ctx.line_to(top_x + tw * 0.18, base_y - height_px * 0.78)
        ctx.line_to(cx + tw * 0.86, base_y + 4.0)
        ctx.close_path()
        ctx.fill()

    def paint_feathered_bough_cluster(bx: float, by: float, rx: float, ry: float, col: np.ndarray, alpha: float) -> None:
        n_v = 18
        pts = []
        for k in range(n_v):
            a = (k / n_v) * math.tau
            jag = float(srng.uniform(0.70, 1.24)) * (1.16 if (k & 1) == 0 else 0.76)
            droop = abs(math.cos(a)) * ry * 0.24
            pts.append((bx + math.cos(a) * rx * jag, by + math.sin(a) * ry * jag + droop))
        ctx.set_source_rgba(*col, alpha)
        ctx.move_to(*pts[0])
        for k in range(n_v):
            p_cur = pts[k]
            p_nxt = pts[(k + 1) % n_v]
            quad_to(ctx, p_cur[0], p_cur[1], 0.5 * (p_cur[0] + p_nxt[0]), 0.5 * (p_cur[1] + p_nxt[1]))
        ctx.close_path()
        ctx.fill()

    # Scale tier count with tree height so monumental 550-850px cedars have crisp detailed branching!
    n_tiers = int(min(22, max(12, round(height_px / 36.0))))
    for ti in range(n_tiers):
        frac = ti / max(1, n_tiers - 1)
        ty = base_y - height_px * (0.14 + 0.85 * frac)
        tx_c = spine_x(0.14 + 0.85 * frac)
        # Authentic Japanese sugi profile: graceful wide base tapering steadily to a feathery crown spire
        t_env = math.pow(max(0.0, 1.0 - frac * 0.90), 0.78) * (0.82 + 0.18 * math.sin(frac * math.pi * 1.6 + lean_ph))
        tw_tier = width_px * (0.76 * t_env + 0.08) * float(srng.uniform(0.84, 1.18))
        th_tier = height_px * (0.068 - 0.022 * frac) * (1.12 if n_tiers <= 14 else 0.88)

        # Occasional visible dark cedar branch limb peeking under major boughs on large foreground trees
        if width_px > 95.0 and frac < 0.78 and (ti % 2 == 0):
            ctx.set_source_rgba(*c_trunk, 0.82)
            ctx.set_line_width(max(1.8, width_px * 0.024 * (1.0 - 0.65 * frac)))
            for s_b in (-1.0, 1.0):
                ctx.move_to(tx_c, ty + th_tier * 0.15)
                ctx.line_to(tx_c + s_b * tw_tier * 0.52, ty + th_tier * 0.42)
                ctx.stroke()

        # Layered asymmetric bough pads (deep under-shadow -> cool shade -> rich cedar -> golden rim!)
        paint_feathered_bough_cluster(tx_c - tw_tier * 0.34, ty + th_tier * 0.24, tw_tier * 0.56, th_tier * 0.84, c_deep, 0.92)
        paint_feathered_bough_cluster(tx_c - tw_tier * 0.10, ty + th_tier * 0.14, tw_tier * 0.58, th_tier * 0.86, c_sh, 0.92)
        paint_feathered_bough_cluster(tx_c + tw_tier * 0.12, ty + th_tier * 0.02, tw_tier * 0.54, th_tier * 0.78, c_mid, 0.91)
        paint_feathered_bough_cluster(tx_c + tw_tier * 0.34, ty - th_tier * 0.06, tw_tier * 0.40, th_tier * 0.62, c_lit, 0.86)

        # Extra outer drooping bough tufts on tall trees so silhouette reads as painterly conifer boughs!
        if width_px > 75.0 and frac < 0.86:
            paint_feathered_bough_cluster(
                tx_c - tw_tier * float(srng.uniform(0.54, 0.74)),
                ty + th_tier * float(srng.uniform(0.18, 0.42)),
                tw_tier * 0.34,
                th_tier * 0.54,
                c_sh,
                0.88,
            )
            paint_feathered_bough_cluster(
                tx_c + tw_tier * float(srng.uniform(0.50, 0.70)),
                ty + th_tier * float(srng.uniform(0.06, 0.30)),
                tw_tier * 0.32,
                th_tier * 0.50,
                c_lit if frac > 0.18 else c_mid,
                0.86,
            )

        if frac > 0.18:
            paint_feathered_bough_cluster(tx_c + tw_tier * 0.38, ty - th_tier * 0.18, tw_tier * 0.24, th_tier * 0.42, c_rim, 0.74)

        # Soft painterly drooping outer needle-tuft tips (zero stiff straight-line pins!)
        if width_px > 42.0 and frac < 0.90:
            for side in (-1.0, 1.0):
                sx = tx_c + side * tw_tier * float(srng.uniform(0.56, 0.82))
                sy = ty + float(srng.uniform(-th_tier * 0.10, th_tier * 0.34))
                c_sp = c_lit if side > 0 else c_sh
                paint_feathered_bough_cluster(
                    sx, sy,
                    max(4.0, tw_tier * float(srng.uniform(0.14, 0.24))),
                    max(3.0, th_tier * float(srng.uniform(0.26, 0.42))),
                    c_sp,
                    0.82,
                )

    # Tapered wispy sugi leader spire at crown apex
    apex_x = spine_x(0.99)
    apex_y = base_y - height_px
    paint_feathered_bough_cluster(apex_x, apex_y + height_px * 0.025, max(5.0, width_px * 0.09), max(8.0, height_px * 0.045), c_mid, 0.90)
    paint_feathered_bough_cluster(apex_x + width_px * 0.03, apex_y + height_px * 0.018, max(3.5, width_px * 0.055), max(6.0, height_px * 0.032), c_rim, 0.80)

    # Soft leafy hillside underbrush tucking around the trunk base so tree feels rooted in soil!
    for bk in range(3):
        bx = cx + (bk - 1) * width_px * 0.22
        paint_feathered_bough_cluster(bx, base_y - height_px * 0.025, width_px * 0.24, max(5.0, height_px * 0.042), c_mid if bk > 0 else c_sh, 0.88)


def draw_distant_tea_shed(ctx: cairo.Context, cx: float, cy: float, sc: float, mist_t: float, fog_rgb: np.ndarray) -> None:
    """Finely crafted miniature weathered wooden harvest tea-shed (chagoya) nestled into tea terraces."""
    def fm(c):
        v = np.array(c, dtype=np.float64) * (1.0 - mist_t) + fog_rgb * mist_t
        return (float(v[0]), float(v[1]), float(v[2]))

    bw, bh = 62.0 * sc, 30.0 * sc
    # Shaded left cedar wall & sunlit right cedar wall with horizontal clapboard lines
    ctx.set_source_rgba(*fm([0.20, 0.17, 0.14]), 0.98)
    ctx.rectangle(cx - bw * 0.52, cy - bh, bw * 0.48, bh)
    ctx.fill()
    ctx.set_source_rgba(*fm([0.42, 0.35, 0.25]), 0.98)
    ctx.rectangle(cx - bw * 0.04, cy - bh, bw * 0.56, bh)
    ctx.fill()
    # Delicate cedar plank seams & shadowed under-eave band
    ctx.set_source_rgba(*fm([0.13, 0.12, 0.09]), 0.55)
    ctx.rectangle(cx - bw * 0.52, cy - bh, bw * 1.04, bh * 0.18)
    ctx.fill()
    ctx.set_line_width(0.8 * sc)
    for k in range(1, 5):
        ly = cy - bh * (k / 5.0)
        ctx.move_to(cx - bw * 0.52, ly)
        ctx.line_to(cx + bw * 0.52, ly)
        ctx.stroke()

    # Tiny warm glowing wooden-framed drying-room window + dark doorway
    ctx.set_source_rgba(*fm([0.14, 0.11, 0.08]), 0.92)
    ctx.rectangle(cx - bw * 0.36, cy - bh * 0.68, bw * 0.18, bh * 0.68)
    ctx.fill()
    ctx.set_source_rgba(*fm([0.94, 0.79, 0.42]), 0.92)
    ctx.rectangle(cx + bw * 0.16, cy - bh * 0.62, bw * 0.17, bh * 0.32)
    ctx.fill()

    # Pitched overhanging mossy cedar shingle roof with dark ridge cap
    ctx.set_source_rgba(*fm([0.22, 0.23, 0.18]), 0.99)
    ctx.move_to(cx - bw * 0.72, cy - bh * 0.90)
    ctx.line_to(cx - bw * 0.05, cy - bh * 1.78)
    ctx.line_to(cx + bw * 0.68, cy - bh * 0.88)
    ctx.line_to(cx + bw * 0.56, cy - bh * 0.84)
    ctx.line_to(cx - bw * 0.60, cy - bh * 0.85)
    ctx.close_path()
    ctx.fill()
    ctx.set_source_rgba(*fm([0.52, 0.49, 0.34]), 0.96)
    ctx.move_to(cx - bw * 0.05, cy - bh * 1.78)
    ctx.line_to(cx + bw * 0.68, cy - bh * 0.88)
    ctx.line_to(cx + bw * 0.03, cy - bh * 0.89)
    ctx.close_path()
    ctx.fill()
    # Ground shed naturally behind a rounded scalloped row of foreground tea-bush foliage
    for k in range(8):
        bx = cx - bw * 0.62 + (k / 7.0) * bw * 1.24
        ctx.set_source_rgba(*fm([0.28 + 0.06 * (k & 1), 0.46 + 0.05 * (k & 1), 0.22]), 0.96)
        ctx.arc(bx, cy + 1.5 * sc, (6.5 + 2.0 * (k & 1)) * sc, 0, math.tau)
        ctx.fill()


def paint_terraced_tea_valley_band(
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    mottle: np.ndarray,
    dabs: np.ndarray,
) -> np.ndarray:
    """
    Stage 3 Band 1: Volumetric Studio Ghibli *Only Yesterday* sculpted Uji terraced green tea
    plantations (chabatake) rendered directly as continuous shaded 3D-contour rounded tea-bush
    hedgerows with lush velvety emerald-to-matcha-lime crowns, soft organic leaf-stipple foliage
    micro-texture (ZERO black geometric lined-paper stripes!), diagonal sunlit/cloud-shadowed
    valley light bands, winding warm-ochre harvest paths, distant wooden tea shed, billowing
    between-hill silver-mint mist ribbons, and feathered soft-crowned cedar/cypress trees!
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = xx / W
    v = yy / H

    # Fine isotropic foliage stipple fields so every tea bush reads as velvety hand-painted leaves at 1:1!
    leaf_stip1 = fbm_gouache(H, W, rng, scales=((20, 20), (8.5, 8.5), (4.0, 4.0), (2.2, 2.2)))
    leaf_stip2 = fbm_gouache(H, W, rng, scales=((11, 11), (5.0, 5.0), (2.6, 2.6), (1.6, 1.6)))
    cloud_sh_field = fbm_gouache(H, W, rng, scales=((220, 220), (110, 110), (50, 50), (22, 22)))

    # Accumulate 5 sweeping sculpted tea hills strictly back-to-front in premultiplied float32 RGBA
    out_rgb = np.zeros((H, W, 3), dtype=np.float32)
    out_a = np.zeros((H, W, 1), dtype=np.float32)

    # Diagonal morning sunbeam / soft cloud shadow sweep across the terraced valley (aligned with SUN_DIR!)
    sun_sweep = np.clip(0.56 + 0.44 * np.sin((u * 1.15 + v * 0.72) * 8.5 + cloud_sh_field * 0.85), 0.16, 1.0)

    hill_specs = [
        # (x0_f, x1_f, crest_v, tilt_v, bow_amp, row_freq, row_exp, mist_t, sun_bias, seed)
        # True-to-scale waist-high (~0.9m) Uji terraced tea rows matching the 4.2m cedar watermill & 16m sugi cedars!
        (0.32, 0.82, 0.486,  0.032, 0.028, 94.0, 0.90, 0.56, 0.68, 111),  # Far valley left-center spur
        (0.54, 1.04, 0.464, -0.042, 0.034, 82.0, 0.88, 0.44, 0.86, 112),  # Far right sunlit Uji ridge
        (0.36, 1.04, 0.536,  0.020, 0.038, 66.0, 0.84, 0.28, 0.78, 113),  # Mid-valley broad sweeping slope
        (0.46, 1.05, 0.626, -0.056, 0.042, 48.0, 0.80, 0.14, 0.90, 114),  # Near right sculpted slope
        (0.56, 1.05, 0.748, -0.068, 0.044, 32.0, 0.76, 0.04, 0.94, 115),  # Foreground-right lush tea terraces
    ]

    # Store ridge crest coordinates for placing organic cypress trees & distant tea shed accurately!
    hill_crests_v: list[np.ndarray] = []

    for h_idx, (x0_f, x1_f, cv_0, tilt, bow, r_freq, r_exp, mist_t, s_bias, hseed) in enumerate(hill_specs):
        hrng = np.random.default_rng(hseed)
        u_rel = np.clip((u - x0_f) / max(0.01, x1_f - x0_f), 0.0, 1.0)
        u1 = np.linspace(0.0, 1.0, W, dtype=np.float32)
        ur1 = np.clip((u1 - x0_f) / max(0.01, x1_f - x0_f), 0.0, 1.0)
        ph = float(hrng.uniform(0.0, math.tau))

        # Broad gentle natural mountain-valley contour crest (smooth sweeping hills, zero bouncy sine waves!)
        crest_1d = (
            cv_0
            + tilt * (ur1 - 0.5)
            - bow * np.sin(ur1 * math.pi)
            + (0.007 + 0.002 * h_idx) * np.cos(ur1 * math.pi * 1.65 + ph)
        )
        hill_crests_v.append(crest_1d)
        crest_2d = crest_1d[None, :] + (0.0018 + 0.0009 * h_idx) * leaf_stip1

        # Distance down the sculpted hillside below its crest
        dy_hill = v - crest_2d
        x_mask = np.clip((u - x0_f) / 0.035, 0.0, 1.0) * np.clip((x1_f - u) / 0.035, 0.0, 1.0)
        h_mask = np.clip(dy_hill / 0.0028, 0.0, 1.0) * x_mask

        # Gentle broad contour-following phase for natural manicured tea rows (NO wavy corrugated sine ripples!)
        dy_pos = np.maximum(0.0, dy_hill)
        contour_warp = (
            (0.006 + 0.003 * h_idx) * np.sin(u_rel * math.pi * 0.95 + dy_pos * 2.2)
            + 0.004 * mottle
        )
        row_coord = np.power(dy_pos + 0.008, r_exp) * r_freq + contour_warp * r_freq * 0.25
        # Soft foliage-edge perturbation so rows feel organic & hand-painted
        row_coord_j = row_coord + (0.085 + 0.025 * h_idx) * leaf_stip1 + 0.048 * leaf_stip2

        # Phase inside each rounded tea-bush hedge row: 0 -> furrow trough, 0.5 -> rounded bush crown!
        phase = np.mod(row_coord_j, 1.0)
        row_id = np.floor(row_coord_j).astype(np.int32)
        row_var = np.sin(row_id.astype(np.float32) * 12.9898 + hseed) * 0.5

        bush_profile = np.clip(np.sin(np.clip((phase - 0.08) / 0.90, 0.0, 1.0) * math.pi), 0.0, 1.0)
        bush_dome = np.power(bush_profile, 0.54)
        crown_facing = np.exp(-((phase - 0.36) / 0.26) ** 2)
        furrow_ao = np.exp(-((np.minimum(phase, 1.0 - phase)) / 0.080) ** 2)

        # Hillside macro slope illumination: right-facing slopes glow golden-matcha, left flanks deeper emerald!
        slope_du = np.gradient(crest_1d)[None, :]
        hill_sun = np.clip(s_bias - slope_du * 1.65 + 0.20 * (u_rel - 0.4), 0.22, 1.0) * (0.48 + 0.52 * sun_sweep)
        lit = np.clip(crown_facing * (0.52 + 0.62 * hill_sun) + 0.14 * leaf_stip2, 0.0, 1.12)

        # Rich Matcha / Uji green tea palette (deep emerald shadows -> velvety matcha -> sunlit golden-lime!)
        c_furrow = np.array([0.095, 0.205, 0.132], dtype=np.float32)
        c_emerald = np.array([0.168, 0.340, 0.195], dtype=np.float32)
        c_matcha = np.array([0.320, 0.510, 0.245], dtype=np.float32)
        c_crown = np.array([0.555, 0.715, 0.330], dtype=np.float32)
        c_gold = np.array([0.755, 0.838, 0.440], dtype=np.float32)

        col = (
            c_furrow[None, None, :] * (1.0 - bush_dome[..., None])
            + c_emerald[None, None, :] * (bush_dome * (1.0 - 0.68 * hill_sun))[..., None]
            + c_matcha[None, None, :] * (bush_dome * 0.68 * hill_sun)[..., None]
        )
        col = col * (1.0 - 0.52 * np.clip(lit, 0.0, 1.0)[..., None]) + c_crown[None, None, :] * (0.52 * np.clip(lit, 0.0, 1.0)[..., None])
        gold_k = np.clip((lit - 0.58) * 1.8 * (0.65 + 0.45 * leaf_stip2), 0.0, 1.0)
        col = col * (1.0 - 0.46 * gold_k[..., None]) + c_gold[None, None, :] * (0.46 * gold_k[..., None])

        # Distance-attenuated velvet furrow shadow (gentle on far/mid slopes -> velvety tea hills, crisp in foreground!)
        furrow_str = 0.17 + 0.048 * h_idx
        col *= (1.0 - furrow_str * furrow_ao)[..., None]

        # Fine gouache foliage leaf-stipple + row variation + angled flat-brush dabs!
        stip_mod = (leaf_stip2 * 0.034 + leaf_stip1 * 0.024 + dabs * 0.015 + row_var * 0.016)[..., None]
        col += stip_mod * np.array([0.65, 1.0, 0.45], dtype=np.float32)

        # Winding narrow crisp warm-ochre earthen harvest footpath with downhill edge shadow
        if h_idx in (2, 3):
            u_path_c = (0.68 if h_idx == 2 else 0.79) + 0.022 * np.sin(dy_pos * 18.0 + h_idx * 1.6) - dy_pos * (0.14 if h_idx == 2 else -0.17)
            pw = 0.0034 + 0.0026 * dy_pos + 0.0012 * (h_idx - 2)
            du_p = (u - u_path_c) / pw
            p_m = np.clip(1.0 - np.abs(du_p), 0.0, 1.0)
            p_m = p_m * p_m * (3.0 - 2.0 * p_m) * np.clip((dy_pos - 0.012) / 0.022, 0.0, 1.0)
            p_edge = np.exp(-((du_p + 0.85) / 0.45) ** 2) * np.clip((dy_pos - 0.012) / 0.022, 0.0, 1.0)
            p_col = np.array([0.58, 0.51, 0.36], dtype=np.float32) + (leaf_stip2 * 0.028)[..., None]
            col *= (1.0 - 0.30 * p_edge)[..., None]
            col = col * (1.0 - 0.84 * p_m[..., None]) + p_col * (0.84 * p_m[..., None])

        # Atmospheric distance mist + luminous silver-mint unkai valley mist hugging hill bases!
        hill_mist = np.clip(mist_t + 0.16 * np.exp(-dy_pos / 0.048) * (0.55 + 0.45 * mottle), 0.0, 0.86)
        col = col * (1.0 - hill_mist[..., None]) + fog_rgb[None, None, :] * hill_mist[..., None]
        col = apply_shadow_bounce(col)

        hm = h_mask[..., None].astype(np.float32)
        out_rgb = np.clip(col, 0.0, 1.25) * hm + out_rgb * (1.0 - hm)
        out_a = hm + out_a * (1.0 - hm)

        if h_idx in (0, 1, 2):
            m_band_v = cv_0 + 0.082 + 0.016 * np.sin(u * 4.5 + h_idx) + 0.022 * mottle
            m_veil = np.exp(-((v - m_band_v) / (0.044 + 0.009 * h_idx)) ** 2) * x_mask * (0.48 - 0.08 * h_idx) * np.clip(0.68 + 0.45 * mottle, 0.0, 1.25)
            mv = m_veil[..., None].astype(np.float32)
            veil_col = np.array([0.90, 0.95, 0.87], dtype=np.float32)[None, None, :]
            out_rgb = veil_col * mv + out_rgb * (1.0 - mv)
            out_a = mv + out_a * (1.0 - mv)

    # Crisp 2x PyCairo MONUMENTAL true-scale Japanese cedar groves (sugi = 22-28m tall, 5-7x the 4.2m watermill!)
    surf, ctx = new_cairo_2x()
    trng = np.random.default_rng(199)

    def crest_px(h_i: int, u_val: float) -> float:
        idx = min(W - 1, max(0, int(u_val * (W - 1))))
        return float(hill_crests_v[h_i][idx]) * H

    # Far-right Uji ridge clustered misty sugi groves (leaves u=0.50..0.68 OPEN for luminous misty mountain valley vista!)
    for uf, hpx, wpx, mt in [
        (0.695, 225, 74, 0.50),
        (0.720, 275, 88, 0.48),
        (0.812, 240, 78, 0.45),
        (0.842, 305, 96, 0.42),
        (0.870, 255, 82, 0.43),
    ]:
        draw_painterly_cypress_tree(ctx, uf * W, crest_px(1, uf) + 18.0, hpx, wpx, trng, mt, fog_rgb)

    # Mid-valley weathered wooden tea-drying shed (chagoya, ~38px tall) dwarfed 8-10x by clustered 26m sugi cedars!
    shed_u = 0.746
    shed_y = crest_px(2, shed_u) + 34.0
    draw_painterly_cypress_tree(ctx, shed_u * W - 62.0, shed_y + 6.0, 365, 114, trng, 0.30, fog_rgb)
    draw_painterly_cypress_tree(ctx, shed_u * W - 26.0, shed_y + 2.0, 290, 92, trng, 0.31, fog_rgb)
    draw_painterly_cypress_tree(ctx, shed_u * W + 56.0, shed_y + 8.0, 330, 102, trng, 0.30, fog_rgb)
    draw_distant_tea_shed(ctx, shed_u * W, shed_y, sc=0.72, mist_t=0.25, fog_rgb=fog_rgb)

    # Receding mid-right cedar stepping smoothly into the monumental foreground-right framing grove
    draw_painterly_cypress_tree(ctx, 0.856 * W, crest_px(2, 0.856) + 46.0, 425, 132, trng, 0.24, fog_rgb)

    # Near-right MONUMENTAL Japanese sugi cedars (true 22-26m height towering grandly into the morning sky!)
    draw_painterly_cypress_tree(ctx, 0.892 * W, crest_px(3, 0.892) + 66.0, 575, 168, trng, 0.16, fog_rgb)
    draw_painterly_cypress_tree(ctx, 0.938 * W, crest_px(3, 0.938) + 86.0, 745, 212, trng, 0.12, fog_rgb)
    draw_painterly_cypress_tree(ctx, 0.988 * W, crest_px(3, 0.988) + 98.0, 910, 255, trng, 0.08, fog_rgb)

    # Towering misty mountain cedar woodland rising majestically ABOVE and behind the left cedar watermill!
    # (Cleanly behind the roof, stopping at u=0.45 so the right roof corner opens cleanly onto the misty tea valley!)
    for uf, vf, hpx, wpx, mt in [
        (0.452, 0.550, 485, 148, 0.40),
        (0.395, 0.555, 575, 172, 0.35),
        (0.318, 0.560, 640, 192, 0.30),
        (0.215, 0.565, 695, 208, 0.26),
        (0.090, 0.570, 730, 218, 0.22),
    ]:
        draw_painterly_cypress_tree(ctx, uf * W, vf * H, hpx, wpx, trng, mt, fog_rgb)

    cairo_trees = finish_cairo_2x(surf)
    ta = cairo_trees[:, :, 3:4]
    out_rgb = cairo_trees[:, :, :3] + out_rgb * (1.0 - ta)
    out_a = ta + out_a * (1.0 - ta)

    return np.concatenate([np.clip(out_rgb, 0.0, 1.25), np.clip(out_a, 0.0, 1.0)], axis=-1)


# =============================================================================
# Stage 1 & 2 — Metric 3x3 Voronoi Stone Surfaces & Weathered Cedar Mill Planes
# =============================================================================

def compute_voronoi_stones(
    wx: np.ndarray,
    wy: np.ndarray,
    cell_x: float,
    cell_y: float,
    rng: np.random.Generator,
    warp_amp: float = 0.042,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical-meter 3x3 jittered Voronoi (d2 - d1) stone field with smooth bicubic perturbation."""
    th, tw = wx.shape
    nx = resize_bicubic(rng.standard_normal((max(4, th // 26), max(4, tw // 26))).astype(np.float32), tw, th) * warp_amp
    ny = resize_bicubic(rng.standard_normal((max(4, th // 26), max(4, tw // 26))).astype(np.float32), tw, th) * warp_amp
    px = wx + nx
    py = wy + ny

    ix = np.floor(px / cell_x).astype(np.int32)
    iy = np.floor(py / cell_y).astype(np.int32)
    jx_tab = rng.uniform(0.16, 0.84, size=256)
    jy_tab = rng.uniform(0.16, 0.84, size=256)

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

    return (d2 - d1).astype(np.float32), best_id, d1.astype(np.float32)


def paint_voronoi_footpath_uv(
    rng: np.random.Generator,
    x_min: float = -1.30,
    x_max: float = 3.55,
    y_min: float = 1.20,
    y_max: float = 26.2,
    mask_to_path: bool = True,
    tw: int = 1024,
    th: int = 1536,
) -> np.ndarray:
    """
    Stage 1 & 2: Curving damp mossy granite highland footpath (z = 0) with exact metric
    3x3 Voronoi (d2 - d1) flagstones, Sobel stone bevels, natural dark velvety moss-tucked
    seams (ZERO bright green or white cartoon webs!), sunlit morning dew, and 6-step -SUN_DIR shadows!
    """
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    wx = x_min + u * (x_max - x_min)
    wy = y_min + v * (y_max - y_min)
    wz = np.full_like(wx, Z_PATH)

    edge_dist, best_id, _ = compute_voronoi_stones(wx, wy, 0.44, 0.48, rng, warp_amp=0.042)
    tone_tab = rng.uniform(-0.082, 0.082, size=256)
    warm_tab = rng.uniform(-0.030, 0.036, size=256)
    moss_tab = rng.uniform(0.0, 1.0, size=256)

    mottle = fbm_gouache(th, tw, rng)
    mottle_f = fbm_gouache(th, tw, rng, scales=((24, 24), (10, 10), (4.8, 4.8), (2.4, 2.4)))
    dabs = flat_brush_dabs(th, tw, rng, count=92)
    tooth = paper_tooth(th, tw, rng)

    left_edge = millrace_right_x(wy)
    right_edge = path_right_x(wy)
    d_left = (wx - left_edge).astype(np.float32)
    d_right = (right_edge - wx).astype(np.float32)

    if mask_to_path:
        shoulder_moss = (
            np.clip((0.38 - d_left) / 0.38, 0.0, 1.0) * 0.65
            + np.clip((0.55 - d_right) / 0.55, 0.0, 1.0) * 0.85
        )
    else:
        shoulder_moss = np.full((th, tw), 0.34, dtype=np.float32)
    shoulder_moss = np.clip(shoulder_moss + mottle * 0.24 + dabs * 0.12, 0.0, 1.0)

    stone_h = np.clip((edge_dist - 0.016) / 0.034, 0.0, 1.0)
    stone_h = stone_h * stone_h * (3.0 - 2.0 * stone_h)
    is_moss_cell = (moss_tab[best_id] < (0.06 + 0.62 * shoulder_moss)).astype(np.float32)
    eff_stone = stone_h * (1.0 - 0.84 * is_moss_cell)

    gx = cv2.Sobel(eff_stone, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(eff_stone, cv2.CV_32F, 0, 1, ksize=3)
    bevel_lit = np.clip(-(gx * float(SUN_DIR[0]) + gy * float(SUN_DIR[1])) * 0.52, -0.35, 0.35)

    sv = tone_tab[best_id].astype(np.float32)
    wv = warm_tab[best_id].astype(np.float32)
    # Warm highland weathered grey-ochre granite
    stone = np.zeros((th, tw, 3), dtype=np.float32)
    stone[:, :, 0] = 0.275 + sv * 0.58 + wv * 0.95 + mottle * 0.032 + mottle_f * 0.020 + dabs * 0.022
    stone[:, :, 1] = 0.282 + sv * 0.54 + wv * 0.55 + mottle * 0.030 + mottle_f * 0.018 + dabs * 0.020
    stone[:, :, 2] = 0.242 + sv * 0.44 - wv * 0.45 + mottle * 0.024 + mottle_f * 0.014 + dabs * 0.016
    stone *= (1.0 + bevel_lit * 0.55)[..., None]

    # Deep natural dark-emerald velvety moss tucked inside stone crevices (ZERO bright double-rim halos!)
    moss_depth = np.clip((0.022 - edge_dist) / 0.022, 0.0, 1.0)
    moss = np.zeros((th, tw, 3), dtype=np.float32)
    moss[:, :, 0] = 0.068 + mottle * 0.022 + mottle_f * 0.016 + dabs * 0.016
    moss[:, :, 1] = 0.158 + mottle * 0.036 + mottle_f * 0.022 + dabs * 0.022
    moss[:, :, 2] = 0.062 + mottle * 0.014 + mottle_f * 0.010 + dabs * 0.010
    moss *= (1.0 - 0.56 * moss_depth)[..., None]

    # Peak shadow AO strictly at edge_dist == 0 so crevices stay dark & grounded!
    crevice_ao = np.exp(-((edge_dist / 0.020) ** 2)) * (1.0 - 0.52 * is_moss_cell)
    rgb = eff_stone[..., None] * stone + (1.0 - eff_stone[..., None]) * moss
    rgb *= (1.0 - 0.62 * crevice_ao)[..., None]

    if mask_to_path:
        # Rough-hewn granite coping curb bordering the crystal millrace edge
        curb_m = np.clip((d_left + 0.02) / 0.03, 0.0, 1.0) * np.clip((0.26 - d_left) / 0.04, 0.0, 1.0)
        curb_joint = np.exp(-((np.abs(np.mod(wy + 0.25, 0.72) - 0.36) - 0.36) / 0.018) ** 2)
        curb_col = np.zeros((th, tw, 3), dtype=np.float32)
        curb_col[:, :, 0] = 0.295 + mottle * 0.028 + dabs * 0.018
        curb_col[:, :, 1] = 0.305 + mottle * 0.028 + dabs * 0.018
        curb_col[:, :, 2] = 0.262 + mottle * 0.024 + dabs * 0.016
        curb_col *= (1.0 - 0.58 * curb_joint)[..., None]
        curb_moss = np.clip(0.20 + 0.36 * mottle + 0.22 * mottle_f, 0.0, 0.62)
        curb_mix = curb_col * (1.0 - curb_moss[..., None]) + moss * curb_moss[..., None]
        rgb = rgb * (1.0 - curb_m[..., None]) + curb_mix * curb_m[..., None]

    # Stage 2: 6-step -SUN_DIR morning sunlight raking from upper-right!
    sun = trace_valley_sunlight(wx, wy, wz)
    sun_stone = np.array([0.36, 0.35, 0.20], dtype=np.float32)
    sun_moss = np.array([0.24, 0.34, 0.11], dtype=np.float32)
    rgb += sun[..., None] * (eff_stone[..., None] * sun_stone + (1.0 - eff_stone[..., None]) * sun_moss) * (1.0 + 0.32 * bevel_lit[..., None])

    dew = eff_stone * np.clip(0.22 + 0.34 * mottle, 0.0, 0.68) * (1.0 - shoulder_moss * 0.65)
    rgb = rgb * (1.0 - 0.16 * dew[..., None]) + np.array([0.74, 0.82, 0.72], dtype=np.float32)[None, None, :] * (0.16 * dew[..., None])

    hearth = eval_hearth_lights(wx, wy, wz)
    rgb += hearth[..., None] * np.array([0.42, 0.32, 0.14], dtype=np.float32)

    rgb = apply_shadow_bounce(rgb)
    rgb += tooth[..., None] * 0.008

    if mask_to_path:
        alpha_l = np.clip((wx - (left_edge - 0.02)) / 0.045, 0.0, 1.0).astype(np.float32)
        # Organic scalloped mossy rim along right path shoulder & vanishing hillside crest
        edge_jit = 0.14 * mottle + 0.06 * mottle_f
        alpha_r = np.clip((right_edge + edge_jit - wx) / 0.18, 0.0, 1.0).astype(np.float32)
        alpha_f = np.clip((20.8 + 1.2 * mottle - wy) / 3.8, 0.0, 1.0).astype(np.float32)
        alpha = alpha_l * alpha_r * alpha_f
    else:
        # Left yard masked strictly behind millrace_left_x so it NEVER overlaps crystal stream water!
        alpha = np.clip((millrace_left_x(wy) + 0.02 - wx) / 0.06, 0.0, 1.0).astype(np.float32)
    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.32) * alpha[..., None]
    rgba[:, :, 3] = alpha
    return rgba


def paint_voronoi_stone_wall_uv(
    rng: np.random.Generator,
    len_m: float,
    h_m: float,
    facing_sun: float,
    wet_bottom: bool = True,
    tw: int = 896,
    th: int = 384,
) -> np.ndarray:
    """
    Metric jittered 3x3 Voronoi dry-stone retaining terrace wall (ishigaki) with rough-hewn
    granite boulders, crisp Sobel bevels, dark velvety moss-tucked seams (ZERO bright webs!),
    and natural stone course aspect ratios!
    """
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)  # v=0 top, v=1 bottom
    sx = u * len_m
    sy = v * h_m

    edge_dist, best_id, _ = compute_voronoi_stones(sx, sy, 0.34, 0.21, rng, warp_amp=0.024)
    tone_tab = rng.uniform(-0.090, 0.090, size=256)
    warm_tab = rng.uniform(-0.032, 0.036, size=256)

    mottle = fbm_gouache(th, tw, rng, scales=((42, 42), (16, 16), (7.5, 7.5), (3.4, 3.4)))
    dabs = flat_brush_dabs(th, tw, rng, count=88)
    tooth = paper_tooth(th, tw, rng)

    stone_h = np.clip((edge_dist - 0.011) / 0.025, 0.0, 1.0)
    stone_h = stone_h * stone_h * (3.0 - 2.0 * stone_h)

    gx = cv2.Sobel(stone_h, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(stone_h, cv2.CV_32F, 0, 1, ksize=3)
    bevel = np.clip((-gy * 0.68 - gx * 0.32), -0.44, 0.44)

    sv = tone_tab[best_id].astype(np.float32)
    wv = warm_tab[best_id].astype(np.float32)
    stone = np.zeros((th, tw, 3), dtype=np.float32)
    stone[:, :, 0] = 0.235 + sv * 0.62 + wv * 0.90 + mottle * 0.032 + dabs * 0.022
    stone[:, :, 1] = 0.242 + sv * 0.58 + wv * 0.48 + mottle * 0.030 + dabs * 0.020
    stone[:, :, 2] = 0.205 + sv * 0.48 - wv * 0.48 + mottle * 0.026 + dabs * 0.016
    stone *= (1.0 + bevel * 0.60)[..., None]

    # Deep natural dark-emerald moss tucked cleanly in mortar crevices & upper coping edge
    moss = np.zeros((th, tw, 3), dtype=np.float32)
    moss[:, :, 0] = 0.068 + mottle * 0.024 + dabs * 0.018
    moss[:, :, 1] = 0.175 + mottle * 0.042 + dabs * 0.026
    moss[:, :, 2] = 0.066 + mottle * 0.016 + dabs * 0.012

    top_moss = np.clip((0.26 - v.astype(np.float32)) / 0.26, 0.0, 1.0) * np.clip(0.50 + 0.50 * mottle, 0.0, 1.0)
    crev_moss = 1.0 - stone_h
    moss_mix = np.clip(crev_moss * 0.88 + top_moss * 0.62, 0.0, 1.0)

    rgb = stone * (1.0 - moss_mix[..., None]) + moss * moss_mix[..., None]
    crev_dark = np.exp(-((edge_dist / 0.015) ** 2))
    rgb *= (1.0 - 0.72 * crev_dark)[..., None]

    rgb *= (0.70 + 0.42 * facing_sun)
    rgb += (facing_sun * np.clip(bevel, 0.0, 1.0))[..., None] * np.array([0.32, 0.31, 0.17], dtype=np.float32)

    if wet_bottom:
        wet = np.clip((v.astype(np.float32) - 0.58) / 0.42, 0.0, 1.0)
        rgb *= (1.0 - 0.46 * wet)[..., None]

    rgb = apply_shadow_bounce(rgb)
    rgb += tooth[..., None] * 0.008

    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.25)
    rgba[:, :, 3] = 1.0
    return rgba


def paint_mill_right_wall_uv(rng: np.random.Generator, tw: int = 1024, th: int = 640) -> np.ndarray:
    """
    Stage 1 & 2: Exact planar right cedar clapboard wall (u_m = 0, facing +MILL_RGT toward sun & camera!)
    with weathered overlapping cedar planks, true un-stretched metric Voronoi granite foundation plinth,
    wet mossy splash staining behind the spoked waterwheel, atmospheric glowing amber shoji/lattice
    tea-drying room window (#f5d580) with subtle interior depth, and 6-step -SUN_DIR eave shadows!
    """
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    l_m = u * MILL_LENGTH
    z_m = MILL_Z_EAVE - v * (MILL_Z_EAVE - Z_MILL_BASE)
    wx = MILL_ORIGIN[0] + l_m * MILL_FWD[0]
    wy = MILL_ORIGIN[1] + l_m * MILL_FWD[1]

    mottle = fbm_gouache(th, tw, rng, scales=((48, 48), (18, 18), (8, 8), (3.5, 3.5)))
    dabs = flat_brush_dabs(th, tw, rng, count=90)
    tooth = paper_tooth(th, tw, rng)

    # Weathered silvered-brown cedar clapboard planks
    board_h = 0.185
    b_idx = np.floor((z_m - Z_MILL_BASE) / board_h).astype(np.int32)
    b_frac = np.mod((z_m - Z_MILL_BASE) / board_h, 1.0).astype(np.float32)
    b_tone = rng.uniform(-0.052, 0.052, size=64)[b_idx & 63].astype(np.float32)
    grain = resize_bicubic(rng.standard_normal((th // 2, max(8, tw // 28))).astype(np.float32), tw, th) * 0.018
    lap_sh = np.exp(-(b_frac / 0.11) ** 2) * 0.46
    lap_hi = np.exp(-((1.0 - b_frac) / 0.09) ** 2) * 0.14

    rgb = np.zeros((th, tw, 3), dtype=np.float32)
    rgb[:, :, 0] = 0.315 + b_tone + grain + mottle * 0.034 + dabs * 0.022
    rgb[:, :, 1] = 0.260 + b_tone * 0.90 + grain * 0.85 + mottle * 0.028 + dabs * 0.018
    rgb[:, :, 2] = 0.205 + b_tone * 0.78 + grain * 0.72 + mottle * 0.022 + dabs * 0.014
    rgb *= (1.0 - lap_sh + lap_hi)[..., None]

    # Vertical cedar corner posts & structural batten strips
    for pl_c in (0.12, 1.95, 3.65, 5.72, MILL_LENGTH - 0.12):
        pw = 0.11 if pl_c in (0.12, MILL_LENGTH - 0.12) else 0.06
        pm = np.clip((pw - np.abs(l_m - pl_c)) / 0.015, 0.0, 1.0).astype(np.float32)
        p_col = np.array([0.21, 0.17, 0.135], dtype=np.float32) + (mottle * 0.018)[..., None]
        rgb = rgb * (1.0 - pm[..., None]) + p_col * pm[..., None]

    # Properly metric-scaled Voronoi stone foundation plinth (z_m < 0.24m — ZERO stretching!)
    plinth_m = np.clip((0.24 - z_m) / 0.035, 0.0, 1.0).astype(np.float32)
    edge_p, id_p, _ = compute_voronoi_stones(l_m, z_m, 0.36, 0.22, rng, warp_amp=0.020)
    sh_p = np.clip((edge_p - 0.011) / 0.024, 0.0, 1.0)
    sh_p = sh_p * sh_p * (3.0 - 2.0 * sh_p)
    sv_p = rng.uniform(-0.07, 0.07, size=256)[id_p].astype(np.float32)
    pl_col = np.zeros((th, tw, 3), dtype=np.float32)
    pl_col[:, :, 0] = (0.21 + sv_p + mottle * 0.028) * sh_p + 0.065 * (1.0 - sh_p)
    pl_col[:, :, 1] = (0.23 + sv_p + mottle * 0.030) * sh_p + 0.165 * (1.0 - sh_p)
    pl_col[:, :, 2] = (0.195 + sv_p + mottle * 0.024) * sh_p + 0.065 * (1.0 - sh_p)
    pl_col *= (1.0 - 0.42 * np.clip((0.02 - z_m) / 0.36, 0.0, 1.0).astype(np.float32))[..., None]
    rgb = rgb * (1.0 - plinth_m[..., None]) + pl_col * plinth_m[..., None]

    # Wet dark cedar splash staining & green velvet moss creeping behind the turning waterwheel!
    d_sp = np.hypot((l_m - WHEEL_L_CENTER) / 1.55, (z_m - (WHEEL_Z_CENTER - 0.10)) / 1.38).astype(np.float32)
    wet_z = np.exp(-d_sp * d_sp * 1.12) * np.clip(0.75 + 0.35 * mottle, 0.0, 1.0)
    rgb *= (1.0 - 0.48 * wet_z)[..., None]
    wall_moss = wet_z * np.clip((0.92 - z_m.astype(np.float32)) / 1.15, 0.0, 1.0) * np.clip(0.45 + 0.55 * mottle, 0.0, 1.0)
    rgb = rgb * (1.0 - 0.60 * wall_moss[..., None]) + np.array([0.095, 0.215, 0.092], dtype=np.float32)[None, None, :] * (0.60 * wall_moss[..., None])

    # Stage 2: Exact 6-step -SUN_DIR overhanging pitched roof eave & spoked waterwheel shadow!
    sun = trace_valley_sunlight(wx + 0.03 * MILL_RGT[0], wy + 0.03 * MILL_RGT[1], z_m)
    rgb *= (0.56 + 0.44 * sun)[..., None]
    rgb += (sun * 0.25)[..., None] * np.array([0.88, 0.76, 0.46], dtype=np.float32)

    # Atmospheric warm glowing amber tea-drying pavilion shoji/lattice window (l_m in [3.90, 5.52], z in [1.02, 1.98])
    l0, l1 = 3.90, 5.52
    z0, z1 = 1.02, 1.98
    fr_m = ((l_m >= l0 - 0.11) & (l_m <= l1 + 0.11) & (z_m >= z0 - 0.12) & (z_m <= z1 + 0.10)).astype(np.float32)
    in_m = ((l_m >= l0) & (l_m <= l1) & (z_m >= z0) & (z_m <= z1)).astype(np.float32)
    rgb = rgb * (1.0 - fr_m[..., None]) + np.array([0.135, 0.105, 0.082], dtype=np.float32)[None, None, :] * fr_m[..., None]

    wu = np.clip((l_m - l0) / (l1 - l0), 0.0, 1.0).astype(np.float32)
    wv = np.clip((z_m - z0) / (z1 - z0), 0.0, 1.0).astype(np.float32)
    # Warm glowing hearth core lower-center + soft amber falloff + subtle interior tea-rack silhouette
    hearth_g = np.exp(-((wu - 0.48) ** 2 * 2.1 + (wv - 0.34) ** 2 * 2.4))
    rack_sil = np.clip(np.exp(-((wv - 0.62) / 0.05) ** 2) * 0.22 + np.exp(-((wu - 0.72) / 0.08) ** 2) * np.clip((0.68 - wv) * 4.0, 0.0, 1.0) * 0.18, 0.0, 0.32)
    gcol = np.zeros((th, tw, 3), dtype=np.float32)
    gcol[:, :, 0] = (0.74 + 0.46 * hearth_g + mottle * 0.032) * (1.0 - rack_sil)
    gcol[:, :, 1] = (0.50 + 0.44 * hearth_g + mottle * 0.025) * (1.0 - rack_sil * 1.15)
    gcol[:, :, 2] = (0.18 + 0.28 * hearth_g + mottle * 0.015) * (1.0 - rack_sil * 1.30)

    # Fine wooden shoji kumiko muntins softened slightly by glowing washi scatter
    m_v = (np.abs(np.mod(wu * 6.0 + 0.5, 1.0) - 0.5) < 0.048)
    m_h = (np.abs(np.mod(wv * 4.0 + 0.5, 1.0) - 0.5) < 0.058)
    m_c = (np.abs(wu - 0.5) < 0.026)
    munt = (m_v | m_h | m_c).astype(np.float32)
    wcol = gcol * (1.0 - munt[..., None] * 0.78) + np.array([0.19, 0.13, 0.08], dtype=np.float32)[None, None, :] * (munt[..., None] * 0.78)
    rgb = rgb * (1.0 - in_m[..., None]) + wcol * in_m[..., None]

    # Warm amber bounce light spilling around wooden window frame & eaves
    d_win = np.hypot(np.maximum(0.0, np.abs(l_m - 4.71) - 0.78), np.maximum(0.0, np.abs(z_m - 1.50) - 0.46)).astype(np.float32)
    rgb += (np.exp(-d_win * d_win * 3.5) * (1.0 - in_m))[..., None] * np.array([0.40, 0.26, 0.08], dtype=np.float32)

    rgb = apply_shadow_bounce(rgb)
    rgb += tooth[..., None] * 0.008

    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.38)
    rgba[:, :, 3] = 1.0
    return rgba


def paint_mill_front_gable_uv(rng: np.random.Generator, tw: int = 896, th: int = 896) -> np.ndarray:
    """
    Stage 1 & 2: Exact planar front wall & gable facade (l_m = 0), featuring aged cedar planks,
    warm weathered lime-plaster upper gable (shikkui) with exposed timber tie-beams, atmospheric
    glowing front shoji window with soft interior hearth depth, and ajar wooden sliding door!
    """
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    u_m = -MILL_WIDTH + u * MILL_WIDTH
    z_m = MILL_Z_RIDGE - v * (MILL_Z_RIDGE - Z_MILL_BASE)
    wx = MILL_ORIGIN[0] + u_m * MILL_RGT[0]
    wy = MILL_ORIGIN[1] + u_m * MILL_RGT[1]

    mottle = fbm_gouache(th, tw, rng, scales=((48, 48), (18, 18), (8, 8), (3.5, 3.5)))
    dabs = flat_brush_dabs(th, tw, rng, count=90)
    tooth = paper_tooth(th, tw, rng)

    u_ridge = -0.5 * MILL_WIDTH
    slope = (MILL_Z_RIDGE - MILL_Z_EAVE) / (0.0 - u_ridge)
    z_gable_max = MILL_Z_RIDGE - slope * np.abs(u_m - u_ridge) + 0.06
    in_gable = np.clip((z_gable_max - z_m) / 0.03, 0.0, 1.0).astype(np.float32)

    b_idx = np.floor((z_m - Z_MILL_BASE) / 0.185).astype(np.int32)
    b_frac = np.mod((z_m - Z_MILL_BASE) / 0.185, 1.0).astype(np.float32)
    b_tone = rng.uniform(-0.045, 0.045, size=64)[b_idx & 63].astype(np.float32)
    lap_sh = np.exp(-(b_frac / 0.11) ** 2) * 0.44

    wood = np.zeros((th, tw, 3), dtype=np.float32)
    wood[:, :, 0] = 0.235 + b_tone + mottle * 0.028 + dabs * 0.018
    wood[:, :, 1] = 0.195 + b_tone * 0.90 + mottle * 0.025 + dabs * 0.016
    wood[:, :, 2] = 0.155 + b_tone * 0.80 + mottle * 0.020 + dabs * 0.014
    wood *= (1.0 - lap_sh)[..., None]

    plaster_m = np.clip((z_m - 2.48) / 0.05, 0.0, 1.0).astype(np.float32)
    plaster = np.zeros((th, tw, 3), dtype=np.float32)
    plaster[:, :, 0] = 0.625 + mottle * 0.055 + dabs * 0.032
    plaster[:, :, 1] = 0.610 + mottle * 0.052 + dabs * 0.030
    plaster[:, :, 2] = 0.535 + mottle * 0.045 + dabs * 0.026
    rgb = wood * (1.0 - plaster_m[..., None]) + plaster * plaster_m[..., None]

    timber = (
        (np.abs(z_m - 2.50) < 0.085)
        | (np.abs(z_m - 3.30) < 0.065)
        | ((z_m > 2.48) & (np.abs(u_m - u_ridge) < 0.085))
        | ((z_m > 2.48) & (np.abs(u_m - (u_ridge + 0.92)) < 0.06))
        | ((z_m > 2.48) & (np.abs(u_m - (u_ridge - 0.92)) < 0.06))
        | (np.abs(u_m - (-0.10)) < 0.11)
        | (np.abs(u_m - (-MILL_WIDTH + 0.12)) < 0.11)
        | ((z_m < 2.50) & (np.abs(u_m - (-2.12)) < 0.085))
    ).astype(np.float32)
    t_col = np.array([0.165, 0.130, 0.100], dtype=np.float32) + (mottle * 0.018)[..., None]
    rgb = rgb * (1.0 - timber[..., None]) + t_col * timber[..., None]

    eave_dist = np.maximum(0.0, z_gable_max - z_m).astype(np.float32)
    eave_ao = np.exp(-eave_dist / 0.42) * 0.54
    sun = trace_valley_sunlight(wx - 0.04 * MILL_FWD[0], wy - 0.04 * MILL_FWD[1], z_m)
    rgb *= (0.64 + 0.28 * sun - eave_ao * 0.52)[..., None]
    rgb += (sun * 0.12)[..., None] * np.array([0.84, 0.74, 0.44], dtype=np.float32)

    # Atmospheric warm glowing front shoji window (u_m in [-1.92, -0.72], z in [1.02, 1.98])
    u0, u1 = -1.92, -0.72
    z0, z1 = 1.02, 1.98
    fr = ((u_m >= u0 - 0.10) & (u_m <= u1 + 0.10) & (z_m >= z0 - 0.10) & (z_m <= z1 + 0.09)).astype(np.float32)
    inside = ((u_m >= u0) & (u_m <= u1) & (z_m >= z0) & (z_m <= z1)).astype(np.float32)
    rgb = rgb * (1.0 - fr[..., None]) + np.array([0.125, 0.098, 0.078], dtype=np.float32)[None, None, :] * fr[..., None]

    wu = np.clip((u_m - u0) / (u1 - u0), 0.0, 1.0).astype(np.float32)
    wv = np.clip((z_m - z0) / (z1 - z0), 0.0, 1.0).astype(np.float32)
    # Rich radial amber hearth glow fading softly into warm amber corners + subtle interior rack shadow
    h_glow = np.exp(-((wu - 0.46) ** 2 * 2.2 + (wv - 0.34) ** 2 * 2.4))
    sil = np.clip(np.exp(-((wv - 0.58) / 0.055) ** 2) * 0.18 + np.exp(-((wu - 0.25) / 0.10) ** 2) * np.clip((0.52 - wv) * 4.0, 0.0, 1.0) * 0.22, 0.0, 0.34)
    gcol = np.zeros((th, tw, 3), dtype=np.float32)
    gcol[:, :, 0] = (0.72 + 0.46 * h_glow + mottle * 0.032) * (1.0 - sil)
    gcol[:, :, 1] = (0.48 + 0.44 * h_glow + mottle * 0.026) * (1.0 - sil * 1.15)
    gcol[:, :, 2] = (0.17 + 0.28 * h_glow + mottle * 0.015) * (1.0 - sil * 1.30)

    # Traditional delicate shoji kumiko grid + wider central sash rail
    munt = (
        (np.abs(np.mod(wu * 6.0 + 0.5, 1.0) - 0.5) < 0.044)
        | (np.abs(np.mod(wv * 4.0 + 0.5, 1.0) - 0.5) < 0.052)
        | (np.abs(wu - 0.50) < 0.028)
    ).astype(np.float32)
    wcol = gcol * (1.0 - munt[..., None] * 0.78) + np.array([0.18, 0.12, 0.075], dtype=np.float32)[None, None, :] * (munt[..., None] * 0.78)
    rgb = rgb * (1.0 - inside[..., None]) + wcol * inside[..., None]

    # Rustic dark timber sliding entry door (u_m in [-3.35, -2.30], z in [0.08, 1.98]) slightly ajar
    d0, d1 = -3.35, -2.30
    door_m = ((u_m >= d0) & (u_m <= d1) & (z_m >= 0.08) & (z_m <= 1.98)).astype(np.float32)
    du = np.clip((u_m - d0) / (d1 - d0), 0.0, 1.0).astype(np.float32)
    dv = np.clip((z_m - 0.08) / 1.90, 0.0, 1.0).astype(np.float32)
    ajar = (du > 0.84).astype(np.float32)
    slat = ((np.abs(np.mod(du * 8.0, 1.0) - 0.5) < 0.08) & (dv > 0.45)).astype(np.float32)
    dcol = np.array([0.14, 0.11, 0.085], dtype=np.float32)[None, None, :] * (1.0 - 0.28 * slat[..., None])
    dcol = dcol * (1.0 - ajar[..., None]) + np.array([0.66, 0.44, 0.16], dtype=np.float32)[None, None, :] * ajar[..., None]
    rgb = rgb * (1.0 - door_m[..., None]) + dcol * door_m[..., None]

    d_w = np.hypot(np.maximum(0.0, np.abs(u_m - (-1.32)) - 0.58), np.maximum(0.0, np.abs(z_m - 1.50) - 0.46)).astype(np.float32)
    rgb += (np.exp(-d_w * d_w * 3.0) * (1.0 - inside))[..., None] * np.array([0.42, 0.28, 0.09], dtype=np.float32)

    rgb = apply_shadow_bounce(rgb)
    rgb += tooth[..., None] * 0.008

    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.38) * in_gable[..., None]
    rgba[:, :, 3] = in_gable
    return rgba


def paint_mossy_shingle_roof_uv(rng: np.random.Generator, tw: int = 1024, th: int = 640) -> np.ndarray:
    """
    Stage 1 & 2: Exact pitched cedar shingle roof slope (kokerabuki) bathed in golden morning light,
    with staggered weathered silvered-cedar shingles, lush emerald-olive moss cushions colonizing
    the damp lower courses, horizontal bamboo weight battens (osae-buchi), and heavy dark ridge cap!
    """
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)

    mottle = fbm_gouache(th, tw, rng, scales=((44, 44), (16, 16), (7.5, 7.5), (3.4, 3.4)))
    mottle_f = fbm_gouache(th, tw, rng, scales=((20, 20), (8.5, 8.5), (4.0, 4.0), (2.0, 2.0)))
    dabs = flat_brush_dabs(th, tw, rng, count=92)
    tooth = paper_tooth(th, tw, rng)

    row_f = v * 34.0
    r_idx = np.floor(row_f).astype(np.int32)
    r_frac = (row_f - r_idx).astype(np.float32)
    col_f = u * 54.0 + (r_idx & 1) * 0.5
    c_idx = np.floor(col_f).astype(np.int32)
    c_frac = (col_f - c_idx).astype(np.float32)

    sh_id = ((r_idx * 73856093) ^ (c_idx * 19349663)) & 255
    s_tone = rng.uniform(-0.075, 0.075, size=256)[sh_id].astype(np.float32)
    s_warm = rng.uniform(-0.030, 0.034, size=256)[sh_id].astype(np.float32)

    butt_sh = np.exp(-((1.0 - r_frac) / 0.15) ** 2) * 0.46
    side_gap = np.exp(-(np.minimum(c_frac, 1.0 - c_frac) / 0.052) ** 2) * 0.36

    rgb = np.zeros((th, tw, 3), dtype=np.float32)
    rgb[:, :, 0] = 0.345 + s_tone + s_warm + mottle * 0.042 + dabs * 0.026
    rgb[:, :, 1] = 0.292 + s_tone * 0.92 + s_warm * 0.5 + mottle * 0.036 + dabs * 0.024
    rgb[:, :, 2] = 0.236 + s_tone * 0.80 - s_warm * 0.4 + mottle * 0.030 + dabs * 0.020
    rgb *= ((1.0 - butt_sh) * (1.0 - side_gap))[..., None]

    # Natural irregular emerald-olive velvet moss cushions colonizing damp lower & shaded cedar courses
    moss_env = np.clip((v.astype(np.float32) - 0.14) / 0.62, 0.0, 1.0)
    moss_field = np.clip((mottle * 0.58 + mottle_f * 0.42 + moss_env * 0.58 - 0.28) * 2.1, 0.0, 1.0)
    moss_col = np.zeros((th, tw, 3), dtype=np.float32)
    moss_col[:, :, 0] = 0.148 + mottle_f * 0.040 + dabs * 0.026
    moss_col[:, :, 1] = 0.312 + mottle_f * 0.058 + dabs * 0.038
    moss_col[:, :, 2] = 0.114 + mottle_f * 0.022 + dabs * 0.016
    rgb = rgb * (1.0 - 0.80 * moss_field[..., None]) + moss_col * (0.80 * moss_field[..., None])

    # Soft golden morning sunlight catching the upper-right facing roof plane!
    sun_grad = (0.78 + 0.32 * (1.0 - v.astype(np.float32))).astype(np.float32)
    rgb *= sun_grad[..., None]
    rgb += (0.20 * (1.0 - 0.48 * v.astype(np.float32)))[..., None] * np.array([0.85, 0.78, 0.46], dtype=np.float32)

    # Subtle weathered ridge shadow & outer eave drip-lip shadow (NO stiff flat horizontal stripes!)
    ridge_m = np.clip((0.055 - v.astype(np.float32)) / 0.022, 0.0, 1.0)
    eave_m = np.clip((v.astype(np.float32) - 0.942) / 0.024, 0.0, 1.0)
    rgb = rgb * (1.0 - ridge_m[..., None] * 0.78) + np.array([0.16, 0.14, 0.11], dtype=np.float32)[None, None, :] * (ridge_m[..., None] * 0.78)
    rgb = rgb * (1.0 - eave_m[..., None] * 0.82) + np.array([0.14, 0.12, 0.095], dtype=np.float32)[None, None, :] * (eave_m[..., None] * 0.82)

    rgb = apply_shadow_bounce(rgb)
    rgb += tooth[..., None] * 0.008

    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.32)
    rgba[:, :, 3] = 1.0
    return rgba


def paint_mill_porch_deck_uv(rng: np.random.Generator, tw: int = 512, th: int = 384) -> np.ndarray:
    """Weathered cedar timber porch floor deck in angled mill-local coordinates."""
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float64)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    u_m = -MILL_WIDTH + u * (MILL_WIDTH + 0.08)
    l_m = -PORCH_DEPTH + v * (PORCH_DEPTH + 0.10)
    wx = MILL_ORIGIN[0] + u_m * MILL_RGT[0] + l_m * MILL_FWD[0]
    wy = MILL_ORIGIN[1] + u_m * MILL_RGT[1] + l_m * MILL_FWD[1]
    wz = np.full_like(wx, Z_PORCH)

    mottle = fbm_gouache(th, tw, rng, scales=((36, 36), (14, 14), (6, 6), (3, 3)))
    dabs = flat_brush_dabs(th, tw, rng, count=85)
    tooth = paper_tooth(th, tw, rng)

    p_idx = np.floor((l_m + PORCH_DEPTH) / 0.21).astype(np.int32)
    p_frac = np.mod((l_m + PORCH_DEPTH) / 0.21, 1.0).astype(np.float32)
    p_tone = rng.uniform(-0.045, 0.045, size=32)[p_idx & 31].astype(np.float32)
    seam = np.exp(-(np.minimum(p_frac, 1.0 - p_frac) / 0.045) ** 2) * 0.56

    rgb = np.zeros((th, tw, 3), dtype=np.float32)
    rgb[:, :, 0] = 0.275 + p_tone + mottle * 0.026 + dabs * 0.018
    rgb[:, :, 1] = 0.230 + p_tone * 0.90 + mottle * 0.024 + dabs * 0.016
    rgb[:, :, 2] = 0.182 + p_tone * 0.78 + mottle * 0.020 + dabs * 0.014
    rgb *= (1.0 - seam)[..., None]

    sun = trace_valley_sunlight(wx, wy, wz)
    rgb *= (0.58 + 0.42 * sun)[..., None]
    rgb += (sun * 0.22)[..., None] * np.array([0.85, 0.76, 0.44], dtype=np.float32)
    hearth = eval_hearth_lights(wx, wy, wz)
    rgb += hearth[..., None] * np.array([0.52, 0.36, 0.13], dtype=np.float32)
    rgb = apply_shadow_bounce(rgb)
    rgb += tooth[..., None] * 0.007

    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.30)
    rgba[:, :, 3] = 1.0
    return rgba


def paint_granite_bridge_slab_uv(rng: np.random.Generator, tw: int = 512, th: int = 256) -> np.ndarray:
    """Chiselled weathered rough-hewn highland granite foot-slab top texture (NEVER flat grey!)."""
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    mottle = fbm_gouache(th, tw, rng, scales=((44, 44), (16, 16), (7, 7), (3, 3)))
    mottle_f = fbm_gouache(th, tw, rng, scales=((18, 18), (7, 7), (3.2, 3.2), (1.8, 1.8)))
    dabs = flat_brush_dabs(th, tw, rng, count=88)
    tooth = paper_tooth(th, tw, rng)

    # Rough hand-quarried granite chiselling & natural warm-cool stone patina
    chisel = np.sin(u * 28.0 + mottle * 2.2) * 0.016
    rgb = np.zeros((th, tw, 3), dtype=np.float32)
    rgb[:, :, 0] = 0.375 + chisel + mottle * 0.048 + mottle_f * 0.024 + dabs * 0.026
    rgb[:, :, 1] = 0.390 + chisel + mottle * 0.046 + mottle_f * 0.024 + dabs * 0.024
    rgb[:, :, 2] = 0.335 + chisel + mottle * 0.038 + mottle_f * 0.018 + dabs * 0.020

    # Natural lichen & velvety moss creeping softly along damp left/right bridge ends & front rim
    edge_end = np.clip((0.20 - u) / 0.20, 0.0, 1.0) + np.clip((u - 0.80) / 0.20, 0.0, 1.0)
    edge_rim = np.clip((0.18 - v) / 0.18, 0.0, 1.0) * 0.55
    moss_m = np.clip((edge_end * 0.82 + edge_rim + mottle * 0.38 + mottle_f * 0.28 - 0.34) * 1.9, 0.0, 1.0)
    moss_c = np.zeros((th, tw, 3), dtype=np.float32)
    moss_c[:, :, 0] = 0.125 + mottle_f * 0.034
    moss_c[:, :, 1] = 0.268 + mottle_f * 0.052
    moss_c[:, :, 2] = 0.102 + mottle_f * 0.020
    rgb = rgb * (1.0 - 0.86 * moss_m[..., None]) + moss_c * (0.86 * moss_m[..., None])

    # Morning golden sunlight gradient (stronger toward right footpath end!)
    sun_g = (0.76 + 0.34 * u).astype(np.float32)
    rgb *= sun_g[..., None]
    rgb += (0.16 * u)[..., None] * np.array([0.82, 0.78, 0.48], dtype=np.float32)

    # Organic hand-hewn edge taper (alpha mask slightly scalloped so slab isn't a razor rectangle)
    a_v = np.clip((v - (0.04 + 0.022 * mottle_f)) / 0.04, 0.0, 1.0) * np.clip(((0.96 + 0.022 * mottle_f) - v) / 0.04, 0.0, 1.0)
    rgb = apply_shadow_bounce(rgb)
    rgb += tooth[..., None] * 0.008

    rgba = np.zeros((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.32) * a_v[..., None]
    rgba[:, :, 3] = a_v
    return rgba


# =============================================================================
# Stage 2 & 3 — Open 3D Spoked Waterwheel (Suisha), Porch, Baskets & Flora
# =============================================================================

def draw_3d_poly(ctx: cairo.Context, cam: Camera3D, pts_3d: list[np.ndarray], mirrored: bool = False) -> bool:
    if mirrored:
        pts_3d = [mirror_point_z(p, Z_STREAM) for p in pts_3d]
    cl = cam.clip_polygon_3d(pts_3d)
    if len(cl) < 3:
        return False
    u0, v0, _ = cam.project(cl[0])
    ctx.move_to(float(u0), float(v0))
    for pt in cl[1:]:
        u, v, _ = cam.project(pt)
        ctx.line_to(float(u), float(v))
    ctx.close_path()
    return True


def draw_3d_box_beam(
    ctx: cairo.Context,
    cam: Camera3D,
    p0: np.ndarray,
    p1: np.ndarray,
    half_w: float,
    half_d: float,
    base_col: tuple[float, float, float],
    fog_rgb: np.ndarray,
    mirrored: bool = False,
) -> None:
    axis = p1 - p0
    ln = float(np.linalg.norm(axis))
    if ln < 1e-4:
        return
    a = axis / ln
    up_ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.85 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, up_ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)

    c_mid = 0.5 * (p0 + p1)
    dist = float(np.linalg.norm(c_mid - cam.eye))
    offs = [(-e1 * half_w - e2 * half_d), (+e1 * half_w - e2 * half_d), (+e1 * half_w + e2 * half_d), (-e1 * half_w + e2 * half_d)]
    faces = []
    for k in range(4):
        o0, o1 = offs[k], offs[(k + 1) % 4]
        quad = [p0 + o0, p0 + o1, p1 + o1, p1 + o0]
        fc = 0.25 * (quad[0] + quad[1] + quad[2] + quad[3])
        nrm = np.cross(quad[1] - quad[0], quad[3] - quad[0])
        nrm /= float(np.linalg.norm(nrm)) + 1e-6
        if float(np.dot(nrm, fc - c_mid)) < 0.0:
            nrm = -nrm
        faces.append((float(np.linalg.norm(fc - cam.eye)), quad, nrm))

    faces.sort(key=lambda x: -x[0])
    bc = np.array(base_col, dtype=np.float64)
    for _, quad, nrm in faces:
        ndotl = max(0.0, float(np.dot(nrm, -SUN_DIR)))
        col = bc * (0.50 + 0.64 * ndotl) + np.array([0.14, 0.12, 0.05]) * (ndotl ** 1.6)
        cf = fog_mix(np.clip(col, 0.0, 1.15), dist, fog_rgb)
        if draw_3d_poly(ctx, cam, quad, mirrored=mirrored):
            ctx.set_source_rgba(*cf, 0.99)
            ctx.fill()


def draw_waterwheel_and_flume(
    ctx: cairo.Context,
    cam: Camera3D,
    fog_rgb: np.ndarray,
    mirrored: bool = False,
) -> dict:
    """
    Stage 3 Band 2: Authentic Studio Ghibli open-spoked cedar waterwheel (suisha) clearing
    cleanly below the mill roof eave (z_top=1.90m < 2.28m):
      1) Unmistakably visible elevated mossy wooden water-supply trough (kakehi/toyu) running
         cleanly BELOW the eave (z=2.12m -> 1.97m) with glistening stream inside & ONE neat
         cross-braced timber trestle (leaving the warm side window completely unobstructed!)
      2) Open-spoked wooden wheel with deep shadowed bucket cavities between 16 angled cedar
         trough paddles, crisp dark iron rim straps, and sunlit felloe timber joinery
      3) Compact iron-banded octagonal wheel hub resting cleanly on a grounded outer wooden
         bearing post (jiku-uke) rooted in a mossy stream boulder (ZERO floating cannon axle!)
      4) Sparkling mountain water pouring physically OUT OF the open wooden trough mouth onto
         the upper-back buckets, cascading gently into aerated splash froth on the millrace!
    """
    wrng = np.random.default_rng(2028)
    u_mid = 0.5 * (WHEEL_U_INNER + WHEEL_U_OUTER)
    center = mill_pt(u_mid, WHEEL_L_CENTER, WHEEL_Z_CENTER)
    dist = float(np.linalg.norm(center - cam.eye))

    # 1. Clearly Visible Elevated Mossy Cedar Water Flume (kakehi / toyu) strictly BELOW roof eave!
    #    Originates cleanly from behind the mill's rear corner under the back eave (ZERO stick poking into empty sky!)
    l_fl0 = MILL_LENGTH + 0.16
    l_fl1 = WHEEL_L_CENTER + 0.26   # spout mouth poised directly above the open top wheel bucket cavity!
    z_fl0 = 2.13
    z_fl1 = 1.99
    flume_p0 = mill_pt(u_mid, l_fl0, z_fl0)
    flume_p1 = mill_pt(u_mid, l_fl1, z_fl1)

    # Sturdy twin-post cedar trestles (front at l=3.68 LEFT of window; back at l=5.82 RIGHT of window -> window stays wide open!)
    for tl in (5.82, 3.68):
        tz_top = z_fl1 + (z_fl0 - z_fl1) * (tl - l_fl1) / max(0.1, l_fl0 - l_fl1)
        draw_3d_box_beam(ctx, cam, mill_pt(u_mid - 0.15, tl, Z_STREAM - 0.04), mill_pt(u_mid - 0.12, tl, tz_top - 0.04), 0.040, 0.040, (0.21, 0.17, 0.13), fog_rgb, mirrored)
        draw_3d_box_beam(ctx, cam, mill_pt(u_mid + 0.15, tl, Z_STREAM - 0.04), mill_pt(u_mid + 0.12, tl, tz_top - 0.04), 0.040, 0.040, (0.26, 0.21, 0.15), fog_rgb, mirrored)
        draw_3d_box_beam(ctx, cam, mill_pt(u_mid - 0.19, tl, tz_top - 0.07), mill_pt(u_mid + 0.19, tl, tz_top - 0.07), 0.034, 0.042, (0.24, 0.19, 0.14), fog_rgb, mirrored)
        if tl < 4.5:
            draw_3d_box_beam(ctx, cam, mill_pt(u_mid - 0.13, tl, tz_top - 0.56), mill_pt(u_mid + 0.13, tl, tz_top - 0.14), 0.024, 0.024, (0.21, 0.17, 0.13), fog_rgb, mirrored)
        if not mirrored:
            for pu_t in (u_mid - 0.15, u_mid + 0.15):
                up_m, vp_m, zp_m = cam.project(mill_pt(pu_t, tl, Z_STREAM + 0.005))
                if zp_m > Z_NEAR:
                    rm_px = 0.085 * cam.K[0, 0] / float(zp_m)
                    ctx.save()
                    ctx.translate(float(up_m), float(vp_m))
                    ctx.scale(rm_px * 1.35, rm_px * 0.34)
                    ctx.set_source_rgba(0.90, 0.97, 0.94, 0.48)
                    ctx.set_line_width(max(0.08, 1.4 / max(5.0, rm_px)))
                    ctx.arc(0, 0, 1.0, 0, math.tau)
                    ctx.stroke()
                    ctx.restore()

    # Open U-shaped wooden water trough (floor plank + shaded inner cheek + sparkling flowing stream + sunlit outer cheek!)
    draw_3d_box_beam(ctx, cam, flume_p0, flume_p1, 0.165, 0.032, (0.23, 0.18, 0.135), fog_rgb, mirrored)
    draw_3d_box_beam(ctx, cam, flume_p0 - 0.135 * MILL_RGT + 0.062 * MILL_UP, flume_p1 - 0.135 * MILL_RGT + 0.062 * MILL_UP, 0.020, 0.066, (0.18, 0.14, 0.105), fog_rgb, mirrored)
    if not mirrored:
        if draw_3d_poly(ctx, cam, [
            flume_p0 - 0.105 * MILL_RGT + 0.056 * MILL_UP,
            flume_p0 + 0.105 * MILL_RGT + 0.056 * MILL_UP,
            flume_p1 + 0.105 * MILL_RGT + 0.056 * MILL_UP,
            flume_p1 - 0.105 * MILL_RGT + 0.056 * MILL_UP,
        ], mirrored=False):
            ctx.set_source_rgba(0.76, 0.93, 0.89, 0.90)
            ctx.fill()
    draw_3d_box_beam(ctx, cam, flume_p0 + 0.135 * MILL_RGT + 0.062 * MILL_UP, flume_p1 + 0.135 * MILL_RGT + 0.062 * MILL_UP, 0.020, 0.066, (0.31, 0.25, 0.18), fog_rgb, mirrored)

    # Inner drive shaft segment (wall u=-0.04 -> wheel center u_mid) drawn BEFORE spokes so spokes cleanly overlap it in 3D!
    draw_3d_box_beam(
        ctx, cam,
        mill_pt(-0.04, WHEEL_L_CENTER, WHEEL_Z_CENTER),
        mill_pt(u_mid + 0.02, WHEEL_L_CENTER, WHEEL_Z_CENTER),
        0.048, 0.048, (0.16, 0.13, 0.095), fog_rgb, mirrored,
    )

    n_seg = 36
    thetas = np.linspace(0.0, math.tau, n_seg + 1)

    def ring_pts(u_val: float, r_val: float) -> list[np.ndarray]:
        return [
            mill_pt(u_val, WHEEL_L_CENTER + r_val * math.cos(th), WHEEL_Z_CENTER + r_val * math.sin(th))
            for th in thetas
        ]

    def paint_annular_rim(u_val: float, shade: float, with_joints: bool) -> None:
        out_p = ring_pts(u_val, WHEEL_R_OUTER)
        in_p = ring_pts(u_val, WHEEL_R_INNER)
        for k in range(n_seg):
            th_m = 0.5 * (thetas[k] + thetas[k + 1])
            wet = max(0.0, -math.sin(th_m))
            lit = max(0.0, math.sin(th_m) * 0.65 - math.cos(th_m) * 0.25)
            felloe_var = 0.038 * math.sin((k // 6) * 2.3)
            base = (np.array([0.27, 0.21, 0.145]) + felloe_var) * shade * (1.0 - 0.34 * wet) + np.array([0.19, 0.16, 0.08]) * lit * shade
            if with_joints and ((k % 6) in (1, 4)):
                base = base * 0.76 + np.array([0.11, 0.25, 0.11]) * 0.24
            cf = fog_mix(np.clip(base, 0.0, 1.15), dist, fog_rgb)
            if draw_3d_poly(ctx, cam, [out_p[k], out_p[k + 1], in_p[k + 1], in_p[k]], mirrored=mirrored):
                ctx.set_source_rgba(*cf, 0.99)
                ctx.fill()
            if with_joints and (k % 6 == 0):
                p_a = mirror_point_z(out_p[k], Z_STREAM) if mirrored else out_p[k]
                p_b = mirror_point_z(in_p[k], Z_STREAM) if mirrored else in_p[k]
                ua, va, za = cam.project(p_a)
                ub, vb, zb = cam.project(p_b)
                if za > Z_NEAR and zb > Z_NEAR:
                    ctx.set_source_rgba(*fog_mix([0.08, 0.07, 0.06], dist, fog_rgb), 0.86)
                    ctx.set_line_width(2.2)
                    ctx.move_to(float(ua), float(va))
                    ctx.line_to(float(ub), float(vb))
                    ctx.stroke()

    # 2. Inner rim ring (closest to cedar wall, drawn FIRST!)
    paint_annular_rim(WHEEL_U_INNER, shade=0.56, with_joints=False)

    # Dark shadowed inner bucket sole-drum floor (R_INNER) — outer circumference R_OUTER stays OPEN for water buckets!
    in_drum_i = ring_pts(WHEEL_U_INNER, WHEEL_R_INNER)
    in_drum_o = ring_pts(WHEEL_U_OUTER, WHEEL_R_INNER)
    for k in range(n_seg):
        th_m = 0.5 * (thetas[k] + thetas[k + 1])
        if math.cos(th_m) < 0.35 or math.sin(th_m) > -0.35:
            cf = fog_mix([0.095, 0.08, 0.058], dist, fog_rgb)
            if draw_3d_poly(ctx, cam, [in_drum_i[k], in_drum_i[k + 1], in_drum_o[k + 1], in_drum_o[k]], mirrored=mirrored):
                ctx.set_source_rgba(*cf, 0.97)
                ctx.fill()

    # 3. 16 crisp upward-cupped overshot cedar bucket paddles bridging inner & outer rim (sorted back-to-front!)
    #    th_out = th - 0.20 angles every paddle to catch water from the upper-right flume and cradle it down the left side!
    n_pad = 16
    paddles = []
    for k in range(n_pad):
        th = (k / n_pad) * math.tau + 0.14
        th_out = th - 0.20
        p0 = mill_pt(WHEEL_U_INNER, WHEEL_L_CENTER + WHEEL_R_INNER * math.cos(th), WHEEL_Z_CENTER + WHEEL_R_INNER * math.sin(th))
        p1 = mill_pt(WHEEL_U_OUTER, WHEEL_L_CENTER + WHEEL_R_INNER * math.cos(th), WHEEL_Z_CENTER + WHEEL_R_INNER * math.sin(th))
        p2 = mill_pt(WHEEL_U_OUTER, WHEEL_L_CENTER + WHEEL_R_OUTER * math.cos(th_out), WHEEL_Z_CENTER + WHEEL_R_OUTER * math.sin(th_out))
        p3 = mill_pt(WHEEL_U_INNER, WHEEL_L_CENTER + WHEEL_R_OUTER * math.cos(th_out), WHEEL_Z_CENTER + WHEEL_R_OUTER * math.sin(th_out))
        pc = 0.25 * (p0 + p1 + p2 + p3)
        paddles.append((float(np.linalg.norm(pc - cam.eye)), th, [p0, p1, p2, p3]))

    paddles.sort(key=lambda x: -x[0])
    for _, th, quad in paddles:
        nrm = -math.sin(th) * MILL_FWD + math.cos(th) * MILL_UP
        ndotl = abs(float(np.dot(nrm, -SUN_DIR)))
        wet = max(0.0, -math.cos(th) * 0.65 - math.sin(th) * 0.45)
        c_pad = np.array([0.23, 0.175, 0.125]) * (0.50 + 0.68 * ndotl) * (1.0 - 0.34 * wet)
        # Water-filled wet sheen on descending left-side buckets (cos(th) < -0.15)
        if not mirrored and math.cos(th) < -0.15 and math.sin(th) > -0.82:
            c_pad = c_pad * 0.68 + np.array([0.52, 0.72, 0.66]) * 0.32
        cf = fog_mix(np.clip(c_pad, 0.0, 1.15), dist, fog_rgb)
        if draw_3d_poly(ctx, cam, quad, mirrored=mirrored):
            ctx.set_source_rgba(*cf, 0.99)
            ctx.fill()
            ctx.set_source_rgba(*fog_mix([0.07, 0.055, 0.04], dist, fog_rgb), 0.82)
            ctx.set_line_width(1.2)
            ctx.stroke()

    # 4. Crisp OPEN traditional Japanese igeta-gumi cross-braced wooden spokes (slender & airy!)
    u_sp = u_mid + 0.03
    for off_s in (-0.13, 0.13):
        ang0 = math.radians(18.0)
        ca, sa = math.cos(ang0), math.sin(ang0)
        span = math.sqrt(max(0.1, (WHEEL_R_INNER + 0.04) ** 2 - off_s ** 2))
        draw_3d_box_beam(
            ctx, cam,
            mill_pt(u_sp, WHEEL_L_CENTER - ca * span - sa * off_s, WHEEL_Z_CENTER - sa * span + ca * off_s),
            mill_pt(u_sp, WHEEL_L_CENTER + ca * span - sa * off_s, WHEEL_Z_CENTER + sa * span + ca * off_s),
            0.032, 0.034, (0.27, 0.21, 0.15), fog_rgb, mirrored,
        )
        draw_3d_box_beam(
            ctx, cam,
            mill_pt(u_sp, WHEEL_L_CENTER + sa * span + ca * off_s, WHEEL_Z_CENTER - ca * span + sa * off_s),
            mill_pt(u_sp, WHEEL_L_CENTER - sa * span + ca * off_s, WHEEL_Z_CENTER + ca * span + sa * off_s),
            0.032, 0.034, (0.25, 0.19, 0.135), fog_rgb, mirrored,
        )
    for ang_d in (63.0, 153.0):
        a = math.radians(ang_d)
        dl, dz = math.cos(a) * (WHEEL_R_INNER + 0.03), math.sin(a) * (WHEEL_R_INNER + 0.03)
        draw_3d_box_beam(
            ctx, cam,
            mill_pt(u_sp - 0.03, WHEEL_L_CENTER - dl, WHEEL_Z_CENTER - dz),
            mill_pt(u_sp - 0.03, WHEEL_L_CENTER + dl, WHEEL_Z_CENTER + dz),
            0.026, 0.028, (0.22, 0.17, 0.125), fog_rgb, mirrored,
        )

    # 5. TRUE 3D HYDRAULIC WATER FLOW — Drawn INSIDE the bucket channel BEFORE the outer wheel shroud!
    #    a) Sparkling mountain stream arcs cleanly OUT OF the kakehi spout mouth directly INTO the open top bucket
    #       cavity (ending at r=0.91m between R_INNER=0.86m and R_OUTER=1.08m — physically occluded by outer rim!)
    #    b) Churning water fills the descending left-hand buckets & spills in translucent shimmering curtains
    #       down the outer descending rim all the way into the foaming millrace splash pool!
    if not mirrored:
        # (5a) Main water jet pouring physically INTO the top open bucket between inner & outer rims!
        l_pts, r_pts = [], []
        for s in range(18):
            tf = s / 17.0
            la = l_fl1 - tf * 0.42          # l: 2.54 -> 2.12 (just past top-dead-center 2.28)
            za = (z_fl1 + 0.035) - 0.25 * (tf * tf) - 0.06 * tf   # z: 2.025 -> 1.715 (inside bucket R_IN=1.68..R_OUT=1.90!)
            w_jet = 0.090 + 0.032 * tf
            ul, vl, zl = cam.project(mill_pt(u_mid - w_jet, la, za))
            ur, vr, zr = cam.project(mill_pt(u_mid + w_jet, la, za))
            if zl > Z_NEAR and zr > Z_NEAR:
                l_pts.append((float(ul), float(vl)))
                r_pts.append((float(ur), float(vr)))
        if len(l_pts) >= 4:
            g_jet = cairo.LinearGradient(l_pts[0][0], l_pts[0][1], l_pts[-1][0], l_pts[-1][1])
            g_jet.add_color_stop_rgba(0.00, 0.72, 0.90, 0.87, 0.92)
            g_jet.add_color_stop_rgba(0.55, 0.88, 0.97, 0.94, 0.88)
            g_jet.add_color_stop_rgba(1.00, 0.95, 0.99, 0.97, 0.80)
            ctx.set_source(g_jet)
            ctx.move_to(*l_pts[0])
            for pt in l_pts[1:]:
                ctx.line_to(*pt)
            for pt in reversed(r_pts):
                ctx.line_to(*pt)
            ctx.close_path()
            ctx.fill()
            for frac_s, a_s, lw_s in [(0.26, 0.54, 1.5), (0.50, 0.72, 2.0), (0.76, 0.46, 1.4)]:
                ctx.set_source_rgba(0.98, 1.00, 0.97, a_s)
                ctx.set_line_width(lw_s)
                mx0 = l_pts[0][0] * (1.0 - frac_s) + r_pts[0][0] * frac_s
                my0 = l_pts[0][1] * (1.0 - frac_s) + r_pts[0][1] * frac_s
                ctx.move_to(mx0, my0)
                for (lx, ly), (rx, ry) in zip(l_pts[1:], r_pts[1:]):
                    ctx.line_to(lx * (1.0 - frac_s) + rx * frac_s, ly * (1.0 - frac_s) + ry * frac_s)
                ctx.stroke()

        # (5b) Organic churning water cradled STRICTLY inside the exact descending left paddle pockets (aligned to k!)
        for k_b in range(n_pad):
            th_b = (k_b / n_pad) * math.tau + 0.14
            if not (math.radians(95.0) <= th_b <= math.radians(225.0)):
                continue
            th_lip = th_b - 0.20
            th_back = th_b + 0.12
            r_in_w = WHEEL_R_INNER + 0.02
            r_lip_w = WHEEL_R_OUTER - 0.03
            b_quad = [
                mill_pt(WHEEL_U_INNER + 0.03, WHEEL_L_CENTER + r_in_w * math.cos(th_back), WHEEL_Z_CENTER + r_in_w * math.sin(th_back)),
                mill_pt(WHEEL_U_OUTER - 0.03, WHEEL_L_CENTER + r_in_w * math.cos(th_back), WHEEL_Z_CENTER + r_in_w * math.sin(th_back)),
                mill_pt(WHEEL_U_OUTER - 0.03, WHEEL_L_CENTER + r_lip_w * math.cos(th_lip), WHEEL_Z_CENTER + r_lip_w * math.sin(th_lip)),
                mill_pt(WHEEL_U_INNER + 0.03, WHEEL_L_CENTER + r_lip_w * math.cos(th_lip), WHEEL_Z_CENTER + r_lip_w * math.sin(th_lip)),
            ]
            if draw_3d_poly(ctx, cam, b_quad, mirrored=False):
                u_a, v_a, _ = cam.project(b_quad[0])
                u_b, v_b, _ = cam.project(b_quad[2])
                g_bk = cairo.LinearGradient(float(u_a), float(v_a), float(u_b), float(v_b))
                g_bk.add_color_stop_rgba(0.0, 0.56, 0.78, 0.74, 0.52)
                g_bk.add_color_stop_rgba(0.75, 0.84, 0.95, 0.92, 0.70)
                g_bk.add_color_stop_rgba(1.0, 0.96, 1.00, 0.98, 0.82)
                ctx.set_source(g_bk)
                ctx.fill()

            # (5c) Painterly organic multi-strand water threads & soft translucent spill veils pouring off each paddle lip!
            #      Upper buckets (95..158 deg) only overflow briefly onto the next lower bucket (NEVER falling through open spokes!);
            #      outer descending left buckets (158..228 deg) cascade cleanly outside the rim all the way into the splash pool!
            l_lip = WHEEL_L_CENTER + r_lip_w * math.cos(th_lip)
            z_lip = WHEEL_Z_CENTER + r_lip_w * math.sin(th_lip)
            if th_b < math.radians(158.0):
                z_drop = min(0.28, max(0.12, z_lip - (Z_STREAM + 0.15)))
                spill_strength = 0.24
                lat_push = 0.14
            else:
                z_drop = max(0.14, z_lip - (Z_STREAM + 0.015))
                spill_strength = 0.56 if th_b < math.radians(202.0) else 0.46
                lat_push = 0.10
            for sub_u_off, sub_w, sub_a in ((-0.075, 0.032, spill_strength * 0.72), (0.010, 0.042, spill_strength), (0.085, 0.028, spill_strength * 0.68)):
                sl_pts, sr_pts = [], []
                for ss in range(12):
                    sf = ss / 11.0
                    la_s = l_lip - (lat_push + 0.030 * math.sin(k_b * 2.1)) * sf - 0.016 * math.sin(sf * math.pi)
                    za_s = z_lip - z_drop * (0.24 * sf + 0.76 * sf * sf)
                    ws = sub_w * (0.85 + 0.52 * sf)
                    u1, v1, z1 = cam.project(mill_pt(u_mid + sub_u_off - ws, la_s, za_s))
                    u2, v2, z2 = cam.project(mill_pt(u_mid + sub_u_off + ws, la_s, za_s))
                    if z1 > Z_NEAR and z2 > Z_NEAR:
                        sl_pts.append((float(u1), float(v1)))
                        sr_pts.append((float(u2), float(v2)))
                if len(sl_pts) >= 4:
                    g_sp = cairo.LinearGradient(sl_pts[0][0], sl_pts[0][1], sl_pts[-1][0], sl_pts[-1][1])
                    g_sp.add_color_stop_rgba(0.00, 0.90, 0.98, 0.96, sub_a)
                    g_sp.add_color_stop_rgba(0.55, 0.86, 0.96, 0.93, sub_a * 0.62)
                    g_sp.add_color_stop_rgba(0.86, 0.92, 0.99, 0.96, sub_a * 0.24)
                    g_sp.add_color_stop_rgba(1.00, 0.96, 1.00, 0.98, 0.00)
                    ctx.set_source(g_sp)
                    ctx.move_to(*sl_pts[0])
                    for pt in sl_pts[1:]:
                        ctx.line_to(*pt)
                    for pt in reversed(sr_pts):
                        ctx.line_to(*pt)
                    ctx.close_path()
                    ctx.fill()
                    if th_b >= math.radians(158.0):
                        g_th = cairo.LinearGradient(sl_pts[0][0], sl_pts[0][1], sl_pts[-1][0], sl_pts[-1][1])
                        g_th.add_color_stop_rgba(0.0, 0.99, 1.00, 0.98, min(0.80, sub_a * 1.12))
                        g_th.add_color_stop_rgba(0.75, 0.98, 1.00, 0.98, sub_a * 0.42)
                        g_th.add_color_stop_rgba(1.0, 0.98, 1.00, 0.98, 0.0)
                        ctx.set_source(g_th)
                        ctx.set_line_width(1.30)
                        ctx.move_to(0.5 * (sl_pts[0][0] + sr_pts[0][0]), 0.5 * (sl_pts[0][1] + sr_pts[0][1]))
                        for (lx, ly), (rx, ry) in zip(sl_pts[1:], sr_pts[1:]):
                            ctx.line_to(0.5 * (lx + rx), 0.5 * (ly + ry))
                        ctx.stroke()

    # 6. Outer wheel rim ring (drawn AFTER bucket water so outer wooden shroud cleanly frames & occludes water entering buckets!)
    paint_annular_rim(WHEEL_U_OUTER, shade=1.02, with_joints=True)

    # Subtle outer plank edge highlight strictly along the thin outer rim board (0.04m thick, keeping bucket cavities wide open!)
    out_rim_back = ring_pts(WHEEL_U_OUTER - 0.042, WHEEL_R_OUTER)
    out_rim_front = ring_pts(WHEEL_U_OUTER, WHEEL_R_OUTER)
    for k in range(n_seg):
        th_m = 0.5 * (thetas[k] + thetas[k + 1])
        if math.cos(th_m) < 0.0 or math.sin(th_m) > 0.25:
            lit = max(0.0, math.sin(th_m) * 0.70)
            col = np.array([0.21, 0.165, 0.118]) + np.array([0.22, 0.19, 0.09]) * lit
            cf = fog_mix(np.clip(col, 0.0, 1.1), dist, fog_rgb)
            if draw_3d_poly(ctx, cam, [out_rim_back[k], out_rim_back[k + 1], out_rim_front[k + 1], out_rim_front[k]], mirrored=mirrored):
                ctx.set_source_rgba(*cf, 0.98)
                ctx.fill()

    # 7. Authentic Compact Iron-Banded Wooden Hub + Short Outer Axle Journal Resting on Grounded Bearing Post!
    hub_pt = mill_pt(u_sp + 0.04, WHEEL_L_CENTER, WHEEL_Z_CENTER)
    if mirrored:
        hub_pt = mirror_point_z(hub_pt, Z_STREAM)
    hu, hv, hz = cam.project(hub_pt)
    if hz > Z_NEAR:
        hr = 0.105 * cam.K[0, 0] / float(hz)
        ctx.set_source_rgba(*fog_mix([0.11, 0.10, 0.09], dist, fog_rgb), 0.98)
        ctx.arc(float(hu), float(hv), hr, 0, math.tau)
        ctx.fill()
        ctx.set_source_rgba(*fog_mix([0.30, 0.24, 0.17], dist, fog_rgb), 0.94)
        ctx.arc(float(hu), float(hv), hr * 0.70, 0, math.tau)
        ctx.fill()

    u_bear = WHEEL_U_OUTER + 0.19
    draw_3d_box_beam(
        ctx, cam,
        mill_pt(u_sp + 0.03, WHEEL_L_CENTER, WHEEL_Z_CENTER),
        mill_pt(u_bear + 0.03, WHEEL_L_CENTER, WHEEL_Z_CENTER),
        0.046, 0.046, (0.18, 0.145, 0.11), fog_rgb, mirrored,
    )
    draw_3d_box_beam(
        ctx, cam,
        mill_pt(u_bear, WHEEL_L_CENTER, Z_STREAM - 0.06),
        mill_pt(u_bear, WHEEL_L_CENTER, WHEEL_Z_CENTER + 0.04),
        0.068, 0.076, (0.22, 0.175, 0.13), fog_rgb, mirrored,
    )
    draw_3d_box_beam(
        ctx, cam,
        mill_pt(u_bear, WHEEL_L_CENTER - 0.34, Z_STREAM - 0.04),
        mill_pt(u_bear, WHEEL_L_CENTER - 0.04, WHEEL_Z_CENTER - 0.22),
        0.040, 0.046, (0.19, 0.15, 0.11), fog_rgb, mirrored,
    )
    cap_pt = mill_pt(u_bear + 0.05, WHEEL_L_CENTER, WHEEL_Z_CENTER)
    if mirrored:
        cap_pt = mirror_point_z(cap_pt, Z_STREAM)
    cu_p, cv_p, cz_p = cam.project(cap_pt)
    if cz_p > Z_NEAR:
        cr_p = 0.055 * cam.K[0, 0] / float(cz_p)
        ctx.set_source_rgba(*fog_mix([0.11, 0.11, 0.10], dist, fog_rgb), 0.96)
        ctx.arc(float(cu_p), float(cv_p), cr_p, 0, math.tau)
        ctx.fill()
        ctx.set_source_rgba(*fog_mix([0.38, 0.34, 0.26], dist, fog_rgb), 0.85)
        ctx.arc(float(cu_p) + cr_p * 0.15, float(cv_p) - cr_p * 0.15, cr_p * 0.45, 0, math.tau)
        ctx.fill()

    # 8. Delicate Top Bucket-Entry Spray Mist, Descending Water Droplets & Waterline Splash Froth!
    if not mirrored:
        # Compact fine mist spray strictly at the TOP rim entry point (z = 1.90, NEVER dangling over spokes!)
        impact_pt = mill_pt(u_mid + 0.04, WHEEL_L_CENTER - 0.04, WHEEL_Z_CENTER + WHEEL_R_OUTER - 0.02)
        iu, iv, iz = cam.project(impact_pt)
        if iz > Z_NEAR:
            for _ in range(9):
                ox = float(wrng.normal(0.0, 6.5))
                oy = float(wrng.normal(0.0, 4.2))
                rr = float(wrng.uniform(3.8, 8.8))
                g_im = cairo.RadialGradient(float(iu) + ox, float(iv) + oy, 1.0, float(iu) + ox, float(iv) + oy, rr)
                g_im.add_color_stop_rgba(0.0, 0.98, 1.00, 0.97, 0.52)
                g_im.add_color_stop_rgba(1.0, 0.86, 0.96, 0.92, 0.0)
                ctx.set_source(g_im)
                ctx.arc(float(iu) + ox, float(iv) + oy, rr, 0, math.tau)
                ctx.fill()

        # Delicate shimmering water droplets trailing off the descending wet left buckets into the stream!
        for k in range(22):
            th = math.radians(125.0 + (k / 21.0) * 115.0)
            pl_d = WHEEL_L_CENTER + (WHEEL_R_OUTER + 0.02) * math.cos(th)
            pz_d = WHEEL_Z_CENTER + (WHEEL_R_OUTER + 0.02) * math.sin(th)
            if pz_d < Z_STREAM + 0.08:
                continue
            u_d = float(wrng.uniform(WHEEL_U_INNER + 0.06, WHEEL_U_OUTER + 0.01))
            for step_d in range(3):
                dz_d = step_d * float(wrng.uniform(0.07, 0.15))
                if pz_d - dz_d < Z_STREAM + 0.04:
                    break
                ud, vd, zd = cam.project(mill_pt(u_d, pl_d - 0.025 * step_d, pz_d - dz_d))
                if zd > Z_NEAR:
                    ctx.set_source_rgba(0.95, 0.99, 0.97, float(wrng.uniform(0.32, 0.66)) * (1.0 - 0.22 * step_d))
                    ctx.save()
                    ctx.translate(float(ud), float(vd))
                    ctx.scale(0.65, 2.3)
                    ctx.arc(0, 0, float(wrng.uniform(1.0, 1.9)), 0, math.tau)
                    ctx.fill()
                    ctx.restore()

        # Painterly waterline aerated splash froth & concentric ripple rings sitting strictly ON the stream surface!
        sp_pos = mill_pt(u_mid + 0.04, WHEEL_L_CENTER - 0.24, Z_STREAM + 0.01)
        su, sv, sz = cam.project(sp_pos)
        if sz > Z_NEAR:
            sr = 0.50 * cam.K[0, 0] / float(sz)
            for k in range(26):
                ox = float(wrng.normal(0.0, sr * 0.58))
                oy = float(wrng.normal(sr * 0.05, sr * 0.09))
                pr = sr * float(wrng.uniform(0.09, 0.22))
                ctx.save()
                ctx.translate(float(su) + ox, float(sv) + oy)
                ctx.scale(2.20, 0.38)
                g_f = cairo.RadialGradient(0, 0, pr * 0.12, 0, 0, pr)
                g_f.add_color_stop_rgba(0.0, 0.98, 1.00, 0.97, float(wrng.uniform(0.36, 0.66)))
                g_f.add_color_stop_rgba(0.62, 0.85, 0.96, 0.92, float(wrng.uniform(0.16, 0.30)))
                g_f.add_color_stop_rgba(1.0, 0.74, 0.90, 0.86, 0.0)
                ctx.set_source(g_f)
                ctx.arc(0, 0, pr, 0, math.tau)
                ctx.fill()
                ctx.restore()
            for rk in range(4):
                rx_r = sr * (0.58 + rk * 0.40)
                ry_r = rx_r * 0.22
                ctx.save()
                ctx.translate(float(su) + rk * 5.5, float(sv) + sr * 0.08 + rk * 4.0)
                ctx.scale(rx_r, ry_r)
                ctx.set_source_rgba(0.95, 0.99, 0.96, 0.44 - 0.08 * rk)
                ctx.set_line_width(max(0.02, 1.9 / max(15.0, rx_r)))
                ctx.arc(0, 0, 1.0, -0.25, math.pi + 0.25)
                ctx.stroke()
                ctx.restore()
            for _ in range(20):
                ox = float(wrng.normal(0.0, sr * 0.56))
                oy = float(wrng.normal(-sr * 0.06, sr * 0.14))
                ctx.set_source_rgba(0.99, 1.0, 0.96, float(wrng.uniform(0.30, 0.74)))
                ctx.arc(float(su) + ox, float(sv) + oy, float(wrng.uniform(0.8, 1.7)), 0, math.tau)
                ctx.fill()

    cu, cv, _ = cam.project(mill_pt(u_mid + 0.04, WHEEL_L_CENTER - 0.24, Z_STREAM + 0.01))
    return {"splash_uv": (float(cu) / W, float(cv) / H)}


def draw_mill_under_eave_rafters(ctx: cairo.Context, cam: Camera3D, fog_rgb: np.ndarray, mirrored: bool = False) -> None:
    """Exposed dark-cedar rafters tucked cleanly UNDER the mossy roof plane (ZERO spiky sticks protruding past eave!)."""
    for rl in np.linspace(-ROOF_OVER_F + 0.28, MILL_LENGTH + ROOF_OVER_B - 0.28, 12):
        draw_3d_box_beam(
            ctx, cam,
            mill_pt(-0.04, float(rl), MILL_Z_EAVE - 0.02),
            mill_pt(ROOF_OVER_R - 0.04, float(rl), ROOF_Z_EAVE_R - 0.05),
            0.032, 0.038, (0.16, 0.13, 0.095), fog_rgb, mirrored=mirrored,
        )


def draw_mill_porch_under_roof(ctx: cairo.Context, cam: Camera3D, fog_rgb: np.ndarray, mirrored: bool = False) -> None:
    """Sturdy cedar porch support posts & horizontal tie beams drawn BEFORE the roof so the mossy roof cleanly shelters them!"""
    for u_p in (-0.12, -1.55, -3.20):
        draw_3d_box_beam(
            ctx, cam,
            mill_pt(u_p, -PORCH_DEPTH + 0.18, Z_PORCH),
            mill_pt(u_p, -PORCH_DEPTH + 0.18, MILL_Z_EAVE - 0.08),
            0.088, 0.088, (0.24, 0.19, 0.14), fog_rgb, mirrored,
        )
    draw_3d_box_beam(
        ctx, cam,
        mill_pt(-MILL_WIDTH + 0.10, -PORCH_DEPTH + 0.18, MILL_Z_EAVE - 0.10),
        mill_pt(-0.02, -PORCH_DEPTH + 0.18, MILL_Z_EAVE - 0.10),
        0.088, 0.10, (0.22, 0.17, 0.13), fog_rgb, mirrored,
    )
    draw_3d_box_beam(
        ctx, cam,
        mill_pt(-0.08, -PORCH_DEPTH + 0.18, MILL_Z_EAVE - 0.08),
        mill_pt(-0.08, 0.10, MILL_Z_EAVE - 0.08),
        0.080, 0.095, (0.24, 0.19, 0.14), fog_rgb, mirrored,
    )


def draw_mill_roof_trim(ctx: cairo.Context, cam: Camera3D, fog_rgb: np.ndarray, mirrored: bool = False) -> None:
    """Sculpted Japanese cedar eave fascia (hanakakushi), front bargeboard (hafu) & tapered layered ridge cap (mune)."""
    u_ridge = -0.5 * MILL_WIDTH
    l0 = -ROOF_OVER_F - 0.02
    l1 = MILL_LENGTH + ROOF_OVER_B + 0.02

    # 1. Clean continuous dark-cedar outer eave fascia board (hanakakushi) capping the tucked-in rafters flush!
    draw_3d_box_beam(
        ctx, cam,
        mill_pt(ROOF_OVER_R, l0, ROOF_Z_EAVE_R - 0.03),
        mill_pt(ROOF_OVER_R, l1, ROOF_Z_EAVE_R - 0.03),
        0.036, 0.068, (0.18, 0.145, 0.11), fog_rgb, mirrored,
    )

    # 2. Clean flush front gable bargeboards (hafu) hugging the roof plane with zero corner notch!
    p_rf = mill_pt(u_ridge, l0, MILL_Z_RIDGE + 0.02)
    p_er = mill_pt(ROOF_OVER_R + 0.01, l0, ROOF_Z_EAVE_R - 0.01)
    p_el = mill_pt(-MILL_WIDTH - 0.55, l0, ROOF_Z_EAVE_R + 0.06)
    draw_3d_box_beam(ctx, cam, p_rf, p_er, 0.052, 0.082, (0.20, 0.16, 0.12), fog_rgb, mirrored)
    draw_3d_box_beam(ctx, cam, p_rf, p_el, 0.052, 0.082, (0.16, 0.13, 0.10), fog_rgb, mirrored)

    # 3. Sculpted traditional layered Japanese dark-slate & cedar ridge cap (mune) snugly seating the roof peak!
    p_rb = mill_pt(u_ridge, l1, MILL_Z_RIDGE + 0.02)
    # Lower weathered cedar ridge saddle
    draw_3d_box_beam(ctx, cam, p_rf, p_rb, 0.11, 0.055, (0.19, 0.16, 0.125), fog_rgb, mirrored)
    # Sleek slightly overhanging upper slate/kawara ridge cap tile (tapered, never a giant chopped block!)
    draw_3d_box_beam(
        ctx, cam,
        mill_pt(u_ridge, l0 - 0.06, MILL_Z_RIDGE + 0.085),
        mill_pt(u_ridge, l1 + 0.06, MILL_Z_RIDGE + 0.085),
        0.135, 0.032, (0.16, 0.145, 0.125), fog_rgb, mirrored,
    )
    # Subtle sculpted end-cap finial (onigawara-style weathered timber ridge stop) at far gable peak!
    draw_3d_box_beam(
        ctx, cam,
        mill_pt(u_ridge, l1 + 0.02, MILL_Z_RIDGE + 0.01),
        mill_pt(u_ridge, l1 + 0.02, MILL_Z_RIDGE + 0.14),
        0.075, 0.055, (0.15, 0.135, 0.115), fog_rgb, mirrored,
    )


def draw_woven_tea_basket(
    ctx: cairo.Context,
    cam: Camera3D,
    base_pos: np.ndarray,
    radius_m: float,
    height_m: float,
    seed: int,
    fog_rgb: np.ndarray,
) -> None:
    """
    Handcrafted Japanese bamboo tea-picking basket (chatsumi-kago) with authentic alternating
    over-under split-bamboo weave shading, rolled bound-bamboo rim, soft feathered shadow, and
    a lush heaped dome of individually shaded curled fresh bright-green Uji tea leaves!
    """
    brng = np.random.default_rng(seed)
    ub, vb, zb = cam.project(base_pos)
    ut, vt, zt = cam.project(base_pos + np.array([0.0, 0.0, height_m]))
    if zb < Z_NEAR or zt < Z_NEAR:
        return

    dist = float(np.linalg.norm(base_pos - cam.eye))
    rx_b = radius_m * 0.80 * cam.K[0, 0] / float(zb)
    rx_t = radius_m * 1.06 * cam.K[0, 0] / float(zt)
    ry_t = rx_t * 0.34
    cx, cy_b, cy_t = float(ub), float(vb), float(vt)

    # Soft feathered radial contact shadow (never a hard cartoon oval!)
    ctx.save()
    ctx.translate(cx - rx_b * 0.20, cy_b + rx_b * 0.06)
    ctx.scale(1.36, 0.36)
    g_sh = cairo.RadialGradient(0, 0, rx_b * 0.12, 0, 0, rx_b * 1.25)
    g_sh.add_color_stop_rgba(0.0, 0.04, 0.07, 0.04, 0.62)
    g_sh.add_color_stop_rgba(0.55, 0.04, 0.07, 0.04, 0.28)
    g_sh.add_color_stop_rgba(1.0, 0.04, 0.07, 0.04, 0.0)
    ctx.set_source(g_sh)
    ctx.arc(0, 0, rx_b * 1.25, 0, math.tau)
    ctx.fill()
    ctx.restore()

    # Tapered woven basket body with alternating over-under bamboo weave cells!
    ctx.save()
    ctx.move_to(cx - rx_t, cy_t)
    ctx.curve_to(cx - rx_t * 0.96, cy_t + (cy_b - cy_t) * 0.55, cx - rx_b * 1.04, cy_b - ry_t * 0.2, cx - rx_b, cy_b)
    ctx.curve_to(cx - rx_b * 0.45, cy_b + ry_t * 0.78, cx + rx_b * 0.45, cy_b + ry_t * 0.78, cx + rx_b, cy_b)
    ctx.curve_to(cx + rx_b * 1.04, cy_b - ry_t * 0.2, cx + rx_t * 0.96, cy_t + (cy_b - cy_t) * 0.55, cx + rx_t, cy_t)
    ctx.close_path()

    # Deep shadowed interior backing behind weave crevices
    ctx.set_source_rgba(*fog_mix([0.16, 0.12, 0.07], dist, fog_rgb), 0.99)
    ctx.fill_preserve()
    ctx.clip()

    n_rows = 10
    n_cols = 14
    for r in range(n_rows):
        t0 = r / n_rows
        t1 = (r + 0.92) / n_rows
        y0 = cy_t + (cy_b - cy_t) * t0
        y1 = cy_t + (cy_b - cy_t) * t1
        rw0 = rx_t + (rx_b - rx_t) * t0
        rw1 = rx_t + (rx_b - rx_t) * t1
        for c in range(n_cols):
            u0 = (c / n_cols) * 2.0 - 1.0
            u1 = ((c + 0.92) / n_cols) * 2.0 - 1.0
            # Curved bamboo strip elevation bow across cylindrical basket front
            bow0 = math.cos(u0 * math.pi * 0.5) * ry_t * 0.52
            bow1 = math.cos(u1 * math.pi * 0.5) * ry_t * 0.52
            # Cylinder directional sunlight (stronger on right side toward morning sun!)
            um = 0.5 * (u0 + u1)
            cyl_lit = max(0.18, min(1.0, 0.48 + 0.52 * um))
            over_h = ((r + c) & 1) == 0
            b_tone = float(brng.uniform(-0.04, 0.04))
            if over_h:
                col = np.array([0.56 + b_tone, 0.43 + b_tone * 0.8, 0.24]) * (0.48 + 0.58 * cyl_lit)
            else:
                col = np.array([0.48 + b_tone, 0.37 + b_tone * 0.8, 0.20]) * (0.42 + 0.52 * cyl_lit)
            ctx.set_source_rgba(*fog_mix(np.clip(col, 0.0, 1.1), dist, fog_rgb), 0.96)
            ctx.move_to(cx + u0 * rw0, y0 + bow0)
            ctx.line_to(cx + u1 * rw0, y0 + bow1)
            ctx.line_to(cx + u1 * rw1, y1 + bow1)
            ctx.line_to(cx + u0 * rw1, y1 + bow0)
            ctx.close_path()
            ctx.fill()
    # Soft cylindrical morning sunlight wrap across the woven bamboo surface
    g_cyl = cairo.LinearGradient(cx - rx_t, cy_t, cx + rx_t, cy_t)
    g_cyl.add_color_stop_rgba(0.0, 0.06, 0.05, 0.03, 0.38)
    g_cyl.add_color_stop_rgba(0.65, 0.85, 0.72, 0.40, 0.10)
    g_cyl.add_color_stop_rgba(1.0, 0.08, 0.06, 0.04, 0.24)
    ctx.set_source(g_cyl)
    ctx.rectangle(cx - rx_t * 1.1, cy_t, rx_t * 2.2, cy_b - cy_t + ry_t)
    ctx.fill()
    ctx.restore()

    # Shadowed deep-green tea bed inside mouth
    ctx.save()
    ctx.translate(cx, cy_t)
    ctx.scale(rx_t * 0.95, ry_t * 0.95)
    ctx.set_source_rgba(*fog_mix([0.09, 0.18, 0.08], dist, fog_rgb), 0.98)
    ctx.arc(0, 0, 1.0, 0, math.tau)
    ctx.fill()
    ctx.restore()

    # Lush heaped dome of 125 overlapping glossy first-flush Uji tea leaves & tender buds!
    n_leaves = 125
    for lk in range(n_leaves):
        lf = lk / n_leaves
        ang = float(brng.uniform(0.0, math.tau))
        rr = float(brng.uniform(0.0, 0.96)) ** 0.72
        lx = cx + math.cos(ang) * rx_t * 0.92 * rr
        mound_h = (1.0 - rr * rr) * ry_t * 1.45
        ly = cy_t + math.sin(ang) * ry_t * 0.62 - mound_h + float(brng.uniform(-3.0, 2.5))
        ll = rx_t * float(brng.uniform(0.12, 0.23))
        lw = ll * float(brng.uniform(0.36, 0.48))
        rot = float(brng.uniform(0.0, math.pi))

        sun_l = max(0.0, min(1.0, 0.25 + 0.58 * lf + 0.30 * math.cos(ang)))
        c_leaf = np.array([0.10, 0.25, 0.11]) * (1.0 - sun_l) + np.array([0.52, 0.74, 0.26]) * sun_l
        if sun_l > 0.76:
            c_leaf = c_leaf * 0.60 + np.array([0.84, 0.92, 0.46]) * 0.40
        ctx.save()
        ctx.translate(lx, ly)
        ctx.rotate(rot)
        ctx.set_source_rgba(*fog_mix(c_leaf, dist, fog_rgb), 0.96)
        ctx.move_to(-ll * 0.52, 0)
        ctx.curve_to(-ll * 0.15, -lw, ll * 0.28, -lw * 0.80, ll * 0.54, 0)
        ctx.curve_to(ll * 0.28, lw * 0.80, -ll * 0.15, lw, -ll * 0.52, 0)
        ctx.close_path()
        ctx.fill()
        # Subtle bright leaf midrib highlight on sunlit top leaves
        if sun_l > 0.65 and ll > 6.5:
            ctx.set_source_rgba(*fog_mix([0.88, 0.95, 0.58], dist, fog_rgb), 0.55)
            ctx.set_line_width(0.85)
            ctx.move_to(-ll * 0.32, 0)
            ctx.line_to(ll * 0.36, 0)
            ctx.stroke()
        ctx.restore()

    # Thick rolled split-bamboo bound rim hoop with cross-lashing marks
    ctx.save()
    ctx.translate(cx, cy_t)
    ctx.scale(rx_t, ry_t)
    ctx.set_source_rgba(*fog_mix([0.24, 0.17, 0.09], dist, fog_rgb), 0.98)
    ctx.set_line_width(max(0.11, 5.8 / max(10.0, rx_t)))
    ctx.arc(0.0, 0.0, 1.0, 0.0, math.pi)
    ctx.stroke()
    ctx.set_source_rgba(*fog_mix([0.66, 0.52, 0.28], dist, fog_rgb), 0.96)
    ctx.set_line_width(max(0.07, 3.4 / max(10.0, rx_t)))
    ctx.arc(0.0, -0.04, 0.99, 0.04, math.pi - 0.04)
    ctx.stroke()
    ctx.restore()


def draw_burlap_tea_sacks_and_bench(ctx: cairo.Context, cam: Camera3D, fog_rgb: np.ndarray) -> None:
    """Rustic cedar mill bench + folded textured burlap tea-leaf sacks (asabukuro) under porch."""
    draw_3d_box_beam(ctx, cam, mill_pt(-2.15, -0.68, 0.46), mill_pt(-0.42, -0.68, 0.46), 0.22, 0.042, (0.31, 0.25, 0.18), fog_rgb)
    for bu in (-1.95, -0.62):
        draw_3d_box_beam(ctx, cam, mill_pt(bu, -0.68, Z_PORCH), mill_pt(bu, -0.68, 0.44), 0.055, 0.16, (0.21, 0.17, 0.13), fog_rgb)

    sacks = [
        (mill_pt(-2.72, -0.65, Z_PORCH), 0.30, 0.64, 501),
        (mill_pt(-2.34, -0.52, Z_PORCH), 0.27, 0.58, 502),
    ]
    for sp, rad, ht, ssd in sacks:
        srng = np.random.default_rng(ssd)
        ub, vb, zb = cam.project(sp)
        ut, vt, zt = cam.project(sp + np.array([0.0, 0.0, ht]))
        if zb < Z_NEAR or zt < Z_NEAR:
            continue
        dist = float(np.linalg.norm(sp - cam.eye))
        rx = rad * cam.K[0, 0] / float(zb)
        cx, cy_b, cy_t = float(ub), float(vb), float(vt)
        # Soft pinched sack body with bunched cloth collar ears on top
        ctx.save()
        ctx.move_to(cx - rx * 0.88, cy_b)
        ctx.curve_to(cx - rx * 1.18, cy_b - (cy_b - cy_t) * 0.36, cx - rx * 0.86, cy_t + (cy_b - cy_t) * 0.24, cx - rx * 0.28, cy_t + rx * 0.22)
        ctx.line_to(cx - rx * 0.44, cy_t - rx * 0.05)
        ctx.line_to(cx + rx * 0.04, cy_t + rx * 0.08)
        ctx.line_to(cx + rx * 0.42, cy_t - rx * 0.06)
        ctx.line_to(cx + rx * 0.28, cy_t + rx * 0.22)
        ctx.curve_to(cx + rx * 0.86, cy_t + (cy_b - cy_t) * 0.24, cx + rx * 1.18, cy_b - (cy_b - cy_t) * 0.36, cx + rx * 0.88, cy_b)
        ctx.curve_to(cx + rx * 0.35, cy_b + rx * 0.24, cx - rx * 0.35, cy_b + rx * 0.24, cx - rx * 0.88, cy_b)
        ctx.close_path()
        g = cairo.LinearGradient(cx - rx, cy_t, cx + rx, cy_b)
        g.add_color_stop_rgba(0.0, *fog_mix([0.28, 0.23, 0.16], dist, fog_rgb), 0.99)
        g.add_color_stop_rgba(0.55, *fog_mix([0.48, 0.40, 0.28], dist, fog_rgb), 0.99)
        g.add_color_stop_rgba(0.85, *fog_mix([0.56, 0.46, 0.31], dist, fog_rgb), 0.99)
        g.add_color_stop_rgba(1.0, *fog_mix([0.24, 0.20, 0.14], dist, fog_rgb), 0.99)
        ctx.set_source(g)
        ctx.fill_preserve()
        ctx.clip()
        # Woven jute cloth folds & subtle cross-hatch sackcloth texture
        for fk in range(-2, 3):
            ctx.set_source_rgba(*fog_mix([0.18, 0.14, 0.09], dist, fog_rgb), 0.38)
            ctx.set_line_width(max(1.2, rx * 0.045))
            ctx.move_to(cx + fk * rx * 0.12, cy_t + rx * 0.22)
            ctx.curve_to(cx + fk * rx * 0.26, cy_t + (cy_b - cy_t) * 0.55, cx + fk * rx * 0.34, cy_b - rx * 0.2, cx + fk * rx * 0.30, cy_b)
            ctx.stroke()
        ctx.restore()
        # Hemp twine tie cord knotted around sack neck
        ctx.set_source_rgba(*fog_mix([0.22, 0.16, 0.09], dist, fog_rgb), 0.96)
        ctx.set_line_width(max(2.2, rx * 0.09))
        ctx.move_to(cx - rx * 0.32, cy_t + rx * 0.23)
        ctx.line_to(cx + rx * 0.32, cy_t + rx * 0.23)
        ctx.stroke()


# =============================================================================
# Stage 3 Band 3 — Razor-Tapered Cubic-Bezier Meadow Sedge & Highland Tea Flora
# =============================================================================

def draw_tapered_sedge_blade(
    ctx: cairo.Context,
    bx: float,
    by: float,
    c1x: float,
    c1y: float,
    c2x: float,
    c2y: float,
    ex: float,
    ey: float,
    base_half_w: float,
    c_base: tuple[float, float, float],
    c_tip: tuple[float, float, float],
) -> None:
    """Silky anti-aliased tapered cubic-Bezier meadow sedge grass blade (zero sub-pixel aliasing!)."""
    tx = ex - bx
    ty = ey - by
    tl = math.hypot(tx, ty) + 1e-6
    nx, ny = -ty / tl, tx / tl
    tip_w = max(0.45, base_half_w * 0.18)
    g = cairo.LinearGradient(bx, by, ex, ey)
    g.add_color_stop_rgba(0.0, *c_base, 0.96)
    g.add_color_stop_rgba(0.68, *c_tip, 0.95)
    g.add_color_stop_rgba(1.0, *c_tip, 0.85)
    ctx.set_source(g)
    ctx.move_to(bx - nx * base_half_w, by - ny * base_half_w)
    ctx.curve_to(c1x - nx * base_half_w * 0.74, c1y - ny * base_half_w * 0.74, c2x - nx * base_half_w * 0.38, c2y - ny * base_half_w * 0.38, ex - nx * tip_w, ey - ny * tip_w)
    ctx.line_to(ex + nx * tip_w, ey + ny * tip_w)
    ctx.curve_to(c2x + nx * base_half_w * 0.38, c2y + ny * base_half_w * 0.38, c1x + nx * base_half_w * 0.74, c1y + ny * base_half_w * 0.74, bx + nx * base_half_w, by + ny * base_half_w)
    ctx.close_path()
    ctx.fill()


def draw_white_tea_blossom(ctx: cairo.Context, x: float, y: float, r: float, srng: np.random.Generator, alpha: float = 0.96) -> None:
    """Delicate 5-petal Japanese highland green-tea blossom (cha-no-hana) with golden stamen cluster."""
    rot0 = float(srng.uniform(0.0, math.tau))
    for pk in range(5):
        a = rot0 + pk * (math.tau / 5.0) + float(srng.uniform(-0.08, 0.08))
        px = x + math.cos(a) * r * 0.56
        py = y + math.sin(a) * r * 0.56
        ctx.save()
        ctx.translate(px, py)
        ctx.rotate(a)
        ctx.scale(r * 0.62, r * 0.50)
        g = cairo.RadialGradient(0.15, -0.15, 0.1, 0.0, 0.0, 1.0)
        g.add_color_stop_rgba(0.0, 0.99, 0.99, 0.95, alpha)
        g.add_color_stop_rgba(0.75, 0.92, 0.94, 0.86, alpha)
        g.add_color_stop_rgba(1.0, 0.76, 0.82, 0.72, alpha * 0.92)
        ctx.set_source(g)
        ctx.arc(0, 0, 1.0, 0, math.tau)
        ctx.fill()
        ctx.restore()
    ctx.set_source_rgba(0.92, 0.74, 0.22, alpha)
    ctx.arc(x, y, max(1.2, r * 0.34), 0, math.tau)
    ctx.fill()
    ctx.set_source_rgba(1.00, 0.94, 0.58, alpha * 0.90)
    for sk in range(5):
        sa = sk * 1.25 + rot0
        ctx.arc(x + math.cos(sa) * r * 0.22, y + math.sin(sa) * r * 0.22, max(0.7, r * 0.12), 0, math.tau)
        ctx.fill()


def draw_meadow_sedge_and_tea_shrubs(ctx: cairo.Context, cam: Camera3D, fog_rgb: np.ndarray) -> None:
    """
    Stage 3 Band 3: Dewy highland meadow sedge blades (razor-tapered cubic Beziers!),
    lush serrated glossy green-tea shrubs with soft feathered ground shadows (ZERO dark ovals!),
    and delicate white highland tea blossoms along the stone footpath & dry-stone retaining walls!
    """
    shrub_clusters = [
        (np.array([float(path_right_x(3.3)) - 0.08, 3.30, Z_PATH - 0.03]), 0.52, True,  4101),
        (np.array([float(path_right_x(5.1)) - 0.08, 5.10, Z_PATH - 0.04]), 0.58, True,  4102),
        (np.array([float(path_right_x(7.3)) - 0.06, 7.30, Z_PATH - 0.05]), 0.64, True,  4103),
        (np.array([float(path_right_x(10.2)) - 0.06, 10.2, Z_PATH - 0.06]), 0.68, False, 4104),
        (np.array([float(path_right_x(13.5)) - 0.08, 13.5, Z_PATH - 0.07]), 0.72, False, 4105),
        # Soft sunlit tea-bush crest wrapping naturally around the far footpath bend (ZERO floating cut edge!)
        (np.array([float(millrace_right_x(18.5)) + 0.65, 18.5, Z_PATH - 0.06]), 0.92, False, 4106),
        (np.array([float(millrace_right_x(20.0)) + 0.15, 20.0, Z_PATH - 0.08]), 0.98, False, 4107),
        # Left foreground mossy stone retaining wall & mill-yard clusters
        (np.array([float(millrace_left_x(3.6)) - 0.42, 3.60, Z_PATH + 0.06]), 0.50, True, 4111),
        (np.array([float(millrace_left_x(4.9)) - 0.58, 4.90, Z_PATH + 0.06]), 0.54, True, 4112),
    ]

    for pos, sc, has_fl, sd in shrub_clusters:
        srng = np.random.default_rng(sd)
        ux, uy, uz = cam.project(pos)
        if uz < Z_NEAR:
            continue
        dist = float(np.linalg.norm(pos - cam.eye))
        px_r = sc * cam.K[0, 0] / float(uz)
        if px_r < 8.0:
            continue

        # Soft feathered multi-stop radial ground shadow (never a dark sharp circle!)
        ctx.save()
        ctx.translate(float(ux) - px_r * 0.22, float(uy) + px_r * 0.12)
        ctx.scale(1.42, 0.34)
        g_s = cairo.RadialGradient(0, 0, px_r * 0.10, 0, 0, px_r * 0.96)
        g_s.add_color_stop_rgba(0.0, *fog_mix([0.04, 0.09, 0.05], dist, fog_rgb), 0.52)
        g_s.add_color_stop_rgba(0.55, *fog_mix([0.04, 0.09, 0.05], dist, fog_rgb), 0.22)
        g_s.add_color_stop_rgba(1.0, *fog_mix([0.04, 0.09, 0.05], dist, fog_rgb), 0.0)
        ctx.set_source(g_s)
        ctx.arc(0, 0, px_r * 0.96, 0, math.tau)
        ctx.fill()
        ctx.restore()

        # Razor-tapered arching cubic-Bezier meadow sedge blades
        n_blades = int(min(34, max(12, px_r * 0.26)))
        for _ in range(n_blades):
            bx = float(ux) + float(srng.uniform(-px_r * 0.88, px_r * 0.88))
            by = float(uy) + float(srng.uniform(-px_r * 0.08, px_r * 0.18))
            lean = float(srng.uniform(-0.92, 0.98))
            blen = px_r * float(srng.uniform(0.52, 1.38))
            c1x = bx + lean * blen * 0.32
            c1y = by - blen * 0.56
            c2x = bx + lean * blen * 0.76
            c2y = by - blen * 0.94
            ex = bx + lean * blen * 1.15
            ey = by - blen * float(srng.uniform(0.64, 0.96))
            is_sun = lean > -0.15
            draw_tapered_sedge_blade(
                ctx, bx, by, c1x, c1y, c2x, c2y, ex, ey,
                base_half_w=max(1.7, px_r * float(srng.uniform(0.020, 0.034))),
                c_base=fog_mix([0.07, 0.17, 0.08], dist, fog_rgb),
                c_tip=fog_mix([0.52, 0.72, 0.28] if is_sun else [0.20, 0.39, 0.16], dist, fog_rgb),
            )

        # Multi-layered glossy serrated Uji tea leaves forming an organic bush crown
        n_lf = int(min(96, max(28, px_r * 0.72)))
        for k in range(n_lf):
            kf = k / n_lf
            ang = float(srng.uniform(-math.pi, 0.12))
            rad = px_r * (float(srng.uniform(0.06, 0.86)) ** 0.82)
            lx = float(ux) + math.cos(ang) * rad * 1.15
            ly = float(uy) + math.sin(ang) * rad * 0.70 - px_r * 0.14
            ll = px_r * float(srng.uniform(0.11, 0.22))
            lw = ll * 0.40
            lrot = float(srng.uniform(-1.15, 1.15))
            sun_f = max(0.0, min(1.0, 0.20 + 0.58 * kf + 0.34 * math.cos(ang)))
            c_lf = np.array([0.09, 0.21, 0.10]) * (1.0 - sun_f) + np.array([0.46, 0.67, 0.25]) * sun_f
            if sun_f > 0.76:
                c_lf = c_lf * 0.62 + np.array([0.78, 0.88, 0.42]) * 0.38
            ctx.save()
            ctx.translate(lx, ly)
            ctx.rotate(lrot)
            ctx.set_source_rgba(*fog_mix(c_lf, dist, fog_rgb), 0.96)
            ctx.move_to(-ll * 0.5, 0)
            ctx.curve_to(-ll * 0.15, -lw, ll * 0.25, -lw * 0.82, ll * 0.52, 0)
            ctx.curve_to(ll * 0.25, lw * 0.82, -ll * 0.15, lw, -ll * 0.5, 0)
            ctx.close_path()
            ctx.fill()
            ctx.restore()

        if has_fl and px_r > 22.0:
            n_bl = int(srng.integers(3, 6))
            for _ in range(n_bl):
                ba = float(srng.uniform(-math.pi * 0.86, -0.14))
                br = px_r * float(srng.uniform(0.16, 0.66))
                draw_white_tea_blossom(
                    ctx,
                    float(ux) + math.cos(ba) * br * 1.05,
                    float(uy) + math.sin(ba) * br * 0.64 - px_r * 0.15,
                    max(4.0, px_r * float(srng.uniform(0.08, 0.12))),
                    srng,
                )


# =============================================================================
# Stage 2 — Crystal Mountain Millrace Stream with Fresnel Reflections & Foam
# =============================================================================

def paint_millrace_fresnel_water(
    canvas: np.ndarray,
    ref_backdrop: np.ndarray,
    mirrored_3d_rgba: np.ndarray,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    mottle_full: np.ndarray,
) -> tuple[np.ndarray, list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """
    Stage 2: Clear jade-turquoise mountain millrace stream (z = -0.38m) combining cv2.remap
    flowing capillary ripples (y_src = 2 * horizon_y - y) with 3D plane-mirrored quads of the
    spoked waterwheel, angled cedar mill wall, glowing window, mossy roof & stone foot-slab,
    weighted by fresnel_min + (fresnel_max - fresnel_min) * (1 - dist_t) ** 1.4!
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    rx_c = (xx - cam.K[0, 2]) / cam.K[0, 0]
    ry_c = (yy - cam.K[1, 2]) / cam.K[1, 1]
    RT = cam.R.T
    dx_w = RT[0, 0] * rx_c + RT[0, 1] * ry_c + RT[0, 2]
    dy_w = RT[1, 0] * rx_c + RT[1, 1] * ry_c + RT[1, 2]
    dz_w = RT[2, 0] * rx_c + RT[2, 1] * ry_c + RT[2, 2]

    hit = dz_w < -0.004
    t_w = np.where(hit, (Z_STREAM - cam.eye[2]) / np.where(hit, dz_w, -1.0), -1.0)
    wx = cam.eye[0] + dx_w * t_w
    wy = cam.eye[1] + dy_w * t_w

    xl = millrace_left_x(wy)
    xr = millrace_right_x(wy)
    valid = hit & (t_w > Z_NEAR) & (wy > 1.2) & (wy < 22.5) & (wx >= xl - 0.16) & (wx <= xr + 0.16)
    stream_mask = (
        valid.astype(np.float32)
        * np.clip((wx - (xl - 0.16)) / 0.10, 0.0, 1.0).astype(np.float32)
        * np.clip((xr + 0.16 - wx) / 0.10, 0.0, 1.0).astype(np.float32)
        * np.clip((21.0 - wy) / 3.6, 0.0, 1.0).astype(np.float32)
    )

    left_uvs, right_uvs = [], []
    for y_p in (3.0, 4.5, 6.2, 8.6, 11.2, 14.5, 18.0):
        ul, vl, zl = cam.project(np.array([float(millrace_left_x(y_p)), y_p, Z_STREAM]))
        ur, vr, zr = cam.project(np.array([float(millrace_right_x(y_p)), y_p, Z_STREAM]))
        if zl > Z_NEAR:
            left_uvs.append((y_p, float(ul) / W, float(vl) / H))
        if zr > Z_NEAR:
            right_uvs.append((y_p, float(ur) / W, float(vr) / H))

    if not np.any(stream_mask > 1e-4):
        return canvas, left_uvs, right_uvs

    rip1 = resize_bicubic(rng.standard_normal((H // 6, W // 42)).astype(np.float32), W, H)
    rip2 = resize_bicubic(rng.standard_normal((H // 3, W // 18)).astype(np.float32), W, H)
    rip3 = resize_bicubic(rng.standard_normal((H // 12, W // 80)).astype(np.float32), W, H)
    rip = 0.48 * rip1 + 0.34 * rip2 + 0.18 * rip3

    dist_t = np.clip((yy.astype(np.float32) - float(cam.horizon_y)) / max(1.0, H - float(cam.horizon_y)), 0.0, 1.0)
    rip_scale = (1.6 + 7.4 * dist_t).astype(np.float32)

    map_x = (xx.astype(np.float32) + rip * 1.05 * rip_scale).astype(np.float32)
    map_y = np.clip((2.0 * float(cam.horizon_y) - yy.astype(np.float32)) + rip * 1.35 * rip_scale, 0.0, H - 1.001).astype(np.float32)

    refl_bg = cv2.remap(ref_backdrop.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mx_3d = (xx.astype(np.float32) + rip * 1.15 * rip_scale).astype(np.float32)
    my_3d = (yy.astype(np.float32) + rip * 0.85 * rip_scale).astype(np.float32)
    refl_3d = cv2.remap(
        mirrored_3d_rgba.astype(np.float32), mx_3d, my_3d, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )
    refl_all = over_premul(refl_bg, refl_3d) * np.array([0.82, 0.94, 0.89], dtype=np.float32)[None, None, :]

    d_edge = np.minimum(np.maximum(0.0, wx - xl), np.maximum(0.0, xr - wx)).astype(np.float32)
    shallow = np.clip(1.0 - d_edge / 0.50, 0.0, 1.0)
    peb = np.sin(wx * 21.0 + mottle_full * 2.4) * np.cos(wy * 18.0 - mottle_full * 2.0)
    bed_deep = np.array([0.068, 0.245, 0.228], dtype=np.float32)
    bed_shal = np.array([0.185, 0.395, 0.320], dtype=np.float32)
    bed_col = bed_deep[None, None, :] * (1.0 - 0.58 * shallow[..., None]) + bed_shal[None, None, :] * (0.58 * shallow[..., None])
    bed_col += (mottle_full * 0.032 + peb.astype(np.float32) * 0.020)[..., None] * np.array([0.45, 0.88, 0.74], dtype=np.float32)

    fresnel = 0.24 + (0.86 - 0.24) * np.power(1.0 - dist_t, 1.4)
    water_rgb = bed_col * (1.0 - fresnel[..., None]) + refl_all * fresnel[..., None]

    sun_w = trace_valley_sunlight(wx, wy, np.full_like(wx, Z_STREAM))
    water_rgb *= (0.62 + 0.38 * sun_w)[..., None]
    glint = np.clip((rip - 0.74) * 2.5, 0.0, 1.0) * sun_w
    water_rgb += glint[..., None] * np.array([0.62, 0.68, 0.50], dtype=np.float32)

    # Shimmering warm golden amber reflection beneath the glowing side tea-drying window!
    ref_win = mill_pt(0.55, 4.65, Z_STREAM)
    dx_w = (wx - ref_win[0]) / 0.36
    dy_w = (wy - ref_win[1]) / 1.55
    w_streak = np.exp(-dx_w * dx_w * 2.0 - dy_w * dy_w * 0.85).astype(np.float32)
    water_rgb += (w_streak * np.clip(0.55 + 0.90 * rip, 0.0, 1.7) * 0.44)[..., None] * np.array([1.00, 0.80, 0.34], dtype=np.float32)

    # Flowing crisp white foam wake & aerated splash turbulence downstream from waterwheel paddles!
    wheel_sp = mill_pt(0.38, WHEEL_L_CENTER - 0.20, Z_STREAM)
    u_ml, l_ml = world_to_mill_local(wx, wy)
    d_sp = np.hypot((u_ml - 0.40) / 0.42, (l_ml - (WHEEL_L_CENTER - 0.25)) / 1.05).astype(np.float32)
    foam_core = np.exp(-d_sp * d_sp * 1.5) * np.clip(0.58 + 0.55 * mottle_full + 0.45 * rip, 0.0, 1.35)
    wake_env = np.clip((WHEEL_L_CENTER - l_ml) / 0.8, 0.0, 1.0) * np.exp(-np.maximum(0.0, WHEEL_L_CENTER - l_ml) / 4.4)
    wake_u = 0.46 + 0.14 * np.sin(l_ml * 1.2)
    wake_str = np.exp(-((u_ml - wake_u) / 0.36) ** 2) * wake_env.astype(np.float32)
    foam_rib = wake_str * np.clip((rip * 0.72 + mottle_full * 0.48 - 0.10) * 1.85, 0.0, 1.0)
    foam_all = np.clip(foam_core * 0.78 + foam_rib * 0.64, 0.0, 0.92)
    foam_col = np.array([0.93, 0.98, 0.95], dtype=np.float32) * (0.76 + 0.24 * sun_w[..., None])
    water_rgb = water_rgb * (1.0 - foam_all[..., None]) + foam_col * foam_all[..., None]

    wall_ao = np.exp(-d_edge / 0.22) * 0.42
    water_rgb *= (1.0 - wall_ao)[..., None]

    fog_t = 1.0 - np.exp(-np.maximum(0.0, t_w.astype(np.float32) - 6.8) * 0.036)
    water_rgb = water_rgb * (1.0 - fog_t[..., None]) + fog_rgb[None, None, :] * fog_t[..., None]
    out = canvas * (1.0 - stream_mask[..., None]) + water_rgb * stream_mask[..., None]
    return out, left_uvs, right_uvs


# =============================================================================
# Stage 2 & 3 — Diagonal Golden Dawn Sunbeams Strictly Aligned With SUN_DIR
# =============================================================================

def paint_valley_dawn_sunbeams(
    cam: Camera3D, rng: np.random.Generator, intensity: float = 0.34
) -> tuple[np.ndarray, tuple[float, float]]:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    p_ref = np.array([1.2, 11.5, 2.0], dtype=np.float64)
    u0, v0, _ = cam.project(p_ref)
    u1, v1, _ = cam.project(p_ref + SUN_DIR * 3.0)
    dx, dy = float(u1 - u0), float(v1 - v0)
    dl = math.hypot(dx, dy) + 1e-6
    rx, ry = dx / dl, dy / dl
    nx, ny = -ry, rx

    ox, oy = W * 0.85, H * 0.12
    s_coord = (xx - ox) * rx + (yy - oy) * ry
    p_coord = (xx - ox) * nx + (yy - oy) * ny

    acc = np.zeros((H, W), dtype=np.float32)
    shafts = [(-380, 74, 0.36), (-165, 92, 0.66), (55, 84, 0.74), (265, 95, 0.56), (485, 82, 0.32)]
    for cp, wp, st in shafts:
        w_eff = wp * (1.0 + np.maximum(0.0, s_coord) / (H * 1.45))
        beam = np.exp(-0.5 * ((p_coord - cp) / w_eff) ** 2)
        vert = np.clip((s_coord + 220.0) / 260.0, 0.0, 1.0) * np.exp(-np.maximum(0.0, s_coord) / (H * 1.30))
        acc += beam * vert * st

    turb = 0.78 + 0.22 * resize_bicubic(fbm_gouache(H // 4, W // 4, rng), W, H)
    acc = np.clip(acc * turb * intensity, 0.0, 1.0)
    col = np.array([0.98, 0.96, 0.76], dtype=np.float32)
    return acc[..., None] * col[None, None, :], (rx, ry)


# =============================================================================
# Stage 4 — Foreground Circular-Disc Bokeh Tea Branch & 35mm Film Optics
# =============================================================================

def make_disc_kernel(radius: float = 7.8) -> np.ndarray:
    rad_i = int(math.ceil(radius + 1.5))
    yy, xx = np.mgrid[-rad_i : rad_i + 1, -rad_i : rad_i + 1].astype(np.float32)
    r = np.hypot(xx, yy)
    disc = np.clip((radius - r + 0.65) / 1.15, 0.0, 1.0)
    rim = np.exp(-((r - radius * 0.86) / (radius * 0.16)) ** 2) * 0.26 * disc
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


def paint_foreground_bokeh_tea_twig() -> np.ndarray:
    """
    Stage 4: Near-camera dewy tea/camellia twig with glossy backlit emerald leaves & two
    soft white tea blossoms arching gracefully into the lower-right corner, convolved in
    premultiplied RGBA with make_disc_kernel(radius=7.8) + separate cv2.warpAffine ~1px
    opposite R/B lateral shift!
    """
    surf, ctx = new_cairo_2x()
    srng = np.random.default_rng(9104)

    twigs = [
        [(W + 24, H - 90),  (W - 110, H - 118), (W - 248, H - 82),  (W - 365, H - 22)],
        [(W - 15, H + 42),  (W - 98,  H - 64),  (W - 180, H - 154), (W - 255, H - 210)],
    ]
    for pts in twigs:
        ctx.set_source_rgba(0.13, 0.15, 0.09, 0.96)
        ctx.set_line_width(7.0)
        ctx.move_to(*pts[0])
        ctx.curve_to(pts[1][0], pts[1][1], pts[2][0], pts[2][1], pts[3][0], pts[3][1])
        ctx.stroke()

        for k in range(1, 10):
            tf = k / 10.0
            omt = 1.0 - tf
            bx = omt**3 * pts[0][0] + 3 * omt**2 * tf * pts[1][0] + 3 * omt * tf**2 * pts[2][0] + tf**3 * pts[3][0]
            by = omt**3 * pts[0][1] + 3 * omt**2 * tf * pts[1][1] + 3 * omt * tf**2 * pts[2][1] + tf**3 * pts[3][1]
            for side in (-1.0, 1.0):
                ang = -2.35 + side * float(srng.uniform(0.55, 1.05))
                llen = float(srng.uniform(68.0, 112.0)) * (1.1 - 0.35 * tf)
                lwid = llen * 0.36
                ctx.save()
                ctx.translate(bx, by)
                ctx.rotate(ang)
                g = cairo.LinearGradient(0, 0, llen, 0)
                lit = 0.72 if side > 0 else 0.34
                g.add_color_stop_rgba(0.0, 0.07 + 0.08 * lit, 0.19 + 0.14 * lit, 0.08 + 0.04 * lit, 0.96)
                g.add_color_stop_rgba(0.75, 0.22 + 0.32 * lit, 0.42 + 0.30 * lit, 0.13 + 0.14 * lit, 0.96)
                g.add_color_stop_rgba(1.0, 0.54 + 0.28 * lit, 0.72 + 0.18 * lit, 0.24 + 0.14 * lit, 0.94)
                ctx.set_source(g)
                ctx.move_to(0, 0)
                ctx.curve_to(llen * 0.28, -lwid, llen * 0.74, -lwid * 0.65, llen, 0)
                ctx.curve_to(llen * 0.74, lwid * 0.65, llen * 0.28, lwid, 0, 0)
                ctx.close_path()
                ctx.fill()
                ctx.set_source_rgba(0.95, 0.98, 0.85, 0.52)
                ctx.arc(llen * 0.48, -lwid * 0.18, 4.0, 0, math.tau)
                ctx.fill()
                ctx.restore()

    draw_white_tea_blossom(ctx, W - 165, H - 112, 28.0, srng, alpha=0.95)
    draw_white_tea_blossom(ctx, W - 268, H - 62, 22.0, srng, alpha=0.94)

    fg_1x = finish_cairo_2x(surf)
    blurred = convolve_fft_2d_bbox(fg_1x, make_disc_kernel(radius=7.8))

    out = blurred.copy()
    M_r = np.array([[1.0, 0.0, 1.05], [0.0, 1.0, 0.0]], dtype=np.float32)
    M_b = np.array([[1.0, 0.0, -1.05], [0.0, 1.0, 0.0]], dtype=np.float32)
    out[:, :, 0] = cv2.warpAffine(blurred[:, :, 0], M_r, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    out[:, :, 2] = cv2.warpAffine(blurred[:, :, 2], M_b, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return np.clip(out, 0.0, 1.0)


def apply_35mm_finish(
    rgb: np.ndarray, rng: np.random.Generator, window_uvs: list[tuple[float, float]]
) -> np.ndarray:
    img = rgb.astype(np.float32).copy()

    # 1. 3-scale warm cream-mint highlight halation (sigma = 6.5, 24, 72 px above lum > 0.72)
    lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    hl_mask = np.clip((lum - 0.72) / 0.28, 0.0, 1.5)
    hl_src = img * hl_mask[..., None]

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for idx_w, (wu, wv) in enumerate(window_uvs):
        rw = 22.0 if idx_w == 0 else 19.0
        sw = 0.44 if idx_w == 0 else 0.38
        d2 = ((xx - wu * W) / rw) ** 2 + ((yy - wv * H) / (rw * 0.85)) ** 2
        hl_src += (np.exp(-d2) * sw)[..., None] * np.array([1.00, 0.84, 0.38], dtype=np.float32)

    tint = np.array([1.03, 1.04, 0.90], dtype=np.float32)
    hal = (
        0.20 * gaussian_blur_2d(hl_src, 6.5)
        + 0.14 * gaussian_blur_2d(hl_src, 24.0)
        + 0.09 * gaussian_blur_2d(hl_src, 72.0)
    ) * tint[None, None, :]
    img = 1.0 - (1.0 - np.clip(img, 0.0, 1.0)) * (1.0 - np.clip(hal, 0.0, 0.82))

    # 2. Radial corner CA + gentle optical vignette
    nx = (xx - W * 0.5) / (W * 0.5)
    ny = (yy - H * 0.5) / (H * 0.5)
    r2 = nx * nx + ny * ny
    ca_scale = 0.00095
    img[:, :, 0] = cv2.remap(
        img[:, :, 0],
        (xx + nx * r2 * (W * ca_scale)).astype(np.float32),
        (yy + ny * r2 * (H * ca_scale)).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    img[:, :, 2] = cv2.remap(
        img[:, :, 2],
        (xx - nx * r2 * (W * ca_scale)).astype(np.float32),
        (yy - ny * r2 * (H * ca_scale)).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    img *= np.clip(1.0 - 0.17 * np.power(r2, 1.15), 0.75, 1.0)[..., None]

    # 3. Shadow black lift + C1-continuous tanh highlight shoulder (knee=0.80, ceiling=0.965)
    img = 0.012 + img * 0.988
    knee, ceiling = 0.80, 0.965
    span = ceiling - knee
    img = np.where(img > knee, knee + span * np.tanh((img - knee) / span), img)

    # 4. Luminance-weighted silver-halide grain
    lum2 = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    env = np.clip(1.10 - 0.94 * np.power(np.clip(lum2, 0.0, 1.0), 1.55), 0.12, 1.10)
    grain = rng.normal(0.0, 0.0095, size=(H, W, 3)).astype(np.float32)
    mono_g = rng.normal(0.0, 0.0078, size=(H, W, 1)).astype(np.float32)
    img = np.clip(img + (0.65 * mono_g + 0.35 * grain) * env[..., None], 0.0, 1.0)

    # 5. Faint YIQ analog transfer patina (1.0 px horizontal chroma softening, BORDER_REFLECT_101, <= 0.75% scanlines)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    I = 0.596 * R - 0.274 * G - 0.322 * B
    Q = 0.211 * R - 0.523 * G + 0.312 * B

    I_soft = cv2.GaussianBlur(I, (7, 1), sigmaX=1.0, sigmaY=0.0, borderType=cv2.BORDER_REFLECT_101)
    Q_soft = cv2.GaussianBlur(Q, (7, 1), sigmaX=1.0, sigmaY=0.0, borderType=cv2.BORDER_REFLECT_101)
    Y_ring = Y + 0.032 * (Y - cv2.GaussianBlur(Y, (9, 1), sigmaX=1.1, sigmaY=0.0, borderType=cv2.BORDER_REFLECT_101))

    out = np.empty_like(img)
    out[:, :, 0] = Y_ring + 0.956 * I_soft + 0.621 * Q_soft
    out[:, :, 1] = Y_ring - 0.272 * I_soft - 0.647 * Q_soft
    out[:, :, 2] = Y_ring - 1.106 * I_soft + 1.703 * Q_soft

    line_shift = cv2.GaussianBlur(
        rng.normal(0.0, 0.14, size=(H, 1)).astype(np.float32),
        (1, 9),
        sigmaX=0.0,
        sigmaY=1.2,
        borderType=cv2.BORDER_REFLECT_101,
    )
    jx = (np.broadcast_to(np.arange(W, dtype=np.float32)[None, :], (H, W)) + line_shift).astype(np.float32)
    jy = np.broadcast_to(np.arange(H, dtype=np.float32)[:, None], (H, W)).astype(np.float32)
    out = cv2.remap(out, jx, jy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    out *= (1.0 - 0.0065 * (0.5 - 0.5 * np.cos(np.arange(H, dtype=np.float32) * math.pi)))[:, None, None]
    return np.clip(out, 0.0, 1.0)


# =============================================================================
# Full 5-Stage Scene Assembly, UV Telemetry Printout & Deterministic Export
# =============================================================================

def render_scene(
    seed: int = 202622,
    audit_dir: pathlib.Path | None = None,
    export_webps: list[pathlib.Path] | None = None,
) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # Stage 1: Off-axis Camera3D (VP ~60.0% width, horizon ~43.5% height)
    # Left showcases full mossy cedar watermill roof, open spoked 3/4 waterwheel & crystal millrace;
    # right opens broadly across undulating sunlit emerald Uji terraced green tea plantations!
    cam = Camera3D(W, H, eye=(1.55, 0.0, 1.72), target=(0.22, 12.0, 1.24), fov_x_deg=58.0)
    print(f"[matcha-02] Camera VP: ({cam.vp_x:.1f}, {cam.vp_y:.1f}) -> {cam.vp_x / W * 100:.2f}% width, horizon_v={cam.horizon_y / H:.4f}")

    # Stage 3 Band 0: Dawn highland sky, sculptural clouds, 4 misty blue-jade ridges & silver-mint sea of clouds
    canvas, fog_rgb, m_global, d_global = paint_dawn_valley_backdrop(cam, rng)

    # Stage 3 Band 1: Volumetric sculpted Uji terraced green tea plantations + distant tea shed & feathered cypresses
    band1 = paint_terraced_tea_valley_band(cam, rng, fog_rgb, m_global, d_global)
    canvas = over_premul(canvas, band1)

    # Snapshot horizon reflection backdrop BEFORE painting crystal millrace water
    ref_backdrop = canvas.copy()

    # Pre-paint all Stage 1 planar homography UV textures
    right_wall_uv = paint_mill_right_wall_uv(rng)
    front_wall_uv = paint_mill_front_gable_uv(rng)
    roof_uv = paint_mossy_shingle_roof_uv(rng)
    porch_uv = paint_mill_porch_deck_uv(rng)
    footpath_uv = paint_voronoi_footpath_uv(rng, x_min=-1.20, x_max=4.40, y_min=1.20, y_max=23.5, mask_to_path=True)
    left_yard_uv = paint_voronoi_footpath_uv(rng, x_min=-5.20, x_max=-1.70, y_min=1.40, y_max=7.40, mask_to_path=False, tw=512, th=512)
    bridge_top_uv = paint_granite_bridge_slab_uv(rng)
    bridge_front_uv = paint_voronoi_stone_wall_uv(rng, 2.4, 0.24, facing_sun=0.25, wet_bottom=True, tw=512, th=128)

    # Stepped dry-stone retaining walls (ishigaki) with metric Voronoi stones!
    # Left millrace retaining wall faces +X (visible to camera & sun!); right channel wall is a backface (not drawn over water!)
    mr_left_wall_uv = paint_voronoi_stone_wall_uv(rng, 6.8, 0.44, facing_sun=0.84, wet_bottom=True)
    yard_low_wall_uv = paint_voronoi_stone_wall_uv(rng, 3.0, 0.56, facing_sun=0.86, wet_bottom=False, tw=768, th=256)

    def warp_mill_and_bridge_planes(dst: np.ndarray, mirrored: bool = False, rgba_mode: bool = False) -> np.ndarray:
        def mp(p: np.ndarray) -> np.ndarray:
            return mirror_point_z(p, Z_STREAM) if mirrored else p

        # 1. Right cedar clapboard wall (u_m = 0, facing +MILL_RGT toward morning sun & viewer!)
        p0 = mp(mill_pt(0.0, 0.0, MILL_Z_EAVE))
        p1 = mp(mill_pt(0.0, MILL_LENGTH, MILL_Z_EAVE))
        p2 = mp(mill_pt(0.0, 0.0, Z_MILL_BASE))
        dst = cam.warp_plane_over(dst, right_wall_uv, p0, p1 - p0, p2 - p0, fog_rgb, 6.8, 0.034, dst_rgba_mode=rgba_mode)

        # 2. Front cedar wall & weathered lime-plaster gable (l_m = 0)
        q0 = mp(mill_pt(-MILL_WIDTH, 0.0, MILL_Z_RIDGE))
        q1 = mp(mill_pt(0.0, 0.0, MILL_Z_RIDGE))
        q2 = mp(mill_pt(-MILL_WIDTH, 0.0, Z_MILL_BASE))
        dst = cam.warp_plane_over(dst, front_wall_uv, q0, q1 - q0, q2 - q0, fog_rgb, 6.8, 0.034, dst_rgba_mode=rgba_mode)

        # 2b. Porch posts, horizontal tie beams & tucked-in under-eave rafters drawn BEFORE roof so mossy roof cleanly shelters them!
        s_raf, c_raf = new_cairo_2x()
        draw_mill_porch_under_roof(c_raf, cam, fog_rgb, mirrored=mirrored)
        if not mirrored:
            draw_mill_under_eave_rafters(c_raf, cam, fog_rgb, mirrored=False)
        raf_1x = finish_cairo_2x(s_raf)
        if rgba_mode:
            ra_r = raf_1x[:, :, 3:4]
            dst[:, :, :3] = raf_1x[:, :, :3] + dst[:, :, :3] * (1.0 - ra_r)
            dst[:, :, 3:4] = ra_r + dst[:, :, 3:4] * (1.0 - ra_r)
        else:
            dst = over_premul(dst, raf_1x)

        # 3. Sunlit pitched mossy cedar shingle roof — right slope (cleanly covers porch tie-beams & rafter tops!)
        u_ridge = -0.5 * MILL_WIDTH
        r0 = mp(mill_pt(u_ridge, -ROOF_OVER_F, MILL_Z_RIDGE))
        r1 = mp(mill_pt(u_ridge, MILL_LENGTH + ROOF_OVER_B, MILL_Z_RIDGE))
        r2 = mp(mill_pt(ROOF_OVER_R, -ROOF_OVER_F, ROOF_Z_EAVE_R))
        dst = cam.warp_plane_over(dst, roof_uv, r0, r1 - r0, r2 - r0, fog_rgb, 6.8, 0.034, dst_rgba_mode=rgba_mode)

        # 4. Textured rough-hewn granite foot-slab bridge spanning cleanly across the crystal millrace!
        by0, by1 = 6.20, 7.05
        bx0 = float(millrace_left_x(6.62)) - 0.16
        bx1 = float(millrace_right_x(6.62)) + 0.18
        bf0 = mp(np.array([bx0, by0, 0.07]))
        bf1 = mp(np.array([bx1, by0, 0.07]))
        bf2 = mp(np.array([bx0, by0, -0.14]))
        dst = cam.warp_plane_over(dst, bridge_front_uv, bf0, bf1 - bf0, bf2 - bf0, fog_rgb, 6.8, 0.036, dst_rgba_mode=rgba_mode)
        if not mirrored:
            bt0 = np.array([bx0, by0, 0.07])
            bt1 = np.array([bx1, by0, 0.07])
            bt2 = np.array([bx0, by1, 0.09])
            dst = cam.warp_plane_over(dst, bridge_top_uv, bt0, bt1 - bt0, bt2 - bt0, fog_rgb, 6.8, 0.036, dst_rgba_mode=rgba_mode)
        return dst

    # Stage 2 Dedicated 3D Plane-Mirrored Quads Pass across z = Z_STREAM!
    mirrored_3d = np.zeros((H, W, 4), dtype=np.float32)
    mirrored_3d = warp_mill_and_bridge_planes(mirrored_3d, mirrored=True, rgba_mode=True)
    s_ref, c_ref = new_cairo_2x()
    draw_mill_roof_trim(c_ref, cam, fog_rgb, mirrored=True)
    draw_waterwheel_and_flume(c_ref, cam, fog_rgb, mirrored=True)
    ref_cairo = finish_cairo_2x(s_ref)
    ra = ref_cairo[:, :, 3:4]
    mirrored_3d[:, :, :3] = ref_cairo[:, :, :3] + mirrored_3d[:, :, :3] * (1.0 - ra)
    mirrored_3d[:, :, 3:4] = ra + mirrored_3d[:, :, 3:4] * (1.0 - ra)

    # Stage 2: Crystal jade mountain millrace stream with Fresnel reflections & aerated splash wake!
    canvas, stream_left_uvs, stream_right_uvs = paint_millrace_fresnel_water(
        canvas, ref_backdrop, mirrored_3d, cam, rng, fog_rgb, m_global
    )

    # Stage 1: Left-bank sunlit Voronoi dry-stone retaining wall (ishigaki) bordering crystal stream!
    xl_near = float(millrace_left_x(2.0))
    xl_mid = float(millrace_left_x(8.7))
    canvas = cam.warp_plane_over(
        canvas,
        mr_left_wall_uv,
        p0=np.array([xl_near, 2.0, 0.08]),
        e1=np.array([xl_mid - xl_near, 6.7, 0.0]),
        e2=np.array([0.0, 0.0, -0.36]),
        fog_rgb=fog_rgb,
        fog_start=6.8,
        fog_density=0.036,
    )

    # Left foreground mossy granite mill-yard terrace strictly behind millrace_left_x!
    canvas = cam.warp_plane_over(
        canvas,
        left_yard_uv,
        p0=np.array([-5.20, 1.40, 0.06]),
        e1=np.array([3.50, 0.0, 0.0]),
        e2=np.array([0.0, 6.00, 0.0]),
        fog_rgb=fog_rgb,
        fog_start=6.8,
        fog_density=0.036,
    )

    # Prominent stepped dry-stone retaining wall (ishigaki) framing the foreground mill-yard terrace!
    w_x0 = float(millrace_left_x(2.6)) - 0.16
    w_x1 = float(millrace_left_x(5.3)) - 0.16
    canvas = cam.warp_plane_over(
        canvas,
        yard_low_wall_uv,
        p0=np.array([w_x0, 2.60, 0.44]),
        e1=np.array([w_x1 - w_x0, 2.70, 0.0]),
        e2=np.array([0.06, 0.0, -0.46]),
        fog_rgb=fog_rgb,
        fog_start=6.8,
        fog_density=0.036,
    )
    coping_uv = paint_granite_bridge_slab_uv(rng, tw=512, th=128)
    canvas = cam.warp_plane_over(
        canvas,
        coping_uv,
        p0=np.array([w_x0 - 0.38, 2.60, 0.46]),
        e1=np.array([w_x1 - w_x0, 2.70, 0.0]),
        e2=np.array([0.40, -0.04, -0.02]),
        fog_rgb=fog_rgb,
        fog_start=6.8,
        fog_density=0.036,
    )

    # Curving damp mossy Voronoi highland stone footpath (z = 0.0, cleanly RIGHT of crystal millrace!)
    canvas = cam.warp_plane_over(
        canvas,
        footpath_uv,
        p0=np.array([-1.20, 1.20, Z_PATH]),
        e1=np.array([5.60, 0.0, 0.0]),
        e2=np.array([0.0, 22.3, 0.0]),
        fog_rgb=fog_rgb,
        fog_start=6.8,
        fog_density=0.036,
    )

    # Timber mill porch deck (exact angled homography in mill-local space!)
    d0 = mill_pt(-MILL_WIDTH, -PORCH_DEPTH, Z_PORCH)
    d1 = mill_pt(0.08, -PORCH_DEPTH, Z_PORCH)
    d2 = mill_pt(-MILL_WIDTH, 0.10, Z_PORCH)
    canvas = cam.warp_plane_over(canvas, porch_uv, d0, d1 - d0, d2 - d0, fog_rgb, 6.8, 0.034)

    # Stage 1: Exact planar angled cedar watermill walls, gable, mossy shingle roof & textured granite foot-slab!
    canvas = warp_mill_and_bridge_planes(canvas, mirrored=False, rgba_mode=False)

    # Stage 3 Band 2: Mill porch timbers, hero open 3/4 spoked wooden waterwheel (suisha) + kakehi flume,
    # dry-brush velvet moss cushions, cedar bench, burlap tea sacks & woven bamboo tea baskets!
    s_b2, c_b2 = new_cairo_2x()
    draw_mill_roof_trim(c_b2, cam, fog_rgb, mirrored=False)
    wheel_info = draw_waterwheel_and_flume(c_b2, cam, fog_rgb, mirrored=False)
    draw_burlap_tea_sacks_and_bench(c_b2, cam, fog_rgb)

    mrng = np.random.default_rng(778)
    for mp_w, mrad in [
        (np.array([float(millrace_left_x(6.35)) - 0.04, 6.35, 0.08]), 0.11),
        (np.array([float(millrace_left_x(6.85)) - 0.05, 6.85, 0.09]), 0.10),
        (np.array([float(millrace_right_x(6.35)) + 0.06, 6.35, 0.08]), 0.12),
        (np.array([float(millrace_right_x(6.85)) + 0.08, 6.85, 0.09]), 0.10),
        (np.array([w_x0 + 0.08, 3.60, 0.45]), 0.12),
        (np.array([w_x1 - 0.06, 4.75, 0.45]), 0.11),
    ]:
        um, vm, zm = cam.project(mp_w)
        if zm > Z_NEAR:
            md = float(np.linalg.norm(mp_w - cam.eye))
            draw_velvet_moss_tuft(c_b2, float(um), float(vm), mrad * cam.K[0, 0] / float(zm), mrng, md, fog_rgb, sun_w=0.68)

    # Interlocking woven bamboo tea-picking baskets (chatsumi-kago) brimming with glossy fresh tea leaves!
    draw_woven_tea_basket(c_b2, cam, mill_pt(-1.05, -0.92, Z_PORCH), 0.30, 0.42, 601, fog_rgb)
    draw_woven_tea_basket(c_b2, cam, mill_pt(-0.46, -1.25, Z_PORCH), 0.27, 0.37, 602, fog_rgb)
    # Hero foreground tea baskets resting cleanly on the mossy footpath approaching the granite foot-slab!
    draw_woven_tea_basket(c_b2, cam, np.array([float(millrace_right_x(5.30)) + 0.38, 5.30, Z_PATH]), 0.28, 0.38, 603, fog_rgb)
    draw_woven_tea_basket(c_b2, cam, np.array([float(millrace_right_x(5.78)) + 0.80, 5.78, Z_PATH]), 0.25, 0.34, 604, fog_rgb)

    b2_1x = finish_cairo_2x(s_b2)
    h_tooth = paper_tooth(H, W, rng)
    a2 = b2_1x[:, :, 3:4]
    b2_1x[:, :, :3] = np.clip(b2_1x[:, :, :3] + a2 * (m_global * 0.015 + d_global * 0.010 + h_tooth * 0.006)[..., None], 0.0, 1.28)
    canvas = over_premul(canvas, b2_1x)

    # Stage 2 & 3: Diagonal golden dawn sunbeams strictly aligned with SUN_DIR!
    beams, sun_axis_2d = paint_valley_dawn_sunbeams(cam, rng, intensity=0.34)
    canvas = 1.0 - (1.0 - np.clip(canvas, 0.0, 1.0)) * (1.0 - np.clip(beams, 0.0, 0.80))

    # Stage 3 Band 3: Dewy foreground meadow sedge, flowering highland tea shrubs & white tea blossoms!
    s_b3, c_b3 = new_cairo_2x()
    draw_meadow_sedge_and_tea_shrubs(c_b3, cam, fog_rgb)
    for _ in range(95):
        mx = float(rng.uniform(120, W - 80))
        my = float(rng.uniform(70, H - 100))
        mr = float(rng.uniform(0.9, 2.0))
        mv = float(rng.uniform(0.16, 0.48)) * (0.45 + 0.55 * math.exp(-((mx / W - 0.68) / 0.34) ** 2))
        c_b3.set_source_rgba(0.99, 0.96, 0.72, mv)
        c_b3.arc(mx, my, mr, 0, math.tau)
        c_b3.fill()
    b3_1x = finish_cairo_2x(s_b3)
    a3 = b3_1x[:, :, 3:4]
    b3_1x[:, :, :3] = np.clip(b3_1x[:, :, :3] + a3 * (m_global * 0.014 + h_tooth * 0.006)[..., None], 0.0, 1.25)
    canvas = over_premul(canvas, b3_1x)

    # Stage 4: Lower-right near-camera disc bokeh dewy tea branch + 35mm theatrical optics!
    fg_bokeh = paint_foreground_bokeh_tea_twig()
    canvas = over_premul(canvas, fg_bokeh)

    su_w, sv_w, _ = cam.project(SIDE_WIN_CENTER)
    fu_w, fv_w, _ = cam.project(FRONT_WIN_CENTER)
    window_uvs = [(float(su_w) / W, float(sv_w) / H), (float(fu_w) / W, float(fv_w) / H)]

    final_rgb = apply_35mm_finish(canvas, rng, window_uvs)
    elapsed = time.perf_counter() - t0
    print(f"[matcha-02] Rendered {W}x{H} in {elapsed:.2f}s")

    du_dv = (sun_axis_2d[0] / W) / (sun_axis_2d[1] / H)
    print("[matcha-02] === EXACT CAMERA3D UV CALIBRATION FOR wall-02.frag ===")
    print(f"  Horizon v           : {cam.horizon_y / H:.4f}   VP uv: ({cam.vp_x / W:.4f}, {cam.vp_y / H:.4f})")
    print(f"  Sunbeam dir 2D      : ({sun_axis_2d[0]:.4f}, {sun_axis_2d[1]:.4f})  (UV slope du/dv = {du_dv:.4f})")
    print(f"  Side window UV      : ({window_uvs[0][0]:.4f}, {window_uvs[0][1]:.4f})")
    print(f"  Front window UV     : ({window_uvs[1][0]:.4f}, {window_uvs[1][1]:.4f})")
    print(f"  Waterwheel splash UV: ({wheel_info['splash_uv'][0]:.4f}, {wheel_info['splash_uv'][1]:.4f})")
    print(f"  Valley mist band v  : {cam.horizon_y / H + 0.02:.4f} .. {cam.horizon_y / H + 0.14:.4f} (u >= 0.44)")
    print("  Millrace LEFT boundary (y_m -> u, v):")
    for y_p, cu, cv in stream_left_uvs:
        print(f"    y={y_p:4.1f}m : u={cu:.4f}, v={cv:.4f}")
    print("  Millrace RIGHT boundary (y_m -> u, v):")
    for y_p, cu, cv in stream_right_uvs:
        print(f"    y={y_p:4.1f}m : u={cu:.4f}, v={cv:.4f}")

    u8 = np.round(np.clip(final_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    full_im = Image.fromarray(u8, mode="RGB")

    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        full_im.save(audit_dir / "w02_full.png", optimize=True)
        sp_u, sp_v = wheel_info["splash_uv"]
        crops = {
            "crop1_tea_valley_mist.png": (int(W * 0.56), int(H * 0.28)),
            "crop2_watermill_wheel_stream.png": (int(sp_u * W - 260), int(sp_v * H - 280)),
            "crop3_foreground_wall_baskets.png": (int(W * 0.09), H - 450),
        }
        for cname, (cx0, cy0) in crops.items():
            x0 = max(0, min(W - 600, cx0))
            y0 = max(0, min(H - 450, cy0))
            full_im.crop((x0, y0, x0 + 600, y0 + 450)).save(audit_dir / cname)
            print(f"[matcha-02] Saved 1:1 crop {audit_dir / cname} @ ({x0}, {y0})")

    if export_webps:
        for wp in export_webps:
            wp.parent.mkdir(parents=True, exist_ok=True)
            full_im.save(wp, format="WEBP", quality=94, method=6)
            print(f"[matcha-02] Exported {wp} ({wp.stat().st_size / 1024:.1f} KiB)")

    return final_rgb, {
        "elapsed": elapsed,
        "vp_uv": (cam.vp_x / W, cam.vp_y / H),
        "window_uvs": window_uvs,
        "splash_uv": wheel_info["splash_uv"],
        "sun_axis_2d": sun_axis_2d,
        "du_dv": du_dv,
        "stream_left_uvs": stream_left_uvs,
        "stream_right_uvs": stream_right_uvs,
    }


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    default_webps = [
        repo_root / "session" / "themes" / "matcha" / "walls" / "wall-02.webp",
        repo_root / "docs" / "assets" / "walls" / "matcha-02.webp",
    ]
    ap = argparse.ArgumentParser(description='Render Matcha 02 — "Misty Terraced Tea Valley & Cedar Watermill" (2048x1152)')
    ap.add_argument("--seed", type=int, default=202622)
    ap.add_argument("--audit-dir", type=pathlib.Path, default=None)
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()
    render_scene(seed=args.seed, audit_dir=args.audit_dir, export_webps=None if args.no_export else default_webps)


if __name__ == "__main__":
    main()

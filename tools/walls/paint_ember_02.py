#!/usr/bin/env python3
"""
Ember 02 ("Twilight Onsen Bathhouse Canal") — Pure-Python 3D-Grounded 1990s Theatrical Anime Pipeline (2048 x 1152)

Five-stage deterministic painting pipeline (zero image models, zero external assets):
  Stage 1: Metric right-handed Camera3D gazing upstream along a narrow steaming mountain
           hot-spring town canal at autumn dusk (off-axis vanishing point ~58.9% width),
           exact 3x3 UV-to-screen plane homographies masked by valid UV and exact ray-plane
           depth z_cam > 0.05 with exponential twilight mist haze, near-plane 3D segment
           clipping, coursed irregular metric jittered 3x3 Voronoi (d2 - d1) wet basalt
           promenade flagstones & vertical stacked stone canal embankments (ishigaki), and
           a towering 3-story charred-timber & warm-plaster ryokan bathhouse facade with
           glowing amber kumiko shoji bays, cantilevered carved kōran balconies, pitched
           curved kawara tiled eaves, 3F marumado loft window, and cedar bath exhaust louver.
  Stage 2: Single locked normalized low twilight SUN_DIR (z < 0) skimming over the mountain
           gorge ridges (6 height-refinement steps across pitched ryokan eaves & arched
           footbridge) unified with explicit 3D point lights in every hanging paper chochin
           lantern and glowing shoji bay; steaming deep slate-teal & amber-lit thermal canal
           water (z = 0) combining cv2.remap capillary-rippled horizon reflections with 3D
           plane-mirrored quads (pts - 2 * dot(pts - p0, n) * n) of the entire glowing
           multi-story ryokan, hanging chochin lanterns, arched red-lacquer footbridge
           (taiko-bashi), riverside pavilion, and autumn maple cliffs weighted by
           fresnel_min + (fresnel_max - fresnel_min) * (1 - dist_t) ** 1.4.
  Stage 3: Depth-interleaved 2x supersampled PyCairo bands (Lanczos4-downsampled, cleanly
           unpremultiplied) back-to-front:
             (0) Luminous twilight plum-to-rose-apricot sky with sweeping raymarched dusk
                 clouds, layered misty Kyoto autumn gorge ridges, and detailed distant
                 warm lit multi-story ryokan facades receding upstream;
             (1) Midground arched red-lacquer wooden footbridge (taiko-bashi) with bronze
                 giboshi finials & hanging bridge chochin lanterns, flanked by a curved-eave
                 weathered wooden riverside onsen gazebo/pavilion and organic cubic-Bezier
                 mossy volcanic gorge rock outcrops;
             (2) Left 3-story ryokan timber joinery, cantilevered carved kōran balcony,
                 narrow tiled 1F eave drip visor & overhanging kawara rafters, hanging
                 oval red-and-cream paper chochin lanterns with varied brush calligraphy &
                 warm upward soffit bounce, split bathhouse noren curtain ('ゆ' motif),
                 mossy cedar thermal spring spout pouring frothy water into the canal,
                 steaming cedar bath tubs with folded tenugui towel & promenade railings;
             (3) Gnarled overhanging crimson-amber Japanese maple (momiji) branches from the
                 right gorge cliff with multi-pass backlit 5-lobed cubic-Bezier leaf clusters
                 (zero stamped circles) plus delicate wispy translucent painted steam ribbons
                 drifting low above the thermal river — modulated with isotropic fbm_gouache,
                 flat-brush dabs, cold-press paper tooth, and warm shadow bounce (lum < 0.28).
  Stage 4: Foreground upper-right overhanging backlit autumn maple bough convolved in
           premultiplied RGBA with a circular aperture disc kernel (r = 7.8) + separate
           cv2.warpAffine ~1.05 px opposite R/B shift, followed by 35mm 3-scale highlight
           halation (sigma = 6.5, 24, 72 px above lum > 0.72 + lantern glints), radial
           corner CA, optical vignette, shadow black lift + C1 tanh highlight shoulder
           (knee = 0.80, ceiling = 0.965), luminance-weighted silver-halide grain, and faint
           analog YIQ chroma softening (cv2.BORDER_REFLECT_101), slight luma ring, sub-pixel
           line jitter, and <= 0.75% scanlines.
  Stage 5: Deterministic render (<60s CPU) exporting identical 2048x1152 WebP (quality=94),
           optional full PNG + 4 1:1 (600x450) audit crops, and exact Camera3D projected
           shader calibration coordinates for wall-02.frag.
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
# Unified 3D World Geometry & Stage 2 Locked Light Rig (Right-Handed Meters)
# =============================================================================
# +X right across canal, +Y forward upstream, +Z up from steaming thermal water (z = 0)

_raw_sun = np.array([-0.44, -0.87, -0.220], dtype=np.float64)
SUN_DIR = _raw_sun / np.linalg.norm(_raw_sun)

Z_WATER = 0.0
Z_PROM = 1.10            # Left damp basalt stone promenade elevation (meters)
Z_RIGHT_BANK = 1.40      # Right ishigaki retaining wall top elevation

X_LEFT_BANK = 0.00       # Left canal ishigaki retaining wall face
X_RIGHT_BANK = 6.15      # Right canal ishigaki retaining wall face
X_RYOKAN_WALL = -2.35    # Left multi-story ryokan 1F & 2F timber facade plane
X_BALCONY_OUT = -1.42    # 2F cantilevered wooden veranda outer railing plane
X_EAVE1_OUT = -0.84      # 1F narrow tiled eave drip visor (hisashi) projecting past balcony
X_EAVE2_OUT = -0.78      # 2F-to-3F grand curved kawara roof eave overhang
X_LOFT_WALL = -2.10      # 3F bathhouse loft clerestory wall plane
X_EAVE3_OUT = -1.06      # 3F crown kawara roof eave overhang

Y_RYOKAN_0 = -1.50
Y_RYOKAN_1 = 17.50
Z_1F_TOP = 3.32          # 1F ceiling / 2F cantilevered balcony underside
Z_2F_FLOOR = 3.46        # 2F veranda floor deck elevation
Z_2F_TOP = 5.56          # 2F main overhanging kawara roof eave line
Z_3F_TOP = 7.36          # 3F attic clerestory crown eave line

# Midground Arched Red-Lacquer Wooden Footbridge (taiko-bashi / soribashi)
Y_BRIDGE_C = 19.25
BRIDGE_HALF_W = 1.28     # y in [17.97, 20.53]
BRIDGE_X0 = -0.48        # Grounded onto left cut-stone bridgehead abutment
BRIDGE_X1 = 6.62         # Grounded onto right cut-stone bridgehead abutment
BRIDGE_ARCH_H = 0.72     # Graceful gentle Japanese timber camber -> peak z ~ 1.98m

# Right Bank Riverside Wooden Gazebo / Onsen Pavilion (azumaya)
PAV_X0, PAV_X1 = 6.38, 9.25
PAV_Y0, PAV_Y1 = 11.95, 15.65
PAV_Z_BASE = Z_RIGHT_BANK
PAV_Z_EAVE = 3.74
PAV_Z_PEAK = 4.90

# Steaming Cedar Thermal Spring Spout (kakehi) jutting from left ishigaki embankment
SPOUT_X, SPOUT_Y, SPOUT_Z = 0.44, 8.65, 0.72

# Structural Ryokan Posts along Y (meters)
RYOKAN_POST_YS = (-2.00, 1.60, 5.20, 8.80, 12.40, 16.00, 17.45)

# Explicit 3D Hanging Paper Chochin Lanterns:
# (x, y, z_hook, z_center, radius_m, height_m, intensity, style, kanji_id, label)
CHOCHIN_SPECS: list[tuple[float, float, float, float, float, float, float, str, int, str]] = [
    # 1F Hisashi eave drip visor lanterns (attached cleanly under physical visor eave at x=-0.92, z=3.16!)
    (X_EAVE1_OUT - 0.08, 4.15, 3.16, 2.80, 0.160, 0.40, 1.85, "ryokan_red",   0, "chochin_1f_near"),
    (X_EAVE1_OUT - 0.08, 7.20, 3.16, 2.82, 0.165, 0.42, 1.98, "ryokan_cream", 1, "chochin_1f_entry"),
    (X_EAVE1_OUT - 0.08, 10.65, 3.16, 2.82, 0.155, 0.40, 1.72, "ryokan_red",  2, "chochin_1f_mid"),
    (X_EAVE1_OUT - 0.08, 14.20, 3.16, 2.82, 0.150, 0.38, 1.60, "ryokan_cream",3, "chochin_1f_far"),
    # 2F Overhanging kawara eaves lanterns (attached cleanly at x=-0.86, z=5.48!)
    (X_EAVE2_OUT - 0.08, 6.20, 5.48, 5.06, 0.165, 0.43, 1.75, "ryokan_red",   3, "chochin_2f_near"),
    (X_EAVE2_OUT - 0.08, 11.40, 5.48, 5.06, 0.160, 0.42, 1.65, "ryokan_cream",0, "chochin_2f_mid"),
    (X_EAVE2_OUT - 0.08, 15.80, 5.48, 5.06, 0.155, 0.40, 1.55, "ryokan_red",  1, "chochin_2f_far"),
    # Arched Red-Lacquer Footbridge (taiko-bashi) paired post lanterns (snug compact roof-cap atop pf=0.25 & 0.75!)
    (1.295, Y_BRIDGE_C - BRIDGE_HALF_W - 0.16, 2.84, 2.58, 0.145, 0.35, 1.74, "bridge_red", 0, "chochin_bridge_L"),
    (4.845, Y_BRIDGE_C - BRIDGE_HALF_W - 0.16, 2.92, 2.66, 0.145, 0.35, 1.74, "bridge_red", 1, "chochin_bridge_R"),
    # Right riverside wooden pavilion lantern
    (PAV_X0 + 0.28, 13.75, PAV_Z_EAVE - 0.05, PAV_Z_EAVE - 0.54, 0.165, 0.42, 1.78, "pavilion_gold", 2, "chochin_pavilion"),
]

# Key Glowing Shoji Bays on left ryokan facade (y0, y1, z0, z1, intensity, label)
SHOJI_BAYS: list[tuple[float, float, float, float, float, str]] = [
    (2.15, 4.75, 1.56, 3.06, 0.88, "shoji_1f_lounge_near"),
    (5.75, 8.30, 1.22, 3.08, 1.10, "shoji_1f_entry"),
    (9.35, 11.90, 1.56, 3.06, 0.85, "shoji_1f_lounge_mid"),
    (12.90, 15.50, 1.70, 3.04, 0.74, "shoji_1f_louver"),
    (2.10, 4.95, 3.70, 5.22, 0.98, "shoji_2f_near"),
    (5.80, 8.55, 3.70, 5.22, 1.16, "shoji_2f_hero"),
    (9.35, 12.10, 3.70, 5.22, 0.96, "shoji_2f_mid"),
    (12.90, 15.65, 3.70, 5.22, 0.92, "shoji_2f_far"),
    (6.10, 8.45, 5.95, 6.95, 0.80, "shoji_3f_loft_near"),
    (10.00, 12.40, 5.95, 6.95, 0.76, "shoji_3f_loft_mid"),
    (13.80, 15.30, 5.95, 7.05, 0.82, "shoji_3f_marumado"),
]

ALL_POINT_LIGHTS: list[tuple[float, float, float, float, bool]] = (
    [(cx, cy, cz, inten, False) for (cx, cy, _zh, cz, _r, _h, inten, _st, _k, _lb) in CHOCHIN_SPECS]
    + [
        (X_RYOKAN_WALL + 0.25, 0.5 * (y0 + y1), 0.5 * (z0 + z1), inten, True)
        for (y0, y1, z0, z1, inten, _lb) in SHOJI_BAYS
    ]
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_THEME_WEBP = REPO_ROOT / "session" / "themes" / "ember" / "walls" / "wall-02.webp"
DEFAULT_DOCS_WEBP = REPO_ROOT / "docs" / "assets" / "walls" / "ember-02.webp"


def bridge_deck_z(x: np.ndarray | float) -> np.ndarray | float:
    """Graceful, low-camber traditional Japanese timber arch curve (soribashi / taiko-bashi) seated on both stone abutments."""
    t = np.clip((x - BRIDGE_X0) / (BRIDGE_X1 - BRIDGE_X0), 0.0, 1.0)
    z_lin = (Z_PROM + 0.05) * (1.0 - t) + (Z_RIGHT_BANK - 0.04) * t
    arch = BRIDGE_ARCH_H * np.power(np.maximum(0.0, 4.0 * t * (1.0 - t)), 0.92)
    return z_lin + arch


# =============================================================================
# Fast Vectorized Painterly Primitives (NumPy + OpenCV + PyCairo)
# =============================================================================

def gaussian_blur_2d(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.1:
        return arr.copy()
    if sigma <= 14.0:
        return cv2.GaussianBlur(arr.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    h, w = arr.shape[:2]
    factor = 4 if sigma < 42 else 8
    sw, sh = max(8, w // factor), max(8, h // factor)
    small = cv2.resize(arr.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma / factor, sigmaY=sigma / factor, borderType=cv2.BORDER_REFLECT_101)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def gaussian_blur_1d(arr: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    if sigma <= 0.05:
        return arr.copy()
    if axis == 1:
        return cv2.GaussianBlur(arr.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=1e-4, borderType=cv2.BORDER_REFLECT_101)
    return cv2.GaussianBlur(arr.astype(np.float32), (0, 0), sigmaX=1e-4, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)


def resize_bicubic(arr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    return cv2.resize(arr.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_CUBIC)


def remap_bilinear(src: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
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
    scales=((72, 68), (28, 30), (11, 10), (4.6, 4.9)),
) -> np.ndarray:
    """Isotropic square-scale multi-octave gouache mottle (NEVER high-aspect-ratio stretched!)."""
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
        cx = float(rng.uniform(0, qw))
        cy = float(rng.uniform(0, qh))
        ang = float(rng.uniform(0, math.pi))
        rx = float(rng.uniform(qw * 0.030, qw * 0.095))
        ry = float(rng.uniform(rx * 0.30, rx * 0.58))
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
        buf = np.frombuffer(s_one.get_data(), dtype=np.uint8).reshape(qh, stride)[:, :qw].astype(np.float32) / 255.0
        arr_sum += strength * buf
    arr_sum = gaussian_blur_2d(arr_sum, 3.0)
    arr_sum -= arr_sum.mean()
    arr_sum /= float(arr_sum.std()) + 1e-6
    return resize_bicubic(arr_sum, w, h)


def over_premul(dst_rgb: np.ndarray, src_premul_rgba: np.ndarray) -> np.ndarray:
    a = np.clip(src_premul_rgba[:, :, 3:4], 0.0, 1.0)
    return src_premul_rgba[:, :, :3] + dst_rgb * (1.0 - a)


def apply_warm_shadow_bounce(rgb: np.ndarray, strength: float = 0.015) -> np.ndarray:
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    mask = np.clip((0.28 - lum) / 0.28, 0.0, 1.0)[..., None]
    bounce = np.array([1.22, 0.52, 0.30], dtype=np.float32) * strength
    return rgb + mask * bounce


def mirror_pts_plane(pts: np.ndarray, p0: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Exact 3D plane reflection: pts - 2 * dot(pts - p0, n) * n."""
    d = np.sum((pts - p0) * n, axis=-1, keepdims=True)
    return pts - 2.0 * d * n


def mirror_z0(pts: np.ndarray) -> np.ndarray:
    p0 = np.array([0.0, 0.0, Z_WATER], dtype=np.float64)
    n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return mirror_pts_plane(np.asarray(pts, dtype=np.float64), p0, n)


# =============================================================================
# Stage 1 — Metric 3D Pinhole Camera & Exact 3x3 Plane Homography
# =============================================================================

class Camera3D:
    def __init__(self, w: int, h: int, eye: Iterable[float], target: Iterable[float], fov_x_deg: float = 61.0):
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
        self.fwd = fwd
        self.right = right
        self.up = up
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
        fog_start: float = 8.5,
        fog_density: float = 0.030,
        opacity: float = 1.0,
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
        sampled = cv2.remap(
            tex_premul_rgba.astype(np.float32),
            u_cl.astype(np.float32),
            v_cl.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        v_f = (valid.astype(np.float32) * np.float32(opacity))[..., None]
        alpha = sampled[:, :, 3:4] * v_f
        rgb_valid = sampled[:, :, :3] * v_f

        fg_rgb = rgb_valid * (1.0 - fog_t[..., None]) + (fog_rgb[None, None, :] * alpha) * fog_t[..., None]
        patch = np.concatenate([fg_rgb, alpha], axis=-1)
        dst_rgb[y0:y1, x0:x1] = over_premul(dst_rgb[y0:y1, x0:x1], patch)
        return dst_rgb


# =============================================================================
# Stage 2 — Unified 3D Twilight Shadow Solver (6-Step Roof & Bridge Arch) + Point Lights
# =============================================================================

def trace_onsen_sun_visibility(wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> np.ndarray:
    """
    Trace any 3D point backward along -SUN_DIR using 6 height-refinement steps across:
      1. The 1F balcony + eave drip visor overhang,
      2. The 2F grand overhanging kawara eaves, and
      3. The arched red-lacquer wooden footbridge (taiko-bashi) spanning the canal.
    """
    inv_sz = np.float32(1.0 / (-SUN_DIR[2]))
    sx = np.float32(SUN_DIR[0])
    sy = np.float32(SUN_DIR[1])

    # 1. 1F hisashi eave drip visor + 2F balcony overhang
    z_e1 = np.full_like(wx, 3.30, dtype=np.float32)
    for _ in range(6):
        t = np.maximum(0.0, (z_e1 - wz) * inv_sz)
        rx = wx - sx * t
        z_e1 = np.float32(3.22) + np.clip(np.float32(X_EAVE1_OUT) - rx, 0.0, 2.2) * np.float32(0.32)
    t1 = np.maximum(0.0, (z_e1 - wz) * inv_sz)
    rx1, ry1 = wx - sx * t1, wy - sy * t1
    hit1 = (wz < z_e1 - 0.02) & (rx1 >= -4.8) & (rx1 <= X_EAVE1_OUT) & (ry1 >= Y_RYOKAN_0) & (ry1 <= Y_RYOKAN_1)
    soft1 = np.clip((rx1 - (X_EAVE1_OUT - 0.24)) / 0.24, 0.0, 1.0)
    vis = np.where(hit1, 0.24 + 0.76 * soft1, 1.0).astype(np.float32)

    # 2. 2F grand overhanging kawara eaves
    z_e2 = np.full_like(wx, 5.65, dtype=np.float32)
    for _ in range(6):
        t = np.maximum(0.0, (z_e2 - wz) * inv_sz)
        rx = wx - sx * t
        z_e2 = np.float32(5.52) + np.clip(np.float32(X_EAVE2_OUT) - rx, 0.0, 2.8) * np.float32(0.40)
    t2 = np.maximum(0.0, (z_e2 - wz) * inv_sz)
    rx2, ry2 = wx - sx * t2, wy - sy * t2
    hit2 = (wz < z_e2 - 0.02) & (rx2 >= -5.2) & (rx2 <= X_EAVE2_OUT) & (ry2 >= Y_RYOKAN_0) & (ry2 <= Y_RYOKAN_1)
    soft2 = np.clip((rx2 - (X_EAVE2_OUT - 0.28)) / 0.28, 0.0, 1.0)
    vis = np.minimum(vis, np.where(hit2, 0.28 + 0.72 * soft2, 1.0).astype(np.float32))

    # 3. Arched red-lacquer footbridge (taiko-bashi) deck shadow
    z_br = np.full_like(wx, 2.10, dtype=np.float32)
    for _ in range(6):
        t = np.maximum(0.0, (z_br - wz) * inv_sz)
        rx = wx - sx * t
        z_br = bridge_deck_z(rx).astype(np.float32)
    tb = np.maximum(0.0, (z_br - wz) * inv_sz)
    rxb, ryb = wx - sx * tb, wy - sy * tb
    hit_b = (
        (wz < z_br - 0.04)
        & (rxb >= BRIDGE_X0)
        & (rxb <= BRIDGE_X1)
        & (np.abs(ryb - Y_BRIDGE_C) <= BRIDGE_HALF_W + 0.15)
    )
    soft_b = np.clip((np.abs(ryb - Y_BRIDGE_C) - (BRIDGE_HALF_W - 0.20)) / 0.35, 0.0, 1.0)
    vis = np.minimum(vis, np.where(hit_b, 0.32 + 0.68 * soft_b, 1.0).astype(np.float32))
    return vis


def eval_onsen_point_lights(
    wx: np.ndarray,
    wy: np.ndarray,
    wz: np.ndarray,
    is_horizontal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    lantern_diff = np.zeros_like(wx, dtype=np.float32)
    shoji_spill = np.zeros_like(wx, dtype=np.float32)
    for lx, ly, lz, inten, is_shoji in ALL_POINT_LIGHTS:
        dx = wx - np.float32(lx)
        dy = wy - np.float32(ly)
        dz = wz - np.float32(lz)
        d2 = dx * dx + dy * dy + dz * dz
        if is_shoji:
            att = np.float32(inten) / (1.0 + 0.42 * d2)
            if is_horizontal:
                att *= np.exp(-np.maximum(0.0, dx) / np.float32(2.6))
            shoji_spill += att
        else:
            lantern_diff += np.float32(inten) / (1.0 + 0.60 * d2)
    return np.clip(lantern_diff, 0.0, 2.4), np.clip(shoji_spill, 0.0, 1.8)


# =============================================================================
# Stage 1 & Band 0 — Sweeping Makoto Shinkai / Ghibli Twilight Sky & Autumn Gorge
# =============================================================================

def paint_twilight_gorge_sky(
    cam: Camera3D,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Luminous twilight indigo-plum to glowing rose-apricot sky with sweeping multi-layered
    raymarched twilight strata clouds (ZERO awkward isolated patches!) framing 4 misty
    autumn mountain gorge ridges!
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = xx / W
    v = yy / H
    hy = float(cam.horizon_y)
    hv = hy / H

    mottle = fbm_gouache(H, W, rng)
    dabs = flat_brush_dabs(H, W, rng, count=92)
    tooth = paper_tooth(H, W, rng)

    sun_p = cam.K @ (cam.R @ (-SUN_DIR))
    sun_u = float(np.clip(sun_p[0] / sun_p[2] / W, 0.62, 0.74))
    sun_v = float(np.clip(sun_p[1] / sun_p[2] / H, 0.31, 0.43))

    t_y = np.clip(v / max(0.35, hv + 0.03), 0.0, 1.0)
    c_zen = np.array([0.048, 0.030, 0.074], dtype=np.float32)   # deep twilight indigo-plum
    c_mid = np.array([0.195, 0.088, 0.168], dtype=np.float32)   # dusky twilight plum-mauve
    c_rose = np.array([0.585, 0.220, 0.238], dtype=np.float32)  # twilight rose-crimson
    c_apr = np.array([0.960, 0.500, 0.275], dtype=np.float32)   # glowing ember-apricot
    c_gold = np.array([1.055, 0.845, 0.560], dtype=np.float32)  # honey-apricot gorge pass

    s1 = np.clip(t_y / 0.26, 0.0, 1.0)[..., None]
    s2 = np.clip((t_y - 0.22) / 0.28, 0.0, 1.0)[..., None]
    s3 = np.clip((t_y - 0.48) / 0.28, 0.0, 1.0)[..., None]
    s4 = np.clip((t_y - 0.74) / 0.26, 0.0, 1.0)[..., None]

    sky = c_zen * (1.0 - s1) + c_mid * s1
    sky = sky * (1.0 - s2) + c_rose * s2
    sky = sky * (1.0 - s3) + c_apr * s3
    sky = sky * (1.0 - s4) + c_gold * s4

    du = (u - sun_u) * (W / H)
    dv = v - sun_v
    d_sun = np.hypot(du, dv)
    glow_broad = np.exp(-(d_sun / 0.58) ** 1.42).astype(np.float32)
    glow_mid = np.exp(-(d_sun / 0.24) ** 1.65).astype(np.float32)
    glow_core = np.exp(-(d_sun / 0.105) ** 1.80).astype(np.float32)

    sky += glow_broad[..., None] * np.array([0.26, 0.125, 0.088], dtype=np.float32)
    sky += glow_mid[..., None] * np.array([0.40, 0.235, 0.125], dtype=np.float32)
    sky += glow_core[..., None] * np.array([0.36, 0.285, 0.185], dtype=np.float32)

    # Faint twilight evening stars in deep indigo zenith
    for _ in range(30):
        sx_p = float(rng.uniform(W * 0.34, W * 0.94))
        sy_p = float(rng.uniform(H * 0.015, H * 0.18))
        sbr = float(rng.uniform(0.22, 0.55)) * max(0.0, 1.0 - sy_p / (H * 0.19))
        y0s, y1s = max(0, int(sy_p) - 2), min(H, int(sy_p) + 3)
        x0s, x1s = max(0, int(sx_p) - 2), min(W, int(sx_p) + 3)
        ds = np.hypot(np.arange(x0s, x1s, dtype=np.float32)[None, :] - sx_p,
                      np.arange(y0s, y1s, dtype=np.float32)[:, None] - sy_p)
        sky[y0s:y1s, x0s:x1s] += np.exp(-(ds / 1.12) ** 2)[..., None] * np.array([1.0, 0.93, 0.85], dtype=np.float32) * sbr

    # Sweeping Horizontal Twilight Strata & Backlit Dusk Cloud Ribbons (Makoto Shinkai / Ghibli twilight sky!)
    qh, qw = H // 3, W // 3
    uq = np.linspace(0.0, 1.0, qw, dtype=np.float32)[None, :]
    vq = np.linspace(0.0, 1.0, qh, dtype=np.float32)[:, None]

    def cloud_fbm(rgen: np.random.Generator, stretch_x: float = 3.2) -> np.ndarray:
        acc = np.zeros((qh, qw), dtype=np.float32)
        for gh, gw, wt in ((8, 10, 0.42), (16, 20, 0.28), (34, 42, 0.18), (70, 88, 0.09), (120, 160, 0.03)):
            gw_s = max(4, int(round(gw / (stretch_x * 0.55))))
            acc += wt * resize_bicubic(rgen.standard_normal((gh, gw_s)).astype(np.float32), qw, qh)
        acc = gaussian_blur_2d(acc, 1.4)
        acc -= acc.mean()
        return acc / (float(acc.std()) + 1e-6)

    wx_q = cloud_fbm(rng, stretch_x=3.4) * 28.0
    wy_q = cloud_fbm(rng, stretch_x=4.2) * 3.8
    c_main = cloud_fbm(rng, stretch_x=3.2)
    c_billow = cloud_fbm(rng, stretch_x=2.4)
    bx_q = np.broadcast_to(np.arange(qw, dtype=np.float32)[None, :], (qh, qw))
    by_q = np.broadcast_to(np.arange(qh, dtype=np.float32)[:, None], (qh, qw))
    c_warped = 0.64 * remap_bilinear(c_main, bx_q + wx_q, by_q + wy_q) + 0.36 * c_billow

    # Sleek sweeping twilight cloud decks streaming cleanly across the mountain gorge sky
    slant = (uq - 0.58) * 0.12
    band_hi = np.exp(-((vq - (0.075 - slant * 0.70)) / 0.046) ** 2)
    band_mid = np.exp(-((vq - (0.185 - slant * 0.45)) / 0.044) ** 2)
    band_low = np.exp(-((vq - (0.295 - slant * 0.18)) / 0.036) ** 2)

    raw_d = (
        band_hi * np.clip(0.58 + 0.48 * c_warped, 0.0, 1.35) * 0.94
        + band_mid * np.clip(0.54 + 0.50 * c_warped, 0.0, 1.38) * 0.96
        + band_low * np.clip(0.50 + 0.46 * c_billow, 0.0, 1.28) * 0.82
    )
    a_q = np.clip((raw_d - 0.28) / 0.58, 0.0, 1.0)
    a_q = a_q * a_q * (3.0 - 2.0 * a_q)

    # Multi-step optical raymarch toward low sunset sun_uv for glowing molten cloud undersides
    sdx = sun_u * qw - bx_q
    sdy = sun_v * qh - by_q
    slen = np.hypot(sdx, sdy) + 1e-4
    ux_s, uy_s = sdx / slen, sdy / slen
    occ_q = np.zeros_like(a_q)
    for step_d, wt in ((4.5, 0.40), (11.5, 0.34), (21.5, 0.26)):
        occ_q += wt * remap_bilinear(a_q, bx_q + ux_s * step_d, by_q + uy_s * step_d)
    occ_q = gaussian_blur_2d(occ_q, 2.2)
    rim_q = np.clip((a_q - occ_q * 0.80) * 1.95, 0.0, 1.0)
    trans_q = np.exp(-occ_q * 1.32)

    cl_a = resize_bicubic(a_q, W, H)
    cl_rim = resize_bicubic(rim_q, W, H)
    cl_trans = resize_bicubic(trans_q, W, H)

    cir_q = cloud_fbm(rng, stretch_x=5.0)
    cir_band = np.exp(-((vq - 0.16 + (uq - 0.60) * 0.09) / 0.10) ** 2)
    cirrus = resize_bicubic(np.clip((cir_q - 0.24) * 0.90 * cir_band, 0.0, 0.54), W, H)

    c_sh = np.array([0.145, 0.072, 0.145], dtype=np.float32)[None, None, :]
    c_mid = np.array([0.62, 0.230, 0.232], dtype=np.float32)[None, None, :]
    c_warm = np.array([0.96, 0.515, 0.280], dtype=np.float32)[None, None, :]
    c_gold = np.array([1.10, 0.84, 0.54], dtype=np.float32)[None, None, :]

    sun_prox = np.clip(0.28 + 0.84 * glow_broad, 0.0, 1.0)[..., None]
    t_lit = np.clip(cl_trans[..., None] * (0.44 + 0.72 * sun_prox), 0.0, 1.0)
    cl_col = c_sh * (1.0 - t_lit) + (c_mid * (1.0 - sun_prox * 0.56) + c_warm * (sun_prox * 0.56)) * t_lit
    cl_col += (cl_rim * (0.38 + 0.88 * glow_broad))[..., None] * (c_gold * 0.68)
    cl_col += (mottle * 0.012 + dabs * 0.010)[..., None]

    sky_mask = np.clip((hv + 0.03 - v) / 0.08, 0.0, 1.0)[..., None]
    sky = sky * (1.0 - cl_a[..., None] * 0.76 * sky_mask) + cl_col * (cl_a[..., None] * 0.76 * sky_mask)
    sky += (cirrus * (0.44 + 0.68 * glow_broad))[..., None] * np.array([1.02, 0.60, 0.38], dtype=np.float32) * sky_mask

    # =========================================================================
    # Layered Misty Autumn Mountain Gorge Ridges Framing the Upstream Canal Pass
    # =========================================================================
    u_r = np.linspace(0.0, 1.0, W, dtype=np.float32)

    def fbm_1d(rgen: np.random.Generator, octaves) -> np.ndarray:
        out = np.zeros(W, dtype=np.float32)
        for nk, amp in octaves:
            vals = rgen.standard_normal(nk).astype(np.float32)
            out += amp * np.interp(np.linspace(0, nk - 1, W), np.arange(nk), vals)
        return out

    def foliage_crown(rgen: np.random.Generator, freq: int, amp: float) -> np.ndarray:
        xs = np.linspace(0, freq, W, dtype=np.float32)
        base = np.abs(np.sin(xs * math.pi))
        jitter = np.interp(
            np.linspace(0, freq // 2, W),
            np.arange(freq // 2 + 1),
            rgen.uniform(0.35, 1.20, size=freq // 2 + 1).astype(np.float32),
        )
        fine = np.abs(np.sin(xs * math.pi * 2.37 + 0.8)) * 0.45
        return -amp * (0.64 * base + 0.36 * fine) * jitter

    mist_tex = resize_bicubic(cloud_fbm(rng, stretch_x=2.8), W, H)

    # Ridge 1: Distant hazy rose-violet gorge saddle behind the upstream bend
    r1_n = fbm_1d(rng, ((12, 0.55), (28, 0.30), (64, 0.15)))
    r1_h = (
        hv - 0.068
        - 0.095 * np.exp(-((u_r - 0.42) / 0.16) ** 2)
        - 0.110 * np.exp(-((u_r - 0.88) / 0.18) ** 2)
        - 0.036 * np.exp(-((u_r - 0.62) / 0.11) ** 2)
        - 0.015 * np.abs(r1_n)
    )
    a_r1 = np.clip((v - r1_h[None, :]) / 0.0040, 0.0, 1.0) * (v <= hv + 0.025).astype(np.float32)
    c_r1 = (
        np.array([0.36, 0.165, 0.225], dtype=np.float32)[None, None, :]
        + glow_broad[..., None] * np.array([0.30, 0.16, 0.105], dtype=np.float32)
    )
    sky = sky * (1.0 - a_r1[..., None]) + c_r1 * a_r1[..., None]

    # Warm apricot-rose valley mist ribbon drifting in the mountain pass
    gorge_mist1 = np.exp(-((v - (hv - 0.035)) / 0.036) ** 2) * np.clip(0.55 + 0.38 * mist_tex, 0.0, 1.15)
    sky += gorge_mist1[..., None] * np.array([0.46, 0.24, 0.18], dtype=np.float32) * (0.45 + 0.68 * glow_broad[..., None])

    # Ridge 2: Mid-distant autumn-clad left & right gorge walls converging smoothly to VP
    r2_n = fbm_1d(rng, ((14, 0.48), (36, 0.30), (92, 0.22)))
    gorge_v = (
        0.135 * np.power(np.clip((0.56 - u_r) / 0.34, 0.0, 1.0), 1.35)
        + 0.205 * np.power(np.clip((u_r - 0.65) / 0.35, 0.0, 1.0), 1.15)
    )
    r2_h = hv - 0.024 - gorge_v - 0.013 * np.abs(r2_n) + foliage_crown(rng, freq=140, amp=0.009)
    a_r2 = np.clip((v - r2_h[None, :]) / 0.0020, 0.0, 1.0) * (v <= hv + 0.030).astype(np.float32)
    aut_patch = np.clip(0.32 + 0.48 * mottle + 0.28 * dabs, 0.0, 1.0)
    c_r2 = (
        np.array([0.185, 0.082, 0.128], dtype=np.float32)[None, None, :]
        + aut_patch[..., None] * np.array([0.175, 0.050, 0.036], dtype=np.float32)
        + glow_broad[..., None] * np.array([0.16, 0.082, 0.056], dtype=np.float32)
    )
    sky = sky * (1.0 - a_r2[..., None]) + c_r2 * a_r2[..., None]

    # Ridge 3: Steep right-side near autumn maple gorge cliff framing the upper-right bank!
    r3_n = fbm_1d(rng, ((16, 0.45), (44, 0.30), (120, 0.25)))
    cliff_ramp = np.clip((u_r - 0.72) / 0.28, 0.0, 1.0)
    r3_h = (
        hv + 0.012
        - 0.275 * np.power(cliff_ramp, 0.82)
        - 0.055 * np.exp(-((u_r - 0.86) / 0.09) ** 2)
        - 0.016 * np.abs(r3_n) * np.clip((u_r - 0.70) / 0.08, 0.0, 1.0)
        + foliage_crown(rng, freq=98, amp=0.015) * np.clip((u_r - 0.71) / 0.07, 0.0, 1.0)
    )
    a_r3 = np.clip((v - r3_h[None, :]) / 0.0018, 0.0, 1.0) * np.clip((u_r[None, :] - 0.705) / 0.035, 0.0, 1.0)
    fol_hill = np.clip(0.45 + 0.55 * mottle + 0.32 * dabs, 0.0, 1.0) * np.clip((hv + 0.08 - v) / 0.25, 0.0, 1.0)
    c_r3 = (
        np.array([0.072, 0.036, 0.056], dtype=np.float32)[None, None, :]
        + fol_hill[..., None] * np.array([0.30, 0.072, 0.042], dtype=np.float32)
        + glow_broad[..., None] * np.array([0.095, 0.045, 0.030], dtype=np.float32)
        + (mottle * 0.010 + dabs * 0.009)[..., None]
    )
    rim_r3 = np.exp(-((v - r3_h[None, :]) / 0.0055) ** 2) * np.clip((u_r[None, :] - 0.72) / 0.08, 0.0, 1.0)
    c_r3 += rim_r3[..., None] * np.array([0.48, 0.20, 0.095], dtype=np.float32)
    sky = sky * (1.0 - a_r3[..., None]) + c_r3 * a_r3[..., None]

    # Soft misty veil over upstream canal gorge entrance
    pass_mist = (
        np.exp(-((v - hv) / 0.036) ** 2)
        * np.exp(-((u - 0.59) / 0.18) ** 2)
        * np.clip(0.55 + 0.40 * mist_tex, 0.0, 1.15)
    )
    sky += pass_mist[..., None] * np.array([0.34, 0.18, 0.15], dtype=np.float32)

    sky += (mottle * 0.008 + dabs * 0.007 + tooth * 0.004)[..., None]
    fog_rgb = np.array([0.34, 0.165, 0.180], dtype=np.float32)
    meta = {"sun_uv": (sun_u, sun_v), "horizon_v": hv, "r3_h": r3_h}
    return np.clip(sky, 0.0, 1.48), fog_rgb, meta


# =============================================================================
# Stage 1 & 2 — Authentic Coursed Wet Basalt Promenade & Ishigaki Retaining Walls
# =============================================================================

def paint_voronoi_ishigaki_uv(
    rng: np.random.Generator,
    u_span: tuple[float, float],
    v_span: tuple[float, float],
    plane_kind: str,  # "promenade" | "left_ishigaki" | "right_ishigaki"
    tw: int = 1024,
    th: int = 768,
) -> np.ndarray:
    """
    Authentic Japanese coursed weathered volcanic basalt masonry with subtle metric
    3x3 Voronoi (d2 - d1) hand-hewn ashlar/boulder stone joints (NEVER giraffe/mud cracks!):
      - "promenade": staggered Kyoto basalt flagstones with thin recessed wet mortar,
        rain-slicked glowing amber puddles reflecting shoji & chochin lanterns, dressed
        outer granite stone coping curb, and Stage 2 overhanging eave shadows.
      - "left_ishigaki" / "right_ishigaki": horizontal coursed dry-stacked basalt canal
        retaining walls with staggered hand-hewn stone blocks, thin realistic shadowed
        crevices, damp waterline darkening, velvety emerald-olive moss tufts in wet joints,
        and upward shimmering amber/teal thermal canal bounce light!
    """
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    u = xx / (tw - 1)
    v = yy / (th - 1)
    a_m = np.float32(u_span[0]) + u * np.float32(u_span[1] - u_span[0])
    b_m = np.float32(v_span[0]) + v * np.float32(v_span[1] - v_span[0])

    # Gentle organic hand-hewn chisel perturbation in meters
    nx = resize_bicubic(rng.standard_normal((max(4, th // 18), max(4, tw // 18))).astype(np.float32), tw, th) * np.float32(0.011)
    ny = resize_bicubic(rng.standard_normal((max(4, th // 18), max(4, tw // 18))).astype(np.float32), tw, th) * np.float32(0.009)
    px, py = a_m + nx, b_m + ny

    if plane_kind == "promenade":
        # Coursed wet Kyoto slate flagstones (wide slabs running along Y, staggered across X)
        cell_a, cell_b = np.float32(0.58), np.float32(0.84)
        j_a_lo, j_a_hi = 0.42, 0.58
        j_b_lo, j_b_hi = 0.44, 0.56
        stagger_x = 0.48
        crevice_w = 0.011
        crevice_edge = 0.004
    else:
        # Authentic horizontal coursed Japanese ishigaki basalt canal masonry (crisp horizontal courses!)
        cell_a, cell_b = np.float32(0.68), np.float32(0.24)
        j_a_lo, j_a_hi = 0.38, 0.62
        j_b_lo, j_b_hi = 0.46, 0.54
        stagger_x = 0.50
        crevice_w = 0.009
        crevice_edge = 0.003

    iy = np.floor(py / cell_b).astype(np.int32)
    # Apply running-bond course stagger along a_m so blocks stack naturally like real ashlar/ishigaki!
    px_st = px + (iy & 1).astype(np.float32) * (cell_a * np.float32(stagger_x))
    ix = np.floor(px_st / cell_a).astype(np.int32)

    jx_tab = rng.uniform(j_a_lo, j_a_hi, size=256).astype(np.float32)
    jy_tab = rng.uniform(j_b_lo, j_b_hi, size=256).astype(np.float32)
    tone_tab = rng.uniform(-0.028, 0.034, size=256).astype(np.float32)
    warm_tab = rng.uniform(-0.010, 0.016, size=256).astype(np.float32)
    sheen_tab = rng.uniform(0.76, 1.24, size=256).astype(np.float32)

    d1 = np.full((th, tw), 1e9, dtype=np.float32)
    d2 = np.full((th, tw), 1e9, dtype=np.float32)
    best_id = np.zeros((th, tw), dtype=np.int32)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cy = iy + dy
            cx = ix + dx
            h_id = ((cx * 73856093) ^ (cy * 19349663)) & 255
            sx = (cx + jx_tab[h_id]) * cell_a - (cy & 1).astype(np.float32) * (cell_a * np.float32(stagger_x))
            sy = (cy + jy_tab[h_id]) * cell_b
            # Metric anisotropic norm keeps horizontal ashlar courses authentically stacked!
            dist = np.hypot((px - sx), (py - sy) * np.float32(1.0 if plane_kind == "promenade" else 2.25))
            closer = dist < d1
            d2 = np.where(closer, d1, np.minimum(d2, dist))
            d1 = np.where(closer, dist, d1)
            best_id = np.where(closer, h_id, best_id)

    # Subtle realistic thin shadowed damp mortar joint (ZERO thick cartoon black giraffe stripes!)
    delta_d = d2 - d1
    mortar = np.clip((delta_d - np.float32(crevice_edge)) / np.float32(crevice_w), 0.0, 1.0)
    mortar_soft = 0.38 + 0.62 * mortar
    bevel = np.clip((delta_d - np.float32(crevice_edge + crevice_w * 0.6)) / np.float32(0.032), 0.0, 1.0)

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=88)
    tooth = paper_tooth(th, tw, rng)

    sv = tone_tab[best_id]
    sw = warm_tab[best_id]
    ss = sheen_tab[best_id]

    stone = np.zeros((th, tw, 3), dtype=np.float32)
    stone[:, :, 0] = 0.088 + sv * 0.42 + sw * 0.44 + mottle * 0.014 + dabs * 0.011
    stone[:, :, 1] = 0.072 + sv * 0.36 + sw * 0.18 + mottle * 0.012 + dabs * 0.009
    stone[:, :, 2] = 0.082 + sv * 0.38 + mottle * 0.012 + dabs * 0.009

    # Subtle chiselled top-edge bevel highlight per stone block
    crevice_col = np.array([0.022, 0.016, 0.020], dtype=np.float32)
    rgb = mortar_soft[..., None] * stone + (1.0 - mortar_soft[..., None]) * crevice_col
    rgb += (bevel * (1.0 - bevel) * 4.0 * mortar)[..., None] * np.array([0.026, 0.018, 0.018], dtype=np.float32)

    if plane_kind == "promenade":
        wx, wy = a_m, b_m
        wz = np.full_like(wx, Z_PROM)
        vis = trace_onsen_sun_visibility(wx, wy, wz)
        lantern, shoji = eval_onsen_point_lights(wx, wy, wz, is_horizontal=True)

        # Dressed cut-granite outer curb border along canal edge (x in [-0.26, 0.0])
        is_curb = (wx > -0.26).astype(np.float32)
        curb_j = np.clip(np.minimum(np.mod(wy / 0.85, 1.0), 1.0 - np.mod(wy / 0.85, 1.0)) / 0.022, 0.0, 1.0)
        curb_col = (np.array([0.112, 0.090, 0.094], dtype=np.float32) + mottle[..., None] * 0.012) * (0.48 + 0.52 * curb_j[..., None])
        curb_inner_seam = np.clip(np.abs(wx - (-0.26)) / 0.016, 0.0, 1.0)
        rgb = (rgb * (1.0 - is_curb[..., None]) + curb_col * is_curb[..., None]) * (0.42 + 0.58 * curb_inner_seam[..., None])

        # Smooth glossy rainwater puddles reflecting warm amber shoji & twilight sky!
        puddle_n = resize_bicubic(fbm_gouache(th // 2, tw // 2, rng, scales=((30, 28), (14, 15), (6, 6), (3, 3))), tw, th)
        puddle = np.clip((puddle_n + 0.14 + 0.12 * np.sin(wy * 0.52 + wx * 0.7)) * 0.92, 0.0, 1.0) * mortar_soft
        puddle = puddle * puddle * (3.0 - 2.0 * puddle)

        rgb *= (0.42 + 0.58 * vis[..., None])
        y_n = np.clip((wy - v_span[0]) / max(1.0, v_span[1] - v_span[0]), 0.0, 1.0)
        sky_sheen = (0.18 + 0.68 * puddle) * (0.35 + 0.65 * vis) * ss * 0.54
        sheen_c = (
            np.array([0.24, 0.145, 0.205], dtype=np.float32)[None, None, :] * (1.0 - y_n[..., None])
            + np.array([0.62, 0.30, 0.20], dtype=np.float32)[None, None, :] * y_n[..., None]
        )
        rgb += sky_sheen[..., None] * sheen_c * 0.52

        warm_pool = lantern * 0.56 + shoji * 0.50
        rgb += (warm_pool * (0.36 + 0.64 * puddle) * mortar_soft)[..., None] * np.array([0.96, 0.50, 0.18], dtype=np.float32) * 0.64
        lip_hi = np.clip((wx - (-0.06)) / 0.06, 0.0, 1.0) * mortar_soft
        rgb += lip_hi[..., None] * np.array([0.36, 0.18, 0.11], dtype=np.float32) * (0.45 + 0.55 * vis[..., None])
    else:
        wy, wz = a_m, b_m
        wx = np.full_like(wy, X_LEFT_BANK if plane_kind == "left_ishigaki" else X_RIGHT_BANK)
        vis = trace_onsen_sun_visibility(wx, wy, wz)
        lantern, shoji = eval_onsen_point_lights(wx, wy, wz, is_horizontal=False)

        # Damp waterline darkening & subtle velvety olive-emerald moss tucked in damp mortar crevices!
        wet_zone = np.clip((0.40 - wz) / 0.40, 0.0, 1.0)
        moss_patch = np.clip(0.35 + 0.55 * mottle + 0.25 * dabs, 0.0, 1.0)
        moss = np.clip((0.54 - wz + 0.18 * mottle) / 0.34, 0.0, 1.0) * moss_patch * (0.32 + 0.68 * (1.0 - mortar))
        rgb *= (1.0 - 0.52 * wet_zone[..., None])
        moss_col = np.array([0.042, 0.076, 0.044], dtype=np.float32)
        rgb = rgb * (1.0 - 0.52 * moss[..., None]) + moss_col * (0.52 * moss[..., None])

        # Under-coping overhang ambient occlusion shadow
        z_top = np.float32(Z_PROM if plane_kind == "left_ishigaki" else Z_RIGHT_BANK)
        coping_ao = np.exp(-np.maximum(0.0, z_top - wz) / 0.13)
        rgb *= (1.0 - 0.52 * coping_ao[..., None])

        # Directional shading + warm amber glow from canal lanterns
        rgb *= (0.48 + 0.52 * vis[..., None])
        rgb += (lantern * 0.42 + shoji * 0.30)[..., None] * (0.45 + 0.55 * bevel[..., None]) * mortar_soft[..., None] * np.array([0.78, 0.38, 0.15], dtype=np.float32)

        # Upward shimmering thermal river reflection bounce kissing lower stone courses
        water_bounce = np.exp(-wz / 0.45) * (0.48 + 0.52 * np.sin(wy * 3.8 + mottle * 2.5)) * mortar_soft
        if plane_kind == "left_ishigaki":
            rgb += water_bounce[..., None] * np.array([0.16, 0.15, 0.13], dtype=np.float32)
            spout_wet = np.exp(-((wy - SPOUT_Y) / 0.38) ** 2) * (wz < SPOUT_Z + 0.10).astype(np.float32)
            rgb *= (1.0 - 0.46 * spout_wet[..., None])
            rgb += (spout_wet * np.exp(-wz / 0.35))[..., None] * np.array([0.22, 0.16, 0.11], dtype=np.float32)
        else:
            rgb += water_bounce[..., None] * np.array([0.28, 0.15, 0.080], dtype=np.float32)

    rgb = apply_warm_shadow_bounce(rgb, strength=0.015)
    rgb += tooth[..., None] * 0.0055
    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.38)
    return rgba


def paint_ryokan_facade_uv(
    rng: np.random.Generator,
    y_span: tuple[float, float],
    z_span: tuple[float, float],
    story_mode: str,  # "lower_2f" | "loft_3f"
    tw: int = 1600,
    th: int = 720,
) -> np.ndarray:
    """
    Towering 3-story Ginzan Onsen / Spirited Away dark-timber & warm-plaster ryokan facade:
      - 1F: Charred-cedar (yakisugi) board-and-batten siding, stone plinth, glowing shoji
            lounge bays, grand sliding entrance door, and warm amber backlit cedar bath
            exhaust louver (yuge-nuki).
      - 2F: Warm twilight shikkui plaster framed in aged dark-cedar beams with vermilion
            lintel trim and richly glowing honey-amber kumiko shoji guestroom windows with
            unique softly-blurred warm interior silhouettes!
      - 3F: Recessed traditional bathhouse loft clerestory with glowing shoji bays AND
            an iconic glowing circular marumado round lattice window!
    """
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    wy = np.float32(y_span[0]) + (xx / (tw - 1)) * np.float32(y_span[1] - y_span[0])
    wz = np.float32(z_span[0]) + (yy / (th - 1)) * np.float32(z_span[1] - z_span[0])
    wx = np.full_like(wy, X_RYOKAN_WALL if story_mode == "lower_2f" else X_LOFT_WALL)

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=90)
    tooth = paper_tooth(th, tw, rng)

    b_w = np.float32(0.19)
    bid = np.floor(wy / b_w).astype(np.int32)
    bf = (wy / b_w) - bid
    batten = (np.abs(bf - 0.5) > 0.42).astype(np.float32)
    b_seam = np.clip(np.minimum(bf, 1.0 - bf) / 0.050, 0.0, 1.0)
    bt = rng.uniform(-0.016, 0.022, size=256).astype(np.float32)[bid & 255]
    grain = np.sin(wy * 50.0 + mottle * 3.5 + wz * 1.8) * 0.006

    wall = np.zeros((th, tw, 3), dtype=np.float32)
    wall[:, :, 0] = 0.068 + bt * 0.45 + mottle * 0.014 + dabs * 0.011 - batten * 0.011 + grain
    wall[:, :, 1] = 0.036 + bt * 0.25 + mottle * 0.009 + dabs * 0.007 - batten * 0.006 + grain * 0.5
    wall[:, :, 2] = 0.038 + bt * 0.24 + mottle * 0.008 + dabs * 0.006 - batten * 0.005 + grain * 0.4
    wall = b_seam[..., None] * wall + (1.0 - b_seam[..., None]) * np.array([0.016, 0.008, 0.010], dtype=np.float32)

    # Warm dusk-toned shikkui lime plaster on 2F & 3F stories (rich warm amber-ochre tint, never cold grey!)
    plaster = np.zeros((th, tw, 3), dtype=np.float32)
    plaster[:, :, 0] = 0.252 + mottle * 0.024 + dabs * 0.018
    plaster[:, :, 1] = 0.148 + mottle * 0.016 + dabs * 0.012
    plaster[:, :, 2] = 0.114 + mottle * 0.013 + dabs * 0.010

    if story_mode == "lower_2f":
        plinth = (wz < 1.46).astype(np.float32)
        pl_j = np.clip(np.minimum(np.mod(wy / 0.58, 1.0), 1.0 - np.mod(wy / 0.58, 1.0)) / 0.035, 0.0, 1.0)
        pl_col = (np.array([0.058, 0.040, 0.044], dtype=np.float32) + mottle[..., None] * 0.010) * (0.42 + 0.58 * pl_j[..., None])
        wall = wall * (1.0 - plinth[..., None]) + pl_col * plinth[..., None]

        is_2f = (wz >= Z_2F_FLOOR + 0.04).astype(np.float32)
        wall = wall * (1.0 - is_2f[..., None]) + plaster * is_2f[..., None]
        dado_2f = ((wz >= Z_1F_TOP) & (wz < Z_2F_FLOOR + 0.30)).astype(np.float32)
        wall = wall * (1.0 - dado_2f[..., None]) + np.array([0.046, 0.022, 0.024], dtype=np.float32) * dado_2f[..., None]
        # Solid warm dark smoked-cedar eave wall frieze sealing cleanly into the 2F soffit (ZERO twilight sky gap!)
        eave_frieze_2f = (wz >= 5.46).astype(np.float32)
        frieze_c = (
            np.array([0.095, 0.046, 0.036], dtype=np.float32)
            + (mottle * 0.008 + grain * 0.6)[..., None]
        )
        wall = wall * (1.0 - eave_frieze_2f[..., None]) + frieze_c * eave_frieze_2f[..., None]
    else:
        wall = plaster * 0.88
        dado_3f = (wz < Z_2F_TOP + 0.36).astype(np.float32)
        wall = wall * (1.0 - dado_3f[..., None]) + np.array([0.044, 0.022, 0.025], dtype=np.float32) * dado_3f[..., None]
        eave_frieze_3f = (wz >= 7.22).astype(np.float32)
        wall = wall * (1.0 - eave_frieze_3f[..., None]) + np.array([0.072, 0.036, 0.032], dtype=np.float32) * eave_frieze_3f[..., None]

    # Heavy structural dark-cedar timber framing
    beams = np.zeros_like(wy)
    if story_mode == "lower_2f":
        for zb, hw in ((1.48, 0.055), (3.24, 0.105), (Z_2F_FLOOR, 0.095), (5.34, 0.078), (5.58, 0.090)):
            beams = np.maximum(beams, (np.abs(wz - zb) < hw).astype(np.float32))
        vermil_trim = (np.abs(wz - 5.44) < 0.038).astype(np.float32)
    else:
        for zb, hw in ((Z_2F_TOP + 0.06, 0.070), (5.92, 0.055), (7.02, 0.072), (7.28, 0.095)):
            beams = np.maximum(beams, (np.abs(wz - zb) < hw).astype(np.float32))
        vermil_trim = (np.abs(wz - 7.12) < 0.044).astype(np.float32)

    posts = np.zeros_like(wy)
    for py_post in RYOKAN_POST_YS:
        posts = np.maximum(posts, (np.abs(wy - py_post) < 0.125).astype(np.float32))
    for py_mid in (3.40, 7.00, 10.60, 14.20):
        posts = np.maximum(posts, ((np.abs(wy - py_mid) < 0.052) & (wz > 5.26 if story_mode == "lower_2f" else wz > 5.6)).astype(np.float32))

    frame_m = np.clip(beams + posts, 0.0, 1.0)[..., None]
    frame_c = np.array([0.072, 0.036, 0.032], dtype=np.float32) + mottle[..., None] * 0.008
    wall = wall * (1.0 - frame_m * 0.88) + frame_c * (frame_m * 0.88)
    wall = wall * (1.0 - vermil_trim[..., None] * 0.72) + np.array([0.46, 0.132, 0.082], dtype=np.float32) * (vermil_trim[..., None] * 0.72)

    vis = trace_onsen_sun_visibility(wx, wy, wz)
    lantern, _shoji = eval_onsen_point_lights(wx, wy, wz, is_horizontal=False)
    wall *= (0.42 + 0.58 * vis[..., None])
    wall += (vis * b_seam)[..., None] * np.array([0.088, 0.038, 0.024], dtype=np.float32)
    wall += lantern[..., None] * np.array([0.46, 0.215, 0.072], dtype=np.float32)

    dy_m = float((y_span[1] - y_span[0]) / max(1, tw - 1))
    dz_m = float((z_span[1] - z_span[0]) / max(1, th - 1))

    def aa_bar(coord: np.ndarray, center: float, half_w: float, px_m: float) -> np.ndarray:
        d = np.abs(coord - np.float32(center))
        t = np.clip((np.float32(half_w + 0.90 * px_m) - d) / np.float32(1.80 * px_m), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def paint_shoji_bay(
        wy0: float,
        wy1: float,
        wz0: float,
        wz1: float,
        arch_style: str,
        warmth: float = 1.0,
        lamp_uv: tuple[float, float] = (0.46, 0.64),
        hue_shift: tuple[float, float, float] = (1.0, 1.0, 1.0),
        is_upper_floor: bool = False,
    ) -> None:
        """
        Hand-painted authentic Ginzan Onsen / Spirited Away ryokan shoji & koushi carpentry:
          - Deeply recessed 3D outer timber frame (kamoi top lintel, projecting shikii bottom
            sill, perspective-accurate recessed far-jamb & lintel/sill reveals catching warm
            interior amber light, NEVER stark flat #000000 black!).
          - Genuine architectural variety: koshita cedar lower kick-panels, fine vertical
            ranma transom slats, slender koushi-do entrance cedar lattice, overlapping sliding
            yukimi-shoji doors with delicate smoked-cedar kumiko muntins & optical warm
            amber light-bleed halation, AND a partially slid-open 2F guestroom revealing a
            shadowy cedar interior, glowing hanging paper pendant lamp & lit tatami depth!
          - Multi-layer translucent handmade washi paper glow with organic horizontal kozo
            fiber texture, per-pane subtle tone patina, soft Gaussian-blurred interior room
            silhouettes, and warm amber light spill onto the surrounding plaster/cedar wall!
        """
        nonlocal wall
        pad = 0.34
        ix0 = max(0, int(math.floor((wy0 - pad - y_span[0]) / dy_m)))
        ix1 = min(tw, int(math.ceil((wy1 + pad - y_span[0]) / dy_m)) + 1)
        iz0 = max(0, int(math.floor((wz0 - pad - z_span[0]) / dz_m)))
        iz1 = min(th, int(math.ceil((wz1 + pad - z_span[0]) / dz_m)) + 1)
        if ix0 >= ix1 or iz0 >= iz1:
            return

        sy = wy[iz0:iz1, ix0:ix1]
        sz = wz[iz0:iz1, ix0:ix1]
        sm = mottle[iz0:iz1, ix0:ix1]
        sd = dabs[iz0:iz1, ix0:ix1]
        slan = lantern[iz0:iz1, ix0:ix1]
        sub_wall = wall[iz0:iz1, ix0:ix1].copy()

        bw = max(0.05, wy1 - wy0)
        bh = max(0.05, wz1 - wz0)
        fu = (sy - np.float32(wy0)) / np.float32(bw)
        fv = (sz - np.float32(wz0)) / np.float32(bh)

        # Warm smoked-cedar carpentry palette (#2d1a14 -> #4a2d21, light-responsive, zero flat black!)
        wood_grain_v = np.sin(sy * np.float32(72.0) + sm * np.float32(3.2) + sz * np.float32(2.4)) * np.float32(0.012)
        wood_grain_h = np.sin(sz * np.float32(82.0) + sm * np.float32(2.8) + sy * np.float32(1.8)) * np.float32(0.012)
        cedar_dark = np.array([0.155, 0.082, 0.058], dtype=np.float32)[None, None, :]
        cedar_mid = np.array([0.235, 0.128, 0.086], dtype=np.float32)[None, None, :]
        cedar_warm = np.array([0.335, 0.188, 0.118], dtype=np.float32)[None, None, :]

        # ---------------------------------------------------------------------
        # 1. Exterior Wall Ambient Occlusion & Warm Amber Light Spill Surround
        # ---------------------------------------------------------------------
        fw_side = np.float32(0.082)
        fw_top = np.float32(0.098)
        fw_bot = np.float32(0.086)
        lintel_ear = np.float32(0.068)  # kamoi & shikii extend gracefully past jambs

        dx_open = np.maximum(np.float32(0.0), np.maximum(np.float32(wy0) - sy, sy - np.float32(wy1)))
        dz_open = np.maximum(np.float32(0.0), np.maximum(np.float32(wz0) - sz, sz - np.float32(wz1)))
        d_open = np.hypot(dx_open, dz_open)

        in_open = (sy >= wy0) & (sy <= wy1) & (sz >= wz0) & (sz <= wz1)
        in_jamb_l = (sy >= wy0 - fw_side) & (sy < wy0) & (sz >= wz0 - fw_bot) & (sz <= wz1 + fw_top)
        in_jamb_r = (sy > wy1) & (sy <= wy1 + fw_side) & (sz >= wz0 - fw_bot) & (sz <= wz1 + fw_top)
        in_kamoi = (sy >= wy0 - fw_side - lintel_ear) & (sy <= wy1 + fw_side + lintel_ear) & (sz > wz1) & (sz <= wz1 + fw_top)
        in_shikii = (sy >= wy0 - fw_side - lintel_ear * 0.85) & (sy <= wy1 + fw_side + lintel_ear * 0.85) & (sz >= wz0 - fw_bot) & (sz < wz0)
        in_frame = in_jamb_l | in_jamb_r | in_kamoi | in_shikii

        # Soft exterior shadow tucked right under sill/lintel + warm golden wall spill
        ext_mask = (~in_open) & (~in_frame)
        under_sill_ao = np.exp(-np.maximum(np.float32(0.0), np.float32(wz0 - fw_bot) - sz) / np.float32(0.065)) * (
            (sy >= wy0 - fw_side - lintel_ear) & (sy <= wy1 + fw_side + lintel_ear) & (sz < wz0 - fw_bot)
        ).astype(np.float32)
        side_ao = np.exp(-d_open / np.float32(0.055)) * ext_mask.astype(np.float32)
        sub_wall *= (1.0 - 0.34 * under_sill_ao[..., None] - 0.14 * side_ao[..., None])

        spill_env = np.exp(-((d_open / np.float32(0.24)) ** np.float32(1.35))) * ext_mask.astype(np.float32)
        spill_bias = 0.65 + 0.35 * np.clip((sz - np.float32(wz0 - 0.18)) / np.float32(bh + 0.25), 0.0, 1.0)
        sub_wall += (spill_env * spill_bias)[..., None] * np.array([0.23, 0.112, 0.038], dtype=np.float32) * np.float32(warmth)

        # ---------------------------------------------------------------------
        # 2. Painterly Translucent Washi Paper Glow & Soft Interior Silhouettes
        # ---------------------------------------------------------------------
        lu, lv = lamp_uv
        r_lamp = np.hypot((fu - np.float32(lu)) * np.float32(1.06), (fv - np.float32(lv)) * np.float32(1.12))
        lamp_core = np.exp(-((r_lamp / np.float32(0.42)) ** 2))
        lamp_broad = np.exp(-((r_lamp / np.float32(0.84)) ** 2))

        # Organic handmade kozo washi paper fibers + subtle per-pane paste variation
        washi_fiber_h = (
            np.sin(sz * np.float32(175.0) + sm * np.float32(5.2)) * np.float32(0.016)
            + np.sin(sz * np.float32(64.0) - sm * np.float32(3.1) + sy * np.float32(2.0)) * np.float32(0.014)
        )
        washi_mottle = np.clip(sm * np.float32(0.042) + sd * np.float32(0.022), -0.065, 0.065)

        # Soft overhang shadow from top kamoi lintel + gentle edge darkening
        top_ao = np.exp(-np.maximum(np.float32(0.0), np.float32(1.0) - fv) / np.float32(0.115))
        corner_vign = np.clip(
            np.minimum(np.minimum(fu, np.float32(1.0) - fu), np.minimum(fv, np.float32(1.0) - fv)) / np.float32(0.14),
            0.0, 1.0,
        )
        hue_v = np.array(hue_shift, dtype=np.float32)[None, None, :]

        # Rich honey-cream-gold (#fff0c8 -> #f7b85c -> #c46834) layered translucent paper
        pane_c = (
            np.array([0.74, 0.345, 0.135], dtype=np.float32)[None, None, :]
            + lamp_broad[..., None] * np.array([0.34, 0.32, 0.165], dtype=np.float32)[None, None, :]
            + lamp_core[..., None] * np.array([0.24, 0.34, 0.32], dtype=np.float32)[None, None, :]
            - top_ao[..., None] * np.array([0.22, 0.145, 0.075], dtype=np.float32)[None, None, :]
            + (washi_fiber_h + washi_mottle)[..., None] * np.array([0.95, 0.72, 0.42], dtype=np.float32)[None, None, :]
        ) * np.float32(warmth) * hue_v * (np.float32(0.84) + np.float32(0.16) * corner_vign[..., None])

        # Soft Gaussian-blurred interior room silhouettes (unique per bay!)
        sil_near = np.zeros_like(fu, dtype=np.float32)
        sil_far = np.zeros_like(fu, dtype=np.float32)
        if arch_style == "1f_lounge_near":
            # Hanging cylindrical andon shade + cord, low chabudai tea-table + teapot, fusuma screen on right, rafters
            cord = (np.abs(fu - 0.38) < 0.012) & (fv > 0.74)
            pendant = (np.abs(fu - 0.38) < 0.068) & (fv > 0.62) & (fv < 0.76)
            table = (fv > 0.23) & (fv < 0.37) & (fu > 0.18) & (fu < 0.58)
            teapot = np.hypot((fu - 0.34) * 1.6, (fv - 0.41) * 1.4) < 0.075
            fusuma_edge = (fu > 0.78) & (fv < 0.84)
            rafters = (fv > 0.84) & ((np.abs(fu - 0.22) < 0.04) | (np.abs(fu - 0.56) < 0.04) | (fv > 0.91))
            sil_near = np.maximum(sil_near, (table | teapot).astype(np.float32) * 0.55)
            sil_near = np.maximum(sil_near, (cord | pendant).astype(np.float32) * 0.28)
            sil_far = np.maximum(sil_far, fusuma_edge.astype(np.float32) * 0.44)
            sil_far = np.maximum(sil_far, rafters.astype(np.float32) * 0.42)
        elif arch_style == "1f_entry":
            # Bathhouse entrance vestibule: warm flanking cedar partitions, high hanging globe lantern, reception bandai
            flanks = ((fu < 0.18) | (fu > 0.82)) & (fv < 0.86)
            bandai = (fu > 0.62) & (fu < 0.84) & (fv < 0.48)
            globe = np.hypot((fu - 0.50) * 1.35, (fv - 0.76) * 1.35) < 0.085
            beam = (fv > 0.86)
            sil_near = np.maximum(sil_near, bandai.astype(np.float32) * 0.42)
            sil_far = np.maximum(sil_far, flanks.astype(np.float32) * 0.40)
            sil_far = np.maximum(sil_far, (globe | beam).astype(np.float32) * 0.32)
        elif arch_style == "1f_lounge_mid":
            # Graceful diagonal bonsai pine branch & foliage silhouette + cozy seated guest shadow
            branch = (np.abs((fv - 0.24) - (fu - 0.42) * 0.92) < 0.028) & (fu > 0.40) & (fu < 0.84)
            twig = (np.abs((fv - 0.52) + (fu - 0.68) * 0.65) < 0.022) & (fu > 0.52) & (fu < 0.78)
            bonsai_c1 = np.hypot((fu - 0.68) * 1.4, (fv - 0.58) * 1.7) < 0.14
            bonsai_c2 = np.hypot((fu - 0.82) * 1.5, (fv - 0.66) * 1.8) < 0.11
            pot = (fu > 0.38) & (fu < 0.54) & (fv > 0.20) & (fv < 0.30)
            sil_near = np.maximum(sil_near, (branch | twig | bonsai_c1 | bonsai_c2 | pot).astype(np.float32) * 0.56)
            sil_far = np.maximum(sil_far, ((fu < 0.28) & (fv < 0.78)).astype(np.float32) * 0.38)
        elif arch_style == "2f_near":
            # Silhouetted bamboo/maple spray in tokonoma vase + hanging cylindrical lantern + ceiling beam
            stem = (np.abs((fu - 0.28) - (fv - 0.25) * 0.35) < 0.020) & (fv > 0.24) & (fv < 0.74)
            canopy = np.hypot((fu - 0.36) * 1.35, (fv - 0.62) * 1.55) < 0.13
            canopy2 = np.hypot((fu - 0.24) * 1.5, (fv - 0.54) * 1.7) < 0.095
            vase = (np.abs(fu - 0.27) < 0.048) & (fv > 0.22) & (fv < 0.38)
            post_in = (np.abs(fu - 0.76) < 0.036)
            sil_near = np.maximum(sil_near, (stem | canopy | canopy2 | vase).astype(np.float32) * 0.52)
            sil_far = np.maximum(sil_far, (post_in | (fv > 0.85)).astype(np.float32) * 0.40)
        elif arch_style == "2f_hero_open":
            # Left closed shoji panels show soft rafters + warm interior screen edge (right panel is slid OPEN!)
            rafter_l = (fv > 0.82) | ((fu < 0.14) & (fv < 0.82))
            table_l = (fu > 0.16) & (fu < 0.48) & (fv > 0.22) & (fv < 0.36)
            sil_far = np.maximum(sil_far, rafter_l.astype(np.float32) * 0.38)
            sil_near = np.maximum(sil_near, table_l.astype(np.float32) * 0.44)
        elif arch_style == "2f_mid":
            # Seated guest silhouette enjoying tea beside folding byobu screen
            guest_torso = np.hypot((fu - 0.36) * 1.7, (fv - 0.36) * 1.25) < 0.14
            guest_head = np.hypot((fu - 0.36) * 2.1, (fv - 0.52) * 1.9) < 0.072
            byobu = (fu > 0.68) & (fv > 0.20) & (fv < 0.68)
            pendant = (np.abs(fu - 0.42) < 0.055) & (fv > 0.66) & (fv < 0.78)
            sil_near = np.maximum(sil_near, (guest_torso | guest_head).astype(np.float32) * 0.52)
            sil_far = np.maximum(sil_far, (byobu | pendant | (fv > 0.84)).astype(np.float32) * 0.42)
        else:
            # 2f_far_crack & 3f_loft
            rafter = (fv > 0.82) | ((fu > 0.72) & (fv < 0.76))
            pendant = (np.abs(fu - lu) < 0.065) & (fv > 0.62) & (fv < 0.76)
            sil_far = np.maximum(sil_far, (rafter | pendant).astype(np.float32) * 0.42)

        sig_near = max(3.2, 0.022 / dy_m)
        sig_far = max(5.5, 0.044 / dy_m)
        sil_comb = np.clip(
            gaussian_blur_2d(sil_near, sig_near) * np.float32(0.72)
            + gaussian_blur_2d(sil_far, sig_far) * np.float32(0.58),
            0.0, 0.85,
        )
        sil_tint = np.array([0.23, 0.095, 0.048], dtype=np.float32)[None, None, :]
        pane_c = pane_c * (1.0 - sil_comb[..., None] * np.float32(0.56)) + sil_tint * (sil_comb[..., None] * np.float32(0.56))

        # ---------------------------------------------------------------------
        # 3. Architectural Joinery: Koshita Kick-Panels, Ranma Transoms,
        #    Koushi Slats, Delicate Kumiko Shoji, & Open 2F Interior Depth!
        # ---------------------------------------------------------------------
        wood_a = np.zeros_like(fu, dtype=np.float32)
        solid_wood_a = np.zeros_like(fu, dtype=np.float32)  # solid koshita/stiles (less edge bleed than thin kumiko)
        wood_col_var = np.zeros_like(fu, dtype=np.float32)
        open_mask = np.zeros_like(fu, dtype=np.float32)
        open_rgb = np.zeros((sy.shape[0], sy.shape[1], 3), dtype=np.float32)

        # Subtle pane-by-pane handmade washi tone variation
        def apply_pane_patina(u_divs: int, v_divs: int, v_lo: float, v_hi: float) -> None:
            nonlocal pane_c
            in_band = (fv >= v_lo) & (fv <= v_hi) & (fu >= 0.0) & (fu <= 1.0)
            ci = np.floor(np.clip(fu, 0.0, 0.999) * u_divs).astype(np.int32)
            ri = np.floor(np.clip((fv - v_lo) / max(0.01, v_hi - v_lo), 0.0, 0.999) * v_divs).astype(np.int32)
            ph = ((ci * 37 + ri * 91 + int(wy0 * 10)) & 63)
            p_jit = (np.sin(ph.astype(np.float32) * 1.73) * 0.036).astype(np.float32)
            pane_c = np.where(in_band[..., None], pane_c * (1.0 + p_jit[..., None]), pane_c)

        # Helper to paint cedar koshita lower wainscoting kick-panel (horizontal planks + top naka-zan rail!)
        def add_koshita(v_top: float, u_lo: float = 0.0, u_hi: float = 1.0, plank_n: int = 4) -> None:
            nonlocal wood_a, solid_wood_a, wood_col_var
            z_top = wz0 + v_top * bh
            m_u = np.clip((fu - np.float32(u_lo) + 0.004) / 0.008, 0.0, 1.0) * np.clip((np.float32(u_hi) - fu + 0.004) / 0.008, 0.0, 1.0)
            m_v = np.clip((np.float32(z_top) - sz + np.float32(dz_m)) / np.float32(2.0 * dz_m), 0.0, 1.0)
            km = m_u * m_v
            # Plank grooves & slight per-plank tone variation
            pf = np.mod(np.clip(fv / max(0.01, v_top), 0.0, 0.999) * plank_n, 1.0)
            pid = np.floor(np.clip(fv / max(0.01, v_top), 0.0, 0.999) * plank_n)
            p_seam = np.clip(np.minimum(pf, 1.0 - pf) / 0.09, 0.0, 1.0)
            # Top naka-zan wooden cap rail catching warm light from shoji directly above!
            rail_hi = aa_bar(sz, z_top - 0.014, 0.018, dz_m) * m_u
            wood_a = np.maximum(wood_a, km)
            solid_wood_a = np.maximum(solid_wood_a, km)
            wood_col_var += km * (-0.28 * (1.0 - p_seam) + 0.10 * np.sin(pid * 2.1 + wy0) + 0.36 * rail_hi)

        # Helper to paint delicate top ranma transom strip with fine vertical cedar slats
        def add_ranma(v_bot: float, n_slats: int = 20) -> None:
            nonlocal wood_a, solid_wood_a, wood_col_var
            z_bot = wz0 + v_bot * bh
            in_r = (sz >= z_bot) & (sz <= wz1)
            # Horizontal transom divider rail (naka-kamoi)
            rail = aa_bar(sz, z_bot, 0.022, dz_m)
            wood_a = np.maximum(wood_a, rail)
            solid_wood_a = np.maximum(solid_wood_a, rail * 0.85)
            wood_col_var += rail * 0.16
            # Fine vertical ranma slats
            step_y = bw / n_slats
            for k in range(1, n_slats):
                yk = wy0 + k * step_y
                sl = aa_bar(sy, yk, 0.0085, dy_m) * in_r.astype(np.float32)
                wood_a = np.maximum(wood_a, sl * 0.94)
            # Delicate central horizontal kumiko tie inside ranma
            mid_r = aa_bar(sz, 0.5 * (z_bot + wz1), 0.0075, dz_m)
            wood_a = np.maximum(wood_a, mid_r * 0.88)

        # Helper to add slender cedar kumiko shoji grid inside a sliding sash [u0..u1, v0..v1]
        def add_sash_kumiko(
            u0: float, u1: float, v0: float, v1: float,
            n_c: int, n_r: int, stile_w: float = 0.032, bar_w: float = 0.0105,
            yukimi_glass_v: tuple[float, float] | None = None,
        ) -> None:
            nonlocal wood_a, solid_wood_a, wood_col_var, pane_c
            y_a, y_b = wy0 + u0 * bw, wy0 + u1 * bw
            z_a, z_b = wz0 + v0 * bh, wz0 + v1 * bh
            in_s = (sy >= y_a - 0.005) & (sy <= y_b + 0.005) & (sz >= z_a - 0.005) & (sz <= z_b + 0.005)
            in_sf = in_s.astype(np.float32)

            # Optional clear warm yukimi-shoji viewing glass strip near lower rail!
            if yukimi_glass_v is not None:
                gv0, gv1 = yukimi_glass_v
                zg0, zg1 = wz0 + gv0 * bh, wz0 + gv1 * bh
                in_gl = (sy > y_a + stile_w) & (sy < y_b - stile_w) & (sz > zg0) & (sz < zg1)
                # Warm darker interior tatami/room depth + subtle diagonal window glass reflection!
                gl_t = np.clip((sz - zg0) / max(0.01, zg1 - zg0), 0.0, 1.0)
                glass_depth = (
                    np.array([0.42, 0.22, 0.095], dtype=np.float32)[None, None, :] * (1.0 - gl_t[..., None])
                    + np.array([0.62, 0.34, 0.155], dtype=np.float32)[None, None, :] * gl_t[..., None]
                ) * np.float32(warmth)
                spec_streak = np.exp(-(((fu - 0.5 * (u0 + u1)) * 2.2 - (gl_t - 0.5) * 0.9) / 0.16) ** 2) * 0.18
                glass_depth += spec_streak[..., None] * np.array([0.96, 0.72, 0.44], dtype=np.float32)
                pane_c = np.where(in_gl[..., None], glass_depth, pane_c)
                # Wooden muntin rail bordering top & bottom of glass viewing strip
                for zg in (zg0, zg1):
                    gr = aa_bar(sz, zg, 0.015, dz_m) * in_sf
                    wood_a = np.maximum(wood_a, gr)
                    solid_wood_a = np.maximum(solid_wood_a, gr * 0.75)

            # Sash outer stiles (left/right vertical + top/bottom horizontal rails)
            for ys_c in (y_a + stile_w * 0.5, y_b - stile_w * 0.5):
                st = aa_bar(sy, ys_c, stile_w * 0.5, dy_m) * in_sf
                wood_a = np.maximum(wood_a, st)
                solid_wood_a = np.maximum(solid_wood_a, st * 0.85)
                wood_col_var += st * 0.14
            for zs_c in (z_a + 0.016, z_b - 0.016):
                st = aa_bar(sz, zs_c, 0.016, dz_m) * in_sf
                wood_a = np.maximum(wood_a, st)
                solid_wood_a = np.maximum(solid_wood_a, st * 0.80)

            # Delicate kumiko interior muntins (thin warm smoked cedar!)
            in_inner = (sy > y_a + stile_w * 0.6) & (sy < y_b - stile_w * 0.6) & (sz > z_a + 0.015) & (sz < z_b - 0.015)
            if yukimi_glass_v is not None:
                zg0, zg1 = wz0 + yukimi_glass_v[0] * bh, wz0 + yukimi_glass_v[1] * bh
                in_kumiko = in_inner & ((sz <= zg0) | (sz >= zg1))
            else:
                in_kumiko = in_inner
            in_kf = in_kumiko.astype(np.float32)
            for ci in range(1, n_c):
                yc = y_a + stile_w + (ci / n_c) * (y_b - y_a - 2.0 * stile_w)
                bar = aa_bar(sy, yc, bar_w, dy_m) * in_kf
                wood_a = np.maximum(wood_a, bar)
            for ri in range(1, n_r):
                zc = z_a + 0.016 + (ri / n_r) * (z_b - z_a - 0.032)
                bar = aa_bar(sz, zc, bar_w * 0.92, dz_m) * in_kf
                wood_a = np.maximum(wood_a, bar)

        # Helper to paint fine vertical koushi cedar lattice slats over [u0..u1, v0..v1]
        def add_koushi_panel(u0: float, u1: float, v0: float, v1: float, n_slats: int, n_cross: int = 2) -> None:
            nonlocal wood_a, solid_wood_a, wood_col_var
            y_a, y_b = wy0 + u0 * bw, wy0 + u1 * bw
            z_a, z_b = wz0 + v0 * bh, wz0 + v1 * bh
            in_s = ((sy >= y_a) & (sy <= y_b) & (sz >= z_a) & (sz <= z_b)).astype(np.float32)
            # Outer stiles
            for ys_c in (y_a + 0.018, y_b - 0.018):
                st = aa_bar(sy, ys_c, 0.018, dy_m) * in_s
                wood_a = np.maximum(wood_a, st)
                solid_wood_a = np.maximum(solid_wood_a, st * 0.85)
            # Fine vertical cedar slats (koushi)
            for k in range(1, n_slats):
                yk = y_a + (k / n_slats) * (y_b - y_a)
                sl = aa_bar(sy, yk, 0.0095, dy_m) * in_s
                wood_a = np.maximum(wood_a, sl * 0.96)
            # Delicate horizontal tie rails (nuki)
            for r in range(1, n_cross + 1):
                zr = z_a + (r / (n_cross + 1)) * (z_b - z_a)
                hr = aa_bar(sz, zr, 0.011, dz_m) * in_s
                wood_a = np.maximum(wood_a, hr * 0.92)

        # Helper to paint genuine layered 3D open ryokan room interior with glowing pendant lamp & tatami!
        def paint_open_room_interior(u_open0: float, u_open1: float, v_open_top: float, has_pendant: bool = True) -> None:
            nonlocal open_mask, open_rgb
            y_oa, y_ob = wy0 + u_open0 * bw, wy0 + u_open1 * bw
            z_ot = wz0 + v_open_top * bh
            om = (
                np.clip((sy - np.float32(y_oa)) / np.float32(1.5 * dy_m), 0.0, 1.0)
                * np.clip((np.float32(y_ob) - sy) / np.float32(1.5 * dy_m), 0.0, 1.0)
                * np.clip((sz - np.float32(wz0)) / np.float32(1.5 * dz_m), 0.0, 1.0)
                * np.clip((np.float32(z_ot) - sz) / np.float32(1.5 * dz_m), 0.0, 1.0)
            )
            ru = (sy - np.float32(y_oa)) / np.float32(max(0.05, y_ob - y_oa))
            rv = (sz - np.float32(wz0)) / np.float32(max(0.05, z_ot - wz0))

            # Glowing hanging Japanese paper pendant lamp (andon) inside open room!
            lamp_ru, lamp_rv = 0.48, 0.67
            d_pl = np.hypot((ru - np.float32(lamp_ru)) * np.float32(0.88), (rv - np.float32(lamp_rv)) * np.float32(1.10))
            room_pool = np.exp(-((d_pl / np.float32(0.68)) ** 2))

            # Shadowy warm cedar room walls + tokonoma back alcove
            back_wall = (
                np.array([0.20, 0.095, 0.056], dtype=np.float32)[None, None, :]
                + room_pool[..., None] * np.array([0.50, 0.27, 0.115], dtype=np.float32)[None, None, :]
                + sm[..., None] * np.array([0.018, 0.010, 0.006], dtype=np.float32)
            )
            # Subtle warm hanging scroll (kakejiku) & dark cedar tokobashira post on back wall
            in_scroll = (ru > 0.56) & (ru < 0.84) & (rv > 0.36) & (rv < 0.78)
            scroll_c = (
                np.array([0.52, 0.34, 0.18], dtype=np.float32)[None, None, :]
                + room_pool[..., None] * np.array([0.28, 0.18, 0.08], dtype=np.float32)[None, None, :]
            )
            scroll_ink = np.exp(-(((ru - 0.70) / 0.07) ** 2 + ((rv - 0.54) / 0.12) ** 2)) * 0.28
            scroll_c *= (1.0 - scroll_ink[..., None])
            back_wall = np.where(in_scroll[..., None], scroll_c, back_wall)

            # Vertical dark cedar tokobashira pillar + sliding fusuma jamb inside room
            in_pillar = (np.abs(ru - 0.28) < 0.055) & (rv > 0.18) & (rv < 0.86)
            back_wall = np.where(in_pillar[..., None], np.array([0.135, 0.065, 0.044], dtype=np.float32), back_wall)

            # Shadowed wooden room ceiling & dark rafters (rv > 0.83)
            is_ceil = np.clip((rv - 0.80) / 0.06, 0.0, 1.0)
            ceil_c = (
                np.array([0.115, 0.054, 0.038], dtype=np.float32)[None, None, :]
                + room_pool[..., None] * np.array([0.30, 0.15, 0.064], dtype=np.float32)[None, None, :]
            )
            room_c = back_wall * (1.0 - is_ceil[..., None]) + ceil_c * is_ceil[..., None]

            # Warm golden-ochre tatami floor catching downward pendant light (rv < 0.24)
            is_tatami = np.clip((0.24 - rv) / 0.04, 0.0, 1.0)
            tatami_pool = np.exp(-(((ru - 0.48) / 0.52) ** 2))
            tatami_weave = np.sin(sz * np.float32(190.0)) * np.float32(0.015)
            tatami_c = (
                np.array([0.32, 0.185, 0.088], dtype=np.float32)[None, None, :]
                + tatami_pool[..., None] * np.array([0.46, 0.28, 0.115], dtype=np.float32)[None, None, :]
                + tatami_weave[..., None]
            )
            # Crisp dark cloth tatami border (heri) + low dark-lacquer chabudai tea table!
            heri = (np.abs(ru - 0.32) < 0.025) & (rv < 0.24)
            tatami_c = np.where(heri[..., None], np.array([0.12, 0.062, 0.044], dtype=np.float32), tatami_c)
            room_c = room_c * (1.0 - is_tatami[..., None]) + tatami_c * is_tatami[..., None]

            # Low lacquered tea-table & glowing rim inside room
            in_tbl = (ru > 0.34) & (ru < 0.84) & (rv > 0.08) & (rv < 0.17)
            tbl_top = (ru > 0.34) & (ru < 0.84) & (np.abs(rv - 0.17) < 0.018)
            room_c = np.where(in_tbl[..., None], np.array([0.16, 0.072, 0.044], dtype=np.float32), room_c)
            room_c = np.where(tbl_top[..., None], np.array([0.64, 0.35, 0.15], dtype=np.float32), room_c)

            # Hanging luminous paper pendant lamp (andon) + slender cord!
            if has_pendant:
                cord_m = aa_bar(ru, lamp_ru, 0.012, 0.010) * (rv > lamp_rv + 0.07).astype(np.float32)
                room_c = room_c * (1.0 - cord_m[..., None] * 0.78) + np.array([0.08, 0.04, 0.03], dtype=np.float32) * (cord_m[..., None] * 0.78)

                # Glowing cylindrical paper lamp body + warm aura halo
                r_body = np.hypot((ru - np.float32(lamp_ru)) / np.float32(0.115), (rv - np.float32(lamp_rv)) / np.float32(0.105))
                halo = np.exp(-((r_body / np.float32(2.6)) ** 2))
                room_c += halo[..., None] * np.array([0.52, 0.30, 0.11], dtype=np.float32)
                in_shade = np.clip((1.05 - r_body) / 0.14, 0.0, 1.0)
                shade_core = np.exp(-((r_body / np.float32(0.75)) ** 2))
                shade_c = (
                    np.array([0.96, 0.62, 0.24], dtype=np.float32)[None, None, :]
                    + shade_core[..., None] * np.array([0.30, 0.38, 0.36], dtype=np.float32)[None, None, :]
                )
                room_c = room_c * (1.0 - in_shade[..., None]) + shade_c * in_shade[..., None]
                # Delicate dark-wood top & bottom rims on paper pendant shade
                for dv_r in (-0.095, 0.095):
                    rim = (np.abs(rv - (lamp_rv + dv_r)) < 0.014) & (np.abs(ru - lamp_ru) < 0.105)
                    room_c = np.where(rim[..., None], np.array([0.16, 0.078, 0.048], dtype=np.float32), room_c)

            # Soft contact shadow cast by slid-back left door stile onto the open room edge
            edge_ao = np.exp(-np.maximum(np.float32(0.0), ru) / np.float32(0.10))
            room_c *= (1.0 - 0.42 * edge_ao[..., None])

            open_mask = np.maximum(open_mask, om)
            open_rgb = np.where(om[..., None] > 0.01, room_c, open_rgb)

        # Dispatch specific traditional Japanese ryokan bay architecture!
        if arch_style == "1f_lounge_near":
            apply_pane_patina(6, 4, 0.24, 0.84)
            add_koshita(v_top=0.24, plank_n=4)
            add_ranma(v_bot=0.83, n_slats=18)
            add_sash_kumiko(0.00, 0.505, 0.24, 0.83, n_c=3, n_r=3, stile_w=0.036, bar_w=0.0105)
            add_sash_kumiko(0.495, 1.00, 0.24, 0.83, n_c=3, n_r=3, stile_w=0.036, bar_w=0.0105)
            # Subtle dark-bronze recessed oval hikite finger pulls beside center meeting stile!
            for hu in (0.455, 0.545):
                hp = np.hypot((sy - (wy0 + hu * bw)) / 0.022, (sz - (wz0 + 0.34 * bh)) / 0.036) < 1.0
                wood_a = np.where(hp, 1.0, wood_a)
                solid_wood_a = np.where(hp, 1.0, solid_wood_a)
                wood_col_var = np.where(hp, -0.35, wood_col_var)
        elif arch_style == "1f_entry":
            # Grand bathhouse entrance: koshita cedar door base + fine vertical koushi sidelights + glowing sliding shoji doors!
            apply_pane_patina(6, 5, 0.23, 0.86)
            add_koshita(v_top=0.23, plank_n=4)
            add_ranma(v_bot=0.86, n_slats=20)
            add_koushi_panel(0.00, 0.23, 0.23, 0.86, n_slats=7, n_cross=2)
            add_koushi_panel(0.77, 1.00, 0.23, 0.86, n_slats=7, n_cross=2)
            add_sash_kumiko(0.23, 0.505, 0.23, 0.86, n_c=2, n_r=4, stile_w=0.034, bar_w=0.010)
            add_sash_kumiko(0.495, 0.77, 0.23, 0.86, n_c=2, n_r=4, stile_w=0.034, bar_w=0.010)
        elif arch_style == "1f_lounge_mid":
            # Asymmetric Kyoto machiya bay: fine senbon-goushi lattice on left + 2 delicate shoji sashes on right!
            apply_pane_patina(6, 4, 0.22, 0.84)
            add_koshita(v_top=0.22, plank_n=3)
            add_ranma(v_bot=0.84, n_slats=18)
            add_koushi_panel(0.00, 0.34, 0.22, 0.84, n_slats=9, n_cross=2)
            add_sash_kumiko(0.34, 0.675, 0.22, 0.84, n_c=2, n_r=3, stile_w=0.032, bar_w=0.010)
            add_sash_kumiko(0.665, 1.00, 0.22, 0.84, n_c=2, n_r=3, stile_w=0.032, bar_w=0.010)
        elif arch_style == "2f_near":
            # 3-panel sliding yukimi-shoji with clear warm glass snow-viewing strip + koshita base + ranma transom!
            apply_pane_patina(6, 3, 0.23, 0.83)
            add_koshita(v_top=0.23, plank_n=3)
            add_ranma(v_bot=0.83, n_slats=22)
            add_sash_kumiko(0.00, 0.34, 0.23, 0.83, n_c=2, n_r=3, stile_w=0.032, bar_w=0.0095, yukimi_glass_v=(0.24, 0.39))
            add_sash_kumiko(0.33, 0.67, 0.23, 0.83, n_c=2, n_r=3, stile_w=0.032, bar_w=0.0095, yukimi_glass_v=(0.24, 0.39))
            add_sash_kumiko(0.66, 1.00, 0.23, 0.83, n_c=2, n_r=3, stile_w=0.032, bar_w=0.0095, yukimi_glass_v=(0.24, 0.39))
        elif arch_style == "2f_hero_open":
            # HERO 2F Veranda Bay: Left 2 yukimi-shoji sashes closed, RIGHT sash slid OPEN revealing glowing hanging pendant lamp & tatami room!
            apply_pane_patina(4, 3, 0.23, 0.82)
            add_koshita(v_top=0.23, u_lo=0.00, u_hi=0.635, plank_n=3)
            add_ranma(v_bot=0.82, n_slats=22)
            add_sash_kumiko(0.00, 0.335, 0.23, 0.82, n_c=2, n_r=3, stile_w=0.032, bar_w=0.0095)
            add_sash_kumiko(0.315, 0.635, 0.23, 0.82, n_c=2, n_r=3, stile_w=0.034, bar_w=0.0095)
            paint_open_room_interior(0.635, 0.965, 0.815, has_pendant=True)
        elif arch_style == "2f_mid":
            apply_pane_patina(6, 3, 0.23, 0.83)
            add_koshita(v_top=0.23, plank_n=3)
            add_ranma(v_bot=0.83, n_slats=20)
            add_sash_kumiko(0.00, 0.34, 0.23, 0.83, n_c=2, n_r=3, stile_w=0.032, bar_w=0.0095)
            add_sash_kumiko(0.33, 0.67, 0.23, 0.83, n_c=2, n_r=3, stile_w=0.032, bar_w=0.0095)
            add_sash_kumiko(0.66, 1.00, 0.23, 0.83, n_c=2, n_r=3, stile_w=0.032, bar_w=0.0095)
        elif arch_style == "2f_far_crack":
            # Slightly cracked-open left sliding panel revealing sliver of warm lantern-lit interior!
            apply_pane_patina(6, 3, 0.23, 0.83)
            add_koshita(v_top=0.23, u_lo=0.20, u_hi=1.00, plank_n=3)
            add_ranma(v_bot=0.83, n_slats=20)
            paint_open_room_interior(0.015, 0.205, 0.825, has_pendant=False)
            add_sash_kumiko(0.20, 0.605, 0.23, 0.83, n_c=3, n_r=3, stile_w=0.032, bar_w=0.0095)
            add_sash_kumiko(0.595, 1.00, 0.23, 0.83, n_c=3, n_r=3, stile_w=0.032, bar_w=0.0095)
        else:
            # 3F attic clerestory shoji bays
            apply_pane_patina(6, 3, 0.20, 0.88)
            add_koshita(v_top=0.20, plank_n=2)
            add_sash_kumiko(0.00, 0.505, 0.20, 1.00, n_c=3, n_r=2, stile_w=0.032, bar_w=0.0105)
            add_sash_kumiko(0.495, 1.00, 0.20, 1.00, n_c=3, n_r=2, stile_w=0.032, bar_w=0.0105)

        # Blend open interior room cleanly into the opening background
        bg_open = pane_c * (1.0 - open_mask[..., None]) + open_rgb * open_mask[..., None]

        # Build richly textured warm smoked-cedar wood color for muntins/stiles/koshita
        wood_c = (
            cedar_dark * (1.0 - np.clip(wood_col_var, 0.0, 1.0)[..., None])
            + cedar_warm * np.clip(wood_col_var, 0.0, 1.0)[..., None]
            - np.clip(-wood_col_var, 0.0, 1.0)[..., None] * np.array([0.06, 0.032, 0.022], dtype=np.float32)
            + (wood_grain_v * 0.5 + wood_grain_h * 0.5 + sm * 0.010)[..., None] * np.array([1.0, 0.58, 0.34], dtype=np.float32)
            + slan[..., None] * np.array([0.16, 0.076, 0.028], dtype=np.float32)
        )

        # Optical warm amber light-bleed halation wrapping gently around thin kumiko & koushi bars!
        glow_emit = bg_open * (1.0 - wood_a)[..., None] * in_open[..., None].astype(np.float32)
        bleed_tight = gaussian_blur_2d(glow_emit, max(1.6, 0.012 / dy_m))
        bleed_soft = gaussian_blur_2d(glow_emit, max(3.6, 0.030 / dy_m))
        thin_bar_factor = np.clip(wood_a * (1.0 - solid_wood_a * 0.72), 0.0, 1.0)[..., None]

        interior_rgb = bg_open * (1.0 - wood_a[..., None]) + wood_c * wood_a[..., None]
        interior_rgb += (0.19 * bleed_tight + 0.08 * bleed_soft) * thin_bar_factor * np.array([1.10, 0.80, 0.46], dtype=np.float32)
        # Subtle warm upward bounce kissing the top edge of solid koshita panels
        interior_rgb += (0.10 * bleed_tight) * (solid_wood_a * wood_a)[..., None] * np.array([1.05, 0.68, 0.34], dtype=np.float32)

        # ---------------------------------------------------------------------
        # 4. Perspective-Accurate 3D Recessed Inner Reveals (Far Jamb, Kamoi Soffit, Shikii Shelf!)
        # ---------------------------------------------------------------------
        # Camera at y = -4.90 looking toward +Y -> sees the recessed FAR (+Y) inner jamb inside wy1!
        rev_r_w = np.float32(0.062)
        rev_r = np.clip((sy - np.float32(wy1 - rev_r_w)) / np.float32(1.6 * dy_m), 0.0, 1.0) * in_open.astype(np.float32)
        t_rev_r = np.clip((sy - np.float32(wy1 - rev_r_w)) / rev_r_w, 0.0, 1.0)  # 0 inner glass -> 1 outer trim
        rev_r_c = (
            cedar_warm * (1.0 - t_rev_r[..., None] * 0.55)
            + cedar_dark * (t_rev_r[..., None] * 0.55)
            + ((1.0 - t_rev_r * 0.62) * (0.45 + 0.55 * lamp_broad))[..., None]
              * np.array([0.52, 0.255, 0.082], dtype=np.float32) * np.float32(warmth)
            + wood_grain_v[..., None] * np.array([0.8, 0.45, 0.25], dtype=np.float32)
        )
        interior_rgb = interior_rgb * (1.0 - rev_r[..., None]) + rev_r_c * rev_r[..., None]

        # Thin near-side (-Y) shadow tuck right inside wy0
        rev_l_ao = np.exp(-np.maximum(np.float32(0.0), sy - np.float32(wy0)) / np.float32(0.026)) * in_open.astype(np.float32)
        interior_rgb *= (1.0 - 0.36 * rev_l_ao[..., None])

        # Top kamoi recessed soffit reveal (strongly visible on 2F/3F since z > z_eye = 2.50!)
        rev_t_w = np.float32(0.068 if is_upper_floor else 0.038)
        rev_t = np.clip((sz - np.float32(wz1 - rev_t_w)) / np.float32(1.6 * dz_m), 0.0, 1.0) * in_open.astype(np.float32)
        t_rev_t = np.clip((sz - np.float32(wz1 - rev_t_w)) / rev_t_w, 0.0, 1.0)
        rev_t_c = (
            cedar_dark * np.float32(0.85)
            + ((1.0 - t_rev_t * 0.55) * (0.45 + 0.55 * lamp_broad))[..., None]
              * np.array([0.46, 0.22, 0.072], dtype=np.float32) * np.float32(warmth)
        )
        interior_rgb = interior_rgb * (1.0 - rev_t[..., None]) + rev_t_c * rev_t[..., None]

        # Bottom shikii recessed wooden sill ledge (visible on 1F since wz0 < z_eye = 2.50!)
        if not is_upper_floor:
            rev_b_w = np.float32(0.044)
            rev_b = np.clip((np.float32(wz0 + rev_b_w) - sz) / np.float32(1.6 * dz_m), 0.0, 1.0) * in_open.astype(np.float32)
            t_rev_b = np.clip((sz - np.float32(wz0)) / rev_b_w, 0.0, 1.0)
            rev_b_c = (
                cedar_mid
                + ((0.40 + 0.60 * t_rev_b) * (0.45 + 0.55 * lamp_broad))[..., None]
                  * np.array([0.34, 0.165, 0.058], dtype=np.float32) * np.float32(warmth)
            )
            interior_rgb = interior_rgb * (1.0 - rev_b[..., None]) + rev_b_c * rev_b[..., None]

        # ---------------------------------------------------------------------
        # 5. Sculpted Outer Timber Frame (Kamoi Lintel, Shikii Sill, Jamb Posts!)
        # ---------------------------------------------------------------------
        # Warm dark smoked-cedar outer casing with beveled top catch-light & warm lantern response
        frame_rgb = (
            cedar_mid * np.float32(0.88)
            + (wood_grain_h * np.where(in_kamoi | in_shikii, 1.0, 0.25) + wood_grain_v * np.where(in_jamb_l | in_jamb_r, 1.0, 0.25) + sm * 0.012)[..., None]
              * np.array([1.0, 0.58, 0.34], dtype=np.float32)
            + slan[..., None] * np.array([0.24, 0.115, 0.042], dtype=np.float32)
        )
        # Kamoi lintel top lip catches subtle dusk/lantern rim; underside shadows softly
        kamoi_t = np.clip((sz - np.float32(wz1)) / fw_top, 0.0, 1.0)
        kamoi_hi = np.exp(-(((kamoi_t - 0.85) / 0.16) ** 2)) * in_kamoi.astype(np.float32)
        kamoi_sh = np.exp(-((kamoi_t / 0.24) ** 2)) * in_kamoi.astype(np.float32)
        frame_rgb += kamoi_hi[..., None] * np.array([0.16, 0.088, 0.048], dtype=np.float32)
        frame_rgb *= (1.0 - 0.24 * kamoi_sh[..., None])

        # Shikii sill top ledge catches rich warm golden spill from the shoji!
        shikii_t = np.clip((sz - np.float32(wz0 - fw_bot)) / fw_bot, 0.0, 1.0)
        shikii_top_hi = np.exp(-(((1.0 - shikii_t) / 0.28) ** 2)) * in_shikii.astype(np.float32)
        shikii_bot_sh = np.exp(-((shikii_t / 0.26) ** 2)) * in_shikii.astype(np.float32)
        frame_rgb += shikii_top_hi[..., None] * np.array([0.34, 0.175, 0.068], dtype=np.float32) * np.float32(warmth)
        frame_rgb *= (1.0 - 0.28 * shikii_bot_sh[..., None])

        # Inner bevel on jamb posts right next to glowing shoji catches warm amber rim!
        inner_bevel = np.exp(-d_open / np.float32(0.022)) * in_frame.astype(np.float32)
        frame_rgb += inner_bevel[..., None] * np.array([0.28, 0.138, 0.048], dtype=np.float32) * np.float32(warmth)

        # Composite frame and recessed window interior smoothly onto wall slice!
        sub_wall = np.where(in_frame[..., None], frame_rgb, sub_wall)
        sub_wall = np.where(in_open[..., None], interior_rgb, sub_wall)
        wall[iz0:iz1, ix0:ix1] = sub_wall

    if story_mode == "lower_2f":
        # Ground Floor Bays — distinct architectural rhythms & cozy interior lighting!
        paint_shoji_bay(2.15, 4.75, 1.56, 3.06, arch_style="1f_lounge_near", warmth=0.98, lamp_uv=(0.38, 0.64), hue_shift=(1.02, 0.99, 0.94), is_upper_floor=False)
        paint_shoji_bay(5.75, 8.30, 1.22, 3.08, arch_style="1f_entry", warmth=1.12, lamp_uv=(0.50, 0.68), hue_shift=(1.04, 1.02, 0.96), is_upper_floor=False)
        paint_shoji_bay(9.35, 11.90, 1.56, 3.06, arch_style="1f_lounge_mid", warmth=0.93, lamp_uv=(0.66, 0.60), hue_shift=(1.04, 0.94, 0.86), is_upper_floor=False)

        # Ground-floor Cedar Bath Exhaust Louver (yuge-nuki) with recessed smoked-cedar frame & backlit steam glow!
        lv_y0, lv_y1, lv_z0, lv_z1 = 12.90, 15.50, 1.70, 3.04
        in_lv = (wy >= lv_y0) & (wy <= lv_y1) & (wz >= lv_z0) & (wz <= lv_z1)
        in_lv_f = (wy >= lv_y0 - 0.085) & (wy <= lv_y1 + 0.085) & (wz >= lv_z0 - 0.085) & (wz <= lv_z1 + 0.095)
        slat_f = np.mod((wz - lv_z0) / 0.112, 1.0)
        slat_wood = np.clip((slat_f - 0.30) / 0.08, 0.0, 1.0) * np.clip((0.98 - slat_f) / 0.06, 0.0, 1.0)
        slat_bevel = np.exp(-(((slat_f - 0.38) / 0.10) ** 2))
        lv_glow = np.exp(-((wy - 14.20) / 1.18) ** 2) * np.exp(-((wz - 2.36) / 0.68) ** 2)
        slat_gap_c = (
            np.array([0.88, 0.44, 0.16], dtype=np.float32) * (0.48 + 0.72 * lv_glow[..., None])
            + np.exp(-((wy - 14.20) / 0.70) ** 2)[..., None] * np.array([0.24, 0.26, 0.16], dtype=np.float32)
        )
        slat_bar_c = (
            np.array([0.155, 0.082, 0.056], dtype=np.float32)
            + slat_bevel[..., None] * np.array([0.18, 0.095, 0.042], dtype=np.float32)
            + lv_glow[..., None] * np.array([0.16, 0.076, 0.028], dtype=np.float32)
        )
        lv_col = slat_wood[..., None] * slat_bar_c + (1.0 - slat_wood[..., None]) * slat_gap_c
        lv_mid = aa_bar(wy, 14.20, 0.042, dy_m) * in_lv.astype(np.float32)
        lv_rev_r = ((wy > lv_y1 - 0.055) & in_lv).astype(np.float32)
        lv_frame_c = np.array([0.185, 0.098, 0.066], dtype=np.float32) + lantern[..., None] * np.array([0.22, 0.105, 0.036], dtype=np.float32)
        lv_col = lv_col * (1.0 - lv_mid[..., None]) + lv_frame_c * lv_mid[..., None]
        lv_col = lv_col * (1.0 - lv_rev_r[..., None]) + (lv_frame_c + np.array([0.24, 0.115, 0.038], dtype=np.float32)) * lv_rev_r[..., None]
        wall = np.where(in_lv_f[..., None], lv_frame_c, wall)
        wall = np.where(in_lv[..., None], lv_col, wall)

        # Traditional smoked-cedar sliding storm-shutter pocket (tobukuro) framing the far-left 2F corner bay!
        tb_y0, tb_y1, tb_z0, tb_z1 = -1.25, 1.38, 3.68, 5.24
        in_tb = (wy >= tb_y0) & (wy <= tb_y1) & (wz >= tb_z0) & (wz <= tb_z1)
        in_tb_f = (wy >= tb_y0 - 0.075) & (wy <= tb_y1 + 0.075) & (wz >= tb_z0 - 0.075) & (wz <= tb_z1 + 0.085)
        tb_plank = np.mod((wz - tb_z0) / 0.135, 1.0)
        tb_groove = np.clip(np.minimum(tb_plank, 1.0 - tb_plank) / 0.12, 0.0, 1.0)
        tb_c = (
            np.array([0.135, 0.068, 0.048], dtype=np.float32)
            + (mottle * 0.010 + grain * 0.5)[..., None]
            + lantern[..., None] * np.array([0.18, 0.085, 0.032], dtype=np.float32)
        ) * (0.68 + 0.32 * tb_groove[..., None])
        tb_fc = np.array([0.115, 0.056, 0.042], dtype=np.float32) + lantern[..., None] * np.array([0.16, 0.076, 0.030], dtype=np.float32)
        wall = np.where(in_tb_f[..., None], tb_fc, wall)
        wall = np.where(in_tb[..., None], tb_c, wall)

        # 2nd Floor Iconic Glowing Yukimi-Shoji Veranda Bays (including HERO open guestroom with hanging lamp!)
        paint_shoji_bay(2.10, 4.95, 3.70, 5.22, arch_style="2f_near", warmth=1.00, lamp_uv=(0.44, 0.66), hue_shift=(1.02, 0.99, 0.94), is_upper_floor=True)
        paint_shoji_bay(5.80, 8.55, 3.70, 5.22, arch_style="2f_hero_open", warmth=1.10, lamp_uv=(0.36, 0.64), hue_shift=(1.03, 1.01, 0.96), is_upper_floor=True)
        paint_shoji_bay(9.35, 12.10, 3.70, 5.22, arch_style="2f_mid", warmth=0.90, lamp_uv=(0.38, 0.60), hue_shift=(1.03, 0.94, 0.87), is_upper_floor=True)
        paint_shoji_bay(12.90, 15.65, 3.70, 5.22, arch_style="2f_far_crack", warmth=0.96, lamp_uv=(0.60, 0.65), hue_shift=(1.02, 0.98, 0.92), is_upper_floor=True)
    else:
        # 3rd Floor Bathhouse Clerestory Shoji Bays + Signature Circular Marumado Window!
        paint_shoji_bay(2.60, 4.80, 5.95, 6.94, arch_style="3f_loft", warmth=0.90, lamp_uv=(0.46, 0.60), is_upper_floor=True)
        paint_shoji_bay(6.10, 8.45, 5.95, 6.94, arch_style="3f_loft", warmth=1.02, lamp_uv=(0.54, 0.62), is_upper_floor=True)
        paint_shoji_bay(9.80, 12.30, 5.95, 6.94, arch_style="3f_loft", warmth=0.95, lamp_uv=(0.42, 0.58), is_upper_floor=True)
        # Glowing circular Japanese marumado window at y = 14.55, z = 6.48 with smoked-cedar ring & delicate kumiko!
        r_maru = np.hypot((wy - 14.55) / 0.64, (wz - 6.48) / 0.50)
        in_maru = r_maru < 1.0
        in_maru_ring = (r_maru >= 0.94) & (r_maru < 1.15)
        maru_bars = np.maximum.reduce([
            aa_bar(wy, 14.55, 0.014, dy_m),
            aa_bar(wy, 14.24, 0.011, dy_m),
            aa_bar(wy, 14.86, 0.011, dy_m),
            aa_bar(wz, 6.48, 0.012, dz_m),
            aa_bar(wz, 6.24, 0.010, dz_m),
        ])
        maru_glow = np.exp(-(r_maru / 0.80) ** 2)
        maru_col = np.array([0.86, 0.43, 0.15], dtype=np.float32) + maru_glow[..., None] * np.array([0.46, 0.52, 0.34], dtype=np.float32)
        maru_wood = np.array([0.19, 0.10, 0.068], dtype=np.float32) + maru_glow[..., None] * np.array([0.18, 0.088, 0.032], dtype=np.float32)
        wall = np.where(in_maru_ring[..., None], maru_wood, wall)
        wall = np.where(in_maru[..., None], maru_col * (1.0 - maru_bars[..., None]) + maru_wood * maru_bars[..., None], wall)

    wall = apply_warm_shadow_bounce(wall, strength=0.016)
    wall += tooth[..., None] * 0.006
    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(wall, 0.0, 1.50)
    return rgba


def paint_kawara_roof_uv(
    rng: np.random.Generator,
    y_len: float,
    slope_len: float,
    tw: int = 1152,
    th: int = 384,
) -> np.ndarray:
    """Curved ibushi-kawara Japanese clay roof tiles with twilight rose-apricot rim glaze."""
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    wy = (xx / (tw - 1)) * np.float32(y_len)
    ws = (yy / (th - 1)) * np.float32(slope_len)

    mottle = fbm_gouache(th, tw, rng)
    dabs = flat_brush_dabs(th, tw, rng, count=86)
    tooth = paper_tooth(th, tw, rng)

    tile_w = np.float32(0.28)
    course_l = np.float32(0.32)
    col_id = np.floor(wy / tile_w).astype(np.int32)
    row_id = np.floor(ws / course_l).astype(np.int32)
    col_f = (wy / tile_w) - col_id
    row_f = (ws / course_l) - row_id

    cyl_curve = np.cos(np.clip((col_f - 0.18) / 0.18, -1.0, 1.0) * math.pi * 0.5)
    is_roll = (col_f < 0.36).astype(np.float32)
    pan_dip = np.sin(np.clip((col_f - 0.36) / 0.64, 0.0, 1.0) * math.pi)
    step_shadow = np.clip(row_f / 0.16, 0.0, 1.0)
    lip_hi = np.exp(-((row_f - 0.07) / 0.045) ** 2)

    t_hash = ((col_id * 73856093) ^ (row_id * 19349663)) & 255
    tv = rng.uniform(-0.020, 0.024, size=256).astype(np.float32)[t_hash]

    base = np.zeros((th, tw, 3), dtype=np.float32)
    base[:, :, 0] = 0.064 + tv * 0.44 + mottle * 0.011 + dabs * 0.009
    base[:, :, 1] = 0.048 + tv * 0.36 + mottle * 0.009 + dabs * 0.007
    base[:, :, 2] = 0.074 + tv * 0.42 + mottle * 0.010 + dabs * 0.008

    roll_hi = np.power(np.clip(cyl_curve, 0.0, 1.0), 1.8) * is_roll
    sun_side = np.clip((col_f - 0.16) / 0.16, 0.0, 1.0) * is_roll
    shade = (0.56 + 0.44 * step_shadow) * (0.72 + 0.32 * roll_hi - 0.18 * pan_dip)
    rgb = base * shade[..., None]
    rgb += (roll_hi * 0.50 + sun_side * 0.68 + lip_hi * 0.28)[..., None] * np.array([0.28, 0.135, 0.092], dtype=np.float32)

    rgb = apply_warm_shadow_bounce(rgb, strength=0.015)
    rgb += tooth[..., None] * 0.0055
    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.25)
    return rgba


def paint_eave_soffit_uv(
    rng: np.random.Generator,
    y_span: tuple[float, float],
    eave_z: float,
    x_in: float,
    x_out: float,
    tw: int = 960,
    th: int = 256,
) -> np.ndarray:
    """Shadowed cedar eaves soffit with physically placed warm upward chochin bounce light."""
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    wy = np.float32(y_span[0]) + (xx / (tw - 1)) * np.float32(y_span[1] - y_span[0])
    wx_f = yy / (th - 1)
    wx = np.float32(x_in) + wx_f * np.float32(x_out - x_in)
    wz = np.full_like(wy, eave_z)

    mottle = fbm_gouache(th, tw, rng)
    tooth = paper_tooth(th, tw, rng)

    plank_f = np.mod(wx_f * 8.0, 1.0)
    seam = np.clip(np.minimum(plank_f, 1.0 - plank_f) / 0.08, 0.0, 1.0)
    raf_f = np.mod(wy / 0.52, 1.0)
    raf_ao = 0.58 + 0.42 * np.clip(np.minimum(raf_f, 1.0 - raf_f) / 0.14, 0.0, 1.0)
    # Soft natural overhang shading: warmer near inner shoji wall & lantern pools, deeper timber shadow near outer eave
    outer_sh = 1.0 - 0.34 * np.power(wx_f, 1.3)

    rgb = np.zeros((th, tw, 3), dtype=np.float32)
    rgb[:, :, 0] = 0.064 + mottle * 0.010
    rgb[:, :, 1] = 0.032 + mottle * 0.006
    rgb[:, :, 2] = 0.028 + mottle * 0.005
    rgb *= (0.48 + 0.52 * seam[..., None]) * raf_ao[..., None] * outer_sh[..., None]

    lantern, shoji = eval_onsen_point_lights(wx, wy, wz, is_horizontal=False)
    rgb += (lantern * 0.34 + shoji * 0.14)[..., None] * (0.45 + 0.55 * seam[..., None]) * raf_ao[..., None] * np.array([0.86, 0.42, 0.15], dtype=np.float32)
    rgb = apply_warm_shadow_bounce(rgb, strength=0.014)
    rgb += tooth[..., None] * 0.005
    rgba = np.ones((th, tw, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.15)
    return rgba


# =============================================================================
# Stage 3 — Fast Tight-BBox 2x Supersampled PyCairo Helpers
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

    sub_f = sub.astype(np.float32) * (1.0 / 255.0)
    down = cv2.resize(sub_f, (bw, bh), interpolation=cv2.INTER_LANCZOS4)
    a1 = np.clip(down[:, :, 3], 0.0, 1.0)
    out[y0:y1, x0:x1, 3] = a1
    for out_c, src_c in ((0, 2), (1, 1), (2, 0)):
        c1 = np.clip(down[:, :, src_c], 0.0, 1.0)
        straight = np.where(a1 > 1e-4, np.clip(c1 / np.maximum(a1, 1e-4), 0.0, 1.0), 0.0)
        out[y0:y1, x0:x1, out_c] = straight * a1
    return out


def fog_mix(rgb: Iterable[float], dist: float, fog_rgb: np.ndarray, fog_start: float = 8.5, fog_density: float = 0.028) -> tuple[float, float, float]:
    ft = 1.0 - math.exp(-max(0.0, dist - fog_start) * fog_density)
    r, g, b = rgb
    return (
        float(r * (1.0 - ft) + fog_rgb[0] * ft),
        float(g * (1.0 - ft) + fog_rgb[1] * ft),
        float(b * (1.0 - ft) + fog_rgb[2] * ft),
    )


def fill_poly(ctx: cairo.Context, pts: list[tuple[float, float]]) -> None:
    if len(pts) < 3:
        return
    ctx.move_to(pts[0][0], pts[0][1])
    for x, y in pts[1:]:
        ctx.line_to(x, y)
    ctx.close_path()
    ctx.fill()


def stroke_catmull_rom(
    ctx: cairo.Context,
    pts: list[tuple[float, float]],
    base_w: float,
    z_depth: float,
    rgba: tuple[float, float, float, float],
    rng: np.random.Generator,
    fade_dist: float = 28.0,
    taper: bool = False,
    tremor: float = 0.14,
) -> None:
    if len(pts) < 2:
        return
    decay = float(np.clip(1.0 / (1.0 + z_depth / (fade_dist * 0.55)), 0.25, 1.0))
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
                (2.0 * p1[0])
                + (-p0[0] + p2[0]) * t
                + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2.0 * p1[1])
                + (-p0[1] + p2[1]) * t
                + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3
            )
            if tremor > 0.0:
                x += float(rng.normal(0.0, tremor * decay))
                y += float(rng.normal(0.0, tremor * decay))
            dense.append((x, y))
    dense.append(pts[-1])

    ctx.set_source_rgba(*rgba)
    if not taper:
        ctx.set_line_width(max(0.65, w_eff))
        ctx.move_to(*dense[0])
        for p in dense[1:]:
            ctx.line_to(*p)
        ctx.stroke()
    else:
        n = len(dense)
        for i in range(n - 1):
            frac = i / max(1, n - 1)
            env = math.sin(frac * math.pi) ** 0.55
            ctx.set_line_width(max(0.5, w_eff * (0.35 + 0.65 * env)))
            ctx.move_to(*dense[i])
            ctx.line_to(*dense[i + 1])
            ctx.stroke()


def project_quad_3d(cam: Camera3D, pts3d: np.ndarray, mirror: bool = False) -> list[tuple[float, float]] | None:
    pts = mirror_z0(pts3d) if mirror else np.asarray(pts3d, dtype=np.float64)
    out: list[tuple[float, float]] = []
    n = len(pts)
    for i in range(n):
        cl = cam.clip_segment_3d(pts[i], pts[(i + 1) % n])
        if cl is None:
            continue
        a, b = cl
        ua, va, _ = cam.project(a)
        ub, vb, _ = cam.project(b)
        if not out or math.hypot(out[-1][0] - float(ua), out[-1][1] - float(va)) > 0.25:
            out.append((float(ua), float(va)))
        out.append((float(ub), float(vb)))
    return out if len(out) >= 3 else None


def draw_box_3d_faces(
    ctx: cairo.Context,
    cam: Camera3D,
    x0: float, x1: float,
    y0: float, y1: float,
    z0: float, z1: float,
    c_front: tuple[float, float, float],
    c_side: tuple[float, float, float],
    c_top: tuple[float, float, float],
    fog_rgb: np.ndarray,
    mirror: bool = False,
    alpha: float = 0.98,
    c_bot: tuple[float, float, float] | None = None,
) -> None:
    yc, xc, zc = 0.5 * (y0 + y1), 0.5 * (x0 + x1), 0.5 * (z0 + z1)
    dist = float(np.linalg.norm(np.array([xc, yc, zc]) - cam.eye))

    if cam.eye[0] > x1:
        q = project_quad_3d(cam, np.array([[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]]), mirror=mirror)
        if q:
            ctx.set_source_rgba(*fog_mix(c_side, dist, fog_rgb), alpha)
            fill_poly(ctx, q)
    elif cam.eye[0] < x0:
        q = project_quad_3d(cam, np.array([[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]]), mirror=mirror)
        if q:
            ctx.set_source_rgba(*fog_mix(c_side, dist, fog_rgb), alpha)
            fill_poly(ctx, q)

    if cam.eye[2] > z1:
        q = project_quad_3d(cam, np.array([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]), mirror=mirror)
        if q:
            ctx.set_source_rgba(*fog_mix(c_top, dist, fog_rgb), alpha)
            fill_poly(ctx, q)
    elif cam.eye[2] < z0:
        q = project_quad_3d(cam, np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]]), mirror=mirror)
        if q:
            bot_col = c_bot if c_bot is not None else tuple(c * 0.68 for c in c_front)
            ctx.set_source_rgba(*fog_mix(bot_col, dist, fog_rgb), alpha)
            fill_poly(ctx, q)

    qf = project_quad_3d(cam, np.array([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]), mirror=mirror)
    if qf:
        ctx.set_source_rgba(*fog_mix(c_front, dist, fog_rgb), alpha)
        fill_poly(ctx, qf)


# =============================================================================
# Stage 3 Band 0b — Richly Detailed Distant Upstream Ryokan Village
# =============================================================================

def draw_shoji_quad_with_mullions(
    ctx: cairo.Context,
    cam: Camera3D,
    p_bl: np.ndarray,
    p_br: np.ndarray,
    p_tr: np.ndarray,
    p_tl: np.ndarray,
    n_cols: int,
    n_rows: int,
    dist: float,
    fog_rgb: np.ndarray,
    br: float = 1.0,
    mirror: bool = False,
    lamp_u: float = 0.48,
    lamp_v: float = 0.62,
    koshita_frac: float = 0.22,
    open_bay: bool = False,
) -> None:
    """
    Painterly recessed distant Japanese ryokan shoji & koushi window:
      - Soft warm amber light spill onto surrounding timber/plaster wall,
      - Recessed warm smoked-cedar outer frame (kamoi top lintel, projecting shikii sill
        with warm top amber catch-light, side jambs — NEVER flat black wireframes!),
      - Authentic koshita lower cedar wood kick-panel + upper translucent washi glow with
        unique per-window lamp placement, top kamoi shadow gradient, soft interior silhouette
        depth, optional slightly-open warm room bay, and delicate smoked-cedar kumiko bars!
    """
    e_u = p_br - p_bl
    e_v = p_tl - p_bl

    # 1. Soft exterior warm light spill + recessed outer timber frame (kamoi & shikii)
    q_outer = project_quad_3d(
        cam,
        np.array([
            p_bl - e_u * 0.055 - e_v * 0.075,
            p_br + e_u * 0.055 - e_v * 0.075,
            p_tr + e_u * 0.065 + e_v * 0.080,
            p_tl - e_u * 0.065 + e_v * 0.080,
        ]),
        mirror=mirror,
    )
    q = project_quad_3d(cam, np.array([p_bl, p_br, p_tr, p_tl]), mirror=mirror)
    if not q or len(q) < 4:
        return

    cx = 0.25 * sum(p[0] for p in q)
    cy = 0.25 * sum(p[1] for p in q)
    rad = max(6.0, 0.65 * math.hypot(q[2][0] - q[0][0], q[2][1] - q[0][1]))

    # Subtle warm amber wall spill halo behind the recessed timber frame
    g_sp = cairo.RadialGradient(cx, cy, rad * 0.18, cx, cy, rad * 1.42)
    g_sp.add_color_stop_rgba(0.0, *fog_mix([0.96 * br, 0.50 * br, 0.18 * br], dist, fog_rgb, fog_density=0.022), 0.26)
    g_sp.add_color_stop_rgba(1.0, *fog_mix([0.85 * br, 0.40 * br, 0.14 * br], dist, fog_rgb, fog_density=0.026), 0.0)
    ctx.set_source(g_sp)
    ctx.arc(cx, cy, rad * 1.42, 0, math.tau)
    ctx.fill()

    # Recessed outer smoked-cedar timber surround (kamoi + shikii + jambs)
    if q_outer and len(q_outer) >= 4:
        ctx.set_source_rgba(*fog_mix([0.145, 0.074, 0.056], dist, fog_rgb, fog_density=0.028), 0.96)
        fill_poly(ctx, q_outer)

    # 2. Translucent glowing handmade washi interior (upper portion above koshita wood base)
    kf = max(0.0, min(0.32, koshita_frac))
    p_kl = p_bl + e_v * kf
    p_kr = p_br + e_v * kf
    q_pane = project_quad_3d(cam, np.array([p_kl, p_kr, p_tr, p_tl]), mirror=mirror)
    if not q_pane or len(q_pane) < 4:
        return

    p_lamp = p_bl + e_u * lamp_u + e_v * (kf + (1.0 - kf) * lamp_v)
    if mirror:
        p_lamp = mirror_z0(p_lamp)
    ul_p, vl_p, _ = cam.project(p_lamp)
    g_w = cairo.RadialGradient(float(ul_p), float(vl_p), rad * 0.06, float(ul_p), float(vl_p), rad * 1.06)
    g_w.add_color_stop_rgba(0.0, *fog_mix([1.14 * br, 0.88 * br, 0.46 * br], dist, fog_rgb, fog_density=0.019), 0.98)
    g_w.add_color_stop_rgba(0.52, *fog_mix([0.98 * br, 0.58 * br, 0.21 * br], dist, fog_rgb, fog_density=0.022), 0.97)
    g_w.add_color_stop_rgba(1.0, *fog_mix([0.66 * br, 0.28 * br, 0.115 * br], dist, fog_rgb, fog_density=0.025), 0.96)
    ctx.set_source(g_w)
    fill_poly(ctx, q_pane)

    # Soft overhanging top kamoi lintel shadow + subtle interior room silhouette depth
    ctx.save()
    ctx.move_to(*q_pane[0])
    for qp in q_pane[1:]:
        ctx.line_to(*qp)
    ctx.close_path()
    ctx.clip()

    top_mid_x, top_mid_y = 0.5 * (q_pane[2][0] + q_pane[3][0]), 0.5 * (q_pane[2][1] + q_pane[3][1])
    bot_mid_x, bot_mid_y = 0.5 * (q_pane[0][0] + q_pane[1][0]), 0.5 * (q_pane[0][1] + q_pane[1][1])
    g_ao = cairo.LinearGradient(top_mid_x, top_mid_y, bot_mid_x, bot_mid_y)
    g_ao.add_color_stop_rgba(0.0, *fog_mix([0.22, 0.095, 0.052], dist, fog_rgb, fog_density=0.026), 0.44)
    g_ao.add_color_stop_rgba(0.28, *fog_mix([0.22, 0.095, 0.052], dist, fog_rgb, fog_density=0.026), 0.0)
    g_ao.add_color_stop_rgba(1.0, *fog_mix([0.24, 0.10, 0.055], dist, fog_rgb, fog_density=0.026), 0.14)
    ctx.set_source(g_ao)
    fill_poly(ctx, q_pane)

    # Optional subtle warm shadowy interior room panel (one sash slid slightly open!)
    if open_bay and n_cols >= 3:
        q_op = project_quad_3d(
            cam,
            np.array([p_bl + e_u * 0.68, p_br, p_tr - e_v * 0.14, p_tl + e_u * 0.68 - e_v * 0.14]),
            mirror=mirror,
        )
        if q_op and len(q_op) >= 4:
            ctx.set_source_rgba(*fog_mix([0.24, 0.11, 0.062], dist, fog_rgb, fog_density=0.024), 0.88)
            fill_poly(ctx, q_op)
            # Tiny warm glowing hanging room lamp inside the open bay!
            p_in_l = p_bl + e_u * 0.84 + e_v * 0.62
            if mirror:
                p_in_l = mirror_z0(p_in_l)
            ui, vi, _ = cam.project(p_in_l)
            ri = max(1.8, rad * 0.09)
            g_il = cairo.RadialGradient(float(ui), float(vi), ri * 0.15, float(ui), float(vi), ri * 2.2)
            g_il.add_color_stop_rgba(0.0, *fog_mix([1.15, 0.90, 0.48], dist, fog_rgb, fog_density=0.018), 0.95)
            g_il.add_color_stop_rgba(0.55, *fog_mix([0.92, 0.48, 0.18], dist, fog_rgb, fog_density=0.022), 0.48)
            g_il.add_color_stop_rgba(1.0, *fog_mix([0.80, 0.36, 0.12], dist, fog_rgb, fog_density=0.024), 0.0)
            ctx.set_source(g_il)
            ctx.arc(float(ui), float(vi), ri * 2.2, 0, math.tau)
            ctx.fill()
    else:
        # Soft blurred interior room screen shadow on one side for lived-in depth
        sil_u0, sil_u1 = (0.0, 0.24) if lamp_u > 0.5 else (0.76, 1.0)
        q_sil = project_quad_3d(
            cam,
            np.array([p_kl + e_u * sil_u0, p_kl + e_u * sil_u1, p_tl + e_u * sil_u1, p_tl + e_u * sil_u0]),
            mirror=mirror,
        )
        if q_sil and len(q_sil) >= 4:
            ctx.set_source_rgba(*fog_mix([0.28, 0.115, 0.058], dist, fog_rgb, fog_density=0.025), 0.30)
            fill_poly(ctx, q_sil)

    ctx.restore()

    # 3. Solid warm smoked-cedar koshita lower kick-panel + warm amber top catch-light rail!
    if kf > 0.02:
        q_kosh = project_quad_3d(cam, np.array([p_bl, p_br, p_kr, p_kl]), mirror=mirror)
        if q_kosh and len(q_kosh) >= 4:
            ctx.set_source_rgba(*fog_mix([0.165, 0.084, 0.060], dist, fog_rgb, fog_density=0.028), 0.97)
            fill_poly(ctx, q_kosh)
            # Warm naka-zan horizontal cap rail catching glow from washi above!
            ctx.set_source_rgba(*fog_mix([0.44, 0.23, 0.115], dist, fog_rgb, fog_density=0.024), 0.72)
            ctx.set_line_width(max(0.7, rad * 0.018))
            ctx.move_to(*q_kosh[3])
            ctx.line_to(*q_kosh[2])
            ctx.stroke()

    # 4. Delicate warm smoked-cedar kumiko lattice (thin, misty, warm amber-edged — NOT flat black!)
    m_col = fog_mix([0.195, 0.098, 0.068], dist, fog_rgb, fog_density=0.026)
    m_hi = fog_mix([0.48, 0.25, 0.12], dist, fog_rgb, fog_density=0.024)
    bar_w = max(0.55, rad * 0.015)

    for ci in range(1, n_cols):
        f = ci / n_cols
        if open_bay and f > 0.67:
            continue
        is_mid_stile = (n_cols >= 4 and ci == n_cols // 2)
        pb = p_kl * (1.0 - f) + p_kr * f
        pt = p_tl * (1.0 - f) + p_tr * f
        if mirror:
            pb, pt = mirror_z0(pb), mirror_z0(pt)
        u0, v0, _ = cam.project(pb)
        u1, v1, _ = cam.project(pt)
        ctx.set_source_rgba(*m_col, 0.84 if not is_mid_stile else 0.92)
        ctx.set_line_width(bar_w * (1.55 if is_mid_stile else 1.0))
        ctx.move_to(float(u0), float(v0))
        ctx.line_to(float(u1), float(v1))
        ctx.stroke()

    for ri in range(1, n_rows):
        f = ri / n_rows
        pl = p_kl * (1.0 - f) + p_tl * f
        pr = p_kr * (1.0 - f) + p_tr * f
        if open_bay:
            pr = pl * 0.32 + pr * 0.68
        if mirror:
            pl, pr = mirror_z0(pl), mirror_z0(pr)
        u0, v0, _ = cam.project(pl)
        u1, v1, _ = cam.project(pr)
        ctx.set_source_rgba(*m_col, 0.80)
        ctx.set_line_width(bar_w * 0.92)
        ctx.move_to(float(u0), float(v0))
        ctx.line_to(float(u1), float(v1))
        ctx.stroke()

    # Subtle outer recessed timber frame edge + warm amber inner shikii sill highlight!
    ctx.set_source_rgba(*fog_mix([0.135, 0.068, 0.052], dist, fog_rgb, fog_density=0.028), 0.90)
    ctx.set_line_width(max(0.90, rad * 0.026))
    ctx.move_to(*q[0])
    for qp in q[1:]:
        ctx.line_to(*qp)
    ctx.close_path()
    ctx.stroke()

    ctx.set_source_rgba(*m_hi, 0.52)
    ctx.set_line_width(max(0.65, rad * 0.016))
    ctx.move_to(*q[0])
    ctx.line_to(*q[1])
    ctx.stroke()


def draw_distant_upstream_ryokan(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    mirror: bool = False,
) -> None:
    """
    Richly detailed multi-story Ginzan Onsen timber inns receding upstream beyond the bridge:
    dark charred-cedar posts, warm plaster panels, recessed glowing yukimi-shoji bays, wooden
    balcony rails, curved upturned kawara roofs, and tastefully scaled distant eave lanterns!
    """
    buildings = [
        # (side, x_face, y0, y1, z_base, z_eave, z_ridge, n_floors)
        ("R", 6.35, 34.5, 42.0, 1.30, 5.10, 6.85, 2),
        ("L", -0.45, 34.0, 41.5, 1.15, 5.45, 7.25, 2),
        ("R", 6.25, 27.5, 34.8, 1.35, 6.05, 7.95, 2),
        ("L", -0.58, 26.8, 34.2, 1.12, 6.60, 8.55, 3),
        ("R", 6.22, 21.8, 27.8, 1.38, 6.35, 8.15, 2),
        ("L", -0.78, 21.0, 27.0, 1.10, 6.95, 8.85, 3),
    ]
    for b_idx, (side, xf, y0, y1, zb, ze, zr, nfl) in enumerate(buildings):
        dist = float(np.linalg.norm(np.array([xf, 0.5 * (y0 + y1), 0.5 * (zb + ze)]) - cam.eye))
        depth = 4.2 if side == "R" else -4.2
        x_back = xf + depth
        x_lo, x_hi = min(xf, x_back), max(xf, x_back)

        # Stone canal embankment base
        q_base = project_quad_3d(
            cam, np.array([[xf, y0, 0.0], [xf, y1, 0.0], [xf, y1, zb], [xf, y0, zb]]), mirror=mirror,
        )
        if q_base:
            ctx.set_source_rgba(*fog_mix([0.072, 0.048, 0.060], dist, fog_rgb, fog_density=0.034), 0.96)
            fill_poly(ctx, q_base)

        # Charred-timber & dusky warm-plaster inn body
        c_wall = [0.098, 0.052, 0.062] if side == "L" else [0.115, 0.062, 0.070]
        c_front = [0.074, 0.040, 0.050]
        draw_box_3d_faces(
            ctx, cam, x_lo, x_hi, y0, y1, zb, ze,
            c_front=tuple(c_front), c_side=tuple(c_wall), c_top=(0.08, 0.04, 0.05),
            fog_rgb=fog_rgb, mirror=mirror, alpha=0.98,
        )

        # Floor-by-floor recessed glowing shoji bays, dark timber balcony rails & mid-story eaves!
        fh = (ze - zb - 0.30) / nfl
        over = 0.68 if side == "L" else -0.68
        for fl in range(nfl):
            z_fl = zb + fl * fh
            z_w0 = z_fl + 0.34
            z_w1 = z_w0 + fh * 0.58
            # Subtle warm plaster frieze band on upper floors
            if fl >= 1:
                qf_pl = project_quad_3d(
                    cam, np.array([[x_lo, y0 - 0.02, z_fl + 0.14], [x_hi, y0 - 0.02, z_fl + 0.14],
                                   [x_hi, y0 - 0.02, z_fl + fh - 0.10], [x_lo, y0 - 0.02, z_fl + fh - 0.10]]),
                    mirror=mirror,
                )
                if qf_pl:
                    ctx.set_source_rgba(*fog_mix([0.165, 0.100, 0.098], dist, fog_rgb, fog_density=0.032), 0.88)
                    fill_poly(ctx, qf_pl)

            # Canal-facing recessed shoji bays (2 cleaner rhythmic wide bays per facade instead of cramped narrow slits!)
            n_side_bays = 2
            span_y = y1 - y0 - 0.90
            for iw in range(n_side_bays):
                wy0 = y0 + 0.45 + iw * (span_y / n_side_bays)
                wy1 = wy0 + (span_y / n_side_bays) * 0.80
                xs_w = xf + (0.04 if side == "L" else -0.04)
                br = 0.88 + 0.24 * float(rng.random())
                lu = 0.36 + 0.28 * float(rng.random())
                lv = 0.52 + 0.18 * float(rng.random())
                draw_shoji_quad_with_mullions(
                    ctx, cam,
                    np.array([xs_w, wy0, z_w0]), np.array([xs_w, wy1, z_w0]),
                    np.array([xs_w, wy1, z_w1]), np.array([xs_w, wy0, z_w1]),
                    n_cols=4, n_rows=2, dist=dist, fog_rgb=fog_rgb, br=br, mirror=mirror,
                    lamp_u=lu, lamp_v=lv, koshita_frac=0.22, open_bay=(b_idx >= 4 and fl == 1 and iw == 0),
                )

            # Front-facing recessed shoji window (facing viewer!) with warm smoked-cedar carpentry!
            x_fw0 = xf + (0.35 if side == "R" else -2.25)
            x_fw1 = xf + (2.25 if side == "R" else -0.35)
            br_f = 0.90 + 0.22 * float(rng.random())
            lu_f = 0.34 + 0.32 * float(rng.random())
            lv_f = 0.54 + 0.16 * float(rng.random())
            draw_shoji_quad_with_mullions(
                ctx, cam,
                np.array([x_fw0, y0 - 0.05, z_w0]), np.array([x_fw1, y0 - 0.05, z_w0]),
                np.array([x_fw1, y0 - 0.05, z_w1]), np.array([x_fw0, y0 - 0.05, z_w1]),
                n_cols=4, n_rows=3, dist=dist, fog_rgb=fog_rgb, br=br_f, mirror=mirror,
                lamp_u=lu_f, lamp_v=lv_f, koshita_frac=0.22, open_bay=(b_idx == 5 and fl == 1),
            )

            # Slender wooden balcony railing across upper front & canal faces
            if fl >= 1:
                for zr_off in (0.14, 0.52):
                    qr_f = project_quad_3d(
                        cam, np.array([[x_fw0 - 0.22, y0 - 0.16, z_fl + zr_off], [x_fw1 + 0.22, y0 - 0.16, z_fl + zr_off],
                                       [x_fw1 + 0.22, y0 - 0.16, z_fl + zr_off + 0.08], [x_fw0 - 0.22, y0 - 0.16, z_fl + zr_off + 0.08]]),
                        mirror=mirror,
                    )
                    if qr_f:
                        col_r = [0.44, 0.135, 0.092] if zr_off > 0.4 else [0.125, 0.062, 0.052]
                        ctx.set_source_rgba(*fog_mix(col_r, dist, fog_rgb, fog_density=0.032), 0.94)
                        fill_poly(ctx, qr_f)
                # Projecting curved mid-story tiled eave belt wrapping front & canal side!
                z_me = z_fl + 0.04
                q_mef = project_quad_3d(
                    cam, np.array([[x_lo - 0.25, y0 - 0.45, z_me - 0.14], [x_hi + 0.25, y0 - 0.45, z_me - 0.14],
                                   [x_hi + 0.10, y0, z_me + 0.20], [x_lo - 0.10, y0, z_me + 0.20]]),
                    mirror=mirror,
                )
                if q_mef:
                    ctx.set_source_rgba(*fog_mix([0.058, 0.032, 0.048], dist, fog_rgb, fog_density=0.034), 0.98)
                    fill_poly(ctx, q_mef)

        # Grand overhanging curved kawara gable roof silhouette with sunlit dusk ridge rim!
        x_eave = xf + over
        x_mid = xf + depth * 0.46
        q_roof_side = project_quad_3d(
            cam, np.array([[x_eave, y0 - 0.38, ze - 0.12], [x_eave, y1 + 0.38, ze - 0.12],
                           [x_mid, y1 + 0.15, zr], [x_mid, y0 - 0.15, zr]]),
            mirror=mirror,
        )
        if q_roof_side:
            ctx.set_source_rgba(*fog_mix([0.070, 0.042, 0.066], dist, fog_rgb, fog_density=0.034), 0.98)
            fill_poly(ctx, q_roof_side)
        q_gable = project_quad_3d(
            cam, np.array([[x_eave, y0 - 0.28, ze - 0.12], [x_mid, y0 - 0.18, zr], [x_back, y0 - 0.18, ze - 0.12]]),
            mirror=mirror,
        )
        if q_gable:
            ctx.set_source_rgba(*fog_mix([0.054, 0.030, 0.046], dist, fog_rgb, fog_density=0.034), 0.98)
            fill_poly(ctx, q_gable)
            ctx.set_source_rgba(*fog_mix([0.80, 0.40, 0.24], dist, fog_rgb, fog_density=0.030), 0.56)
            ctx.set_line_width(1.6)
            ctx.move_to(*q_gable[0])
            ctx.line_to(*q_gable[1])
            ctx.stroke()

        # Subtle small distant glowing chochin dots under far eaves (tastefully scaled — NO giant orange blobs!)
        if nfl >= 2 and y0 > 24.0:
            z_dl = zb + fh - 0.24
            for yl in (y0 + 1.6, y1 - 1.6):
                p_l = np.array([xf + over * 0.65, yl, z_dl])
                if mirror:
                    p_l = mirror_z0(p_l)
                ul, vl, zl = cam.project(p_l)
                if zl > 1.0:
                    rl = max(1.4, float(cam.K[0, 0] * 0.095 / zl))
                    g_l = cairo.RadialGradient(float(ul), float(vl), rl * 0.1, float(ul), float(vl), rl * 2.4)
                    g_l.add_color_stop_rgba(0.0, 1.0, 0.88, 0.52, 0.90)
                    g_l.add_color_stop_rgba(0.50, 0.96, 0.42, 0.16, 0.40)
                    g_l.add_color_stop_rgba(1.0, 0.94, 0.34, 0.12, 0.0)
                    ctx.set_source(g_l)
                    ctx.arc(float(ul), float(vl), rl * 2.4, 0, math.tau)
                    ctx.fill()


# =============================================================================
# Stage 3 Band 1 — Hanging Paper Chochin Lanterns, Arched Taiko-Bashi, Curved-Eave Pavilion & Gorge Rocks
# =============================================================================

def draw_chochin_lantern_3d(
    ctx: cairo.Context,
    cam: Camera3D,
    cx: float, cy: float, z_hook: float, z_center: float,
    rad_m: float, h_m: float, inten: float, style: str, kanji_id: int,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    mirror: bool = False,
) -> None:
    """
    Sculpted 3D hanging Japanese paper chochin lantern with varied brush-painted kanji/hiragana
    calligraphy ('湯', 'ゆ', '泉', '宿'), soft horizontal bamboo ribs, dark-lacquer collars &
    warm radial candle aura!
    """
    p_c = np.array([cx, cy, z_center], dtype=np.float64)
    p_h = np.array([cx, cy, z_hook], dtype=np.float64)
    p_top = np.array([cx, cy, z_center + h_m * 0.5], dtype=np.float64)
    p_bot = np.array([cx, cy, z_center - h_m * 0.5], dtype=np.float64)
    if mirror:
        p_c, p_h, p_top, p_bot = mirror_z0(p_c), mirror_z0(p_h), mirror_z0(p_top), mirror_z0(p_bot)

    uc, vc, zc = cam.project(p_c)
    if zc < 0.8 or uc < -120 or uc > W + 120 or vc < -120 or vc > H + 120:
        return
    uh, vh, _ = cam.project(p_h)
    ut, vt, _ = cam.project(p_top)
    ub, vb, _ = cam.project(p_bot)

    rx = max(2.2, float(cam.K[0, 0] * rad_m / zc))
    ry = max(3.4, 0.5 * abs(float(vb - vt)))
    dist = float(zc)

    # 1. Warm radial lantern aura halo
    halo_r = rx * (5.0 if not mirror else 5.8)
    g_halo = cairo.RadialGradient(float(uc), float(vc), rx * 0.25, float(uc), float(vc), halo_r)
    if style in ("ryokan_red", "bridge_red"):
        g_halo.add_color_stop_rgba(0.0, 1.0, 0.72, 0.30, 0.42 * min(1.0, inten * 0.54))
        g_halo.add_color_stop_rgba(0.38, 0.96, 0.36, 0.14, 0.19 * min(1.0, inten * 0.54))
        g_halo.add_color_stop_rgba(1.0, 0.90, 0.24, 0.10, 0.0)
    else:
        g_halo.add_color_stop_rgba(0.0, 1.0, 0.86, 0.46, 0.44 * min(1.0, inten * 0.54))
        g_halo.add_color_stop_rgba(0.42, 0.98, 0.52, 0.18, 0.19 * min(1.0, inten * 0.54))
        g_halo.add_color_stop_rgba(1.0, 0.94, 0.36, 0.12, 0.0)
    ctx.set_source(g_halo)
    ctx.arc(float(uc), float(vc), halo_r, 0, math.tau)
    ctx.fill()

    # 2. Dark iron hanging cord cleanly from physical eave/post hook to lantern top collar
    ctx.set_source_rgba(*fog_mix([0.04, 0.02, 0.024], dist, fog_rgb), 0.92)
    ctx.set_line_width(max(0.85, rx * 0.10))
    ctx.move_to(float(uh), float(vh))
    ctx.line_to(float(ut), float(vt))
    ctx.stroke()

    # 3. Oval Tapered Chochin Paper Body
    ctx.save()
    ctx.translate(float(uc), float(vc))
    if mirror:
        ctx.scale(1.0, -1.0)

    ctx.move_to(-rx * 0.62, -ry)
    ctx.curve_to(-rx * 1.06, -ry * 0.52, -rx * 1.06, ry * 0.52, -rx * 0.62, ry)
    ctx.line_to(rx * 0.62, ry)
    ctx.curve_to(rx * 1.06, ry * 0.52, rx * 1.06, -ry * 0.52, rx * 0.62, -ry)
    ctx.close_path()
    ctx.save()
    ctx.clip()

    g_body = cairo.RadialGradient(0.0, ry * 0.06, rx * 0.08, 0.0, 0.0, max(rx, ry) * 1.18)
    if style in ("ryokan_red", "bridge_red"):
        g_body.add_color_stop_rgba(0.0, 1.0, 0.94, 0.68, 0.99)
        g_body.add_color_stop_rgba(0.36, 1.0, 0.64, 0.24, 0.99)
        g_body.add_color_stop_rgba(0.76, 0.84, 0.23, 0.12, 0.99)
        g_body.add_color_stop_rgba(1.0, 0.54, 0.11, 0.08, 0.99)
    else:
        g_body.add_color_stop_rgba(0.0, 1.0, 0.97, 0.78, 0.99)
        g_body.add_color_stop_rgba(0.45, 1.0, 0.80, 0.38, 0.99)
        g_body.add_color_stop_rgba(0.82, 0.92, 0.48, 0.18, 0.99)
        g_body.add_color_stop_rgba(1.0, 0.66, 0.26, 0.12, 0.99)
    ctx.set_source(g_body)
    ctx.rectangle(-rx * 1.2, -ry * 1.1, rx * 2.4, ry * 2.2)
    ctx.fill()

    if style == "ryokan_cream":
        ctx.set_source_rgba(0.76, 0.16, 0.10, 0.80)
        ctx.rectangle(-rx * 1.1, -ry, rx * 2.2, ry * 0.32)
        ctx.fill()
        ctx.rectangle(-rx * 1.1, ry * 0.68, rx * 2.2, ry * 0.32)
        ctx.fill()

    # Soft curved horizontal bamboo spiral ribs
    n_ribs = 7 if ry > 12 else 4
    ctx.set_source_rgba(0.32, 0.09, 0.05, 0.26)
    ctx.set_line_width(max(0.5, ry * 0.028))
    for k in range(1, n_ribs):
        yk = -ry + (2.0 * ry * k) / n_ribs
        ctx.move_to(-rx * 1.08, yk - ry * 0.03)
        ctx.curve_to(-rx * 0.35, yk + ry * 0.055, rx * 0.35, yk + ry * 0.055, rx * 1.08, yk - ry * 0.03)
        ctx.stroke()

    # Varied organic brush-painted Japanese calligraphy motifs (zero stiff repeated '+'!)
    if rx > 7.5 and not mirror:
        ctx.set_source_rgba(0.14, 0.042, 0.032, 0.60)
        ctx.set_line_width(max(0.95, rx * 0.105))
        kid = kanji_id % 4
        if kid == 0:
            # Flowing hiragana 'ゆ' (hot bath)
            ctx.move_to(-rx * 0.34, -ry * 0.26)
            ctx.curve_to(-rx * 0.40, 0.05, -rx * 0.32, ry * 0.28, -rx * 0.16, ry * 0.24)
            ctx.curve_to(0.02, ry * 0.18, 0.06, -ry * 0.22, rx * 0.30, -ry * 0.20)
            ctx.curve_to(rx * 0.46, -ry * 0.12, rx * 0.40, ry * 0.20, rx * 0.14, ry * 0.22)
            ctx.stroke()
            ctx.move_to(rx * 0.10, -ry * 0.34)
            ctx.curve_to(rx * 0.12, 0.0, rx * 0.06, ry * 0.20, -rx * 0.04, ry * 0.38)
            ctx.stroke()
        elif kid == 1:
            # Stylized kanji '湯' (hot spring)
            # Left water radical 3 brush drops
            for dy_d in (-0.24, 0.0, 0.24):
                ctx.move_to(-rx * 0.38, ry * dy_d)
                ctx.line_to(-rx * 0.26, ry * (dy_d + 0.06))
                ctx.stroke()
            # Right sun + rising steam strokes
            ctx.rectangle(-rx * 0.12, -ry * 0.30, rx * 0.42, ry * 0.22)
            ctx.stroke()
            ctx.move_to(-rx * 0.16, 0.02)
            ctx.line_to(rx * 0.38, 0.02)
            ctx.move_to(rx * 0.06, 0.02)
            ctx.curve_to(0.0, ry * 0.18, -rx * 0.12, ry * 0.28, -rx * 0.20, ry * 0.36)
            ctx.move_to(rx * 0.26, 0.02)
            ctx.curve_to(rx * 0.26, ry * 0.22, rx * 0.20, ry * 0.32, rx * 0.08, ry * 0.36)
            ctx.stroke()
        elif kid == 2:
            # Stylized kanji '泉' (spring)
            ctx.rectangle(-rx * 0.24, -ry * 0.32, rx * 0.48, ry * 0.24)
            ctx.stroke()
            ctx.move_to(0.0, -ry * 0.04)
            ctx.line_to(0.0, ry * 0.36)
            ctx.move_to(-rx * 0.28, ry * 0.10)
            ctx.curve_to(-rx * 0.10, ry * 0.14, -rx * 0.16, ry * 0.26, -rx * 0.32, ry * 0.32)
            ctx.move_to(rx * 0.28, ry * 0.10)
            ctx.curve_to(rx * 0.12, ry * 0.16, rx * 0.16, ry * 0.26, rx * 0.34, ry * 0.32)
            ctx.stroke()
        else:
            # Traditional circular ryokan crest + wave mark
            ctx.arc(0.0, 0.0, rx * 0.40, 0.3, math.tau - 0.3)
            ctx.stroke()
            for wx_s in (-0.16, 0.0, 0.16):
                ctx.move_to(rx * wx_s, -ry * 0.18)
                ctx.curve_to(rx * (wx_s + 0.08), -ry * 0.05, rx * (wx_s - 0.08), ry * 0.06, rx * wx_s, ry * 0.18)
                ctx.stroke()

    ctx.restore()

    # 4. Dark-lacquer top & bottom wooden collars + brass lip + tassel
    collar_c = fog_mix([0.052, 0.022, 0.026], dist, fog_rgb)
    ctx.set_source_rgba(*collar_c, 0.98)
    cw, ch = rx * 0.68, max(1.6, ry * 0.14)
    ctx.rectangle(-cw, -ry - ch * 0.55, 2.0 * cw, ch)
    ctx.fill()
    ctx.rectangle(-cw * 0.96, ry - ch * 0.35, 1.92 * cw, ch)
    ctx.fill()

    ctx.set_source_rgba(0.88, 0.56, 0.24, 0.68)
    ctx.set_line_width(max(0.6, ry * 0.04))
    ctx.move_to(-cw, -ry - ch * 0.45)
    ctx.line_to(cw, -ry - ch * 0.45)
    ctx.stroke()

    ctx.set_source_rgba(*fog_mix([0.46, 0.11, 0.08], dist, fog_rgb), 0.88)
    ctx.set_line_width(max(0.8, rx * 0.13))
    ctx.move_to(0.0, ry + ch * 0.6)
    ctx.line_to(0.0, ry + ch * 2.0)
    ctx.stroke()
    ctx.restore()


def draw_taiko_bashi_bridge(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    mirror: bool = False,
) -> None:
    """
    Graceful, architecturally grounded traditional Japanese wooden arch footbridge (soribashi / taiko-bashi):
      - Gentle natural camber (~0.72m rise over 6.15m canal) seated cleanly on dressed cut-stone
        bridgehead abutments (hashizume) joining both embankments with zero floating ends!
      - Angled dark-cedar structural knee-braces (hoozue) anchoring the curved outer main arch
        girder (keta) into stone corbels on both canal walls, shadowed vaulted timber underside,
        projecting square cross-beam ends (yokogi), and visible curving weathered cedar deck planks!
      - Clean depth separation between misty recessed far balustrade (y_back) and crisp vermilion-
        lacquer near balustrade (y_front), with unified wooden lantern standards (andon-bashira)
        at pf = 0.25 & 0.75 and aged-bronze giboshi finials on end & crown posts (zero bunny-ear overlap!).
    """
    y_front = Y_BRIDGE_C - BRIDGE_HALF_W
    y_back = Y_BRIDGE_C + BRIDGE_HALF_W
    n_seg = 28
    xs = np.linspace(BRIDGE_X0, BRIDGE_X1, n_seg + 1)
    zs = bridge_deck_z(xs)
    dist_c = float(np.linalg.norm(np.array([3.1, Y_BRIDGE_C, 1.8]) - cam.eye))

    def p3(pt: np.ndarray) -> tuple[float, float]:
        p = mirror_z0(pt) if mirror else pt
        u, v, _ = cam.project(p)
        return (float(u), float(v))

    def girder_depth(t_k: float) -> float:
        # Deeper haunched timber arch girder at stone abutments (0.50m) tapering gracefully to 0.31m at crown!
        return 0.31 + 0.19 * (2.0 * t_k - 1.0) ** 2

    # 1. Cut-Stone Bridgehead Abutments (Hashizume) & Angled Timber Knee-Braces (Hoozue) Grounding BOTH Banks!
    # Left stone abutment pier seated cleanly into left ishigaki embankment & promenade
    draw_box_3d_faces(
        ctx, cam, BRIDGE_X0 - 0.28, 0.22, y_front - 0.18, y_back + 0.18, -0.05, Z_PROM + 0.04,
        c_front=(0.115, 0.078, 0.082), c_side=(0.145, 0.095, 0.092), c_top=(0.185, 0.125, 0.112),
        fog_rgb=fog_rgb, mirror=mirror, alpha=0.99,
    )
    # Left dressed granite cap slab + low entry step onto bridge
    draw_box_3d_faces(
        ctx, cam, BRIDGE_X0 - 0.34, 0.28, y_front - 0.22, y_back + 0.22, Z_PROM - 0.04, Z_PROM + 0.10,
        c_front=(0.165, 0.112, 0.105), c_side=(0.195, 0.130, 0.118), c_top=(0.24, 0.155, 0.132),
        fog_rgb=fog_rgb, mirror=mirror, alpha=0.99,
    )
    # Right stone abutment pier seated cleanly into right embankment
    draw_box_3d_faces(
        ctx, cam, 5.92, BRIDGE_X1 + 0.36, y_front - 0.18, y_back + 0.18, -0.05, Z_RIGHT_BANK - 0.02,
        c_front=(0.105, 0.070, 0.076), c_side=(0.155, 0.102, 0.096), c_top=(0.185, 0.122, 0.112),
        fog_rgb=fog_rgb, mirror=mirror, alpha=0.99,
    )
    draw_box_3d_faces(
        ctx, cam, 5.86, BRIDGE_X1 + 0.42, y_front - 0.22, y_back + 0.22, Z_RIGHT_BANK - 0.12, Z_RIGHT_BANK + 0.03,
        c_front=(0.155, 0.105, 0.100), c_side=(0.195, 0.130, 0.118), c_top=(0.23, 0.148, 0.128),
        fog_rgb=fog_rgb, mirror=mirror, alpha=0.99,
    )

    # Sturdy angled dark-cedar structural knee-braces (hoozue) supporting the arch from both canal walls!
    for y_br in (y_back - 0.10, y_front + 0.10):
        is_f = (y_br < Y_BRIDGE_C)
        c_br = [0.115, 0.056, 0.046] if is_f else [0.068, 0.034, 0.040]
        # Left diagonal timber strut (x=0.14, z=0.42 -> x=1.22, z=girder underside)
        z_l_hi = float(bridge_deck_z(1.22)) - girder_depth((1.22 - BRIDGE_X0) / (BRIDGE_X1 - BRIDGE_X0)) + 0.06
        q_bl = project_quad_3d(
            cam,
            np.array([[0.12, y_br, 0.36], [1.26, y_br, z_l_hi - 0.10],
                      [1.18, y_br, z_l_hi + 0.06], [0.12, y_br, 0.56]]),
            mirror=mirror,
        )
        if q_bl:
            ctx.set_source_rgba(*fog_mix(c_br, dist_c, fog_rgb), 0.97)
            fill_poly(ctx, q_bl)
        # Right diagonal timber strut (x=6.00, z=0.46 -> x=4.92, z=girder underside)
        z_r_hi = float(bridge_deck_z(4.92)) - girder_depth((4.92 - BRIDGE_X0) / (BRIDGE_X1 - BRIDGE_X0)) + 0.06
        q_br = project_quad_3d(
            cam,
            np.array([[6.02, y_br, 0.42], [4.88, y_br, z_r_hi - 0.10],
                      [4.96, y_br, z_r_hi + 0.06], [6.02, y_br, 0.62]]),
            mirror=mirror,
        )
        if q_br:
            ctx.set_source_rgba(*fog_mix(c_br, dist_c, fog_rgb), 0.97)
            fill_poly(ctx, q_br)

    # Helper for rendering far (recessed misty) vs near (crisp vermilion-lacquer) kōran balustrades!
    def draw_giboshi(xp: float, y_rail: float, z_top: float, scale_r: float = 1.0, misty: bool = False) -> None:
        p_fin = np.array([xp, y_rail, z_top + 0.115 * scale_r])
        if mirror:
            p_fin = mirror_z0(p_fin)
        uf, vf, zf = cam.project(p_fin)
        if zf <= 1.0:
            return
        rf = max(2.0, float(cam.K[0, 0] * 0.088 * scale_r / zf))
        fd = 0.038 if misty else 0.026
        ctx.save()
        ctx.translate(float(uf), float(vf))
        if mirror:
            ctx.scale(1.0, -1.0)
        # Aged-bronze collar ring
        ctx.set_source_rgba(*fog_mix([0.34, 0.22, 0.12], dist_c, fog_rgb, fog_density=fd), 0.96 if not misty else 0.78)
        ctx.rectangle(-rf * 1.08, rf * 0.46, rf * 2.16, rf * 0.40)
        ctx.fill()
        # Sculpted aged-bronze lotus-bud giboshi finial
        g_gb = cairo.RadialGradient(rf * 0.22, -rf * 0.26, rf * 0.10, 0.0, 0.0, rf * 1.30)
        if misty:
            g_gb.add_color_stop_rgba(0.0, *fog_mix([0.62, 0.46, 0.32], dist_c, fog_rgb, fog_density=fd), 0.85)
            g_gb.add_color_stop_rgba(1.0, *fog_mix([0.22, 0.14, 0.14], dist_c, fog_rgb, fog_density=fd), 0.85)
        else:
            g_gb.add_color_stop_rgba(0.0, *fog_mix([1.00, 0.86, 0.52], dist_c, fog_rgb, fog_density=fd), 0.99)
            g_gb.add_color_stop_rgba(0.48, *fog_mix([0.64, 0.42, 0.20], dist_c, fog_rgb, fog_density=fd), 0.99)
            g_gb.add_color_stop_rgba(1.0, *fog_mix([0.20, 0.12, 0.08], dist_c, fog_rgb, fog_density=fd), 0.99)
        ctx.set_source(g_gb)
        ctx.move_to(-rf * 0.84, rf * 0.50)
        ctx.curve_to(-rf * 1.12, -rf * 0.05, -rf * 0.68, -rf * 0.62, 0.0, -rf * 1.22)
        ctx.curve_to(rf * 0.68, -rf * 0.62, rf * 1.12, -rf * 0.05, rf * 0.84, rf * 0.50)
        ctx.close_path()
        ctx.fill()
        ctx.restore()

    def draw_railing(y_rail: float, is_front: bool) -> None:
        rail_h = 0.88 if is_front else 0.84
        fd = 0.025 if is_front else 0.044
        al = 0.98 if is_front else 0.78
        c_lacq_main = [0.72, 0.190, 0.122] if is_front else [0.36, 0.115, 0.110]
        c_lacq_hi = [1.00, 0.52, 0.26] if is_front else [0.54, 0.24, 0.20]
        c_lacq_sh = [0.34, 0.095, 0.072] if is_front else [0.22, 0.075, 0.082]
        c_cedar = [0.145, 0.068, 0.054] if is_front else [0.12, 0.060, 0.068]

        # Clean rhythmic vertical balusters (less dense on back rail so zero visual clutter!)
        bal_step = 2 if is_front else 4
        for k in range(2, n_seg - 1, bal_step):
            xk, zk = float(xs[k]), float(zs[k])
            bw = 0.036 if is_front else 0.030
            q_bal = project_quad_3d(
                cam,
                np.array([[xk - bw, y_rail, zk + 0.08], [xk + bw, y_rail, zk + 0.08],
                          [xk + bw, y_rail, zk + rail_h - 0.04], [xk - bw, y_rail, zk + rail_h - 0.04]]),
                mirror=mirror,
            )
            if q_bal:
                col_b = [0.52, 0.150, 0.105] if is_front else [0.28, 0.095, 0.098]
                ctx.set_source_rgba(*fog_mix(col_b, dist_c, fog_rgb, fog_density=fd), al * 0.96)
                fill_poly(ctx, q_bal)

        # Curved horizontal rails: bottom timber curb (jifuku), mid rail (hirageta), graceful top handrail (kasagi)
        rail_specs = (
            (0.11, 0.078, c_cedar),
            (0.48, 0.068, c_lacq_sh),
            (rail_h, 0.105, c_lacq_main),
        ) if is_front else (
            (0.46, 0.058, c_lacq_sh),
            (rail_h, 0.088, c_lacq_main),
        )
        for z_off, thick, col_r in rail_specs:
            top_pts, bot_pts = [], []
            for k in range(n_seg + 1):
                # Subtle classic upturned sori ends on the top handrail (kasagi)!
                t_k = k / n_seg
                end_flare = 0.045 * max(0.0, (abs(2.0 * t_k - 1.0) - 0.80) / 0.20) ** 2 if z_off > 0.7 else 0.0
                p_t = np.array([xs[k], y_rail, zs[k] + z_off + end_flare + thick * 0.5])
                p_b = np.array([xs[k], y_rail, zs[k] + z_off + end_flare - thick * 0.5])
                if mirror:
                    p_t, p_b = mirror_z0(p_t), mirror_z0(p_b)
                ut, vt, _ = cam.project(p_t)
                ub, vb, _ = cam.project(p_b)
                top_pts.append((float(ut), float(vt)))
                bot_pts.append((float(ub), float(vb)))
            ctx.set_source_rgba(*fog_mix(col_r, dist_c, fog_rgb, fog_density=fd), al)
            fill_poly(ctx, top_pts + bot_pts[::-1])
            if is_front and z_off >= 0.45:
                stroke_catmull_rom(
                    ctx, top_pts, base_w=2.2 if z_off > 0.7 else 1.2, z_depth=dist_c,
                    rgba=(*fog_mix(c_lacq_hi, dist_c, fog_rgb, fog_density=fd), 0.86), rng=rng, tremor=0.04,
                )

        # Sturdy symmetric square newel posts (oyabashira) — clamping down onto the main arch girder!
        post_fracs = (0.02, 0.25, 0.50, 0.75, 0.98)
        for pf in post_fracs:
            xp = BRIDGE_X0 + pf * (BRIDGE_X1 - BRIDGE_X0)
            zp = float(bridge_deck_z(xp))
            is_lantern_post = is_front and (abs(pf - 0.25) < 0.02 or abs(pf - 0.75) < 0.02)
            pw = 0.092 if (pf in (0.02, 0.50, 0.98) and is_front) else 0.078
            ph = rail_h + (0.14 if is_front else 0.10)
            z_bot_p = zp - (0.28 if is_front else 0.10)

            if is_lantern_post:
                # Unified tall cedar-and-vermilion bridge lantern standard (andon-bashira) — ZERO tacked-on sticks!
                l_spec = [c for c in CHOCHIN_SPECS if c[9] == ("chochin_bridge_L" if pf < 0.5 else "chochin_bridge_R")][0]
                zh = l_spec[2]
                draw_box_3d_faces(
                    ctx, cam, xp - pw, xp + pw, y_rail - pw, y_rail + pw, z_bot_p, zp + ph,
                    c_front=tuple(c_lacq_main), c_side=tuple(c_lacq_sh), c_top=tuple(c_cedar),
                    fog_rgb=fog_rgb, mirror=mirror, alpha=al,
                )
                # Compact dark-cedar lantern collar rising snugly from the newel post (no tall T-stick!)
                pw_u = 0.056
                draw_box_3d_faces(
                    ctx, cam, xp - pw_u, xp + pw_u, y_rail - pw_u, y_rail + pw_u, zp + ph - 0.02, zh + 0.05,
                    c_front=(0.24, 0.095, 0.068), c_side=(0.14, 0.058, 0.048), c_top=(0.28, 0.12, 0.075),
                    fog_rgb=fog_rgb, mirror=mirror, alpha=0.98,
                )
                # Sleek miniature curved cedar rain-cap nestled directly over the bridge lantern!
                q_cap = project_quad_3d(
                    cam,
                    np.array([[xp - 0.15, y_rail - 0.14, zh + 0.02], [xp + 0.15, y_rail - 0.14, zh + 0.02],
                              [xp + 0.05, y_rail, zh + 0.11], [xp - 0.05, y_rail, zh + 0.11]]),
                    mirror=mirror,
                )
                if q_cap:
                    ctx.set_source_rgba(*fog_mix([0.18, 0.075, 0.060], dist_c, fog_rgb), 0.98)
                    fill_poly(ctx, q_cap)
                # Short neat bracket arm suspending the bridge chochin lantern directly under the cap!
                p_a0, p_a1 = p3(np.array([xp, y_rail, zh + 0.01])), p3(np.array([l_spec[0], l_spec[1], zh]))
                ctx.set_source_rgba(*fog_mix([0.20, 0.082, 0.060], dist_c, fog_rgb), 0.96)
                ctx.set_line_width(2.0)
                ctx.move_to(*p_a0)
                ctx.line_to(*p_a1)
                ctx.stroke()
            else:
                draw_box_3d_faces(
                    ctx, cam, xp - pw, xp + pw, y_rail - pw, y_rail + pw, z_bot_p, zp + ph,
                    c_front=tuple(c_lacq_main), c_side=tuple(c_lacq_sh), c_top=tuple(c_cedar),
                    fog_rgb=fog_rgb, mirror=mirror, alpha=al,
                )
                # Aged-bronze giboshi finials ONLY on front end/crown posts & subtle smaller ones on back end posts
                # (skip back center crown giboshi so it NEVER double-stacks like bunny ears over the front center post!)
                if is_front and pf in (0.02, 0.50, 0.98):
                    draw_giboshi(xp, y_rail, zp + ph, scale_r=1.0, misty=False)
                elif (not is_front) and pf in (0.02, 0.98):
                    draw_giboshi(xp, y_rail, zp + ph, scale_r=0.82, misty=True)

    # 2. Far Misty Balustrade (drawn behind deck so deck planks & near rail clearly overlap it!)
    draw_railing(y_back, is_front=False)

    # 3. Vaulted Shadowed Timber Bridge Underside (Soffit) with Subtle Canal Water Bounce & Cross-Joists
    bot_back, bot_front = [], []
    for k in range(n_seg + 1):
        th_k = girder_depth(k / n_seg)
        bot_back.append(p3(np.array([xs[k], y_back, zs[k] - th_k])))
        bot_front.append(p3(np.array([xs[k], y_front, zs[k] - th_k])))
    ctx.set_source_rgba(*fog_mix([0.072, 0.036, 0.040], dist_c, fog_rgb), 0.98)
    fill_poly(ctx, bot_front + bot_back[::-1])

    for k in range(2, n_seg - 1, 2):
        th_k = girder_depth(k / n_seg)
        qj = project_quad_3d(
            cam,
            np.array([[xs[k] - 0.055, y_front, zs[k] - th_k], [xs[k] + 0.055, y_front, zs[k] - th_k],
                      [xs[k] + 0.055, y_back, zs[k] - th_k], [xs[k] - 0.055, y_back, zs[k] - th_k]]),
            mirror=mirror,
        )
        if qj:
            ctx.set_source_rgba(*fog_mix([0.19, 0.088, 0.058], dist_c, fog_rgb), 0.68)
            fill_poly(ctx, qj)

    # 4. Curving Weathered Cedar Bridge Deck Planks (Clearly Visible Over the Gentle Arch!)
    for k in range(n_seg):
        qd = project_quad_3d(
            cam,
            np.array([[xs[k], y_front, zs[k]], [xs[k + 1], y_front, zs[k + 1]],
                      [xs[k + 1], y_back, zs[k + 1]], [xs[k], y_back, zs[k]]]),
            mirror=mirror,
        )
        if qd:
            t_mid = (k + 0.5) / n_seg
            crest = math.sin(t_mid * math.pi)
            lan_warm = math.exp(-((t_mid - 0.25) / 0.14) ** 2) + math.exp(-((t_mid - 0.75) / 0.14) ** 2)
            pl_c = [
                0.19 + 0.14 * crest + 0.14 * lan_warm,
                0.105 + 0.072 * crest + 0.072 * lan_warm,
                0.082 + 0.042 * crest + 0.030 * lan_warm,
            ]
            ctx.set_source_rgba(*fog_mix(pl_c, dist_c, fog_rgb), 0.98)
            fill_poly(ctx, qd)
            ctx.set_source_rgba(*fog_mix([0.056, 0.028, 0.030], dist_c, fog_rgb), 0.56)
            ctx.set_line_width(0.95)
            ctx.move_to(*qd[0])
            ctx.line_to(*qd[3])
            ctx.stroke()

    # 5. Curved Main Outer Smoked-Cedar Arch Girder (Keta) + Vermilion Fascia Skirt + Cross-Beam Ends (Yokogi)!
    top_f, mid_f, bot_f = [], [], []
    for k in range(n_seg + 1):
        th_k = girder_depth(k / n_seg)
        top_f.append(p3(np.array([xs[k], y_front, zs[k] + 0.04])))
        mid_f.append(p3(np.array([xs[k], y_front, zs[k] - 0.12])))
        bot_f.append(p3(np.array([xs[k], y_front, zs[k] - th_k])))

    # Lower structural dark smoked-cedar arch girder (warm umber timber, NEVER flat black!)
    ctx.set_source_rgba(*fog_mix([0.135, 0.064, 0.052], dist_c, fog_rgb), 0.99)
    fill_poly(ctx, mid_f + bot_f[::-1])
    # Warm rose-amber river water bounce rim skimming the underside edge of the timber arch!
    stroke_catmull_rom(
        ctx, bot_f, base_w=2.2, z_depth=dist_c,
        rgba=(*fog_mix([0.52, 0.24, 0.13], dist_c, fog_rgb), 0.66), rng=rng, tremor=0.04,
    )
    # Upper vermilion-lacquer deck skirt fascia board
    ctx.set_source_rgba(*fog_mix([0.62, 0.162, 0.106], dist_c, fog_rgb), 0.99)
    fill_poly(ctx, top_f + mid_f[::-1])
    # Crisp dark-cedar shadow line separating upper red skirt from lower dark-cedar girder
    stroke_catmull_rom(
        ctx, mid_f, base_w=1.4, z_depth=dist_c,
        rgba=(*fog_mix([0.075, 0.034, 0.032], dist_c, fog_rgb), 0.85), rng=rng, tremor=0.03,
    )

    # Projecting square dark-cedar cross-beam ends (yokogi) jutting from the arch girder + aged brass studs!
    for k in range(3, n_seg - 2, 3):
        xk, zk = float(xs[k]), float(zs[k])
        draw_box_3d_faces(
            ctx, cam, xk - 0.055, xk + 0.055, y_front - 0.09, y_front + 0.02, zk - 0.25, zk - 0.13,
            c_front=(0.20, 0.098, 0.068), c_side=(0.12, 0.058, 0.046), c_top=(0.30, 0.145, 0.088),
            fog_rgb=fog_rgb, mirror=mirror, alpha=0.97,
        )
        p_rv = np.array([xk, y_front - 0.02, zk - 0.04])
        if mirror:
            p_rv = mirror_z0(p_rv)
        ur, vr, zr = cam.project(p_rv)
        if zr > 1.0:
            rr = max(1.3, float(cam.K[0, 0] * 0.036 / zr))
            ctx.set_source_rgba(*fog_mix([0.90, 0.65, 0.30], dist_c, fog_rgb), 0.90)
            ctx.arc(float(ur), float(vr), rr, 0, math.tau)
            ctx.fill()

    # 6. Near Vermilion-Lacquer Kōran Balustrade + Unified Hanging Bridge Chochin Lanterns!
    draw_railing(y_front, is_front=True)
    for (cx, cy, zh, cz, rm, hm, inten, st, kid, lbl) in CHOCHIN_SPECS:
        if lbl.startswith("chochin_bridge"):
            draw_chochin_lantern_3d(ctx, cam, cx, cy, zh, cz, rm, hm, inten, st, kid, rng, fog_rgb, mirror=mirror)


def draw_sculpted_volcanic_boulder_2x(
    ctx: cairo.Context,
    cx: float, cy: float, rx: float, ry: float,
    dist: float, moss_amt: float,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    mirror: bool = False,
) -> None:
    """
    Craggy angular stratified Kyoto volcanic gorge basalt boulder with multi-plane
    chiselled rock facets, deep shadow clefts, warm amber canal-side bounce, wet
    waterline darkening, and irregular clustered moss cushions tucked along flat top
    ledges (ZERO smooth round brown bread-roll domes!).
    """
    ctx.save()
    ctx.translate(cx, cy)
    if mirror:
        ctx.scale(1.0, -1.0)

    # Outer angular craggy silhouette built from fractured rock shelf points (subtle slight curve bevels)
    j = lambda a, b: float(rng.uniform(a, b))
    pts_outer = [
        (-rx * j(0.95, 1.08), ry * j(0.45, 0.62)),
        (-rx * j(0.84, 0.96), -ry * j(0.18, 0.34)),
        (-rx * j(0.42, 0.56), -ry * j(0.68, 0.86)),
        (rx * j(-0.08, 0.14), -ry * j(0.76, 0.94)),
        (rx * j(0.46, 0.62), -ry * j(0.58, 0.76)),
        (rx * j(0.86, 1.02), -ry * j(0.08, 0.24)),
        (rx * j(0.94, 1.08), ry * j(0.46, 0.64)),
        (rx * j(0.28, 0.42), ry * j(0.68, 0.82)),
        (-rx * j(0.32, 0.48), ry * j(0.66, 0.80)),
    ]

    def trace_crag(pts: list[tuple[float, float]]) -> None:
        ctx.move_to(*pts[0])
        for px, py in pts[1:]:
            ctx.line_to(px, py)
        ctx.close_path()

    # Base deep volcanic basalt crag body
    trace_crag(pts_outer)
    g_base = cairo.LinearGradient(-rx, -ry * 0.6, rx, ry * 0.7)
    g_base.add_color_stop_rgba(0.0, *fog_mix([0.145, 0.082, 0.084], dist, fog_rgb), 0.99)
    g_base.add_color_stop_rgba(0.45, *fog_mix([0.070, 0.046, 0.056], dist, fog_rgb), 0.99)
    g_base.add_color_stop_rgba(1.0, *fog_mix([0.026, 0.016, 0.024], dist, fog_rgb), 0.99)
    ctx.set_source(g_base)
    ctx.fill_preserve()

    ctx.save()
    ctx.clip()

    # Planar Facet 1: Top-left sunlit/lantern-lit fractured stone table catching warm amber & twilight sky!
    facet_top = [
        pts_outer[1], pts_outer[2], pts_outer[3], pts_outer[4],
        (rx * 0.18, -ry * 0.14), (-rx * 0.36, -ry * 0.08),
    ]
    g_top = cairo.LinearGradient(-rx * 0.8, -ry * 0.8, rx * 0.5, 0.0)
    g_top.add_color_stop_rgba(0.0, *fog_mix([0.24, 0.128, 0.100], dist, fog_rgb), 0.92)
    g_top.add_color_stop_rgba(1.0, *fog_mix([0.105, 0.068, 0.076], dist, fog_rgb), 0.85)
    ctx.set_source(g_top)
    fill_poly(ctx, facet_top)

    # Planar Facet 2: Deep shadowed right-side fractured rock cliff plane
    facet_sh = [
        (rx * 0.14, -ry * 0.18), pts_outer[4], pts_outer[5], pts_outer[6], pts_outer[7], (rx * 0.02, ry * 0.32),
    ]
    ctx.set_source_rgba(*fog_mix([0.024, 0.014, 0.022], dist, fog_rgb), 0.84)
    fill_poly(ctx, facet_sh)

    # Planar Facet 3: Canal-facing left chiselled basalt ledge illuminated by ryokan lantern warmth!
    facet_left = [
        pts_outer[0], pts_outer[1], (-rx * 0.36, -ry * 0.08), (-rx * 0.18, ry * 0.42), pts_outer[8],
    ]
    g_lf = cairo.LinearGradient(-rx, 0.0, -rx * 0.15, ry * 0.4)
    g_lf.add_color_stop_rgba(0.0, *fog_mix([0.34, 0.165, 0.095], dist, fog_rgb), 0.72)
    g_lf.add_color_stop_rgba(1.0, *fog_mix([0.088, 0.052, 0.056], dist, fog_rgb), 0.55)
    ctx.set_source(g_lf)
    fill_poly(ctx, facet_left)

    # Crisp dark fracture crease lines delineating basalt strata!
    ctx.set_source_rgba(*fog_mix([0.016, 0.010, 0.016], dist, fog_rgb), 0.72)
    ctx.set_line_width(max(1.2, rx * 0.025))
    ctx.move_to(-rx * 0.78, -ry * 0.14)
    ctx.line_to(-rx * 0.34, -ry * 0.08)
    ctx.line_to(rx * 0.16, -ry * 0.16)
    ctx.line_to(rx * 0.62, -ry * 0.34)
    ctx.stroke()
    ctx.move_to(-rx * 0.34, -ry * 0.08)
    ctx.line_to(-rx * 0.18, ry * 0.46)
    ctx.stroke()

    # Damp dark mineral waterline clinging to lower base of rock
    g_wet = cairo.LinearGradient(0, ry * 0.20, 0, ry * 0.82)
    g_wet.add_color_stop_rgba(0.0, 0.014, 0.010, 0.016, 0.0)
    g_wet.add_color_stop_rgba(1.0, 0.012, 0.008, 0.014, 0.78)
    ctx.set_source(g_wet)
    ctx.rectangle(-rx * 1.2, ry * 0.16, rx * 2.4, ry * 0.80)
    ctx.fill()

    # Textured organic moss cushions tucked naturally in top rock shelf & crevices (crisp dabs, not soft green dome!)
    for _ in range(int(7 + 8 * moss_amt)):
        mx = float(rng.uniform(-0.58, 0.52)) * rx
        my = float(rng.uniform(-0.76, -0.14)) * ry
        mr = rx * float(rng.uniform(0.08, 0.19))
        ctx.set_source_rgba(*fog_mix([0.088 + 0.055 * rng.random(), 0.128 + 0.052 * rng.random(), 0.056], dist, fog_rgb), 0.76 * moss_amt)
        ctx.save()
        ctx.translate(mx, my)
        ctx.scale(1.45, 0.56)
        ctx.arc(0, 0, mr, 0, math.tau)
        ctx.fill()
        ctx.restore()

    ctx.restore()
    ctx.restore()


def draw_right_bank_pavilion_and_boulders(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    mirror: bool = False,
) -> None:
    """
    Right canal bank mountain-gorge charm:
      1. Curved-eave traditional wooden riverside onsen gazebo (azumaya) with sweeping
         concave upturned kawara roof (sori), tile ribs, bronze ridge finials, timber
         balustrade, warm footbath bench, and glowing hanging chochin lantern!
      2. Sculpted organic cubic-Bezier volcanic gorge rock outcrops & velvety moss ledges
         lining the right embankment & stepping naturally into the steaming canal!
    """
    dist_p = float(np.linalg.norm(np.array([PAV_X0 + 1.0, 0.5 * (PAV_Y0 + PAV_Y1), 2.8]) - cam.eye))

    # Stone foundation plinth & dark wooden deck
    draw_box_3d_faces(
        ctx, cam, PAV_X0 - 0.12, PAV_X1, PAV_Y0, PAV_Y1, PAV_Z_BASE - 0.05, PAV_Z_BASE + 0.24,
        c_front=(0.088, 0.048, 0.048), c_side=(0.118, 0.064, 0.058), c_top=(0.14, 0.080, 0.064),
        fog_rgb=fog_rgb, mirror=mirror,
    )
    q_floor = project_quad_3d(
        cam,
        np.array([[PAV_X0, PAV_Y0 + 0.15, PAV_Z_BASE + 0.25], [PAV_X1 - 0.2, PAV_Y0 + 0.15, PAV_Z_BASE + 0.25],
                  [PAV_X1 - 0.2, PAV_Y1 - 0.15, PAV_Z_BASE + 0.25], [PAV_X0, PAV_Y1 - 0.15, PAV_Z_BASE + 0.25]]),
        mirror=mirror,
    )
    if q_floor:
        ctx.set_source_rgba(*fog_mix([0.56, 0.28, 0.12], dist_p, fog_rgb), 0.74)
        fill_poly(ctx, q_floor)

    # Corner dark-cedar timber posts + carved eave tie beams
    for px, py in ((PAV_X1 - 0.25, PAV_Y1 - 0.20), (PAV_X0 + 0.15, PAV_Y1 - 0.20),
                   (PAV_X1 - 0.25, PAV_Y0 + 0.20), (PAV_X0 + 0.15, PAV_Y0 + 0.20)):
        draw_box_3d_faces(
            ctx, cam, px - 0.09, px + 0.09, py - 0.09, py + 0.09, PAV_Z_BASE + 0.22, PAV_Z_EAVE + 0.08,
            c_front=(0.086, 0.042, 0.038), c_side=(0.122, 0.058, 0.044), c_top=(0.06, 0.03, 0.03),
            fog_rgb=fog_rgb, mirror=mirror,
        )

    # Low wooden balustrade & soaking-bench interior
    for zr in (PAV_Z_BASE + 0.56, PAV_Z_BASE + 0.96):
        draw_box_3d_faces(
            ctx, cam, PAV_X0 + 0.05, PAV_X0 + 0.22, PAV_Y0 + 0.15, PAV_Y1 - 0.15, zr - 0.045, zr + 0.045,
            c_front=(0.11, 0.046, 0.038), c_side=(0.26, 0.112, 0.065), c_top=(0.32, 0.148, 0.084),
            fog_rgb=fog_rgb, mirror=mirror,
        )
    draw_box_3d_faces(
        ctx, cam, PAV_X0 + 0.72, PAV_X1 - 0.45, PAV_Y0 + 0.55, PAV_Y1 - 0.55, PAV_Z_BASE + 0.24, PAV_Z_BASE + 0.74,
        c_front=(0.19, 0.090, 0.050), c_side=(0.30, 0.142, 0.066), c_top=(0.46, 0.24, 0.12),
        fog_rgb=fog_rgb, mirror=mirror,
    )

    # Hanging glowing gold-amber pavilion chochin lantern!
    for (cx, cy, zh, cz, rm, hm, inten, st, kid, lbl) in CHOCHIN_SPECS:
        if lbl == "chochin_pavilion":
            draw_chochin_lantern_3d(ctx, cam, cx, cy, zh, cz, rm, hm, inten, st, kid, rng, fog_rgb, mirror=mirror)

    # Sweeping CURVED CONCAVE Traditional Japanese Kawara Pavilion Roof (Sori!)
    e_ov = 0.68
    rx0, rx1 = PAV_X0 - e_ov, PAV_X1 + e_ov
    ry0, ry1 = PAV_Y0 - e_ov, PAV_Y1 + e_ov
    peak_x = 0.5 * (PAV_X0 + PAV_X1)
    peak_y0, peak_y1 = PAV_Y0 + 0.85, PAV_Y1 - 0.85

    def pproj(pt: np.ndarray) -> tuple[float, float]:
        p = mirror_z0(pt) if mirror else pt
        u, v, _ = cam.project(p)
        return (float(u), float(v))

    # Warm glowing timber soffit underside
    q_sof = project_quad_3d(
        cam, np.array([[rx0, ry0, PAV_Z_EAVE], [rx1, ry0, PAV_Z_EAVE], [rx1, ry1, PAV_Z_EAVE], [rx0, ry1, PAV_Z_EAVE]]),
        mirror=mirror,
    )
    if q_sof:
        ctx.set_source_rgba(*fog_mix([0.28, 0.125, 0.058], dist_p, fog_rgb), 0.96)
        fill_poly(ctx, q_sof)

    # Exposed wooden rafters under the pavilion front & canal eaves
    for k in range(1, 7):
        t = k / 7.0
        p0 = pproj(np.array([rx0 + 0.06, ry0 + t * (ry1 - ry0), PAV_Z_EAVE - 0.03]))
        p1 = pproj(np.array([PAV_X0 + 0.2, ry0 + t * (ry1 - ry0), PAV_Z_EAVE + 0.16]))
        ctx.set_source_rgba(*fog_mix([0.078, 0.036, 0.032], dist_p, fog_rgb), 0.90)
        ctx.set_line_width(1.6)
        ctx.move_to(*p0)
        ctx.line_to(*p1)
        ctx.stroke()

    # Canal-facing CURVED concave roof slope (-X) with upturned eave corners (sori!)
    fl_c = pproj(np.array([rx0 - 0.10, ry0 - 0.08, PAV_Z_EAVE + 0.24]))  # upturned front-left corner
    bl_c = pproj(np.array([rx0 - 0.08, ry1 + 0.08, PAV_Z_EAVE + 0.20]))  # upturned back-left corner
    mid_l = pproj(np.array([rx0, 0.5 * (ry0 + ry1), PAV_Z_EAVE + 0.04]))
    pk_f = pproj(np.array([peak_x, peak_y0, PAV_Z_PEAK]))
    pk_b = pproj(np.array([peak_x, peak_y1, PAV_Z_PEAK]))
    fr_c = pproj(np.array([rx1 + 0.08, ry0 - 0.08, PAV_Z_EAVE + 0.24]))
    sag_fl = pproj(np.array([0.5 * (rx0 + peak_x), 0.5 * (ry0 + peak_y0), 0.5 * (PAV_Z_EAVE + PAV_Z_PEAK) - 0.18]))
    sag_bl = pproj(np.array([0.5 * (rx0 + peak_x), 0.5 * (ry1 + peak_y1), 0.5 * (PAV_Z_EAVE + PAV_Z_PEAK) - 0.18]))
    sag_fr = pproj(np.array([0.5 * (rx1 + peak_x), 0.5 * (ry0 + peak_y0), 0.5 * (PAV_Z_EAVE + PAV_Z_PEAK) - 0.18]))

    # Left curved slope fill
    g_rl = cairo.LinearGradient(pk_f[0], pk_f[1], fl_c[0], fl_c[1])
    g_rl.add_color_stop_rgba(0.0, *fog_mix([0.135, 0.072, 0.088], dist_p, fog_rgb), 0.99)
    g_rl.add_color_stop_rgba(1.0, *fog_mix([0.058, 0.032, 0.050], dist_p, fog_rgb), 0.99)
    ctx.set_source(g_rl)
    ctx.move_to(*fl_c)
    ctx.curve_to(mid_l[0], mid_l[1], mid_l[0], mid_l[1], *bl_c)
    ctx.curve_to(sag_bl[0], sag_bl[1], sag_bl[0], sag_bl[1], *pk_b)
    ctx.line_to(*pk_f)
    ctx.curve_to(sag_fl[0], sag_fl[1], sag_fl[0], sag_fl[1], *fl_c)
    ctx.close_path()
    ctx.fill()

    # Curved kawara tile rib lines flowing gracefully down the concave slope!
    for k in range(1, 9):
        t = k / 9.0
        pt_bot = pproj(np.array([rx0 - 0.06 * abs(2 * t - 1), ry0 + t * (ry1 - ry0), PAV_Z_EAVE + 0.05 + 0.16 * (2 * t - 1) ** 2]))
        pt_top = pproj(np.array([peak_x, peak_y0 + t * (peak_y1 - peak_y0), PAV_Z_PEAK]))
        pt_mid = pproj(np.array([0.5 * (rx0 + peak_x), 0.5 * (ry0 + peak_y0) + t * (0.5 * (ry1 - ry0 + peak_y1 - peak_y0)), 0.5 * (PAV_Z_EAVE + PAV_Z_PEAK) - 0.16]))
        ctx.set_source_rgba(*fog_mix([0.28, 0.14, 0.12], dist_p, fog_rgb), 0.45)
        ctx.set_line_width(1.2)
        ctx.move_to(*pt_top)
        ctx.curve_to(pt_mid[0], pt_mid[1], pt_mid[0], pt_mid[1], *pt_bot)
        ctx.stroke()

    # Front curved concave triangular gable/hip slope (-Y)
    g_rf = cairo.LinearGradient(pk_f[0], pk_f[1], fl_c[0], fl_c[1])
    g_rf.add_color_stop_rgba(0.0, *fog_mix([0.088, 0.048, 0.064], dist_p, fog_rgb), 0.99)
    g_rf.add_color_stop_rgba(1.0, *fog_mix([0.044, 0.024, 0.038], dist_p, fog_rgb), 0.99)
    ctx.set_source(g_rf)
    ctx.move_to(*fl_c)
    ctx.curve_to(sag_fl[0], sag_fl[1], sag_fl[0], sag_fl[1], *pk_f)
    ctx.curve_to(sag_fr[0], sag_fr[1], sag_fr[0], sag_fr[1], *fr_c)
    ctx.close_path()
    ctx.fill()

    # Heavy curved ceramic ridge tile & bronze twilight sunlit rim along roof hip!
    ctx.set_source_rgba(*fog_mix([0.82, 0.42, 0.24], dist_p, fog_rgb), 0.78)
    ctx.set_line_width(2.4)
    ctx.move_to(*fl_c)
    ctx.curve_to(sag_fl[0], sag_fl[1], sag_fl[0], sag_fl[1], *pk_f)
    ctx.stroke()
    # Sculpted onigawara ceramic ridge finial crowning the pavilion peak!
    ctx.set_source_rgba(*fog_mix([0.12, 0.06, 0.07], dist_p, fog_rgb), 0.98)
    ctx.arc(pk_f[0], pk_f[1] - (4.0 if not mirror else -4.0), 5.5, 0, math.tau)
    ctx.fill()

    # 2. Natural Craggy Volcanic Gorge Boulders & Terraced Mossy Rock Ledges on Right Bank (far-to-near!)
    boulders = [
        # (x, y, z_base, rx_m, rz_m, moss_amt) — varied natural rock sizes hugging the right ishigaki!
        (6.02, 17.10, -0.04, 0.76, 0.56, 0.65),
        (6.62, 16.20, 1.12, 0.96, 0.74, 0.72),
        (5.94, 10.90, -0.05, 0.72, 0.52, 0.68),
        (6.95, 10.05, 1.12, 1.15, 0.96, 0.78),
        (6.06, 8.20, -0.05, 0.66, 0.48, 0.66),
        (6.95, 7.15, 1.15, 1.22, 1.08, 0.76),
        (5.98, 5.40, -0.06, 0.82, 0.62, 0.72),
        (7.20, 4.70, 1.10, 1.38, 1.22, 0.80),
    ]
    for bx, by, bz, rx_m, rz_m, moss in boulders:
        p_c = np.array([bx, by, bz + rz_m * 0.42], dtype=np.float64)
        if mirror:
            p_c = mirror_z0(p_c)
        uc, vc, zc = cam.project(p_c)
        if zc < 1.0 or uc < -200 or uc > W + 200 or vc < -200 or vc > H + 200:
            continue
        rx_p = float(cam.K[0, 0] * rx_m / zc)
        ry_p = float(cam.K[0, 0] * rz_m * 0.62 / zc)
        draw_sculpted_volcanic_boulder_2x(ctx, float(uc), float(vc), rx_p, ry_p, float(zc), moss, rng, fog_rgb, mirror=mirror)


# =============================================================================
# Stage 3 Band 2 — Left 3-Story Ryokan Joinery, Clean Eaves, Noren, Tubs, Spout & Chochin
# =============================================================================

def draw_left_ryokan_3d_joinery(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    mirror: bool = False,
) -> None:
    """
    Cleanly engineered 3-story dark-timber Ginzan Onsen ryokan joinery:
      - 3F crown kawara roof eaves, 2F overhanging kawara roof eaves, AND clean narrow
        tiled 1F eave drip visor (hisashi) projecting cleanly forward of the 2F balcony
        (ZERO awkward triangles cutting across 1F windows!).
      - Cantilevered 2F wooden kōran balcony with slender corbel beam ends, vertical
        lattice spindles & vermilion-lacquer handrail.
      - Split bathhouse noren curtain ('ゆ' hiragana + rising steam motif) across entrance.
      - Dressed 3D granite canal coping curb cap along x = -0.22..0.04, z = 1.04..1.14.
      - Steaming wooden cedar bath tubs with copper hoops & neatly folded ivory tenugui towel!
      - Weathered bamboo/cedar thermal spring spout (kakehi) with timber bracket & frothy
        painterly pouring water arc + organic splash droplets!
      - Full row of glowing red-and-cream paper chochin lanterns with varied brush kanji!
    """
    # 1. 3F & 2F Overhanging Kawara Eaves: Full-Span 3D Smoked-Cedar Rafters (Taruki), Purlin, Fascia & Nokimaru Tiles
    #    (All start cleanly off-screen at Y_RYOKAN_0 so rafters & fascia NEVER chop off inside the upper-left corner!)
    eave_tiers = [
        (X_EAVE3_OUT, X_LOFT_WALL, Z_3F_TOP - 0.10, Y_RYOKAN_0, Y_RYOKAN_1 + 0.35, 0.56, 0.34),
        (X_EAVE2_OUT, X_RYOKAN_WALL, Z_2F_TOP - 0.04, Y_RYOKAN_0, Y_RYOKAN_1 + 0.45, 0.52, 0.36),
    ]
    for x_out, x_in, z_e, y0_e, y1_e, step_y, rise_in in eave_tiers:
        # Inner wall-junction cedar cornice beam (keta) + mid-span eave purlin sealing soffit cleanly to the wall!
        for frac_p, hw_p, c_p in (
            (0.04, 0.065, [0.105, 0.048, 0.038]),
            (0.54, 0.045, [0.088, 0.040, 0.034]),
        ):
            xp = x_in + frac_p * (x_out - x_in)
            zp = (z_e + rise_in) - frac_p * rise_in
            qp = project_quad_3d(
                cam,
                np.array([[xp - hw_p, y0_e, zp - 0.035], [xp + hw_p, y0_e, zp - 0.035],
                          [xp + hw_p, y1_e, zp - 0.035], [xp - hw_p, y1_e, zp - 0.035]]),
                mirror=mirror,
            )
            if qp:
                ctx.set_source_rgba(*fog_mix(c_p, 11.0, fog_rgb), 0.94)
                fill_poly(ctx, qp)

        # Continuous 3D smoked-cedar rafters (taruki) running off-screen left -> far end (with warm lantern-lit near side!)
        ys_raf = np.arange(y0_e + 0.16, y1_e, step_y)
        rw = 0.034
        rh = 0.058
        for ty in ys_raf[::-1]:
            p_out = np.array([x_out, ty, z_e - 0.04])
            p_in = np.array([x_in, ty, z_e + rise_in - 0.04])
            dist = float(np.linalg.norm(p_out - cam.eye))
            # Underside of timber rafter
            q_rb = project_quad_3d(
                cam,
                np.array([[x_in, ty - rw, z_e + rise_in - rh], [x_out, ty - rw, z_e - rh],
                          [x_out, ty + rw, z_e - rh], [x_in, ty + rw, z_e + rise_in - rh]]),
                mirror=mirror,
            )
            if q_rb:
                ctx.set_source_rgba(*fog_mix([0.078, 0.036, 0.032], dist, fog_rgb), 0.94)
                fill_poly(ctx, q_rb)
            # Near (-Y) side face of rafter catching warm golden upward bounce from 2F shoji & chochin!
            q_rs = project_quad_3d(
                cam,
                np.array([[x_in, ty - rw, z_e + rise_in - rh], [x_out, ty - rw, z_e - rh],
                          [x_out, ty - rw, z_e], [x_in, ty - rw, z_e + rise_in]]),
                mirror=mirror,
            )
            if q_rs:
                ctx.set_source_rgba(*fog_mix([0.165, 0.078, 0.046], dist, fog_rgb), 0.92)
                fill_poly(ctx, q_rs)

        # Two-step sculpted dark smoked-cedar eave fascia (kayaoi) + ceramic kawara eave drip course!
        q_fasc = project_quad_3d(
            cam,
            np.array([[x_out, y0_e, z_e - 0.08], [x_out, y1_e, z_e - 0.08],
                      [x_out, y1_e, z_e + 0.16], [x_out, y0_e, z_e + 0.16]]),
            mirror=mirror,
        )
        if q_fasc:
            ctx.set_source_rgba(*fog_mix([0.048, 0.024, 0.032], 11.0, fog_rgb), 0.99)
            fill_poly(ctx, q_fasc)
        q_lip = project_quad_3d(
            cam,
            np.array([[x_out + 0.03, y0_e, z_e + 0.05], [x_out + 0.03, y1_e, z_e + 0.05],
                      [x_out + 0.03, y1_e, z_e + 0.18], [x_out + 0.03, y0_e, z_e + 0.18]]),
            mirror=mirror,
        )
        if q_lip:
            ctx.set_source_rgba(*fog_mix([0.068, 0.038, 0.054], 11.0, fog_rgb), 0.99)
            fill_poly(ctx, q_lip)
            p_r0 = np.array([x_out + 0.03, y0_e, z_e + 0.18])
            p_r1 = np.array([x_out + 0.03, y1_e, z_e + 0.18])
            if mirror:
                p_r0, p_r1 = mirror_z0(p_r0), mirror_z0(p_r1)
            seg = cam.project_polyline(np.array([p_r0, p_r1]))
            if len(seg) >= 2:
                stroke_catmull_rom(ctx, [(p[0], p[1]) for p in seg], base_w=2.1, z_depth=10.5,
                                   rgba=(0.84, 0.43, 0.24, 0.74), rng=rng, tremor=0.04)

        for ty in ys_raf[::-1]:
            p_med = np.array([x_out + 0.04, ty, z_e + 0.11])
            if mirror:
                p_med = mirror_z0(p_med)
            dist = float(np.linalg.norm(p_med - cam.eye))
            um, vm, zm = cam.project(p_med)
            if zm > 1.2 and -30 <= um <= W + 30 and -30 <= vm <= H + 30:
                rk = max(1.6, float(cam.K[0, 0] * 0.064 / zm))
                ctx.set_source_rgba(*fog_mix([0.072, 0.042, 0.056], dist, fog_rgb), 0.97)
                ctx.arc(float(um), float(vm), rk, 0, math.tau)
                ctx.fill()
                ctx.set_source_rgba(*fog_mix([0.82, 0.44, 0.24], dist, fog_rgb), 0.64)
                ctx.set_line_width(max(0.65, rk * 0.22))
                ctx.arc(float(um), float(vm), rk * 0.82, -0.9, 0.9)
                ctx.stroke()

    # 2. 2nd-Floor Continuous Cantilevered Veranda (Kōran) + Properly Layered 1F Hisashi Drip Visor!
    #    Starts cleanly off-screen at Y_RYOKAN_0 (-1.50) -> ZERO chopped-off railing or left black triangle!
    y_b0, y_b1 = Y_RYOKAN_0, Y_RYOKAN_1

    # 2a. Back-most layer: Shadowed warm-cedar balcony underside ceiling + continuous wall header beam & flush timber ribs!
    q_bfloor = project_quad_3d(
        cam,
        np.array([[X_RYOKAN_WALL, y_b0, Z_1F_TOP + 0.02], [X_BALCONY_OUT, y_b0, Z_1F_TOP + 0.02],
                  [X_BALCONY_OUT, y_b1, Z_1F_TOP + 0.02], [X_RYOKAN_WALL, y_b1, Z_1F_TOP + 0.02]]),
        mirror=mirror,
    )
    if q_bfloor:
        ctx.set_source_rgba(*fog_mix([0.092, 0.044, 0.036], 11.0, fog_rgb), 0.99)
        fill_poly(ctx, q_bfloor)

    # Continuous dark smoked-cedar wall header beam (keta) anchoring the balcony underside cleanly into the 1F wall
    q_bwall_beam = project_quad_3d(
        cam,
        np.array([[X_RYOKAN_WALL + 0.04, y_b0, Z_1F_TOP - 0.06], [X_RYOKAN_WALL + 0.04, y_b1, Z_1F_TOP - 0.06],
                  [X_RYOKAN_WALL + 0.04, y_b1, Z_1F_TOP + 0.03], [X_RYOKAN_WALL + 0.04, y_b0, Z_1F_TOP + 0.03]]),
        mirror=mirror,
    )
    if q_bwall_beam:
        ctx.set_source_rgba(*fog_mix([0.115, 0.054, 0.042], 11.0, fog_rgb), 0.99)
        fill_poly(ctx, q_bwall_beam)

    # Sleek flush under-veranda ceiling joists (hari) seated neatly inside the soffit (ZERO protruding sawtooth blocks!)
    for py_br in np.arange(y_b0 + 0.20, y_b1 - 0.08, 0.62)[::-1]:
        p_j0 = np.array([X_RYOKAN_WALL + 0.04, py_br, Z_1F_TOP + 0.005])
        p_j1 = np.array([X_BALCONY_OUT - 0.02, py_br, Z_1F_TOP + 0.005])
        if mirror:
            p_j0, p_j1 = mirror_z0(p_j0), mirror_z0(p_j1)
        seg_j = cam.project_polyline(np.array([p_j0, p_j1]))
        if len(seg_j) >= 2:
            dist_j = float(np.linalg.norm(p_j1 - cam.eye))
            stroke_catmull_rom(
                ctx, [(p[0], p[1]) for p in seg_j], base_w=2.4, z_depth=dist_j,
                rgba=(*fog_mix([0.155, 0.074, 0.048], dist_j, fog_rgb), 0.86), rng=rng, tremor=0.02,
            )

    # Warm golden balcony deck boards & continuous smoked-cedar sill fascia beam (keta)
    q_bdeck = project_quad_3d(
        cam,
        np.array([[X_RYOKAN_WALL, y_b0, Z_2F_FLOOR], [X_BALCONY_OUT, y_b0, Z_2F_FLOOR],
                  [X_BALCONY_OUT, y_b1, Z_2F_FLOOR], [X_RYOKAN_WALL, y_b1, Z_2F_FLOOR]]),
        mirror=mirror,
    )
    if q_bdeck:
        ctx.set_source_rgba(*fog_mix([0.24, 0.125, 0.068], 11.0, fog_rgb), 0.97)
        fill_poly(ctx, q_bdeck)
    q_bfasc = project_quad_3d(
        cam,
        np.array([[X_BALCONY_OUT, y_b0, Z_1F_TOP], [X_BALCONY_OUT, y_b1, Z_1F_TOP],
                  [X_BALCONY_OUT, y_b1, Z_2F_FLOOR + 0.06], [X_BALCONY_OUT, y_b0, Z_2F_FLOOR + 0.06]]),
        mirror=mirror,
    )
    if q_bfasc:
        ctx.set_source_rgba(*fog_mix([0.125, 0.058, 0.044], 11.0, fog_rgb), 0.99)
        fill_poly(ctx, q_bfasc)

    # 2b. Front-layered Sleek 1F Hisashi Tiled Drip Visor projecting cleanly from balcony sill (X_BALCONY_OUT -> X_EAVE1_OUT)!
    #     Drawn AFTER balcony underside & joists so painter's order is 100% true 3D (strictly below 2F railing!).
    z_h_out = Z_1F_TOP - 0.14   # 3.18m outer drip lip (supports 1F chochin hooks at z=3.16m!)
    z_h_in = Z_1F_TOP + 0.10    # 3.42m meets balcony sill fascia cleanly below z_rbot (3.52m)!
    y_h0, y_h1 = Y_RYOKAN_0, Y_RYOKAN_1 + 0.35
    q_hslope = project_quad_3d(
        cam,
        np.array([[X_EAVE1_OUT, y_h0, z_h_out], [X_EAVE1_OUT, y_h1, z_h_out],
                  [X_BALCONY_OUT, y_h1, z_h_in], [X_BALCONY_OUT, y_h0, z_h_in]]),
        mirror=mirror,
    )
    if q_hslope:
        ctx.set_source_rgba(*fog_mix([0.072, 0.044, 0.058], 10.5, fog_rgb), 0.99)
        fill_poly(ctx, q_hslope)
        # Subtle pitched ibushi-kawara tile ribs flowing down the narrow 1F rain visor
        for ty_t in np.arange(y_h0 + 0.18, y_h1, 0.36)[::-1]:
            p_t0 = np.array([X_BALCONY_OUT, ty_t, z_h_in])
            p_t1 = np.array([X_EAVE1_OUT, ty_t, z_h_out])
            if mirror:
                p_t0, p_t1 = mirror_z0(p_t0), mirror_z0(p_t1)
            seg_t = cam.project_polyline(np.array([p_t0, p_t1]))
            if len(seg_t) >= 2:
                dist_t = float(np.linalg.norm(p_t1 - cam.eye))
                stroke_catmull_rom(
                    ctx, [(p[0], p[1]) for p in seg_t], base_w=1.4, z_depth=dist_t,
                    rgba=(*fog_mix([0.24, 0.12, 0.10], dist_t, fog_rgb), 0.56), rng=rng, tremor=0.02,
                )

    # Crisp dark smoked-cedar outer fascia edge of 1F hisashi + circular nokimaru tile-end medallions!
    q_hfasc = project_quad_3d(
        cam,
        np.array([[X_EAVE1_OUT, y_h0, z_h_out - 0.065], [X_EAVE1_OUT, y_h1, z_h_out - 0.065],
                  [X_EAVE1_OUT, y_h1, z_h_out + 0.035], [X_EAVE1_OUT, y_h0, z_h_out + 0.035]]),
        mirror=mirror,
    )
    if q_hfasc:
        ctx.set_source_rgba(*fog_mix([0.048, 0.024, 0.030], 10.5, fog_rgb), 0.99)
        fill_poly(ctx, q_hfasc)
        p_hr0 = np.array([X_EAVE1_OUT, y_h0, z_h_out + 0.035])
        p_hr1 = np.array([X_EAVE1_OUT, y_h1, z_h_out + 0.035])
        if mirror:
            p_hr0, p_hr1 = mirror_z0(p_hr0), mirror_z0(p_hr1)
        h_rim = cam.project_polyline(np.array([p_hr0, p_hr1]))
        if len(h_rim) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in h_rim], base_w=1.7, z_depth=10.5,
                               rgba=(0.76, 0.38, 0.20, 0.66), rng=rng, tremor=0.03)
    for ty_m in np.arange(y_h0 + 0.18, y_h1, 0.52)[::-1]:
        p_med = np.array([X_EAVE1_OUT + 0.01, ty_m, z_h_out + 0.005])
        if mirror:
            p_med = mirror_z0(p_med)
        dist_m = float(np.linalg.norm(p_med - cam.eye))
        um, vm, zm = cam.project(p_med)
        if zm > 1.2 and -30 <= um <= W + 30 and -30 <= vm <= H + 30:
            rk = max(1.4, float(cam.K[0, 0] * 0.048 / zm))
            ctx.set_source_rgba(*fog_mix([0.072, 0.040, 0.052], dist_m, fog_rgb), 0.96)
            ctx.arc(float(um), float(vm), rk, 0, math.tau)
            ctx.fill()
            ctx.set_source_rgba(*fog_mix([0.78, 0.40, 0.22], dist_m, fog_rgb), 0.58)
            ctx.set_line_width(max(0.6, rk * 0.22))
            ctx.arc(float(um), float(vm), rk * 0.80, -0.9, 0.9)
            ctx.stroke()

    # 2c. Continuous 2F Carved Wooden Kōran Balustrade Seated Cleanly on the Veranda Sill!
    z_rbot = Z_2F_FLOOR + 0.06
    z_rmid = Z_2F_FLOOR + 0.44
    z_rtop = Z_2F_FLOOR + 0.86
    # Gracefully spaced warm smoked-cedar kōran spindles running continuously from y_b0 to y_b1
    for sy in np.arange(y_b0 + 0.14, y_b1 - 0.08, 0.28)[::-1]:
        dist_s = float(np.linalg.norm(np.array([X_BALCONY_OUT, sy, z_rmid]) - cam.eye))
        p0, p1 = np.array([X_BALCONY_OUT, sy, z_rbot]), np.array([X_BALCONY_OUT, sy, z_rtop])
        if mirror:
            p0, p1 = mirror_z0(p0), mirror_z0(p1)
        seg = cam.project_polyline(np.array([p0, p1]))
        if len(seg) >= 2:
            # Soft backlit warm amber rim along back edge of foreground spindle (depth separation!)
            if not mirror:
                stroke_catmull_rom(
                    ctx, [(p[0] - 1.0, p[1]) for p in seg], base_w=1.1, z_depth=dist_s,
                    rgba=(*fog_mix([0.78, 0.42, 0.18], dist_s, fog_rgb), 0.56), rng=rng, tremor=0.02,
                )
            stroke_catmull_rom(
                ctx, [(p[0], p[1]) for p in seg], base_w=1.95, z_depth=dist_s,
                rgba=(*fog_mix([0.145, 0.068, 0.052], dist_s, fog_rgb), 0.94), rng=rng, tremor=0.03,
            )

    for zr, hw, col_r, rim_r in (
        (z_rbot + 0.05, 0.048, [0.130, 0.060, 0.048], [0.56, 0.28, 0.13]),
        (z_rmid, 0.042, [0.220, 0.088, 0.064], [0.76, 0.38, 0.18]),
        (z_rtop, 0.066, [0.640, 0.172, 0.110], [0.98, 0.54, 0.26]),
    ):
        qr = project_quad_3d(
            cam,
            np.array([[X_BALCONY_OUT, y_b0, zr - hw], [X_BALCONY_OUT, y_b1, zr - hw],
                      [X_BALCONY_OUT, y_b1, zr + hw], [X_BALCONY_OUT, y_b0, zr + hw]]),
            mirror=mirror,
        )
        if qr:
            ctx.set_source_rgba(*fog_mix(col_r, 11.0, fog_rgb), 0.98)
            fill_poly(ctx, qr)
        p_h0, p_h1 = np.array([X_BALCONY_OUT, y_b0, zr + hw]), np.array([X_BALCONY_OUT, y_b1, zr + hw])
        if mirror:
            p_h0, p_h1 = mirror_z0(p_h0), mirror_z0(p_h1)
        hseg = cam.project_polyline(np.array([p_h0, p_h1]))
        if len(hseg) >= 2:
            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in hseg], base_w=2.1 if zr > z_rmid else 1.2, z_depth=10.5,
                               rgba=(*rim_r, 0.80 if zr > z_rmid else 0.52), rng=rng, tremor=0.03)

    # Sturdy square newel posts (oyabashira) seated cleanly ON TOP of the veranda sill (z0 = Z_2F_FLOOR + 0.02, dark c_bot!)
    for py_p in RYOKAN_POST_YS[::-1]:
        if py_p < y_b0 - 0.05:
            continue
        draw_box_3d_faces(
            ctx, cam, X_BALCONY_OUT - 0.056, X_BALCONY_OUT + 0.056, py_p - 0.056, py_p + 0.056,
            Z_2F_FLOOR + 0.02, z_rtop + 0.10,
            c_front=(0.145, 0.068, 0.054), c_side=(0.205, 0.098, 0.068),
            c_top=(0.66, 0.20, 0.12), c_bot=(0.082, 0.038, 0.034),
            fog_rgb=fog_rgb, mirror=mirror,
        )
        # Warm golden backlit rim along the shoji-facing inner edge of each foreground balcony post!
        if not mirror:
            p_rim0 = np.array([X_BALCONY_OUT - 0.056, py_p - 0.056, Z_2F_FLOOR + 0.04])
            p_rim1 = np.array([X_BALCONY_OUT - 0.056, py_p - 0.056, z_rtop + 0.09])
            dist_p = float(np.linalg.norm(p_rim1 - cam.eye))
            r_seg = cam.project_polyline(np.array([p_rim0, p_rim1]))
            if len(r_seg) >= 2:
                stroke_catmull_rom(
                    ctx, [(p[0], p[1]) for p in r_seg], base_w=1.8, z_depth=dist_p,
                    rgba=(*fog_mix([0.88, 0.48, 0.20], dist_p, fog_rgb), 0.72), rng=rng, tremor=0.02,
                )

    # 3. Ground-Floor Bathhouse Split Noren Entrance Curtain ('ゆ' Motif in Crisp 2x PyCairo!)
    x_nor = X_RYOKAN_WALL + 0.14
    dist_nor = float(np.linalg.norm(np.array([x_nor, 7.03, 2.58]) - cam.eye))
    p_rod0, p_rod1 = np.array([x_nor, 5.75, 3.04]), np.array([x_nor, 8.32, 3.04])
    if mirror:
        p_rod0, p_rod1 = mirror_z0(p_rod0), mirror_z0(p_rod1)
    rseg = cam.project_polyline(np.array([p_rod0, p_rod1]))
    if len(rseg) >= 2:
        stroke_catmull_rom(ctx, [(p[0], p[1]) for p in rseg], base_w=3.8, z_depth=dist_nor,
                           rgba=(*fog_mix([0.28, 0.17, 0.09], dist_nor, fog_rgb), 0.96), rng=rng, tremor=0.03)

    noren_panels = [(5.90, 6.60), (6.68, 7.38), (7.46, 8.16)]
    for pi, (ny0, ny1) in enumerate(noren_panels):
        sway = 0.08 * math.sin(pi * 1.4 + 0.5)
        qn = project_quad_3d(
            cam,
            np.array([[x_nor, ny0, 3.02], [x_nor, ny1, 3.02],
                      [x_nor + sway, ny1, 2.12], [x_nor + sway, ny0, 2.12]]),
            mirror=mirror,
        )
        if not qn:
            continue
        g_nor = cairo.LinearGradient(qn[0][0], qn[0][1], qn[3][0], qn[3][1])
        g_nor.add_color_stop_rgba(0.0, *fog_mix([0.44, 0.115, 0.115], dist_nor, fog_rgb), 0.97)
        g_nor.add_color_stop_rgba(0.65, *fog_mix([0.58, 0.165, 0.125], dist_nor, fog_rgb), 0.96)
        g_nor.add_color_stop_rgba(1.0, *fog_mix([0.74, 0.27, 0.145], dist_nor, fog_rgb), 0.93)
        ctx.set_source(g_nor)
        fill_poly(ctx, qn)

        if not mirror:
            p_mc = np.array([x_nor + sway * 0.5, 0.5 * (ny0 + ny1), 2.58])
            p_mr = np.array([x_nor + sway * 0.5, ny1 - 0.12, 2.58])
            p_mt = np.array([x_nor + sway * 0.5, 0.5 * (ny0 + ny1), 2.86])
            um, vm, zm = cam.project(p_mc)
            ur, vr, _ = cam.project(p_mr)
            ut, vt, _ = cam.project(p_mt)
            if zm > 1.5:
                vx = np.array([float(ur - um), float(vr - vm)])
                vy = np.array([float(ut - um), float(vt - vm)])
                def npt(a: float, b: float) -> tuple[float, float]:
                    p = np.array([float(um), float(vm)]) + a * vx + b * vy
                    return (float(p[0]), float(p[1]))

                ctx.set_source_rgba(0.98, 0.88, 0.68, 0.88)
                ctx.set_line_width(max(1.5, float(np.linalg.norm(vx)) * 0.11))
                if pi == 1:
                    ctx.move_to(*npt(-0.55, 0.55))
                    ctx.curve_to(*npt(-0.62, 0.0), *npt(-0.58, -0.50), *npt(-0.32, -0.52))
                    ctx.curve_to(*npt(-0.08, -0.45), *npt(-0.05, 0.42), *npt(0.45, 0.46))
                    ctx.curve_to(*npt(0.75, 0.35), *npt(0.70, -0.35), *npt(0.28, -0.42))
                    ctx.stroke()
                    ctx.move_to(*npt(0.18, 0.68))
                    ctx.curve_to(*npt(0.20, 0.10), *npt(0.16, -0.30), *npt(-0.02, -0.72))
                    ctx.stroke()
                else:
                    for wy_s in (-0.34, 0.0, 0.34):
                        ctx.move_to(*npt(wy_s - 0.08, -0.42))
                        ctx.curve_to(*npt(wy_s + 0.16, -0.14), *npt(wy_s - 0.16, 0.16), *npt(wy_s + 0.06, 0.44))
                        ctx.stroke()

    # 4. Carved Warm Smoked-Cedar Vertical Ryokan Signboard ('湯宿') on Foremost Pillar
    if not mirror:
        x_sb = X_RYOKAN_WALL + 0.12
        qs = project_quad_3d(
            cam, np.array([[x_sb, 4.96, 1.84], [x_sb, 5.42, 1.84], [x_sb, 5.42, 2.94], [x_sb, 4.96, 2.94]]),
        )
        if qs and len(qs) >= 4:
            dist_sb = float(np.linalg.norm(np.array([x_sb, 5.19, 2.39]) - cam.eye))
            g_sb = cairo.LinearGradient(qs[3][0], qs[3][1], qs[0][0], qs[0][1])
            g_sb.add_color_stop_rgba(0.0, *fog_mix([0.165, 0.086, 0.056], dist_sb, fog_rgb), 0.98)
            g_sb.add_color_stop_rgba(0.55, *fog_mix([0.220, 0.115, 0.070], dist_sb, fog_rgb), 0.98)
            g_sb.add_color_stop_rgba(1.0, *fog_mix([0.135, 0.068, 0.046], dist_sb, fog_rgb), 0.98)
            ctx.set_source(g_sb)
            fill_poly(ctx, qs)
            ctx.set_source_rgba(*fog_mix([0.72, 0.42, 0.19], dist_sb, fog_rgb), 0.78)
            ctx.set_line_width(1.6)
            ctx.move_to(*qs[0])
            for qp in qs[1:]:
                ctx.line_to(*qp)
            ctx.close_path()
            ctx.stroke()
            # Carved warm gold-ivory brush calligraphy strokes on signboard
            def spt(u_s: float, v_s: float) -> tuple[float, float]:
                u, v, _ = cam.project(np.array([x_sb + 0.005, 4.96 + u_s * 0.46, 1.84 + v_s * 1.10]))
                return (float(u), float(v))
            ctx.set_source_rgba(*fog_mix([0.92, 0.72, 0.40], dist_sb, fog_rgb), 0.78)
            ctx.set_line_width(1.8)
            # Upper stylized '湯' + lower stylized '宿' calligraphic brushwork
            for z0_c in (0.70, 0.36):
                ctx.move_to(*spt(0.26, z0_c + 0.14))
                ctx.line_to(*spt(0.74, z0_c + 0.14))
                ctx.move_to(*spt(0.50, z0_c + 0.18))
                ctx.line_to(*spt(0.50, z0_c - 0.12))
                ctx.move_to(*spt(0.30, z0_c + 0.02))
                ctx.curve_to(*spt(0.42, z0_c + 0.05), *spt(0.58, z0_c - 0.02), *spt(0.72, z0_c + 0.02))
                ctx.move_to(*spt(0.34, z0_c - 0.10))
                ctx.line_to(*spt(0.68, z0_c - 0.08))
                ctx.stroke()

    # 5. Steaming Wooden Cedar Bath Tubs / Buckets (oke) with Folded Ivory Onsen Towel (tenugui)!
    tub_specs = [
        # (x, y, z_base, radius_m, height_m, has_towel)
        (-1.82, 4.55, Z_PROM, 0.34, 0.42, True),
        (-1.58, 4.16, Z_PROM, 0.18, 0.22, False),
        (-1.90, 9.05, Z_PROM, 0.28, 0.34, False),
    ]
    for tx, ty, tz, tr, th_m, has_towel in sorted(tub_specs, key=lambda s: -s[1]):
        p_bot, p_top = np.array([tx, ty, tz]), np.array([tx, ty, tz + th_m])
        if mirror:
            p_bot, p_top = mirror_z0(p_bot), mirror_z0(p_top)
        ub, vb, zb = cam.project(p_bot)
        ut, vt, _ = cam.project(p_top)
        if zb < 1.2 or ub < -80 or ub > W + 80:
            continue
        rx = float(cam.K[0, 0] * tr / zb)
        ry = rx * 0.34
        dist_t = float(zb)
        if not mirror:
            ctx.save()
            ctx.translate(float(ub), float(vb) + ry * 0.14)
            ctx.scale(1.0, 0.38)
            g_sh = cairo.RadialGradient(0, 0, rx * 0.2, 0, 0, rx * 1.48)
            g_sh.add_color_stop_rgba(0.0, 0.014, 0.008, 0.010, 0.58)
            g_sh.add_color_stop_rgba(1.0, 0.014, 0.008, 0.010, 0.0)
            ctx.set_source(g_sh)
            ctx.arc(0, 0, rx * 1.48, 0, math.tau)
            ctx.fill()
            ctx.restore()
        g_tub = cairo.LinearGradient(float(ub) - rx, float(vt), float(ub) + rx, float(vt))
        g_tub.add_color_stop_rgba(0.0, *fog_mix([0.14, 0.068, 0.038], dist_t, fog_rgb), 0.98)
        g_tub.add_color_stop_rgba(0.55, *fog_mix([0.38, 0.195, 0.092], dist_t, fog_rgb), 0.98)
        g_tub.add_color_stop_rgba(1.0, *fog_mix([0.18, 0.086, 0.044], dist_t, fog_rgb), 0.98)
        ctx.set_source(g_tub)
        ctx.move_to(float(ut) - rx, float(vt))
        ctx.line_to(float(ub) - rx * 0.92, float(vb))
        ctx.curve_to(float(ub) - rx * 0.5, float(vb) + ry, float(ub) + rx * 0.5, float(vb) + ry, float(ub) + rx * 0.92, float(vb))
        ctx.line_to(float(ut) + rx, float(vt))
        ctx.close_path()
        ctx.fill()
        for hf in (0.30, 0.74):
            yh = float(vt) + (float(vb) - float(vt)) * hf
            rh = rx * (1.0 - 0.06 * hf)
            ctx.set_source_rgba(*fog_mix([0.68, 0.42, 0.19], dist_t, fog_rgb), 0.94)
            ctx.set_line_width(max(1.2, rx * 0.09))
            ctx.move_to(float(ut) - rh, yh)
            ctx.curve_to(float(ut) - rh * 0.45, yh + ry * 0.78, float(ut) + rh * 0.45, yh + ry * 0.78, float(ut) + rh, yh)
            ctx.stroke()
        ctx.save()
        ctx.translate(float(ut), float(vt))
        ctx.scale(1.0, 0.34)
        ctx.set_source_rgba(*fog_mix([0.44, 0.24, 0.12], dist_t, fog_rgb), 0.96)
        ctx.arc(0, 0, rx, 0, math.tau)
        ctx.fill()
        # Warm glowing mineral hot-spring water inside the cedar tub!
        g_tw = cairo.RadialGradient(-rx * 0.18, -rx * 0.15, rx * 0.08, 0, 0, rx * 0.86)
        g_tw.add_color_stop_rgba(0.0, *fog_mix([0.68, 0.44, 0.24], dist_t, fog_rgb), 0.96)
        g_tw.add_color_stop_rgba(1.0, *fog_mix([0.16, 0.14, 0.14], dist_t, fog_rgb), 0.96)
        ctx.set_source(g_tw)
        ctx.arc(0, 0, rx * 0.84, 0, math.tau)
        ctx.fill()
        ctx.restore()
        # Neatly draped warm ivory cotton onsen towel (tenugui) hanging over foreground tub rim!
        if has_towel and not mirror:
            ctx.set_source_rgba(*fog_mix([0.86, 0.74, 0.58], dist_t, fog_rgb), 0.94)
            ctx.move_to(float(ut) - rx * 0.65, float(vt) + ry * 0.25)
            ctx.curve_to(float(ut) - rx * 0.25, float(vt) + ry * 0.02, float(ut) - rx * 0.05, float(vt) + ry * 0.12, float(ut) - rx * 0.08, float(vt) + ry * 1.45)
            ctx.line_to(float(ut) - rx * 0.56, float(vt) + ry * 1.60)
            ctx.close_path()
            ctx.fill()

    # 6. Dressed 3D Granite Stone Canal Coping Cap (starts well behind camera at y=-3.0 -> ZERO cut-off!) & Railings!
    draw_box_3d_faces(
        ctx, cam, -0.22, 0.04, -3.0, 18.2, Z_PROM - 0.06, Z_PROM + 0.04,
        c_front=(0.08, 0.058, 0.064), c_side=(0.11, 0.082, 0.086), c_top=(0.15, 0.105, 0.100),
        fog_rgb=fog_rgb, mirror=mirror, alpha=0.98,
    )

    rail_x = -0.14
    rail_posts = (1.8, 4.6, 7.4, 10.2, 13.0, 15.8, 17.6)
    for idx in range(len(rail_posts) - 1, -1, -1):
        py_r = rail_posts[idx]
        if idx < len(rail_posts) - 1:
            py_next = rail_posts[idx + 1]
            for zr, hw, cr in (
                (Z_PROM + 0.36, 0.036, [0.062, 0.030, 0.030]),
                (Z_PROM + 0.78, 0.052, [0.115, 0.048, 0.038]),
            ):
                qr = project_quad_3d(
                    cam,
                    np.array([[rail_x, py_r, zr - hw], [rail_x, py_next, zr - hw],
                              [rail_x, py_next, zr + hw], [rail_x, py_r, zr + hw]]),
                    mirror=mirror,
                )
                if qr:
                    dist_r = float(np.linalg.norm(np.array([rail_x, 0.5 * (py_r + py_next), zr]) - cam.eye))
                    ctx.set_source_rgba(*fog_mix(cr, dist_r, fog_rgb), 0.98)
                    fill_poly(ctx, qr)
                    if zr > Z_PROM + 0.6:
                        p0, p1 = np.array([rail_x, py_r, zr + hw]), np.array([rail_x, py_next, zr + hw])
                        if mirror:
                            p0, p1 = mirror_z0(p0), mirror_z0(p1)
                        seg = cam.project_polyline(np.array([p0, p1]))
                        if len(seg) >= 2:
                            stroke_catmull_rom(ctx, [(p[0], p[1]) for p in seg], base_w=1.8, z_depth=dist_r,
                                               rgba=(*fog_mix([0.56, 0.26, 0.13], dist_r, fog_rgb), 0.72), rng=rng, tremor=0.04)
        pw = 0.062
        draw_box_3d_faces(
            ctx, cam, rail_x - pw, rail_x + pw, py_r - pw, py_r + pw, Z_PROM + 0.02, Z_PROM + 0.88,
            c_front=(0.062, 0.028, 0.030), c_side=(0.095, 0.045, 0.038), c_top=(0.36, 0.19, 0.10),
            fog_rgb=fog_rgb, mirror=mirror,
        )

    # 7. Mossy Weathered Bamboo/Cedar Thermal Spring Spout (Kakehi) + Frothy Painterly Pour & Splash!
    dist_sp = float(np.linalg.norm(np.array([SPOUT_X, SPOUT_Y, SPOUT_Z]) - cam.eye))
    # Angled diagonal dark-timber support strut bracing the spout against the ishigaki wall
    p_st0, p_st1 = np.array([0.02, SPOUT_Y, SPOUT_Z - 0.34]), np.array([SPOUT_X - 0.10, SPOUT_Y, SPOUT_Z - 0.06])
    if mirror:
        p_st0, p_st1 = mirror_z0(p_st0), mirror_z0(p_st1)
    u0s, v0s, _ = cam.project(p_st0)
    u1s, v1s, _ = cam.project(p_st1)
    ctx.set_source_rgba(*fog_mix([0.072, 0.038, 0.030], dist_sp, fog_rgb), 0.96)
    ctx.set_line_width(4.2)
    ctx.move_to(float(u0s), float(v0s))
    ctx.line_to(float(u1s), float(v1s))
    ctx.stroke()

    draw_box_3d_faces(
        ctx, cam, -0.06, SPOUT_X, SPOUT_Y - 0.11, SPOUT_Y + 0.11, SPOUT_Z - 0.08, SPOUT_Z + 0.07,
        c_front=(0.13, 0.072, 0.046), c_side=(0.19, 0.102, 0.062), c_top=(0.26, 0.145, 0.084),
        fog_rgb=fog_rgb, mirror=mirror,
    )
    # Multi-layered frothy backlit hot-spring water stream (broad translucent mist veil + glistening ribbons!)
    n_arc = 14
    for spread, al_mult, col_top, col_bot in (
        (1.65, 0.34, (0.98, 0.78, 0.56), (1.00, 0.88, 0.68)),
        (0.82, 0.68, (1.00, 0.88, 0.68), (1.00, 0.95, 0.82)),
    ):
        arc_l, arc_r = [], []
        for k in range(n_arc + 1):
            t = k / n_arc
            ax = SPOUT_X + 0.02 + 0.36 * t
            az = SPOUT_Z * (1.0 - t * t)
            w_str = (0.042 + 0.065 * t) * spread
            p_l = np.array([ax, SPOUT_Y - w_str, az])
            p_r = np.array([ax, SPOUT_Y + w_str, az])
            if mirror:
                p_l, p_r = mirror_z0(p_l), mirror_z0(p_r)
            ul, vl, _ = cam.project(p_l)
            ur, vr, _ = cam.project(p_r)
            arc_l.append((float(ul), float(vl)))
            arc_r.append((float(ur), float(vr)))
        g_pour = cairo.LinearGradient(arc_l[0][0], arc_l[0][1], arc_l[-1][0], arc_l[-1][1])
        g_pour.add_color_stop_rgba(0.0, *col_top, 0.78 * al_mult)
        g_pour.add_color_stop_rgba(1.0, *col_bot, 0.86 * al_mult)
        ctx.set_source(g_pour)
        fill_poly(ctx, arc_l + arc_r[::-1])

    # Painterly broken foam splash cluster & segmented ripple shimmer arcs (NO sterile solid vector ellipses!)
    if not mirror:
        p_sp = np.array([SPOUT_X + 0.38, SPOUT_Y, Z_WATER + 0.01])
        usp, vsp, zsp = cam.project(p_sp)
        if zsp > 1.2:
            r0 = float(cam.K[0, 0] * 0.24 / zsp)
            # Soft frothy glowing splash pillow
            g_sp = cairo.RadialGradient(float(usp), float(vsp), r0 * 0.1, float(usp), float(vsp), r0 * 1.4)
            g_sp.add_color_stop_rgba(0.0, 1.0, 0.90, 0.72, 0.72)
            g_sp.add_color_stop_rgba(0.5, 0.96, 0.76, 0.52, 0.34)
            g_sp.add_color_stop_rgba(1.0, 0.96, 0.76, 0.52, 0.0)
            ctx.save()
            ctx.translate(float(usp), float(vsp))
            ctx.scale(1.0, 0.34)
            ctx.set_source(g_sp)
            ctx.arc(0, 0, r0 * 1.4, 0, math.tau)
            ctx.fill()
            # Broken calligraphic shimmer arcs radiating organically!
            for r_m, al in ((0.22, 0.62), (0.44, 0.38), (0.72, 0.22)):
                rx_r = float(cam.K[0, 0] * r_m / zsp)
                ctx.set_source_rgba(1.0, 0.84, 0.58, al)
                ctx.set_line_width(max(1.1, rx_r * 0.055))
                for a_s, a_e in ((0.2, 1.5), (2.1, 3.6), (4.4, 5.7)):
                    ctx.new_sub_path()
                    ctx.arc(0, 0, rx_r, a_s, a_e)
                    ctx.stroke()
            ctx.restore()

    # 8. Hanging Glowing Paper Chochin Lanterns along Left Ryokan 1F & 2F Eaves (far-to-near!)
    ryokan_lanterns = [c for c in CHOCHIN_SPECS if c[9].startswith("chochin_1f") or c[9].startswith("chochin_2f")]
    for (cx, cy, zh, cz, rm, hm, inten, st, kid, _lbl) in sorted(ryokan_lanterns, key=lambda c: -c[1]):
        draw_chochin_lantern_3d(ctx, cam, cx, cy, zh, cz, rm, hm, inten, st, kid, rng, fog_rgb, mirror=mirror)


# =============================================================================
# Stage 2 — Combined Fresnel Steaming Thermal Canal Water (cv2.remap + 3D Mirrors!)
# =============================================================================

def paint_fresnel_steaming_canal(
    sky_bg: np.ndarray,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    left_ishigaki_uv: np.ndarray,
    right_ishigaki_uv: np.ndarray,
    ryokan_12f_uv: np.ndarray,
    ryokan_3f_uv: np.ndarray,
    kawara_1f_uv: np.ndarray,
    kawara_2f_uv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deep mineral slate-teal & amber-lit steaming thermal river canal (z = 0):
      Combines cv2.remap capillary-rippled horizon sky/gorge reflections with exact 3D
      plane-mirrored quads (pts - 2 * dot(pts - p0, n) * n) of the entire 3-story ryokan,
      hanging chochin lanterns, arched red-lacquer taiko-bashi footbridge, curved-eave
      riverside pavilion, and crimson autumn maple cliffs weighted by:
        fresnel = fresnel_min + (fresnel_max - fresnel_min) * (1 - dist_t) ** 1.4
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    hy = float(cam.horizon_y)
    dist_t = np.clip((yy - hy) / max(1.0, H - hy), 0.0, 1.0).astype(np.float32)

    y_hor_src = np.clip(2.0 * hy - yy, 0.0, hy - 1.0).astype(np.float32)
    refl_src = cv2.remap(
        sky_bg.astype(np.float32), xx, y_hor_src,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )

    def warp_mirrored_plane(buf: np.ndarray, tex: np.ndarray, p0: np.ndarray, e1: np.ndarray, e2: np.ndarray, fog_d: float = 0.028) -> np.ndarray:
        mp0 = mirror_z0(p0)
        mp1 = mirror_z0(p0 + e1)
        mp2 = mirror_z0(p0 + e2)
        return cam.warp_plane_over(buf, tex, mp0, mp1 - mp0, mp2 - mp0, fog_rgb=fog_rgb, fog_start=8.5, fog_density=fog_d)

    refl_src = warp_mirrored_plane(
        refl_src, right_ishigaki_uv,
        p0=np.array([X_RIGHT_BANK, 1.0, Z_WATER]),
        e1=np.array([0.0, 21.0, 0.0]),
        e2=np.array([0.0, 0.0, Z_RIGHT_BANK]),
    )
    refl_src = warp_mirrored_plane(
        refl_src, ryokan_3f_uv,
        p0=np.array([X_LOFT_WALL, Y_RYOKAN_0, Z_2F_TOP]),
        e1=np.array([0.0, Y_RYOKAN_1 - Y_RYOKAN_0, 0.0]),
        e2=np.array([0.0, 0.0, Z_3F_TOP + 0.28 - Z_2F_TOP]),
    )
    refl_src = warp_mirrored_plane(
        refl_src, ryokan_12f_uv,
        p0=np.array([X_RYOKAN_WALL, Y_RYOKAN_0, Z_PROM]),
        e1=np.array([0.0, Y_RYOKAN_1 - Y_RYOKAN_0, 0.0]),
        e2=np.array([0.0, 0.0, Z_2F_TOP + 0.36 - Z_PROM]),
    )
    refl_src = warp_mirrored_plane(
        refl_src, left_ishigaki_uv,
        p0=np.array([X_LEFT_BANK, -1.5, Z_WATER]),
        e1=np.array([0.0, 21.5, 0.0]),
        e2=np.array([0.0, 0.0, Z_PROM]),
    )

    # Full 3D plane-mirrored 2x PyCairo reflections (inns, arched red taiko-bashi, pavilion, lanterns, maples!)
    s_mir, c_mir = new_cairo_2x()
    draw_distant_upstream_ryokan(c_mir, cam, rng, fog_rgb, mirror=True)
    draw_taiko_bashi_bridge(c_mir, cam, rng, fog_rgb, mirror=True)
    draw_right_bank_pavilion_and_boulders(c_mir, cam, rng, fog_rgb, mirror=True)
    draw_left_ryokan_3d_joinery(c_mir, cam, rng, fog_rgb, mirror=True)
    draw_right_gorge_maples(c_mir, cam, rng, fog_rgb, mirror=True)
    mir_1x = finish_cairo_2x(s_mir)
    refl_src = over_premul(refl_src, mir_1x)

    # Anisotropic horizontal capillary river ripples (strictly for canal water surface!)
    persp_y = np.power(dist_t + 0.024, 0.44)
    rip_swell = resize_bicubic(rng.standard_normal((H // 6, W // 48)).astype(np.float32), W, H)
    rip_mid = resize_bicubic(rng.standard_normal((H // 3, W // 22)).astype(np.float32), W, H)
    rip_fine = resize_bicubic(rng.standard_normal((H // 2, W // 11)).astype(np.float32), W, H)

    u_n = xx / W
    wave_phase = persp_y * 188.0 + rip_swell * 3.5 + np.sin(u_n * 18.0 + rip_swell * 1.4) * 1.15
    wave_train = np.sin(wave_phase) * 0.44 + np.sin(wave_phase * 2.25 + rip_mid * 2.4) * 0.28
    ripple = (0.38 * rip_swell + 0.30 * rip_mid + 0.14 * rip_fine + 0.44 * wave_train).astype(np.float32)
    wave_dy = np.gradient(ripple, axis=0).astype(np.float32)

    disp_x = (rip_mid * 0.54 + rip_fine * 0.46 + wave_train * 0.34) * (2.2 + 16.5 * np.power(dist_t, 0.72))
    disp_y = ripple * (1.6 + 14.0 * np.power(dist_t, 0.78))

    refl_soft = gaussian_blur_1d(refl_src, 1.9, axis=0)
    refl_soft = gaussian_blur_1d(refl_soft, 0.95, axis=1)
    map_x = np.clip(xx + disp_x, 0.0, W - 1.0).astype(np.float32)
    map_y = np.clip(yy + disp_y, hy, H - 1.0).astype(np.float32)
    refl_rippled = cv2.remap(
        refl_soft.astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )

    # Exact Stage 2 Fresnel formula: fresnel_min + (fresnel_max - fresnel_min) * (1 - dist_t) ** 1.4
    fresnel = 0.42 + (0.91 - 0.42) * np.power(1.0 - dist_t, 1.4)

    # Rich deep mineral slate-teal / twilight indigo thermal river body contrasting vividly against ember reflections!
    c_water_near = np.array([0.020, 0.072, 0.102], dtype=np.float32)[None, None, :]
    c_water_far = np.array([0.048, 0.125, 0.148], dtype=np.float32)[None, None, :]
    t_body = np.power(dist_t, 0.68)[..., None]
    water_body = c_water_near * t_body + c_water_far * (1.0 - t_body)

    water_col = water_body * (1.0 - fresnel[..., None]) + refl_rippled * (fresnel[..., None] * 0.98)

    # Shimmering golden-vermilion liquid reflection columns beneath every hanging chochin lantern!
    facet_up = np.clip(0.50 + 0.54 * wave_dy * 3.1 + 0.34 * wave_train, 0.0, 1.0)
    for (cx, cy, _zh, cz, _rm, _hm, inten, st, _kid, _lbl) in CHOCHIN_SPECS:
        um, vm, zm = cam.project(np.array([max(0.15, cx), cy, 0.0]))
        if zm < 1.5:
            continue
        col_w = max(14.0, float(cam.K[0, 0] * 0.44 / zm))
        col_h = max(56.0, float(cam.K[0, 0] * 3.15 / zm))
        x_d = (xx - float(um) + disp_x * 1.40) / col_w
        y_d = np.maximum(0.0, yy - float(vm)) / col_h
        below = np.clip((yy - float(vm) + 12.0) / 26.0, 0.0, 1.0)
        ribbon = np.exp(-0.5 * x_d * x_d) * np.exp(-y_d * 1.08) * below
        shimmer = np.power(facet_up, 2.7) * ribbon * min(1.0, inten * 0.58)
        tint = np.array([1.18, 0.58, 0.22], dtype=np.float32) if "red" in st else np.array([1.24, 0.80, 0.34], dtype=np.float32)
        water_col += shimmer[..., None] * tint * 0.54

    # Subtle glowing rose-amber twilight corridor reflection down the center canal channel
    sun_p = cam.K @ (cam.R @ (-SUN_DIR))
    sun_u = float(np.clip(sun_p[0] / sun_p[2] / W, 0.56, 0.72))
    ch_w = 0.042 + 0.078 * dist_t
    ch_glow = np.exp(-0.5 * ((u_n - (sun_u * 0.45 + 0.59 * 0.55)) / ch_w) ** 2) * (1.0 - dist_t * 0.40)
    water_col += (np.power(facet_up, 4.0) * ch_glow * 0.34)[..., None] * np.array([1.12, 0.68, 0.36], dtype=np.float32)

    trough = np.clip(-ripple * 0.42, 0.0, 0.64) * (0.28 + 0.72 * dist_t)
    water_col *= (1.0 - 0.30 * trough[..., None])

    horiz_blend = np.clip((yy - hy + 1.0) / 3.5, 0.0, 1.0)[..., None]
    out = sky_bg * (1.0 - horiz_blend) + water_col * horiz_blend
    return np.clip(out, 0.0, 1.50), dist_t


# =============================================================================
# Stage 3 Band 3 — Gnarled Overhanging Autumn Maples (5-Lobed Cubic Beziers!)
# =============================================================================

def draw_tapered_branch_limb(
    ctx: cairo.Context,
    pts: list[tuple[float, float]],
    w_start: float,
    w_end: float,
    bark_col: tuple[float, float, float, float],
    rim_col: tuple[float, float, float, float] | None,
    rng: np.random.Generator,
) -> None:
    if len(pts) < 2:
        return
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
            dense.append((x, y))
    dense.append(pts[-1])

    left_edge, right_edge = [], []
    n = len(dense)
    for i, (x, y) in enumerate(dense):
        frac = i / max(1, n - 1)
        prev_p = dense[max(0, i - 1)]
        next_p = dense[min(n - 1, i + 1)]
        tx, ty = next_p[0] - prev_p[0], next_p[1] - prev_p[1]
        tl = math.hypot(tx, ty) + 1e-6
        nx, ny = -ty / tl, tx / tl
        knot = 1.0 + 0.14 * math.sin(frac * 13.0 + float(rng.uniform(-0.3, 0.3)))
        half_w = 0.5 * (w_start * (1.0 - frac) + w_end * frac) * knot
        left_edge.append((x + nx * half_w, y + ny * half_w))
        right_edge.append((x - nx * half_w, y - ny * half_w))

    ctx.set_source_rgba(*bark_col)
    fill_poly(ctx, left_edge + right_edge[::-1])

    if rim_col is not None:
        edge_hi = left_edge if left_edge[len(left_edge) // 2][1] < right_edge[len(right_edge) // 2][1] else right_edge
        ctx.set_source_rgba(*rim_col)
        ctx.set_line_width(max(1.0, w_start * 0.15))
        ctx.move_to(*edge_hi[0])
        for p in edge_hi[1:]:
            ctx.line_to(*p)
        ctx.stroke()


def draw_painted_momiji_leaf(
    ctx: cairo.Context,
    cx: float,
    cy: float,
    size: float,
    angle: float,
    rgba: tuple[float, float, float, float],
    rng: np.random.Generator,
    tilt_y: float = 0.78,
    backlit_glow: tuple[float, float, float, float] | None = None,
    with_petiole: bool = True,
) -> None:
    """Seeded 5-lobed Japanese momiji maple leaf using pure cubic Beziers (ZERO stamped circles!)."""
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(angle)
    ctx.scale(1.0, tilt_y)

    if with_petiole and size > 9.5:
        ctx.set_source_rgba(rgba[0] * 0.55, rgba[1] * 0.35, rgba[2] * 0.35, rgba[3] * 0.72)
        ctx.set_line_width(max(0.65, size * 0.040))
        ctx.move_to(0.0, size * 0.42)
        ctx.curve_to(size * 0.06, size * 0.25, -size * 0.04, size * 0.16, 0.0, size * 0.06)
        ctx.stroke()

    lobe_specs = [(-1.92, 0.48), (-0.96, 0.80), (0.00, 1.00), (0.96, 0.80), (1.92, 0.48)]
    bend = float(rng.uniform(-0.12, 0.12))

    if backlit_glow is not None:
        g_l = cairo.RadialGradient(0.0, -size * 0.28, size * 0.08, 0.0, -size * 0.10, size * 1.08)
        g_l.add_color_stop_rgba(0.0, *rgba)
        g_l.add_color_stop_rgba(0.58, *rgba)
        g_l.add_color_stop_rgba(1.0, *backlit_glow)
        ctx.set_source(g_l)
    else:
        ctx.set_source_rgba(*rgba)

    ctx.move_to(0.0, size * 0.10)
    for la, lscale in lobe_specs:
        r_tip = size * lscale * float(rng.uniform(0.86, 1.14))
        a_j = la + float(rng.uniform(-0.09, 0.09)) + bend * (abs(la) * 0.35)
        w_l = 0.205 * r_tip
        dx, dy = math.sin(a_j), -math.cos(a_j)
        px, py = math.cos(a_j), math.sin(a_j)
        ctx.curve_to(
            dx * r_tip * 0.28 - px * w_l * 1.25, dy * r_tip * 0.28 - py * w_l * 1.25,
            dx * r_tip * 0.68 - px * w_l * 0.85, dy * r_tip * 0.68 - py * w_l * 0.85,
            dx * r_tip, dy * r_tip,
        )
        ctx.curve_to(
            dx * r_tip * 0.68 + px * w_l * 0.85, dy * r_tip * 0.68 + py * w_l * 0.85,
            dx * r_tip * 0.28 + px * w_l * 1.18, dy * r_tip * 0.28 + py * w_l * 1.18,
            dx * size * 0.14 + px * size * 0.08, dy * size * 0.14 + py * size * 0.08,
        )
    ctx.close_path()
    ctx.fill()

    if backlit_glow is not None and size > 13.0:
        ctx.set_source_rgba(backlit_glow[0], backlit_glow[1], backlit_glow[2], 0.30)
        ctx.set_line_width(max(0.55, size * 0.030))
        ctx.move_to(0.0, size * 0.08)
        ctx.line_to(0.0, -size * 0.64)
        ctx.stroke()

    ctx.restore()


def draw_momiji_foliage_cluster(
    ctx: cairo.Context,
    cx: float,
    cy: float,
    radius: float,
    base_leaf_sz: float,
    rng: np.random.Generator,
    density: int = 22,
    is_foreground: bool = False,
    alpha_scale: float = 1.0,
) -> None:
    """Cohesive 4-pass painterly gouache autumn momiji canopy cluster (soft depth under-wash + 3 leaf passes)."""
    # Pass 0: Soft translucent painterly gouache foliage under-wash gluing sprays into cohesive canopy volume
    ctx.save()
    ctx.translate(cx, cy)
    ctx.scale(1.18, 0.76)
    g_w = cairo.RadialGradient(0, 0, radius * 0.08, 0, 0, radius * 0.90)
    if is_foreground:
        g_w.add_color_stop_rgba(0.0, 0.16, 0.028, 0.032, 0.52 * alpha_scale)
        g_w.add_color_stop_rgba(0.65, 0.24, 0.042, 0.032, 0.24 * alpha_scale)
        g_w.add_color_stop_rgba(1.0, 0.24, 0.042, 0.032, 0.0)
    else:
        g_w.add_color_stop_rgba(0.0, 0.34, 0.062, 0.048, 0.40 * alpha_scale)
        g_w.add_color_stop_rgba(0.65, 0.52, 0.110, 0.058, 0.18 * alpha_scale)
        g_w.add_color_stop_rgba(1.0, 0.52, 0.110, 0.058, 0.0)
    ctx.set_source(g_w)
    ctx.arc(0, 0, radius * 0.90, 0, math.tau)
    ctx.fill()
    ctx.restore()

    def sample_offset(rx: float, ry: float) -> tuple[float, float]:
        r = math.sqrt(float(rng.uniform(0.02, 1.0)))
        a = float(rng.uniform(0.0, math.tau))
        return cx + math.cos(a) * r * rx, cy + math.sin(a) * r * ry

    n_sh = max(6, density // 2)
    for _ in range(n_sh):
        lx, ly = sample_offset(radius * 0.78, radius * 0.56)
        sz = base_leaf_sz * float(rng.uniform(0.80, 1.26))
        ang = float(rng.uniform(-math.pi, math.pi))
        tilt = float(rng.uniform(0.55, 0.92))
        col = (0.095, 0.022, 0.030, 0.94 * alpha_scale) if is_foreground else (0.19, 0.038, 0.044, 0.92 * alpha_scale)
        draw_painted_momiji_leaf(ctx, lx, ly, sz, ang, col, rng, tilt_y=tilt, backlit_glow=None, with_petiole=False)

    for _ in range(density):
        lx, ly = sample_offset(radius * 0.88, radius * 0.64)
        sz = base_leaf_sz * float(rng.uniform(0.65, 1.14))
        ang = float(rng.uniform(-math.pi, math.pi))
        tilt = float(rng.uniform(0.48, 0.94))
        t_c = float(rng.random())
        if is_foreground:
            col = (0.30 + 0.28 * t_c, 0.048 + 0.056 * t_c, 0.032 + 0.022 * t_c, 0.94 * alpha_scale)
            glow = (0.86, 0.28, 0.10, 0.88 * alpha_scale) if t_c > 0.40 else None
        else:
            col = (0.56 + 0.34 * t_c, 0.098 + 0.145 * t_c, 0.055 + 0.042 * t_c, 0.94 * alpha_scale)
            glow = (0.98, 0.52, 0.18, 0.92 * alpha_scale) if t_c > 0.34 else None
        draw_painted_momiji_leaf(ctx, lx, ly, sz, ang, col, rng, tilt_y=tilt, backlit_glow=glow, with_petiole=True)

    n_hi = max(5, density // 3)
    for _ in range(n_hi):
        lx, ly = sample_offset(radius * 0.82, radius * 0.56)
        lx -= radius * 0.12
        ly -= radius * 0.12
        sz = base_leaf_sz * float(rng.uniform(0.56, 0.96))
        ang = float(rng.uniform(-math.pi, math.pi))
        tilt = float(rng.uniform(0.50, 0.90))
        if is_foreground:
            col = (0.62, 0.155, 0.056, 0.92 * alpha_scale)
            glow = (0.98, 0.54, 0.19, 0.92 * alpha_scale)
        else:
            col = (0.94, 0.35, 0.12, 0.94 * alpha_scale)
            glow = (1.0, 0.82, 0.38, 0.94 * alpha_scale)
        draw_painted_momiji_leaf(ctx, lx, ly, sz, ang, col, rng, tilt_y=tilt, backlit_glow=glow, with_petiole=True)


def draw_right_gorge_maples(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
    fog_rgb: np.ndarray,
    mirror: bool = False,
) -> None:
    """
    Gnarled fiery crimson-amber Japanese autumn maples (momiji) rooted seamlessly into the
    right mossy crags & arching over the right bank — framed so BOTH paired red bridge
    lanterns AND the curved-eave riverside pavilion lantern & roof shine cleanly!
    """
    tree_specs = [
        # Tree 1 (Behind & right of the bridge/pavilion — crowns the upper-right gorge above the curved pavilion roof!)
        (
            np.array([[7.45, 17.6, 1.35], [7.15, 17.3, 3.05], [6.65, 17.0, 4.65], [5.95, 16.8, 5.45]]),
            [
                np.array([[6.92, 17.16, 3.92], [6.25, 17.35, 4.60], [5.55, 17.5, 4.82]]),
                np.array([[7.18, 17.32, 3.15], [7.95, 17.0, 4.45], [8.70, 16.8, 5.30]]),
            ],
            [
                (5.80, 16.9, 5.42, 0.88, 0.26, 16),
                (6.55, 17.0, 4.95, 1.00, 0.28, 18),
                (5.65, 17.5, 4.82, 0.76, 0.24, 14),
                (7.55, 17.1, 5.05, 1.12, 0.30, 18),
                (8.50, 16.8, 5.35, 1.18, 0.32, 18),
            ],
        ),
        # Tree 2 (Near-right cliff overhanging maple rooted at y=7.6m — clean Y-forks right on parent trunk vertices!)
        (
            np.array([[7.38, 7.6, 1.15], [7.02, 7.5, 2.38], [6.32, 7.4, 3.45], [5.48, 7.3, 3.98], [4.90, 7.2, 3.92]]),
            [
                np.array([[6.32, 7.40, 3.45], [5.75, 7.35, 4.10], [5.22, 7.30, 4.52]]),
                np.array([[6.32, 7.40, 3.45], [5.68, 7.65, 3.08], [5.12, 7.85, 2.98]]),
                np.array([[7.02, 7.50, 2.38], [7.72, 7.42, 3.38], [8.42, 7.35, 4.22]]),
            ],
            [
                (4.95, 7.2, 3.98, 0.78, 0.23, 18),
                (5.40, 7.3, 4.42, 0.90, 0.26, 20),
                (5.28, 7.8, 3.02, 0.80, 0.24, 18),
                (6.18, 7.4, 3.85, 1.04, 0.28, 22),
                (6.95, 7.5, 3.55, 1.12, 0.30, 22),
                (7.85, 7.4, 4.15, 1.24, 0.33, 24),
            ],
        ),
    ]

    for trunk_3d, subs_3d, clusts_3d in tree_specs:
        t_pts = mirror_z0(trunk_3d) if mirror else trunk_3d
        dist_t = float(np.linalg.norm(trunk_3d[1] - cam.eye))
        proj_t = cam.project_polyline(t_pts)
        bark_c = (*fog_mix([0.054, 0.022, 0.028], dist_t, fog_rgb), 0.96 if not mirror else 0.78)
        rim_c = (*fog_mix([0.78, 0.30, 0.14], dist_t, fog_rgb), 0.62 if not mirror else 0.42)

        if len(proj_t) >= 2:
            w0 = max(4.0, float(cam.K[0, 0] * 0.22 / max(2.0, proj_t[0][2])))
            w1 = max(1.8, float(cam.K[0, 0] * 0.055 / max(2.0, proj_t[-1][2])))
            draw_tapered_branch_limb(ctx, [(p[0], p[1]) for p in proj_t], w0, w1, bark_c, rim_c, rng)

        for sb in subs_3d:
            s_pts = mirror_z0(sb) if mirror else sb
            ps = cam.project_polyline(s_pts)
            if len(ps) >= 2:
                sw0 = max(2.8, float(cam.K[0, 0] * 0.11 / max(2.0, ps[0][2])))
                sw1 = max(1.2, float(cam.K[0, 0] * 0.032 / max(2.0, ps[-1][2])))
                draw_tapered_branch_limb(ctx, [(p[0], p[1]) for p in ps], sw0, sw1, bark_c, rim_c, rng)

        for cx, cy, cz, rad_m, leaf_m, dens in clusts_3d:
            pc = np.array([cx, cy, cz], dtype=np.float64)
            if mirror:
                pc = mirror_z0(pc)
            uc, vc, zc = cam.project(pc)
            if zc < 1.5 or uc < -180 or uc > W + 180 or vc < -180 or vc > H + 180:
                continue
            rad_px = max(18.0, float(cam.K[0, 0] * rad_m / zc))
            lsz_px = max(7.5, float(cam.K[0, 0] * leaf_m / zc))
            draw_momiji_foliage_cluster(
                ctx, float(uc), float(vc), rad_px, lsz_px, rng,
                density=dens if not mirror else max(8, dens // 2),
                is_foreground=False,
                alpha_scale=1.0 if not mirror else 0.80,
            )


def draw_fallen_and_drifting_maple_leaves(
    ctx: cairo.Context,
    cam: Camera3D,
    rng: np.random.Generator,
) -> None:
    """Wet fallen autumn momiji leaves along damp basalt flagstones + poetic drifting breeze leaves."""
    fallen_3d = [
        (-1.85, 2.8, Z_PROM + 0.005, 0.072), (-1.42, 3.1, Z_PROM + 0.005, 0.068),
        (-0.85, 3.7, Z_PROM + 0.005, 0.076), (-0.42, 4.2, Z_PROM + 0.005, 0.070),
        (-1.68, 5.2, Z_PROM + 0.005, 0.074), (-0.52, 6.4, Z_PROM + 0.005, 0.072),
        (-1.25, 7.6, Z_PROM + 0.005, 0.070), (-0.36, 9.2, Z_PROM + 0.005, 0.068),
        (-1.55, 10.8, Z_PROM + 0.005, 0.072), (-0.64, 12.6, Z_PROM + 0.005, 0.066),
    ]
    for fx, fy, fz, fsz in fallen_3d:
        u, v, z = cam.project(np.array([fx, fy, fz]))
        if z < 1.2 or u < -30 or u > W + 30 or v < -30 or v > H + 30:
            continue
        px_sz = float(cam.K[0, 0] * fsz / z)
        ang = float(rng.uniform(-math.pi, math.pi))
        draw_painted_momiji_leaf(
            ctx, float(u) + 0.8, float(v) + 1.4, px_sz * 1.04, ang,
            (0.015, 0.006, 0.008, 0.34), rng, tilt_y=0.30, backlit_glow=None, with_petiole=False,
        )
        col = (0.82, 0.20, 0.10, 0.94) if rng.random() > 0.45 else (0.48, 0.095, 0.060, 0.94)
        glow = (1.0, 0.56, 0.22, 0.86)
        draw_painted_momiji_leaf(ctx, float(u), float(v), px_sz, ang, col, rng, tilt_y=0.30, backlit_glow=glow, with_petiole=False)

    drifters = [
        (1140, 310, 12.5, 0.56), (1325, 255, 11.5, 0.48), (1505, 345, 14.0, 0.62),
        (1240, 465, 14.5, 0.52), (1450, 515, 15.0, 0.68), (1055, 555, 16.0, 0.50),
        (1175, 700, 17.0, 0.48), (1385, 770, 16.5, 0.58),
    ]
    for lx, ly, sz, tilt in drifters:
        ang = float(rng.uniform(-math.pi, math.pi))
        col = (0.90, 0.28, 0.11, 0.92) if rng.random() > 0.48 else (0.58, 0.12, 0.078, 0.92)
        glow = (1.0, 0.74, 0.32, 0.94)
        draw_painted_momiji_leaf(ctx, lx, ly, sz, ang, col, rng, tilt_y=tilt, backlit_glow=glow, with_petiole=True)


# =============================================================================
# Stage 3 — Delicate Wispy Translucent Painted Onsen Steam Ribbons
# =============================================================================

def paint_onsen_steam_ribbons(cam: Camera3D, rng: np.random.Generator) -> np.ndarray:
    """
    Delicate gossamer horizontal curling hot-spring vapour ribbons gliding gently over
    the thermal river (ZERO blobby/spotty cloud masks obscuring the reflected red bridge!):
      - Warm peach-amber curling plume rising cleanly above the pouring spring spout,
      - Warm golden bath steam drifting out from the 1F cedar bath exhaust louver (y = 14.2m),
      - Sleek low horizontal vapour ribbons skimming right along the steaming water surface!
    """
    qh, qw = H // 2, W // 2
    yy, xx = np.mgrid[0:qh, 0:qw].astype(np.float32)

    def steam_fbm(rgen: np.random.Generator, horiz_stretch: float = 2.6) -> np.ndarray:
        acc = np.zeros((qh, qw), dtype=np.float32)
        for gh, gw, wt in ((10, 8, 0.42), (22, 16, 0.28), (46, 34, 0.18), (92, 68, 0.12)):
            gw_s = max(4, int(round(gw / (horiz_stretch * 0.5))))
            acc += wt * resize_bicubic(rgen.standard_normal((gh, gw_s)).astype(np.float32), qw, qh)
        acc -= acc.mean()
        return acc / (float(acc.std()) + 1e-6)

    wx = steam_fbm(rng, 2.8) * 34.0
    wy = steam_fbm(rng, 3.4) * 8.0
    base_n = steam_fbm(rng, 2.6)
    wisps = remap_bilinear(base_n, np.clip(xx + wx, 0, qw - 1), np.clip(yy + wy, 0, qh - 1))
    wisps = np.clip(0.54 + 0.46 * wisps, 0.0, 1.12)

    alpha_acc = np.zeros((qh, qw), dtype=np.float32)
    col_acc = np.zeros((qh, qw, 3), dtype=np.float32)

    def add_ribbon(x3d: float, y3d: float, z3d: float, rx_m: float, ry_m: float, strength: float, rgb_c: tuple[float, float, float]) -> None:
        nonlocal alpha_acc, col_acc
        u, v, z = cam.project(np.array([x3d, y3d, z3d], dtype=np.float64))
        if z < 1.5:
            return
        cx, cy = float(u) * 0.5, float(v) * 0.5
        rx = max(26.0, float(cam.K[0, 0] * rx_m / z) * 0.5)
        ry = max(9.0, float(cam.K[0, 0] * ry_m / z) * 0.5)
        dx = (xx - cx) / rx
        dy = (yy - cy - np.sin(dx * 2.2 + y3d) * (ry * 0.28)) / ry
        env = np.exp(-0.5 * (dx * dx + dy * dy)) * wisps * np.float32(strength)
        a_p = np.clip(env, 0.0, 0.24)
        col_acc += a_p[..., None] * np.array(rgb_c, dtype=np.float32)[None, None, :]
        alpha_acc = np.clip(alpha_acc + a_p * (1.0 - alpha_acc), 0.0, 0.30)

    # 1. Delicate curling backlit vapour rising from the pouring thermal spring spout splash!
    add_ribbon(SPOUT_X + 0.40, SPOUT_Y, 0.36, 1.35, 0.48, 0.34, (1.02, 0.82, 0.60))
    add_ribbon(SPOUT_X + 0.85, SPOUT_Y + 0.35, 0.86, 1.65, 0.60, 0.22, (0.98, 0.74, 0.56))

    # 2. Warm golden-amber bath steam drifting out from the 1F cedar bath exhaust louver (y = 14.2m)!
    add_ribbon(X_RYOKAN_WALL + 0.95, 14.20, 2.38, 1.55, 0.66, 0.32, (1.04, 0.76, 0.46))
    add_ribbon(X_RYOKAN_WALL + 2.15, 14.60, 2.90, 2.15, 0.82, 0.20, (0.98, 0.68, 0.46))

    # 3. Sleek gossamer horizontal mist ribbons skimming low across the steaming canal (preserves bridge mirror!)
    add_ribbon(1.85, 7.80, 0.16, 2.85, 0.26, 0.16, (0.92, 0.76, 0.68))
    add_ribbon(4.45, 11.40, 0.18, 2.90, 0.26, 0.15, (0.92, 0.74, 0.66))
    add_ribbon(2.90, 15.80, 0.26, 3.50, 0.30, 0.16, (0.96, 0.74, 0.62))
    add_ribbon(3.10, 20.40, 0.36, 3.65, 0.38, 0.18, (0.96, 0.70, 0.60))
    add_ribbon(3.10, 26.50, 0.60, 3.90, 0.60, 0.22, (0.94, 0.66, 0.58))
    add_ribbon(5.15, 13.70, 0.34, 2.05, 0.45, 0.18, (0.98, 0.76, 0.54))

    alpha_s = gaussian_blur_2d(alpha_acc, 3.0)
    rgb_s = gaussian_blur_2d(col_acc, 3.0)
    out = np.zeros((H, W, 4), dtype=np.float32)
    for c in range(3):
        out[:, :, c] = np.clip(resize_bicubic(rgb_s[:, :, c], W, H), 0.0, 1.15)
    out[:, :, 3] = np.clip(resize_bicubic(alpha_s, W, H), 0.0, 0.30)
    return out


def paint_twilight_haze_beams(cam: Camera3D, rng: np.random.Generator, intensity: float = 0.20) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sun_p = cam.K @ (cam.R @ (-SUN_DIR))
    su, sv = float(sun_p[0] / sun_p[2]), float(sun_p[1] / sun_p[2])
    dx, dy = xx - su, yy - sv
    dist = np.hypot(dx, dy) + 1e-4
    ang = np.arctan2(dy, dx)

    acc = np.zeros((H, W), dtype=np.float32)
    for a0, aw, st in ((2.42, 0.14, 0.34), (2.74, 0.12, 0.46), (3.02, 0.15, 0.38), (-2.85, 0.13, 0.32)):
        da = np.arctan2(np.sin(ang - a0), np.cos(ang - a0))
        acc += np.exp(-0.5 * (da / aw) ** 2) * np.exp(-dist / (W * 0.56)) * np.clip(dist / 85.0, 0.0, 1.0) * st

    turb = 0.84 + 0.16 * resize_bicubic(fbm_gouache(H // 4, W // 4, rng), W, H)
    acc = np.clip(acc * turb * intensity, 0.0, 0.52)
    return acc[..., None] * np.array([1.04, 0.56, 0.30], dtype=np.float32)[None, None, :]


# =============================================================================
# Stage 4 — Foreground Circular Disc Bokeh & 35mm Analog Optics
# =============================================================================

def make_disc_kernel(radius: float = 7.8) -> np.ndarray:
    rad_i = int(math.ceil(radius + 1.5))
    yy, xx = np.mgrid[-rad_i : rad_i + 1, -rad_i : rad_i + 1].astype(np.float32)
    r = np.hypot(xx, yy)
    disc = np.clip((radius - r + 0.65) / 1.15, 0.0, 1.0)
    rim = np.exp(-((r - radius * 0.86) / (radius * 0.16)) ** 2) * 0.25 * disc
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


def paint_foreground_bokeh_maple_bough(rng: np.random.Generator) -> np.ndarray:
    """Near-camera upper-right backlit autumn maple bough convolved with circular aperture disc (r = 7.8)."""
    surf, ctx = new_cairo_2x()
    bark_fg = (0.036, 0.014, 0.020, 0.99)
    rim_fg = (0.76, 0.28, 0.12, 0.56)

    main_bough = [(W + 30, 30), (W - 125, 72), (W - 285, 96), (W - 435, 68), (W - 535, 84)]
    twig_1 = [(W - 180, 82), (W - 255, 168), (W - 322, 215)]
    twig_2 = [(W - 96, 66), (W - 142, 188), (W - 178, 270)]
    twig_3 = [(W - 320, 90), (W - 395, 26), (W - 468, 6)]
    twig_4 = [(W - 60, 56), (W - 32, 184), (W - 64, 294)]

    draw_tapered_branch_limb(ctx, main_bough, 22.0, 5.0, bark_fg, rim_fg, rng)
    draw_tapered_branch_limb(ctx, twig_1, 11.0, 2.7, bark_fg, rim_fg, rng)
    draw_tapered_branch_limb(ctx, twig_2, 10.5, 2.5, bark_fg, rim_fg, rng)
    draw_tapered_branch_limb(ctx, twig_3, 9.8, 2.4, bark_fg, rim_fg, rng)
    draw_tapered_branch_limb(ctx, twig_4, 11.5, 2.9, bark_fg, rim_fg, rng)

    fg_clusters = [
        (W - 525, 88, 58, 40, 14),
        (W - 438, 66, 68, 46, 16),
        (W - 415, 16, 60, 42, 14),
        (W - 320, 208, 56, 40, 14),
        (W - 268, 118, 74, 48, 17),
        (W - 176, 262, 58, 42, 14),
        (W - 152, 158, 78, 52, 18),
        (W - 80, 78, 82, 54, 18),
        (W - 56, 268, 62, 44, 15),
        (W - 26, 158, 72, 48, 16),
    ]
    for cx, cy, rad, lsz, dens in fg_clusters:
        draw_momiji_foliage_cluster(ctx, cx, cy, rad, lsz, rng, density=dens, is_foreground=True)

    fg_1x = finish_cairo_2x(surf)
    blurred = convolve_fft_2d_bbox(fg_1x, make_disc_kernel(radius=7.8))

    out = blurred.copy()
    M_r = np.array([[1.0, 0.0, 1.05], [0.0, 1.0, 0.0]], dtype=np.float32)
    M_b = np.array([[1.0, 0.0, -1.05], [0.0, 1.0, 0.0]], dtype=np.float32)
    out[:, :, 0] = cv2.warpAffine(blurred[:, :, 0], M_r, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    out[:, :, 2] = cv2.warpAffine(blurred[:, :, 2], M_b, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return np.clip(out, 0.0, 1.0)


def add_lantern_specular_glints(img: np.ndarray, cam: Camera3D) -> np.ndarray:
    out = img.copy()
    for (cx, cy, _zh, cz, _rm, _hm, inten, st, _kid, _lbl) in CHOCHIN_SPECS:
        u, v, z = cam.project(np.array([cx, cy, cz], dtype=np.float64))
        if z < 1.2 or u < 0 or u >= W or v < 0 or v >= H:
            continue
        r = max(14, int(round(cam.K[0, 0] * 0.34 / z)))
        x0, x1 = max(0, int(u) - r), min(W, int(u) + r + 1)
        y0, y1 = max(0, int(v) - r), min(H, int(v) + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy = (np.arange(y0, y1, dtype=np.float32)[:, None] - float(v)) / max(1.0, r * 0.36)
        xx = (np.arange(x0, x1, dtype=np.float32)[None, :] - float(u)) / max(1.0, r * 0.36)
        core = np.exp(-0.82 * (xx * xx + yy * yy))
        wing_h = np.exp(-0.20 * (xx * xx) - 4.4 * (yy * yy)) * 0.36
        wing_v = np.exp(-4.4 * (xx * xx) - 0.45 * (yy * yy)) * 0.22
        glint = (core * 0.22 + wing_h + wing_v) * min(1.0, inten * 0.46)
        tint = np.array([1.0, 0.62, 0.26], dtype=np.float32) if "red" in st else np.array([1.0, 0.78, 0.36], dtype=np.float32)
        out[y0:y1, x0:x1] += glint[..., None] * tint * 0.22
    return out


def apply_35mm_finish(rgb: np.ndarray, cam: Camera3D, rng: np.random.Generator) -> np.ndarray:
    img = add_lantern_specular_glints(rgb.astype(np.float32), cam)

    lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    hl_mask = np.clip((lum - 0.72) / 0.28, 0.0, 1.65)
    hl_src = img * hl_mask[..., None]
    tint = np.array([1.16, 0.92, 0.74], dtype=np.float32)
    hal = (
        0.27 * gaussian_blur_2d(hl_src, 6.5)
        + 0.20 * gaussian_blur_2d(hl_src, 24.0)
        + 0.13 * gaussian_blur_2d(hl_src, 72.0)
    ) * tint[None, None, :]
    img = 1.0 - (1.0 - np.clip(img, 0.0, 1.0)) * (1.0 - np.clip(hal, 0.0, 0.86))

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (xx - W * 0.5) / (W * 0.5)
    ny = (yy - H * 0.5) / (H * 0.5)
    r2 = nx * nx + ny * ny
    ca_scale = 0.00075
    img[:, :, 0] = remap_bilinear(img[:, :, 0], xx + nx * r2 * (W * ca_scale), yy + ny * r2 * (H * ca_scale))
    img[:, :, 2] = remap_bilinear(img[:, :, 2], xx - nx * r2 * (W * ca_scale), yy - ny * r2 * (H * ca_scale))
    img *= np.clip(1.0 - 0.19 * (r2 ** 1.15), 0.73, 1.0)[..., None]

    img = 0.011 + img * 0.989
    knee, ceiling = 0.80, 0.965
    span = ceiling - knee
    img = np.where(img > knee, knee + span * np.tanh((img - knee) / span), img)

    lum2 = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    env = np.clip(1.12 - 0.95 * np.power(np.clip(lum2, 0.0, 1.0), 1.55), 0.12, 1.12)
    grain = rng.normal(0.0, 0.0098, size=(H, W, 3)).astype(np.float32)
    mono_g = rng.normal(0.0, 0.0080, size=(H, W, 1)).astype(np.float32)
    img = np.clip(img + (0.65 * mono_g + 0.35 * grain) * env[..., None], 0.0, 1.0)

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
    out *= (1.0 - 0.007 * (0.5 - 0.5 * np.cos(np.arange(H, dtype=np.float32) * math.pi)))[:, None, None]
    return np.clip(out, 0.0, 1.0)


# =============================================================================
# Stage 5 — Exact Shader Calibration Coordinate Printout & Scene Assembly
# =============================================================================

def print_shader_calibration_uvs(cam: Camera3D) -> dict[str, tuple[float, float]]:
    uvs: dict[str, tuple[float, float]] = {}
    print("[ember-02] === Exact Camera3D Projected Shader Calibration (u, v) Coordinates ===")

    print("  -- Thermal Canal Water Boundaries (z = 0) --")
    for name, pt in (
        ("canal_L_y3",       (X_LEFT_BANK, 3.0, 0.0)),
        ("canal_L_y6",       (X_LEFT_BANK, 6.0, 0.0)),
        ("canal_L_y12",      (X_LEFT_BANK, 12.0, 0.0)),
        ("canal_L_bridge",   (X_LEFT_BANK, Y_BRIDGE_C - BRIDGE_HALF_W, 0.0)),
        ("canal_L_y34",      (X_LEFT_BANK, 34.0, 0.0)),
        ("canal_R_y5",       (X_RIGHT_BANK, 5.0, 0.0)),
        ("canal_R_y12",      (X_RIGHT_BANK, 12.0, 0.0)),
        ("canal_R_bridge",   (X_RIGHT_BANK, Y_BRIDGE_C - BRIDGE_HALF_W, 0.0)),
        ("canal_R_y34",      (X_RIGHT_BANK, 34.0, 0.0)),
        ("bridge_arch_bot_L",(0.6, Y_BRIDGE_C - BRIDGE_HALF_W, float(bridge_deck_z(0.6)) - 0.45)),
        ("bridge_arch_bot_C",(3.1, Y_BRIDGE_C - BRIDGE_HALF_W, float(bridge_deck_z(3.1)) - 0.36)),
        ("bridge_arch_bot_R",(5.6, Y_BRIDGE_C - BRIDGE_HALF_W, float(bridge_deck_z(5.6)) - 0.45)),
        ("bridge_rail_top_C",(3.1, Y_BRIDGE_C - BRIDGE_HALF_W, float(bridge_deck_z(3.1)) + 1.05)),
    ):
        u, v, z = cam.project(np.array(pt, dtype=np.float64))
        uvs[name] = (float(u / W), float(v / H))
        print(f"  {name:22s}: uv = ({float(u / W):+.4f}, {float(v / H):+.4f})  px = ({float(u):7.1f}, {float(v):6.1f})  z = {float(z):.2f}m")

    print("  -- Hanging Chochin Lantern Centers & Water Reflection Columns --")
    for (cx, cy, _zh, cz, _rm, _hm, inten, _st, _kid, lbl) in CHOCHIN_SPECS:
        u, v, z = cam.project(np.array([cx, cy, cz], dtype=np.float64))
        ur, vr, _ = cam.project(np.array([max(0.25, cx), cy, 0.0], dtype=np.float64))
        uvs[lbl] = (float(u / W), float(v / H))
        uvs[f"refl_{lbl}"] = (float(ur / W), float(vr / H))
        print(f"  {lbl:22s}: uv = ({float(u / W):.4f}, {float(v / H):.4f})  refl_top = ({float(ur / W):.4f}, {float(vr / H):.4f})  inten = {inten:.2f}")

    print("  -- Key Shoji Bays & Bath Steam Vents --")
    for (y0, y1, z0, z1, inten, lbl) in SHOJI_BAYS:
        u, v, z = cam.project(np.array([X_RYOKAN_WALL, 0.5 * (y0 + y1), 0.5 * (z0 + z1)], dtype=np.float64))
        uvs[lbl] = (float(u / W), float(v / H))
        print(f"  {lbl:22s}: uv = ({float(u / W):.4f}, {float(v / H):.4f})  z = {float(z):.2f}m")
    for name, pt in (
        ("steam_spout_splash", (SPOUT_X + 0.38, SPOUT_Y, 0.15)),
        ("steam_louver_vent",  (X_RYOKAN_WALL + 0.45, 14.20, 2.38)),
        ("steam_canal_mid",    (3.10, 13.50, 0.30)),
    ):
        u, v, z = cam.project(np.array(pt, dtype=np.float64))
        uvs[name] = (float(u / W), float(v / H))
        print(f"  {name:22s}: uv = ({float(u / W):.4f}, {float(v / H):.4f})  z = {float(z):.2f}m")

    return uvs


def render_scene(
    seed: int = 199809,
    audit_dir: pathlib.Path | None = None,
    webp_targets: list[pathlib.Path] | None = None,
) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # Off-axis vanishing point ~58.9% width, horizon ~50.0% height — shows all 3 ryokan stories + full bridge mirror!
    cam = Camera3D(W, H, eye=(2.35, -4.90, 2.50), target=(-0.25, 20.00, 2.52), fov_x_deg=61.0)
    print(f"[ember-02] Camera VP: ({cam.vp_x:.1f}, {cam.vp_y:.1f}) -> {cam.vp_x / W * 100:.1f}% width, horizon {cam.horizon_y / H * 100:.1f}% height")
    uvs = print_shader_calibration_uvs(cam)

    # Stage 1 & Band 0: Sweeping twilight sky, raymarched dusk cloud ribbons & autumn mountain gorge ridges
    sky_bg, fog_rgb, _sky_meta = paint_twilight_gorge_sky(cam, rng)

    # Pre-render Planar Architectural UV Textures (shared by real geometry AND exact 3D canal mirrors!)
    prom_uv = paint_voronoi_ishigaki_uv(rng, (X_RYOKAN_WALL, X_LEFT_BANK), (Y_RYOKAN_0, 20.2), "promenade", tw=896, th=1152)
    left_ishi_uv = paint_voronoi_ishigaki_uv(rng, (-1.5, 20.0), (Z_WATER, Z_PROM), "left_ishigaki", tw=1408, th=384)
    right_ishi_uv = paint_voronoi_ishigaki_uv(rng, (1.0, 22.0), (Z_WATER, Z_RIGHT_BANK), "right_ishigaki", tw=1408, th=384)
    ryokan_12f_uv = paint_ryokan_facade_uv(rng, (Y_RYOKAN_0, Y_RYOKAN_1), (Z_PROM, Z_2F_TOP + 0.36), "lower_2f", tw=2800, th=1180)
    ryokan_3f_uv = paint_ryokan_facade_uv(rng, (Y_RYOKAN_0, Y_RYOKAN_1), (Z_2F_TOP, Z_3F_TOP + 0.28), "loft_3f", tw=1600, th=480)
    kawara_1f_uv = paint_kawara_roof_uv(rng, Y_RYOKAN_1 - Y_RYOKAN_0 + 0.35, 0.72, tw=1024, th=224)
    kawara_2f_uv = paint_kawara_roof_uv(rng, Y_RYOKAN_1 - Y_RYOKAN_0 + 0.45, 1.75, tw=1152, th=352)
    soffit_2f_uv = paint_eave_soffit_uv(rng, (Y_RYOKAN_0, Y_RYOKAN_1 + 0.45), Z_2F_TOP - 0.04, X_RYOKAN_WALL, X_EAVE2_OUT)
    soffit_3f_uv = paint_eave_soffit_uv(rng, (Y_RYOKAN_0, Y_RYOKAN_1 + 0.35), Z_3F_TOP - 0.08, X_LOFT_WALL, X_EAVE3_OUT)

    # Stage 2: Combined Fresnel Steaming Thermal Canal Water (cv2.remap horizon + 3D plane-mirrored quads!)
    canvas, _dist_t = paint_fresnel_steaming_canal(
        sky_bg, cam, rng, fog_rgb,
        left_ishi_uv, right_ishi_uv, ryokan_12f_uv, ryokan_3f_uv, kawara_1f_uv, kawara_2f_uv,
    )

    g_mottle = fbm_gouache(H, W, rng)
    g_dabs = flat_brush_dabs(H, W, rng, count=92)
    g_tooth = paper_tooth(H, W, rng)

    # Stage 3 Band 0b: Detailed Distant Warm Lit Upstream Ryokan Village beyond the arched bridge
    s_b0, c_b0 = new_cairo_2x()
    draw_distant_upstream_ryokan(c_b0, cam, rng, fog_rgb, mirror=False)
    b0_1x = finish_cairo_2x(s_b0)
    b0_a = b0_1x[:, :, 3:4]
    b0_1x[:, :, :3] = np.clip(b0_1x[:, :, :3] + b0_a * (g_mottle * 0.012 + g_dabs * 0.009)[..., None], 0.0, 1.40)
    canvas = over_premul(canvas, b0_1x)

    # Stage 1 & 2 Planar Homography Architectural Surfaces (Back-to-Front!)
    canvas = cam.warp_plane_over(
        canvas, right_ishi_uv,
        p0=np.array([X_RIGHT_BANK, 1.0, Z_WATER]),
        e1=np.array([0.0, 21.0, 0.0]),
        e2=np.array([0.0, 0.0, Z_RIGHT_BANK]),
        fog_rgb=fog_rgb, fog_start=8.5, fog_density=0.026,
    )
    # 3rd-story attic clerestory wall & crown overhanging timber soffit (sealed cleanly up to Z_3F_TOP + 0.28!)
    canvas = cam.warp_plane_over(
        canvas, ryokan_3f_uv,
        p0=np.array([X_LOFT_WALL, Y_RYOKAN_0, Z_2F_TOP]),
        e1=np.array([0.0, Y_RYOKAN_1 - Y_RYOKAN_0, 0.0]),
        e2=np.array([0.0, 0.0, Z_3F_TOP + 0.28 - Z_2F_TOP]),
        fog_rgb=fog_rgb, fog_start=8.5, fog_density=0.026,
    )
    canvas = cam.warp_plane_over(
        canvas, soffit_3f_uv,
        p0=np.array([X_LOFT_WALL, Y_RYOKAN_0, Z_3F_TOP + 0.24]),
        e1=np.array([0.0, Y_RYOKAN_1 - Y_RYOKAN_0 + 0.35, 0.0]),
        e2=np.array([X_EAVE3_OUT - X_LOFT_WALL, 0.0, -0.34]),
        fog_rgb=fog_rgb, fog_start=8.5, fog_density=0.026,
    )
    # 1F & 2F ryokan timber & plaster facade (extends solidly to Z_2F_TOP + 0.36 to seal tightly with soffit_2f_uv!)
    canvas = cam.warp_plane_over(
        canvas, ryokan_12f_uv,
        p0=np.array([X_RYOKAN_WALL, Y_RYOKAN_0, Z_PROM]),
        e1=np.array([0.0, Y_RYOKAN_1 - Y_RYOKAN_0, 0.0]),
        e2=np.array([0.0, 0.0, Z_2F_TOP + 0.36 - Z_PROM]),
        fog_rgb=fog_rgb, fog_start=8.5, fog_density=0.025,
    )
    # 2F grand overhanging cedar eave soffit (front-facing from street level; backfacing kawara top slope omitted!)
    canvas = cam.warp_plane_over(
        canvas, soffit_2f_uv,
        p0=np.array([X_RYOKAN_WALL, Y_RYOKAN_0, Z_2F_TOP + 0.32]),
        e1=np.array([0.0, Y_RYOKAN_1 - Y_RYOKAN_0 + 0.45, 0.0]),
        e2=np.array([X_EAVE2_OUT - X_RYOKAN_WALL, 0.0, -0.36]),
        fog_rgb=fog_rgb, fog_start=8.5, fog_density=0.026,
    )
    # Left coursed ishigaki stone embankment & damp basalt promenade flagstones
    canvas = cam.warp_plane_over(
        canvas, left_ishi_uv,
        p0=np.array([X_LEFT_BANK, -1.5, Z_WATER]),
        e1=np.array([0.0, 21.5, 0.0]),
        e2=np.array([0.0, 0.0, Z_PROM]),
        fog_rgb=fog_rgb, fog_start=8.5, fog_density=0.024,
    )
    canvas = cam.warp_plane_over(
        canvas, prom_uv,
        p0=np.array([X_RYOKAN_WALL, Y_RYOKAN_0, Z_PROM]),
        e1=np.array([X_LEFT_BANK - X_RYOKAN_WALL, 0.0, 0.0]),
        e2=np.array([0.0, 20.2 - Y_RYOKAN_0, 0.0]),
        fog_rgb=fog_rgb, fog_start=8.5, fog_density=0.024,
    )

    # Stage 3 Band 1: Midground Arched Taiko-Bashi Footbridge + Curved-Eave Riverside Gazebo & Sculpted Gorge Rocks
    s_b1, c_b1 = new_cairo_2x()
    draw_taiko_bashi_bridge(c_b1, cam, rng, fog_rgb, mirror=False)
    draw_right_bank_pavilion_and_boulders(c_b1, cam, rng, fog_rgb, mirror=False)
    b1_1x = finish_cairo_2x(s_b1)
    b1_a = b1_1x[:, :, 3:4]
    b1_1x[:, :, :3] = np.clip(b1_1x[:, :, :3] + b1_a * (g_mottle * 0.015 + g_dabs * 0.011 + g_tooth * 0.004)[..., None], 0.0, 1.42)
    canvas = over_premul(canvas, b1_1x)

    # Subtle volumetric twilight haze rays aligned with SUN_DIR
    beams = paint_twilight_haze_beams(cam, rng, intensity=0.20)
    canvas = 1.0 - (1.0 - np.clip(canvas, 0.0, 1.0)) * (1.0 - np.clip(beams, 0.0, 0.78))

    # Stage 3 Band 2: Left 3-Story Ryokan Joinery, Chochin Lanterns, 'ゆ' Noren, Cedar Spout, Tubs & Promenade Rail
    s_b2, c_b2 = new_cairo_2x()
    draw_left_ryokan_3d_joinery(c_b2, cam, rng, fog_rgb, mirror=False)
    b2_1x = finish_cairo_2x(s_b2)
    b2_a = b2_1x[:, :, 3:4]
    b2_1x[:, :, :3] = np.clip(b2_1x[:, :, :3] + b2_a * (g_mottle * 0.016 + g_dabs * 0.011 + g_tooth * 0.005)[..., None], 0.0, 1.44)
    canvas = over_premul(canvas, b2_1x)

    # Delicate Wispy Translucent Painted Onsen Steam Ribbons
    steam_rgba = paint_onsen_steam_ribbons(cam, rng)
    canvas = over_premul(canvas, steam_rgba)

    # Stage 3 Band 3: Gnarled Overhanging Right-Cliff Autumn Maples (5-lobed Beziers) + Fallen & Drifting Leaves
    s_b3, c_b3 = new_cairo_2x()
    draw_right_gorge_maples(c_b3, cam, rng, fog_rgb, mirror=False)
    draw_fallen_and_drifting_maple_leaves(c_b3, cam, rng)
    b3_1x = finish_cairo_2x(s_b3)
    b3_a = b3_1x[:, :, 3:4]
    b3_1x[:, :, :3] = np.clip(b3_1x[:, :, :3] + b3_a * (g_mottle * 0.016 + g_dabs * 0.011 + g_tooth * 0.005)[..., None], 0.0, 1.42)
    canvas = over_premul(canvas, b3_1x)

    canvas = apply_warm_shadow_bounce(canvas, strength=0.013)

    # Stage 4: Foreground Circular Disc Bokeh Maple Bough (r = 7.8 + R/B warpAffine shift) & 35mm Finish
    fg_bokeh = paint_foreground_bokeh_maple_bough(rng)
    canvas = over_premul(canvas, fg_bokeh)

    final_rgb = apply_35mm_finish(canvas, cam, rng)
    elapsed = time.perf_counter() - t0
    print(f"[ember-02] Rendered {W}x{H} in {elapsed:.2f}s")

    u8 = np.round(np.clip(final_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    full_im = Image.fromarray(u8, mode="RGB")

    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        full_im.save(audit_dir / "w02_full.png", optimize=True)
        u_br, v_br = uvs["bridge_arch_bot_C"]
        u_ch, v_ch = uvs["chochin_1f_entry"]
        crops = {
            "crop1_bridge_canal_refl.png": (int(u_br * W) - 310, int(v_br * H) - 115),
            "crop2_ryokan_balcony_chochin.png": (int(u_ch * W) - 300, int(v_ch * H) - 220),
            "crop3_promenade_maple_bokeh.png": (W - 600, int(H * 0.14)),
            "crop4_fg_promenade_spout.png": (int(W * 0.14), H - 450),
        }
        for cname, (cx0, cy0) in crops.items():
            x0 = max(0, min(W - 600, cx0))
            y0 = max(0, min(H - 450, cy0))
            full_im.crop((x0, y0, x0 + 600, y0 + 450)).save(audit_dir / cname)
        print(f"[ember-02] Exported full PNG + 4 1:1 macro crops to {audit_dir}")

    targets = webp_targets if webp_targets is not None else [DEFAULT_THEME_WEBP, DEFAULT_DOCS_WEBP]
    if targets:
        first = targets[0]
        first.parent.mkdir(parents=True, exist_ok=True)
        full_im.save(first, format="WEBP", quality=94, method=6)
        print(f"[ember-02] Saved {W}x{H} WebP (q=94): {first} ({first.stat().st_size / 1024:.1f} KiB)")
        for extra in targets[1:]:
            extra.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(first, extra)
            print(f"[ember-02] Copied {W}x{H} WebP: {extra} ({extra.stat().st_size / 1024:.1f} KiB)")

    return final_rgb, {"elapsed": elapsed, "vp_ratio": cam.vp_x / W, "uvs": uvs}


def main() -> None:
    ap = argparse.ArgumentParser(description="Render Ember 02 (Twilight Onsen Bathhouse Canal) 2048x1152 wallpaper")
    ap.add_argument("--seed", type=int, default=199809)
    ap.add_argument("--audit-dir", type=pathlib.Path, default=None)
    ap.add_argument("--no-webp", action="store_true")
    args = ap.parse_args()
    targets = [] if args.no_webp else [DEFAULT_THEME_WEBP, DEFAULT_DOCS_WEBP]
    render_scene(seed=args.seed, audit_dir=args.audit_dir, webp_targets=targets)


if __name__ == "__main__":
    main()

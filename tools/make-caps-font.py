#!/usr/bin/env python3
# =============================================================
# Builds assets/fonts/MiraCaps.ttf: the two end caps of the
# mira-islands bar capsules, a pill end slanted like "/"
# (Nerd Font has slanted or rounded caps, not both).
#   U+E100  left cap  (left edge leans "/")
#   U+E101  right cap (right edge leans "/")
#
# Each cap starts as an upright rounded end (two quarter circles of
# radius --radius; at the maximum, half the height, a full half
# circle like a pill) and is then sheared by --slant. Shearing keeps
# horizontal lines horizontal, so the curve runs flat into the top and
# bottom edge of the capsule body, and the result is one smooth curve
# with no corner anywhere. The right cap is the left cap turned 180
# degrees. The advance ends exactly where the flat top (left cap) or
# flat bottom (right cap) begins, so the capsule body joins there.
#
# The glyphs span the full em (descent to ascent), so at N px rendered size
# they are exactly N px tall. Note: in Polybar (96 DPI), fontconfig converts
# `pixelsize=N` via 75 DPI to points first (`N * 96/75` px), so use
# `size=<N * 72 / 96>` (e.g. `size=22.5` for a 30px bar, offset `;4`) with
# `hinting=false:antialias=true:rgba=none` so the cap fits flush without
# top/bottom clipping. Each one reaches OVER past its advance, under the
# capsule body, so no hairline shows at the join.
# --check prints the tightest curve radius at the bar's pixel size.
# Requires python3-fonttools.  Copyright (C) 2026 Misaalya
# SPDX-License-Identifier: GPL-3.0-only
# =============================================================
import argparse
import math
import os

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

UPM, ASC, DESC = 1000, 800, -200
HEIGHT = ASC - DESC
OVER = 40          # drawn past the advance, under the capsule body
ARC_STEP = 30      # degrees per quadratic piece; keeps the arcs truly round


class Cap:
    def __init__(self, radius, slant):
        if not 0 < radius <= HEIGHT / 2:
            raise SystemExit(f"--radius must be in 1..{HEIGHT // 2}")
        self.rad = radius
        self.run = slant
        self.adv = round(radius + slant)   # where the flat top begins

    def shear(self, x, y):
        return x + (y - DESC) * self.run / HEIGHT, y

    def arc(self, cx, cy, start, end):
        """Quadratic pieces (control, end point) of a circular arc, in degrees."""
        k = max(1, math.ceil(abs(end - start) / ARC_STEP))
        step = math.radians((end - start) / k)
        a = math.radians(start)
        r_ctrl = self.rad / math.cos(step / 2)
        out = []
        for _ in range(k):
            mid = a + step / 2
            a += step
            out.append(((cx + r_ctrl * math.cos(mid), cy + r_ctrl * math.sin(mid)),
                        (cx + self.rad * math.cos(a), cy + self.rad * math.sin(a))))
        return out

    def segments(self, flip=False):
        """Clockwise outline (y up) as a start point and a list of
        ("line", p) / ("qcurve", ctrl, p). flip turns it 180 degrees."""
        R, right = self.rad, self.adv + OVER

        def P(x, y, slanted=True):
            if slanted:
                x, y = self.shear(x, y)
            if flip:
                x, y = self.adv - x, (ASC + DESC) - y
            return (round(x), round(y))

        segs = []
        if ASC - R > DESC + R:
            segs.append(("line", P(0, ASC - R)))
        for c, e in self.arc(R, ASC - R, 180, 90):          # upper half of the end
            segs.append(("qcurve", P(*c), P(*e)))
        segs.append(("line", P(right, ASC, slanted=False)))
        segs.append(("line", P(right, DESC, slanted=False)))
        segs.append(("line", P(R, DESC)))
        for c, e in self.arc(R, DESC + R, 270, 180):        # lower half of the end
            segs.append(("qcurve", P(*c), P(*e)))
        return P(0, DESC + R), segs

    def glyph(self, flip):
        start, segs = self.segments(flip)
        pen = TTGlyphPen(None)
        pen.moveTo(start)
        for s in segs:
            if s[0] == "line":
                pen.lineTo(s[1])
            else:
                pen.qCurveTo(s[1], s[2])
        pen.closePath()
        return pen.glyph()

    def tightest_radius(self):
        """Smallest radius of curvature along the visible curve, in font units."""
        start, segs = self.segments()
        best, p0 = math.inf, start
        for s in segs:
            if s[0] == "qcurve":
                (x0, y0), (x1, y1), (x2, y2) = p0, s[1], s[2]
                ddx, ddy = 2 * (x2 - 2 * x1 + x0), 2 * (y2 - 2 * y1 + y0)
                for i in range(41):
                    t = i / 40
                    dx = 2 * (1 - t) * (x1 - x0) + 2 * t * (x2 - x1)
                    dy = 2 * (1 - t) * (y1 - y0) + 2 * t * (y2 - y1)
                    cross = abs(dx * ddy - dy * ddx)
                    if cross:
                        best = min(best, (dx * dx + dy * dy) ** 1.5 / cross)
            p0 = s[-1]
        return best


def build(cap, out):
    empty = TTGlyphPen(None).glyph()
    fb = FontBuilder(UPM, isTTF=True)
    order = [".notdef", "space", "capl", "capr"]
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap({0x20: "space", 0xE100: "capl", 0xE101: "capr"})
    fb.setupGlyf({".notdef": empty, "space": empty,
                  "capl": cap.glyph(False), "capr": cap.glyph(True)})
    # The left side bearing must be each glyph's real xMin: FreeType moves
    # the outline to match it, so a wrong value shifts the cap away from
    # the capsule body and leaves a gap.
    glyf = fb.font["glyf"]
    lsb = {}
    for name in order:
        g = glyf[name]
        g.recalcBounds(glyf)
        lsb[name] = getattr(g, "xMin", 0)
    fb.setupHorizontalMetrics({name: (cap.adv, lsb[name]) for name in order})
    fb.setupHorizontalHeader(ascent=ASC, descent=DESC)
    fb.setupNameTable({"familyName": "Mira Caps", "styleName": "Regular",
                       "uniqueFontIdentifier": "MiraCaps-Regular",
                       "fullName": "Mira Caps", "psName": "MiraCaps-Regular",
                       "version": "Version 1.0",
                       "licenseDescription": "GPL-3.0-only"})
    fb.setupOS2(sTypoAscender=ASC, sTypoDescender=DESC, sTypoLineGap=0,
                usWinAscent=ASC, usWinDescent=-DESC, fsType=0)
    fb.setupPost()
    fb.save(out)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Build the Mira Caps font.")
    ap.add_argument("out", nargs="?",
                    default=os.path.join(here, "..", "assets", "fonts", "MiraCaps.ttf"))
    ap.add_argument("--radius", type=float, default=500,
                    help="corner radius before the slant, 1..500 (500 = full pill end)")
    ap.add_argument("--slant", type=float, default=400,
                    help="horizontal run of the slant over the full height")
    ap.add_argument("--check", type=float, metavar="PX",
                    help="print the tightest curve radius at this pixel size")
    args = ap.parse_args()
    cap = Cap(args.radius, args.slant)
    build(cap, args.out)
    if args.check:
        scale = args.check / UPM
        print(f"advance {cap.adv * scale:.1f}px, tightest curve radius "
              f"{cap.tightest_radius() * scale:.1f}px at {args.check:g}px")


if __name__ == "__main__":
    main()

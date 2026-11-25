#version 120
// Ember wall-01 — "Crimson Viaduct & Seaside Tram Crossing" (1:1 Cell-FBO GLSL)
// Bold 3px 4x4 ordered-Bayer pixel-art dither, molten-copper sunset sea shimmer
// strictly masked outside the stone viaduct piers, coastal parapet, far overlook
// balustrade, crossing posts & foreground bokeh guardrail; rippled wet cobblestone
// rainwater puddle mirroring the alternating twin fumikiri crossing warning lamps;
// gentle warm breathing on the retro Enoden tram windows, headlamp, bakery porch
// lantern & shopfront windows; pulsing distant cape beacon; distant flapping pixel
// coastal gulls strictly in open dusk sky above the tram; plus 2-layer drifting
// golden sunset dust & rising warm embers.
uniform sampler2D uTex;
uniform vec2 uRes;
uniform vec4 uCover;
uniform float uTime;
uniform vec2 uMouse;
uniform float uCell;
varying vec2 vUv;

float sstep(float a, float b, float x) {
    float t = clamp((x - a) / (b - a), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float bayer4(vec2 p) {
    vec2 b0 = floor(mod(p, 2.0));
    vec2 b1 = floor(mod(floor(p * 0.5), 2.0));
    float q0 = abs(b0.x - b0.y) * 2.0 + b0.y;
    float q1 = abs(b1.x - b1.y) * 2.0 + b1.y;
    return (q0 * 4.0 + q1) * 0.0625;
}

vec3 applyDither(vec3 col, vec2 cell, vec3 shadowInk) {
    vec3 c = clamp(col, 0.0, 1.0);
    float b = bayer4(cell);
    float lum = dot(c, vec3(0.299, 0.587, 0.114));
    float lumP = mix(lum, sqrt(max(lum, 0.0)), 0.19);
    float tL = clamp(lumP, 0.0, 1.0) * 6.0;
    float i0 = floor(tL);
    float qL = (i0 + step(b + 0.0001, tL - i0)) * 0.1666667;
    float qLin = qL * (0.83 + 0.17 * qL);

    vec3 chroma = c / max(lum, 0.035);
    float satMask = sstep(0.025, 0.16, lum);
    vec3 toned = max(mix(shadowInk * (0.24 + 1.55 * qLin), chroma * qLin, satMask), shadowInk * 0.24);
    vec3 tc = clamp(toned, 0.0, 1.0);

    vec3 ct = tc * 5.0;
    vec3 ci = floor(ct);
    vec3 qRgb = (ci + step(vec3(b + 0.0001), ct - ci)) * 0.2;
    float cMax = max(tc.r, max(tc.g, tc.b));
    float cMin = min(tc.r, min(tc.g, tc.b));
    return mix(tc, qRgb, 0.55 * sstep(0.03, 0.18, cMax - cMin));
}

void main() {
    float cellPx = max(1.0, uCell);
    vec2 grid = max(vec2(1.0), floor(uRes / cellPx));
    vec2 cell = floor(vUv * grid);
    vec2 uv = (cell + 0.5) / grid;
    float t = uTime;

    vec2 p = uv + vec2(uMouse.x * 0.0026, uMouse.y * 0.0015);

    // =========================================================================
    // 1. Strict Camera3D Sea Mask (horizon_v = 0.4570)
    //    Masked cleanly off coastal parapet, far overlook rail, viaduct piers,
    //    far crossing post, right coastal bluff & foreground lower bokeh rail!
    // =========================================================================
    float parapetEdge = mix(0.646, 0.968, clamp((uv.y - 0.520) / 0.480, 0.0, 1.0));
    float rightBay = sstep(0.457, 0.474, uv.y) * sstep(parapetEdge + 0.008, parapetEdge + 0.026, uv.x);

    // Sea seen cleanly through Arch[0] above far overlook rail (v in 0.457..0.514, u in 0.488..0.622)
    float arch0Sea = sstep(0.457, 0.470, uv.y) * sstep(0.516, 0.502, uv.y)
                   * sstep(0.485, 0.498, uv.x) * sstep(0.626, 0.612, uv.x);

    // Exclude solid stone viaduct piers (Pier[1]: u 0.658..0.710, Pier[2]: u 0.874..0.922)
    float pier1 = sstep(0.650, 0.662, uv.x) * sstep(0.718, 0.706, uv.x) * sstep(0.640, 0.626, uv.y);
    float pier2 = sstep(0.866, 0.878, uv.x) * sstep(0.930, 0.918, uv.x) * sstep(0.634, 0.620, uv.y);

    // Exclude right coastal rocky bluff/pine (u > 0.952, v < 0.51) & lower-right bokeh guardrail (u > 0.858, v > 0.715)
    float rightBluff = sstep(0.946, 0.962, uv.x) * sstep(0.515, 0.492, uv.y);
    float fgRail = sstep(0.852, 0.876, uv.x) * sstep(0.705, 0.735, uv.y);

    float seaMask = max(rightBay, arch0Sea) * (1.0 - pier1) * (1.0 - pier2) * (1.0 - rightBluff) * (1.0 - fgRail);

    // =========================================================================
    // 2. Strict Camera3D Wet Cobblestone Rainwater Puddle Mask
    //    BBox u:(0.2837..0.6313), v:(0.7040..0.9991), strictly off curb & parapet!
    // =========================================================================
    vec2 pudD = (uv - vec2(0.452, 0.852)) / vec2(0.148, 0.132);
    float pudMask = sstep(1.08, 0.62, dot(pudD, pudD))
                  * sstep(0.706, 0.738, uv.y)
                  * sstep(0.280, 0.312, uv.x)
                  * sstep(0.630, 0.592, uv.x);

    // Capillary sea waves on open ocean bay
    if (seaMask > 0.01) {
        float depth = clamp((uv.y - 0.457) * 1.84, 0.0, 1.0);
        float w1 = sin(uv.y * (172.0 - depth * 64.0) - t * 2.65 + sin(uv.x * 28.0 + t * 1.02) * 1.32);
        float w2 = cos(uv.y * 96.0 + t * 1.90 - uv.x * 42.0);
        p.x += (w1 * 0.0022 + w2 * 0.0009) * seaMask * (0.42 + 0.58 * depth);
        p.y += w2 * 0.0006 * seaMask;
    }

    // Gentle shallow micro-ripples inside the wet cobblestone puddle
    if (pudMask > 0.01) {
        float pw1 = sin(uv.y * 118.0 - t * 2.10 + uv.x * 22.0);
        float pw2 = cos(uv.y * 74.0 + t * 1.55 - uv.x * 31.0);
        p.x += (pw1 * 0.00105 + pw2 * 0.00045) * pudMask;
        p.y += pw2 * 0.00032 * pudMask;
    }

    vec2 tUV = clamp(p, 0.0, 1.0) * uCover.xy + uCover.zw;
    vec3 col = texture2D(uTex, tUV).rgb;

    // =========================================================================
    // 3. Distant Flapping Pixel Coastal Gulls Strictly in Open Dusk Sky Above Tram
    //    tram_v top = 0.1534, upper bokeh u > 0.812 -> safe open sky u:0.41..0.80, v:0.02..0.14
    // =========================================================================
    if (uv.x > 0.40 && uv.x < 0.81 && uv.y < 0.145) {
        float skyOpen = sstep(0.40, 0.44, uv.x) * sstep(0.81, 0.77, uv.x) * sstep(0.145, 0.125, uv.y);
        if (skyOpen > 0.01) {
            for (int b = 0; b < 4; b++) {
                float bf = float(b);
                float bx = 0.43 + fract(0.82 - bf * 0.23 - t * (0.014 + bf * 0.0028)) * 0.35;
                float by = 0.048 + bf * 0.022 + sin(t * (1.15 + bf * 0.26) + bf * 2.0) * 0.007;
                vec2 bd = (uv - vec2(bx, by)) * grid;
                float span = b < 2 ? 4.2 : 3.0;
                if (abs(bd.x) < span && abs(bd.y) < 2.8) {
                    float flap = sin(t * (5.0 + bf * 0.85) + bf * 1.6);
                    float ax = abs(bd.x);
                    float wingY = -ax * (0.24 + 0.28 * flap) + ax * ax * 0.042;
                    if (abs(bd.y - wingY) < 0.75) {
                        vec3 gullCol = (bd.y < wingY) ? vec3(0.42, 0.18, 0.12) : vec3(0.15, 0.055, 0.058);
                        col = mix(col, gullCol, 0.88 * skyOpen);
                    }
                }
            }
        }
    }

    // =========================================================================
    // 4. Molten-Copper Solar Glints Strictly on Open Sea Aligned with sun_u = 0.7894
    // =========================================================================
    if (seaMask > 0.35) {
        float lum = dot(col, vec3(0.3, 0.55, 0.15));
        float dxSun = (uv.x - 0.7894) * 3.8;
        float sunCol = exp(-dxSun * dxSun);
        float sp = hash12(vec2(cell.x, cell.y + floor(t * 2.4)));
        float spark = sstep(0.946, 0.995, sp) * sstep(0.28, 0.64, lum);
        col += vec3(1.0, 0.89, 0.58) * spark * (0.36 + 0.64 * sunCol) * seaMask * 0.45;
    }

    // =========================================================================
    // 5. Alternating Japanese Fumikiri Twin Crossing Warning Lamps!
    //    Exact Camera3D UVs: L=(0.3766, 0.4210), R=(0.4031, 0.4209)
    // =========================================================================
    float fumiWave = sin(t * 4.2);
    float fumiL = sstep(-0.35, 0.45, fumiWave);
    float fumiR = 1.0 - fumiL;

    if (abs(uv.x - 0.390) < 0.11 && abs(uv.y - 0.421) < 0.11) {
        vec2 dL = (uv - vec2(0.3766, 0.4210)) * vec2(1.75, 1.0);
        vec2 dR = (uv - vec2(0.4031, 0.4209)) * vec2(1.75, 1.0);
        float gL = exp(-dot(dL, dL) * 420.0);
        float gR = exp(-dot(dR, dR) * 420.0);
        float haloL = exp(-dot(dL, dL) * 68.0);
        float haloR = exp(-dot(dR, dR) * 68.0);
        col += vec3(1.0, 0.48, 0.18) * (gL * fumiL + gR * fumiR) * 0.34;
        col += vec3(0.98, 0.24, 0.11) * (haloL * fumiL + haloR * fumiR) * 0.16;
    }

    // Synchronized crimson-amber lamp reflection shimmer down inside the wet puddle!
    if (pudMask > 0.05) {
        vec2 dPL = (uv - vec2(0.382, 0.920)) * vec2(5.5, 3.2);
        vec2 dPR = (uv - vec2(0.408, 0.920)) * vec2(5.5, 3.2);
        float reflPulse = exp(-dot(dPL, dPL)) * fumiL + exp(-dot(dPR, dPR)) * fumiR;
        col += vec3(0.98, 0.30, 0.12) * reflPulse * pudMask * 0.14;
    }

    // =========================================================================
    // 6. Distant Cape Beacon (0.9450, 0.3930), Enoden Tram Windows & Headlamp,
    //    and Left Seaside Bakery Porch Lantern & Glowing Mullioned Windows
    // =========================================================================
    if (abs(uv.x - 0.9450) < 0.10 && abs(uv.y - 0.3930) < 0.10) {
        vec2 vBcn = (uv - vec2(0.9450, 0.3930)) * vec2(1.75, 1.0);
        float bw = 0.5 + 0.5 * sin(t * 1.65);
        float b2 = bw * bw;
        float beacon = b2 * b2 * b2;
        col += vec3(1.0, 0.91, 0.66) * exp(-dot(vBcn, vBcn) * 260.0) * (0.12 + 0.48 * beacon);
    }

    // Retro Enoden tram glowing passenger window row (u:0.484..0.794, v:0.153..0.245) & headlamp (0.7836, 0.1695)
    if (uv.x > 0.47 && uv.x < 0.82 && uv.y > 0.13 && uv.y < 0.27) {
        float tramWin = sstep(0.485, 0.502, uv.x) * sstep(0.792, 0.775, uv.x)
                      * sstep(0.162, 0.178, uv.y) * sstep(0.244, 0.228, uv.y);
        float tramPulse = 0.55 + 0.45 * sin(t * 1.85 + uv.x * 6.0);
        col += vec3(1.0, 0.76, 0.34) * tramWin * tramPulse * 0.042;
        vec2 vHl = (uv - vec2(0.7836, 0.1695)) * vec2(2.4, 1.4);
        col += vec3(1.0, 0.88, 0.54) * exp(-dot(vHl, vHl) * 220.0) * (0.08 + 0.07 * sin(t * 2.4));
    }

    // Bakery hanging brass porch lantern (0.2831, 0.4584) & cozy mullioned windows warm breathing
    if (uv.x < 0.36 && uv.y > 0.12 && uv.y < 0.68) {
        vec2 dPorch = (uv - vec2(0.2831, 0.4584)) * vec2(3.8, 2.4);
        col += vec3(1.0, 0.74, 0.34) * exp(-dot(dPorch, dPorch) * 58.0) * (0.06 + 0.05 * sin(t * 2.1 + 0.5));

        vec2 dW0 = (uv - vec2(0.0971, 0.5489)) * vec2(6.2, 4.4);
        vec2 dW1 = (uv - vec2(0.2312, 0.5602)) * vec2(7.0, 4.8);
        vec2 dW2 = (uv - vec2(0.2990, 0.5121)) * vec2(8.5, 5.8);
        vec2 dW3 = (uv - vec2(0.0948, 0.1836)) * vec2(6.5, 4.6);
        vec2 dW4 = (uv - vec2(0.2330, 0.2582)) * vec2(7.5, 5.4);
        float wGlow = exp(-dot(dW0, dW0)) * (0.55 + 0.45 * sin(t * 1.9))
                    + exp(-dot(dW1, dW1)) * (0.55 + 0.45 * sin(t * 2.1 + 1.2))
                    + exp(-dot(dW2, dW2)) * (0.55 + 0.45 * sin(t * 2.0 + 2.4))
                    + exp(-dot(dW3, dW3)) * (0.55 + 0.45 * sin(t * 1.7 + 3.1))
                    + exp(-dot(dW4, dW4)) * (0.55 + 0.45 * sin(t * 1.8 + 4.2));
        col += vec3(0.98, 0.62, 0.24) * wGlow * 0.044;
    }

    // =========================================================================
    // 7. 2-Layer Sunset Coastal Atmosphere: Drifting Golden Dusk Dust & Embers
    // =========================================================================
    for (int l = 0; l < 2; l++) {
        float lf = float(l);
        vec2 dim = l == 0 ? vec2(26.0, 15.0) : vec2(14.0, 9.0);
        vec2 eCoord = vec2(
            uv.x - t * (0.018 + lf * 0.014) + sin(t * (0.7 + lf * 0.4) + uv.y * 6.0) * 0.010,
            uv.y + t * (0.016 + lf * 0.022)
        );
        vec2 gid = floor(eCoord * dim);
        float rnd = hash12(gid + lf * 73.0 + 13.0);
        if (rnd > (l == 0 ? 0.55 : 0.60)) {
            vec2 c = (gid + 0.18 + 0.64 * vec2(hash12(gid + 3.7), hash12(gid + 11.3))) / dim;
            vec2 dpx = abs((eCoord - c) * grid);
            float rad = l == 0 ? 0.58 : 0.85;
            if (dpx.x < rad && dpx.y < rad) {
                float tw = 0.50 + 0.50 * sin(t * (2.4 + rnd * 2.6) + rnd * 15.0);
                vec3 eCol = l == 0 ? vec3(1.0, 0.82, 0.52) : vec3(1.0, 0.64, 0.24);
                col = mix(col, eCol, (l == 0 ? 0.58 : 0.86) * tw);
            }
        }
    }

    gl_FragColor = vec4(applyDither(col, cell, vec3(0.068, 0.036, 0.032)), 1.0);
}

#version 120
// Ember wall-02 — "Twilight Onsen Bathhouse Canal" (Hyper-Optimized 1:1 Cell-FBO GLSL)
// Bold 3px 4x4 ordered-Bayer pixel-art dither, strictly Camera3D-masked shimmering
// thermal canal water ripples (zero wobble on arched bridge, ishigaki embankments,
// ryokan facade, boulders, or bokeh), phase-staggered organic candle flicker dead-
// centered on all 10 hanging paper chochin lanterns & water reflection columns,
// curling translucent warm onsen steam wisps above the canal, pouring spring spout
// & bathhouse exhaust louver, rising golden lantern sparks, and 2-layer tumbling
// crimson-amber autumn momiji maple leaves.
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

float glowPt(vec2 uv, vec2 c, vec2 sc) {
    vec2 d = (uv - c) * sc;
    return exp(-dot(d, d));
}

void main() {
    float cellPx = max(1.0, uCell);
    vec2 grid = max(vec2(1.0), floor(uRes / cellPx));
    vec2 cell = floor(vUv * grid);
    vec2 uv = (cell + 0.5) / grid;
    float t = uTime;

    vec2 p = uv + vec2(uMouse.x * 0.0028, uMouse.y * 0.0016);

    // Strictly calibrated Camera3D thermal canal water mask (z = 0):
    // Left ishigaki waterline:  u_L(v) = 0.5014 - (v - 0.6653) * 0.5316
    // Right ishigaki waterline: u_R(v) = 0.7337 + (v - 0.6700) * 0.8597 (inset safely inside boulders!)
    // Arched taiko-bashi underside: crown (0.6169, 0.5601), shoulders (0.5235, 0.5918) & (0.7124, 0.5851)
    float leftBankU = 0.5014 - (uv.y - 0.6653) * 0.5316;
    float rightBankU = 0.7337 + (uv.y - 0.6700) * 0.8597 - 0.056;
    float dxArch = (uv.x - 0.6169) / 0.095;
    float archCutV = 0.562 + 0.032 * dxArch * dxArch;

    float water = sstep(leftBankU + 0.014, leftBankU + 0.034, uv.x)
                * sstep(rightBankU + 0.012, rightBankU - 0.018, uv.x)
                * sstep(archCutV, archCutV + 0.022, uv.y);

    // Exclude the pouring hot-spring spout stream jet near (0.478, 0.69..0.76) from horizontal wobble
    float spoutJet = sstep(0.455, 0.468, uv.x) * sstep(0.505, 0.492, uv.x)
                   * sstep(0.675, 0.692, uv.y) * sstep(0.768, 0.748, uv.y);
    water *= (1.0 - spoutJet);

    if (water > 0.01) {
        float depth = clamp((uv.y - 0.52) * 2.08, 0.0, 1.0);
        float w1 = sin(uv.y * (162.0 - depth * 62.0) - t * 2.4 + sin(uv.x * 32.0 + t * 0.95) * 1.25);
        float w2 = cos(uv.y * 92.0 + t * 1.75 - uv.x * 46.0);
        p.x += (w1 * 0.0020 + w2 * 0.00085) * water * (0.40 + 0.60 * depth);
        p.y += w2 * 0.00055 * water;
    }

    vec2 tUV = clamp(p, 0.0, 1.0) * uCover.xy + uCover.zw;
    vec3 col = texture2D(uTex, tUV).rgb;

    // Fine 1-cell liquid amber water sparkles strictly inside bright canal reflection zones
    if (water > 0.35) {
        float lum = dot(col, vec3(0.3, 0.55, 0.15));
        float sp = hash12(vec2(cell.x, cell.y + floor(t * 1.75)));
        float spark = sstep(0.958, 0.996, sp) * sstep(0.28, 0.64, lum);
        col += vec3(0.96, 0.78, 0.44) * spark * water * 0.24;
    }

    // Calm, gentle candle-breathing warmth around all 10 hanging chochin lanterns & water reflections
    // (Slow 4s-loop-locked breathing, highlight-protected so lantern cores stay cozy & never glare/strobe!)
    float baseLum = dot(col, vec3(0.299, 0.587, 0.114));
    float glareGuard = 1.0 - 0.78 * sstep(0.62, 0.92, baseLum);

    // Phase A: 1f_near (0.2899, 0.4527), 1f_mid (0.4121, 0.4707), 2f_near (0.3478, 0.1616), bridge_L (0.5490, 0.4959)
    float fA = 0.017 * sin(t * 1.5708) + 0.006 * sin(t * 3.1416 + 1.1);
    float gA = glowPt(uv, vec2(0.2899, 0.4527), vec2(11.5, 8.2))
             + glowPt(uv, vec2(0.4121, 0.4707), vec2(13.5, 9.6)) * 0.85
             + glowPt(uv, vec2(0.3478, 0.1616), vec2(12.0, 8.6)) * 0.86
             + glowPt(uv, vec2(0.5490, 0.4959), vec2(15.0, 10.8)) * 0.84
             + glowPt(uv, vec2(0.4150, 0.9400), vec2(8.0, 5.0)) * water * 0.42
             + glowPt(uv, vec2(0.5490, 0.6950), vec2(10.0, 5.6)) * water * 0.36;

    // Phase B: 1f_entry (0.3631, 0.4622), 2f_mid (0.4231, 0.2677), bridge_R (0.6840, 0.4904), hero shoji bays
    float fB = 0.016 * sin(t * 1.5708 + 2.1) + 0.006 * sin(t * 3.1416 + 0.4);
    float gB = glowPt(uv, vec2(0.3631, 0.4622), vec2(12.0, 8.6)) * 0.96
             + glowPt(uv, vec2(0.4231, 0.2677), vec2(14.0, 10.0)) * 0.82
             + glowPt(uv, vec2(0.6840, 0.4904), vec2(15.0, 10.8)) * 0.84
             + glowPt(uv, vec2(0.2638, 0.5440), vec2(6.8, 6.2)) * 0.24
             + glowPt(uv, vec2(0.2677, 0.2646), vec2(7.0, 6.4)) * 0.22
             + glowPt(uv, vec2(0.4580, 0.8450), vec2(8.5, 5.2)) * water * 0.40
             + glowPt(uv, vec2(0.6840, 0.6980), vec2(10.0, 5.6)) * water * 0.36;

    // Phase C: 1f_far (0.4443, 0.4762), 2f_far (0.4577, 0.3166), pavilion (0.7918, 0.4429), 3f marumado
    float fC = 0.015 * sin(t * 1.5708 + 4.2) + 0.005 * sin(t * 3.1416 + 2.5);
    float gC = glowPt(uv, vec2(0.4443, 0.4762), vec2(15.0, 10.8)) * 0.78
             + glowPt(uv, vec2(0.4577, 0.3166), vec2(15.5, 11.2)) * 0.76
             + glowPt(uv, vec2(0.7918, 0.4429), vec2(13.5, 9.6)) * 0.86
             + glowPt(uv, vec2(0.3864, 0.1969), vec2(13.5, 10.0)) * 0.32
             + glowPt(uv, vec2(0.5020, 0.7450), vec2(10.2, 5.8)) * water * 0.34
             + glowPt(uv, vec2(0.7450, 0.7650), vec2(9.2, 5.4)) * water * 0.36;

    col += vec3(0.92, 0.50, 0.18) * (gA * fA + gB * fB + gC * fC) * glareGuard;

    // Curling translucent warm onsen steam wisps rising lazily over canal, pouring spout & bathhouse louver
    if (uv.x > 0.34 && uv.x < 0.84 && uv.y > 0.40 && uv.y < 0.92) {
        float stSpout = glowPt(uv, vec2(0.502, 0.725), vec2(11.0, 8.5));
        float stLouver = glowPt(uv, vec2(0.415, 0.495), vec2(12.5, 9.0));
        float stCanal = glowPt(uv, vec2(0.618, 0.675), vec2(4.6, 7.2)) * sstep(0.525, 0.565, uv.y);
        float stEnv = clamp(stSpout * 0.85 + stLouver * 0.78 + stCanal * 0.62, 0.0, 1.0);
        if (stEnv > 0.02) {
            float sx = uv.x * 26.0 + sin(uv.y * 18.0 - t * 0.95) * 1.4;
            float sy = uv.y * 16.0 + t * 0.55;
            float wisp = 0.5 + 0.5 * sin(sx + cos(sy * 1.3 + t * 0.45) * 1.6);
            float wisp2 = 0.5 + 0.5 * cos(sx * 0.65 - sy * 1.15 + sin(t * 0.35));
            float curl = sstep(0.34, 0.86, wisp * wisp2);
            vec3 steamCol = mix(vec3(0.96, 0.74, 0.62), vec3(1.0, 0.82, 0.52), stLouver + stSpout * 0.6);
            col = mix(col, steamCol, curl * stEnv * 0.14);
        }
    }

    // Soft warm golden lantern embers drifting lazily upward into twilight air
    vec2 sCoord = vec2(uv.x + sin(t * 0.7854 + uv.y * 7.0) * 0.009, uv.y + t * 0.028);
    vec2 sgid = floor(sCoord * vec2(22.0, 14.0));
    float srnd = hash12(sgid + 131.0);
    if (srnd > 0.66) {
        vec2 sc = (sgid + 0.20 + 0.60 * vec2(hash12(sgid + 5.1), hash12(sgid + 12.7))) / vec2(22.0, 14.0);
        vec2 dpx = abs((sCoord - sc) * grid);
        if (dpx.x < 0.56 && dpx.y < 0.56) {
            float tw = 0.50 + 0.50 * sin(t * (1.5708 + floor(srnd * 2.0) * 1.5708) + srnd * 18.0);
            col = mix(col, vec3(0.96, 0.76, 0.38), 0.54 * tw);
        }
    }

    // 2-layer crisp tumbling crimson-amber autumn momiji maple leaves floating in evening breeze
    for (int l = 0; l < 2; l++) {
        float lf = float(l);
        vec2 dim = l == 0 ? vec2(12.0, 7.5) : vec2(8.0, 5.5);
        vec2 lCoord = vec2(uv.x + t * (0.044 + lf * 0.024), uv.y - t * (0.034 + lf * 0.018));
        vec2 gid = floor(lCoord * dim);
        float rnd = hash12(gid + lf * 67.0 + 19.0);
        if (rnd > 0.61) {
            vec2 center = (gid + 0.20 + 0.60 * vec2(hash12(gid + 3.7), hash12(gid + 19.3))) / dim;
            center.x += sin(t * (1.7 + rnd * 1.4) + rnd * 6.28) * 0.013;
            center.y += cos(t * (1.2 + rnd) + rnd * 4.0) * 0.009;
            vec2 d = (lCoord - center) * grid;
            float sz = l == 0 ? 1.12 : 1.65;
            if (dot(d, d) < sz * sz) {
                float ang = t * (1.5 + rnd * 2.0) + rnd * 6.28;
                float ca = cos(ang), sa = sin(ang);
                vec2 rd = vec2(ca * d.x - sa * d.y, sa * d.x + ca * d.y);
                if (abs(rd.x) < sz && abs(rd.y) < sz * 0.68) {
                    vec3 leafCol = mix(vec3(0.88, 0.24, 0.14), vec3(1.0, 0.54, 0.20), rnd);
                    col = mix(col, leafCol, 0.92);
                }
            }
        }
    }

    gl_FragColor = vec4(applyDither(col, cell, vec3(0.068, 0.036, 0.032)), 1.0);
}

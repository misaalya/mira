#version 120
// Ember wall-01 — "Crimson Harbor Pier" (Hyper-Optimized 1:1 Cell-FBO GLSL)
// Bold 3px 4x4 ordered-Bayer pixel-art dither, molten-copper harbor water strictly
// outside the timber pier, pilings, boat hull & mast, distant flapping pixel harbor
// gulls gliding across the crimson sunset sky, pulsing lighthouse beacon, plus
// 2-layer warm drifting sunset sea-mist dust & rising golden harbor embers.
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

    vec2 p = uv + vec2(uMouse.x * 0.0028, uMouse.y * 0.0016);

    // Strictly mask open harbor water: right of pier edge & pilings, outside boat hull, mast & foreground post
    float pierEdge = mix(0.57, 0.60, sstep(1.0, 0.50, uv.y));
    float boatBody = sstep(0.57, 0.60, uv.x) * sstep(0.88, 0.84, uv.x)
                   * sstep(0.52, 0.56, uv.y) * sstep(0.84, 0.79, uv.y);
    float boatMast = sstep(0.75, 0.77, uv.x) * sstep(0.84, 0.82, uv.x)
                   * sstep(0.36, 0.40, uv.y) * sstep(0.58, 0.54, uv.y);
    float fgPost = sstep(0.91, 0.94, uv.x) * sstep(0.68, 0.74, uv.y);
    float water = sstep(0.45, 0.50, uv.y)
                * sstep(pierEdge, pierEdge + 0.025, uv.x)
                * (1.0 - boatBody) * (1.0 - boatMast) * (1.0 - fgPost);

    if (water > 0.01) {
        float depth = clamp((uv.y - 0.45) * 1.81818, 0.0, 1.0);
        float w1 = sin(uv.y * (165.0 - depth * 65.0) - t * 2.8 + sin(uv.x * 30.0 + t * 1.1) * 1.4);
        float w2 = cos(uv.y * 96.0 + t * 2.0 - uv.x * 44.0);
        p.x += (w1 * 0.0024 + w2 * 0.0010) * water * (0.45 + 0.55 * depth);
        p.y += w2 * 0.0007 * water;
    }

    vec2 tUV = clamp(p, 0.0, 1.0) * uCover.xy + uCover.zw;
    vec3 col = texture2D(uTex, tUV).rgb;

    // 4 sculpted distant flapping pixel harbor gulls strictly in the open sunset sky
    if (uv.x > 0.40 && uv.y < 0.38) {
        float roofLine = 0.02 + uv.x * 0.68;
        float skyOpen = sstep(0.40, 0.46, uv.x) * sstep(0.38, 0.28, uv.y) * sstep(roofLine, roofLine - 0.03, uv.y);
        if (skyOpen > 0.01) {
            for (int b = 0; b < 4; b++) {
                float bf = float(b);
                float bx = 0.44 + fract(0.84 - bf * 0.20 - t * (0.016 + bf * 0.003)) * 0.52;
                float by = 0.12 + bf * 0.044 + sin(t * (1.2 + bf * 0.3) + bf * 2.1) * 0.011;
                vec2 bd = (uv - vec2(bx, by)) * grid;
                float span = b < 2 ? 4.4 : 3.2;
                if (abs(bd.x) < span && abs(bd.y) < 3.0) {
                    float flap = sin(t * (5.2 + bf * 0.9) + bf * 1.7);
                    float ax = abs(bd.x);
                    float wingY = -ax * (0.24 + 0.30 * flap) + ax * ax * 0.042;
                    if (abs(bd.y - wingY) < 0.78) {
                        vec3 gullCol = (bd.y < wingY) ? vec3(0.42, 0.18, 0.12) : vec3(0.16, 0.06, 0.06);
                        col = mix(col, gullCol, 0.90 * skyOpen);
                    }
                }
            }
        }
    }

    // Fine 1-cell pixel solar glints ONLY on bright open water ripples
    if (water > 0.35) {
        float lum = dot(col, vec3(0.3, 0.55, 0.15));
        float dxSun = (uv.x - 0.72) * 3.8;
        float sunCol = exp(-dxSun * dxSun);
        float sp = hash12(vec2(cell.x, cell.y + floor(t * 2.4)));
        float spark = sstep(0.945, 0.995, sp) * sstep(0.30, 0.65, lum);
        col += vec3(1.0, 0.88, 0.56) * spark * (0.4 + 0.6 * sunCol) * water * 0.45;
    }

    // Distant headland lighthouse beacon pulse at (0.78, 0.365) (localized early-exit)
    if (abs(uv.x - 0.78) < 0.14 && abs(uv.y - 0.365) < 0.16) {
        vec2 vLight = (uv - vec2(0.78, 0.365)) * vec2(1.6, 1.0);
        float bWave = 0.5 + 0.5 * sin(t * 1.7);
        float b2 = bWave * bWave;
        float beacon = b2 * b2 * b2;
        col += vec3(1.0, 0.90, 0.65) * exp(-dot(vLight, vLight) * 200.0) * (0.14 + 0.52 * beacon);
    }

    // Warm warehouse window & bulkhead lamp subtle breathing (localized early-exit)
    if (abs(uv.x - 0.25) < 0.32 && abs(uv.y - 0.40) < 0.30) {
        vec2 vLamp = (uv - vec2(0.25, 0.40)) * vec2(5.5, 6.5);
        col += vec3(0.98, 0.56, 0.24) * exp(-dot(vLamp, vLamp)) * (0.038 * sin(t * 2.4));
    }

    // 2-layer twilight harbor atmosphere: drifting warm sunset sea-mist dust & rising golden embers
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

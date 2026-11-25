#version 120
// Fjord wall-01 — "Blue-Hour Fjord Boathouse" (Hyper-Optimized 1:1 Cell-FBO GLSL)
// Bold 3px 4x4 ordered-Bayer pixel-art dither (matching Website & SDDM),
// drifting blue-hour pixel snow flurries, icy mirrored fjord water strictly
// outside the timber pier & moored rowboat, and warm boathouse hearth flicker.
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

    // Strictly mask open icy fjord water: right of pier edge & posts, outside rowboat hull
    float pierRight = mix(0.47, 0.61, sstep(1.0, 0.56, uv.y));
    float boatHull = sstep(0.58, 0.62, uv.x) * sstep(0.90, 0.86, uv.x)
                   * sstep(0.69, 0.73, uv.y) * sstep(0.98, 0.93, uv.y);
    float fjord = sstep(0.44, 0.50, uv.y)
                * sstep(pierRight, pierRight + 0.03, uv.x)
                * (1.0 - boatHull);
    if (fjord > 0.01) {
        float w1 = sin(uv.y * 120.0 - t * 1.9 + cos(uv.x * 24.0 + t * 0.8) * 1.3);
        p.x += w1 * 0.0018 * fjord;
        p.y += cos(uv.y * 68.0 + t * 1.3) * 0.0005 * fjord;
    }

    vec2 tUV = clamp(p, 0.0, 1.0) * uCover.xy + uCover.zw;
    vec3 col = texture2D(uTex, tUV).rgb;

    // Cozy warm hearth flicker inside boathouse window (0.325, 0.45)
    vec2 vWin = (uv - vec2(0.325, 0.45)) * vec2(7.0, 5.2);
    float hearth = 0.048 * sin(t * 3.2) + 0.026 * sin(t * 7.8 + 1.4);
    col += vec3(1.0, 0.72, 0.34) * exp(-dot(vWin, vWin)) * hearth;

    // 2-layer crisp drifting Nordic pixel snow flurries
    for (int l = 0; l < 2; l++) {
        float lf = float(l);
        vec2 dim = l == 0 ? vec2(26.0, 17.0) : vec2(15.0, 10.0);
        vec2 sCoord = vec2(uv.x - t * (0.016 + lf * 0.012), uv.y - t * (0.042 + lf * 0.028));
        vec2 gid = floor(sCoord * dim);
        float rnd = hash12(gid + lf * 71.0 + 7.0);
        if (rnd > 0.56) {
            vec2 c = (gid + 0.18 + 0.64 * vec2(hash12(gid + 2.3), hash12(gid + 15.9))) / dim;
            c.x += sin(t * (1.2 + rnd * 1.5) + rnd * 6.28) * 0.008;
            vec2 dpx = abs((sCoord - c) * grid);
            float rad = l == 0 ? 0.55 : 0.85;
            if (dpx.x < rad && dpx.y < rad) {
                float br = (l == 0 ? 0.48 : 0.76) * (0.65 + 0.35 * sin(t * 2.0 + rnd * 10.0));
                col = mix(col, vec3(0.88, 0.95, 1.0), br);
            }
        }
    }

    gl_FragColor = vec4(applyDither(col, cell, vec3(0.035, 0.058, 0.088)), 1.0);
}

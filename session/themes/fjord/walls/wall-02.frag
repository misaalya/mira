#version 120
// Fjord wall-02 — "Glacier Station Overpass" (Hyper-Optimized 1:1 Cell-FBO GLSL)
// Bold 3px 4x4 ordered-Bayer pixel-art dither (matching Website & SDDM),
// blowing Nordic pixel snow & icy mist, and subtle fluorescent station lamp
// shimmer on wet slate pavers (keeping steel bench & columns razor-straight).
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
    vec2 tUV = clamp(p, 0.0, 1.0) * uCover.xy + uCover.zw;
    vec3 col = texture2D(uTex, tUV).rgb;

    // Subtle cold fluorescent platform ceiling lamp shimmer & wet slate reflection
    float hum = 0.032 * sin(t * 5.2) + 0.016 * sin(t * 13.7 + 0.6);
    vec2 vL1 = (uv - vec2(0.535, 0.15)) * vec2(4.5, 5.5);
    vec2 vL2 = (uv - vec2(0.56, 0.72)) * vec2(4.5, 2.8);
    col += vec3(0.58, 0.80, 1.0) * (exp(-dot(vL1, vL1)) + exp(-dot(vL2, vL2)) * 0.55) * hum;

    // Drifting cold fjord mist blowing gently past the tracks
    float mist = (0.5 + 0.5 * sin(uv.x * 4.8 - t * 0.24 + cos(uv.y * 5.0) * 1.2))
               * sstep(0.24, 0.48, uv.y) * sstep(0.72, 0.52, uv.y);
    col += vec3(0.45, 0.62, 0.80) * mist * 0.05;

    // 2-layer crisp diagonal blowing pixel snow flurries
    for (int l = 0; l < 2; l++) {
        float lf = float(l);
        vec2 dim = l == 0 ? vec2(28.0, 17.0) : vec2(16.0, 10.0);
        vec2 sCoord = vec2(uv.x - t * (0.040 + lf * 0.025), uv.y - t * (0.055 + lf * 0.032));
        vec2 gid = floor(sCoord * dim);
        float rnd = hash12(gid + lf * 83.0 + 5.0);
        if (rnd > 0.58) {
            vec2 c = (gid + 0.18 + 0.64 * vec2(hash12(gid + 6.1), hash12(gid + 21.4))) / dim;
            c.x += sin(t * (1.5 + rnd) + rnd * 6.28) * 0.006;
            vec2 dpx = abs((sCoord - c) * grid);
            float rad = l == 0 ? 0.55 : 0.85;
            if (dpx.x < rad && dpx.y < rad) {
                float br = (l == 0 ? 0.45 : 0.72) * (0.6 + 0.4 * sin(t * 2.5 + rnd * 9.0));
                col = mix(col, vec3(0.86, 0.94, 1.0), br);
            }
        }
    }

    gl_FragColor = vec4(applyDither(col, cell, vec3(0.035, 0.058, 0.088)), 1.0);
}

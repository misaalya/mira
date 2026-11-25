#version 120
// Ember wall-02 — "Lantern Shrine Stairway" (Hyper-Optimized 1:1 Cell-FBO GLSL)
// Bold 3px 4x4 ordered-Bayer pixel-art dither, organic candle-flame flicker inside
// stone toro lanterns, rich 2-layer drifting crimson/amber momiji maple leaves,
// rising warm golden lantern embers, and razor-straight basalt steps.
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

    // Warm dual-frequency candle flicker across stone toro lanterns + hanging lamp
    float f1 = 0.058 * sin(t * 3.8) + 0.032 * sin(t * 8.9 + 1.3);
    float f2 = 0.052 * sin(t * 3.1 + 2.1) + 0.030 * sin(t * 7.6 + 0.4);
    vec2 vR = (uv - vec2(0.68, 0.39)) * vec2(4.2, 3.0);
    vec2 vL = (uv - vec2(0.36, 0.38)) * vec2(4.6, 3.4);
    vec2 vM = (uv - vec2(0.45, 0.33)) * vec2(6.0, 4.5);
    vec2 vH = (uv - vec2(0.44, 0.16)) * vec2(7.0, 5.5);
    vec2 vS = (uv - vec2(0.54, 0.78)) * vec2(2.6, 3.8);
    float g1 = exp(-dot(vR, vR)) + exp(-dot(vM, vM)) + exp(-dot(vS, vS)) * 0.35;
    float g2 = exp(-dot(vL, vL)) + exp(-dot(vH, vH));
    col += vec3(1.0, 0.62, 0.24) * (g1 * f1 + g2 * f2);

    // Rising warm golden candle sparks / lantern embers drifting lazily upward
    vec2 sCoord = vec2(uv.x + sin(t * 0.9 + uv.y * 8.0) * 0.010, uv.y + t * 0.034);
    vec2 sgid = floor(sCoord * vec2(22.0, 14.0));
    float srnd = hash12(sgid + 127.0);
    if (srnd > 0.56) {
        vec2 sc = (sgid + 0.20 + 0.60 * vec2(hash12(sgid + 4.9), hash12(sgid + 12.1))) / vec2(22.0, 14.0);
        vec2 dpx = abs((sCoord - sc) * grid);
        if (dpx.x < 0.58 && dpx.y < 0.58) {
            float tw = 0.45 + 0.55 * sin(t * (3.0 + srnd * 3.0) + srnd * 18.0);
            col = mix(col, vec3(1.0, 0.84, 0.42), 0.82 * tw);
        }
    }

    // 2-layer crisp tumbling autumn momiji maple leaves floating diagonally in evening breeze
    for (int l = 0; l < 2; l++) {
        float lf = float(l);
        vec2 dim = l == 0 ? vec2(12.0, 7.5) : vec2(8.0, 5.5);
        vec2 lCoord = vec2(uv.x + t * (0.048 + lf * 0.026), uv.y - t * (0.038 + lf * 0.020));
        vec2 gid = floor(lCoord * dim);
        float rnd = hash12(gid + lf * 67.0 + 11.0);
        if (rnd > 0.60) {
            vec2 center = (gid + 0.20 + 0.60 * vec2(hash12(gid + 3.7), hash12(gid + 19.3))) / dim;
            center.x += sin(t * (1.8 + rnd * 1.4) + rnd * 6.28) * 0.013;
            center.y += cos(t * (1.3 + rnd) + rnd * 4.0) * 0.009;
            vec2 d = (lCoord - center) * grid;
            float sz = l == 0 ? 1.15 : 1.70;
            if (dot(d, d) < sz * sz) {
                float ang = t * (1.6 + rnd * 2.0) + rnd * 6.28;
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

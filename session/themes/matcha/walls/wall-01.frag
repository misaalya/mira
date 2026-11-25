#version 120
// Matcha wall-01 — "Arashiyama Tea Veranda" (Hyper-Optimized 1:1 Cell-FBO GLSL)
// Bold 3px 4x4 ordered-Bayer pixel-art dither, curling morning steam above the
// matcha chawan, swaying bamboo fronds, calm Zen pond ripples, sunlit morning
// pollen motes, and drifting pale-jade bamboo leaf petals on the breeze.
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

    vec2 p = uv + vec2(uMouse.x * 0.003, uMouse.y * 0.0018);

    // Gentle Arashiyama breeze swaying bamboo fronds in upper-right garden
    float bamboo = sstep(0.60, 0.74, uv.x) * sstep(0.54, 0.24, uv.y);
    if (bamboo > 0.01) {
        float sway = sin(t * 1.35 + uv.y * 7.0 + uv.x * 5.0) * 0.0032
                   + sin(t * 2.6 - uv.x * 11.0) * 0.0010;
        p.x += sway * bamboo;
    }

    // Calm pond ripples in lower-right mossy Zen garden
    float pond = sstep(0.56, 0.70, uv.y) * sstep(0.46, 0.62, uv.x);
    if (pond > 0.01) {
        float rip = sin(uv.y * 92.0 - t * 2.0 + sin(uv.x * 26.0 + t * 0.8) * 1.4);
        p.x += rip * 0.0020 * pond;
        p.y += cos(uv.x * 48.0 + t * 1.5) * 0.0007 * pond;
    }

    vec2 tUV = clamp(p, 0.0, 1.0) * uCover.xy + uCover.zw;
    vec3 col = texture2D(uTex, tUV).rgb;

    // Live curling sunlit morning steam rising from the ceramic matcha chawan (0.255, 0.71)
    float sy = (0.71 - uv.y) * 4.0;
    if (sy > 0.0 && sy < 1.0 && abs(uv.x - 0.258) < 0.085) {
        float curl = sin(sy * 7.5 - t * 1.85) * (0.007 + sy * 0.013);
        float dx = (uv.x - (0.258 + curl)) / (0.013 + sy * 0.024);
        float wisp = exp(-dx * dx) * sin(sy * 3.14159) * (0.65 + 0.35 * sin(sy * 12.0 - t * 2.6));
        col += vec3(0.88, 0.94, 0.82) * wisp * 0.24;
    }

    // Warm morning sunbeams across garden
    float mist = (0.5 + 0.5 * sin(uv.x * 4.5 - t * 0.15 + uv.y * 3.0))
               * sstep(0.56, 0.18, uv.y) * sstep(0.55, 0.75, uv.x);
    col += vec3(0.74, 0.84, 0.58) * mist * 0.055;

    // Layer 0: Warm sunlit Kyoto morning pollen & dust motes drifting across veranda & garden
    vec2 mCoord = vec2(uv.x + t * 0.018, uv.y - t * 0.010 + sin(t * 0.7 + uv.x * 9.0) * 0.009);
    vec2 gid = floor(mCoord * vec2(24.0, 14.0));
    float rnd = hash12(gid + 31.0);
    if (rnd > 0.52) {
        vec2 c = (gid + 0.18 + 0.64 * vec2(hash12(gid + 4.3), hash12(gid + 9.1))) / vec2(24.0, 14.0);
        vec2 dpx = abs((mCoord - c) * grid);
        if (dpx.x < 0.60 && dpx.y < 0.60) {
            float tw = 0.50 + 0.50 * sin(t * (2.0 + rnd * 2.0) + rnd * 14.0);
            col = mix(col, vec3(0.95, 0.98, 0.78), 0.72 * tw);
        }
    }

    // Layer 1: Crisp drifting bamboo leaf petals tumbling gently on the morning breeze
    vec2 lCoord = vec2(uv.x + t * 0.042, uv.y - t * 0.024);
    vec2 lgid = floor(lCoord * vec2(11.0, 7.0));
    float lrnd = hash12(lgid + 77.0);
    if (lrnd > 0.64) {
        vec2 lc = (lgid + 0.20 + 0.60 * vec2(hash12(lgid + 2.7), hash12(lgid + 14.9))) / vec2(11.0, 7.0);
        lc.x += sin(t * (1.4 + lrnd) + lrnd * 6.28) * 0.012;
        lc.y += cos(t * (1.1 + lrnd) + lrnd * 4.0) * 0.008;
        vec2 d = (lCoord - lc) * grid;
        if (dot(d, d) < 2.4) {
            float ang = t * (1.5 + lrnd * 1.6) + lrnd * 6.28;
            float ca = cos(ang), sa = sin(ang);
            vec2 rd = vec2(ca * d.x - sa * d.y, sa * d.x + ca * d.y);
            if (abs(rd.x) < 1.35 && abs(rd.y) < 0.62) {
                vec3 leafCol = mix(vec3(0.36, 0.64, 0.34), vec3(0.72, 0.88, 0.54), lrnd);
                col = mix(col, leafCol, 0.88);
            }
        }
    }

    gl_FragColor = vec4(applyDither(col, cell, vec3(0.038, 0.062, 0.048)), 1.0);
}

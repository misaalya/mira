#version 120
// Mira wall-01 — "Obsidian Rain Alley" (Hyper-Optimized 1:1 Cell-FBO GLSL)
// Bold 3px 4x4 ordered-Bayer pixel-art dither, live 2-layer diagonal pixel rain
// catching acid-yellow lantern light, wet alley puddle ripples + raindrop splash
// rings strictly right of the bicycle, and drifting damp mist.
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

    // Gentle parallax + wet alley puddle ripples strictly right of bicycle (u > 0.34, v > 0.65)
    vec2 p = uv + vec2(uMouse.x * 0.003 * (0.4 + 0.6 * uv.y), uMouse.y * 0.0018);
    float puddle = sstep(0.65, 0.74, uv.y) * sstep(0.34, 0.40, uv.x) * sstep(0.86, 0.74, uv.x);
    if (puddle > 0.01) {
        float w1 = sin(uv.y * 135.0 - t * 3.8 + sin(uv.x * 36.0 + t * 1.4) * 1.4);
        float w2 = cos(uv.x * 72.0 + t * 2.5 - uv.y * 92.0);
        p.x += (w1 * 0.0018 + w2 * 0.0008) * puddle;
        p.y += w2 * 0.0006 * puddle;
    }

    vec2 tUV = clamp(p, 0.0, 1.0) * uCover.xy + uCover.zw;
    vec3 col = texture2D(uTex, tUV).rgb;

    // Subtle organic breathing flicker on the acid-yellow paper lantern & puddle reflection
    float flick = 0.045 * sin(t * 2.9) + 0.024 * sin(t * 7.3 + 1.1);
    vec2 vLan = (uv - vec2(0.445, 0.39)) * vec2(1.4, 1.0);
    float dLan2 = dot(vLan, vLan);
    vec2 vRefl = (uv - vec2(0.44, 0.84)) * vec2(2.2, 0.85);
    float warmMask = exp(-dLan2 * 18.0) + exp(-dot(vRefl, vRefl) * 10.0) * 0.55;
    col += vec3(0.86, 0.88, 0.24) * warmMask * flick;

    // Crisp raindrop splash pings popping on the wet puddle surface
    if (puddle > 0.25) {
        vec2 sid = floor(uv * vec2(28.0, 22.0));
        float phase = fract(t * 1.65 + hash12(sid + 41.0));
        if (phase < 0.28) {
            float sr = hash12(sid + floor(t * 1.65) * 7.3);
            if (sr > 0.78) {
                vec2 sc = (sid + 0.2 + 0.6 * vec2(hash12(sid + 5.1), hash12(sid + 9.7))) / vec2(28.0, 22.0);
                vec2 d = (uv - sc) * grid * vec2(0.55, 1.35);
                float r = length(d);
                float ring = abs(r - phase * 7.0);
                if (ring < 0.75) {
                    float fade = (1.0 - phase / 0.28) * puddle;
                    col += vec3(0.88, 0.92, 0.56) * fade * 0.48;
                }
            }
        }
    }

    // Drifting damp alley mist near mid-height
    float mist = (0.5 + 0.5 * sin(uv.x * 5.2 - t * 0.18 + cos(uv.y * 4.0) * 1.3))
               * sstep(0.22, 0.48, uv.y) * sstep(0.86, 0.55, uv.y);
    col += vec3(0.14, 0.16, 0.13) * mist * 0.065;

    // 2-layer crisp 1-cell-wide diagonal falling pixel rain catching lantern light
    float lanHalo = exp(-dLan2 * 7.5);
    float rainGlow = 0.28 + 1.35 * lanHalo;
    vec3 rcol = mix(vec3(0.58, 0.64, 0.72), vec3(0.94, 0.96, 0.50), lanHalo);
    for (int l = 0; l < 2; l++) {
        float lf = float(l);
        vec2 dim = l == 0 ? vec2(60.0, 10.5) : vec2(40.0, 7.0);
        float spd = l == 0 ? 1.50 : 2.15;
        vec2 ruv = vec2(uv.x + uv.y * 0.13, uv.y - t * spd) * dim;
        vec2 gid = floor(ruv);
        float rnd = hash12(gid + lf * 53.0);
        if (rnd > 0.65) {
            vec2 f = fract(ruv);
            float sx = 0.18 + 0.64 * hash12(gid + 17.3);
            float streak = sstep(0.11, 0.0, abs(f.x - sx)) * sstep(0.10, 0.45, f.y) * sstep(0.96, 0.68, f.y);
            col += rcol * streak * (l == 0 ? 0.11 : 0.17) * rainGlow;
        }
    }

    gl_FragColor = vec4(applyDither(col, cell, vec3(0.045, 0.052, 0.062)), 1.0);
}

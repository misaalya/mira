#version 120
// Matcha wall-02 — "Mossy Shrine Canal" (Hyper-Optimized 1:1 Cell-FBO GLSL)
// Bold 3px 4x4 ordered-Bayer pixel-art dither, cedar forest sunbeams, calm
// emerald canal ripples strictly left of the stone lantern pedestal, warm
// firebox glow, drifting sunlit cedar dust, and glowing forest fireflies (hotaru).
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

    // Emerald lily-pad canal ripples strictly left of the stone lantern pillar (u < 0.45, v > 0.62)
    float canalRight = mix(0.42, 0.47, sstep(1.0, 0.60, uv.y));
    float canal = sstep(0.58, 0.66, uv.y) * sstep(canalRight, canalRight - 0.04, uv.x) * sstep(0.10, 0.22, uv.x);
    if (canal > 0.01) {
        float w1 = sin(uv.y * 104.0 - t * 2.0 + cos(uv.x * 28.0 + t * 0.9) * 1.4);
        p.x += w1 * 0.0017 * canal;
        p.y += cos(uv.y * 58.0 + t * 1.4) * 0.0005 * canal;
    }

    vec2 tUV = clamp(p, 0.0, 1.0) * uCover.xy + uCover.zw;
    vec3 col = texture2D(uTex, tUV).rgb;

    // Volumetric sunlit forest mist shafts drifting gently through cedar canopy
    float diag = (uv.x - 0.56) * 0.85 - uv.y * 0.36;
    float beam = (0.5 + 0.5 * sin(diag * 18.0 - t * 0.45)) * exp(-diag * diag * 5.0) * sstep(0.86, 0.06, uv.y);
    col += vec3(0.68, 0.82, 0.58) * beam * 0.08;

    // Warm Kasuga stone lantern firebox gentle glow at (0.52, 0.56)
    vec2 vL = (uv - vec2(0.52, 0.56)) * vec2(1.5, 1.0);
    float flick = 0.048 * sin(t * 2.7) + 0.024 * sin(t * 6.4 + 0.8);
    col += vec3(0.92, 0.95, 0.62) * exp(-dot(vL, vL) * 28.0) * flick;

    // Layer 0: Sunlit cedar forest pollen & dust drifting downward through sunbeams
    vec2 dCoord = vec2(uv.x - t * 0.008 + sin(t * 0.5 + uv.y * 7.0) * 0.008, uv.y - t * 0.016);
    vec2 dgid = floor(dCoord * vec2(26.0, 15.0));
    float drnd = hash12(dgid + 43.0);
    if (drnd > 0.55) {
        vec2 dc = (dgid + 0.18 + 0.64 * vec2(hash12(dgid + 5.1), hash12(dgid + 13.7))) / vec2(26.0, 15.0);
        vec2 dpx = abs((dCoord - dc) * grid);
        if (dpx.x < 0.58 && dpx.y < 0.58) {
            float tw = 0.45 + 0.55 * sin(t * 2.1 + drnd * 14.0);
            col = mix(col, vec3(0.90, 0.96, 0.76), 0.58 * tw);
        }
    }

    // Layer 1: Luminous forest fireflies (hotaru) meandering upward around canal, torii & lantern
    vec2 fCoord = vec2(uv.x + sin(t * 0.45 + uv.y * 5.5) * 0.016, uv.y + t * 0.018);
    vec2 fgid = floor(fCoord * vec2(12.0, 8.0));
    float frnd = hash12(fgid + 89.0);
    if (frnd > 0.56) {
        vec2 fc = (fgid + 0.20 + 0.60 * vec2(hash12(fgid + 8.3), hash12(fgid + 19.1))) / vec2(12.0, 8.0);
        fc.x += sin(t * (1.2 + frnd * 1.4) + frnd * 6.28) * 0.014;
        fc.y += cos(t * (0.9 + frnd) + frnd * 4.5) * 0.010;
        vec2 fd = (fCoord - fc) * grid;
        float dist2 = dot(fd, fd);
        if (dist2 < 6.0) {
            float glow = 0.35 + 0.65 * (0.5 + 0.5 * sin(t * (2.6 + frnd * 2.4) + frnd * 16.0));
            float halo = exp(-dist2 * 0.55) * glow;
            col += vec3(0.86, 0.98, 0.42) * halo * 0.55;
            if (abs(fd.x) < 0.65 && abs(fd.y) < 0.65) {
                col = mix(col, vec3(0.98, 1.0, 0.70), 0.92 * glow);
            }
        }
    }

    gl_FragColor = vec4(applyDither(col, cell, vec3(0.038, 0.062, 0.048)), 1.0);
}

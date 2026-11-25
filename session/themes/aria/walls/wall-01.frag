#version 120
// Aria wall-01 — "Aurora Rooftop Observatory" (Hyper-Optimized 1:1 Cell-FBO GLSL)
// Bold 3px 4x4 ordered-Bayer pixel-art dither, undulating emerald-indigo aurora
// curtains, twinkling starfield, periodic diagonal pixel shooting star, 2-layer
// drifting aurora stardust motes, rooftop puddle shimmer, and lantern breath.
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

    // Rooftop rain puddle shimmer strictly below bench legs (v > 0.85, u in 0.45..0.75)
    float puddle = sstep(0.84, 0.90, uv.y) * sstep(0.42, 0.50, uv.x) * sstep(0.78, 0.68, uv.x);
    if (puddle > 0.01) {
        p.x += sin(uv.y * 120.0 - t * 2.2 + uv.x * 28.0) * 0.0015 * puddle;
    }

    vec2 tUV = clamp(p, 0.0, 1.0) * uCover.xy + uCover.zw;
    vec3 col = texture2D(uTex, tUV).rgb;

    // Live undulating aurora borealis curtain waves across upper-right sky
    float skyMask = sstep(0.44, 0.14, uv.y) * sstep(0.38, 0.54, uv.x);
    if (skyMask > 0.01) {
        float wave = sin(uv.x * 7.5 - t * 0.55 + sin(uv.x * 13.0 + t * 0.35) * 0.8);
        float dy = (uv.y - (0.22 + wave * 0.055)) * 14.0;
        float ribbon = exp(-dy * dy);
        float rays = 0.55 + 0.45 * sin(uv.x * 42.0 - t * 1.15 + uv.y * 10.0);
        vec3 aurCol = mix(vec3(0.28, 0.92, 0.78), vec3(0.56, 0.58, 1.0), sstep(0.18, 0.06, uv.y));
        col += aurCol * ribbon * rays * skyMask * 0.20;
    }

    // Crisp twinkling 1-cell pixel stars + periodic diagonal shooting star in upper-right sky
    if (uv.y < 0.42 && uv.x > 0.38) {
        vec2 sid = floor(uv * vec2(40.0, 22.0));
        float sr = hash12(sid + 91.0);
        if (sr > 0.80) {
            vec2 sc = (sid + 0.25 + 0.5 * vec2(hash12(sid + 4.1), hash12(sid + 8.9))) / vec2(40.0, 22.0);
            vec2 dpx = abs((uv - sc) * grid);
            if (dpx.x < 0.58 && dpx.y < 0.58) {
                float w = 0.5 + 0.5 * sin(t * (1.8 + sr * 3.0) + sr * 20.0);
                col += vec3(0.88, 0.96, 1.0) * (w * w * w) * 0.72 * sstep(0.42, 0.24, uv.y);
            }
        }
        // Subtle periodic diagonal shooting star gliding across aurora sky every ~6.5s
        float cyc = mod(t + 2.4, 6.5);
        if (cyc < 0.85) {
            float prog = cyc * 1.1765;
            vec2 head = vec2(0.50 + prog * 0.44, 0.05 + prog * 0.16);
            vec2 dir = normalize(vec2(0.44, 0.16));
            vec2 rel = uv - head;
            float along = dot(rel, dir);
            float perp = abs(rel.x * dir.y - rel.y * dir.x) * grid.y;
            if (along <= 0.004 && along > -0.095 && perp < 0.85) {
                float tail = sstep(-0.095, 0.0, along) * sin(prog * 3.14159);
                col = mix(col, vec3(0.92, 1.0, 0.98), tail * 0.90);
            }
        }
    }

    // 2-layer drifting aurora stardust & ice-crystal motes floating across night air
    for (int l = 0; l < 2; l++) {
        float lf = float(l);
        vec2 dim = l == 0 ? vec2(26.0, 15.0) : vec2(15.0, 9.0);
        vec2 aCoord = vec2(
            uv.x - t * (0.012 + lf * 0.009) + sin(t * (0.6 + lf * 0.3) + uv.y * 6.0) * 0.009,
            uv.y + t * (0.014 + lf * 0.012)
        );
        vec2 gid = floor(aCoord * dim);
        float rnd = hash12(gid + lf * 61.0 + 37.0);
        if (rnd > (l == 0 ? 0.56 : 0.62)) {
            vec2 c = (gid + 0.18 + 0.64 * vec2(hash12(gid + 2.9), hash12(gid + 13.3))) / dim;
            vec2 dpx = abs((aCoord - c) * grid);
            float rad = l == 0 ? 0.56 : 0.82;
            if (dpx.x < rad && dpx.y < rad) {
                float tw = 0.45 + 0.55 * sin(t * (2.2 + rnd * 2.4) + rnd * 14.0);
                vec3 mCol = mix(vec3(0.54, 0.96, 0.92), vec3(0.76, 0.80, 1.0), rnd);
                col = mix(col, mCol, (l == 0 ? 0.55 : 0.82) * tw);
            }
        }
    }

    // Bench lantern gentle breathing glow (0.635, 0.73)
    vec2 vLan = (uv - vec2(0.635, 0.73)) * vec2(1.6, 1.0);
    col += vec3(0.92, 0.96, 0.76) * exp(-dot(vLan, vLan) * 48.0) * (0.048 * sin(t * 2.5));

    gl_FragColor = vec4(applyDither(col, cell, vec3(0.038, 0.044, 0.082)), 1.0);
}

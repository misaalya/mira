#version 120
// Aria wall-02 — "Starlit Tidal Railway" (Hyper-Optimized 1:1 Cell-FBO GLSL)
// Bold 3px 4x4 ordered-Bayer pixel-art dither, starry tidal ocean ripples strictly
// outside the submerged stone causeway & steel rails, undulating aurora, periodic
// shooting star, and 2-layer drifting bioluminescent sea-mist stardust motes.
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

    // Exact perspective stone causeway mask:
    // Left edge: (0.00, 0.70) -> (0.615, 0.515); Right edge: (0.67, 1.00) -> (0.66, 0.515)
    float tL = clamp((0.70 - uv.y) * 5.4054, 0.0, 1.0);
    float leftEdge = mix(-0.02, 0.61, tL);
    float tR = clamp((1.00 - uv.y) * 2.0619, 0.0, 1.0);
    float rightEdge = mix(0.67, 0.66, tR);
    float onCauseway = sstep(leftEdge - 0.015, leftEdge + 0.015, uv.x)
                     * sstep(rightEdge + 0.025, rightEdge - 0.015, uv.x);
    float seaY = sstep(0.51, 0.55, uv.y);
    float leftSea = seaY * sstep(leftEdge, leftEdge - 0.03, uv.x);
    float rightSea = seaY * sstep(rightEdge, rightEdge + 0.03, uv.x);
    float sea = clamp((leftSea + rightSea) * (1.0 - onCauseway), 0.0, 1.0);

    if (sea > 0.01) {
        float depth = clamp((uv.y - 0.51) * 2.0408, 0.0, 1.0);
        float w1 = sin(uv.y * (165.0 - depth * 70.0) - t * 2.4 + cos(uv.x * 26.0 + t * 0.9) * 1.4);
        float w2 = cos(uv.y * 92.0 + t * 1.7 - uv.x * 40.0);
        p.x += (w1 * 0.0022 + w2 * 0.0009) * sea * (0.45 + 0.55 * depth);
        p.y += w2 * 0.0006 * sea;
    }

    vec2 tUV = clamp(p, 0.0, 1.0) * uCover.xy + uCover.zw;
    vec3 col = texture2D(uTex, tUV).rgb;

    // Fine 1-cell pixel starlight water sparkles ONLY on open tidal water
    if (sea > 0.35) {
        float lum = dot(col, vec3(0.25, 0.55, 0.20));
        float sp = hash12(vec2(cell.x, cell.y + floor(t * 2.2)));
        float spark = sstep(0.95, 0.995, sp) * sstep(0.24, 0.65, lum);
        col += vec3(0.76, 0.96, 1.0) * spark * sea * 0.42;
    }

    // Subtle aurora ribbon, twinkling stars & periodic shooting star across night sky (v < 0.45)
    if (uv.y < 0.45) {
        float wave = sin(uv.x * 6.0 - t * 0.45 + cos(uv.x * 11.0 + t * 0.3) * 0.7);
        float dy = (uv.y - (0.18 + wave * 0.045)) * 16.0;
        float ribbon = exp(-dy * dy);
        float rays = 0.55 + 0.45 * sin(uv.x * 36.0 - t * 0.95);
        col += vec3(0.32, 0.88, 0.80) * ribbon * rays * 0.14;

        vec2 sid = floor(uv * vec2(44.0, 22.0));
        float sr = hash12(sid + 109.0);
        if (sr > 0.80) {
            vec2 sc = (sid + 0.25 + 0.5 * vec2(hash12(sid + 2.9), hash12(sid + 14.1))) / vec2(44.0, 22.0);
            vec2 dpx = abs((uv - sc) * grid);
            if (dpx.x < 0.58 && dpx.y < 0.58) {
                float w = 0.5 + 0.5 * sin(t * (1.7 + sr * 2.8) + sr * 18.0);
                col += vec3(0.88, 0.96, 1.0) * (w * w * w) * 0.65 * sstep(0.44, 0.24, uv.y);
            }
        }
        float cyc = mod(t + 3.1, 7.5);
        if (cyc < 0.85) {
            float prog = cyc * 1.1765;
            vec2 head = vec2(0.16 + prog * 0.46, 0.06 + prog * 0.14);
            vec2 dir = normalize(vec2(0.46, 0.14));
            vec2 rel = uv - head;
            float along = dot(rel, dir);
            float perp = abs(rel.x * dir.y - rel.y * dir.x) * grid.y;
            if (along <= 0.004 && along > -0.10 && perp < 0.85) {
                float tail = sstep(-0.10, 0.0, along) * sin(prog * 3.14159);
                col = mix(col, vec3(0.90, 1.0, 0.98), tail * 0.88);
            }
        }
    }

    // 2-layer drifting bioluminescent sea-mist stardust & rising nocturnal ocean motes
    for (int l = 0; l < 2; l++) {
        float lf = float(l);
        vec2 dim = l == 0 ? vec2(26.0, 15.0) : vec2(14.0, 9.0);
        vec2 mCoord = vec2(
            uv.x - t * (0.014 + lf * 0.010) + sin(t * (0.65 + lf * 0.35) + uv.y * 6.0) * 0.010,
            uv.y + t * (0.016 + lf * 0.014)
        );
        vec2 gid = floor(mCoord * dim);
        float rnd = hash12(gid + lf * 83.0 + 29.0);
        if (rnd > (l == 0 ? 0.56 : 0.62)) {
            vec2 c = (gid + 0.18 + 0.64 * vec2(hash12(gid + 4.7), hash12(gid + 15.3))) / dim;
            vec2 dpx = abs((mCoord - c) * grid);
            float rad = l == 0 ? 0.58 : 0.85;
            if (dpx.x < rad && dpx.y < rad) {
                float tw = 0.45 + 0.55 * sin(t * (2.2 + rnd * 2.5) + rnd * 16.0);
                vec3 mCol = mix(vec3(0.56, 0.98, 0.94), vec3(0.86, 0.94, 1.0), rnd);
                col = mix(col, mCol, (l == 0 ? 0.58 : 0.84) * tw);
            }
        }
    }

    // Distant railway signal lamp gentle pulse at (0.642, 0.37)
    vec2 vSig = (uv - vec2(0.642, 0.37)) * vec2(1.8, 1.0);
    col += vec3(0.75, 1.0, 0.88) * exp(-dot(vSig, vSig) * 160.0) * (0.12 + 0.16 * sin(t * 2.2));

    gl_FragColor = vec4(applyDither(col, cell, vec3(0.038, 0.044, 0.082)), 1.0);
}

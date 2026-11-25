#version 120
// @mira-dirty: 0.42 0.00 0.90 1.00
// Mira wall-02 — "Monolith Reflecting Atrium" (Hyper-Optimized 1:1 Cell-FBO GLSL)
// Bold 3px 4x4 ordered-Bayer pixel-art dither, fine silver drizzle falling strictly
// inside the central skylight shaft (zero noise on dark concrete walls), expanding
// concentric perspective raindrop ripples on the mirrored canal floor, breathing
// skylight curtains, and subtle acid-yellow architectural slit pulse.
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

    // Perspective water canal mask: strictly right of colonnade edge & outside floor lantern
    float py = sstep(1.0, 0.57, uv.y);
    float leftEdge = mix(0.43, 0.575, py);
    float rightEdge = mix(0.87, 0.72, py);
    float lanternBox = sstep(0.74, 0.76, uv.x) * sstep(0.85, 0.82, uv.x) * sstep(0.71, 0.68, uv.y);
    float canal = sstep(0.56, 0.61, uv.y)
                * sstep(leftEdge, leftEdge + 0.025, uv.x)
                * sstep(rightEdge + 0.025, rightEdge, uv.x)
                * (1.0 - lanternBox);
    if (canal > 0.01) {
        float w1 = sin(uv.y * 125.0 - t * 2.2 + cos(uv.x * 28.0 + t * 0.9) * 1.3);
        float w2 = cos(uv.y * 78.0 + t * 1.6 + uv.x * 36.0);
        p.x += (w1 * 0.0018 + w2 * 0.0008) * canal;
        p.y += w2 * 0.0005 * canal;
    }

    vec2 tUV = clamp(p, 0.0, 1.0) * uCover.xy + uCover.zw;
    vec3 col = texture2D(uTex, tUV).rgb;

    // Breathing skylight curtains piercing strictly down the central atrium shaft
    if (uv.x > 0.42 && uv.x < 0.78 && uv.y < 0.88) {
        float diag = (uv.x - 0.58) * 0.92 - uv.y * 0.16;
        float shaft = exp(-diag * diag * 26.0) * sstep(0.86, 0.04, uv.y);
        float pulse = 0.5 + 0.5 * sin(diag * 28.0 - t * 0.65);
        col += vec3(0.50, 0.54, 0.60) * shaft * pulse * 0.10;
    }

    // Acid-yellow architectural slit & water reflection gentle breath
    float slitGlow = 0.0;
    if (abs(uv.x - 0.79) < 0.18 && uv.y > 0.14) {
        float dxSlit = (uv.x - 0.79) * 8.0;
        slitGlow = exp(-dxSlit * dxSlit) * sstep(0.14, 0.34, uv.y) * sstep(0.98, 0.72, uv.y);
        col += vec3(0.86, 0.90, 0.22) * slitGlow * (0.042 * sin(t * 2.1) + 0.02 * sin(t * 5.7));
    }

    // Fine silver drizzle falling STRICTLY inside the open central skylight shaft (u in 0.46..0.74)
    if (uv.x > 0.46 && uv.x < 0.74 && uv.y < 0.74) {
        float shaftMask = sstep(0.46, 0.51, uv.x) * sstep(0.74, 0.67, uv.x) * sstep(0.74, 0.58, uv.y);
        for (int l = 0; l < 2; l++) {
            float lf = float(l);
            vec2 ruv = vec2(uv.x + uv.y * 0.04, uv.y - t * (0.48 + lf * 0.26));
            vec2 dim = l == 0 ? vec2(74.0, 9.0) : vec2(52.0, 6.5);
            vec2 gid = floor(ruv * dim);
            float rnd = hash12(gid + lf * 47.0 + 13.0);
            if (rnd > (l == 0 ? 0.76 : 0.80)) {
                vec2 f = fract(ruv * dim);
                float cx = 0.25 + 0.50 * hash12(gid + 5.3);
                float dx = abs(f.x - cx) * (grid.x / dim.x);
                if (dx < 0.55) {
                    float streak = sstep(0.05, 0.32, f.y) * sstep(0.95, 0.72, f.y);
                    float a = (l == 0 ? 0.38 : 0.62) * streak * shaftMask;
                    col = mix(col, vec3(0.82, 0.86, 0.92), a);
                }
            }
        }
    }

    // Expanding concentric perspective raindrop ripples on the mirrored canal floor
    if (canal > 0.20) {
        vec2 rGrid = vec2(11.0, 9.5);
        vec2 cid = floor(uv * rGrid);
        float rr = hash12(cid + 93.0);
        if (rr > 0.44) {
            float cyc = fract(t * (0.42 + rr * 0.24) + rr * 7.3);
            vec2 center = (cid + 0.24 + 0.52 * vec2(hash12(cid + 4.1), hash12(cid + 17.7))) / rGrid;
            vec2 d = (uv - center) * vec2(1.0, 2.75);
            float dist = length(d);
            float rad = cyc * 0.044;
            float ringW = 0.0042;
            float ring = sstep(ringW, 0.0, abs(dist - rad)) * (1.0 - cyc) * sstep(0.0, 0.14, cyc);
            if (ring > 0.02) {
                vec3 rCol = mix(vec3(0.76, 0.80, 0.86), vec3(0.90, 0.92, 0.38), slitGlow * 0.7);
                col = mix(col, rCol, ring * canal * 0.72);
            }
        }
    }

    gl_FragColor = vec4(applyDither(col, cell, vec3(0.045, 0.052, 0.062)), 1.0);
}

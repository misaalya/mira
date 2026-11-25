#version 120
// Matcha wall-01 — "Sunlit Botanical Conservatory" (Hyper-Optimized 1:1 Cell-FBO GLSL)
// Bold 3px 4x4 ordered-Bayer pixel-art dither, Camera3D-calibrated SUN_DIR (0.6664, 0.7456)
// volumetric golden-sage sunbeam shimmer, strictly floor-puddle-masked serene micro-ripples
// (zero wobble on potting bench, terracotta pots, wrought-iron columns, or stone planter),
// ultra-subtle breeze breath on open-doorway garden foliage & hanging fern frond tips,
// gentle pendant brass lamp shimmer, sunlit conservatory pollen motes, and drifting pale-jade
// botanical petals/spores.
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

    vec2 p = uv + vec2(uMouse.x * 0.0026, uMouse.y * 0.0015);

    // 1. Ultra-subtle breeze breath strictly on distant open-doorway garden foliage
    //    (Camera3D doorway UV: u=[0.4896, 0.6121], v=[0.3418, 0.6411]; inset inside open jambs)
    float doorOpen = sstep(0.502, 0.525, uv.x) * sstep(0.602, 0.578, uv.x)
                   * sstep(0.360, 0.405, uv.y) * sstep(0.605, 0.545, uv.y);

    //    Plus subtle soft drift strictly on cascading hanging fern frond tips below solid bowls
    //    (Camera3D baskets: (0.4272, 0.2223), (0.4972, 0.3063), (0.5899, 0.2686), (0.5717, 0.0579), (0.2691, 0.0257))
    float frondMask = 0.0;
    vec2 db1 = (uv - vec2(0.4272, 0.2850)) * vec2(1.0, 1.15);
    frondMask = max(frondMask, exp(-dot(db1, db1) * 260.0));
    vec2 db2 = (uv - vec2(0.4972, 0.3420)) * vec2(1.0, 1.25);
    frondMask = max(frondMask, exp(-dot(db2, db2) * 520.0));
    vec2 db3 = (uv - vec2(0.5899, 0.3120)) * vec2(1.0, 1.20);
    frondMask = max(frondMask, exp(-dot(db3, db3) * 380.0));
    vec2 db4 = (uv - vec2(0.2691, 0.1350)) * vec2(0.85, 1.0);
    frondMask = max(frondMask, exp(-dot(db4, db4) * 145.0));
    // Exclude solid brass pendant lamp (0.5093, 0.1525) & wrought-iron arch crown
    vec2 dLamp = (uv - vec2(0.5093, 0.1525)) * vec2(1.0, 1.45);
    frondMask *= sstep(0.048, 0.088, length(dLamp));

    float foliageBreath = max(doorOpen, frondMask * 0.78);
    if (foliageBreath > 0.01) {
        float sway = sin(t * 1.12 + uv.y * 7.5 + uv.x * 5.4) * 0.0014
                   + sin(t * 2.15 - uv.x * 11.0) * 0.00045;
        p.x += sway * foliageBreath;
    }

    // 2. Serene conservatory floor watering-puddle micro-ripples strictly inset inside Camera3D puddle mask
    //    (Camera3D rows: v=0.66:[0.549,0.565], v=0.72:[0.452,0.593], v=0.78:[0.490,0.608],
    //     v=0.84:[0.477,0.645], v=0.90:[0.458,0.633], v=0.95:[0.345,0.621])
    float pudL = 0.535 - 0.045 * sstep(0.68, 0.86, uv.y) - 0.115 * sstep(0.89, 0.98, uv.y);
    float pudR = 0.575 + 0.045 * sstep(0.68, 0.85, uv.y) - 0.010 * sstep(0.88, 0.98, uv.y);
    float puddle = sstep(0.675, 0.725, uv.y)
                 * sstep(pudL, pudL + 0.032, uv.x)
                 * sstep(pudR, pudR - 0.032, uv.x);
    if (puddle > 0.01) {
        float rip = sin(uv.y * 102.0 - t * 1.55 + sin(uv.x * 26.0 + t * 0.65) * 1.15);
        p.x += rip * 0.00115 * puddle;
        p.y += cos(uv.x * 52.0 + t * 1.15) * 0.00038 * puddle;
    }

    vec2 tUV = clamp(p, 0.0, 1.0) * uCover.xy + uCover.zw;
    vec3 col = texture2D(uTex, tUV).rgb;

    // Subtle morning water-surface skylight shimmer strictly inside floor watering puddles
    if (puddle > 0.01) {
        float gl = 0.5 + 0.5 * sin(uv.y * 92.0 - t * 1.65 + uv.x * 18.0);
        float glintMask = sstep(0.66, 0.96, gl) * sstep(0.95, 0.70, uv.y);
        col += vec3(0.88, 0.92, 0.74) * glintMask * puddle * 0.038;
    }

    // 3. Delicate warm morning glow breath on the vintage brass pendant conservatory lamp (0.5093, 0.1525)
    float lampPulse = 0.5 + 0.5 * sin(t * 1.5708);
    float lampGlow = exp(-dot(dLamp, dLamp) * 520.0);
    col += vec3(1.00, 0.93, 0.66) * lampGlow * (0.018 + 0.014 * lampPulse);

    // 4. Soft, restrained SUN_DIR morning sunbeam shimmer (highlight-protected, zero washout!)
    vec2 fromSun = uv - vec2(-1.479, -3.263);
    vec2 sunRayDir = normalize(fromSun);
    float rayAxis = sunRayDir.x * 0.7456 - sunRayDir.y * 0.6664;
    float rayFlow = uv.x * 0.6664 + uv.y * 0.7456;

    float mistDrift = sin(rayFlow * 4.8 - t * 0.39 + sin(rayAxis * 32.0 + t * 0.20) * 0.50);
    float wBay = 0.5 + 0.5 * sin(rayAxis * 84.0 + mistDrift * 0.36 - t * 0.10);
    float wMull = 0.5 + 0.5 * sin(rayAxis * 172.0 - mistDrift * 0.22 + 0.7);
    float shafts = pow(wBay, 2.0) * (0.45 + 0.55 * pow(wMull, 1.4));
    float breath = 0.78 + 0.22 * sin(t * 1.5708 + rayAxis * 18.0);

    float rightEdge = sstep(0.64, 0.34, uv.x - uv.y * 0.28);
    float underBench = 1.0 - sstep(0.68, 0.84, uv.y) * sstep(0.38, 0.20, uv.x);
    float topBotEnv = sstep(0.92, 0.16, uv.y) * sstep(0.02, 0.12, uv.x + uv.y * 0.28);
    float baseLum = dot(col, vec3(0.299, 0.587, 0.114));
    float hlGuard = 1.0 - 0.65 * sstep(0.64, 0.90, baseLum);
    float litMask = sstep(0.14, 0.46, baseLum);
    float rayCorridor = rightEdge * underBench * topBotEnv * (0.34 + 0.66 * litMask);
    float rayHit = shafts * rayCorridor;

    vec3 rayTint = mix(vec3(0.99, 0.94, 0.66), vec3(0.88, 0.93, 0.60), sstep(0.18, 0.78, rayFlow));
    col += rayTint * rayHit * breath * hlGuard * 0.060;

    // 5. Layer 0: Gentle sunlit conservatory pollen motes
    vec2 mCoord = vec2(uv.x - t * 0.0095, uv.y - t * 0.0085 + sin(t * 0.58 + uv.x * 8.2) * 0.0065);
    vec2 gid = floor(mCoord * vec2(28.0, 16.0));
    float rnd = hash12(gid + 41.0);
    if (rnd > 0.58) {
        vec2 c = (gid + 0.18 + 0.64 * vec2(hash12(gid + 5.7), hash12(gid + 11.3))) / vec2(28.0, 16.0);
        vec2 dpx = abs((mCoord - c) * grid);
        if (dpx.x < 0.56 && dpx.y < 0.56) {
            float tw = 0.45 + 0.55 * sin(t * (1.5708 + floor(rnd * 2.0) * 1.5708) + rnd * 15.0);
            float sunBoost = 0.28 + 0.85 * clamp(rayHit * 1.45 + rayCorridor * 0.25, 0.0, 1.0);
            vec3 moteCol = mix(vec3(0.90, 0.94, 0.76), vec3(0.98, 0.96, 0.74), clamp(rayHit * 1.8, 0.0, 1.0));
            col = mix(col, moteCol, clamp(0.56 * tw * sunBoost, 0.0, 0.72));
        }
    }

    // Subtle sunlit pollen micro-glints inside active window shafts
    if (rayCorridor > 0.18) {
        vec2 fCoord = vec2(uv.x - t * 0.0065, uv.y - t * 0.0072 + cos(t * 0.72 + uv.y * 9.5) * 0.005);
        vec2 fgid = floor(fCoord * vec2(44.0, 25.0));
        float frnd = hash12(fgid + 127.0);
        if (frnd > 0.74) {
            vec2 fc = (fgid + 0.20 + 0.60 * vec2(hash12(fgid + 9.1), hash12(fgid + 19.7))) / vec2(44.0, 25.0);
            vec2 fd = abs((fCoord - fc) * grid);
            if (fd.x < 0.50 && fd.y < 0.50) {
                float ftw = pow(0.5 + 0.5 * sin(t * (1.5708 + floor(frnd * 2.0) * 1.5708) + frnd * 21.0), 2.0);
                float shaftCatch = clamp(rayHit * 1.7 + rayCorridor * 0.18, 0.0, 1.0);
                col = mix(col, vec3(0.98, 0.96, 0.76), clamp(0.48 * ftw * shaftCatch, 0.0, 0.64));
            }
        }
    }

    // 6. Layer 1: Drifting pale-jade tiny botanical petals & greenhouse fern spores
    vec2 sCoord = vec2(uv.x - t * 0.024, uv.y - t * 0.018);
    vec2 sgid = floor(sCoord * vec2(12.0, 7.5));
    float srnd = hash12(sgid + 89.0);
    if (srnd > 0.70) {
        vec2 sc = (sgid + 0.22 + 0.56 * vec2(hash12(sgid + 3.1), hash12(sgid + 17.4))) / vec2(12.0, 7.5);
        sc.x += sin(t * (1.15 + srnd) + srnd * 6.28) * 0.011;
        sc.y += cos(t * (0.95 + srnd) + srnd * 4.4) * 0.007;
        vec2 d = (sCoord - sc) * grid;
        if (dot(d, d) < 2.3) {
            float ang = t * (1.2 + srnd * 1.4) + srnd * 6.28;
            float ca = cos(ang), sa = sin(ang);
            vec2 rd = vec2(ca * d.x - sa * d.y, sa * d.x + ca * d.y);
            if (abs(rd.x) < 1.25 && abs(rd.y) < 0.54) {
                vec3 petalCol = mix(vec3(0.56, 0.78, 0.50), vec3(0.90, 0.94, 0.74), srnd);
                col = mix(col, petalCol, 0.82);
            }
        }
    }

    gl_FragColor = vec4(applyDither(col, cell, vec3(0.038, 0.062, 0.048)), 1.0);
}

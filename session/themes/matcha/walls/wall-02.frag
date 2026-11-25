#version 120
// Matcha wall-02 — "Misty Terraced Tea Valley & Cedar Watermill" (Hyper-Optimized 1:1 Cell-FBO GLSL)
// Exact Camera3D-calibrated 3px 4x4 ordered-Bayer shader:
//   - Flowing crystal mountain millrace ripples & waterwheel splash shimmer strictly inside
//     the calibrated Camera3D stream mask (excluding granite foot-slab bridge, footpath & baskets),
//   - Warm cozy tea-drying pavilion hearth-breath glow at front (0.1217, 0.4731) & side (0.4472, 0.4626) windows,
//   - Drifting luminous silver-mint unkai valley mist veils gliding across the right-hand terraced tea valley,
//   - SUN_DIR-aligned diagonal golden dawn sunbeams (UV slope du/dv = -0.5672),
//   - Floating golden morning pollen motes & tiny tumbling emerald-lime green tea leaves on the breeze.
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

    // Exact Camera3D crystal millrace stream mask:
    // Region A: mid/upper channel beside waterwheel & stone footpath (v in [0.605, 0.798])
    //   left bank:  (v=0.609->u=0.566, v=0.706->u=0.365, v=0.781->u=0.216)
    //   right curb: (v=0.611->u=0.658, v=0.712->u=0.527, v=0.790->u=0.422)
    float tv = clamp((uv.y - 0.608) / 0.182, 0.0, 1.0);
    float streamLeft = mix(0.566, 0.216, tv) + 0.014;
    float streamRight = mix(0.646, 0.414, tv) - 0.014;
    float streamUpper = sstep(0.602, 0.625, uv.y)
        * sstep(0.808, 0.772, uv.y)
        * sstep(streamLeft - 0.015, streamLeft + 0.022, uv.x)
        * sstep(streamRight + 0.012, streamRight - 0.022, uv.x)
        * sstep(0.628, 0.592, uv.x);

    // Exclude the foreground tea basket tops resting beside the stone bridge approach
    vec2 dBask = (uv - vec2(0.300, 0.805)) * vec2(1.0, 1.35);
    streamUpper *= sstep(0.042, 0.075, length(dBask));

    // Region B: foreground lower-left open crystal stream pool in front of granite foot-slab bridge
    float poolRight = mix(0.075, 0.132, clamp((uv.y - 0.875) / 0.125, 0.0, 1.0));
    float streamFore = sstep(0.872, 0.902, uv.y)
        * sstep(poolRight + 0.018, poolRight - 0.020, uv.x);

    float stream = max(streamUpper, streamFore);
    if (stream > 0.01) {
        // Flowing mountain current moving downstream (toward viewer: +v, -u)
        float flow = uv.y * 88.0 - uv.x * 38.0 - t * 2.55;
        float w1 = sin(flow + cos(uv.x * 28.0 + t * 1.15) * 1.25);
        float w2 = cos(uv.y * 54.0 + uv.x * 22.0 - t * 1.85);
        p.x += (w1 * 0.00145 + w2 * 0.00042) * stream;
        p.y += (w2 * 0.00052) * stream;
    }

    vec2 tUV = clamp(p, 0.0, 1.0) * uCover.xy + uCover.zw;
    vec3 col = texture2D(uTex, tUV).rgb;

    // Subtle churning aerated splash shimmer strictly around Camera3D waterwheel splash UV (0.3684, 0.7210)
    vec2 dSplash = (uv - vec2(0.3684, 0.7210)) * vec2(1.0, 2.35);
    float spDist = length(dSplash);
    if (spDist < 0.095) {
        float spWave = 0.55 + 0.45 * sin(spDist * 145.0 - t * 4.4 + sin(uv.x * 64.0 + t * 2.1));
        float spMask = exp(-spDist * spDist * 280.0) * sstep(0.675, 0.705, uv.y);
        col += vec3(0.86, 0.96, 0.92) * spMask * spWave * 0.085;
    }

    // Drifting luminous silver-mint unkai valley mist veils gliding softly across right-hand tea valley
    float mistY = 0.495 + 0.022 * sin(uv.x * 6.5 - t * 0.14);
    float mistBand = exp(-pow((uv.y - mistY) / 0.068, 2.0)) * sstep(0.42, 0.56, uv.x);
    float mistDrift = 0.55 + 0.45 * sin(uv.x * 9.0 - t * 0.22 + sin(uv.y * 14.0 + t * 0.11) * 0.7);
    col += vec3(0.86, 0.93, 0.84) * mistBand * mistDrift * 0.068;

    // Diagonal golden dawn sunbeams strictly aligned with Camera3D SUN_DIR (UV slope du/dv = -0.5672)
    float diag = (uv.x - 0.85) + 0.5672 * (uv.y - 0.12);
    float beam = (0.5 + 0.5 * sin(diag * 19.0 + t * 0.32))
        * exp(-pow((diag + 0.08) / 0.42, 2.0))
        * sstep(0.88, 0.06, uv.y)
        * sstep(0.24, 0.52, uv.x);
    col += vec3(0.96, 0.93, 0.68) * beam * 0.056;

    // Warm cozy hearth-breath glow at exact Camera3D cedar pavilion tea-drying windows!
    float hearth1 = 0.5 + 0.30 * sin(t * 1.85) + 0.20 * sin(t * 4.1 + 0.8);
    float hearth2 = 0.5 + 0.30 * sin(t * 2.05 + 1.5) + 0.20 * sin(t * 3.7 + 2.3);

    // Front porch glowing shoji window UV (0.1217, 0.4731)
    vec2 vWFront = (uv - vec2(0.1217, 0.4731)) * vec2(1.45, 1.0);
    col += vec3(0.98, 0.80, 0.38) * exp(-dot(vWFront, vWFront) * 44.0) * (0.036 + 0.034 * hearth1);

    // Side millrace glowing shoji window UV (0.4472, 0.4626) + subtle warm shimmer on water below
    vec2 vWSide = (uv - vec2(0.4472, 0.4626)) * vec2(1.65, 1.1);
    col += vec3(0.98, 0.82, 0.40) * exp(-dot(vWSide, vWSide) * 95.0) * (0.028 + 0.028 * hearth2);
    if (streamUpper > 0.01) {
        vec2 vRef = (uv - vec2(0.468, 0.680)) * vec2(5.2, 2.4);
        float rw = 0.60 + 0.40 * sin(uv.y * 86.0 - t * 2.4);
        col += vec3(0.96, 0.80, 0.36) * exp(-dot(vRef, vRef) * 4.5) * (0.032 + 0.032 * hearth2) * rw * streamUpper;
    }

    // Layer 0: Floating sunlit golden morning pollen motes drifting softly on the highland breeze
    vec2 mCoord = vec2(uv.x + t * 0.012 + sin(t * 0.42 + uv.y * 6.5) * 0.007, uv.y - t * 0.008);
    vec2 mgid = floor(mCoord * vec2(26.0, 15.0));
    float mrnd = hash12(mgid + 41.0);
    if (mrnd > 0.56) {
        vec2 mc = (mgid + 0.18 + 0.64 * vec2(hash12(mgid + 5.7), hash12(mgid + 13.3))) / vec2(26.0, 15.0);
        vec2 mpx = abs((mCoord - mc) * grid);
        if (mpx.x < 0.58 && mpx.y < 0.58) {
            float tw = 0.46 + 0.54 * sin(t * (2.0 + mrnd * 1.8) + mrnd * 14.0);
            col = mix(col, vec3(0.98, 0.97, 0.76), 0.58 * tw);
        }
    }

    // Layer 1: Tiny tumbling fresh emerald-chartreuse green tea leaves riding the valley morning breeze
    vec2 lCoord = vec2(uv.x + t * 0.032, uv.y - t * 0.014);
    vec2 lgid = floor(lCoord * vec2(12.0, 8.0));
    float lrnd = hash12(lgid + 83.0);
    if (lrnd > 0.68) {
        vec2 lc = (lgid + 0.20 + 0.60 * vec2(hash12(lgid + 3.1), hash12(lgid + 16.7))) / vec2(12.0, 8.0);
        lc.x += sin(t * (1.25 + lrnd) + lrnd * 6.28) * 0.012;
        lc.y += cos(t * (0.95 + lrnd) + lrnd * 4.2) * 0.009;
        vec2 ld = (lCoord - lc) * grid;
        if (dot(ld, ld) < 2.5) {
            float ang = t * (1.5 + lrnd * 1.4) + lrnd * 6.28;
            float ca = cos(ang), sa = sin(ang);
            vec2 rd = vec2(ca * ld.x - sa * ld.y, sa * ld.x + ca * ld.y);
            if (abs(rd.x) < 1.32 && abs(rd.y) < 0.62) {
                vec3 leafCol = mix(vec3(0.28, 0.52, 0.24), vec3(0.72, 0.85, 0.42), lrnd);
                col = mix(col, leafCol, 0.84);
            }
        }
    }

    gl_FragColor = vec4(applyDither(col, cell, vec3(0.038, 0.062, 0.048)), 1.0);
}

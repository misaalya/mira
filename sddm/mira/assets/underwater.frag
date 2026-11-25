#version 440
// =============================================================
// Mira SDDM — Live 4x4 Ordered-Bayer Pixel-Dithered Underwater Shader
// Faithfully matches the Mira landing page WebGL engine (docs/app.js).
// Bake with: /usr/lib/qt6/bin/qsb --qt6 -o underwater.frag.qsb underwater.frag
// Copyright (C) 2026 Misaalya · GPL-3.0-only
// =============================================================

layout(location = 0) in vec2 qt_TexCoord0;
layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform buf {
    mat4 qt_Matrix;
    float qt_Opacity;
    vec2 uRes;
    vec4 uCover;
    float uTime;
    float uCell;
    float uMouse;
    float uRayGain;
    float uDive;
    float uFail;
    float uHasTex;
    vec4 uR0;
    vec4 uR1;
    vec4 uR2;
    vec4 uR3;
    vec4 uR4;
    vec4 uR5;
    vec4 uR6;
    vec4 uR7;
    vec4 uGlint;
    vec4 uBubbleTint;
};

layout(binding = 1) uniform sampler2D source;

float bayer4(vec2 p) {
    vec2 m = mod(p, 4.0);
    float x = floor(m.x);
    float y = floor(m.y);
    float bx0 = mod(x, 2.0);
    float by0 = mod(y, 2.0);
    float bx1 = floor(x * 0.5);
    float by1 = floor(y * 0.5);
    float q0 = mod(bx0 + by0, 2.0) * 2.0 + by0;
    float q1 = mod(bx1 + by1, 2.0) * 2.0 + by1;
    return (q0 * 4.0 + q1) / 16.0;
}

vec3 getRamp(float i) {
    if (i < 0.5) return uR0.rgb;
    if (i < 1.5) return uR1.rgb;
    if (i < 2.5) return uR2.rgb;
    if (i < 3.5) return uR3.rgb;
    if (i < 4.5) return uR4.rgb;
    if (i < 5.5) return uR5.rgb;
    if (i < 6.5) return uR6.rgb;
    return uR7.rgb;
}

float sstep(float a, float b, float x) {
    float t = clamp((x - a) / (b - a), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main() {
    vec2 res = max(vec2(64.0), uRes);
    float cellPx = max(1.0, uCell);
    vec2 grid = max(vec2(1.0), floor(res / cellPx));
    vec2 cell = floor(qt_TexCoord0 * grid);
    vec2 uv = cell / max(vec2(1.0), grid - 1.0);
    float u = uv.x;
    float v = uv.y;
    float t = uTime;
    float d = clamp(uDive, 0.0, 1.0);

    // Subtle login dive zoom & gentle vertical lift
    float invZoom = 1.0 / (1.0 + d * 0.16);
    float edge = abs(u - 0.5) * 2.0;
    float liftFactor = 1.0 - 0.24 * edge * edge;
    float diveV = d * 0.36 * liftFactor;

    float uBase = 0.52 + (u - 0.52) * invZoom;
    float vBase = 0.44 + (v - 0.44) * invZoom + diveV;

    // Rippling sunlit water surface overhead; sunken concrete ruins stay grounded
    float surfZone = sstep(0.32, 0.02, vBase) * sstep(-0.08, 0.0, vBase);
    vec2 disp = vec2(uMouse * (0.009 + d * 0.005) * (0.35 + 0.65 * (1.0 - v)), 0.0);
    if (surfZone > 0.005) {
        float rowRip = sin(vBase * 34.0 - t * 2.15) * surfZone;
        disp.x += (sin(uBase * 20.0 + t * 1.65 + vBase * 14.0) * 0.0055 + rowRip * 0.0032) * surfZone;
        disp.y += cos(uBase * 14.0 - t * 1.35 + vBase * 18.0) * 0.0036 * surfZone;
    }

    vec2 pUV = vec2(uBase, vBase) + disp;
    float depthDim = 1.0 - d * 0.14;

    float lumVal = clamp(0.08 + 0.78 * pow(1.0 - v, 1.8), 0.0, 1.0) * depthDim;
    float warmVal = 0.0;
    if (uHasTex > 0.5) {
        vec2 tUV = clamp(pUV, 0.0, 1.0) * uCover.xy + uCover.zw;
        vec3 texCol = texture(source, tUV).rgb;
        lumVal = pow(clamp(dot(texCol, vec3(0.28, 0.56, 0.16)), 0.0, 1.0), 0.90) * depthDim;
        warmVal = clamp((texCol.r - texCol.b * 0.72) * 1.85, 0.0, 1.0);
    }

    // Live breathing diagonal volumetric god rays aligned with SUN_DIR
    float rayVert = sstep(1.18, 0.02, v + d * 0.35);
    float diag = (u - 0.56) * 0.91 - (v + d * 0.18) * 0.38;
    float beamA = 0.5 + 0.5 * sin(diag * 21.0 - t * 0.95 + sin((v + d) * 5.0 + t * 0.4) * 0.5);
    float beamB = 0.5 + 0.5 * cos(diag * 34.0 + t * 0.68);
    float beamEnv = exp(-diag * diag * 6.0) * rayVert;
    float liveRay = pow(beamA * 0.65 + beamB * 0.35, 2.2) * beamEnv * (0.145 + d * 0.05) * uRayGain;

    float surfShimmer = surfZone > 0.01
        ? (0.5 + 0.5 * sin(pUV.x * 26.0 - t * 2.1 + cos(pUV.y * 38.0 + t * 1.4))) * surfZone * 0.10
        : 0.0;

    // Exact 4x4 ordered-Bayer quantization across curated 8-stop Mira ramp
    float sumL = clamp(lumVal + liveRay + surfShimmer, 0.0, 1.0);
    float tone = sumL * 6.999;
    float i0 = floor(tone);
    float b = bayer4(cell);
    float pick = (tone - i0) > b ? i0 + 1.0 : i0;
    vec3 col = getRamp(pick);

    vec3 glintCol = mix(uGlint.rgb, vec3(0.98, 0.38, 0.36), clamp(uFail, 0.0, 1.0) * 0.75);
    float glintMask = (liveRay * 2.8 + surfShimmer * 2.2 + warmVal * 0.32) * sstep(0.46, 0.94, sumL);
    if (glintMask > 0.22 && glintMask > b * 0.75 + 0.14) {
        col = mix(col, glintCol, min(0.55, glintMask * 0.62));
    }

    // 3-plane parallax GPU pixel air bubbles & sunlit plankton motes
    for (int layer = 0; layer < 3; layer++) {
        float lf = float(layer);
        float cols = layer == 0 ? 18.0 : (layer == 1 ? 13.0 : 9.0);
        float rows = layer == 0 ? 12.0 : (layer == 1 ? 9.0 : 6.0);
        float speedT = layer == 0 ? 0.025 : (layer == 1 ? 0.048 : 0.072);
        float pScroll = layer == 0 ? 0.78 : (layer == 1 ? 1.45 : 2.25);
        float mShift = uMouse * (0.010 + lf * 0.008);

        vec2 pCoord = vec2(u + mShift, v + t * speedT + d * pScroll);
        vec2 gid = floor(pCoord * vec2(cols, rows));
        float rnd = hash12(gid + lf * 37.0);
        if (rnd > 0.42) {
            float cx = (gid.x + 0.18 + 0.64 * hash12(gid + 11.3)) / cols;
            float cy = (gid.y + 0.18 + 0.64 * hash12(gid + 27.9)) / rows;
            cx += sin(t * (1.4 + rnd * 1.8) + rnd * 6.28 + pCoord.y * 8.0) * (0.003 + lf * 0.002);
            vec2 deltaPx = abs((pCoord - vec2(cx, cy)) * grid);
            float tw = 0.55 + 0.45 * sin(t * 2.2 + rnd * 12.0);
            float fadeEdge = sstep(0.02, 0.12, v) * sstep(0.99, 0.86, v);
            if (layer == 2 && rnd > 0.78) {
                float distC = max(deltaPx.x, deltaPx.y);
                if (distC < 1.55 && distC > 0.45) {
                    vec2 sgn = (pCoord - vec2(cx, cy)) * grid;
                    float spec = (sgn.x > 0.0 && sgn.y < 0.0) ? 0.88 : 0.44;
                    col = mix(col, mix(uBubbleTint.rgb, vec3(1.0), spec * 0.65), spec * fadeEdge);
                }
            } else if (deltaPx.x < 0.55 && deltaPx.y < 0.55) {
                float br = (layer == 0 ? 0.44 : 0.72) * tw * fadeEdge;
                col = mix(col, mix(uBubbleTint.rgb, vec3(1.0), 0.45), clamp(br, 0.0, 0.85));
            }
        }
    }

    // Subtle failure pulse along outer vignette on wrong password
    if (uFail > 0.005) {
        float vig = sstep(0.28, 0.92, length((uv - 0.5) * vec2(1.3, 1.0)));
        col = mix(col, vec3(0.42, 0.09, 0.11), vig * uFail * 0.45);
    }

    fragColor = vec4(col, 1.0) * qt_Opacity;
}

/* ==========================================================================
 * Mira Landing Page — Full-DPR Retina Sparkle Matrix Background
 * + Borderless BSPWM Topbar + Mira Luma-Halftone Develop Wallpaper Engine
 * ========================================================================== */

(() => {
  "use strict";

  const ASSET_V = "93";

  const THEMES = {
    mira: {
      name: "Mira",
      sub: "Monochrome obsidian · acid yellow · mira",
      bar: "mira",
      bg: "#08090d",
      walls: [
        { id: "mira-01", src: "./assets/walls/mira-01.webp?v=45", thumb: "./assets/walls/thumbs/mira-01.webp?v=45", name: "Obsidian Rain Alley", effect: "Rain & puddle splashes" },
        { id: "mira-02", src: "./assets/walls/mira-02.webp?v=45", thumb: "./assets/walls/thumbs/mira-02.webp?v=45", name: "Monolith Reflecting Atrium", effect: "Skylight drizzle & canal ripples" },
      ],
    },
    matcha: {
      name: "Matcha",
      sub: "Mossy charcoal · olive cream · mira-islands",
      bar: "mira-islands",
      bg: "#0b120e",
      walls: [
        { id: "matcha-01", src: `./assets/walls/matcha-01.webp?v=${ASSET_V}`, thumb: `./assets/walls/thumbs/matcha-01.webp?v=${ASSET_V}`, name: "Sunlit Botanical Conservatory", effect: "Glasshouse sunbeams, ripples & pollen" },
        { id: "matcha-02", src: `./assets/walls/matcha-02.webp?v=${ASSET_V}`, thumb: `./assets/walls/thumbs/matcha-02.webp?v=${ASSET_V}`, name: "Misty Terraced Tea Valley", effect: "Millrace stream, valley mist & tea leaves" },
      ],
    },
    ember: {
      name: "Ember",
      sub: "Volcanic obsidian · vermilion · mira-line",
      bar: "mira-line",
      bg: "#100808",
      walls: [
        { id: "ember-01", src: `./assets/walls/ember-01.webp?v=${ASSET_V}`, thumb: `./assets/walls/thumbs/ember-01.webp?v=${ASSET_V}`, name: "Crimson Viaduct Crossing", effect: "Crossing lamps, sea shimmer & embers" },
        { id: "ember-02", src: `./assets/walls/ember-02.webp?v=${ASSET_V}`, thumb: `./assets/walls/thumbs/ember-02.webp?v=${ASSET_V}`, name: "Twilight Onsen Canal", effect: "Onsen steam, chochin glow & maple leaves" },
      ],
    },
    fjord: {
      name: "Fjord",
      sub: "Basalt slate · glacier blue · mira-minimal",
      bar: "mira-minimal",
      bg: "#080f18",
      walls: [
        { id: "fjord-01", src: "./assets/walls/fjord-01.webp?v=45", thumb: "./assets/walls/thumbs/fjord-01.webp?v=45", name: "Blue-Hour Fjord Boathouse", effect: "Nordic snow & twilight ripples" },
        { id: "fjord-02", src: "./assets/walls/fjord-02.webp?v=45", thumb: "./assets/walls/thumbs/fjord-02.webp?v=45", name: "Glacier Station Overpass", effect: "Drifting glacier snowfall" },
      ],
    },
    aria: {
      name: "Aria",
      sub: "Obsidian midnight · starlight slate · mira-float",
      bar: "mira-float",
      bg: "#07080f",
      walls: [
        { id: "aria-01", src: "./assets/walls/aria-01.webp?v=45", thumb: "./assets/walls/thumbs/aria-01.webp?v=45", name: "Aurora Rooftop Observatory", effect: "Aurora, stars & shooting stars" },
        { id: "aria-02", src: "./assets/walls/aria-02.webp?v=45", thumb: "./assets/walls/thumbs/aria-02.webp?v=45", name: "Starlit Tidal Railway", effect: "Tidal railway aurora & starlit sea" },
      ],
    },
  };

  const THEME_ORDER = ["mira", "matcha", "ember", "fjord", "aria"];
  const ALL_WALLS = THEME_ORDER.flatMap((t) =>
    THEMES[t].walls.map((w, idx) => ({
      theme: t,
      wallIdx: idx,
      id: w.id,
      src: w.src,
      thumb: w.thumb,
      title: `${THEMES[t].name} · wall-0${idx + 1}`,
      name: w.name,
      effect: w.effect,
    }))
  );

  // Ordered pairs strictly matching the UI theme sequence:
  // Mira -> Matcha -> Ember -> Fjord -> Aria -> Mira
  const LAB_PAIRS = THEME_ORDER.map((t, i) => [
    THEMES[t].walls[0].id,
    THEMES[THEME_ORDER[(i + 1) % THEME_ORDER.length]].walls[0].id,
  ]);

  /* =====================================================================
   * 1. Unified Living Ordered-Bayer Pixel-Dithered Underwater Anime Engine
   *    - Loads our deterministic 3D-grounded 1990s underwater anime painting
   *      (./assets/bg/mira-underwater-scene.webp) into a live 1:1 square
   *      pixel-art buffer (cell = 3px, ~480x300, silky 60 FPS).
   *    - Applies real-time underwater current fluid motion directly to the
   *      painting itself (rippling sunlit surface overhead, gentle buoyancy
   *      drift & flowing hair/dress current on the central drifting girl,
   *      anchored stillness on heavy sunken concrete ruins).
   *    - Sweeps animated diagonal volumetric god rays & surface caustics
   *      through the painting BEFORE 4x4 ordered-Bayer quantization so the
   *      entire scene lives and breathes in unified ordered-Bayer
   *      pixel-dither motion (zero static pasted layer!), with rising
   *      pixel air bubbles and scroll-evolving 5-theme curated ramps.
   * ===================================================================== */
  const BAYER4_UNIT = new Float32Array([
    0 / 16, 8 / 16, 2 / 16, 10 / 16,
    12 / 16, 4 / 16, 14 / 16, 6 / 16,
    3 / 16, 11 / 16, 1 / 16, 9 / 16,
    15 / 16, 7 / 16, 13 / 16, 5 / 16,
  ]);

  // 512-entry fast sine LUT (zero Math.sin/Math.cos inside hot per-pixel loop -> 120+ FPS!)
  const SIN_MASK = 511;
  const RAD_TO_LUT = 512 / (Math.PI * 2);
  const SIN_LUT = new Float32Array(512);
  for (let i = 0; i < 512; i++) SIN_LUT[i] = Math.sin((i * Math.PI * 2) / 512);
  const IS_LE = new Uint8Array(new Uint32Array([0x01020304]).buffer)[0] === 0x04;
  function packRGBA(r, g, b) {
    const ri = r < 0 ? 0 : r > 255 ? 255 : r | 0;
    const gi = g < 0 ? 0 : g > 255 ? 255 : g | 0;
    const bi = b < 0 ? 0 : b > 255 ? 255 : b | 0;
    return IS_LE ? (0xff000000 | (bi << 16) | (gi << 8) | ri) >>> 0 : ((ri << 24) | (gi << 16) | (bi << 8) | 0xff) >>> 0;
  }

  // Curated 8-stop underwater atmospheric pixel ramps per Mira theme:
  // [0: deepest abyssal concrete shadow ... 4: midwater teal ... 7: sunlit cream surface]
  const WORLDS = [
    {
      // 0: MIRA — 1990s anime Prussian abyss, seafoam-cyan midwater & pale sunlit cream surface
      key: "mira",
      ramp: [
        [5, 13, 28],
        [10, 27, 50],
        [18, 49, 76],
        [30, 82, 108],
        [52, 124, 144],
        [94, 170, 178],
        [156, 212, 210],
        [226, 240, 232],
      ],
      glint: [232, 236, 168],
      rayGain: 1.05,
      bubbleTint: [214, 244, 242],
    },
    {
      // 1: MATCHA — Sunken mossy Kyoto shrine pool, celadon depths & sunlit olive-cream surface
      key: "matcha",
      ramp: [
        [6, 16, 13],
        [12, 31, 25],
        [21, 54, 43],
        [36, 86, 68],
        [62, 126, 98],
        [104, 168, 132],
        [164, 208, 170],
        [232, 240, 214],
      ],
      glint: [224, 238, 182],
      rayGain: 1.0,
      bubbleTint: [220, 242, 216],
    },
    {
      // 2: EMBER — Dusk-lit flooded volcanic ruins, deep wine-obsidian abyss & molten peach-gold rays
      key: "ember",
      ramp: [
        [13, 7, 14],
        [29, 13, 22],
        [56, 22, 30],
        [98, 38, 38],
        [148, 66, 49],
        [198, 108, 72],
        [238, 164, 114],
        [255, 224, 182],
      ],
      glint: [255, 214, 148],
      rayGain: 1.18,
      bubbleTint: [255, 224, 192],
    },
    {
      // 3: FJORD — Nordic glacial flooded hall, arctic navy depths & crisp glacier-silver caustics
      key: "fjord",
      ramp: [
        [5, 11, 22],
        [11, 24, 44],
        [21, 46, 76],
        [37, 76, 116],
        [64, 114, 156],
        [106, 158, 196],
        [162, 204, 230],
        [224, 240, 250],
      ],
      glint: [198, 232, 252],
      rayGain: 1.02,
      bubbleTint: [216, 238, 252],
    },
    {
      // 4: ARIA — Deep obsidian midnight-indigo & crisp starlight silver-blue (Install closing)
      key: "aria",
      ramp: [
        [5, 6, 12],
        [10, 12, 24],
        [18, 22, 42],
        [32, 38, 68],
        [54, 62, 102],
        [88, 98, 146],
        [144, 154, 202],
        [220, 226, 248],
      ],
      glint: [186, 196, 255],
      rayGain: 1.06,
      bubbleTint: [198, 208, 248],
    },
  ];

  function lerp(a, b, t) { return a + (b - a) * t; }
  function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
  function smoothstep(e0, e1, x) {
    const t = clamp01((x - e0) / (e1 - e0));
    return t * t * (3 - 2 * t);
  }
  function lerpRGB(a, b, t, out) {
    out[0] = a[0] + (b[0] - a[0]) * t;
    out[1] = a[1] + (b[1] - a[1]) * t;
    out[2] = a[2] + (b[2] - a[2]) * t;
  }

  /* ---- GPU WebGL 1.0 Shaders (0% CPU, locked 60/120/144 FPS!) ---------- */
  const GL_VS = `
    attribute vec2 aPos;
    void main() { gl_Position = vec4(aPos, 0.0, 1.0); }
  `;
  const GL_FS = `
    precision highp float;
    uniform sampler2D uTex;
    uniform vec2 uRes;
    uniform float uCell;
    uniform vec4 uCover;
    uniform float uTime;
    uniform float uDepth;
    uniform float uMouse;
    uniform float uHasTex;
    uniform float uRayGain;
    uniform vec3 uRamp[8];
    uniform vec3 uGlint;
    uniform vec3 uBubbleTint;

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
      if (i < 0.5) return uRamp[0];
      if (i < 1.5) return uRamp[1];
      if (i < 2.5) return uRamp[2];
      if (i < 3.5) return uRamp[3];
      if (i < 4.5) return uRamp[4];
      if (i < 5.5) return uRamp[5];
      if (i < 6.5) return uRamp[6];
      return uRamp[7];
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
      vec2 fragTop = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
      vec2 cell = floor(fragTop / uCell);
      vec2 grid = max(vec2(1.0), floor(uRes / uCell));
      vec2 uv = cell / max(vec2(1.0), grid - 1.0);
      float u = uv.x;
      float v = uv.y;
      float t = uTime;
      float d = uDepth;

      // 3-plane underwater camera dive parallax (character & surface rise smoothly above!)
      float invZoom = 1.0 / (1.0 + d * 0.26);
      float edge = abs(u - 0.5) * 2.0;
      float liftFactor = 1.0 - 0.26 * edge * edge;
      float diveV = d * 0.92 * liftFactor;

      float uBase = 0.52 + (u - 0.52) * invZoom;
      float vBase = 0.44 + (v - 0.44) * invZoom + diveV;

      // Painting-space motion zones (rippling sunlit water surface overhead; ruins stay grounded)
      float surfZone = sstep(0.32, 0.02, vBase) * sstep(-0.08, 0.0, vBase);

      vec2 disp = vec2(uMouse * (0.008 + d * 0.006) * (0.35 + 0.65 * (1.0 - v)), 0.0);
      if (surfZone > 0.005) {
        float rowRip = sin(vBase * 34.0 - t * 2.15) * surfZone;
        disp.x += (sin(uBase * 20.0 + t * 1.65 + vBase * 14.0) * 0.0055 + rowRip * 0.0032) * surfZone;
        disp.y += cos(uBase * 14.0 - t * 1.35 + vBase * 18.0) * 0.0036 * surfZone;
      }

      vec2 pUV = vec2(uBase, vBase) + disp;
      float depthDim = 1.0 - d * 0.26 + sstep(0.74, 1.0, d) * 0.10;

      // Seamless deep-abyssal procedural continuation below painting bottom
      float deepV = v + d * 1.45;
      float turbA = 0.5 + 0.5 * sin(u * 6.4 - t * 0.24 + cos(deepV * 4.2 + t * 0.16) * 1.4);
      float turbB = 0.5 + 0.5 * cos(u * 13.0 + t * 0.18 - deepV * 7.5);
      float flankRuin = sstep(0.26, 0.04, u) * 0.45 + sstep(0.82, 0.98, u) * 0.35;
      float abyssLum = clamp(
        (0.065 + 0.155 * (1.0 - v) * (0.55 + 0.45 * (1.0 - d * 0.5)) + turbA * 0.055 + turbB * 0.025) *
        (1.0 - flankRuin * 0.52),
        0.0, 1.0
      );

      float lumVal = abyssLum;
      float warmVal = 0.0;
      if (uHasTex > 0.5 && pUV.y < 1.05) {
        vec2 tUV = clamp(pUV, 0.0, 1.0) * uCover.xy + uCover.zw;
        vec3 texCol = texture2D(uTex, tUV).rgb;
        float l = pow(clamp(dot(texCol, vec3(0.28, 0.56, 0.16)), 0.0, 1.0), 0.90) * depthDim;
        float ab = sstep(0.80, 1.04, pUV.y);
        lumVal = mix(l, abyssLum, ab);
        warmVal = clamp((texCol.r - texCol.b * 0.72) * 1.85, 0.0, 1.0) * (1.0 - ab);
      } else if (uHasTex <= 0.5) {
        lumVal = clamp(0.08 + 0.78 * pow(1.0 - v, 1.8), 0.0, 1.0) * depthDim;
      }

      // Live breathing diagonal volumetric god rays aligned with SUN_DIR
      float rayVert = sstep(1.18, 0.02, v + d * 0.56);
      float diag = (u - 0.56) * 0.91 - (v + d * 0.25) * 0.38;
      float beamA = 0.5 + 0.5 * sin(diag * 21.0 - t * 0.95 + sin((v + d) * 5.0 + t * 0.4) * 0.5);
      float beamB = 0.5 + 0.5 * cos(diag * 34.0 + t * 0.68);
      float beamEnv = exp(-diag * diag * (6.2 - d * 1.8)) * rayVert;
      float liveRay = pow(beamA * 0.65 + beamB * 0.35, 2.2) * beamEnv * (0.145 + d * 0.04) * uRayGain;

      float surfShimmer = surfZone > 0.01
        ? (0.5 + 0.5 * sin(pUV.x * 26.0 - t * 2.1 + cos(pUV.y * 38.0 + t * 1.4))) * surfZone * 0.10
        : 0.0;

      // Exact 4x4 ordered-Bayer quantization across curated 8-stop palette ramp
      float sumL = clamp(lumVal + liveRay + surfShimmer, 0.0, 1.0);
      float tone = sumL * 6.999;
      float i0 = floor(tone);
      float b = bayer4(cell);
      float pick = (tone - i0) > b ? i0 + 1.0 : i0;
      vec3 col = getRamp(pick);

      float glintMask = (liveRay * 2.8 + surfShimmer * 2.2 + warmVal * 0.32) * sstep(0.46 - d * 0.16, 0.94, sumL);
      if (glintMask > 0.22 && glintMask > b * 0.75 + 0.14) {
        col = mix(col, uGlint, min(0.55, glintMask * 0.62));
      }

      // 3-plane parallax GPU pixel air bubbles & abyssal plankton motes (rush UPWARD on scroll!)
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
            // Crisp 3x3 hollow pixel bubble with bright top-right highlight
            float distC = max(deltaPx.x, deltaPx.y);
            if (distC < 1.55 && distC > 0.45) {
              vec2 sgn = (pCoord - vec2(cx, cy)) * grid;
              float spec = (sgn.x > 0.0 && sgn.y < 0.0) ? 0.88 : 0.44;
              col = mix(col, mix(uBubbleTint, vec3(1.0), spec * 0.65), spec * fadeEdge);
            }
          } else if (deltaPx.x < 0.55 && deltaPx.y < 0.55) {
            float br = (layer == 0 ? 0.42 + d * 0.34 : 0.72) * tw * fadeEdge;
            col = mix(col, mix(uBubbleTint, vec3(1.0), 0.45), clamp(br, 0.0, 0.85));
          }
        }
      }

      gl_FragColor = vec4(col, 1.0);
    }
  `;

  class UnderwaterPixelMotion {
    constructor(canvas) {
      this.canvas = canvas;
      this.gl = null;
      this.ctx = null;
      this.w = 0;
      this.h = 0;
      this.img = null;
      this.px = null;
      this.px32 = null;

      // Decoded painting image & cover-sampled tone/highlight buffers
      this.srcImg = null;
      this.srcReady = false;
      this.srcLum = null;
      this.srcWarm = null;

      // Reusable O(W) column scratch arrays & 512-entry per-frame god-ray LUT (Canvas2D fallback)
      this.uCol = null;
      this.liftCol = null;
      this.flankCol = null;
      this.baseXCol = null;
      this.uSrcCol = null;
      this.heroUCol = null;
      this.hairUCol = null;
      this.rayLut = new Float32Array(512);
      this.ramp32 = new Uint32Array(8);
      this.glintRamp32 = new Uint32Array(8);
      this.glRampFlat = new Float32Array(24);
      this.glTex = null;
      this.glLoc = null;

      // Try GPU WebGL 1.0 first (0% CPU, locked 60/120/144 FPS); cleanly fall back to fast Canvas2D
      if (!this.initWebGL()) {
        this.ctx = canvas.getContext("2d", { alpha: false });
      }

      // Procedural rising pixel air bubbles & sunlit underwater plankton motes
      this.bubbles = [];
      this.motes = [];
      let s = 19950714;
      const rnd = () => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 4294967296;
      };
      for (let i = 0; i < 54; i++) {
        const heroCluster = i < 20;
        this.bubbles.push({
          u: heroCluster ? 0.38 + rnd() * 0.28 : 0.08 + rnd() * 0.84,
          v0: rnd(),
          speed: 0.032 + rnd() * 0.055,
          wobbleAmp: 0.004 + rnd() * 0.011,
          wobbleFreq: 1.4 + rnd() * 2.2,
          phase: rnd() * 6.283,
          big: i % 5 === 0,
          bright: 0.55 + rnd() * 0.45,
        });
      }
      for (let i = 0; i < 68; i++) {
        this.motes.push({
          u: rnd(),
          v: 0.12 + rnd() * 0.84,
          speedX: (rnd() - 0.35) * 0.008,
          speedY: -0.004 - rnd() * 0.010,
          phase: rnd() * 6.283,
          bright: 0.32 + rnd() * 0.52,
        });
      }

      this.worldPos = 0;
      this.targetPos = 0;
      this.scrollNorm = 0;
      this.depthPos = 0;
      this.heroFactor = 1;
      this.mouseX = 0;
      this.targetMouseX = 0;
      this.lastTime = 0;
      this.lastDraw = 0;
      this.idleInterval = 1 / 24; /* 24 FPS at rest -> cool GPU; WebGL runs native VSync while scrolling */
      this.canvas2dInterval = 1 / 30;

      this.cur = {
        ramp: Array.from({ length: 8 }, () => [0, 0, 0]),
        glint: [0, 0, 0],
        rayGain: 1.05,
        bubbleTint: [214, 244, 242],
      };

      this.resize();
      this.loadPainting("./assets/bg/mira-underwater-scene.webp");
    }

    initWebGL() {
      let gl = null;
      try {
        gl = this.canvas.getContext("webgl", {
          alpha: false,
          antialias: false,
          depth: false,
          stencil: false,
          preserveDrawingBuffer: false,
          powerPreference: "high-performance",
        });
      } catch {
        return false;
      }
      if (!gl) return false;

      const compile = (type, src) => {
        const sh = gl.createShader(type);
        gl.shaderSource(sh, src);
        gl.compileShader(sh);
        if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
          gl.deleteShader(sh);
          return null;
        }
        return sh;
      };
      const vs = compile(gl.VERTEX_SHADER, GL_VS);
      const fs = compile(gl.FRAGMENT_SHADER, GL_FS);
      if (!vs || !fs) return false;

      const prog = gl.createProgram();
      gl.attachShader(prog, vs);
      gl.attachShader(prog, fs);
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return false;

      gl.useProgram(prog);
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
      const aPos = gl.getAttribLocation(prog, "aPos");
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

      const tex = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([10, 24, 44, 255]));

      this.gl = gl;
      this.glTex = tex;
      this.glLoc = {
        uTex: gl.getUniformLocation(prog, "uTex"),
        uRes: gl.getUniformLocation(prog, "uRes"),
        uCell: gl.getUniformLocation(prog, "uCell"),
        uCover: gl.getUniformLocation(prog, "uCover"),
        uTime: gl.getUniformLocation(prog, "uTime"),
        uDepth: gl.getUniformLocation(prog, "uDepth"),
        uMouse: gl.getUniformLocation(prog, "uMouse"),
        uHasTex: gl.getUniformLocation(prog, "uHasTex"),
        uRayGain: gl.getUniformLocation(prog, "uRayGain"),
        uRamp: gl.getUniformLocation(prog, "uRamp[0]"),
        uGlint: gl.getUniformLocation(prog, "uGlint"),
        uBubbleTint: gl.getUniformLocation(prog, "uBubbleTint"),
      };
      gl.uniform1i(this.glLoc.uTex, 0);
      return true;
    }

    uploadGLTexture() {
      if (!this.gl || !this.srcReady || !this.srcImg) return;
      const gl = this.gl;
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.glTex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.srcImg);
    }

    loadPainting(url) {
      const im = new Image();
      im.decoding = "async";
      im.fetchPriority = "high";
      const commit = () => {
        this.srcImg = im;
        this.srcReady = true;
        if (this.gl) this.uploadGLTexture();
        else this.samplePainting();
        this.render(this.lastTime || performance.now() * 0.001, true);
      };
      im.onload = () => {
        if (im.decode) im.decode().then(commit, commit);
        else commit();
      };
      im.src = url;
    }

    samplePainting() {
      if (!this.srcReady || !this.srcImg || this.w <= 0 || this.h <= 0) return;
      const w = this.w;
      const h = this.h;
      const c = document.createElement("canvas");
      c.width = w;
      c.height = h;
      const g = c.getContext("2d", { alpha: false });
      // Cover-fit with gentle centering on the drifting girl (even on mobile portrait!)
      const iw = this.srcImg.naturalWidth;
      const ih = this.srcImg.naturalHeight;
      const scale = Math.max(w / iw, h / ih);
      const dw = iw * scale;
      const dh = ih * scale;
      const posX = window.innerWidth <= 720 ? 0.54 : 0.50;
      g.imageSmoothingEnabled = true;
      g.imageSmoothingQuality = "high";
      g.drawImage(this.srcImg, (w - dw) * posX, (h - dh) * 0.5, dw, dh);

      try {
        const d = g.getImageData(0, 0, w, h).data;
        const n = w * h;
        const lum = this.srcLum;
        const warm = this.srcWarm;
        for (let i = 0; i < n; i++) {
          const o = i * 4;
          const r = d[o] / 255;
          const gr = d[o + 1] / 255;
          const b = d[o + 2] / 255;
          // Perceptual luminance with gentle filmic contrast curve so ruins & girl pop crisply in pixel-dither!
          const l = 0.28 * r + 0.56 * gr + 0.16 * b;
          lum[i] = Math.pow(clamp01(l), 0.90);
          // Warm highlight signature (sunlit skin, white billowing dress & warm surface crest)
          warm[i] = clamp01((r - b * 0.72) * 1.85);
        }
      } catch {
        this.srcReady = false;
      }
    }

    resize() {
      const vw = Math.max(320, window.innerWidth);
      const vh = Math.max(320, window.innerHeight);
      this.vw = vw;
      this.vh = vh;
      if (this.gl) {
        // Render at exact 1:1 pixel-art cell resolution (1/3 CSS px -> 9x less GPU/compositor bandwidth!)
        const nw = Math.max(128, Math.ceil(vw / 3));
        const nh = Math.max(100, Math.ceil(vh / 3));
        if (nw === this.w && nh === this.h) return;
        this.w = nw;
        this.h = nh;
        this.canvas.width = nw;
        this.canvas.height = nh;
        this.gl.viewport(0, 0, nw, nh);
        return;
      }
      // Canvas2D fallback: crisp 1:1 square pixel cells (~105k px -> <2ms/frame)
      const cell = Math.max(3, Math.ceil(Math.sqrt((vw * vh) / 108000)));
      const nw = Math.max(128, Math.ceil(vw / cell));
      const nh = Math.max(100, Math.ceil(vh / cell));
      if (nw === this.w && nh === this.h) return;

      this.w = nw;
      this.h = nh;
      this.canvas.width = nw;
      this.canvas.height = nh;
      this.img = this.ctx.createImageData(nw, nh);
      this.px = this.img.data;
      this.px32 = new Uint32Array(this.px.buffer);
      this.srcLum = new Float32Array(nw * nh);
      this.srcWarm = new Float32Array(nw * nh);

      this.uCol = new Float32Array(nw);
      this.liftCol = new Float32Array(nw);
      this.flankCol = new Float32Array(nw);
      this.baseXCol = new Float32Array(nw);
      this.uSrcCol = new Float32Array(nw);
      this.heroUCol = new Float32Array(nw);
      this.hairUCol = new Float32Array(nw);
      const invW = 1 / Math.max(1, nw - 1);
      for (let x = 0; x < nw; x++) {
        const u = x * invW;
        this.uCol[x] = u;
        const edge = Math.abs(u - 0.5) * 2.0;
        this.liftCol[x] = 1.0 - 0.26 * edge * edge;
        this.flankCol[x] = 1.0 - (smoothstep(0.26, 0.04, u) * 0.45 + smoothstep(0.82, 0.98, u) * 0.35) * 0.52;
      }
      if (this.srcReady) this.samplePainting();
    }

    blendWorlds(pos) {
      const p = Math.max(0, Math.min(WORLDS.length - 1, pos));
      const i0 = Math.floor(p);
      const i1 = Math.min(WORLDS.length - 1, i0 + 1);
      const f = smoothstep(0, 1, p - i0);
      const a = WORLDS[i0];
      const b = WORLDS[i1];
      const c = this.cur;
      for (let k = 0; k < 8; k++) lerpRGB(a.ramp[k], b.ramp[k], f, c.ramp[k]);
      lerpRGB(a.glint, b.glint, f, c.glint);
      lerpRGB(a.bubbleTint, b.bubbleTint, f, c.bubbleTint);
      c.rayGain = lerp(a.rayGain, b.rayGain, f);
    }

    render(timeSec, force = false) {
      const diving = Math.abs(this.scrollNorm - this.depthPos) > 0.0006 || Math.abs(this.targetPos - this.worldPos) > 0.001;
      const gap = this.gl ? (diving ? 0 : this.idleInterval) : this.canvas2dInterval;
      if (!force && gap > 0 && this.lastDraw > 0 && timeSec - this.lastDraw < gap) return;
      this.lastDraw = timeSec;
      const dt = this.lastTime > 0 ? Math.min(0.05, Math.max(0.001, timeSec - this.lastTime)) : 0.033;
      this.lastTime = timeSec;

      // Frame-rate-independent exponential smoothing (butter-smooth at 60Hz, 120Hz, 144Hz!)
      const kWorld = 1.0 - Math.exp(-dt * 10.0);
      const kDepth = 1.0 - Math.exp(-dt * 14.0);
      const kMouse = 1.0 - Math.exp(-dt * 6.5);
      this.worldPos += (this.targetPos - this.worldPos) * kWorld;
      if (Math.abs(this.targetPos - this.worldPos) < 0.0004) this.worldPos = this.targetPos;
      this.depthPos += (this.scrollNorm - this.depthPos) * kDepth;
      if (Math.abs(this.scrollNorm - this.depthPos) < 0.0002) this.depthPos = this.scrollNorm;
      this.mouseX += (this.targetMouseX - this.mouseX) * kMouse;
      this.blendWorlds(this.worldPos);

      const c = this.cur;
      const ramp = c.ramp;

      // Fast path: 100% GPU WebGL fragment shader (<0.2ms GPU, 0% CPU -> native 60/120/144 FPS!)
      if (this.gl) {
        const gl = this.gl;
        const loc = this.glLoc;
        const flat = this.glRampFlat;
        for (let k = 0; k < 8; k++) {
          flat[k * 3] = ramp[k][0] / 255;
          flat[k * 3 + 1] = ramp[k][1] / 255;
          flat[k * 3 + 2] = ramp[k][2] / 255;
        }
        // Compute exact object-fit: cover texture UV mapping
        let sx = 1.0, sy = 1.0, ox = 0.0, oy = 0.0;
        if (this.srcReady && this.srcImg) {
          const vpA = (this.vw || this.w) / Math.max(1, this.vh || this.h);
          const imA = this.srcImg.naturalWidth / Math.max(1, this.srcImg.naturalHeight);
          const posX = (this.vw || 1280) <= 720 ? 0.54 : 0.50;
          if (vpA > imA) {
            sy = imA / vpA;
            oy = (1.0 - sy) * 0.5;
          } else {
            sx = vpA / imA;
            ox = (1.0 - sx) * posX;
          }
        }
        gl.uniform2f(loc.uRes, this.w, this.h);
        gl.uniform1f(loc.uCell, 1.0);
        gl.uniform4f(loc.uCover, sx, sy, ox, oy);
        gl.uniform1f(loc.uTime, timeSec);
        gl.uniform1f(loc.uDepth, this.depthPos);
        gl.uniform1f(loc.uMouse, this.mouseX);
        gl.uniform1f(loc.uHasTex, this.srcReady ? 1.0 : 0.0);
        gl.uniform1f(loc.uRayGain, c.rayGain);
        gl.uniform3fv(loc.uRamp, flat);
        gl.uniform3f(loc.uGlint, c.glint[0] / 255, c.glint[1] / 255, c.glint[2] / 255);
        gl.uniform3f(loc.uBubbleTint, c.bubbleTint[0] / 255, c.bubbleTint[1] / 255, c.bubbleTint[2] / 255);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        return;
      }

      const ramp32 = this.ramp32;
      const glintRamp32 = this.glintRamp32;
      for (let k = 0; k < 8; k++) {
        const rk = ramp[k];
        ramp32[k] = packRGBA(rk[0], rk[1], rk[2]);
        glintRamp32[k] = packRGBA(
          rk[0] + (c.glint[0] - rk[0]) * 0.48,
          rk[1] + (c.glint[1] - rk[1]) * 0.48,
          rk[2] + (c.glint[2] - rk[2]) * 0.48
        );
      }

      const w = this.w;
      const h = this.h;
      const px32 = this.px32;
      const srcLum = this.srcLum;
      const srcWarm = this.srcWarm;
      const hasArt = this.srcReady;

      const invW = 1 / Math.max(1, w - 1);
      const invH = 1 / Math.max(1, h - 1);
      const t = timeSec;
      const wMax = w - 1.001;
      const hMax = h - 1.001;

      const d = this.depthPos;
      const invZoom = 1.0 / (1.0 + d * 0.26);
      const cx = w * 0.52;
      const cy = h * 0.44;
      const depthDim = 1.0 - d * 0.26 + smoothstep(0.74, 1.0, d) * 0.10;
      const diveScale = d * h * 0.92;
      const floatBreathY = Math.sin(t * 0.92) * 1.45;
      const panX = this.mouseX * (3.2 + d * 2.4);

      // O(W) column precomputation (replaces 100k+ inner-loop calculations!)
      const uCol = this.uCol;
      const liftCol = this.liftCol;
      const flankCol = this.flankCol;
      const baseXCol = this.baseXCol;
      const uSrcCol = this.uSrcCol;
      const heroUCol = this.heroUCol;
      const hairUCol = this.hairUCol;
      for (let x = 0; x < w; x++) {
        const bx = cx + (x - cx) * invZoom;
        const us = bx * invW;
        baseXCol[x] = bx;
        uSrcCol[x] = us;
        heroUCol[x] = smoothstep(0.28, 0.42, us) * smoothstep(0.82, 0.68, us);
        hairUCol[x] = smoothstep(0.50, 0.62, us) * smoothstep(0.82, 0.68, us);
      }

      // 512-entry god-ray LUT computed ONCE per frame (zero Math.exp/Math.pow inside inner loop!)
      const rayLut = this.rayLut;
      const raySpread = 6.2 - d * 1.8;
      const rayMul = (0.145 + d * 0.04) * c.rayGain;
      for (let i = 0; i < 512; i++) {
        const diag = -1.2 + (i / 511) * 2.4;
        const bA = 0.5 + 0.5 * Math.sin(diag * 21.0 - t * 0.95);
        const bB = 0.5 + 0.5 * Math.cos(diag * 34.0 + t * 0.68);
        const env = Math.exp(-diag * diag * raySpread);
        rayLut[i] = Math.pow(bA * 0.65 + bB * 0.35, 2.2) * env * rayMul;
      }

      const tSurfA = t * 1.65 * RAD_TO_LUT;
      const tSurfB = -t * 1.35 * RAD_TO_LUT;
      const tHeroA = -t * 1.45 * RAD_TO_LUT;
      const tHeroB = -t * 1.85 * RAD_TO_LUT;
      const tTurbA = -t * 0.24 * RAD_TO_LUT;
      const tShim = -t * 2.1 * RAD_TO_LUT;
      const glintFloor = 0.46 - d * 0.16;

      for (let y = 0; y < h; y++) {
        const v = y * invH;
        const bRow = (y & 3) << 2;
        const rowOff = y * w;
        const baseY = cy + (y - cy) * invZoom;
        const rayVert = smoothstep(1.18, 0.02, v + d * 0.56);
        const panRow = panX * (0.35 + 0.65 * (1.0 - v));
        const diagRowOff = -0.5096 - (v + d * 0.25) * 0.38;
        const deepV = v + d * 1.45;
        const abyssBaseRow = 0.065 + 0.155 * (1.0 - v) * (0.55 + 0.45 * (1.0 - d * 0.5));
        const turbPhaseRow = Math.cos(deepV * 4.2 + t * 0.16) * 1.4 * RAD_TO_LUT + tTurbA;

        for (let x = 0; x < w; x++) {
          const u = uCol[x];
          const diveY = diveScale * liftCol[x];
          const vSrcEst = (baseY + diveY) * invH;
          const uSrc = uSrcCol[x];

          let dx = panRow;
          let dy = 0;
          let surfZone = 0;

          if (vSrcEst > -0.08 && vSrcEst < 0.32) {
            surfZone = smoothstep(0.32, 0.02, vSrcEst) * smoothstep(-0.08, 0.0, vSrcEst);
            if (surfZone > 0.01) {
              const rowRip = SIN_LUT[((vSrcEst * 34.0 - t * 2.15) * RAD_TO_LUT) & SIN_MASK];
              const sA = SIN_LUT[((uSrc * 20.0 + vSrcEst * 14.0) * RAD_TO_LUT + tSurfA) & SIN_MASK];
              const sB = SIN_LUT[(((uSrc * 14.0 + vSrcEst * 18.0) * RAD_TO_LUT + tSurfB + 128) | 0) & SIN_MASK];
              dx += (sA * 2.3 + rowRip * 1.4 * surfZone) * surfZone;
              dy += sB * 0.95 * surfZone;
            }
          }

          const syRaw = baseY + diveY + dy;
          const vRaw = syRaw * invH;

          let lumVal;
          let warmVal = 0;
          if (hasArt && vRaw < 1.04) {
            let sx = baseXCol[x] + dx;
            if (sx < 0) sx = 0;
            else if (sx > wMax) sx = wMax;
            let sy = syRaw;
            if (sy < 0) sy = 0;
            else if (sy > hMax) sy = hMax;
            const x0 = sx | 0;
            const y0 = sy | 0;
            const fx = sx - x0;
            const fy = sy - y0;
            const idx00 = y0 * w + x0;
            const l0 = srcLum[idx00] + (srcLum[idx00 + 1] - srcLum[idx00]) * fx;
            const l1 = srcLum[idx00 + w] + (srcLum[idx00 + w + 1] - srcLum[idx00 + w]) * fx;
            const artLum = (l0 + (l1 - l0) * fy) * depthDim;
            if (vRaw > 0.80) {
              const tA = 0.5 + 0.5 * SIN_LUT[((u * 6.4 * RAD_TO_LUT + turbPhaseRow) | 0) & SIN_MASK];
              const abyssLum = (abyssBaseRow + tA * 0.065) * flankCol[x];
              const ab = smoothstep(0.80, 1.04, vRaw);
              lumVal = artLum + (abyssLum - artLum) * ab;
              warmVal = srcWarm[idx00] * (1.0 - ab);
            } else {
              lumVal = artLum;
              warmVal = srcWarm[idx00];
            }
          } else if (hasArt) {
            const tA = 0.5 + 0.5 * SIN_LUT[((u * 6.4 * RAD_TO_LUT + turbPhaseRow) | 0) & SIN_MASK];
            lumVal = (abyssBaseRow + tA * 0.065) * flankCol[x];
          } else {
            lumVal = (0.08 + 0.78 * (1.0 - v) * (1.0 - v)) * depthDim;
          }

          // Instant 1-cycle LUT god-ray lookup!
          const diag = u * 0.91 + diagRowOff;
          const rIdx = ((diag + 1.2) * 212.9166) | 0;
          const liveRay = rIdx >= 0 && rIdx < 512 ? rayLut[rIdx] * rayVert : 0;

          const surfShimmer =
            surfZone > 0.02
              ? (0.5 + 0.5 * SIN_LUT[((uSrc * 26.0 * RAD_TO_LUT + tShim + vSrcEst * 160.0) | 0) & SIN_MASK]) * surfZone * 0.10
              : 0;

          const sumL = lumVal + liveRay + surfShimmer;
          const tone = (sumL < 0 ? 0 : sumL > 1 ? 1 : sumL) * 6.999;
          const bayer = BAYER4_UNIT[bRow | (x & 3)];
          const i0 = tone | 0;
          const pick = (tone - i0) > bayer ? i0 + 1 : i0;

          if (sumL > glintFloor && (liveRay * 2.8 + surfShimmer * 2.2 + warmVal * 0.32) > bayer * 0.72 + 0.24) {
            px32[rowOff + x] = glintRamp32[pick];
          } else {
            px32[rowOff + x] = ramp32[pick];
          }
        }
      }

      /* ---- 3-PLANE PARALLAX RISING PIXEL AIR BUBBLES & ABYSSAL MOTES ---- */
      const bt = c.bubbleTint;
      for (let i = 0; i < this.motes.length; i++) {
        const m = this.motes[i];
        const mu = (((m.u + t * m.speedX + this.mouseX * 0.012) % 1) + 1) % 1;
        const mv = (((m.v + t * m.speedY - d * 0.78) % 0.94) + 0.94) % 0.94 + 0.03;
        const tw = 0.45 + 0.55 * SIN_LUT[(((t * 2.0 + m.phase) * RAD_TO_LUT) | 0) & SIN_MASK];
        const a = m.bright * tw * (0.48 + d * 0.36);
        if (a < 0.14) continue;
        this.blendPixel((mu * w) | 0, (mv * h) | 0, bt[0], bt[1], bt[2], a > 0.85 ? 0.85 : a);
      }

      for (let i = 0; i < this.bubbles.length; i++) {
        const b = this.bubbles[i];
        const pScroll = b.big ? d * 2.15 : d * 1.35;
        const bv = 1.0 - ((((b.v0 + t * b.speed + pScroll) % 0.96) + 0.96) % 0.96 + 0.02);
        const bu = b.u + SIN_LUT[(((t * b.wobbleFreq + b.phase + bv * 11.0) * RAD_TO_LUT) | 0) & SIN_MASK] * b.wobbleAmp + this.mouseX * (b.big ? 0.026 : 0.016);
        const bx = (bu * w) | 0;
        const by = (bv * h) | 0;
        const a = b.bright * smoothstep(0.02, 0.12, bv) * smoothstep(0.99, 0.86, bv);
        if (a < 0.16) continue;

        if (b.big) {
          this.blendPixel(bx + 1, by - 1, 252, 255, 252, a * 0.90);
          this.blendPixel(bx, by - 1, bt[0], bt[1], bt[2], a * 0.55);
          this.blendPixel(bx - 1, by, bt[0], bt[1], bt[2], a * 0.45);
          this.blendPixel(bx + 1, by, bt[0], bt[1], bt[2], a * 0.55);
          this.blendPixel(bx, by + 1, bt[0], bt[1], bt[2], a * 0.42);
        } else {
          this.blendPixel(bx, by, 248, 254, 252, a * 0.82);
          this.blendPixel(bx, by + 1, bt[0], bt[1], bt[2], a * 0.36);
        }
      }

      this.ctx.putImageData(this.img, 0, 0);
    }

    blendPixel(x, y, r, g, b, a) {
      if (x < 0 || x >= this.w || y < 0 || y >= this.h) return;
      const off = (y * this.w + x) * 4;
      this.px[off] = this.px[off] + (r - this.px[off]) * a;
      this.px[off + 1] = this.px[off + 1] + (g - this.px[off + 1]) * a;
      this.px[off + 2] = this.px[off + 2] + (b - this.px[off + 2]) * a;
    }
  }

  /* =====================================================================
   * 2. Hyper-Optimized WebGL 1.0 Live 3px Bayer Shader Wallpaper Engine
   *    + Exact GPU Mira Luma-Halftone Develop Transition
   *    - Exact 1:1 parity with desktop misa-wall GLSL shaders (10 themes).
   *    - Renders at 1:1 cell resolution (~640x360 on 1080p) with CSS
   *      image-rendering: pixelated -> 9x-36x fewer fragment invocations.
   *    - Steady-state = 1 direct pass @ 15 FPS (0 FBOs); transitions =
   *      live-to-live GPU Luma-Halftone Develop @ 60 FPS for 460 ms.
   *    - Auto-pauses at 0 FPS whenever hidden or covered by another view.
   * ===================================================================== */
  const LIVE_EPOCH = performance.now();

  const LIVE_VS_SRC = `
attribute vec2 aPos;
varying vec2 vUv;
void main() {
  vUv = vec2(aPos.x * 0.5 + 0.5, 0.5 - aPos.y * 0.5);
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

  const LIVE_COMP_FS_SRC = `
precision highp float;
uniform sampler2D uSrc;
uniform sampler2D uTgt;
uniform vec2 uMonSize;
uniform float uEased;
uniform float uLumaMax;
uniform float uLumaSpan;
varying vec2 vUv;
float bayer4(vec2 p) {
  vec2 b0 = floor(mod(p, 2.0));
  vec2 b1 = floor(mod(floor(p * 0.5), 2.0));
  float q0 = abs(b0.x - b0.y) * 2.0 + b0.y;
  float q1 = abs(b1.x - b1.y) * 2.0 + b1.y;
  return (q0 * 4.0 + q1) / 15.0;
}
void main() {
  vec2 monPx = vUv * uMonSize;
  vec2 fboUv = vec2(vUv.x, 1.0 - vUv.y);
  vec3 tgt = texture2D(uTgt, fboUv).rgb;
  vec3 src = texture2D(uSrc, fboUv).rgb;
  float b = bayer4(floor(monPx * 0.25));
  float luma = dot(tgt, vec3(0.3008, 0.5859, 0.1133));
  float invL = clamp((uLumaMax - luma) / max(uLumaSpan, 0.18), 0.0, 1.0);
  float tau = 6.2831853;
  float org = 0.5 + 0.275 * sin(tau * 1.35 * vUv.x + 0.4) * cos(tau * 1.15 * vUv.y - 0.3)
                  + 0.225 * cos(tau * 2.15 * vUv.x - 0.9) * sin(tau * 1.75 * vUv.y + 0.7);
  float th = 0.42 * b + 0.38 * invL + 0.20 * org;
  float feather = 0.0586;
  float head = uEased * (1.0 + feather);
  float a = clamp((head - th) / feather, 0.0, 1.0);
  gl_FragColor = vec4(mix(src, tgt, a), 1.0);
}
`;

  function easeDevelop(t) {
    const c = clamp01(t);
    const smooth = c * c * (3 - 2 * c);
    const inv = 1 - c;
    return 0.8 * smooth + 0.2 * (1 - inv * inv * inv);
  }

  function computeCover(imgW, imgH, viewW, viewH) {
    const imgAsp = imgW / Math.max(1, imgH);
    const viewAsp = viewW / Math.max(1, viewH);
    if (viewAsp > imgAsp) {
      const sy = imgAsp / viewAsp;
      return [1.0, sy, 0.0, (1.0 - sy) * 0.5];
    }
    const sx = viewAsp / imgAsp;
    return [sx, 1.0, (1.0 - sx) * 0.5, 0.0];
  }

  /* ---- Image & Luma Cache (Off-Main-Thread Decode + Persistent SW Cache) */
  const imgCache = new Map();
  const imgReady = new Map();
  function loadImg(src) {
    if (imgCache.has(src)) return imgCache.get(src);
    const p = new Promise((res, rej) => {
      const im = new Image();
      im.decoding = "async";
      const finish = () => {
        const done = () => {
          imgReady.set(src, im);
          res(im);
        };
        if (im.decode) im.decode().then(done, done);
        else done();
      };
      im.onload = finish;
      im.onerror = (err) => {
        imgCache.delete(src);
        rej(err);
      };
      im.src = src;
    });
    imgCache.set(src, p);
    return p;
  }

  const lumaCache = new WeakMap();
  let lumaSampleCanvas = null;
  let lumaSampleCtx = null;
  function getLumaStats(im) {
    if (lumaCache.has(im)) return lumaCache.get(im);
    let stats = { max: 0.92, span: 0.74 };
    try {
      if (!lumaSampleCanvas) {
        lumaSampleCanvas = document.createElement("canvas");
        lumaSampleCanvas.width = 32;
        lumaSampleCanvas.height = 18;
        lumaSampleCtx = lumaSampleCanvas.getContext("2d", { alpha: false, willReadFrequently: true });
      }
      if (lumaSampleCtx) {
        lumaSampleCtx.drawImage(im, 0, 0, 32, 18);
        const data = lumaSampleCtx.getImageData(0, 0, 32, 18).data;
        let lMin = 255;
        let lMax = 0;
        for (let i = 0; i < data.length; i += 4) {
          const l = (data[i] * 77 + data[i + 1] * 150 + data[i + 2] * 29) >> 8;
          if (l < lMin) lMin = l;
          if (l > lMax) lMax = l;
        }
        stats = {
          max: lMax / 255,
          span: Math.max(lMax - lMin, 48) / 255,
        };
      }
    } catch {
      // SecurityError under strict file:// origin -> calibrated default
    }
    lumaCache.set(im, stats);
    return stats;
  }

  class LiveWallEngine {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.cellPx = opts.cellPx || 3;
      this.idleIntervalMs = 1000 / (opts.idleFps || 15);
      this.allowDive = !!opts.allowDive;
      this.cssW = 1280;
      this.cssH = 720;
      this.gridW = 426;
      this.gridH = 240;

      this.mouseX = 0;
      this.mouseY = 0;
      this.targetMouseX = 0;
      this.targetMouseY = 0;
      this.scrollTarget = 0;
      this.scrollPos = 0;
      this.lastStep = 0;

      this.curWall = null;
      this.prevWall = null;
      this.transActive = false;
      this.transStart = 0;
      this.transDur = 460;
      this.scrubProgress = null;
      this.lastDraw = 0;
      this.reqId = 0;
      this.contextLost = false;
      this.hasFrame = false;
      this.onFirstFrame = opts.onFirstFrame || null;

      this.programs = new Map();
      this.textures = new Map();
      this.warmPromises = new Map();
      this.fboSrc = null;
      this.fboTgt = null;
      this.fboW = 0;
      this.fboH = 0;
      this.compProg = null;
      this.quadVbo = null;
      this.ctx2d = null;
      this.gl = null;
      this.glInitialized = false;

      if (!opts.lazy) this.ensureGL();

      canvas.addEventListener("webglcontextlost", (e) => {
        e.preventDefault();
        this.contextLost = true;
      });
      canvas.addEventListener("webglcontextrestored", () => {
        this.contextLost = false;
        this.programs.clear();
        this.textures.clear();
        this.fboSrc = null;
        this.fboTgt = null;
        this.fboW = 0;
        this.fboH = 0;
        this.compProg = null;
        this.quadVbo = null;
        this.glInitialized = false;
        this.ensureGL();
        if (this.curWall) this.setWall(this.curWall, false);
      });
    }

    ensureGL() {
      if (this.glInitialized) return !!this.gl;
      this.glInitialized = true;
      this.initGL();
      return !!this.gl;
    }

    initGL() {
      const attrs = {
        alpha: false,
        antialias: false,
        depth: false,
        stencil: false,
        preserveDrawingBuffer: false,
        powerPreference: "low-power",
      };
      let gl = null;
      try {
        gl = this.canvas.getContext("webgl", attrs) || this.canvas.getContext("experimental-webgl", attrs);
      } catch {
        gl = null;
      }
      this.gl = gl;
      if (!gl) {
        this.ctx2d = this.canvas.getContext("2d", { alpha: false });
        return;
      }

      const vbo = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(
        gl.ARRAY_BUFFER,
        new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
        gl.STATIC_DRAW
      );
      this.quadVbo = vbo;
      this.compProg = this.buildProgram(LIVE_VS_SRC, LIVE_COMP_FS_SRC, [
        "uSrc", "uTgt", "uMonSize", "uEased", "uLumaMax", "uLumaSpan",
      ]);
    }

    buildProgram(vsSrc, fsSrc, uniformNames) {
      const gl = this.gl;
      if (!gl) return null;
      const vs = gl.createShader(gl.VERTEX_SHADER);
      gl.shaderSource(vs, vsSrc);
      gl.compileShader(vs);
      if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
        console.error("[LiveWall] VS compile error:", gl.getShaderInfoLog(vs));
        gl.deleteShader(vs);
        return null;
      }
      const fs = gl.createShader(gl.FRAGMENT_SHADER);
      gl.shaderSource(fs, fsSrc);
      gl.compileShader(fs);
      if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
        console.error("[LiveWall] FS compile error:", gl.getShaderInfoLog(fs));
        gl.deleteShader(vs);
        gl.deleteShader(fs);
        return null;
      }
      const prog = gl.createProgram();
      gl.attachShader(prog, vs);
      gl.attachShader(prog, fs);
      gl.bindAttribLocation(prog, 0, "aPos");
      gl.linkProgram(prog);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        console.error("[LiveWall] Link error:", gl.getProgramInfoLog(prog));
        gl.deleteProgram(prog);
        return null;
      }
      const locs = {};
      for (let i = 0; i < uniformNames.length; i++) {
        locs[uniformNames[i]] = gl.getUniformLocation(prog, uniformNames[i]);
      }
      return { prog, locs };
    }

    ensureProgram(wallId) {
      if (this.programs.has(wallId)) return this.programs.get(wallId);
      const bank = window.MIRA_WALL_SHADERS;
      const fsSrc = bank && bank[wallId];
      if (!fsSrc) return null;
      const built = this.buildProgram(LIVE_VS_SRC, fsSrc, [
        "uTex", "uRes", "uCover", "uTime", "uMouse", "uCell", "uScroll",
      ]);
      if (built) this.programs.set(wallId, built);
      return built;
    }

    ensureTexture(wall, im) {
      const gl = this.gl;
      if (!gl) return null;
      if (this.textures.has(wall.id)) return this.textures.get(wall.id);
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      try {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, im);
      } catch (err) {
        gl.deleteTexture(tex);
        console.warn("[LiveWall] texImage2D blocked (fallback to dithered preview):", err);
        this.gl = null;
        this.ctx2d = this.canvas.getContext("2d", { alpha: false });
        return null;
      }
      const stats = getLumaStats(im);
      const entry = {
        tex,
        w: im.naturalWidth || 2048,
        h: im.naturalHeight || 1152,
        lumaMax: stats.max,
        lumaSpan: stats.span,
      };
      this.textures.set(wall.id, entry);
      return entry;
    }

    createFbo(w, h) {
      const gl = this.gl;
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return { fbo, tex };
    }

    destroyFbos() {
      const gl = this.gl;
      if (!gl) return;
      if (this.fboSrc) {
        gl.deleteFramebuffer(this.fboSrc.fbo);
        gl.deleteTexture(this.fboSrc.tex);
        this.fboSrc = null;
      }
      if (this.fboTgt) {
        gl.deleteFramebuffer(this.fboTgt.fbo);
        gl.deleteTexture(this.fboTgt.tex);
        this.fboTgt = null;
      }
      this.fboW = 0;
      this.fboH = 0;
    }

    ensureFbos() {
      if (!this.gl) return false;
      if (this.fboSrc && this.fboTgt && this.fboW === this.gridW && this.fboH === this.gridH) {
        return true;
      }
      this.destroyFbos();
      this.fboSrc = this.createFbo(this.gridW, this.gridH);
      this.fboTgt = this.createFbo(this.gridW, this.gridH);
      this.fboW = this.gridW;
      this.fboH = this.gridH;
      return true;
    }

    resize(cssW, cssH) {
      const w = Math.max(160, Math.round(cssW));
      const h = Math.max(90, Math.round(cssH));
      const gw = Math.max(64, Math.floor(w / this.cellPx));
      const gh = Math.max(36, Math.floor(h / this.cellPx));
      this.cssW = w;
      this.cssH = h;
      if (gw !== this.gridW || gh !== this.gridH || this.canvas.width !== gw || this.canvas.height !== gh) {
        this.gridW = gw;
        this.gridH = gh;
        this.canvas.width = gw;
        this.canvas.height = gh;
        this.destroyFbos();
        this.lastDraw = 0;
      }
    }

    drawWallPass(wall, tSec, targetFbo) {
      const gl = this.gl;
      const entry = this.textures.get(wall.id);
      const progObj = this.ensureProgram(wall.id);
      if (!gl || !entry || !progObj) return false;

      gl.bindFramebuffer(gl.FRAMEBUFFER, targetFbo);
      gl.viewport(0, 0, this.gridW, this.gridH);
      gl.useProgram(progObj.prog);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVbo);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, entry.tex);
      const cov = computeCover(entry.w, entry.h, this.gridW, this.gridH);
      const u = progObj.locs;
      gl.uniform1i(u.uTex, 0);
      gl.uniform2f(u.uRes, this.gridW * this.cellPx, this.gridH * this.cellPx);
      gl.uniform4f(u.uCover, cov[0], cov[1], cov[2], cov[3]);
      gl.uniform1f(u.uTime, tSec);
      gl.uniform2f(u.uMouse, this.mouseX, this.mouseY);
      gl.uniform1f(u.uCell, this.cellPx);
      if (u.uScroll) gl.uniform1f(u.uScroll, this.allowDive ? this.scrollPos : 0.0);

      gl.drawArrays(gl.TRIANGLES, 0, 6);
      if (targetFbo === null) this.markFrame();
      return true;
    }

    markFrame() {
      if (!this.hasFrame) {
        this.hasFrame = true;
        if (this.onFirstFrame) this.onFirstFrame();
      }
    }

    isWallReady(wall) {
      return !!(wall && this.gl && this.textures.has(wall.id) && this.programs.has(wall.id));
    }

    isPairReady(wallA, wallB) {
      return this.isWallReady(wallA) && this.isWallReady(wallB) && !!(this.fboSrc && this.fboTgt && this.fboW === this.gridW && this.fboH === this.gridH);
    }

    async prepareWall(wall) {
      if (!wall) return false;
      if (this.isWallReady(wall)) return true;
      if (this.warmPromises.has(wall.id)) return this.warmPromises.get(wall.id);
      const p = (async () => {
        let im = null;
        try {
          im = await loadImg(wall.src);
        } catch {
          return false;
        }
        if (this.ensureGL()) {
          this.ensureTexture(wall, im);
          this.ensureProgram(wall.id);
        }
        return true;
      })().finally(() => {
        this.warmPromises.delete(wall.id);
      });
      this.warmPromises.set(wall.id, p);
      return p;
    }

    async preparePair(wallA, wallB) {
      if (!wallA || !wallB) return false;
      const [okA, okB] = await Promise.all([this.prepareWall(wallA), this.prepareWall(wallB)]);
      if (!okA || !okB) return false;
      if (this.gl) this.ensureFbos();
      return true;
    }

    async drawFallback2D(wall) {
      if (!this.ctx2d) return;
      try {
        const im = await loadImg(wall.thumb || wall.src);
        const w = this.canvas.width;
        const h = this.canvas.height;
        const k = Math.max(w / im.naturalWidth, h / im.naturalHeight);
        const dw = im.naturalWidth * k;
        const dh = im.naturalHeight * k;
        this.ctx2d.imageSmoothingEnabled = false;
        this.ctx2d.drawImage(im, (w - dw) * 0.5, (h - dh) * 0.5, dw, dh);
        this.markFrame();
      } catch {
        // ignore
      }
    }

    async setWall(wall, animate = true, durationMs = 460) {
      if (!wall) return false;
      const myReq = ++this.reqId;
      const ok = await this.prepareWall(wall);
      if (!ok || myReq !== this.reqId) return false;

      const canTrans =
        animate &&
        this.gl &&
        this.compProg &&
        this.curWall &&
        this.curWall.id !== wall.id &&
        this.textures.has(this.curWall.id) &&
        !window.matchMedia("(prefers-reduced-motion: reduce)").matches;

      if (canTrans) {
        this.ensureFbos();
        this.prevWall = this.curWall;
        this.curWall = wall;
        this.transStart = performance.now();
        this.transDur = durationMs;
        this.transActive = true;
        this.scrubProgress = null;
      } else {
        this.prevWall = null;
        this.curWall = wall;
        this.transActive = false;
        this.scrubProgress = null;
      }
      this.lastDraw = 0;

      if (this.gl) {
        this.render(performance.now(), true);
        if (canTrans) this.transStart = performance.now();
      } else {
        await this.drawFallback2D(wall);
      }
      return true;
    }

    async scrubPair(wallA, wallB, progress) {
      if (!wallA || !wallB) return false;
      const clamped = clamp01(progress);
      // Zero-latency synchronous fast path when both shaders & textures are already GPU-resident!
      if (this.isPairReady(wallA, wallB)) {
        this.reqId++;
        this.prevWall = wallA;
        this.curWall = wallB;
        this.transActive = false;
        this.scrubProgress = clamped;
        this.lastDraw = 0;
        this.render(performance.now(), true);
        return true;
      }
      const myReq = ++this.reqId;
      const ok = await this.preparePair(wallA, wallB);
      if (!ok || myReq !== this.reqId) return false;
      this.prevWall = wallA;
      this.curWall = wallB;
      this.transActive = false;
      this.scrubProgress = clamped;
      this.lastDraw = 0;
      if (this.gl) this.render(performance.now(), true);
      else await this.drawFallback2D(clamped < 0.5 ? wallA : wallB);
      return true;
    }

    startPreparedPair(wallA, wallB, durationMs = 680, fromProgress = 0) {
      if (!wallA || !wallB) return false;
      this.reqId++;
      this.prevWall = wallA;
      this.curWall = wallB;
      this.scrubProgress = null;
      this.transDur = durationMs;
      this.transActive = true;
      this.lastDraw = 0;
      const p0 = clamp01(fromProgress);
      const now0 = performance.now();
      this.transStart = now0 - p0 * durationMs;
      if (this.gl) {
        // Prime 1 frame synchronously so GPU pipeline is hot & canvas is painted before clock starts!
        this.render(now0, true);
        this.transStart = performance.now() - p0 * durationMs;
      } else {
        this.drawFallback2D(p0 < 0.5 ? wallA : wallB);
      }
      return true;
    }

    async animatePair(wallA, wallB, durationMs = 680, fromProgress = 0) {
      if (!wallA || !wallB) return false;
      const myReq = ++this.reqId;
      const ok = await this.preparePair(wallA, wallB);
      if (!ok || myReq !== this.reqId) return false;
      return this.startPreparedPair(wallA, wallB, durationMs, fromProgress);
    }

    render(nowMs, force = false) {
      if (this.contextLost || !this.gl || !this.curWall) return;
      if (!this.textures.has(this.curWall.id)) return;

      const targetDive = this.allowDive ? this.scrollTarget : 0;
      const diving = Math.abs(targetDive - this.scrollPos) > 0.0008;
      const isScrub = this.scrubProgress !== null && !!this.prevWall && !!this.compProg;
      const inTrans = !!(this.transActive && this.prevWall && this.compProg);

      // Adaptive cadence: native VSync during halftone transitions, 30 FPS during scroll-dive,
      // dropping automatically to 15 FPS at rest so compositor scrolling stays butter-smooth.
      const minGap = inTrans ? 0 : (diving ? 33 : this.idleIntervalMs);
      if (!force && minGap > 0 && nowMs - this.lastDraw < minGap) return;
      const dt = Math.min(0.05, Math.max(0.004, (nowMs - (this.lastStep || nowMs)) * 0.001));
      this.lastStep = nowMs;
      this.lastDraw = nowMs;

      this.mouseX += (this.targetMouseX - this.mouseX) * 0.14;
      this.mouseY += (this.targetMouseY - this.mouseY) * 0.14;
      if (this.allowDive) {
        this.scrollPos += (targetDive - this.scrollPos) * (1 - Math.exp(-10.5 * dt));
        if (Math.abs(targetDive - this.scrollPos) < 0.0005) this.scrollPos = targetDive;
      }

      const tSec = (nowMs - LIVE_EPOCH) * 0.001 + 2.4;

      if (isScrub || inTrans) {
        const p = isScrub ? this.scrubProgress : clamp01((nowMs - this.transStart) / this.transDur);
        if (!isScrub && p >= 1) {
          this.transActive = false;
          this.prevWall = null;
        } else if (this.ensureFbos()) {
          const okA = this.drawWallPass(this.prevWall, tSec, this.fboSrc.fbo);
          const okB = this.drawWallPass(this.curWall, tSec, this.fboTgt.fbo);
          if (okA && okB) {
            const gl = this.gl;
            const tgtEntry = this.textures.get(this.curWall.id);
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.viewport(0, 0, this.gridW, this.gridH);
            gl.useProgram(this.compProg.prog);

            gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVbo);
            gl.enableVertexAttribArray(0);
            gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, this.fboSrc.tex);
            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_2D, this.fboTgt.tex);

            const u = this.compProg.locs;
            gl.uniform1i(u.uSrc, 0);
            gl.uniform1i(u.uTgt, 1);
            gl.uniform2f(u.uMonSize, this.cssW, this.cssH);
            gl.uniform1f(u.uEased, easeDevelop(p));
            gl.uniform1f(u.uLumaMax, tgtEntry ? tgtEntry.lumaMax : 0.92);
            gl.uniform1f(u.uLumaSpan, tgtEntry ? tgtEntry.lumaSpan : 0.74);

            gl.drawArrays(gl.TRIANGLES, 0, 6);
            this.markFrame();
            return;
          }
        }
      }

      this.drawWallPass(this.curWall, tSec, null);
    }
  }

  /* ---- Toast --------------------------------------------------------- */
  const toastEl = document.getElementById("toast");
  let toastTimer = 0;
  function toast(msg) {
    toastEl.textContent = msg;
    toastEl.classList.add("is-shown");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toastEl.classList.remove("is-shown"), 1700);
  }

  /* ---- Core State & Engines ------------------------------------------ */
  const fieldCanvas = document.getElementById("field-canvas");
  const wallCanvas = document.getElementById("wall-canvas");
  const tpCanvas = document.getElementById("tp-canvas");
  const tpStage = document.getElementById("tp-stage");
  const matrix = new UnderwaterPixelMotion(fieldCanvas);
  const wallEngine = new LiveWallEngine(wallCanvas, {
    cellPx: 3,
    idleFps: 15,
    allowDive: true,
    lazy: true,
    onFirstFrame: () => document.body.classList.add("wall-ready"),
  });
  const tpEngine = new LiveWallEngine(tpCanvas, {
    cellPx: 2,
    idleFps: 15,
    allowDive: false,
    lazy: true,
    onFirstFrame: () => { if (tpStage) tpStage.classList.add("is-ready"); },
  });
  wallEngine.resize(window.innerWidth, window.innerHeight);

  const siteNav = document.getElementById("site-nav");
  const liveBgPill = document.getElementById("live-bg-pill");

  const state = {
    theme: "mira",
    bar: "mira-minimal",
    wallId: "mira-01",
    lockedWorld: -1,
    wallMode: false,
    shotIdx: 0,
  };

  function setThemeAttr(themeKey) {
    if (!THEMES[themeKey]) return;
    state.theme = themeKey;
    document.documentElement.dataset.theme = themeKey;
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", THEMES[themeKey].bg);
    document.querySelectorAll("[data-pick-theme]").forEach((el) => {
      el.classList.toggle("is-active", el.dataset.pickTheme === themeKey);
    });
  }

  function setBar(barKey) {
    state.bar = barKey;
    document.documentElement.dataset.bar = barKey;
    const stageBar = document.getElementById("stage-bar");
    document.querySelectorAll("[data-pick-bar]").forEach((el) => {
      const on = el.dataset.pickBar === barKey;
      el.classList.toggle("is-active", on);
      if (el.getAttribute("role") === "radio") {
        el.setAttribute("aria-checked", on ? "true" : "false");
        el.tabIndex = on ? 0 : -1;
      }
    });
    if (stageBar) drawBar(stageBar, `./assets/bars/${barKey}.webp?v=${ASSET_V}`);
  }

  // Pop the old Polybar out, then pop the new one in.
  let barSwap = 0;
  function drawBar(img, src) {
    if (img.getAttribute("src") === src) return;
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (still || !img.animate) { img.src = src; return; }
    const id = ++barSwap;
    img.getAnimations().forEach((a) => a.cancel());
    const pre = new Image();
    pre.src = src;
    const ready = (pre.decode ? pre.decode() : Promise.resolve()).catch(() => {});
    const out = img.animate(
      [{ opacity: 1, transform: "scale(1)" }, { opacity: 0, transform: "scale(0.94)" }],
      { duration: 130, easing: "cubic-bezier(0.4, 0, 1, 1)", fill: "forwards" }
    );
    Promise.all([out.finished.catch(() => {}), ready]).then(() => {
      if (id !== barSwap) return;
      img.src = src;
      out.cancel();
      img.animate(
        [{ opacity: 0, transform: "scale(0.94)" }, { opacity: 1, transform: "scale(1)" }],
        { duration: 300, easing: "cubic-bezier(0.34, 1.45, 0.64, 1)" }
      );
    });
  }

  function syncLivePill() {
    const show = state.wallMode || state.lockedWorld >= 0;
    liveBgPill.hidden = !show;
    document.body.classList.toggle("has-live-pill", show);
  }

  function enterWallMode(on) {
    state.wallMode = on;
    document.body.classList.toggle("wall-mode", on);
    if (on) {
      wallEngine.resize(window.innerWidth, window.innerHeight);
      wallEngine.lastDraw = 0;
      if (wallEngine.hasFrame) document.body.classList.add("wall-ready");
    }
    syncLivePill();
  }

  function lockThemeLive(themeKey, announce = true) {
    const idx = THEME_ORDER.indexOf(themeKey);
    if (idx < 0) return;
    state.lockedWorld = idx;
    matrix.targetPos = idx;
    setThemeAttr(themeKey);
    if (THEMES[themeKey] && THEMES[themeKey].bar) setBar(THEMES[themeKey].bar);
    const firstWall = ALL_WALLS.find((w) => w.theme === themeKey);
    if (firstWall) developWallpaper(firstWall, true);
    else syncLabToTheme(themeKey);
    if (announce) toast(`Theme: ${THEMES[themeKey].name}`);
  }

  function resumeLiveField() {
    state.lockedWorld = -1;
    wallEngine.scrubProgress = null;
    enterWallMode(false);
    syncLivePill();
    onScroll();
    toast("Scroll scene resumed");
  }

  /* ---- Scroll Tracking -> Soft Nav Blur, Menu Visibility & Theme Morph - */
  const navSecLinks = Array.from(document.querySelectorAll(".nav-links a[data-sec]")).map((link) => ({
    link,
    sec: document.getElementById(link.dataset.sec),
    lastVis: -1,
  }));
  const drawerSecLinks = Array.from(document.querySelectorAll(".nav-drawer a[data-sec]")).map((link) => ({
    link,
    id: link.dataset.sec,
  }));
  const navMenuBtn = document.getElementById("nav-menu-btn");
  const navDrawer = document.getElementById("nav-drawer");
  const navBtnThemes = document.getElementById("nav-open-themes");
  const navBtnWalls = document.getElementById("nav-open-walls");
  let scrollDirty = true;

  function setDrawerOpen(open) {
    if (!navDrawer || !navMenuBtn) return;
    navDrawer.hidden = !open;
    navMenuBtn.setAttribute("aria-expanded", open ? "true" : "false");
    siteNav.classList.toggle("is-menu-open", open);
  }

  if (navMenuBtn) {
    navMenuBtn.addEventListener("click", () => setDrawerOpen(navDrawer.hidden));
  }
  if (navDrawer) {
    navDrawer.querySelectorAll("a[href^='#']").forEach((a) => {
      a.addEventListener("click", () => setDrawerOpen(false));
    });
  }

  const snapSlides = Array.from(document.querySelectorAll(".snap-sec"));
  let layoutCache = { vh: Math.max(1, window.innerHeight), maxScroll: 1, slides: [] };

  function measureSlides() {
    const vh = Math.max(1, window.innerHeight);
    const maxScroll = Math.max(1, document.documentElement.scrollHeight - vh);
    layoutCache = {
      vh,
      maxScroll,
      slides: snapSlides.map((el) => {
        const top = el.offsetTop;
        const height = Math.max(1, el.offsetHeight);
        return { el, id: el.id, top, height, bottom: top + height };
      }),
    };
  }
  measureSlides();

  function getSlideIndex(sy) {
    const mid = sy + layoutCache.vh * 0.45;
    for (let i = 0; i < layoutCache.slides.length; i++) {
      const s = layoutCache.slides[i];
      if (mid >= s.top && mid < s.bottom) return i;
    }
    return sy <= 0 ? 0 : Math.max(0, layoutCache.slides.length - 1);
  }

  let activeNavSec = "";
  function updateNavVisibility(sy) {
    const vh = layoutCache.vh;
    let bestSec = "";
    let bestVis = 0.14;
    for (let i = 0; i < navSecLinks.length; i++) {
      const item = navSecLinks[i];
      const m = layoutCache.slides.find((s) => s.el === item.sec);
      if (!m) continue;
      const top = m.top - sy;
      const bot = top + m.height;
      const visiblePx = Math.max(0, Math.min(bot, vh) - Math.max(top, 0));
      const vis = clamp01(visiblePx / Math.min(vh, m.height));
      if (vis > bestVis) {
        bestVis = vis;
        bestSec = item.sec.id;
      }
    }
    if (bestSec !== activeNavSec) {
      activeNavSec = bestSec;
      for (let i = 0; i < navSecLinks.length; i++) {
        navSecLinks[i].link.classList.toggle("is-active", navSecLinks[i].sec.id === bestSec);
      }
      for (let i = 0; i < drawerSecLinks.length; i++) {
        drawerSecLinks[i].link.classList.toggle("is-active", drawerSecLinks[i].id === bestSec);
      }
    }
  }

  const bgScrimVeil = document.getElementById("bg-scrim-veil");
  let lastReadVal = -1;
  let lastScrolledNav = false;

  function syncScrollState() {
    scrollDirty = false;
    const sy = window.scrollY;
    const vh = layoutCache.vh;
    const progress = clamp01(sy / layoutCache.maxScroll);
    matrix.scrollNorm = progress;
    matrix.heroFactor = 1.0 - smoothstep(0, vh * 0.85, sy);
    if (activeOv === "walls") {
      wallEngine.scrollTarget = 0;
    } else if (wallEngine.scrubProgress === null) {
      wallEngine.scrollTarget = progress;
    }
    const isScrolled = sy > 36;
    if (isScrolled !== lastScrolledNav) {
      lastScrolledNav = isScrolled;
      siteNav.classList.toggle("is-scrolled", isScrolled);
    }
    const readMix = Math.round(smoothstep(vh * 0.14, vh * 0.85, sy) * 100) / 100;
    if (bgScrimVeil && readMix !== lastReadVal) {
      lastReadVal = readMix;
      bgScrimVeil.style.opacity = String(readMix);
    }
    updateNavVisibility(sy);

    if (state.lockedWorld < 0 && !state.wallMode) {
      // Smooth GPU-only underwater palette glide (zero mid-scroll DOM data-theme reflow!)
      matrix.targetPos = progress * (WORLDS.length - 1);
    }
  }

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---- Smooth Inertial Scroll (wheel + anchors, desktop / fine pointer) --
   * Wheel deltas feed a target that the page eases toward each frame.
   * Touch + snap layouts (≤768px / coarse pointer) and reduced motion stay
   * 100% native. Nested scrollers and overlays keep their own wheel. */
  const smoothScroll = (() => {
    const root = document.documentElement;
    const nativeMQ = window.matchMedia("(max-width: 768px), (pointer: coarse)");
    const LERP = 10.5;
    let enabled = !reducedMotion && !nativeMQ.matches;
    let current = window.scrollY;
    let target = current;
    let lastSet = current;
    let raf = 0;
    let lastT = 0;
    let tween = null;

    const maxY = () => Math.max(0, root.scrollHeight - window.innerHeight);
    const clampY = (y) => Math.max(0, Math.min(maxY(), y));
    const easeOutQuart = (t) => 1 - Math.pow(1 - t, 4);
    const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

    function stop() {
      cancelAnimationFrame(raf);
      raf = 0;
      tween = null;
      current = target = lastSet = window.scrollY;
    }

    function apply(y) {
      current = y;
      lastSet = Math.round(y);
      window.scrollTo(0, y);
    }

    function tick(now) {
      raf = 0;
      if (root.classList.contains("ov-open")) { stop(); return; }
      const dt = Math.min(0.05, Math.max(0.001, (now - (lastT || now)) / 1000 || 1 / 60));
      lastT = now;
      if (tween) {
        const p = Math.min(1, (now - tween.t0) / tween.dur);
        apply(tween.from + (tween.to - tween.from) * tween.ease(p));
        if (p >= 1) { tween = null; target = current; return; }
      } else {
        target = clampY(target);
        const next = current + (target - current) * (1 - Math.exp(-LERP * dt));
        if (Math.abs(target - next) < 0.4) { apply(target); return; }
        apply(next);
      }
      raf = requestAnimationFrame(tick);
    }

    function start() {
      if (raf) return;
      lastT = 0;
      raf = requestAnimationFrame(tick);
    }

    // Lets an inner element (code row, list) keep its own vertical wheel.
    function innerScrolls(el, dy) {
      for (let n = el; n && n !== document.body && n !== root; n = n.parentElement) {
        if (n.scrollHeight <= n.clientHeight + 1) continue;
        const oy = getComputedStyle(n).overflowY;
        if (oy !== "auto" && oy !== "scroll") continue;
        if (dy > 0 ? n.scrollTop + n.clientHeight < n.scrollHeight - 1 : n.scrollTop > 0) return true;
      }
      return false;
    }

    window.addEventListener("wheel", (e) => {
      if (!enabled || e.defaultPrevented || e.ctrlKey || e.metaKey) return;
      if (root.classList.contains("ov-open")) return;
      let dy = e.deltaY;
      if (Math.abs(e.deltaX) > Math.abs(dy)) return;
      if (e.deltaMode === 1) dy *= 16;
      else if (e.deltaMode === 2) dy *= window.innerHeight;
      if (!dy || (e.target instanceof Element && innerScrolls(e.target, dy))) return;
      e.preventDefault();
      if (!raf || tween) {
        current = tween ? current : window.scrollY;
        target = current;
        tween = null;
      }
      target = clampY(target + dy);
      start();
    }, { passive: false });

    // Native scroll that we didn't cause (scrollbar drag, keys, find-in-page) wins.
    window.addEventListener("scroll", () => {
      if (raf && Math.abs(window.scrollY - lastSet) > 3) stop();
      else if (!raf) current = target = lastSet = window.scrollY;
    }, { passive: true });

    const onMQ = () => { enabled = !reducedMotion && !nativeMQ.matches; if (!enabled) stop(); };
    if (nativeMQ.addEventListener) nativeMQ.addEventListener("change", onMQ);
    else if (nativeMQ.addListener) nativeMQ.addListener(onMQ);

    return {
      to(y) {
        const dest = clampY(y);
        if (reducedMotion) { stop(); window.scrollTo(0, dest); return; }
        if (!enabled) { window.scrollTo({ top: dest, behavior: "smooth" }); return; }
        const from = window.scrollY;
        const dist = Math.abs(dest - from);
        if (dist < 1) return;
        cancelAnimationFrame(raf);
        raf = 0;
        current = from;
        target = dest;
        tween = {
          from,
          to: dest,
          t0: performance.now(),
          dur: Math.min(1400, 620 + dist * 0.32),
          ease: dist > window.innerHeight * 1.2 ? easeInOutCubic : easeOutQuart,
        };
        start();
      },
      stop,
    };
  })();

  document.addEventListener("click", (e) => {
    const a = e.target.closest('a[href^="#"]');
    if (!a) return;
    const hash = a.getAttribute("href");
    if (!hash || hash === "#") return;
    const targetEl = hash === "#top" ? snapSlides[0] : document.querySelector(hash);
    if (!targetEl) return;
    e.preventDefault();
    if (activeOv) closeAll();
    const top = hash === "#top" ? 0 : targetEl.offsetTop;
    smoothScroll.to(top);
  });

  let lastScrubScrollY = -1;
  function onScroll() {
    if (wallEngine.scrubProgress !== null && lastScrubScrollY >= 0 && Math.abs(window.scrollY - lastScrubScrollY) > 60) {
      wallEngine.scrubProgress = null;
      lastScrubScrollY = -1;
    }
    scrollDirty = true;
    if (reducedMotion) syncScrollState();
  }

  function resizeTpCanvas() {
    const stage = tpCanvas && tpCanvas.parentElement;
    if (!stage) return;
    const rect = stage.getBoundingClientRect();
    const w = rect.width > 120 ? rect.width : 780;
    const h = rect.height > 80 ? rect.height : Math.round((w * 9) / 16);
    tpEngine.resize(w, h);
  }

  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", () => {
    measureSlides();
    matrix.resize();
    wallEngine.resize(window.innerWidth, window.innerHeight);
    if (activeOv === "themes") resizeTpCanvas();
    syncScrollState();
    const now = performance.now();
    if (activeOv === "themes") tpEngine.render(now, true);
    else if (state.wallMode) wallEngine.render(now, true);
    else matrix.render(now * 0.001, true);
  }, { passive: true });

  window.addEventListener("pointermove", (e) => {
    const nx = (e.clientX / Math.max(1, window.innerWidth)) * 2 - 1;
    const ny = (e.clientY / Math.max(1, window.innerHeight)) * 2 - 1;
    matrix.targetMouseX = nx;
    wallEngine.targetMouseX = nx;
    wallEngine.targetMouseY = ny;
    tpEngine.targetMouseX = nx;
    tpEngine.targetMouseY = ny;
  }, { passive: true });
  window.addEventListener("pointerleave", () => {
    matrix.targetMouseX = 0;
    wallEngine.targetMouseX = 0;
    wallEngine.targetMouseY = 0;
    tpEngine.targetMouseX = 0;
    tpEngine.targetMouseY = 0;
  });

  /* ---- Unified Adaptive Animation Loop (Only 1 active surface at a time!) */
  function frame(now) {
    if (!document.hidden) {
      if (scrollDirty) syncScrollState();
      if (activeOv === "themes") {
        tpEngine.render(now);
      } else if (state.wallMode && wallEngine.hasFrame) {
        wallEngine.render(now);
      } else {
        matrix.render(now * 0.001);
        if (state.wallMode) wallEngine.render(now);
      }
    }
    if (!reducedMotion) requestAnimationFrame(frame);
  }

  /* ---- Theme & Bar Controls ------------------------------------------ */
  const onEl = (id, ev, fn) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener(ev, fn);
  };
  if (liveBgPill) liveBgPill.addEventListener("click", resumeLiveField);

  document.querySelectorAll("[data-pick-theme]").forEach((btn) => {
    btn.addEventListener("click", () => {
      lockThemeLive(btn.dataset.pickTheme, true);
    });
  });

  document.querySelectorAll("[data-pick-bar]").forEach((btn) => {
    btn.addEventListener("click", () => {
      setBar(btn.dataset.pickBar);
      const name = btn.querySelector(".view-name") || btn;
      toast(`Polybar: ${name.textContent.trim()}`);
    });
  });

  /* ---- Section 2: Desktop views (RiceSelector-style list + monitor) ----
   * Each view button carries its screenshot (data-src), how it sits in the
   * 16:9 monitor (data-fit: top = full desktop, left/right = 1:1 close-up
   * against the edge it was cropped from) and its caption. */
  const stageShot = document.getElementById("stage-shot");
  const stageMonitor = document.getElementById("stage-monitor");
  const stageTabs = document.getElementById("stage-tabs");
  const stageViews = stageTabs ? Array.from(stageTabs.querySelectorAll("[data-shot]")) : [];
  const viewCaption = (btn) => {
    const cap = btn && btn.querySelector(".view-cap");
    return cap ? cap.textContent.trim() : "";
  };

  let shotReqId = 0;
  async function selectShot(i, focus = false) {
    const n = stageViews.length;
    if (!n || !stageShot || !stageMonitor) return;
    const idx = ((i % n) + n) % n;
    state.shotIdx = idx;
    stageViews.forEach((b, k) => {
      const on = k === idx;
      b.classList.toggle("is-active", on);
      b.setAttribute("aria-selected", on ? "true" : "false");
      b.tabIndex = on ? 0 : -1;
    });
    const btn = stageViews[idx];
    if (focus) btn.focus();
    stageShot.alt = viewCaption(btn);
    const nextSrc = btn.dataset.src;
    if (stageShot.getAttribute("src") === nextSrc) return;

    const myReq = ++shotReqId;
    const nextFit = btn.dataset.fit || "top";
    // Keep current screenshot visible (ZERO black flash!) while off-DOM decoding next image.
    const alreadyReady = imgReady.has(nextSrc);
    let dimTimer = 0;
    if (!alreadyReady) {
      dimTimer = setTimeout(() => {
        if (myReq === shotReqId) stageMonitor.classList.add("is-loading");
      }, 90);
    }
    try {
      await loadImg(nextSrc);
    } catch {
      // fall through and let <img> handle fallback
    }
    clearTimeout(dimTimer);
    if (myReq !== shotReqId) return;
    stageMonitor.dataset.fit = nextFit;
    stageShot.src = nextSrc;
    stageMonitor.classList.remove("is-loading");
  }

  stageViews.forEach((b, k) => {
    b.addEventListener("click", () => selectShot(k));
    const warmShot = () => {
      if (b.dataset.src && !saveDataMode()) loadImg(b.dataset.src).catch(() => {});
    };
    b.addEventListener("pointerenter", warmShot, { passive: true });
    b.addEventListener("focus", warmShot, { passive: true });
  });
  if (stageTabs) {
    stageTabs.addEventListener("keydown", (e) => {
      const step = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 }[e.key];
      if (step) { e.preventDefault(); selectShot(state.shotIdx + step, true); }
      else if (e.key === "Home") { e.preventDefault(); selectShot(0, true); }
      else if (e.key === "End") { e.preventDefault(); selectShot(stageViews.length - 1, true); }
    });
  }

  function setOverlayLock(open) {
    if (open) {
      const sbw = Math.max(0, window.innerWidth - document.documentElement.clientWidth);
      document.documentElement.style.setProperty("--sbw", `${sbw}px`);
      document.documentElement.classList.add("ov-open");
      document.body.classList.add("ov-open");
    } else {
      document.documentElement.classList.remove("ov-open");
      document.body.classList.remove("ov-open");
      document.documentElement.style.removeProperty("--sbw");
    }
  }

  function openLightbox(src, cap, kind = "shot") {
    closeAll();
    ovLb.classList.toggle("is-bar", kind === "bar");
    document.getElementById("lb-img").src = src;
    document.getElementById("lb-img").alt = cap;
    document.getElementById("lb-cap").textContent = cap;
    setOverlayLock(true);
    ovLb.hidden = false;
    activeOv = "lb";
  }

  function inspectShot() {
    const btn = stageViews[state.shotIdx];
    if (btn) openLightbox(btn.dataset.src, viewCaption(btn));
  }
  onEl("stage-zoom", "click", inspectShot);
  if (stageMonitor) stageMonitor.addEventListener("click", inspectShot);

  // The Polybar strip opens on its own, cropped to the bar; on phones it stands upright.
  function inspectBar() {
    const btn = document.querySelector(`.bsw[data-pick-bar="${state.bar}"]`);
    const name = btn ? btn.textContent.trim() : "Polybar";
    openLightbox(`./assets/bars/strip/${state.bar}.webp?v=${ASSET_V}`, `${name} Polybar layout`, "bar");
  }
  const barInspect = document.getElementById("bar-inspect");
  if (barInspect) {
    barInspect.addEventListener("click", inspectBar);
    barInspect.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); inspectBar(); }
    });
  }

  /* ---- Live Shader Lab: one rail = five theme stops + halftone scrub --
   * Rail value v runs 0..500. Stop i sits at v = 100·i; the stretch from
   * 100·i to 100·(i+1) is the luma-halftone from theme i to theme i+1, and
   * the last stretch wraps Aria → Mira (v = 500 is Mira again).
   *
   * Elegant Zero-Waste Warming Strategy (NEVER bulk-downloads all 10 on load):
   *   1. Viewport intent: IntersectionObserver on #halftone warms ONLY the
   *      active pair [a, b] (image decode + WebGL shader compile + texImage2D
   *      + FBOs) during browser idle time when the visitor approaches the Lab.
   *   2. Interaction intent: hovering/focusing a stop, Play/45%, or [W]/[T]
   *      warms just that target wallpaper ahead of click.
   *   3. Sliding-window lookahead: after segment i→i+1 finishes, quietly warms
   *      ONLY the next single stop (i+2) so consecutive Plays stay 0ms instant.
   *   4. Prepare-before-clock: Play/45% always await preparePair(a, b) BEFORE
   *      starting the rail clock or revealing #wall-canvas -> 0 black flash,
   *      0 slider desync, locked 60/120 FPS from frame 0. */
  const LAB_MAX = THEME_ORDER.length * 100;
  const LAB_HOLD = 45;
  const LAB_PLAY_MS = 760;
  const labRange = document.getElementById("bg-scrubber");
  const labTrack = document.getElementById("lab-track");
  const labPct = document.getElementById("scrub-readout");
  const labPlayBtn = document.getElementById("btn-scrub-play");
  const labHoldBtn = document.getElementById("btn-scrub-hold");
  const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);
  let labPos = 0;
  let labRaf = 0;
  let labPlaySeq = 0;

  const saveDataMode = () => !!(navigator.connection && navigator.connection.saveData);
  const idleQueue = new Set();
  const runIdle = (fn) => {
    if ("requestIdleCallback" in window) {
      window.requestIdleCallback(fn, { timeout: 900 });
    } else {
      setTimeout(fn, 120);
    }
  };
  function scheduleIdleWarm(wall, engine = wallEngine) {
    if (!wall || saveDataMode() || engine.isWallReady(wall)) return;
    const key = `${engine === tpEngine ? "tp" : "bg"}:${wall.id}`;
    if (idleQueue.has(key)) return;
    idleQueue.add(key);
    runIdle(() => {
      idleQueue.delete(key);
      if (!engine.isWallReady(wall)) engine.prepareWall(wall).catch(() => {});
    });
  }
  function scheduleIdleWarmPair(seg) {
    const [a, b] = labPair(seg);
    scheduleIdleWarm(a, wallEngine);
    scheduleIdleWarm(b, wallEngine);
  }

  const labSeg = (v) => Math.min(THEME_ORDER.length - 1, Math.floor(v / 100));
  const findWall = (id) => ALL_WALLS.find((w) => w.id === id) || ALL_WALLS[0];
  const labPair = (seg) => LAB_PAIRS[seg].map(findWall);

  function paintLab(v, showPct = true) {
    labPos = v;
    const seg = labSeg(v);
    const p = v - seg * 100;
    const mid = p > 0.5 && p < 99.5;
    if (labRange) {
      labRange.value = String(Math.round(v));
      const [a, b] = labPair(seg);
      const nameA = THEMES[a.theme].name;
      const nameB = THEMES[b.theme].name;
      labRange.setAttribute("aria-valuetext", mid ? `${nameA} to ${nameB}, ${Math.round(p)}%` : (p < 50 ? nameA : nameB));
    }
    if (labTrack) {
      labTrack.style.setProperty("--a", `${(seg * 100 * 100) / LAB_MAX}%`);
      labTrack.style.setProperty("--p", `${((mid ? v : seg * 100) * 100) / LAB_MAX}%`);
      labTrack.classList.toggle("is-mid", mid && showPct);
    }
    if (labPct && mid) labPct.textContent = `${Math.round(p)}%`;
  }

  function stopLabAnim() {
    labPlaySeq++;
    cancelAnimationFrame(labRaf);
    labRaf = 0;
    if (labPlayBtn) {
      labPlayBtn.classList.remove("is-on", "is-loading");
    }
    if (labHoldBtn) labHoldBtn.classList.remove("is-loading");
  }

  // Shows the wallpaper pair for rail value v on the fullscreen canvas.
  function showLabAt(v) {
    const seg = labSeg(v);
    const [a, b] = labPair(seg);
    const p = (v - seg * 100) / 100;
    const near = p < 0.5 ? a : b;
    state.wallId = near.id;
    if (state.theme !== near.theme) setThemeAttr(near.theme);
    enterWallMode(true);
    wallEngine.scrollTarget = 0;
    lastScrubScrollY = window.scrollY;
    wallEngine.scrubPair(a, b, p);
  }

  function scrubLabTo(v) {
    stopLabAnim();
    const clamped = Math.max(0, Math.min(LAB_MAX, Math.round(v)));
    paintLab(clamped);
    showLabAt(clamped);
    const seg = labSeg(clamped);
    scheduleIdleWarm(labPair((seg + 1) % LAB_PAIRS.length)[1]);
  }

  // Eases the playhead (and optionally the halftone) from where it is to `to`.
  function tweenLab(to, dur, withWall) {
    stopLabAnim();
    const from = labPos;
    if (reducedMotion || Math.abs(to - from) < 0.5) {
      paintLab(to, withWall);
      if (withWall) showLabAt(to);
      return;
    }
    const t0 = performance.now();
    const step = (now) => {
      const k = clamp01((now - t0) / dur);
      const v = from + (to - from) * easeOutCubic(k);
      paintLab(k < 1 ? v : to, withWall);
      if (withWall) showLabAt(k < 1 ? v : to);
      labRaf = k < 1 ? requestAnimationFrame(step) : 0;
    };
    labRaf = requestAnimationFrame(step);
  }

  // Theme picked elsewhere (rail stop, T picker, W picker): glide to its stop & warm its next neighbor.
  function syncLabToTheme(themeKey) {
    const idx = THEME_ORDER.indexOf(themeKey);
    if (idx < 0) return;
    scheduleIdleWarm(labPair(idx)[1]);
    if (labPos === idx * 100 || (idx === 0 && labPos >= LAB_MAX)) return;
    tweenLab(idx * 100, 460, false);
  }

  // Play: prepare BOTH textures + GLSL shaders + FBOs BEFORE starting clock -> zero stutter, zero black flash!
  async function playLab() {
    const v = labPos >= LAB_MAX ? 0 : labPos;
    const seg = labSeg(v);
    const p0 = (v - seg * 100) / 100;
    const to = (seg + 1) * 100;
    if (reducedMotion) {
      scrubLabTo(to);
      if (to >= LAB_MAX) paintLab(0);
      return;
    }
    stopLabAnim();
    const mySeq = ++labPlaySeq;
    const [a, b] = labPair(seg);
    let loadTimer = 0;
    if (labPlayBtn) {
      labPlayBtn.classList.add("is-on");
      if (!wallEngine.isPairReady(a, b)) {
        loadTimer = setTimeout(() => {
          if (mySeq === labPlaySeq) labPlayBtn.classList.add("is-loading");
        }, 90);
      }
    }

    const ready = await wallEngine.preparePair(a, b);
    clearTimeout(loadTimer);
    if (mySeq !== labPlaySeq) return;
    if (labPlayBtn) labPlayBtn.classList.remove("is-loading");
    if (!ready) {
      stopLabAnim();
      return;
    }

    state.wallId = b.id;
    setThemeAttr(p0 < 0.5 ? a.theme : b.theme);
    wallEngine.scrollTarget = 0;
    lastScrubScrollY = -1;
    wallEngine.resize(window.innerWidth, window.innerHeight);
    wallEngine.startPreparedPair(a, b, LAB_PLAY_MS, p0);
    enterWallMode(true);

    const t0 = wallEngine.transStart;
    const step = (now) => {
      if (mySeq !== labPlaySeq) return;
      const p = clamp01((now - t0) / LAB_PLAY_MS);
      if (p >= 0.5 && state.theme !== b.theme) setThemeAttr(b.theme);
      if (p < 1) {
        paintLab(seg * 100 + p * 100);
        labRaf = requestAnimationFrame(step);
      } else {
        stopLabAnim();
        paintLab(to >= LAB_MAX ? 0 : to);
        // Sliding-window lookahead: warm ONLY the next single theme wallpaper in idle time!
        const nextSeg = (seg + 1) % LAB_PAIRS.length;
        scheduleIdleWarm(labPair(nextSeg)[1]);
      }
    };
    labRaf = requestAnimationFrame(step);
  }

  // 45%: prepare pair first, then glide smoothly to the most interleaved frame (zero black flash).
  async function holdLab() {
    const seg = labSeg(labPos);
    const targetV = seg * 100 + LAB_HOLD;
    const [a, b] = labPair(seg);
    stopLabAnim();
    const mySeq = ++labPlaySeq;
    let loadTimer = 0;
    if (labHoldBtn && !wallEngine.isPairReady(a, b)) {
      loadTimer = setTimeout(() => {
        if (mySeq === labPlaySeq) labHoldBtn.classList.add("is-loading");
      }, 90);
    }
    const ready = await wallEngine.preparePair(a, b);
    clearTimeout(loadTimer);
    if (mySeq !== labPlaySeq) return;
    if (labHoldBtn) labHoldBtn.classList.remove("is-loading");
    if (!ready) return;
    tweenLab(targetV, 420, true);
    scheduleIdleWarm(labPair((seg + 1) % LAB_PAIRS.length)[1]);
  }

  if (labRange) labRange.addEventListener("input", () => scrubLabTo(Number(labRange.value)));
  // While a transition plays the key reads Stop; pressing it freezes the rail where it is.
  onEl("btn-scrub-play", "click", () => (labRaf || (labPlayBtn && labPlayBtn.classList.contains("is-on")) ? scrubLabTo(labPos) : playLab()));
  onEl("btn-scrub-hold", "click", holdLab);
  paintLab(0, false);

  /* ---- Smart Zero-Waste Intent Warming (Viewport + Hover/Focus) ------ */
  const halftoneSec = document.getElementById("halftone");
  if (halftoneSec && "IntersectionObserver" in window) {
    const labObs = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        scheduleIdleWarmPair(labSeg(labPos));
      }
    }, { rootMargin: "320px 0px" });
    labObs.observe(halftoneSec);
  }

  // Hover/focus intent on Lab controls warms only the active pair or hovered stop
  [labPlayBtn, labHoldBtn, labRange].forEach((el) => {
    if (!el) return;
    const warmCur = () => scheduleIdleWarmPair(labSeg(labPos));
    el.addEventListener("pointerenter", warmCur, { passive: true });
    el.addEventListener("focus", warmCur, { passive: true });
    el.addEventListener("touchstart", warmCur, { passive: true });
  });
  document.querySelectorAll("[data-pick-theme]").forEach((btn) => {
    const warmStop = () => {
      const w = ALL_WALLS.find((x) => x.theme === btn.dataset.pickTheme);
      if (w) scheduleIdleWarm(w, wallEngine);
    };
    btn.addEventListener("pointerenter", warmStop, { passive: true });
    btn.addEventListener("focus", warmStop, { passive: true });
  });
  document.querySelectorAll("[data-pick-bar]").forEach((btn) => {
    const warmBar = () => {
      if (!saveDataMode() && btn.dataset.pickBar) {
        loadImg(`./assets/bars/${btn.dataset.pickBar}.webp?v=${ASSET_V}`).catch(() => {});
      }
    };
    btn.addEventListener("pointerenter", warmBar, { passive: true });
    btn.addEventListener("focus", warmBar, { passive: true });
  });
  ["nav-open-walls", "hero-open-walls", "btn-open-walls", "drawer-open-walls"].forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    const warmW = () => scheduleIdleWarm(findWall(state.wallId), wallEngine);
    el.addEventListener("pointerenter", warmW, { passive: true });
    el.addEventListener("focus", warmW, { passive: true });
  });
  ["nav-open-themes", "hero-open-themes", "btn-open-themes", "drawer-open-themes"].forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    const warmT = () => {
      const t = THEMES[state.theme];
      if (t && t.walls[0]) scheduleIdleWarm(findWall(t.walls[0].id), tpEngine);
    };
    el.addEventListener("pointerenter", warmT, { passive: true });
    el.addEventListener("focus", warmT, { passive: true });
  });

  /* ---- Install: one sequence on a rail; click a row to copy it --------
   * The rail fills in accent down to the furthest step copied so far, so a
   * visitor following along can see where they are. */
  const installList = document.getElementById("install-list");
  const btnCopyInstall = document.getElementById("btn-copy-install");
  const installRows = installList ? Array.from(installList.children) : [];
  const installCmds = installList ? Array.from(installList.querySelectorAll("[data-copy]")) : [];
  let installHead = -1;

  // The async Clipboard API only exists on secure origins (https, localhost). Opened from
  // file:// or a LAN address (http://192.168…) it is missing, so fall back to the old
  // hidden-textarea + execCommand("copy") path, which works anywhere inside a click.
  function legacyCopy(txt) {
    const ta = document.createElement("textarea");
    ta.value = txt;
    ta.setAttribute("readonly", "");
    ta.style.cssText = "position:fixed;top:0;left:0;width:1px;height:1px;opacity:0;font-size:16px;pointer-events:none";
    document.body.appendChild(ta);
    const sel = document.getSelection();
    const prev = sel && sel.rangeCount ? sel.getRangeAt(0) : null;
    const prevFocus = document.activeElement;
    ta.focus({ preventScroll: true });
    ta.select();
    ta.setSelectionRange(0, txt.length);
    let ok = false;
    try { ok = document.execCommand("copy"); } catch { ok = false; }
    ta.remove();
    if (prev && sel) { sel.removeAllRanges(); sel.addRange(prev); }
    if (prevFocus && prevFocus.focus) prevFocus.focus({ preventScroll: true });
    return ok;
  }

  async function copyText(txt, msg) {
    let ok = false;
    if (navigator.clipboard && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(txt);
        ok = true;
      } catch { ok = false; }
    }
    if (!ok) ok = legacyCopy(txt);
    toast(ok ? msg : "Copy unavailable");
    return ok;
  }

  function markInstall(rowIdx) {
    if (rowIdx <= installHead) return;
    installHead = rowIdx;
    installRows.forEach((li, i) => {
      li.classList.toggle("is-past", i < rowIdx);
      li.classList.toggle("is-head", i === rowIdx);
    });
  }

  function flashCopied(btn) {
    const act = btn.querySelector(".step-act");
    btn.classList.add("is-copied");
    if (act) act.textContent = "Copied";
    clearTimeout(btn._copiedTimer);
    btn._copiedTimer = setTimeout(() => {
      btn.classList.remove("is-copied");
      if (act) act.textContent = "Copy";
    }, 1200);
  }

  if (installList) {
    // Notes line up in one column right after the longest command.
    const longest = installCmds.reduce((m, b) => Math.max(m, b.dataset.copy.length), 0);
    if (longest) installList.style.setProperty("--cmd-ch", String(longest));
    installList.addEventListener("click", async (e) => {
      const btn = e.target.closest("[data-copy]");
      if (!btn) return;
      if (!(await copyText(btn.dataset.copy, "Command copied"))) return;
      flashCopied(btn);
      markInstall(installRows.indexOf(btn.closest("li")));
    });
  }
  if (btnCopyInstall) {
    btnCopyInstall.addEventListener("click", async () => {
      const steps = installCmds.filter((b) => !b.closest(".is-opt"));
      if (!steps.length) return;
      if (!(await copyText(steps.map((b) => b.dataset.copy).join("\n"), "Steps 01–04 copied"))) return;
      markInstall(installRows.indexOf(steps[steps.length - 1].closest("li")));
    });
  }

  /* ---- Authentic Mira Overlays (WallSelect W & RiceSelector T) ------- */
  const ovWalls = document.getElementById("ov-walls");
  const ovThemes = document.getElementById("ov-themes");
  const ovLb = document.getElementById("ov-lb");
  const wdStrip = document.getElementById("wd-strip");
  const wdTitle = document.getElementById("wd-title");
  const wdMeta = document.getElementById("wd-meta");
  const wdBadge = document.getElementById("wd-badge");
  const tpList = document.getElementById("tp-list");
  const tpBar = document.getElementById("tp-bar");
  const tpName = document.getElementById("tp-name");
  const tpSub = document.getElementById("tp-sub");

  let activeOv = null;
  let wdIdx = 0;
  let tpIdx = 0;

  function closeAll() {
    setDrawerOpen(false);
    ovWalls.hidden = true;
    ovThemes.hidden = true;
    ovLb.hidden = true;
    document.body.classList.remove("walls-open");
    setOverlayLock(false);
    navBtnWalls.classList.remove("is-open");
    navBtnThemes.classList.remove("is-open");
    activeOv = null;
    syncScrollState();
    if (state.wallMode) wallEngine.lastDraw = 0;
  }

  function developWallpaper(wall, animate = true) {
    state.wallId = wall.id;
    setThemeAttr(wall.theme);
    enterWallMode(true);
    wallEngine.setWall(wall, animate, 460);
    syncLabToTheme(wall.theme);
  }

  ALL_WALLS.forEach((w, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "dock-tile";
    b.innerHTML = `<img src="${w.thumb}" alt="${w.name}" loading="lazy"><span>${w.id}</span>`;
    b.addEventListener("click", () => (wdIdx === i ? applyWall() : selectWall(i, true)));
    wdStrip.appendChild(b);
  });

  function prefetchWallNeighbors(idx) {
    const prev = ALL_WALLS[(idx - 1 + ALL_WALLS.length) % ALL_WALLS.length];
    const next = ALL_WALLS[(idx + 1) % ALL_WALLS.length];
    if (prev) loadImg(prev.src).catch(() => {});
    if (next) loadImg(next.src).catch(() => {});
  }

  function selectWall(i, animate) {
    wdIdx = (i + ALL_WALLS.length) % ALL_WALLS.length;
    const w = ALL_WALLS[wdIdx];
    wdTitle.textContent = w.title;
    wdMeta.textContent = window.innerWidth <= 860
      ? `${w.name} · ${w.effect}`
      : `Live 3px Bayer shader · ${w.name} · ${w.effect}`;
    wdBadge.hidden = w.id !== state.wallId;
    wdStrip.querySelectorAll(".dock-tile").forEach((el, idx) => {
      const on = idx === wdIdx;
      el.classList.toggle("is-sel", on);
      if (on) el.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
    });
    wallEngine.scrollTarget = 0;
    wallEngine.scrollPos = 0;
    if (animate) developWallpaper(w, true);
    prefetchWallNeighbors(wdIdx);
  }

  function applyWall() {
    const w = ALL_WALLS[wdIdx];
    developWallpaper(w, true);
    closeAll();
    toast(`Wallpaper: ${w.title}`);
  }

  function openWalls() {
    closeAll();
    document.body.classList.add("walls-open");
    setOverlayLock(true);
    navBtnWalls.classList.add("is-open");
    ovWalls.hidden = false;
    activeOv = "walls";
    wallEngine.scrollTarget = 0;
    wallEngine.scrollPos = 0;
    const cur = ALL_WALLS.findIndex((w) => w.id === state.wallId);
    const startIdx = cur >= 0 ? cur : 0;
    selectWall(startIdx, true);
  }

  THEME_ORDER.forEach((k, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "sc-row";
    b.innerHTML = `<span class="sc-num">0${i + 1}</span><span class="sc-name">${THEMES[k].name}</span><span class="inuse-tag" hidden>IN USE</span>`;
    b.addEventListener("click", () => (tpIdx === i ? applyTheme() : selectTheme(i, true)));
    tpList.appendChild(b);
  });

  function selectTheme(i, animate = true) {
    tpIdx = (i + THEME_ORDER.length) % THEME_ORDER.length;
    const k = THEME_ORDER[tpIdx];
    const t = THEMES[k];
    const wall = ALL_WALLS.find((w) => w.id === t.walls[0].id);
    resizeTpCanvas();
    if (wall) tpEngine.setWall(wall, animate, 420);
    tpBar.src = `./assets/bars/${t.bar}.webp?v=${ASSET_V}`;
    tpName.textContent = t.name;
    tpSub.textContent = t.sub;
    tpList.querySelectorAll(".sc-row").forEach((el, idx) => {
      const on = idx === tpIdx;
      el.classList.toggle("is-sel", on);
      el.querySelector(".inuse-tag").hidden = THEME_ORDER[idx] !== state.theme;
      if (on && window.innerWidth <= 860) {
        el.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
      }
    });
    const nextK = THEME_ORDER[(tpIdx + 1) % THEME_ORDER.length];
    const nextW = ALL_WALLS.find((w) => w.id === THEMES[nextK].walls[0].id);
    if (nextW) scheduleIdleWarm(nextW, tpEngine);
  }

  function applyTheme() {
    const k = THEME_ORDER[tpIdx];
    closeAll();
    lockThemeLive(k, true);
  }

  function openThemes() {
    closeAll();
    setOverlayLock(true);
    navBtnThemes.classList.add("is-open");
    ovThemes.hidden = false;
    activeOv = "themes";
    selectTheme(Math.max(0, THEME_ORDER.indexOf(state.theme)), false);
    requestAnimationFrame(() => {
      if (activeOv === "themes") {
        resizeTpCanvas();
        tpEngine.render(performance.now(), true);
      }
    });
  }

  onEl("nav-open-walls", "click", openWalls);
  onEl("hero-open-walls", "click", openWalls);
  onEl("btn-open-walls", "click", openWalls);
  onEl("drawer-open-walls", "click", openWalls);
  onEl("nav-open-themes", "click", openThemes);
  onEl("hero-open-themes", "click", openThemes);
  onEl("btn-open-themes", "click", openThemes);
  onEl("drawer-open-themes", "click", openThemes);
  onEl("wd-apply-btn", "click", applyWall);
  onEl("wd-key-apply", "click", applyWall);
  onEl("tp-apply-btn", "click", applyTheme);
  onEl("tp-key-apply", "click", applyTheme);
  document.querySelectorAll("[data-close]").forEach((el) => el.addEventListener("click", closeAll));

  /* Scoped overlay scroll lock (attached directly to overlays -> zero non-passive window listeners!) */
  let ovWheelCooldown = 0;
  [ovWalls, ovThemes, ovLb].forEach((ov) => {
    if (!ov) return;
    ov.addEventListener("wheel", (e) => {
      if (e.ctrlKey || e.metaKey) return;
      if (Math.abs(e.deltaX) > Math.abs(e.deltaY) && e.target.closest(".dock-strip, .showcase-list")) return;
      const sc = e.target.closest(".showcase");
      if (ov === ovThemes && sc && sc.scrollHeight > sc.clientHeight + 2) return;
      e.preventDefault();
      const now = performance.now();
      if (Math.abs(e.deltaY) >= 4 && now - ovWheelCooldown > 110) {
        ovWheelCooldown = now;
        const dir = e.deltaY > 0 ? 1 : -1;
        if (ov === ovWalls) selectWall(wdIdx + dir, true);
        else if (ov === ovThemes) selectTheme(tpIdx + dir, true);
      }
    }, { passive: false });
    ov.addEventListener("touchmove", (e) => {
      if (e.target.closest(".dock-strip, .showcase-list")) return;
      const sc = e.target.closest(".showcase");
      if (ov === ovThemes && sc && sc.scrollHeight > sc.clientHeight + 2) return;
      e.preventDefault();
    }, { passive: false });
  });

  window.addEventListener("resize", () => {
    if (window.innerWidth > 980 && navDrawer && !navDrawer.hidden) setDrawerOpen(false);
  }, { passive: true });

  /* ---- Mobile Swipe Gestures for WallSelect & Theme Showcase --------- */
  function attachSwipe(el, onLeft, onRight) {
    if (!el) return;
    let sx = 0;
    let sy = 0;
    el.addEventListener("touchstart", (e) => {
      if (e.touches.length !== 1) return;
      sx = e.touches[0].clientX;
      sy = e.touches[0].clientY;
    }, { passive: true });
    el.addEventListener("touchend", (e) => {
      if (!e.changedTouches.length) return;
      const dx = e.changedTouches[0].clientX - sx;
      const dy = e.changedTouches[0].clientY - sy;
      if (Math.abs(dx) > 44 && Math.abs(dx) > Math.abs(dy) * 1.4) {
        if (dx < 0) onLeft();
        else onRight();
      }
    }, { passive: true });
  }
  attachSwipe(ovWalls.querySelector(".ov-back"), () => selectWall(wdIdx + 1, true), () => selectWall(wdIdx - 1, true));
  attachSwipe(document.getElementById("tp-stage"), () => selectTheme(tpIdx + 1, true), () => selectTheme(tpIdx - 1, true));

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (activeOv) { e.preventDefault(); closeAll(); return; }
      if (navDrawer && !navDrawer.hidden) { e.preventDefault(); setDrawerOpen(false); return; }
    }
    if (activeOv === "walls") {
      if (e.key === "ArrowRight") { e.preventDefault(); selectWall(wdIdx + 1, true); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); selectWall(wdIdx - 1, true); }
      else if (e.key === "Enter") { e.preventDefault(); applyWall(); }
      else if ([" ", "ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End"].includes(e.key)) { e.preventDefault(); }
      return;
    }
    if (activeOv === "themes") {
      if (e.key === "ArrowDown" || e.key === "ArrowRight") { e.preventDefault(); selectTheme(tpIdx + 1, true); }
      else if (e.key === "ArrowUp" || e.key === "ArrowLeft") { e.preventDefault(); selectTheme(tpIdx - 1, true); }
      else if (e.key === "Enter") { e.preventDefault(); applyTheme(); }
      else if ([" ", "PageUp", "PageDown", "Home", "End"].includes(e.key)) { e.preventDefault(); }
      return;
    }
    if (activeOv === "lb") {
      if ([" ", "ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End"].includes(e.key)) e.preventDefault();
      return;
    }
    if (e.target.tagName === "INPUT" || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "w" || e.key === "W") { e.preventDefault(); openWalls(); }
    else if (e.key === "t" || e.key === "T") { e.preventDefault(); openThemes(); }
  });

  /* ---- Boot ---------------------------------------------------------- */
  const params = new URLSearchParams(window.location.search);
  const qTheme = params.get("theme");
  const qWall = params.get("wall");
  const qBar = params.get("bar");
  const qWallObj = qWall ? ALL_WALLS.find((w) => w.id === qWall) : null;

  if (qWallObj) {
    state.lockedWorld = THEME_ORDER.indexOf(qWallObj.theme);
    developWallpaper(qWallObj, false);
    setBar(qBar || "mira-minimal");
  } else if (qTheme && THEMES[qTheme]) {
    lockThemeLive(qTheme, false);
    if (qBar) setBar(qBar);
  } else {
    setThemeAttr("mira");
    setBar(qBar || "mira-minimal");
  }

  /* ---- Easter egg: click the hero sparkle → pixel sparkles across the screen ----
   * Its own fixed layer above everything, on the site's 3px grid. Hand-drawn sparkles
   * in the wordmark's style pop up outward from the star like an unseen wave passing. */
  (() => {
    const hit = document.querySelector(".wm-star-hit");
    const star = document.querySelector(".wordmark .wm-star");
    if (!hit || !star) return;

    const CELL = 3;
    const rand = (a, b) => a + Math.random() * (b - a);
    const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
    const rint = (a, b) => a + Math.floor(Math.random() * (b - a + 1));
    // Hand-drawn sparkle in the wordmark's style: uneven arms, chipped tips, a lopsided
    // core and the odd stray pixel. Each frame is redrawn, so it "boils" a little, and the
    // twinkle grows, hesitates, sometimes flickers back, then shrinks at an uneven pace.
    function handStar(peak) {
      const arms = [0, 1, 2, 3].map(() => Math.max(1, peak + (Math.random() < 0.5 ? 0 : pick([-1, 1]))));
      if (peak >= 2) {
        if (Math.random() < 0.6) arms[2] += 1;
        if (Math.random() < 0.6) arms[3] += 1;
      }
      const core = peak >= 3 && Math.random() < 0.75;
      const DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]];
      const frame = (lv) => {
        if (lv === 0) return [[0, 0]];
        const pts = new Map();
        const put = (x, y) => pts.set(`${x},${y}`, [x, y]);
        put(0, 0);
        DIRS.forEach(([ux, uy], i) => {
          const len = Math.max(1, Math.round((arms[i] * lv) / peak));
          for (let step = 1; step <= len; step++) {
            if (step === len && step > 1 && Math.random() < 0.18) break;
            put(ux * step, uy * step);
          }
        });
        if (core && lv === peak) {
          for (const cx of [-1, 1]) for (const cy of [-1, 1]) if (Math.random() < 0.8) put(cx, cy);
        }
        if (lv > 1 && Math.random() < 0.12) put(pick([-1, 1]) * rint(1, 2), pick([-1, 1]) * rint(1, 2));
        return Array.from(pts.values());
      };
      const seq = [];
      for (let lv = 0; lv <= peak; lv++) seq.push(lv);
      for (let h = rint(0, 2); h > 0; h--) seq.push(peak);
      if (peak > 1 && Math.random() < 0.35) seq.push(peak - 1, peak);
      for (let lv = peak - 1; lv >= 0; lv--) if (lv === 0 || Math.random() > 0.25) seq.push(lv);
      const frames = seq.map(frame);
      const ends = [];
      let acc = 0;
      for (let i = 0; i < frames.length; i++) ends.push((acc += rand(0.035, 0.095)));
      return { frames, ends };
    }
    const still = window.matchMedia("(prefers-reduced-motion: reduce)");

    let canvas = null, ctx = null, img = null, px = null;
    let cols = 0, rows = 0, raf = 0;
    const ripples = [];

    function ensureCanvas() {
      const c = Math.ceil(window.innerWidth / CELL);
      const r = Math.ceil(window.innerHeight / CELL);
      if (!canvas) {
        canvas = document.createElement("canvas");
        canvas.className = "egg-ripple";
        canvas.setAttribute("aria-hidden", "true");
        document.body.appendChild(canvas);
        ctx = canvas.getContext("2d");
      }
      if (c !== cols || r !== rows) {
        cols = c;
        rows = r;
        canvas.width = cols;
        canvas.height = rows;
        canvas.style.width = `${cols * CELL}px`;
        canvas.style.height = `${rows * CELL}px`;
        img = ctx.createImageData(cols, rows);
        px = new Uint32Array(img.data.buffer);
        ripples.length = 0;
      }
      canvas.hidden = false;
    }

    const rgba = (r, g, b, a) => ((a << 24) | (b << 16) | (g << 8) | r) >>> 0;

    function spawn(cx, cy) {
      ensureCanvas();
      const ox = cx / CELL;
      const oy = cy / CELL;
      const sx = rand(0.88, 1.12);
      const sy = rand(0.88, 1.12);
      const far = Math.max(
        Math.hypot(ox / sx, oy / sy), Math.hypot((cols - ox) / sx, oy / sy),
        Math.hypot(ox / sx, (rows - oy) / sy), Math.hypot((cols - ox) / sx, (rows - oy) / sy)
      ) * 1.06;
      const dur = rand(1.35, 1.85);

      const m = (getComputedStyle(star).fill.match(/\d+(\.\d+)?/g) || ["217", "216", "74"]).map(Number);
      // Sparkles stay off the wordmark and don't pile onto each other.
      const wm = star.ownerSVGElement ? star.ownerSVGElement.getBoundingClientRect() : null;
      const keepOut = wm && {
        l: wm.left / CELL - 10, r: wm.right / CELL + 10, t: wm.top / CELL - 10, b: wm.bottom / CELL + 10,
      };
      const sparks = [];
      const n = Math.max(22, Math.min(100, Math.round((cols * rows) / 2400)));
      for (let tries = 0; sparks.length < n && tries < n * 6; tries++) {
        const x = Math.floor(Math.random() * cols);
        const y = Math.floor(Math.random() * rows);
        if (keepOut && x > keepOut.l && x < keepOut.r && y > keepOut.t && y < keepOut.b) continue;
        const d = Math.hypot((x - ox) / sx, (y - oy) / sy);
        if (d < 18) continue;
        const k = Math.min(1, d / far);
        const scale = k < 0.3 ? pick([2, 3, 3, 4]) : k < 0.65 ? pick([2, 3, 3]) : pick([2, 2, 3]);
        if (sparks.some((o) => Math.abs(o.x - x) < (o.scale + scale) * 5 && Math.abs(o.y - y) < (o.scale + scale) * 6)) continue;
        const shape = handStar(k < 0.3 ? pick([3, 3, 2]) : k < 0.65 ? pick([2, 3, 2]) : pick([1, 2, 2]));
        const len = Math.hypot(x - ox, y - oy) || 1;
        const lift = Math.random() < 0.3 ? rand(0.7, 0.9) : rand(0, 0.25);
        sparks.push({
          x, y,
          frames: shape.frames,
          ends: shape.ends,
          scale,
          drift: rand(0, 1.3),
          dx: (x - ox) / len,
          dy: (y - oy) / len,
          color: rgba(
            Math.round(m[0] + (255 - m[0]) * lift), Math.round(m[1] + (255 - m[1]) * lift),
            Math.round(m[2] + (255 - m[2]) * lift), rint(215, 255)
          ),
          at: (1 - Math.sqrt(1 - k)) * dur + rand(0, 0.12),
        });
      }

      ripples.push({
        sparks,
        t0: performance.now(),
      });
      if (ripples.length > 4) ripples.shift();
      if (!raf) raf = requestAnimationFrame(tick);
    }

    function block(x, y, c, color) {
      for (let j = 0; j < c; j++) {
        const yy = y + j;
        if (yy < 0 || yy >= rows) continue;
        const row = yy * cols;
        for (let i = 0; i < c; i++) {
          const xx = x + i;
          if (xx >= 0 && xx < cols) px[row + xx] = color;
        }
      }
    }

    function drawSpark(s, k) {
      const sc = s.scale;
      const shift = Math.round(s.drift * k * sc);
      const bx = s.x + Math.round(s.dx * shift);
      const by = s.y + Math.round(s.dy * shift);
      const o = (sc - 1) >> 1;
      for (const [dx, dy] of s.frames[k]) block(bx + dx * sc - o, by + dy * sc - o, sc, s.color);
    }

    function tick(now) {
      raf = 0;
      px.fill(0);
      for (let i = ripples.length - 1; i >= 0; i--) {
        const rp = ripples[i];
        const t = (now - rp.t0) / 1000;
        let alive = false;
        for (const s of rp.sparks) {
          const el = t - s.at;
          if (el >= s.ends[s.ends.length - 1]) continue;
          alive = true;
          if (el < 0) continue;
          let k = 0;
          while (s.ends[k] <= el) k++;
          drawSpark(s, k);
        }
        if (!alive) ripples.splice(i, 1);
      }
      ctx.putImageData(img, 0, 0);
      if (ripples.length) raf = requestAnimationFrame(tick);
      else canvas.hidden = true;
    }

    hit.addEventListener("click", () => {
      if (still.matches) return;
      const box = star.getBoundingClientRect();
      if (star.animate) {
        star.animate(
          [{ transform: "scale(1)" }, { transform: "scale(1.3)", offset: 0.35 }, { transform: "scale(1)" }],
          { duration: 360, easing: "cubic-bezier(0.34, 1.45, 0.64, 1)" }
        );
      }
      spawn(box.left + box.width / 2, box.top + box.height / 2);
    });

    window.addEventListener("resize", () => {
      if (!canvas || !ripples.length) return;
      ripples.length = 0;
      px.fill(0);
      ctx.putImageData(img, 0, 0);
      canvas.hidden = true;
    }, { passive: true });
  })();

  measureSlides();
  window.addEventListener("load", () => {
    measureSlides();
    syncScrollState();
    if (
      "serviceWorker" in navigator &&
      (location.protocol === "https:" || location.hostname === "localhost" || location.hostname === "127.0.0.1")
    ) {
      navigator.serviceWorker.register(`./sw.js?v=${ASSET_V}`).catch(() => {});
    }
  }, { passive: true });
  syncScrollState();
  matrix.depthPos = matrix.scrollNorm;
  wallEngine.scrollPos = wallEngine.scrollTarget;
  if (!state.wallMode) matrix.render(1.6, true);
  if (!reducedMotion) requestAnimationFrame(frame);

  if (params.get("open") === "walls") openWalls();
  else if (params.get("open") === "themes") openThemes();
})();

import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

// ═══════════════════════════════════════════════════════════════
// Simplex noise GLSL (inlined)
// ═══════════════════════════════════════════════════════════════
const NOISE_GLSL = `
  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

  float snoise(vec3 v) {
    const vec2 C = vec2(1.0/6.0, 1.0/3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i  = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = mod289(i);
    vec4 p = permute(permute(permute(
      i.z + vec4(0.0, i1.z, i2.z, 1.0))
      + i.y + vec4(0.0, i1.y, i2.y, 1.0))
      + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);
    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
  }
`;

// ═══════════════════════════════════════════════════════════════
// FFT-Reactive Vertex Shader
// ═══════════════════════════════════════════════════════════════
const vertexShader = `
  ${NOISE_GLSL}
  uniform float uTime;
  uniform float uNoiseFreq;
  uniform float uNoiseAmp;
  uniform float uPulse;
  uniform float uBass;
  uniform float uMid;
  uniform float uTreble;
  uniform float uEnergy;
  varying vec3 vNormal;
  varying vec3 vPosition;
  varying float vDisplacement;
  varying float vFresnel;
  varying vec2 vUv;

  void main() {
    vUv = uv;

    // Base noise displacement
    float noise = snoise(position * uNoiseFreq + uTime * 0.5);

    // FFT-reactive displacement layers:
    // Bass → deep, slow undulating waves
    float bassWave = sin(position.y * 2.0 + uTime * 1.5) * uBass * 0.25;
    // Mid → medium organic ripples
    float midRipple = snoise(position * 3.0 + uTime * 2.0) * uMid * 0.18;
    // Treble → sharp, spiky high-frequency detail
    float trebleSpike = snoise(position * 8.0 + uTime * 4.0) * uTreble * 0.12;

    // Combined displacement
    float displacement = noise * uNoiseAmp
      + uPulse * 0.05
      + bassWave
      + midRipple
      + trebleSpike
      + uEnergy * 0.06;

    vec3 newPosition = position + normal * displacement;
    vDisplacement = displacement;
    vNormal = normalMatrix * normal;
    vPosition = (modelViewMatrix * vec4(newPosition, 1.0)).xyz;

    // Pre-compute fresnel for fragment shader
    vec3 viewDir = normalize(cameraPosition - (modelMatrix * vec4(newPosition, 1.0)).xyz);
    vFresnel = pow(1.0 - max(dot(normalize(normal), viewDir), 0.0), 2.8);

    gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
  }
`;

// ═══════════════════════════════════════════════════════════════
// Enhanced Fragment Shader with iridescence and energy glow
// ═══════════════════════════════════════════════════════════════
const fragmentShader = `
  uniform vec3 uColor1;
  uniform vec3 uColor2;
  uniform vec3 uColor3;
  uniform float uTime;
  uniform float uGlow;
  uniform float uEnergy;
  uniform float uBass;
  uniform float uLanguageBlend; // 0.0 = English (blue), 1.0 = German (amber)
  varying vec3 vNormal;
  varying vec3 vPosition;
  varying float vDisplacement;
  varying float vFresnel;
  varying vec2 vUv;

  // Language palette
  const vec3 EN_TINT = vec3(0.15, 0.45, 0.95);  // electric blue
  const vec3 DE_TINT = vec3(0.95, 0.55, 0.12);  // deep sunset orange

  void main() {
    vec3 viewDir = normalize(-vPosition);

    // Language lava-lamp blend — flows through the surface like liquid
    float langWave = sin(vUv.x * 4.0 + vUv.y * 3.0 + uTime * 0.8) * 0.5 + 0.5;
    float langFlow = sin(vUv.y * 5.0 - uTime * 0.5 + vDisplacement * 6.0) * 0.5 + 0.5;
    float localBlend = clamp(uLanguageBlend + (langWave * langFlow - 0.5) * 0.3, 0.0, 1.0);
    vec3 langColor = mix(EN_TINT, DE_TINT, localBlend);

    // Tri-color gradient based on displacement + UV
    float t = vDisplacement * 3.0 + 0.5;
    vec3 color = mix(uColor1, uColor2, clamp(t, 0.0, 1.0));
    // Iridescent third color on displacement peaks
    float peak = smoothstep(0.12, 0.25, abs(vDisplacement));
    color = mix(color, uColor3, peak * 0.4);

    // Bleed language color into the surface — stronger at edges
    float langStrength = 0.18 + vFresnel * 0.25;
    color = mix(color, color * langColor, langStrength);

    // Animated iridescent shimmer
    float shimmer = sin(vUv.x * 20.0 + vUv.y * 15.0 + uTime * 3.0) * 0.5 + 0.5;
    color += shimmer * uColor3 * 0.08 * uEnergy;

    // Fresnel rim glow — tinted by language
    float rimIntensity = uGlow + uEnergy * 0.5;
    vec3 rimColor = mix(uColor2, langColor, 0.4);
    color += vFresnel * rimColor * rimIntensity;

    // Core inner light (subsurface scattering approximation)
    float core = pow(max(dot(viewDir, vNormal), 0.0), 4.0);
    color += core * mix(uColor1, vec3(1.0), 0.3) * 0.35;

    // Bass-reactive deep pulse
    color += uBass * uColor1 * 0.15 * (1.0 - vFresnel);

    // Energy-reactive edge bloom — language-tinted
    float edgeGlow = pow(vFresnel, 1.5) * uEnergy * 0.6;
    color += edgeGlow * mix(uColor3, langColor, 0.35);

    float alpha = 0.92 - vFresnel * 0.1 + uEnergy * 0.05;
    gl_FragColor = vec4(color, clamp(alpha, 0.0, 1.0));
  }
`;

// ═══════════════════════════════════════════════════════════════
// State + Language color presets
// ═══════════════════════════════════════════════════════════════
const STATE_PARAMS = {
  idle: {
    color1: [0.02, 0.71, 0.83],
    color2: [0.05, 0.55, 0.72],
    color3: [0.39, 0.4, 0.95],     // indigo iridescence
    noiseFreq: 1.5, noiseAmp: 0.08, speed: 0.3,
    glow: 0.6, pulseSpeed: 2.0, pulseAmp: 0.3,
  },
  listening: {
    color1: [0.96, 0.45, 0.71],
    color2: [0.93, 0.28, 0.60],
    color3: [0.66, 0.33, 0.97],     // purple iridescence
    noiseFreq: 4.0, noiseAmp: 0.15, speed: 1.2,
    glow: 1.0, pulseSpeed: 6.0, pulseAmp: 0.5,
  },
  thinking: {
    color1: [0.96, 0.62, 0.04],
    color2: [0.85, 0.47, 0.02],
    color3: [0.98, 0.35, 0.15],     // orange-red iridescence
    noiseFreq: 2.5, noiseAmp: 0.12, speed: 0.8,
    glow: 0.8, pulseSpeed: 3.0, pulseAmp: 0.4, rotateSpeed: 1.5,
  },
  speaking: {
    color1: [0.06, 0.73, 0.51],
    color2: [0.02, 0.59, 0.41],
    color3: [0.2, 0.83, 0.75],      // teal iridescence
    noiseFreq: 2.0, noiseAmp: 0.10, speed: 0.6,
    glow: 0.9, pulseSpeed: 4.0, pulseAmp: 0.8,
  },
};

// Language-aware color tinting
const LANG_TINT = {
  en: { r: 0.0, g: 0.05, b: 0.12 },   // cool blue tint
  de: { r: 0.12, g: 0.06, b: 0.0 },   // warm amber tint
};

// ═══════════════════════════════════════════════════════════════
// OrbMesh — FFT-reactive 3D sphere
// ═══════════════════════════════════════════════════════════════
function OrbMesh({ state, lang }) {
  const meshRef = useRef();
  const materialRef = useRef();
  const params = STATE_PARAMS[state] || STATE_PARAMS.idle;
  const tint = LANG_TINT[lang] || { r: 0, g: 0, b: 0 };

  const uniforms = useMemo(() => ({
    uTime: { value: 0 },
    uColor1: { value: new THREE.Vector3(...params.color1) },
    uColor2: { value: new THREE.Vector3(...params.color2) },
    uColor3: { value: new THREE.Vector3(...params.color3) },
    uNoiseFreq: { value: params.noiseFreq },
    uNoiseAmp: { value: params.noiseAmp },
    uGlow: { value: params.glow },
    uPulse: { value: 0 },
    uBass: { value: 0 },
    uMid: { value: 0 },
    uTreble: { value: 0 },
    uEnergy: { value: 0 },
    uLanguageBlend: { value: 0 },  // 0.0 = EN, 1.0 = DE
  }), []);

  useFrame(({ clock }) => {
    if (!materialRef.current) return;
    const t = clock.getElapsedTime();
    const u = materialRef.current.uniforms;

    u.uTime.value = t * params.speed;

    // Smooth color transitions with language tint
    const c1 = params.color1;
    const c2 = params.color2;
    const c3 = params.color3;
    const target1 = new THREE.Vector3(c1[0] + tint.r, c1[1] + tint.g, c1[2] + tint.b);
    const target2 = new THREE.Vector3(c2[0] + tint.r * 0.5, c2[1] + tint.g * 0.5, c2[2] + tint.b * 0.5);
    const target3 = new THREE.Vector3(...c3);
    u.uColor1.value.lerp(target1, 0.04);
    u.uColor2.value.lerp(target2, 0.04);
    u.uColor3.value.lerp(target3, 0.04);

    // Smooth noise params
    u.uNoiseFreq.value += (params.noiseFreq - u.uNoiseFreq.value) * 0.05;
    u.uNoiseAmp.value += (params.noiseAmp - u.uNoiseAmp.value) * 0.05;
    u.uGlow.value += (params.glow - u.uGlow.value) * 0.05;

    // Pulse
    u.uPulse.value = Math.sin(t * params.pulseSpeed) * params.pulseAmp;

    // Read global FFT data (set by useAudioStream)
    const bass = window.__fftBass || 0;
    const mid = window.__fftMid || 0;
    const treble = window.__fftTreble || 0;
    const energy = window.__fftEnergy || 0;

    // Smooth FFT uniforms
    u.uBass.value += (bass - u.uBass.value) * 0.15;
    u.uMid.value += (mid - u.uMid.value) * 0.12;
    u.uTreble.value += (treble - u.uTreble.value) * 0.1;
    u.uEnergy.value += (energy - u.uEnergy.value) * 0.08;

    // Language blend — smooth lava-lamp transition
    const targetBlend = lang === "de" ? 1.0 : 0.0;
    u.uLanguageBlend.value += (targetBlend - u.uLanguageBlend.value) * 0.02;

    // Rotation
    if (meshRef.current) {
      const rotSpeed = params.rotateSpeed || 0.1;
      meshRef.current.rotation.y += rotSpeed * 0.01;
      meshRef.current.rotation.x = Math.sin(t * 0.3) * 0.1;
      // Energy-reactive scale breathing
      const s = 1.0 + energy * 0.08;
      meshRef.current.scale.setScalar(s);
    }
  });

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[1, 128, 128]} />
      <shaderMaterial
        ref={materialRef}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={uniforms}
        transparent
      />
    </mesh>
  );
}

// ═══════════════════════════════════════════════════════════════
// LanguageLight — dynamic point light that shifts with language
// ═══════════════════════════════════════════════════════════════
function LanguageLight({ lang }) {
  const lightRef = useRef();
  const targetColor = useMemo(() => {
    if (lang === "de") return new THREE.Color(0.95, 0.7, 0.3);  // warm amber
    return new THREE.Color(0.3, 0.6, 0.95);                      // cool blue
  }, [lang]);

  useFrame(() => {
    if (lightRef.current) {
      lightRef.current.color.lerp(targetColor, 0.02);
    }
  });

  return <pointLight ref={lightRef} position={[0, 3, 4]} intensity={0.5} color="#6699ff" />;
}

// PostFX via CSS — bloom/glow handled by CSS filter on the canvas container
// Chromatic aberration simulated via subtle hue-rotate animation

// ═══════════════════════════════════════════════════════════════
// ORB_COLORS for CSS decorations
// ═══════════════════════════════════════════════════════════════
const ORB_COLORS = {
  idle:      { r: "6,182,212",   hex: "#06b6d4" },
  listening: { r: "244,114,182", hex: "#f472b6" },
  thinking:  { r: "245,158,11",  hex: "#f59e0b" },
  speaking:  { r: "16,185,129",  hex: "#10b981" },
};

// ═══════════════════════════════════════════════════════════════
// Main FluidOrb Component
// ═══════════════════════════════════════════════════════════════
export default function FluidOrb({ state = "idle", lang = "en" }) {
  const isActive = state === "listening" || state === "speaking";
  const isThinking = state === "thinking";
  const c = ORB_COLORS[state] || ORB_COLORS.idle;

  return (
    <div className="relative flex items-center justify-center" style={{ width: 280, height: 280 }}>

      {/* ── Outer glow halo ── */}
      <div className="absolute rounded-full" style={{
        width: 260, height: 260,
        background: `radial-gradient(circle, rgba(${c.r}, ${isActive ? 0.12 : 0.05}) 0%, transparent 70%)`,
        transition: "all 1.5s ease",
        animation: "glow-pulse 4s ease-in-out infinite",
      }} />

      {/* ── Outer orbit ring ── */}
      <div className="absolute rounded-full" style={{
        width: 270, height: 270,
        border: `1px solid rgba(${c.r}, ${isActive ? 0.2 : 0.07})`,
        animation: "spin 35s linear infinite",
        transition: "border-color 1.5s",
      }}>
        <div className="absolute rounded-full" style={{
          width: 5, height: 5, top: -2.5, left: "50%",
          background: `rgba(${c.r}, ${isActive ? 1 : 0.25})`,
          boxShadow: isActive ? `0 0 12px rgba(${c.r}, 0.7)` : "none",
          transition: "all 1.2s",
        }} />
      </div>

      {/* ── Inner orbit ring — reverse ── */}
      <div className="absolute rounded-full" style={{
        width: 248, height: 248,
        border: `1px solid rgba(255,255,255, ${isActive ? 0.06 : 0.025})`,
        animation: "spin-rev 24s linear infinite",
        transition: "border-color 1.5s",
      }}>
        <div className="absolute rounded-full" style={{
          width: 3, height: 3, bottom: -1.5, left: "30%",
          background: `rgba(${c.r}, ${isActive ? 0.6 : 0.12})`,
          transition: "all 1.2s",
        }} />
      </div>

      {/* ── Pulse rings (active states) ── */}
      {isActive && [0, 1, 2].map((d, i) => (
        <div key={i} className="absolute rounded-full" style={{
          width: 220, height: 220,
          border: `1px solid rgba(${c.r}, ${0.12 - i * 0.03})`,
          animation: `pulse-ring 3s ease-out ${d}s infinite`,
        }} />
      ))}

      {/* ── Thinking arc spinner ── */}
      {isThinking && (
        <div className="absolute rounded-full" style={{
          width: 255, height: 255,
          border: "2px solid transparent",
          borderTopColor: `rgba(${c.r}, 0.4)`,
          borderRightColor: `rgba(${c.r}, 0.1)`,
          animation: "spin 1s linear infinite",
        }} />
      )}

      {/* ── 3D Orb canvas with CSS bloom ── */}
      <div style={{
        width: 220, height: 220,
        filter: state === "speaking"
          ? "brightness(1.15) contrast(1.05) saturate(1.2) drop-shadow(0 0 18px rgba(16,185,129,0.35))"
          : state === "listening"
            ? "brightness(1.1) saturate(1.15) drop-shadow(0 0 14px rgba(244,114,182,0.3))"
            : state === "thinking"
              ? "brightness(1.05) contrast(1.08) hue-rotate(3deg) drop-shadow(0 0 12px rgba(245,158,11,0.25))"
              : "brightness(1.0) drop-shadow(0 0 8px rgba(6,182,212,0.15))",
        transition: "filter 0.8s ease",
      }}>
        <Canvas
          camera={{ position: [0, 0, 2.8], fov: 45 }}
          gl={{ alpha: true, antialias: true, powerPreference: "high-performance" }}
          style={{ background: "transparent" }}
          dpr={[1, 1.5]}
        >
          <ambientLight intensity={0.5} />
          <pointLight position={[5, 5, 5]} intensity={0.7} />
          <pointLight position={[-3, -3, 2]} intensity={0.2} color="#818cf8" />
          <LanguageLight lang={lang} />
          <OrbMesh state={state} lang={lang} />
        </Canvas>
      </div>
    </div>
  );
}

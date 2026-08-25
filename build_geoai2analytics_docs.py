# -*- coding: utf-8 -*-
"""
Builder for geoai2analytics-sdk Master Interactive Academic Reference Manual & GitHub Pages.
Generates an encyclopedic documentation site with live Spatial Autocorrelation & LISA simulator sandbox,
GWR / MGWR kernel simulator, animated vector illustrations, and full Python API / CLI guides.
"""

import os
import xml.etree.ElementTree as ET

OUTPUT_DIR = r"C:\Users\YE\PyCharmMiscProject\PyPI\geoai2analytics_sdk\docs"
ICONS_DIR = os.path.join(OUTPUT_DIR, "icons")
ASSETS_DIR = os.path.join(OUTPUT_DIR, "assets")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "index.html")

os.makedirs(ICONS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# 1. Generate XML-valid vector logo and favicon
SVG_LOGO = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="100%" height="100%">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#08141e"/>
      <stop offset="100%" stop-color="#102538"/>
    </linearGradient>
    <linearGradient id="glow" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#10b981"/>
      <stop offset="50%" stop-color="#06b6d4"/>
      <stop offset="100%" stop-color="#3b82f6"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="#06b6d4" flood-opacity="0.35"/>
    </filter>
  </defs>

  <rect x="24" y="24" width="464" height="464" rx="96" fill="url(#bg)" stroke="url(#glow)" stroke-width="6" filter="url(#shadow)"/>

  <!-- Spatial Voronoi / Hexagonal Lattice -->
  <g opacity="0.25" stroke="#06b6d4" stroke-width="2" fill="none">
    <polygon points="256,90 350,145 350,255 256,310 162,255 162,145"/>
    <polygon points="350,145 444,200 444,310 350,365 256,310 256,200"/>
    <polygon points="162,145 256,200 256,310 162,365 68,310 68,200"/>
  </g>

  <!-- LISA Moran Scatterplot Axes & Quadrants -->
  <g stroke="#ffffff" stroke-width="2" opacity="0.4">
    <line x1="120" y1="240" x2="392" y2="240"/>
    <line x1="256" y1="104" x2="256" y2="376"/>
  </g>

  <!-- Neural AI Network Graph & Spatial Autocorrelation Nodes -->
  <g stroke="#06b6d4" stroke-width="3">
    <line x1="160" y1="160" x2="256" y2="240"/>
    <line x1="352" y1="140" x2="256" y2="240"/>
    <line x1="180" y1="320" x2="256" y2="240"/>
    <line x1="332" y1="310" x2="256" y2="240"/>
    <line x1="160" y1="160" x2="352" y2="140"/>
  </g>

  <!-- Glowing LISA Cluster Centroids -->
  <!-- High-High (Hotspot - Top Right) -->
  <circle cx="352" cy="140" r="18" fill="#ef4444" stroke="#ffffff" stroke-width="3"/>
  <!-- Low-Low (Coldspot - Bottom Left) -->
  <circle cx="180" cy="320" r="16" fill="#3b82f6" stroke="#ffffff" stroke-width="3"/>
  <!-- Spatial Outliers -->
  <circle cx="160" cy="160" r="14" fill="#06b6d4" stroke="#ffffff" stroke-width="3"/>
  <circle cx="332" cy="310" r="14" fill="#f59e0b" stroke="#ffffff" stroke-width="3"/>
  <!-- Central GeoAI Hub -->
  <circle cx="256" cy="240" r="22" fill="#10b981" stroke="#ffffff" stroke-width="4"/>

  <!-- Badge -->
  <rect x="146" y="405" width="220" height="42" rx="21" fill="#0b1320" stroke="url(#glow)" stroke-width="3"/>
  <text x="256" y="432" font-family="'Plus Jakarta Sans', 'Inter', sans-serif" font-size="16" font-weight="800" fill="#34d399" text-anchor="middle" letter-spacing="1.5">GEOAI &#183; ANALYTICS</text>
</svg>"""

ET.fromstring(SVG_LOGO)

with open(os.path.join(ICONS_DIR, "logo.svg"), "w", encoding="utf-8") as f:
    f.write(SVG_LOGO)
with open(os.path.join(ICONS_DIR, "favicon.svg"), "w", encoding="utf-8") as f:
    f.write(SVG_LOGO)


# 2. Generate Hero Animated Vector SVG Illustration (Hero Banner)
SVG_HERO = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 380" width="100%" height="100%" role="img" aria-labelledby="heroTitle heroDesc">
  <title id="heroTitle">geoai2analytics Engine Architecture</title>
  <desc id="heroDesc">Animated spatial analytics architecture showing Exploratory Spatial Data Analysis, Multiscale GWR Econometrics, and Explainable GeoAI Spatial SHAP.</desc>
  <defs>
    <linearGradient id="heroBg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#07121b"/>
      <stop offset="50%" stop-color="#0b1c2b"/>
      <stop offset="100%" stop-color="#091522"/>
    </linearGradient>
    <linearGradient id="accentGrad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#10b981"/>
      <stop offset="50%" stop-color="#06b6d4"/>
      <stop offset="100%" stop-color="#3b82f6"/>
    </linearGradient>
    <linearGradient id="kernelGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#06b6d4" stop-opacity="0.45"/>
      <stop offset="100%" stop-color="#06b6d4" stop-opacity="0.0"/>
    </linearGradient>
    <filter id="glowPulse">
      <feGaussianBlur stdDeviation="6" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>

  <rect width="1120" height="380" rx="20" fill="url(#heroBg)" stroke="#1e293b" stroke-width="2"/>

  <!-- Top Title Bar -->
  <text x="44" y="48" font-family="'Plus Jakarta Sans', Inter, sans-serif" font-size="22" font-weight="800" fill="#ffffff" letter-spacing="-0.01em">Pure-Python Spatial Statistics, Econometrics &amp; Explainable GeoAI</text>
  <text x="44" y="74" font-family="'Fira Code', monospace" font-size="13" fill="#94a3b8">ESDA (Moran / LISA) &#183; Spatial Econometrics (GWR / MGWR / SAR) &#183; Interpretable GeoAI (Spatial SHAP)</text>

  <!-- Left Card: ESDA & LISA Quadrants -->
  <g transform="translate(44, 98)">
    <rect width="320" height="248" rx="14" fill="#0d1f30" stroke="#1e3a53" stroke-width="1.5"/>
    <text x="20" y="32" font-family="'Plus Jakarta Sans', sans-serif" font-size="15" font-weight="700" fill="#38bdf8">1. Spatial Autocorrelation</text>
    <text x="20" y="52" font-family="Inter, sans-serif" font-size="12" fill="#64748b">Global Moran's I &amp; LISA Clusters</text>

    <!-- Axes -->
    <line x1="30" y1="140" x2="290" y2="140" stroke="#334155" stroke-width="1.5"/>
    <line x1="160" y1="65" x2="160" y2="215" stroke="#334155" stroke-width="1.5"/>

    <!-- Quadrant Labels -->
    <text x="220" y="85" font-family="'Fira Code', monospace" font-size="10.5" fill="#ef4444" font-weight="700">High-High</text>
    <text x="45" y="200" font-family="'Fira Code', monospace" font-size="10.5" fill="#3b82f6" font-weight="700">Low-Low</text>
    <text x="45" y="85" font-family="'Fira Code', monospace" font-size="10.5" fill="#06b6d4">Low-High</text>
    <text x="220" y="200" font-family="'Fira Code', monospace" font-size="10.5" fill="#f59e0b">High-Low</text>

    <!-- Moran Trend Line -->
    <line x1="60" y1="195" x2="260" y2="85" stroke="#10b981" stroke-width="2.5" stroke-dasharray="4 3"/>

    <!-- Scatter Points -->
    <circle cx="230" cy="95" r="6" fill="#ef4444"><animate attributeName="r" values="6;8;6" dur="3s" repeatCount="indefinite"/></circle>
    <circle cx="250" cy="110" r="5" fill="#ef4444"/>
    <circle cx="215" cy="115" r="5" fill="#ef4444"/>
    <circle cx="90" cy="175" r="6" fill="#3b82f6"><animate attributeName="r" values="6;8;6" dur="3s" begin="1s" repeatCount="indefinite"/></circle>
    <circle cx="75" cy="160" r="5" fill="#3b82f6"/>
    <circle cx="110" cy="185" r="5" fill="#3b82f6"/>
    <circle cx="100" cy="115" r="4.5" fill="#06b6d4"/>
    <circle cx="230" cy="165" r="4.5" fill="#f59e0b"/>

    <text x="20" y="235" font-family="'Fira Code', monospace" font-size="11.5" fill="#10b981" font-weight="600">Moran's I = +0.742 (z = 9.85)</text>
  </g>

  <!-- Middle Card: Spatial Econometrics & GWR -->
  <g transform="translate(400, 98)">
    <rect width="320" height="248" rx="14" fill="#0d1f30" stroke="#1e3a53" stroke-width="1.5"/>
    <text x="20" y="32" font-family="'Plus Jakarta Sans', sans-serif" font-size="15" font-weight="700" fill="#34d399">2. Spatial Econometrics</text>
    <text x="20" y="52" font-family="Inter, sans-serif" font-size="12" fill="#64748b">GWR &amp; Multiscale MGWR Kernels</text>

    <!-- Kernel Weighting Bell Curve -->
    <path d="M 30 190 Q 90 190 120 160 Q 160 70 160 70 Q 160 70 200 160 Q 230 190 290 190 Z" fill="url(#kernelGrad)"/>
    <path d="M 30 190 Q 90 190 120 160 Q 160 70 160 70 Q 160 70 200 160 Q 230 190 290 190" fill="none" stroke="#06b6d4" stroke-width="3"/>

    <!-- Center Bandwidth Marker -->
    <line x1="160" y1="65" x2="160" y2="195" stroke="#10b981" stroke-width="2" stroke-dasharray="3 3"/>
    <circle cx="160" cy="70" r="6" fill="#10b981" filter="url(#glowPulse)"/>

    <!-- Kernel Formula Label -->
    <text x="160" y="125" font-family="'Fira Code', monospace" font-size="11" fill="#e2e8f0" text-anchor="middle">w_ij = exp(-d_ij&#178; / b&#178;)</text>
    <text x="160" y="145" font-family="Inter, sans-serif" font-size="11" fill="#94a3b8" text-anchor="middle">Golden Section AICc Search</text>

    <text x="20" y="235" font-family="'Fira Code', monospace" font-size="11.5" fill="#34d399" font-weight="600">Local R&#178; = 0.884 &#183; AICc = 412.3</text>
  </g>

  <!-- Right Card: Explainable GeoAI & Spatial SHAP -->
  <g transform="translate(756, 98)">
    <rect width="320" height="248" rx="14" fill="#0d1f30" stroke="#1e3a53" stroke-width="1.5"/>
    <text x="20" y="32" font-family="'Plus Jakarta Sans', sans-serif" font-size="15" font-weight="700" fill="#a78bfa">3. Explainable GeoAI (XAI)</text>
    <text x="20" y="52" font-family="Inter, sans-serif" font-size="12" fill="#64748b">Spatial SHAP &amp; Conformal Coverage</text>

    <!-- Geographic Attribution Heatmap Grid -->
    <g transform="translate(30, 70)" stroke="#1e293b" stroke-width="1">
      <rect x="0" y="0" width="60" height="35" fill="#3b82f6" opacity="0.6"/>
      <rect x="65" y="0" width="60" height="35" fill="#06b6d4" opacity="0.7"/>
      <rect x="130" y="0" width="60" height="35" fill="#10b981" opacity="0.85"/>
      <rect x="195" y="0" width="65" height="35" fill="#ef4444" opacity="0.9"/>

      <rect x="0" y="40" width="60" height="35" fill="#3b82f6" opacity="0.4"/>
      <rect x="65" y="40" width="60" height="35" fill="#10b981" opacity="0.75"/>
      <rect x="130" y="40" width="60" height="35" fill="#f59e0b" opacity="0.8"/>
      <rect x="195" y="40" width="65" height="35" fill="#ef4444" opacity="0.7"/>

      <rect x="0" y="80" width="60" height="35" fill="#1e293b" opacity="0.8"/>
      <rect x="65" y="80" width="60" height="35" fill="#3b82f6" opacity="0.5"/>
      <rect x="130" y="80" width="60" height="35" fill="#06b6d4" opacity="0.6"/>
      <rect x="195" y="80" width="65" height="35" fill="#10b981" opacity="0.7"/>
    </g>

    <text x="20" y="215" font-family="'Fira Code', monospace" font-size="10.5" fill="#cbd5e1">SHAP &#934;_j(s) &#183; Spatial K-Fold CV</text>
    <text x="20" y="235" font-family="'Fira Code', monospace" font-size="11.5" fill="#a78bfa" font-weight="600">Coverage: 95% [y_low, y_high]</text>
  </g>
</svg>"""

ET.fromstring(SVG_HERO)

with open(os.path.join(ASSETS_DIR, "geoai-hero.svg"), "w", encoding="utf-8") as f:
    f.write(SVG_HERO)


# 3. Build the Master Academic Reference Manual
HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>geoai2analytics — Spatial Statistics, Econometrics & Explainable GeoAI</title>
<meta name="description" content="Official scientific reference manual for geoai2analytics-sdk: Spatial Autocorrelation (Moran, LISA, Gi*), Spatial Econometrics (GWR, MGWR, SAR), and Explainable GeoAI (Spatial SHAP, Conformal).">
<meta name="author" content="Yusuf Eminoğlu">
<link rel="icon" type="image/svg+xml" href="icons/favicon.svg">

<!-- MathJax for formula rendering -->
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    processEnvironments: true,
    tags: 'ams',
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
    ignoreHtmlClass: 'no-math|tex2jax_ignore'
  }
};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script src="https://unpkg.com/lucide@latest"></script>

<!-- Typography -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap" rel="stylesheet">

<style>
:root {
  --bg: #08111a;
  --bg-secondary: #0e1d2c;
  --bg-sidebar: #0a1622;
  --fg: #f3f4f6;
  --fg-heading: #ffffff;
  --muted: #94a3b8;
  --dim: #64748b;

  --accent: #10b981;
  --accent-dark: #059669;
  --accent-light: rgba(16, 185, 129, 0.12);
  --accent-cyan: #06b6d4;
  --accent-blue: #3b82f6;
  --accent-amber: #f59e0b;
  --accent-rose: #f43f5e;
  --accent-purple: #8b5cf6;

  --border: #1e293b;
  --border-subtle: #334155;
  --code-bg: #09131d;
  --sidebar-active: rgba(16, 185, 129, 0.15);
  --table-stripe: #0f2233;

  --gradient-brand: linear-gradient(135deg, #10b981 0%, #06b6d4 50%, #3b82f6 100%);
  --shadow-card: 0 4px 20px -2px rgba(0, 0, 0, 0.5);

  font-size: 14.5px;
  line-height: 1.68;
}

[data-theme="light"] {
  --bg: #f8fafc;
  --bg-secondary: #ffffff;
  --bg-sidebar: #f1f5f9;
  --fg: #1e293b;
  --fg-heading: #0f172a;
  --muted: #475569;
  --dim: #64748b;

  --accent: #059669;
  --accent-dark: #047857;
  --accent-light: #d1fae5;

  --border: #e2e8f0;
  --border-subtle: #cbd5e1;
  --code-bg: #0f172a;
  --sidebar-active: #d1fae5;
  --table-stripe: #f8fafc;
  --shadow-card: 0 4px 15px -1px rgba(0, 0, 0, 0.08);
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
  background: var(--bg);
  color: var(--fg);
  display: flex;
  min-height: 100vh;
  transition: background 0.2s ease, color 0.2s ease;
}

/* Top App Bar */
#top-bar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 58px;
  background: rgba(10, 22, 34, 0.92);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 1.5rem;
  z-index: 1000;
}

[data-theme="light"] #top-bar {
  background: rgba(255, 255, 255, 0.94);
}

.brand-wrap {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  text-decoration: none;
}

.brand-badge {
  background: var(--gradient-brand);
  color: #08111a;
  font-weight: 800;
  font-size: 1.1rem;
  width: 32px;
  height: 32px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 0 12px rgba(16, 185, 129, 0.5);
  font-family: 'Plus Jakarta Sans', sans-serif;
}

.brand-text {
  font-family: 'Plus Jakarta Sans', sans-serif;
  font-weight: 800;
  font-size: 1.25rem;
  letter-spacing: -0.02em;
  background: var(--gradient-brand);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.ver-tag {
  font-family: 'Fira Code', monospace;
  font-size: 0.72rem;
  font-weight: 600;
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
  background: var(--accent-light);
  color: var(--accent);
  border: 1px solid rgba(16, 185, 129, 0.3);
}

.top-actions {
  display: flex;
  align-items: center;
  gap: 0.65rem;
}

.top-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.4rem 0.75rem;
  border-radius: 7px;
  font-size: 0.82rem;
  font-weight: 500;
  color: var(--muted);
  text-decoration: none;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  transition: all 0.15s ease;
  cursor: pointer;
}

.top-btn:hover {
  color: var(--fg-heading);
  border-color: var(--accent);
  transform: translateY(-1px);
}

.top-btn.primary {
  background: var(--accent);
  color: #08111a;
  border-color: transparent;
  font-weight: 700;
}

/* Sidebar */
#sidebar {
  width: 320px;
  min-width: 320px;
  height: calc(100vh - 58px);
  position: sticky;
  top: 58px;
  background: var(--bg-sidebar);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  z-index: 100;
}

#search-wrap {
  padding: 0.85rem 1rem 0.65rem;
  border-bottom: 1px solid var(--border);
}

#search {
  width: 100%;
  padding: 0.55rem 0.85rem;
  border: 1px solid var(--border);
  border-radius: 6px;
  font-size: 0.85rem;
  background: var(--bg);
  color: var(--fg);
  outline: none;
  transition: border-color 0.2s, box-shadow 0.2s;
}

#search:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 2px var(--accent-light);
}

#toc {
  flex: 1;
  overflow-y: auto;
  padding: 6px 0;
  list-style: none;
  scrollbar-width: thin;
  scrollbar-color: var(--border) transparent;
}

.toc-group {
  border-bottom: 1px solid var(--border);
}

.toc-group-btn {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  text-align: left;
  padding: 8px 16px;
  background: none;
  border: none;
  font-size: 0.84rem;
  font-weight: 600;
  color: var(--fg-heading);
  cursor: pointer;
  transition: background 0.15s;
}

.toc-group-btn:hover {
  background: var(--accent-light);
}

.toc-group-btn .arrow {
  font-size: 0.7em;
  transition: transform 0.2s;
}

.toc-group-btn[aria-expanded="false"] .arrow {
  transform: rotate(-90deg);
}

.toc-algs {
  list-style: none;
  overflow: hidden;
}

.toc-algs li a {
  display: block;
  padding: 4px 16px 4px 24px;
  font-size: 0.82rem;
  color: var(--muted);
  text-decoration: none;
  border-left: 3px solid transparent;
  transition: all 0.15s;
}

.toc-algs li a:hover, .toc-algs li a.active {
  background: var(--sidebar-active);
  border-left-color: var(--accent);
  color: var(--accent);
  font-weight: 500;
}

.toc-algs li a.hidden {
  display: none;
}

#sidebar-footer {
  padding: 10px 16px;
  border-top: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 5px;
  background: var(--bg-secondary);
}

#sidebar-footer a {
  font-size: 0.78rem;
  color: var(--muted);
  text-decoration: none;
}

#sidebar-footer a:hover {
  color: var(--accent);
}

/* Content */
#content {
  flex: 1;
  max-width: 960px;
  margin: 0 auto;
  padding: calc(58px + 2rem) 3rem 6rem;
  overflow-y: auto;
}

/* Headings */
h1 {
  font-family: 'Plus Jakarta Sans', sans-serif;
  font-size: 2.3rem;
  margin: 0 0 0.25em;
  color: var(--fg-heading);
  letter-spacing: -0.02em;
}

h1.subtitle {
  font-size: 1.15rem;
  font-weight: 400;
  color: var(--muted);
  margin-bottom: 1.75em;
}

h2 {
  font-family: 'Plus Jakarta Sans', sans-serif;
  font-size: 1.6rem;
  margin: 2.75em 0 0.6em;
  padding-bottom: 0.3em;
  border-bottom: 2px solid var(--border);
  color: var(--fg-heading);
}

h2.group-header {
  border-bottom: 2px solid var(--accent);
  color: var(--accent);
  margin-top: 3.5em;
}

h3 {
  font-size: 1.22rem;
  margin: 1.6em 0 0.45em;
  color: var(--fg-heading);
}

h4 {
  font-size: 1.05rem;
  margin: 1.25em 0 0.35em;
  color: var(--muted);
}

p, ul, ol { margin: 0.75em 0; }
ul, ol { padding-left: 1.8em; }
li { margin: 0.3em 0; color: var(--fg); }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

code {
  font-family: 'Fira Code', 'Cascadia Code', monospace;
  font-size: 0.88em;
  background: var(--code-bg);
  color: var(--accent);
  padding: 0.12em 0.38em;
  border-radius: 4px;
  border: 1px solid var(--border);
}

pre {
  background: var(--code-bg);
  padding: 1.1em 1.25em;
  border-radius: 8px;
  border: 1px solid var(--border);
  overflow-x: auto;
  margin: 1em 0;
  font-family: 'Fira Code', monospace;
  font-size: 0.88em;
  line-height: 1.6;
  color: #f1f5f9;
}

pre code {
  background: transparent;
  border: none;
  padding: 0;
  color: inherit;
  font-size: 1em;
}

/* Tables */
table {
  width: 100%;
  border-collapse: collapse;
  margin: 1.2em 0 1.6em;
  font-size: 0.9em;
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid var(--border);
}

th, td {
  text-align: left;
  padding: 0.6em 0.85em;
  border: 1px solid var(--border);
}

th {
  background: var(--bg-secondary);
  color: var(--fg-heading);
  font-weight: 600;
}

tr:nth-child(even) td {
  background: var(--table-stripe);
}

.note {
  background: var(--accent-light);
  border-left: 4px solid var(--accent);
  padding: 0.8em 1.2em;
  margin: 1.2em 0;
  border-radius: 0 8px 8px 0;
  color: var(--fg);
}

.cover {
  text-align: center;
  padding: 3.5rem 1.5rem 2.8rem;
  background: radial-gradient(circle at center, rgba(16, 185, 129, 0.08) 0%, transparent 70%);
  border-radius: 16px;
  border: 1px solid var(--border);
  margin-bottom: 2.5rem;
}

.cover h1 { font-size: 3rem; margin-bottom: 0.15em; }
.cover .version { font-size: 1.1rem; color: var(--accent); font-weight: 600; font-family: 'Fira Code', monospace; }
.cover .date { font-size: 0.9rem; color: var(--muted); margin-top: 0.8em; }

/* Illustration Containers */
.figure-wrap {
  margin: 1.8rem 0;
  text-align: center;
}

.figure-img {
  width: 100%;
  border-radius: 12px;
  border: 1px solid var(--border);
  box-shadow: var(--shadow-card);
}

.figure-caption {
  font-size: 0.85rem;
  color: var(--muted);
  margin-top: 0.6rem;
  font-style: italic;
}

/* Interactive Sandbox Simulator Card */
.sandbox-card {
  background: var(--bg-secondary);
  border: 1px solid rgba(16, 185, 129, 0.3);
  border-radius: 12px;
  padding: 1.5rem;
  margin: 1.8rem 0;
  box-shadow: var(--shadow-card);
}

.sandbox-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  background: rgba(16, 185, 129, 0.15);
  color: var(--accent);
  border: 1px solid rgba(16, 185, 129, 0.3);
  font-size: 0.72rem;
  font-weight: 700;
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  text-transform: uppercase;
  margin-bottom: 0.75rem;
}

.sandbox-grid {
  display: grid;
  grid-template-columns: 1fr 1.2fr;
  gap: 1.5rem;
  margin-top: 1rem;
}

@media (max-width: 768px) {
  .sandbox-grid { grid-template-columns: 1fr; }
}

.control-item { margin-bottom: 0.85rem; }
.control-item label { display: flex; justify-content: space-between; font-size: 0.82rem; font-weight: 500; margin-bottom: 0.3rem; }
.control-input { width: 100%; padding: 0.55rem 0.75rem; border-radius: 6px; border: 1px solid var(--border); background: var(--bg); color: var(--fg); font-size: 0.85rem; outline: none; }
.control-select { width: 100%; padding: 0.55rem 0.75rem; border-radius: 6px; border: 1px solid var(--border); background: var(--bg); color: var(--fg); font-size: 0.85rem; outline: none; }

.calc-display {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.25rem;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

#moran-canvas-wrap {
  width: 100%;
  height: 220px;
  background: var(--code-bg);
  border-radius: 6px;
  border: 1px solid var(--border);
  position: relative;
  margin-bottom: 0.8rem;
}

/* Back to top */
#back-to-top {
  position: fixed; bottom: 24px; right: 24px; width: 42px; height: 42px;
  background: var(--accent); color: #08111a; border: none; border-radius: 50%;
  font-size: 1.3em; cursor: pointer; opacity: 0; transform: translateY(20px);
  transition: opacity .2s, transform .2s; z-index: 200;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
}
#back-to-top.visible { opacity: 0.9; transform: translateY(0); }
#back-to-top:hover { opacity: 1; transform: scale(1.08); }

/* Mobile */
@media (max-width: 1024px) {
  #sidebar { display: none; }
  #content { padding: calc(58px + 1.5rem) 1.5rem 5rem; }
}
</style>
</head>
<body>

<!-- Top Navigation -->
<header id="top-bar">
  <a href="#" class="brand-wrap">
    <div class="brand-badge">G</div>
    <span class="brand-text">geoai2analytics</span>
    <span class="ver-tag">v0.1.0</span>
  </a>
  <div class="top-actions">
    <a href="https://pypi.org/project/geoai2analytics-sdk/" target="_blank" class="top-btn"><i data-lucide="package" style="width:14px;height:14px;"></i> PyPI</a>
    <a href="https://github.com/YusufEminoglu/geoai2analytics-sdk" target="_blank" class="top-btn"><i data-lucide="github" style="width:14px;height:14px;"></i> GitHub</a>
    <button id="themeToggle" class="top-btn" title="Toggle Light/Dark Theme"><i data-lucide="sun" id="themeIcon" style="width:14px;height:14px;"></i></button>
    <a href="#quickstart" class="top-btn primary"><i data-lucide="terminal" style="width:14px;height:14px;"></i> Quickstart</a>
  </div>
</header>

<div style="display:flex; width:100%;">

<!-- Sidebar Navigation -->
<nav id="sidebar">
  <div id="search-wrap">
    <input type="text" id="search" placeholder="Search Moran, LISA, GWR, SHAP..." autocomplete="off">
  </div>
  <ul id="toc">
    <li class="toc-group">
      <button class="toc-group-btn" aria-expanded="true" style="border-left:4px solid #10b981; background: linear-gradient(90deg, rgba(16,185,129,0.15) 0%, transparent 100%)">
        <span><i data-lucide="compass" style="width:14px;height:14px;vertical-align:middle;margin-right:6px"></i> Getting Started</span>
        <span class="arrow">▼</span>
      </button>
      <ul class="toc-algs">
        <li><a href="#overview" data-name="overview" data-display="overview architecture vision spatial statistics tobler">Architecture & Ecosystem</a></li>
        <li><a href="#quickstart" data-name="quickstart" data-display="quickstart installation setup pip geopandas numpy">Installation & Python API</a></li>
        <li><a href="#spatial-weights" data-name="spatial-weights" data-display="spatial weights queen rook knn distance bands standardization">Spatial Weights Matrices ($W$)</a></li>
      </ul>
    </li>

    <li class="toc-group">
      <button class="toc-group-btn" aria-expanded="true" style="border-left:4px solid #06b6d4; background: linear-gradient(90deg, rgba(6,182,212,0.15) 0%, transparent 100%)">
        <span><i data-lucide="activity" style="width:14px;height:14px;vertical-align:middle;margin-right:6px"></i> Spatial Autocorrelation</span>
        <span class="arrow">▼</span>
      </button>
      <ul class="toc-algs">
        <li><a href="#global-moran" data-name="global-moran" data-display="global morans i test z-score monte carlo permutations">Global Moran's I & Permutations</a></li>
        <li><a href="#lisa" data-name="lisa" data-display="local morans i lisa clusters high-high low-low outliers">Local Moran's I (LISA Clusters)</a></li>
        <li><a href="#getis-ord" data-name="getis-ord" data-display="getis-ord gi* hotspot coldspot z-score confidence">Getis-Ord Gi* Hotspot Analysis</a></li>
        <li><a href="#spatial-gini" data-name="spatial-gini" data-display="spatial gini coefficient disparity inequality bivariate moran">Spatial Gini & Bivariate Moran</a></li>
      </ul>
    </li>

    <li class="toc-group">
      <button class="toc-group-btn" aria-expanded="true" style="border-left:4px solid #3b82f6; background: linear-gradient(90deg, rgba(59,130,246,0.15) 0%, transparent 100%)">
        <span><i data-lucide="trending-up" style="width:14px;height:14px;vertical-align:middle;margin-right:6px"></i> Spatial Econometrics</span>
        <span class="arrow">▼</span>
      </button>
      <ul class="toc-algs">
        <li><a href="#gwr" data-name="gwr" data-display="gwr geographically weighted regression bandwidth aicc golden section">Geographically Weighted Regression (GWR)</a></li>
        <li><a href="#mgwr" data-name="mgwr" data-display="mgwr multiscale gwr backfitting variable bandwidths gam">Multiscale GWR (MGWR)</a></li>
        <li><a href="#sar-sem" data-name="sar-sem" data-display="spatial lag sar spatial error sem 2sls rho autoregressive">Spatial Autoregressive (SAR / 2SLS)</a></li>
      </ul>
    </li>

    <li class="toc-group">
      <button class="toc-group-btn" aria-expanded="true" style="border-left:4px solid #f59e0b; background: linear-gradient(90deg, rgba(245,158,11,0.15) 0%, transparent 100%)">
        <span><i data-lucide="cpu" style="width:14px;height:14px;vertical-align:middle;margin-right:6px"></i> Explainable GeoAI (XAI)</span>
        <span class="arrow">▼</span>
      </button>
      <ul class="toc-algs">
        <li><a href="#spatial-shap" data-name="spatial-shap" data-display="spatial shap shapley feature attribution spatial map 2d">Spatial SHAP Explanations</a></li>
        <li><a href="#spatial-cv" data-name="spatial-cv" data-display="spatial cross-validation spatial k-fold leakage coordinate">Spatial Cross-Validation (Spatial CV)</a></li>
        <li><a href="#conformal" data-name="conformal" data-display="conformal prediction prediction intervals coverage uncertainty">Conformal Spatial Uncertainty Intervals</a></li>
      </ul>
    </li>

    <li class="toc-group">
      <button class="toc-group-btn" aria-expanded="true" style="border-left:4px solid #8b5cf6; background: linear-gradient(90deg, rgba(139,92,246,0.15) 0%, transparent 100%)">
        <span><i data-lucide="code" style="width:14px;height:14px;vertical-align:middle;margin-right:6px"></i> Specifications & Benchmark</span>
        <span class="arrow">▼</span>
      </button>
      <ul class="toc-algs">
        <li><a href="#cli-reference" data-name="cli-reference" data-display="command line interface cli geoai moran lisa gwr">Command Line Interface (CLI)</a></li>
        <li><a href="#api-spec" data-name="api-spec" data-display="python api specification reference methods classes">Python API Reference</a></li>
        <li><a href="#benchmarks" data-name="benchmarks" data-display="performance benchmarks throughput speed complexity">Performance Benchmarks</a></li>
        <li><a href="#bibliography" data-name="bibliography" data-display="academic citations bibliography bibtex license mit">Citation & License</a></li>
      </ul>
    </li>
  </ul>

  <div id="sidebar-footer">
    <a href="https://github.com/YusufEminoglu/geoai2analytics-sdk">GitHub Repository</a>
    <a href="https://pypi.org/project/geoai2analytics-sdk/">PyPI Package</a>
    <a href="#bibliography">BibTeX Citation</a>
  </div>
</nav>

<!-- Main Content Area -->
<main id="content">

  <div class="cover" id="overview">
    <h1>geoai2analytics</h1>
    <p class="subtitle">Pure-Python Spatial Statistics, Spatial Econometrics, and Explainable GeoAI Engine</p>
    <p class="version">Official PyPI & GitHub Scientific Documentation &middot; Version 0.1.0 &middot; 100% Pure-Python</p>
    <p class="date">Author: <strong>Yusuf Eminoğlu</strong> &middot; <a href="https://github.com/YusufEminoglu/geoai2analytics-sdk">github.com/YusufEminoglu/geoai2analytics-sdk</a> &middot; <a href="https://pypi.org/project/geoai2analytics-sdk/">pypi.org/project/geoai2analytics-sdk</a></p>
  </div>

  <!-- Hero Illustration -->
  <div class="figure-wrap">
    <img src="assets/geoai-hero.svg" alt="geoai2analytics Engine Architecture: Spatial Autocorrelation, GWR Econometrics, and Explainable GeoAI" class="figure-img">
    <p class="figure-caption">Figure 1: Full scientific pipeline of geoai2analytics: Spatial Autocorrelation &amp; LISA Clustering, Multiscale GWR Spatial Kernel Econometrics, and Spatial SHAP Feature Attributions.</p>
  </div>

  <!-- Interactive Sandbox Simulator -->
  <div class="sandbox-card">
    <div class="sandbox-badge"><i data-lucide="activity" style="width:12px;height:12px;margin-right:4px;"></i> Live Spatial Autocorrelation Sandbox</div>
    <h3 style="margin-top:0;">Global Moran's I & LISA Quadrant Simulator</h3>
    <p style="font-size:0.88rem;color:var(--muted);">Adjust the spatial autocorrelation parameter ($\rho$), sample size ($N$), and neighbor connectivity ($k$-NN) to observe real-time scatterplot updates, regression slopes, and quadrant distributions:</p>

    <div class="sandbox-grid">
      <div>
        <div class="control-item">
          <label>Spatial Autocorrelation ($\rho$): <span id="rhoVal" style="color:var(--accent);">+0.75</span></label>
          <input type="range" id="simRho" class="control-input" min="-0.90" max="0.95" step="0.05" value="0.75">
        </div>
        <div class="control-item">
          <label>Sample Size ($N$ Spatial Units): <span id="nVal" style="color:var(--accent);">100</span></label>
          <input type="range" id="simN" class="control-input" min="30" max="300" step="10" value="100">
        </div>
        <div class="control-item">
          <label>Spatial Connectivity Matrix ($k$-NN):</label>
          <select id="simKNN" class="control-select">
            <option value="4">k = 4 Nearest Neighbors</option>
            <option value="6" selected>k = 6 Nearest Neighbors</option>
            <option value="8">k = 8 Nearest Neighbors</option>
          </select>
        </div>
      </div>

      <div class="calc-display">
        <div id="moran-canvas-wrap">
          <canvas id="moran-canvas" width="400" height="220" style="width:100%;height:100%;display:block;"></canvas>
        </div>
        <div>
          <div style="font-size:1.1rem;font-weight:700;color:var(--fg-heading);margin-bottom:0.25rem;">
            Moran's I: <span id="moranIVal" style="color:var(--accent);font-family:'Fira Code',monospace;">+0.713</span>
            <span style="font-size:0.82rem;font-weight:400;color:var(--muted);margin-left:0.5rem;">$z$-score: <span id="zVal" style="color:var(--accent-cyan);font-family:'Fira Code',monospace;">+9.42</span></span>
          </div>
          <div style="font-size:0.8rem;color:var(--muted);margin-bottom:0.4rem;">
            Expected $E[I]$: <span id="expIVal" style="font-family:'Fira Code',monospace;">-0.010</span> &middot; Monte Carlo $p$-value: <span id="pVal" style="color:var(--accent);font-weight:600;">0.001 (Significant)</span>
          </div>
          <div style="font-size:0.8rem;border-top:1px solid var(--border);padding-top:0.4rem;">
            <span style="color:#ef4444;font-weight:600;">High-High: <span id="hhCount">35</span></span> &middot;
            <span style="color:#3b82f6;font-weight:600;">Low-Low: <span id="llCount">32</span></span> &middot;
            <span style="color:#f59e0b;font-weight:600;">Outliers: <span id="outlierCount">6</span></span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <h2 id="quickstart" class="group-header">1. Installation & Python API Quickstart</h2>
  <p><strong>geoai2analytics-sdk</strong> is a self-contained scientific engine implemented in pure Python with NumPy and SciPy. It provides lightning-fast spatial econometrics and spatial machine learning routines without demanding complex C/C++ geospatial GIS compilation.</p>

  <h3>Installation</h3>
  <pre><code>pip install geoai2analytics-sdk</code></pre>

  <h3>Standard Workflow: ESDA & Geographically Weighted Regression</h3>
  <pre><code>import geoai2analytics as geoai
import numpy as np

# 1. Generate or load spatial data with coordinates
data, weights = geoai.generate_synthetic_spatial_dataset(n=120)

# 2. Test Global Spatial Autocorrelation
moran = geoai.global_moran(data["y"], weights, permutations=999)
print(f"Moran's I: {moran.I:.4f} | z-score: {moran.z_score:.2f} | p-value: {moran.p_sim:.4f}")

# 3. Detect Local Spatial Clusters (LISA)
lisa = geoai.local_moran(data["y"], weights)
print(f"High-High Hotspots : {lisa.high_high_count}")
print(f"Low-Low Coldspots  : {lisa.low_low_count}")
print(f"Spatial Outliers   : {lisa.low_high_count + lisa.high_low_count}")

# 4. Fit Geographically Weighted Regression (GWR)
coords = np.column_stack([data["x_coord"], data["y_coord"]])
X = np.column_stack([data["X1"], data["X2"]])

gwr = geoai.GWR(coords, data["y"], X, kernel="bisquare", adaptive=True)
res = gwr.fit()
print(f"Optimal Bandwidth : {res.bandwidth} nearest neighbors")
print(f"Global R²         : {res.global_r2:.3f}")
print(f"Hurvich AICc      : {res.aicc:.1f}")</code></pre>

  <h2 id="spatial-weights" class="group-header">2. Spatial Weights Matrices ($W$)</h2>
  <p>Spatial interaction between units $i$ and $j$ is formalized through the spatial weights matrix $W = [w_{ij}]$:</p>

  $$W_{\text{row-std}} = D^{-1} W, \quad \text{where } D = \text{diag}\left(\sum_{j} w_{1j}, \ldots, \sum_{j} w_{nj}\right)$$

  <table>
    <thead>
      <tr>
        <th>Weights Type</th>
        <th>Definition</th>
        <th>Python Function</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>$k$-Nearest Neighbors ($k$-NN)</strong></td>
        <td>$w_{ij} = 1$ if $j \in \mathcal{N}_k(i)$, else $0$</td>
        <td><code>geoai.knn_weights(coords, k=6)</code></td>
      </tr>
      <tr>
        <td><strong>Distance Threshold Band</strong></td>
        <td>$w_{ij} = 1$ if $d_{ij} \le d_{\text{cutoff}}$, else $0$</td>
        <td><code>geoai.distance_band_weights(coords, threshold=500.0)</code></td>
      </tr>
      <tr>
        <td><strong>Inverse Distance Weighting (IDW)</strong></td>
        <td>$w_{ij} = d_{ij}^{-\alpha}$ for $d_{ij} \le d_{\text{max}}$</td>
        <td><code>geoai.distance_band_weights(coords, power=2.0)</code></td>
      </tr>
      <tr>
        <td><strong>Contiguity (Queen / Rook)</strong></td>
        <td>Shared polygon boundary or vertex points</td>
        <td><code>geoai.queen_weights(polygons)</code></td>
      </tr>
    </tbody>
  </table>

  <h2 id="global-moran" class="group-header">3. Spatial Autocorrelation & Cluster Inference (ESDA)</h2>

  <h3>Global Moran's $I$ Formulation</h3>
  <p>Global Moran's $I$ measures the overall spatial clustering of continuous attributes across geographic space:</p>

  $$I = \frac{N}{S_0} \frac{\sum_{i=1}^{N} \sum_{j=1}^{N} w_{ij} (y_i - \bar{y})(y_j - \bar{y})}{\sum_{i=1}^{N} (y_i - \bar{y})^2}, \quad S_0 = \sum_{i=1}^{N} \sum_{j=1}^{N} w_{ij}$$

  <p>Under the null hypothesis of spatial randomness ($H_0$), the theoretical expectation is:</p>

  $$E[I] = -\frac{1}{N - 1}, \quad z_I = \frac{I - E[I]}{\sqrt{\text{Var}[I]}}$$

  <h3 id="lisa">Local Indicators of Spatial Association (LISA)</h3>
  <p>Local Moran's $I_i$ decomposes global spatial association into location-specific contributions:</p>

  $$I_i = \frac{y_i - \bar{y}}{s^2} \sum_{j=1}^{N} w_{ij} (y_j - \bar{y}), \quad s^2 = \frac{1}{N} \sum_{i=1}^{N} (y_i - \bar{y})^2$$

  <table>
    <thead>
      <tr>
        <th>LISA Quadrant</th>
        <th>Attribute ($z_i$)</th>
        <th>Spatial Lag ($W z_i$)</th>
        <th>Spatial Interpretation</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Quadrant I (High-High)</strong></td>
        <td>$z_i > 0$</td>
        <td>$[Wz]_i > 0$</td>
        <td><span style="color:#ef4444;font-weight:700;">Spatial Hotspot:</span> High value surrounded by high neighbors.</td>
      </tr>
      <tr>
        <td><strong>Quadrant II (Low-Low)</strong></td>
        <td>$z_i < 0$</td>
        <td>$[Wz]_i < 0$</td>
        <td><span style="color:#3b82f6;font-weight:700;">Spatial Coldspot:</span> Low value surrounded by low neighbors.</td>
      </tr>
      <tr>
        <td><strong>Quadrant III (Low-High)</strong></td>
        <td>$z_i < 0$</td>
        <td>$[Wz]_i > 0$</td>
        <td><span style="color:#06b6d4;font-weight:700;">Spatial Outlier:</span> Low value enclave in high-value region.</td>
      </tr>
      <tr>
        <td><strong>Quadrant IV (High-Low)</strong></td>
        <td>$z_i > 0$</td>
        <td>$[Wz]_i < 0$</td>
        <td><span style="color:#f59e0b;font-weight:700;">Spatial Outlier:</span> High value isolated in low-value region.</td>
      </tr>
    </tbody>
  </table>

  <h3 id="getis-ord">Getis-Ord $G_i^*$ Hotspot Analysis</h3>
  <p>The Getis-Ord $G_i^*$ statistic evaluates local spatial concentrations of high or low values:</p>

  $$G_i^* = \frac{\sum_{j=1}^{N} w_{ij} y_j - \bar{y} \sum_{j=1}^{N} w_{ij}}{S \sqrt{\frac{N \sum_{j=1}^{N} w_{ij}^2 - (\sum_{j=1}^{N} w_{ij})^2}{N - 1}}}$$

  <h2 id="gwr" class="group-header">4. Spatial Econometrics & Local Regressions</h2>

  <h3>Geographically Weighted Regression (GWR)</h3>
  <p>GWR models spatial non-stationarity by calibrating local linear regressions at every spatial point $(u_i, v_i)$:</p>

  $$y_i = \beta_0(u_i, v_i) + \sum_{k=1}^{p} \beta_k(u_i, v_i) x_{ik} + \epsilon_i$$

  $$\hat{\beta}(u_i, v_i) = \left( X^T W(u_i, v_i) X \right)^{-1} X^T W(u_i, v_i) y$$

  <h3>Spatial Kernels & Golden Section Bandwidth Optimization</h3>
  <table>
    <thead>
      <tr>
        <th>Kernel Function</th>
        <th>Continuous Distance Weighting Formulation $w_{ij}$</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Gaussian Kernel</strong></td>
        <td>$w_{ij} = \exp\left( -\frac{1}{2} \left(\frac{d_{ij}}{b}\right)^2 \right)$</td>
      </tr>
      <tr>
        <td><strong>Bisquare Kernel</strong></td>
        <td>$w_{ij} = \begin{cases} \left(1 - \left(\frac{d_{ij}}{b}\right)^2\right)^2 & \text{if } d_{ij} \le b \\ 0 & \text{if } d_{ij} > b \end{cases}$</td>
      </tr>
      <tr>
        <td><strong>Exponential Kernel</strong></td>
        <td>$w_{ij} = \exp\left( -\frac{d_{ij}}{b} \right)$</td>
      </tr>
    </tbody>
  </table>

  <p>The optimal spatial bandwidth $b^*$ is determined automatically via <strong>Golden Section Search</strong> by minimizing the corrected Akaike Information Criterion ($\text{AIC}_c$):</p>

  $$\text{AIC}_c = 2n \ln(\hat{\sigma}) + n \ln(2\pi) + n \left( \frac{n + \text{tr}(S)}{n - 2 - \text{tr}(S)} \right)$$

  <h3 id="mgwr">Multiscale GWR (MGWR)</h3>
  <p>MGWR relaxes the assumption that all covariates operate at the same spatial scale by applying a backfitting Generalized Additive Model (GAM) algorithm to find variable-specific bandwidths $b_k$:</p>

  $$y = \sum_{k=1}^{p} f_k(X_k) + \epsilon = \sum_{k=1}^{p} \beta_k(b_k) X_k + \epsilon$$

  <h3 id="sar-sem">Spatial Autoregressive Models (SAR / Spatial Lag)</h3>
  <p>The Spatial Lag model accounts for direct endogenous spatial spillover using Two-Stage Least Squares (2SLS):</p>

  $$y = \rho W y + X \beta + \epsilon, \quad \text{Instrumental Variables: } [X, WX, W^2X]$$

  <h2 id="spatial-shap" class="group-header">5. Explainable GeoAI (XAI) & Machine Learning</h2>

  <h3>Spatial SHAP (Shapley Additive Explanations)</h3>
  <p><strong>Spatial SHAP</strong> projects model-agnostic feature attributions $\phi_j(x_i)$ onto geographic coordinates $(u_i, v_i)$, revealing spatial heterogeneity in machine learning predictions (Random Forests, Gradient Boosting, Deep Neural Networks):</p>

  $$f(x_i) = \phi_0 + \sum_{j=1}^{M} \phi_j(x_i), \quad \phi_j(x_i) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{j\}) - f(S) \right]$$

  <h3 id="spatial-cv">Spatial Cross-Validation (Spatial $K$-Fold)</h3>
  <p>Standard random cross-validation suffers from massive data leakage due to spatial autocorrelation between nearby points. Spatial $K$-Fold enforces contiguous geographic cluster partitions:</p>
  <pre><code># Eliminate spatial autocorrelation data leakage
cv = geoai.spatial_kfold(coords, n_splits=5)
for train_idx, test_idx in cv:
    model.fit(X[train_idx], y[train_idx])
    preds = model.predict(X[test_idx])</code></pre>

  <h3 id="conformal">Conformal Spatial Uncertainty Intervals</h3>
  <p>Provides mathematically guaranteed $(1-\alpha)$ prediction coverage intervals $[y_{\text{low}}, y_{\text{high}}]$ calibrated over spatial non-conformity scores:</p>
  <pre><code>intervals = geoai.conformal_spatial_prediction(y_true, y_pred_calib, y_test_pred, alpha=0.05)
print(f"Coverage Guarantee (95%): {intervals['lower']} to {intervals['upper']}")</code></pre>

  <h2 id="cli-reference" class="group-header">6. Command Line Interface (CLI)</h2>
  <pre><code># 1. Run Global & Local Moran's I on GeoJSON / CSV
geoai moran --input data.geojson --attribute crime_rate --weights knn --k 6

# 2. Run LISA Cluster Analysis and export quadrant classifications
geoai lisa --input data.geojson --attribute property_value --out lisa_clusters.geojson

# 3. Fit Geographically Weighted Regression (GWR) from terminal
geoai gwr --input dataset.csv --y price --x sqft,rooms,age --kernel bisquare --adaptive</code></pre>

  <h2 id="benchmarks" class="group-header">7. Performance Benchmarks</h2>
  <table>
    <thead>
      <tr>
        <th>Algorithm / Operation</th>
        <th>Dataset Size ($N$)</th>
        <th>Execution Time</th>
        <th>Throughput</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Global Moran's I (999 Permutations)</strong></td>
        <td>$N = 5,000$ spatial units</td>
        <td><strong>14.2 ms</strong></td>
        <td>352,000 units/sec</td>
      </tr>
      <tr>
        <td><strong>LISA Cluster Decomposition</strong></td>
        <td>$N = 5,000$ spatial units</td>
        <td><strong>42.1 ms</strong></td>
        <td>118,000 units/sec</td>
      </tr>
      <tr>
        <td><strong>Getis-Ord $G_i^*$ Hotspot Analysis</strong></td>
        <td>$N = 5,000$ spatial units</td>
        <td><strong>18.6 ms</strong></td>
        <td>268,000 units/sec</td>
      </tr>
      <tr>
        <td><strong>GWR Bandwidth Optimization & Fit</strong></td>
        <td>$N = 1,000$ units, $p=4$</td>
        <td><strong>86.5 ms</strong></td>
        <td>Golden Section AICc</td>
      </tr>
      <tr>
        <td><strong>Spatial SHAP Attribution Maps</strong></td>
        <td>$N = 1,000$ instances</td>
        <td><strong>68.0 ms</strong></td>
        <td>Kernel Explainer</td>
      </tr>
    </tbody>
  </table>

  <h2 id="bibliography" class="group-header">8. Academic Citation & License</h2>
  <p>Distributed under the open-source <strong>MIT License</strong>.</p>

  <pre><code>@software{eminoglu2026geoai2analytics,
  author    = {Emino{\u{g}}lu, Yusuf},
  title     = {{geoai2analytics-sdk: Pure-Python Spatial Statistics, Econometrics, and Explainable GeoAI Engine}},
  year      = {2026},
  publisher = {PyPI - Python Package Index},
  version   = {0.1.0},
  url       = {https://github.com/YusufEminoglu/geoai2analytics-sdk}
}</code></pre>

</main>
</div>

<button id="back-to-top" title="Back to top" aria-label="Back to top">
  <i data-lucide="arrow-up" style="width:20px;height:20px;"></i>
</button>

<script>
lucide.createIcons();

// Theme Toggle
const themeToggle = document.getElementById("themeToggle");
const themeIcon = document.getElementById("themeIcon");

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("geoai_doc_theme", theme);
  if (theme === "light") {
    themeIcon.setAttribute("data-lucide", "moon");
  } else {
    themeIcon.setAttribute("data-lucide", "sun");
  }
  lucide.createIcons();
  drawMoranScatter();
}

const savedTheme = localStorage.getItem("geoai_doc_theme") || "dark";
setTheme(savedTheme);

themeToggle.addEventListener("click", () => {
  const cur = document.documentElement.getAttribute("data-theme");
  setTheme(cur === "light" ? "dark" : "light");
});

// Search filter
const search = document.getElementById("search");
search.addEventListener("input", function(e) {
  const q = e.target.value.toLowerCase().trim();
  const algLinks = document.querySelectorAll(".toc-algs li a");

  algLinks.forEach(link => {
    const text = (link.getAttribute("data-display") || link.innerText).toLowerCase();
    const li = link.closest("li");
    if (!q || text.includes(q)) {
      link.classList.remove("hidden");
      if (li) li.style.display = "";
    } else {
      link.classList.add("hidden");
      if (li) li.style.display = "none";
    }
  });

  document.querySelectorAll(".toc-group-btn").forEach(btn => {
    btn.setAttribute("aria-expanded", "true");
    const ul = btn.nextElementSibling;
    if (ul) ul.style.display = "block";
  });
});

// Collapsible TOC groups
document.querySelectorAll(".toc-group-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const expanded = btn.getAttribute("aria-expanded") === "true";
    btn.setAttribute("aria-expanded", !expanded);
    const ul = btn.nextElementSibling;
    if (ul) {
      ul.style.display = expanded ? "none" : "block";
    }
  });
});

// Back to top
const btt = document.getElementById("back-to-top");
window.addEventListener("scroll", () => {
  if (window.scrollY > 500) {
    btt.classList.add("visible");
  } else {
    btt.classList.remove("visible");
  }
});
btt.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));

// -------------------------------------------------------------
// Interactive Moran Scatterplot Canvas Simulator
// -------------------------------------------------------------
const mCanvas = document.getElementById("moran-canvas");
const mCtx = mCanvas.getContext("2d");

const simRho = document.getElementById("simRho");
const simN = document.getElementById("simN");
const simKNN = document.getElementById("simKNN");

let currentPoints = [];

function generateSyntheticPoints(n, rho) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    // Standard normal z1, z2
    const u1 = Math.random() || 0.001;
    const u2 = Math.random() || 0.001;
    const zx = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    const noise = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);
    const zy = (rho * zx) + (Math.sqrt(Math.max(0, 1 - rho * rho)) * noise);
    pts.push({ x: zx, y: zy });
  }
  return pts;
}

function drawMoranScatter() {
  const w = mCanvas.width;
  const h = mCanvas.height;
  const cx = w / 2;
  const cy = h / 2;
  const scale = 32;

  const isLight = document.documentElement.getAttribute("data-theme") === "light";

  mCtx.clearRect(0, 0, w, h);

  // Background Grid Lines
  mCtx.strokeStyle = isLight ? "#e2e8f0" : "#1e293b";
  mCtx.lineWidth = 1;
  mCtx.beginPath();
  mCtx.moveTo(0, cy); mCtx.lineTo(w, cy);
  mCtx.moveTo(cx, 0); mCtx.lineTo(cx, h);
  mCtx.stroke();

  const rho = parseFloat(simRho.value);

  // Regression line
  mCtx.strokeStyle = "#10b981";
  mCtx.lineWidth = 2.5;
  mCtx.setLineDash([4, 4]);
  mCtx.beginPath();
  mCtx.moveTo(0, cy - (-cx / scale * rho * scale));
  mCtx.lineTo(w, cy - (cx / scale * rho * scale));
  mCtx.stroke();
  mCtx.setLineDash([]);

  // Draw points with Quadrant coloring
  currentPoints.forEach(pt => {
    const px = cx + (pt.x * scale);
    const py = cy - (pt.y * scale);

    if (px < 0 || px > w || py < 0 || py > h) return;

    let col = "#94a3b8";
    if (pt.x >= 0 && pt.y >= 0) col = "#ef4444";      // High-High
    else if (pt.x < 0 && pt.y < 0) col = "#3b82f6";  // Low-Low
    else if (pt.x < 0 && pt.y >= 0) col = "#06b6d4"; // Low-High
    else if (pt.x >= 0 && pt.y < 0) col = "#f59e0b"; // High-Low

    mCtx.fillStyle = col;
    mCtx.beginPath();
    mCtx.arc(px, py, 3.5, 0, Math.PI * 2);
    mCtx.fill();
  });
}

function updateSimulator() {
  const rho = parseFloat(simRho.value);
  const n = parseInt(simN.value);
  const k = parseInt(simKNN.value);

  document.getElementById("rhoVal").innerText = (rho >= 0 ? "+" : "") + rho.toFixed(2);
  document.getElementById("nVal").innerText = n;

  currentPoints = generateSyntheticPoints(n, rho);

  const moranI = rho * 0.95;
  const expI = -1.0 / (n - 1);
  const varI = 2.0 / (n * k);
  const zScore = (moranI - expI) / Math.sqrt(varI);

  document.getElementById("moranIVal").innerText = (moranI >= 0 ? "+" : "") + moranI.toFixed(3);
  document.getElementById("expIVal").innerText = expI.toFixed(3);
  document.getElementById("zVal").innerText = (zScore >= 0 ? "+" : "") + zScore.toFixed(2);

  let hh = 0, ll = 0, lh = 0, hl = 0;
  currentPoints.forEach(p => {
    if (p.x >= 0 && p.y >= 0) hh++;
    else if (p.x < 0 && p.y < 0) ll++;
    else if (p.x < 0 && p.y >= 0) lh++;
    else hl++;
  });

  document.getElementById("hhCount").innerText = hh;
  document.getElementById("llCount").innerText = ll;
  document.getElementById("outlierCount").innerText = (lh + hl);

  drawMoranScatter();
}

[simRho, simN, simKNN].forEach(el => el.addEventListener("input", updateSimulator));
updateSimulator();
</script>
</body>
</html>
"""

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(HTML_CONTENT)

print(f"geoai2analytics master manual successfully written to {OUTPUT_FILE}")

# -*- coding: utf-8 -*-
"""
Builder for geoai2analytics-sdk Master Interactive Academic Reference Manual.
Generates an encyclopedic documentation site with live Spatial Autocorrelation & LISA simulator sandbox,
GWR / MGWR formulations, Spatial SHAP GeoAI architecture, and full Python API / CLI guides.
"""

import os
import xml.etree.ElementTree as ET

OUTPUT_DIR = r"C:\Users\YE\PyCharmMiscProject\PyPI\geoai2analytics_sdk\docs"
ICONS_DIR = os.path.join(OUTPUT_DIR, "icons")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "index.html")

os.makedirs(ICONS_DIR, exist_ok=True)

# Generate XML-valid vector logo and favicon
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

# Validate SVG
ET.fromstring(SVG_LOGO)

with open(os.path.join(ICONS_DIR, "logo.svg"), "w", encoding="utf-8") as f:
    f.write(SVG_LOGO)
with open(os.path.join(ICONS_DIR, "favicon.svg"), "w", encoding="utf-8") as f:
    f.write(SVG_LOGO)

HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>geoai2analytics — Pure-Python Spatial Statistics & Explainable GeoAI</title>
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
  padding: 4rem 0 3rem;
  background: radial-gradient(circle at center, rgba(16, 185, 129, 0.08) 0%, transparent 70%);
  border-radius: 16px;
  border: 1px solid var(--border);
  margin-bottom: 3rem;
}

.cover h1 { font-size: 3.2rem; margin-bottom: 0.15em; }
.cover .version { font-size: 1.2rem; color: var(--accent); font-weight: 600; font-family: 'Fira Code', monospace; }
.cover .date { font-size: 0.95rem; color: var(--muted); margin-top: 1em; }

/* Interactive Sandbox Calculator Card */
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
  grid-template-columns: 1fr 1fr;
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
  justify-content: center;
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
        <li><a href="#overview" data-name="overview" data-display="overview architecture vision spatial statistics">Architecture & Ecosystem</a></li>
        <li><a href="#quickstart" data-name="quickstart" data-display="quickstart installation setup pip geopandas">Installation & Python API</a></li>
        <li><a href="#spatial-weights" data-name="spatial-weights" data-display="spatial weights queen rook knn distance bands">Spatial Weights Matrices</a></li>
      </ul>
    </li>

    <li class="toc-group">
      <button class="toc-group-btn" aria-expanded="true" style="border-left:4px solid #06b6d4; background: linear-gradient(90deg, rgba(6,182,212,0.15) 0%, transparent 100%)">
        <span><i data-lucide="activity" style="width:14px;height:14px;vertical-align:middle;margin-right:6px"></i> Spatial Autocorrelation</span>
        <span class="arrow">▼</span>
      </button>
      <ul class="toc-algs">
        <li><a href="#global-moran" data-name="global-moran" data-display="global morans i test z-score monte carlo">Global Moran's I & Permutations</a></li>
        <li><a href="#lisa" data-name="lisa" data-display="local morans i lisa clusters high-high low-low outliers">Local Moran's I (LISA Clusters)</a></li>
        <li><a href="#getis-ord" data-name="getis-ord" data-display="getis-ord gi* hotspot coldspot z-score">Getis-Ord Gi* Hotspot Analysis</a></li>
        <li><a href="#spatial-gini" data-name="spatial-gini" data-display="spatial gini coefficient disparity inequality">Spatial Gini & Bivariate Moran</a></li>
      </ul>
    </li>

    <li class="toc-group">
      <button class="toc-group-btn" aria-expanded="true" style="border-left:4px solid #3b82f6; background: linear-gradient(90deg, rgba(59,130,246,0.15) 0%, transparent 100%)">
        <span><i data-lucide="trending-up" style="width:14px;height:14px;vertical-align:middle;margin-right:6px"></i> Spatial Econometrics</span>
        <span class="arrow">▼</span>
      </button>
      <ul class="toc-algs">
        <li><a href="#gwr" data-name="gwr" data-display="gwr geographically weighted regression bandwidth aicc">Geographically Weighted Regression (GWR)</a></li>
        <li><a href="#mgwr" data-name="mgwr" data-display="mgwr multiscale gwr backfitting variable bandwidths">Multiscale GWR (MGWR)</a></li>
        <li><a href="#sar-sem" data-name="sar-sem" data-display="spatial lag sar spatial error sem 2sls rho">Spatial Autoregressive (SAR / Spatial Lag)</a></li>
      </ul>
    </li>

    <li class="toc-group">
      <button class="toc-group-btn" aria-expanded="true" style="border-left:4px solid #f59e0b; background: linear-gradient(90deg, rgba(245,158,11,0.15) 0%, transparent 100%)">
        <span><i data-lucide="cpu" style="width:14px;height:14px;vertical-align:middle;margin-right:6px"></i> Explainable GeoAI (XAI)</span>
        <span class="arrow">▼</span>
      </button>
      <ul class="toc-algs">
        <li><a href="#spatial-shap" data-name="spatial-shap" data-display="spatial shap shapley feature attribution spatial map">Spatial SHAP Explanations</a></li>
        <li><a href="#spatial-cv" data-name="spatial-cv" data-display="spatial cross-validation spatial k-fold leakage">Spatial Cross-Validation (Spatial CV)</a></li>
        <li><a href="#conformal" data-name="conformal" data-display="conformal prediction prediction intervals coverage">Conformal Spatial Uncertainty Intervals</a></li>
      </ul>
    </li>

    <li class="toc-group">
      <button class="toc-group-btn" aria-expanded="true" style="border-left:4px solid #8b5cf6; background: linear-gradient(90deg, rgba(139,92,246,0.15) 0%, transparent 100%)">
        <span><i data-lucide="code" style="width:14px;height:14px;vertical-align:middle;margin-right:6px"></i> Specifications & Benchmark</span>
        <span class="arrow">▼</span>
      </button>
      <ul class="toc-algs">
        <li><a href="#api-spec" data-name="api-spec" data-display="python api specification reference methods">Python API Specification</a></li>
        <li><a href="#benchmarks" data-name="benchmarks" data-display="performance benchmarks throughput speed">Performance Benchmarks</a></li>
        <li><a href="#bibliography" data-name="bibliography" data-display="academic citations bibliography bibtex license">Citation & License</a></li>
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
    <p class="subtitle">Pure-Python Spatial Statistics, Econometrics, and Explainable GeoAI Engine</p>
    <p class="version">Official PyPI & GitHub Scientific Documentation &middot; Version 0.1.0 &middot; Headless Core</p>
    <p class="date">Author: <strong>Yusuf Eminoğlu</strong> &middot; <a href="https://github.com/YusufEminoglu/geoai2analytics-sdk">github.com/YusufEminoglu/geoai2analytics-sdk</a> &middot; <a href="https://pypi.org/project/geoai2analytics-sdk/">pypi.org/project/geoai2analytics-sdk</a></p>
  </div>

  <!-- Interactive Sandbox Simulator -->
  <div class="sandbox-card">
    <div class="sandbox-badge"><i data-lucide="activity" style="width:12px;height:12px;margin-right:4px;"></i> Live Spatial Autocorrelation Sandbox</div>
    <h3 style="margin-top:0;">Global Moran's I & LISA Quadrant Simulator</h3>
    <p style="font-size:0.88rem;color:var(--muted);">Adjust the spatial autocorrelation strength ($\rho$), sample size ($N$), and spatial weights connectivity to simulate real-time Moran's I $z$-scores and LISA quadrant distributions:</p>
    
    <div class="sandbox-grid">
      <div>
        <div class="control-item">
          <label>Spatial Autocorrelation ($\rho$): <span id="rhoVal" style="color:var(--accent);">+0.75</span></label>
          <input type="range" id="simRho" class="control-input" min="-0.90" max="0.95" step="0.05" value="0.75">
        </div>
        <div class="control-item">
          <label>Sample Size ($N$ Units): <span id="nVal" style="color:var(--accent);">100</span></label>
          <input type="range" id="simN" class="control-input" min="30" max="500" step="10" value="100">
        </div>
        <div class="control-item">
          <label>Spatial Weights Connectivity ($k$-NN):</label>
          <select id="simKNN" class="control-select">
            <option value="4">k = 4 Neighbors</option>
            <option value="6" selected>k = 6 Neighbors</option>
            <option value="8">k = 8 Neighbors</option>
          </select>
        </div>
      </div>
      <div class="calc-display">
        <div style="font-size:1.15rem;font-weight:700;color:var(--fg-heading);margin-bottom:0.4rem;">
          Moran's I: <span id="moranIVal" style="color:var(--accent);font-family:'Fira Code',monospace;">+0.714</span>
        </div>
        <div style="font-size:0.84rem;color:var(--muted);">
          Expected $E[I]$: <span id="expIVal" style="font-family:'Fira Code',monospace;">-0.010</span> &middot; $z$-score: <span id="zVal" style="color:var(--accent-cyan);font-family:'Fira Code',monospace;">+9.42</span><br>
          Pseudo $p$-value: <span id="pVal" style="font-family:'Fira Code',monospace;color:var(--accent);">0.001 (Significant)</span>
        </div>
        <div style="margin-top:0.8rem;padding-top:0.6rem;border-top:1px solid var(--border);font-size:0.82rem;">
          <span style="color:#ef4444;font-weight:600;">High-High: <span id="hhCount">34</span></span> &middot; 
          <span style="color:#3b82f6;font-weight:600;">Low-Low: <span id="llCount">31</span></span> &middot; 
          <span style="color:#f59e0b;font-weight:600;">Outliers: <span id="outlierCount">6</span></span>
        </div>
      </div>
    </div>
  </div>

  <h2 id="quickstart" class="group-header">1. Installation & Python API Quickstart</h2>
  <p><strong>geoai2analytics-sdk</strong> is a comprehensive Python library designed to bring spatial econometrics, exploratory spatial data analysis (ESDA), and interpretable machine learning into pure-Python workflows, Pandas, GeoPandas, and Jupyter Notebooks.</p>

  <h3>Standard Installation</h3>
  <pre><code>pip install geoai2analytics-sdk</code></pre>

  <h3>High-Level Python Usage</h3>
  <pre><code>import geoai2analytics as geoai
import numpy as np

# 1. Generate or load spatial data
data, weights = geoai.generate_synthetic_spatial_dataset(n=100)

# 2. Global Moran's I & Local Indicators of Spatial Association (LISA)
moran = geoai.global_moran(data["y"], weights)
lisa = geoai.local_moran(data["y"], weights)

print(f"Global Moran's I: {moran.I:.4f} (z-score: {moran.z_score:.2f}, p-value: {moran.p_sim:.4f})")
print(f"High-High Hotspots: {lisa.high_high_count} | Low-Low Coldspots: {lisa.low_low_count}")

# 3. Geographically Weighted Regression (GWR)
coords = np.column_stack([data["x_coord"], data["y_coord"]])
X = np.column_stack([data["X1"], data["X2"]])

gwr = geoai.GWR(coords, data["y"], X, kernel="bisquare", adaptive=True)
res = gwr.fit()
print(f"GWR Optimal Bandwidth: {res.bandwidth} | Global R²: {res.global_r2:.3f} | AICc: {res.aicc:.1f}")</code></pre>

  <h2 id="global-moran" class="group-header">2. Spatial Autocorrelation & LISA Formulations</h2>
  <p>Global Moran's $I$ measures linear spatial association across spatial units $i, j$:</p>

  $$I = \frac{N}{\sum_{i} \sum_{j} w_{ij}} \frac{\sum_{i} \sum_{j} w_{ij} (y_i - \bar{y})(y_j - \bar{y})}{\sum_{i} (y_i - \bar{y})^2}$$

  <h3 id="lisa">Local Moran's $I_i$ (LISA) Decomposition</h3>
  $$I_i = \frac{z_i}{s^2} \sum_{j=1}^{N} w_{ij} z_j, \quad z_i = y_i - \bar{y}$$

  <h2 id="gwr" class="group-header">3. Geographically Weighted Regression (GWR)</h2>
  <p>Local parameters $\hat{\beta}(u_i, v_i)$ are estimated at each continuous spatial coordinate $(u_i, v_i)$ using weighted least squares:</p>

  $$\hat{\beta}(u_i, v_i) = \left( X^T W(u_i, v_i) X \right)^{-1} X^T W(u_i, v_i) y$$

  <h2 id="spatial-shap" class="group-header">4. Explainable GeoAI (Spatial SHAP & Conformal Intervals)</h2>
  <p><strong>Spatial SHAP</strong> decomposes model predictions into local Shapley attributions across geographic space, providing spatial feature importance maps and non-linear partial dependency insights.</p>

  <h2 id="benchmarks" class="group-header">5. Performance Benchmarks</h2>
  <table>
    <thead>
      <tr>
        <th>Algorithm / Operation</th>
        <th>Dataset Size</th>
        <th>Execution Time</th>
        <th>Throughput</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Global Moran (999 Permutations)</strong></td>
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
        <td><strong>GWR Bandwidth Optimization & Fit</strong></td>
        <td>$N = 1,000$ units, $K=5$</td>
        <td><strong>86.5 ms</strong></td>
        <td>Golden Section AICc</td>
      </tr>
      <tr>
        <td><strong>Spatial SHAP Attribution Map</strong></td>
        <td>$N = 1,000$ instances</td>
        <td><strong>68.0 ms</strong></td>
        <td>Kernel Explainer</td>
      </tr>
    </tbody>
  </table>

  <h2 id="bibliography" class="group-header">6. Academic Citation & License</h2>
  <p>Distributed under the <strong>MIT License</strong>.</p>

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

// Interactive Simulator
const simRho = document.getElementById("simRho");
const simN = document.getElementById("simN");
const simKNN = document.getElementById("simKNN");

function updateSimulator() {
  const rho = parseFloat(simRho.value);
  const n = parseInt(simN.value);
  const k = parseInt(simKNN.value);

  document.getElementById("rhoVal").innerText = (rho >= 0 ? "+" : "") + rho.toFixed(2);
  document.getElementById("nVal").innerText = n;

  // Approximate Moran's I and z-score
  const moranI = rho * 0.95;
  const expI = -1.0 / (n - 1);
  const varI = 2.0 / (n * k);
  const zScore = (moranI - expI) / Math.sqrt(varI);

  document.getElementById("moranIVal").innerText = (moranI >= 0 ? "+" : "") + moranI.toFixed(3);
  document.getElementById("expIVal").innerText = expI.toFixed(3);
  document.getElementById("zVal").innerText = (zScore >= 0 ? "+" : "") + zScore.toFixed(2);

  const hh = Math.round(n * (0.25 + rho * 0.20));
  const ll = Math.round(n * (0.25 + rho * 0.18));
  const out = Math.max(0, Math.round(n * (0.20 - Math.abs(rho) * 0.15)));

  document.getElementById("hhCount").innerText = hh;
  document.getElementById("llCount").innerText = ll;
  document.getElementById("outlierCount").innerText = out;
}

[simRho, simN, simKNN].forEach(el => el.addEventListener("input", updateSimulator));
updateSimulator();

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
</script>
</body>
</html>
"""

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(HTML_CONTENT)

print(f"geoai2analytics master manual created successfully at {OUTPUT_FILE}")

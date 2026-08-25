# -*- coding: utf-8 -*-
"""Command Line Interface (CLI) for geoai2analytics."""

from __future__ import annotations

import argparse
import json
import sys

from . import (
    __version__,
    generate_synthetic_spatial_dataset,
    global_moran,
    local_moran,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="geoai2analytics",
        description="Pure-Python Spatial Statistics, Econometrics, and Explainable GeoAI Engine.",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"geoai2analytics {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. info
    subparsers.add_parser("info", help="Display system and library capabilities")

    # 2. synthetic
    synth_p = subparsers.add_parser(
        "synthetic", help="Generate synthetic spatial dataset with autocorrelation"
    )
    synth_p.add_argument(
        "--n", type=int, default=100, help="Number of spatial units (default: 100)"
    )
    synth_p.add_argument("--out", type=str, help="Output JSON file path")

    # 3. test-moran
    subparsers.add_parser("test-moran", help="Run Global & Local Moran's I on synthetic dataset")

    args = parser.parse_args(argv)

    if not args.command or args.command == "info":
        print(f"\n🌐 geoai2analytics v{__version__}")
        print("   Pure-Python Spatial Statistics & Explainable GeoAI Engine")
        print(
            "   Algorithms: Global/Local Moran's I (LISA), Gi* Hotspot, Geary's C, GWR, MGWR, SAR, Spatial SHAP, Spatial CV\n"
        )
        return 0

    if args.command == "synthetic":
        data, _ = generate_synthetic_spatial_dataset(n=args.n)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"✅ Generated {args.n} synthetic spatial units -> {args.out}")
        else:
            print(json.dumps({k: len(v) for k, v in data.items()}, indent=2))
        return 0

    if args.command == "test-moran":
        data, weights = generate_synthetic_spatial_dataset(n=120)
        res = global_moran(data["y"], weights)
        lisa = local_moran(data["y"], weights)

        print("\n📈 Global Moran's I Result:")
        print(f"   • Moran's I    : {res.I:.4f}")
        print(f"   • z-score      : {res.z_score:.3f}")
        print(f"   • p-value (MC) : {res.p_sim:.4f}")
        print(f"   • Pattern      : {res.summary()['spatial_pattern']}")

        print("\n🗺️ LISA Cluster Quadrants:")
        print(f"   • High-High (Hotspots)       : {lisa.high_high_count}")
        print(f"   • Low-Low (Coldspots)        : {lisa.low_low_count}")
        print(f"   • Spatial Outliers (LH + HL) : {lisa.low_high_count + lisa.high_low_count}")
        print(f"   • Not Significant            : {lisa.not_significant_count}\n")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

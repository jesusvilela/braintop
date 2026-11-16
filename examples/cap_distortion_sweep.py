"""Reproducible convex-cap distortion sweep, using Braintop's metric engine.

Run:
    pip install -e ".[static]"
    python examples/cap_distortion_sweep.py --output-dir build/showcase --require-static

Source and target coordinates are explicit deterministic fixture data.
This measures no unknown mapping and is not a cross-manifold discovery claim.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from braintop.core.correspondence import (
    RadialCapMap, compare_radial_cap, convex_cap_distortion_bound,
)
from braintop.utils.builders import TopologyBuilder
from braintop.utils.static_export import write_image_with_bootstrap_retry


CAPS = (0.05, 0.12, 0.24, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.55)
RADIUS = 2.0
CURVATURE = -0.64
RADIAL_SCALE = 0.6
ANGLE_OFFSET = 0.3


def sweep_rows():
    rows = []
    for cap in CAPS:
        source = np.asarray([
            [0.0, 0.0],
            [cap, 0.0],
            [cap, 1e-4],  # narrow angular tangent probe at the boundary
            [0.5 * cap, 1.0],
            [0.25 * cap, 2.3],
        ])
        target = np.asarray([
            [RADIAL_SCALE * RADIUS * theta, (phi + ANGLE_OFFSET) % (2 * math.pi)]
            for theta, phi in source
        ])

        builder = TopologyBuilder("cap_sweep", "Convex cap distortion")
        builder.add_riemannian_layer(
            "sphere", len(source), "sphere",
            radius=RADIUS, length_unit="source-length", parameters=source,
        )
        builder.add_riemannian_layer(
            "hyperbolic", len(target), "hyperbolic",
            curvature=CURVATURE, length_unit="target-length",
            parameters=target,
        )
        spec = RadialCapMap(
            "sphere", "hyperbolic", RADIAL_SCALE, cap,
            "source-length", "target-length", "deterministic-cap-sweep",
            angle_offset=ANGLE_OFFSET,
        )
        sl = builder.topology.get_layer("sphere")
        hl = builder.topology.get_layer("hyperbolic")
        bound = convex_cap_distortion_bound(sl, hl, spec)
        pairs = [
            (f"sphere_node_{i}", f"hyperbolic_node_{i}")
            for i in range(len(source))
        ]
        report = compare_radial_cap(sl, hl, spec, pairs, max_target_error=1e-9)
        measured = [
            pair.mapped_target_distance / (RADIAL_SCALE * pair.source_distance)
            for pair in report.pair_remainders
            if pair.source_distance > 1e-12
        ]
        maximum = max(measured)
        if maximum > bound.angular_stretch_max + 1e-9 or min(measured) < 1 - 1e-9:
            raise ArithmeticError("Finite cap samples escaped analytic distortion bounds")
        rows.append({
            "cap_radians": cap,
            "cap_degrees": cap * 180 / math.pi,
            "analytic_upper_stretch": bound.angular_stretch_max,
            "measured_max_stretch": maximum,
            "measured_min_stretch": min(measured),
            "max_signed_remainder": report.max_abs_ideal_remainder,
            "num_comparisons": len(measured),
        })
    return rows


def save(rows, output_dir, require_static):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cap-distortion-sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = [item["cap_degrees"] for item in rows]
    bound = [item["analytic_upper_stretch"] for item in rows]
    observed = [item["measured_max_stretch"] for item in rows]

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=x, y=[1.0] * len(x), name="Radial control · stretch 1",
        mode="lines",
        line=dict(color="#7193b7", width=2, dash="dot"),
    ))
    figure.add_trace(go.Scatter(
        x=x, y=bound, name="Analytic convex-cap upper bound",
        mode="lines+markers",
        fill="tonexty", fillcolor="rgba(255,199,111,0.09)",
        line=dict(color="#ffc76f", width=4),
        marker=dict(size=8),
    ))
    figure.add_trace(go.Scatter(
        x=x, y=observed, name="Finite-sample maximum",
        mode="lines+markers",
        line=dict(color="#73d9d0", width=3),
        marker=dict(size=9),
    ))
    figure.update_layout(
        title=dict(
            text="CURVATURE HAS A PRICE<br><sup>Sphere cap → hyperbolic disk · "
                 "chosen radial correspondence · no inferred transport</sup>",
            x=0.04, xanchor="left", font=dict(size=23),
        ),
        template="plotly_dark", paper_bgcolor="#090e18",
        plot_bgcolor="#10192a", font=dict(family="Inter, sans-serif", size=15),
        width=1450, height=850,
        margin=dict(l=90, r=55, t=130, b=100),
        xaxis=dict(title="Spherical cap angular radius (degrees)", rangemode="tozero",
                   showgrid=True, gridcolor="#2b3850", dtick=15),
        yaxis=dict(title="Target distance / scaled source distance", rangemode="tozero",
                   showgrid=True, gridcolor="#2b3850"),
        legend=dict(orientation="h", x=0.03, y=-0.16, font=dict(size=13)),
    )
    html_path = output_dir / "cap-distortion-sweep.html"
    figure.write_html(str(html_path), include_plotlyjs=True)
    png_path = output_dir / "cap-distortion-sweep.png"
    try:
        write_image_with_bootstrap_retry(figure, str(png_path), width=1450, height=850, scale=2)
    except Exception as exc:
        if require_static:
            raise RuntimeError(
                "PNG export failed. Install the static extra and Chrome."
            ) from exc
        print("PNG export skipped:", exc)

    for row in rows:
        print("CAP_SWEEP", ",".join(
            str(row[key]) for key in
            ("cap_degrees", "analytic_upper_stretch", "measured_max_stretch")
        ))
    print("Wrote", csv_path, html_path)
    if png_path.exists():
        print("Wrote", png_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("build/showcase"))
    parser.add_argument("--require-static", action="store_true")
    args = parser.parse_args()
    save(sweep_rows(), args.output_dir, args.require_static)


if __name__ == "__main__":
    main()

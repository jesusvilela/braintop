"""Reproduce the lower bound and near-south-pole upper obstruction.

Usage:
    python examples/nonconvex_cap_lower_sweep.py --output-dir build/showcase
    python examples/nonconvex_cap_lower_sweep.py --require-static

This uses Braintop's actual declared radial map and geodesic-distance engine.
The paired targets are deterministic and chosen by construction; the picture
does not independently validate a theorem or infer a new manifold transport.
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from braintop.core.correspondence import (
    RadialCapMap, compare_radial_cap, radial_cap_global_lower_bound,
)
from braintop.utils.builders import TopologyBuilder
from braintop.utils.static_export import write_image_with_bootstrap_retry


EPSILONS = (1.3, 0.8, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.001)
LAMBDA = 0.7
RADIUS = 1.3
CURVATURE = -1.1


def collect():
    records = []
    for epsilon in EPSILONS:
        theta = math.pi - epsilon
        # All selected caps are larger than hemispheres, so the source
        # minimizing geodesic for opposite boundary points exits the cap.
        source = np.array([[0., 0.], [theta, 0.], [theta, math.pi]])
        target = np.array([
            [0., 0.2],
            [LAMBDA * RADIUS * theta, 0.2],
            [LAMBDA * RADIUS * theta, math.pi + 0.2],
        ])
        builder = TopologyBuilder("south", "Nonconvex cap lower bound")
        builder.add_riemannian_layer(
            "S", 3, "sphere", radius=RADIUS,
            length_unit="sphere-unit", parameters=source,
        )
        builder.add_riemannian_layer(
            "H", 3, "hyperbolic", curvature=CURVATURE,
            length_unit="hyperbolic-unit", parameters=target,
        )
        spec = RadialCapMap(
            "S", "H", LAMBDA, theta,
            "sphere-unit", "hyperbolic-unit", "south-pole-sweep",
            angle_offset=0.2,
        )
        sl, hl = builder.topology.get_layer("S"), builder.topology.get_layer("H")
        lower = radial_cap_global_lower_bound(sl, hl, spec)
        report = compare_radial_cap(
            sl, hl, spec, [(f"S_node_{i}", f"H_node_{i}") for i in range(3)]
        )
        radial = next(p for p in report.pair_remainders
                      if {p.source_a, p.source_b} == {"S_node_0", "S_node_1"})
        opposite = next(p for p in report.pair_remainders
                        if {p.source_a, p.source_b} == {"S_node_1", "S_node_2"})
        observed_ratio = (
            opposite.mapped_target_distance / (LAMBDA * opposite.source_distance)
        )
        expected = theta / epsilon
        if not math.isclose(observed_ratio, expected, rel_tol=1e-9, abs_tol=1e-8):
            raise ArithmeticError("Opposite-ray analytic control failed")
        radial_ratio = radial.mapped_target_distance / (LAMBDA * radial.source_distance)
        if not math.isclose(radial_ratio, 1.0, rel_tol=1e-11):
            raise ArithmeticError("Sharp radial lower control failed")
        if observed_ratio < lower.lower_stretch:
            raise ArithmeticError("Global lower bound violated")
        records.append({
            "cap_degrees": theta * 180 / math.pi,
            "epsilon_radians": epsilon,
            "observed_ratio": observed_ratio,
            "analytical_ratio": expected,
            "lower_factor": lower.lower_stretch,
            "radial_ratio": radial_ratio,
            "source_opposite_distance": opposite.source_distance,
            "target_opposite_distance": opposite.mapped_target_distance,
        })
    return records


def render(rows, directory, static):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "nonconvex-cap-south-pole.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = [r["cap_degrees"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=[r["observed_ratio"] for r in rows],
        mode="lines+markers", name="Braintop geodesic ratio",
        line=dict(color="#7bd5ce", width=4), marker=dict(size=9),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=[r["analytical_ratio"] for r in rows],
        mode="lines", name="Analytic opposite-ray control",
        line=dict(color="#f1bd67", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=[r["lower_factor"] for r in rows],
        mode="lines", name="Sharp global lower factor = 1",
        line=dict(color="#9eb9f0", width=2),
    ))
    fig.update_layout(
        title=dict(
            text="THE SOUTH POLE IS A BOUNDARY<br>"
                 "<sup>A global lower bound survives. No cap-independent upper factor does.</sup>",
            x=0.04, font=dict(size=23),
        ),
        template="plotly_dark",
        paper_bgcolor="#090d18",
        plot_bgcolor="#101928",
        width=1500, height=880,
        font=dict(family="Inter, sans-serif", size=16),
        margin=dict(l=90, r=70, t=125, b=105),
        xaxis=dict(title="Spherical cap radius (degrees)", range=[90, 182],
                   gridcolor="#304054"),
        yaxis=dict(title="Mapped distance / scaled source distance",
                   type="log", gridcolor="#304054"),
        legend=dict(orientation="h", y=-0.17, x=0.04),
    )
    html = directory / "nonconvex-cap-south-pole.html"
    fig.write_html(str(html), include_plotlyjs=True)
    png = directory / "nonconvex-cap-south-pole.png"
    try:
        write_image_with_bootstrap_retry(fig, png, width=1500, height=880, scale=2)
    except Exception as error:
        if static:
            raise RuntimeError("Nonconvex cap PNG export failed") from error
        print(f"PNG skipped: {error}")
    for r in rows:
        print("SOUTH_CONTROL", r["epsilon_radians"],
              r["observed_ratio"], r["lower_factor"])
    print("Wrote", path, html)
    if png.exists():
        print("Wrote", png)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("build/showcase"))
    parser.add_argument("--require-static", action="store_true")
    args = parser.parse_args()
    render(collect(), args.output_dir, args.require_static)


if __name__ == "__main__":
    main()

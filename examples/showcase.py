"""Generate deterministic, publication-friendly Braintop showcase views.

Run with:
    pip install -e ".[static]"
    python examples/showcase.py --output-dir build/showcase --require-static
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from braintop.core.visualizer import TopologyVisualizer
from braintop.integrations.llm_connector import LLMConnector
from braintop.utils.builders import TopologyBuilder


def _style_figure(figure: Any, title: str, subtitle: str) -> Any:
    """Apply a restrained dark observatory theme to a Braintop figure."""
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#090d18",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#e7edf8"),
        margin=dict(l=18, r=18, t=92, b=34),
        title=dict(
            text=f"{title}<br><sup>{subtitle}</sup>",
            x=0.035,
            y=0.97,
            xanchor="left",
            font=dict(size=22, color="#f4f6fb"),
        ),
        legend=dict(
            orientation="h",
            x=0.035,
            y=0.015,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12, color="#cbd5e1"),
        ),
        scene=dict(
            bgcolor="#090d18",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.55, y=1.65, z=1.25)),
        ),
    )
    figure.update_traces(
        marker=dict(
            opacity=0.92,
            line=dict(width=0.35, color="#dbe7ff"),
        ),
        selector=dict(type="scatter3d"),
    )
    return figure


def geometry_atlas(output_dir: Path, require_static: bool) -> None:
    """Render sphere, torus and hyperbolic point clouds side by side."""
    np.random.seed(17)

    builder = TopologyBuilder("geometry_atlas", "Geometry atlas")
    builder.add_riemannian_layer(
        "sphere", num_nodes=96, manifold_type="sphere", radius=1.25
    )
    builder.add_riemannian_layer(
        "torus",
        num_nodes=128,
        manifold_type="torus",
        radius=1.05,
        minor_radius=0.38,
    )
    builder.add_riemannian_layer(
        "hyperbolic",
        num_nodes=128,
        manifold_type="hyperbolic",
        curvature=-1.0,
        max_radius=2.2,
    )
    topology = builder.build()

    offsets = {
        "sphere": (-4.0, 0.0, 0.0),
        "torus": (0.0, 0.0, 0.0),
        "hyperbolic": (4.0, 0.0, 0.0),
    }
    names = {
        "sphere": "Sphere · K > 0",
        "torus": "Torus · periodic",
        "hyperbolic": "Hyperbolic · K < 0",
    }

    for layer in topology.layers:
        layer.name = names[layer.id]
        dx, dy, dz = offsets[layer.id]
        for node in layer.nodes:
            x, y, z = node.position
            node.position = (x + dx, y + dy, z + dz)
            node.label = ""

    visualizer = TopologyVisualizer(topology)
    figure = visualizer.create_figure(color_by="layer", size_by="normalized_radius")
    figure = _style_figure(
        figure,
        "GEOMETRY ATLAS",
        "Braintop manifold samples · translated for comparison, not a common embedding",
    )
    figure.update_layout(
        scene=dict(
            annotations=[
                dict(
                    x=-4.0, y=0.0, z=1.75, text="SPHERE",
                    showarrow=False, font=dict(size=13, color="#f2f5fb"),
                ),
                dict(
                    x=0.0, y=0.0, z=0.75, text="TORUS",
                    showarrow=False, font=dict(size=13, color="#f2f5fb"),
                ),
                dict(
                    x=4.0, y=0.0, z=2.55, text="HYPERBOLIC DISK",
                    showarrow=False, font=dict(size=13, color="#f2f5fb"),
                ),
            ]
        )
    )
    _save(figure, output_dir, "geometry-atlas", require_static)


def semantic_topology(output_dir: Path, require_static: bool) -> None:
    """Render a deterministic concept network from synthetic embedding vectors."""
    rng = np.random.default_rng(42)
    np.random.seed(42)

    concepts = [
        "attention", "context", "token", "sequence", "decoder",
        "embedding", "vector", "latent", "representation", "similarity",
        "training", "gradient", "optimizer", "loss", "inference",
    ]
    cluster_centres = rng.normal(size=(3, 24))
    embeddings = np.vstack(
        [
            cluster_centres[group] + rng.normal(scale=0.12, size=24)
            for group in range(3)
            for _ in range(5)
        ]
    )

    connector = LLMConnector(
        provider="custom",
        model_name="synthetic-demo",
        config={
            "embedding_fn": lambda texts: embeddings,
            "similarity_threshold": 0.72,
            "top_k": 2,
        },
    )
    connector.connect()

    builder = TopologyBuilder("semantic_topology", "Semantic topology")
    builder.add_layer_from_llm(
        "concepts",
        connector,
        concepts,
        projection_method="pca",
        spacing_factor=1.7,
    )
    topology = builder.build()
    connector.disconnect()

    layer = topology.get_layer("concepts")
    layer.name = "Concept network · PCA view"
    group_names = [
        "ATTENTION / SEQUENCE",
        "REPRESENTATION",
        "OPTIMIZATION / INFERENCE",
    ]
    positions = np.asarray([node.position for node in layer.nodes], dtype=float)
    annotations = []
    for group_index, group_name in enumerate(group_names):
        centre = positions[group_index * 5 : (group_index + 1) * 5].mean(axis=0)
        annotations.append(
            dict(
                x=float(centre[0]),
                y=float(centre[1]),
                z=float(centre[2] + 0.55),
                text=group_name,
                showarrow=False,
                font=dict(size=12, color="#f4f6fb"),
            )
        )

    # Keep the static composition uncluttered; full concept names remain in hover metadata.
    for node in layer.nodes:
        node.label = ""

    visualizer = TopologyVisualizer(topology)
    figure = visualizer.create_figure(
        color_by="degree",
        size_by="degree",
    )
    figure = _style_figure(
        figure,
        "SEMANTIC TOPOLOGY",
        "Synthetic vectors · cosine-similarity edges · PCA display coordinates",
    )
    figure.update_layout(
        width=1800,
        height=1200,
        scene=dict(
            aspectmode="cube",
            camera=dict(eye=dict(x=0.85, y=0.95, z=0.75)),
            annotations=annotations,
        ),
    )
    _save(figure, output_dir, "semantic-topology", require_static)

def _save(figure: Any, output_dir: Path, stem: str, require_static: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{stem}.html"
    png_path = output_dir / f"{stem}.png"

    figure.write_html(str(html_path), include_plotlyjs=True, full_html=True)
    try:
        figure.write_image(str(png_path), width=1800, height=1200, scale=2)
    except Exception as exc:
        if require_static:
            raise RuntimeError(
                "Static export failed. Install the 'static' extra and ensure "
                "Chrome is available: pip install -e \".[static]\""
            ) from exc
        print(f"PNG skipped for {stem}: {exc}")

    print(f"wrote {html_path}")
    if png_path.exists():
        print(f"wrote {png_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/showcase"),
        help="Directory for HTML and PNG outputs",
    )
    parser.add_argument(
        "--require-static",
        action="store_true",
        help="Fail if Plotly cannot export PNG files",
    )
    args = parser.parse_args()

    geometry_atlas(args.output_dir, args.require_static)
    semantic_topology(args.output_dir, args.require_static)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

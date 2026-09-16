from __future__ import annotations

from pathlib import Path
from typing import Any


VISUAL_POLICY = {
    "component_inventory": False,
    "value_context": False,
    "component_geometry": False,
    "local_topology": True,
    "path_topology": True,
}


def build_head(embedding_dim: int, hidden_dim: int, dropout: float, output_dim: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(embedding_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


def load_router(model_dir: str | Path, device: str = "cpu") -> dict[str, Any]:
    import torch

    model_dir = Path(model_dir)
    checkpoint = torch.load(model_dir / "best.pt", map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    labels = checkpoint.get("label_names", checkpoint.get("subtype_names"))
    if not labels:
        raise ValueError("Router checkpoint has no label names")
    model = build_head(
        int(config["embedding_dim"]),
        int(config["hidden_dim"]),
        float(config["dropout"]),
        len(labels),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return {"model": model, **checkpoint}


def infer_route(question: str, model_dir: str | Path, device: str = "cpu") -> dict[str, Any]:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    bundle = load_router(model_dir, device)
    encoder = SentenceTransformer(bundle["encoder_path"], local_files_only=True)
    embedding = encoder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=bool(bundle["model_config"].get("normalize_embeddings", True)),
        show_progress_bar=False,
    ).astype(np.float32)
    with torch.no_grad():
        logits = bundle["model"](torch.tensor(embedding, dtype=torch.float32, device=device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    labels = bundle.get("label_names", bundle.get("subtype_names"))
    index = int(np.argmax(probabilities))
    label = labels[index]
    target_type = bundle.get("target_type", "subtype")
    if target_type == "route_bundle":
        route_bundle = label
        route = bundle["bundle_mapping"][label]
    else:
        route = bundle["route_mapping"][label]
        route_bundle = route["route_bundle"]
    return {
        "label": label,
        "confidence": float(probabilities[index]),
        "route_bundle": route_bundle,
        "qtype": route["qtype"],
        "use_raw_image": True,
        "use_marked_image": VISUAL_POLICY[route_bundle],
    }

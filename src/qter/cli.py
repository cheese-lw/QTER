from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evidence import assemble_query_conditioned_evidence


ROUTES = (
    "component_inventory",
    "value_context",
    "component_geometry",
    "local_topology",
    "path_topology",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build question-specific QTER evidence.")
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--route", choices=ROUTES)
    parser.add_argument("--router-model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if bool(args.route) == bool(args.router_model):
        parser.error("Provide exactly one of --route or --router-model")

    payload = json.loads(args.structure.read_text(encoding="utf-8-sig"))
    context = payload.get("structure", payload)
    texts = payload.get("texts", context.get("texts", []))
    route = args.route
    route_prediction = None
    if args.router_model:
        from .router import infer_route

        route_prediction = infer_route(args.question, args.router_model)
        route = route_prediction["route_bundle"]

    result = assemble_query_conditioned_evidence(
        question=args.question,
        route_bundle=route,
        compact_context=context,
        texts=texts,
    )
    if route_prediction:
        result["route_prediction"] = route_prediction

    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 0

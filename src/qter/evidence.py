from __future__ import annotations

import re
from typing import Any

from .structure import (
    StructureIndex,
    component_center,
    component_id,
    component_type,
    edge_component_id,
    edge_node_id,
    edge_terminal_side,
)


TYPE_FORMS = {
    "voltage source": "voltage_source",
    "voltage sources": "voltage_source",
    "current source": "current_source",
    "current sources": "current_source",
    "resistor": "resistor",
    "resistors": "resistor",
    "capacitor": "capacitor",
    "capacitors": "capacitor",
    "inductor": "inductor",
    "inductors": "inductor",
    "ground": "ground",
    "grounds": "ground",
    "port": "port",
    "ports": "port",
}
DISPLAY_TYPE = {
    "voltage_source": "voltage source",
    "current_source": "current source",
    "resistor": "resistor",
    "capacitor": "capacitor",
    "inductor": "inductor",
    "ground": "ground",
    "port": "port",
}
TYPE_PATTERN = "|".join(sorted((re.escape(key) for key in TYPE_FORMS), key=len, reverse=True))
RANKED_PATTERN = re.compile(
    rf"the (?P<rank>\d+)(?:st|nd|rd|th) (?P<type>{TYPE_PATTERN}) from the (?P<origin>left|top)",
    flags=re.IGNORECASE,
)
POSITIONED_PATTERN = re.compile(
    rf"the (?P<position>top|bottom|left-side|right-side|central|upper-branch|lower-branch) "
    rf"(?P<ac>AC )?(?P<type>{TYPE_PATTERN})",
    flags=re.IGNORECASE,
)
GENERIC_PATTERN = re.compile(
    rf"the (?P<ac>AC )?(?P<type>{TYPE_PATTERN}) in the diagram",
    flags=re.IGNORECASE,
)
ALL_PATTERN = re.compile(rf"all (?P<type>{TYPE_PATTERN})", flags=re.IGNORECASE)
VALUE_NAMED_PATTERN = re.compile(
    rf"the (?P<value>[-+]?\d+(?:\.\d+)?)\s*[- ]?(?P<unit>ohm|f|h|v|a) "
    rf"(?P<type>{TYPE_PATTERN})",
    flags=re.IGNORECASE,
)
TERMINAL_PREFIX_PATTERN = re.compile(
    r"(?:the )?(?P<side>left|right|upper|lower|top|bottom) terminal of\s*$",
    flags=re.IGNORECASE,
)


def _normalize_type(value: str) -> str:
    return TYPE_FORMS.get(value.lower(), "")


def _question_component_type(question: str) -> str:
    lower = question.lower()
    for visible in sorted(TYPE_FORMS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(visible)}\b", lower):
            return TYPE_FORMS[visible]
    return ""


def _terminal_side_before(question: str, start: int) -> str:
    prefix = question[:start]
    match = TERMINAL_PREFIX_PATTERN.search(prefix)
    if not match:
        return ""
    side = match.group("side").lower()
    return {"upper": "top", "lower": "bottom"}.get(side, side)


def _candidate_ids(candidates: list[dict[str, Any]]) -> list[str]:
    return [component_id(component) for component in candidates]


def _resolve_position(
    candidates: list[dict[str, Any]], position: str
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    if position in {"left-side"}:
        return [min(candidates, key=lambda item: (*component_center(item), component_id(item)))]
    if position in {"right-side"}:
        return [max(candidates, key=lambda item: (*component_center(item), component_id(item)))]
    if position in {"top", "upper-branch"}:
        return [min(candidates, key=lambda item: (component_center(item)[1], component_center(item)[0], component_id(item)))]
    if position in {"bottom", "lower-branch"}:
        return [max(candidates, key=lambda item: (component_center(item)[1], component_center(item)[0], component_id(item)))]
    center_x = sum(component_center(item)[0] for item in candidates) / len(candidates)
    center_y = sum(component_center(item)[1] for item in candidates) / len(candidates)
    return [
        min(
            candidates,
            key=lambda item: (
                (component_center(item)[0] - center_x) ** 2
                + (component_center(item)[1] - center_y) ** 2,
                component_id(item),
            ),
        )
    ]


def _normalize_value(value: str, unit: str) -> str:
    unit_alias = {"ohm": "ohm", "f": "f", "h": "h", "v": "v", "a": "a"}
    return f"{value.lower().replace(' ', '')}{unit_alias.get(unit.lower(), unit.lower())}"


def _text_value(text: dict[str, Any]) -> str:
    raw = str(
        text.get("value_text")
        or text.get("normalized_text")
        or text.get("text")
        or text.get("raw_text")
        or ""
    ).lower()
    raw = raw.replace("Ω", "ohm").replace("ω", "ohm")
    return re.sub(r"[^a-z0-9.+-]", "", raw)


def _ground_match(
    question: str,
    match: re.Match[str],
    index: StructureIndex,
    resolution: str,
) -> dict[str, Any]:
    comp_type = _normalize_type(match.group("type"))
    candidates = list(index.components_by_type.get(comp_type, []))
    resolved: list[dict[str, Any]] = []
    selector: dict[str, Any] = {"component_type": comp_type, "resolution": resolution}
    if resolution == "spatial_rank":
        axis = "left_to_right" if match.group("origin").lower() == "left" else "top_to_bottom"
        rank = int(match.group("rank"))
        ordered = index.ordered_components(comp_type, axis)
        selector.update({"axis": axis, "rank": rank})
        if 0 < rank <= len(ordered):
            resolved = [ordered[rank - 1]]
    elif resolution == "spatial_position":
        position = match.group("position").lower()
        selector["position"] = position
        resolved = _resolve_position(candidates, position)
    elif resolution == "all_matching_type":
        resolved = candidates
    elif resolution == "value_identity":
        expected = _normalize_value(match.group("value"), match.group("unit"))
        selector.update({"value": match.group("value"), "unit": match.group("unit")})
        linked_ids = {
            str(text.get("linked_component_id") or "")
            for text in index.texts
            if expected in _text_value(text)
        }
        resolved = [item for item in candidates if component_id(item) in linked_ids]
    elif len(candidates) == 1:
        resolved = candidates

    if resolved:
        status = "resolved"
    elif candidates and resolution == "unique_if_possible":
        status = "ambiguous"
    else:
        status = "unresolved"
    return {
        "mention": match.group(0),
        "entity_type": "component",
        "terminal_side": _terminal_side_before(question, match.start()),
        "selector": selector,
        "candidate_component_ids": _candidate_ids(candidates),
        "resolved_component_ids": _candidate_ids(resolved),
        "component_grounding_status": status,
        "terminal_grounding_status": (
            "pending" if _terminal_side_before(question, match.start()) else "not_requested"
        ),
    }


def ground_visible_entities(question: str, index: StructureIndex) -> list[dict[str, Any]]:
    matches: list[tuple[int, int, re.Match[str], str]] = []
    patterns = (
        (RANKED_PATTERN, "spatial_rank"),
        (POSITIONED_PATTERN, "spatial_position"),
        (GENERIC_PATTERN, "unique_if_possible"),
        (ALL_PATTERN, "all_matching_type"),
        (VALUE_NAMED_PATTERN, "value_identity"),
    )
    for pattern, resolution in patterns:
        for match in pattern.finditer(question):
            matches.append((match.start(), match.end(), match, resolution))
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    accepted: list[tuple[int, int, re.Match[str], str]] = []
    for item in matches:
        if any(item[0] < old_end and item[1] > old_start for old_start, old_end, _, _ in accepted):
            continue
        accepted.append(item)
    accepted.sort(key=lambda item: item[0])
    return [_ground_match(question, match, index, resolution) for _, _, match, resolution in accepted]


def _deduplicate_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in edges:
        key = (
            edge_component_id(edge),
            str(edge.get("terminal_id") or edge.get("terminal_candidate_id") or ""),
            edge_node_id(edge),
        )
        result[key] = edge
    return list(result.values())


def _component_value_text(text: dict[str, Any]) -> bool:
    return (
        str(text.get("text_type") or "") == "component_value"
        and bool(text.get("linked_component_id"))
    )


def _filter_component_values(
    question: str,
    anchors: list[dict[str, Any]],
    index: StructureIndex,
    component_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_values = {
        _normalize_value(str(anchor["selector"]["value"]), str(anchor["selector"]["unit"]))
        for anchor in anchors
        if anchor.get("selector", {}).get("resolution") == "value_identity"
    }
    selected: list[dict[str, Any]] = []
    filtered: list[dict[str, str]] = []
    for text in index.texts:
        text_id = str(text.get("text_id") or "")
        linked = str(text.get("linked_component_id") or "")
        if not _component_value_text(text):
            filtered.append({"text_id": text_id, "reason": "not_component_value"})
            continue
        if linked not in component_ids:
            filtered.append({"text_id": text_id, "reason": "not_linked_to_target"})
            continue
        if expected_values and not any(value in _text_value(text) for value in expected_values):
            filtered.append({"text_id": text_id, "reason": "value_does_not_match_question"})
            continue
        selected.append(text)
    return selected, {
        "question_uses_numeric_grounding": bool(expected_values),
        "selected_text_ids": [str(text.get("text_id") or "") for text in selected],
        "filtered": filtered,
        "filter_policy": "linked_component_value_only",
    }


def _role_add(
    roles: dict[str, dict[str, list[str]]], category: str, record_id: str, role: str
) -> None:
    values = roles[category].setdefault(record_id, [])
    if role not in values:
        values.append(role)


def _structured_projection(selected: dict[str, Any]) -> dict[str, Any]:
    return {
        "components": [
            {
                "component_id": component_id(component),
                "type": component_type(component),
                "bbox": component.get("bbox"),
                "center": component.get("center"),
            }
            for component in selected["components"]
        ],
        "terminals": [
            {
                "component_id": terminal.get("component_id"),
                "terminal_side": terminal.get("terminal_side"),
                "node_id": terminal.get("node_id"),
            }
            for terminal in selected["terminals"]
        ],
        "nodes": [
            {
                "node_id": node.get("node_id"),
                "terminal_members": node.get("terminal_members", []),
                "component_ids": node.get("component_ids", []),
            }
            for node in selected["nodes"]
        ],
        "edges": [
            {
                "component_id": edge_component_id(edge),
                "terminal_side": edge_terminal_side(edge),
                "node_id": edge_node_id(edge),
            }
            for edge in selected["edges"]
        ],
        "texts": [
            {
                "linked_component_id": text.get("linked_component_id"),
                "value_text": text.get("value_text") or text.get("normalized_text"),
                "unit": text.get("unit"),
                "text_type": "component_value",
            }
            for text in selected["texts"]
        ],
        "node_stars": selected["node_stars"],
        "evidence_roles": selected["evidence_roles"],
    }


def _facts_for_selection(
    route_bundle: str,
    anchors: list[dict[str, Any]],
    selected: dict[str, list[dict[str, Any]]],
    graph_search: dict[str, Any] | None,
) -> list[str]:
    facts: list[str] = []
    for anchor in anchors:
        resolved = anchor.get("resolved_component_ids", [])
        if resolved:
            facts.append(f'Visible reference "{anchor["mention"]}" resolves to {", ".join(resolved)}.')
    if route_bundle == "component_inventory":
        ids = [component_id(item) for item in selected["components"]]
        comp_type = component_type(selected["components"][0]) if selected["components"] else "requested type"
        facts.append(f"Detected {DISPLAY_TYPE.get(comp_type, comp_type)} records: {', '.join(ids) if ids else 'none'}.")
    elif route_bundle == "component_geometry":
        for component in selected["components"]:
            x, y = component_center(component)
            visible_type = DISPLAY_TYPE.get(component_type(component), component_type(component))
            article = "an" if visible_type[:1].lower() in "aeiou" else "a"
            facts.append(
                f"{component_id(component)} is {article} {visible_type} with center ({x:g}, {y:g})."
            )
    elif route_bundle == "value_context":
        for text in selected["texts"]:
            linked = str(text.get("linked_component_id") or "unlinked")
            value = str(
                text.get("value_text")
                or text.get("normalized_text")
                or text.get("text")
                or text.get("raw_text")
                or ""
            )
            facts.append(f'OCR text linked to {linked} reads "{value}".')
    elif route_bundle == "local_topology":
        target_terminal_ids = {
            record_id
            for record_id, roles in selected["evidence_roles"]["terminals"].items()
            if "target_terminal" in roles
        }
        for terminal in selected["terminals"]:
            side = terminal.get("terminal_side") or terminal.get("terminal_id") or "terminal"
            terminal_ref = f'{terminal["component_id"]}.{side}'
            if terminal_ref in target_terminal_ids:
                facts.append(
                    f'{terminal_ref} is assigned to electrical node {terminal["node_id"]}.'
                )
    elif route_bundle == "path_topology" and graph_search:
        components = graph_search.get("dag_component_vertices", [])
        if components:
            facts.append(
                "Predicted shortest-path DAG component set: " + ", ".join(components) + "."
            )
        for link in graph_search.get("dag_transitions", []):
            facts.append(
                f'{link["from_component"]} and {link["to_component"]} meet through electrical node '
                f'{link["via_node"]}.'
            )
    if route_bundle in {"local_topology", "path_topology"}:
        for star in selected["node_stars"]:
            members = ", ".join(star.get("terminal_members", [])) or "no represented terminals"
            facts.append(
                f'Complete one-hop neighborhood of {star["node_id"]}: {members}.'
            )
    return facts


def assemble_query_conditioned_evidence(
    *,
    question: str,
    route_bundle: str,
    compact_context: dict[str, Any],
    texts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build V3.1 node-centered evidence from predicted structure only."""
    index = StructureIndex.from_context(compact_context, texts)
    anchors = ground_visible_entities(question, index)
    warnings: list[str] = []
    graph_search: dict[str, Any] | None = None
    selected_components: set[str] = set()
    selected_nodes: set[str] = set()
    selected_edges: list[dict[str, Any]] = []
    selected_texts: list[dict[str, Any]] = []
    node_stars: list[dict[str, Any]] = []
    target_components: set[str] = set()
    target_terminal_edge_keys: set[str] = set()
    path_core_components: set[str] = set()
    path_core_nodes: set[str] = set()
    path_core_edge_keys: set[str] = set()
    prediction_issue = False
    target_component_missing = False
    ocr_filter_trace: dict[str, Any] = {
        "question_uses_numeric_grounding": False,
        "selected_text_ids": [],
        "filtered": [],
        "filter_policy": "linked_component_value_only",
    }

    anchor_component_states = {
        str(anchor.get("component_grounding_status") or "") for anchor in anchors
    }
    if "unresolved" in anchor_component_states:
        target_component_missing = True
    if "ambiguous" in anchor_component_states:
        prediction_issue = True

    if route_bundle == "component_inventory":
        comp_type = _question_component_type(question)
        selected_components = {
            component_id(component)
            for component in index.components_by_type.get(comp_type, [])
        }
        target_components.update(selected_components)
    elif route_bundle == "component_geometry":
        selected_components = set(index.component_by_id)
        target_components.update(selected_components)
    elif route_bundle == "value_context":
        selected_components = {
            comp_id
            for anchor in anchors
            for comp_id in anchor.get("resolved_component_ids", [])
        }
        target_components.update(selected_components)
        selected_texts, ocr_filter_trace = _filter_component_values(
            question, anchors, index, selected_components
        )
        if not selected_texts:
            warnings.append("linked_ocr_text_not_found")
            prediction_issue = True
    elif route_bundle == "local_topology":
        for anchor in anchors:
            for comp_id in anchor.get("resolved_component_ids", []):
                selected_components.add(comp_id)
                target_components.add(comp_id)
                side = str(anchor.get("terminal_side") or "")
                terminal_edges = index.terminal_edges(comp_id, side)
                if not terminal_edges:
                    anchor["terminal_grounding_status"] = "missing"
                    prediction_issue = True
                    warnings.append(
                        f"requested_terminal_not_available:{comp_id}:{side or 'unspecified'}"
                    )
                    terminal_edges = index.terminal_edges(comp_id)
                else:
                    anchor["terminal_grounding_status"] = "resolved"
                    if side:
                        target_terminal_edge_keys.update(
                            index.edge_key(edge) for edge in terminal_edges
                        )
                    else:
                        target_terminal_edge_keys.update(
                            index.edge_key(edge) for edge in terminal_edges
                        )
                selected_nodes.update(
                    edge_node_id(edge) for edge in terminal_edges if edge_node_id(edge)
                )
        node_stars, star_components, star_edges = index.expand_node_stars(selected_nodes)
        node_stars = [{**star, "trigger": "local_topology"} for star in node_stars]
        selected_components.update(star_components)
        selected_edges = _deduplicate_edges(star_edges)
    elif route_bundle == "path_topology":
        endpoint_groups = [
            list(anchor.get("resolved_component_ids", [])) for anchor in anchors[:2]
        ]
        target_components.update(
            comp_id for group in endpoint_groups for comp_id in group
        )
        if len(endpoint_groups) == 2:
            path_result = index.shortest_component_path_dag(
                endpoint_groups[0], endpoint_groups[1]
            )
        else:
            path_result = index.shortest_component_path_dag([], [])
        path_core_components = set(path_result.component_ids)
        path_core_nodes = set(path_result.node_ids)
        path_core_edge_keys = {
            index.edge_key(edge) for edge in path_result.path_edges
        }
        selected_nodes = set(path_result.node_ids)
        if path_result.status != "found":
            prediction_issue = True
            warnings.append("predicted_graph_path_not_found")
            for comp_id in sorted(target_components):
                selected_nodes.update(
                    edge_node_id(edge)
                    for edge in index.terminal_edges(comp_id)
                    if edge_node_id(edge)
                )
        node_stars, star_components, star_edges = index.expand_node_stars(selected_nodes)
        node_stars = [{**star, "trigger": "path_core"} for star in node_stars]
        selected_components = set(star_components).union(
            path_core_components, target_components
        )
        selected_edges = _deduplicate_edges(star_edges)
        graph_search = {
            "algorithm": "bidirectional_bfs_shortest_path_dag_on_component_adjacency_graph",
            "endpoint_component_candidates": endpoint_groups,
            "path_status": path_result.status,
            "shortest_distance": path_result.shortest_distance,
            "shortest_path_count": path_result.shortest_path_count,
            "dag_component_vertices": list(path_result.component_ids),
            "dag_node_vertices": list(path_result.node_ids),
            "dag_transitions": list(path_result.transitions),
            "supporting_edge_ids": sorted(path_core_edge_keys),
            "representative_component_path": list(
                path_result.representative_component_path
            ),
            "representative_node_path": list(path_result.representative_node_path),
            "derived_answer": None,
        }
    else:
        raise KeyError(f"Unsupported route bundle: {route_bundle}")

    numeric_anchor_components = {
        comp_id
        for anchor in anchors
        if anchor.get("selector", {}).get("resolution") == "value_identity"
        for comp_id in anchor.get("resolved_component_ids", [])
    }
    if route_bundle in {"local_topology", "path_topology"} and numeric_anchor_components:
        selected_texts, ocr_filter_trace = _filter_component_values(
            question, anchors, index, numeric_anchor_components
        )

    selected = index.selected_records(
        selected_components,
        selected_nodes,
        selected_edges,
        selected_texts,
    )
    roles: dict[str, dict[str, list[str]]] = {
        "components": {},
        "terminals": {},
        "nodes": {},
        "edges": {},
    }
    for component in selected["components"]:
        comp_id = component_id(component)
        _role_add(
            roles,
            "components",
            comp_id,
            "target_component" if comp_id in target_components else "node_context",
        )
        if comp_id in path_core_components:
            _role_add(roles, "components", comp_id, "path_core")
    for node in selected["nodes"]:
        node_id = str(node.get("node_id") or "")
        _role_add(
            roles,
            "nodes",
            node_id,
            "path_core" if node_id in path_core_nodes else "node_context",
        )
    for edge in selected["edges"]:
        edge_key = index.edge_key(edge)
        terminal_ref = (
            f"{edge_component_id(edge)}."
            f"{edge_terminal_side(edge) or edge_terminal_id(edge) or 'terminal'}"
        )
        edge_role = "path_core" if edge_key in path_core_edge_keys else "node_context"
        terminal_role = (
            "target_terminal"
            if edge_key in target_terminal_edge_keys
            else edge_role
        )
        _role_add(roles, "edges", edge_key, edge_role)
        if edge_key in target_terminal_edge_keys:
            _role_add(roles, "edges", edge_key, "target_terminal")
        _role_add(roles, "terminals", terminal_ref, terminal_role)
    selected["node_stars"] = node_stars
    selected["evidence_roles"] = roles

    if target_component_missing:
        prediction_coverage_status = "missing"
    elif prediction_issue:
        prediction_coverage_status = "partial"
    else:
        prediction_coverage_status = "complete"
    selected_record_count = sum(
        len(selected[key]) for key in ("components", "nodes", "edges", "texts")
    )
    if prediction_coverage_status == "complete":
        assembly_status = "success"
        evidence_status = "complete"
    elif selected_record_count:
        assembly_status = "partial"
        evidence_status = "partial"
    else:
        assembly_status = "failed"
        evidence_status = "empty"

    facts = _facts_for_selection(route_bundle, anchors, selected, graph_search)
    structured_evidence = _structured_projection(selected)
    audit_metadata = {
        "assembly_status": assembly_status,
        "prediction_coverage_status": prediction_coverage_status,
        "evidence_status": evidence_status,
        "route_prediction": {"route_bundle": route_bundle},
        "anchor_trace": anchors,
        "graph_query_trace": graph_search or {},
        "ocr_filter_trace": ocr_filter_trace,
        "warnings": sorted(set(warnings)),
        "record_counts": {
            "full": index.full_record_count(),
            "selected": selected_record_count,
        },
        "offline_gt_comparison": None,
    }
    return {
        "assembly_version": "question_evidence_v3_1",
        "selection_source": "english_question_and_predicted_structure_only",
        "uses_private_answer": False,
        "selected_evidence": selected,
        "vlm_payload": {
            "question": question,
            "text_facts": facts,
            "structured_evidence": structured_evidence,
            "visual_evidence": {
                "raw_image": True,
                "ours_labeled_image": route_bundle
                in {"local_topology", "path_topology"},
            },
        },
        "audit_metadata": audit_metadata,
    }

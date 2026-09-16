from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable


def component_id(component: dict[str, Any]) -> str:
    return str(component.get("component_id") or component.get("comp_id") or "")


def component_type(component: dict[str, Any]) -> str:
    return str(component.get("type") or component.get("comp_type") or "")


def component_center(component: dict[str, Any]) -> tuple[float, float]:
    center = component.get("center") or []
    if len(center) >= 2:
        return float(center[0]), float(center[1])
    bbox = component.get("bbox") or [0, 0, 0, 0]
    return (
        (float(bbox[0]) + float(bbox[2])) / 2.0,
        (float(bbox[1]) + float(bbox[3])) / 2.0,
    )


def edge_component_id(edge: dict[str, Any]) -> str:
    return str(edge.get("component_id") or edge.get("source_id") or "")


def edge_node_id(edge: dict[str, Any]) -> str:
    return str(edge.get("node_id") or edge.get("target_id") or "")


def edge_terminal_id(edge: dict[str, Any]) -> str:
    return str(
        edge.get("terminal_id")
        or edge.get("terminal_candidate_id")
        or edge.get("component_terminal_candidate_id")
        or ""
    )


def edge_terminal_side(edge: dict[str, Any]) -> str:
    value = str(
        edge.get("terminal_side")
        or edge.get("terminal")
        or edge.get("component_terminal_role")
        or ""
    ).lower()
    aliases = {
        "upper": "top",
        "lower": "bottom",
        "terminal_a": "a",
        "terminal_b": "b",
    }
    return aliases.get(value, value)


@dataclass(frozen=True)
class PathResult:
    status: str
    component_path: tuple[str, ...]
    node_path: tuple[str, ...]
    path_edges: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PathDAGResult:
    status: str
    shortest_distance: int | None
    component_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    transitions: tuple[dict[str, Any], ...]
    path_edges: tuple[dict[str, Any], ...]
    representative_component_path: tuple[str, ...]
    representative_node_path: tuple[str, ...]
    shortest_path_count: int


class StructureIndex:
    """Question-time indices over predicted circuit structure only."""

    def __init__(
        self,
        components: Iterable[dict[str, Any]],
        nodes: Iterable[dict[str, Any]],
        edges: Iterable[dict[str, Any]],
        texts: Iterable[dict[str, Any]],
    ) -> None:
        self.components = list(components)
        self.nodes = list(nodes)
        self.edges = list(edges)
        self.texts = list(texts)
        self.component_by_id = {
            component_id(component): component
            for component in self.components
            if component_id(component)
        }
        self.node_by_id = {
            str(node.get("node_id") or ""): node
            for node in self.nodes
            if node.get("node_id")
        }
        self.components_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.edges_by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.edges_by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.edge_by_terminal: dict[str, dict[str, Any]] = {}
        self.texts_by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for component in self.components:
            self.components_by_type[component_type(component)].append(component)
        for edge in self.edges:
            comp_id = edge_component_id(edge)
            node_id = edge_node_id(edge)
            terminal_id = edge_terminal_id(edge)
            if comp_id:
                self.edges_by_component[comp_id].append(edge)
            if node_id:
                self.edges_by_node[node_id].append(edge)
            if terminal_id:
                self.edge_by_terminal[terminal_id] = edge
        for text in self.texts:
            linked = str(text.get("linked_component_id") or "")
            if linked:
                self.texts_by_component[linked].append(text)

    @classmethod
    def from_context(
        cls,
        compact_context: dict[str, Any],
        texts: Iterable[dict[str, Any]] = (),
    ) -> "StructureIndex":
        return cls(
            compact_context.get("components", []),
            compact_context.get("nodes", []),
            compact_context.get("edges", []),
            texts,
        )

    def ordered_components(self, comp_type: str, axis: str) -> list[dict[str, Any]]:
        candidates = list(self.components_by_type.get(comp_type, []))
        if axis == "left_to_right":
            candidates.sort(
                key=lambda item: (
                    component_center(item)[0],
                    component_center(item)[1],
                    component_id(item),
                )
            )
        elif axis == "top_to_bottom":
            candidates.sort(
                key=lambda item: (
                    component_center(item)[1],
                    component_center(item)[0],
                    component_id(item),
                )
            )
        else:
            raise ValueError(f"Unsupported ordering axis: {axis}")
        return candidates

    def terminal_edges(self, comp_id: str, side: str = "") -> list[dict[str, Any]]:
        edges = list(self.edges_by_component.get(comp_id, []))
        if not side:
            return edges
        normalized = {"upper": "top", "lower": "bottom"}.get(side, side)
        return [edge for edge in edges if edge_terminal_side(edge) == normalized]

    @staticmethod
    def edge_key(edge: dict[str, Any]) -> str:
        explicit = str(edge.get("edge_id") or edge.get("id") or "")
        if explicit:
            return explicit
        terminal = edge_terminal_side(edge) or edge_terminal_id(edge) or "terminal"
        return f"{edge_component_id(edge)}.{terminal}->{edge_node_id(edge)}"

    def node_star(self, node_id: str) -> dict[str, Any] | None:
        node = self.node_by_id.get(node_id)
        if node is None:
            return None
        edges = sorted(
            self.edges_by_node.get(node_id, []),
            key=lambda edge: (
                edge_component_id(edge),
                edge_terminal_side(edge),
                edge_terminal_id(edge),
            ),
        )
        component_ids = sorted(
            {edge_component_id(edge) for edge in edges if edge_component_id(edge)}
        )
        terminal_members = [
            f"{edge_component_id(edge)}.{edge_terminal_side(edge) or edge_terminal_id(edge) or 'terminal'}"
            for edge in edges
        ]
        return {
            "node_id": node_id,
            "terminal_members": terminal_members,
            "incident_edge_ids": [self.edge_key(edge) for edge in edges],
            "incident_component_ids": component_ids,
        }

    def expand_node_stars(
        self, node_ids: Iterable[str]
    ) -> tuple[list[dict[str, Any]], set[str], list[dict[str, Any]]]:
        stars: list[dict[str, Any]] = []
        component_ids: set[str] = set()
        edge_by_key: dict[str, dict[str, Any]] = {}
        for node_id in sorted(set(node_ids)):
            star = self.node_star(node_id)
            if star is None:
                continue
            stars.append(star)
            component_ids.update(star["incident_component_ids"])
            for edge in self.edges_by_node.get(node_id, []):
                edge_by_key[self.edge_key(edge)] = edge
        edges = [edge_by_key[key] for key in sorted(edge_by_key)]
        return stars, component_ids, edges

    def selected_records(
        self,
        component_ids: Iterable[str],
        node_ids: Iterable[str],
        selected_edges: Iterable[dict[str, Any]],
        selected_texts: Iterable[dict[str, Any]] = (),
    ) -> dict[str, list[dict[str, Any]]]:
        comp_set = set(component_ids)
        node_set = set(node_ids)
        edges = list(selected_edges)
        node_records = []
        for node in self.nodes:
            node_id = str(node.get("node_id") or "")
            if node_id not in node_set:
                continue
            star = self.node_star(node_id)
            normalized = dict(node)
            if star is not None:
                normalized["terminal_members"] = list(star["terminal_members"])
                normalized["component_ids"] = list(star["incident_component_ids"])
            node_records.append(normalized)
        return {
            "components": [
                component for component in self.components if component_id(component) in comp_set
            ],
            "terminals": [
                {
                    "terminal_id": edge_terminal_id(edge),
                    "component_id": edge_component_id(edge),
                    "terminal_side": edge_terminal_side(edge),
                    "node_id": edge_node_id(edge),
                }
                for edge in edges
            ],
            "nodes": node_records,
            "edges": edges,
            "texts": list(selected_texts),
        }

    def _component_adjacency(
        self,
    ) -> dict[str, list[tuple[str, str, dict[str, Any], dict[str, Any]]]]:
        adjacency: dict[
            str, dict[tuple[str, str, str, str], tuple[str, str, dict[str, Any], dict[str, Any]]]
        ] = defaultdict(dict)
        for node_id, node_edges in self.edges_by_node.items():
            ordered = sorted(
                node_edges,
                key=lambda edge: (
                    edge_component_id(edge),
                    edge_terminal_side(edge),
                    edge_terminal_id(edge),
                ),
            )
            for current_edge in ordered:
                current = edge_component_id(current_edge)
                if not current:
                    continue
                for neighbor_edge in ordered:
                    neighbor = edge_component_id(neighbor_edge)
                    if not neighbor or neighbor == current:
                        continue
                    key = (
                        neighbor,
                        node_id,
                        self.edge_key(current_edge),
                        self.edge_key(neighbor_edge),
                    )
                    adjacency[current][key] = (
                        neighbor,
                        node_id,
                        current_edge,
                        neighbor_edge,
                    )
        return {
            component: [records[key] for key in sorted(records)]
            for component, records in adjacency.items()
        }

    @staticmethod
    def _distances(
        adjacency: dict[str, list[tuple[str, str, dict[str, Any], dict[str, Any]]]],
        sources: Iterable[str],
    ) -> dict[str, int]:
        distances: dict[str, int] = {}
        queue: deque[str] = deque()
        for source in sorted(set(sources)):
            distances[source] = 0
            queue.append(source)
        while queue:
            current = queue.popleft()
            for neighbor, _, _, _ in adjacency.get(current, []):
                if neighbor in distances:
                    continue
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
        return distances

    def shortest_component_path_dag(
        self,
        start_candidates: Iterable[str],
        end_candidates: Iterable[str],
    ) -> PathDAGResult:
        starts = sorted(set(start_candidates).intersection(self.component_by_id))
        ends = sorted(set(end_candidates).intersection(self.component_by_id))
        if not starts or not ends:
            return PathDAGResult("not_found", None, (), (), (), (), (), (), 0)

        adjacency = self._component_adjacency()
        from_start = self._distances(adjacency, starts)
        from_end = self._distances(adjacency, ends)
        reachable_ends = [end for end in ends if end in from_start]
        if not reachable_ends:
            return PathDAGResult("not_found", None, (), (), (), (), (), (), 0)
        shortest_distance = min(from_start[end] for end in reachable_ends)
        dag_components = {
            component
            for component in self.component_by_id
            if component in from_start
            and component in from_end
            and from_start[component] + from_end[component] == shortest_distance
        }

        transition_by_key: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
        path_edge_by_key: dict[str, dict[str, Any]] = {}
        for source in sorted(dag_components):
            for target, node_id, source_edge, target_edge in adjacency.get(source, []):
                if target not in dag_components:
                    continue
                if from_start.get(target) != from_start[source] + 1:
                    continue
                if from_start[source] + 1 + from_end.get(target, shortest_distance + 1) != shortest_distance:
                    continue
                key = (
                    source,
                    target,
                    node_id,
                    self.edge_key(source_edge),
                    self.edge_key(target_edge),
                )
                transition_by_key[key] = {
                    "from_component": source,
                    "to_component": target,
                    "via_node": node_id,
                    "from_terminal": edge_terminal_side(source_edge),
                    "to_terminal": edge_terminal_side(target_edge),
                    "supporting_edge_ids": [
                        self.edge_key(source_edge),
                        self.edge_key(target_edge),
                    ],
                }
                path_edge_by_key[self.edge_key(source_edge)] = source_edge
                path_edge_by_key[self.edge_key(target_edge)] = target_edge

        transitions = [transition_by_key[key] for key in sorted(transition_by_key)]
        node_ids = sorted({str(item["via_node"]) for item in transitions})

        transition_targets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for transition in transitions:
            transition_targets[str(transition["from_component"])].append(transition)
        representative_components = [
            min(starts, key=lambda item: (from_start.get(item, 10**9), item))
        ]
        representative_nodes: list[str] = []
        current = representative_components[0]
        target_ends = {end for end in reachable_ends if from_start[end] == shortest_distance}
        while current not in target_ends:
            choices = sorted(
                transition_targets.get(current, []),
                key=lambda item: (
                    str(item["to_component"]),
                    str(item["via_node"]),
                ),
            )
            if not choices:
                break
            choice = choices[0]
            representative_nodes.append(str(choice["via_node"]))
            current = str(choice["to_component"])
            representative_components.append(current)

        path_counts: dict[str, int] = {component: 0 for component in dag_components}
        for start in starts:
            if start in path_counts:
                path_counts[start] += 1
        for component in sorted(dag_components, key=lambda item: (from_start[item], item)):
            for transition in transition_targets.get(component, []):
                target = str(transition["to_component"])
                path_counts[target] += path_counts[component]
        shortest_path_count = sum(path_counts.get(end, 0) for end in target_ends)

        return PathDAGResult(
            "found",
            shortest_distance,
            tuple(sorted(dag_components, key=lambda item: (from_start[item], item))),
            tuple(node_ids),
            tuple(transitions),
            tuple(path_edge_by_key[key] for key in sorted(path_edge_by_key)),
            tuple(representative_components),
            tuple(representative_nodes),
            shortest_path_count,
        )

    def shortest_component_path(
        self,
        start_candidates: Iterable[str],
        end_candidates: Iterable[str],
    ) -> PathResult:
        starts = sorted(set(start_candidates).intersection(self.component_by_id))
        ends = set(end_candidates).intersection(self.component_by_id)
        if not starts or not ends:
            return PathResult("not_found", (), (), ())

        queue: deque[tuple[str, tuple[str, ...], tuple[str, ...], tuple[dict[str, Any], ...]]] = deque()
        visited_distance: dict[str, int] = {}
        for start in starts:
            queue.append((start, (start,), (), ()))
            visited_distance[start] = 0

        while queue:
            current, comp_path, node_path, path_edges = queue.popleft()
            if current in ends:
                return PathResult("found", comp_path, node_path, path_edges)
            distance = len(comp_path) - 1
            neighbors: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
            for current_edge in self.edges_by_component.get(current, []):
                node_id = edge_node_id(current_edge)
                for neighbor_edge in self.edges_by_node.get(node_id, []):
                    neighbor = edge_component_id(neighbor_edge)
                    if neighbor and neighbor != current:
                        neighbors.append((neighbor, node_id, current_edge, neighbor_edge))
            neighbors.sort(key=lambda item: (item[0], item[1]))
            for neighbor, node_id, current_edge, neighbor_edge in neighbors:
                next_distance = distance + 1
                if neighbor in visited_distance and visited_distance[neighbor] <= next_distance:
                    continue
                visited_distance[neighbor] = next_distance
                queue.append(
                    (
                        neighbor,
                        (*comp_path, neighbor),
                        (*node_path, node_id),
                        (*path_edges, current_edge, neighbor_edge),
                    )
                )
        return PathResult("not_found", (), (), ())

    def full_record_count(self) -> int:
        return len(self.components) + len(self.nodes) + len(self.edges) + len(self.texts)

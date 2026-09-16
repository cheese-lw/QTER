from qter import StructureIndex, assemble_query_conditioned_evidence


def circuit():
    return {
        "components": [
            {"component_id": "L1", "type": "inductor", "bbox": [10, 40, 30, 60]},
            {"component_id": "C1", "type": "capacitor", "bbox": [50, 10, 70, 30]},
            {"component_id": "V1", "type": "voltage_source", "bbox": [50, 70, 70, 90]},
            {"component_id": "R1", "type": "resistor", "bbox": [90, 40, 110, 60]},
        ],
        "nodes": [{"node_id": f"N{i}"} for i in range(1, 5)],
        "edges": [
            {"component_id": "L1", "terminal_side": "left", "node_id": "N1"},
            {"component_id": "C1", "terminal_side": "left", "node_id": "N1"},
            {"component_id": "L1", "terminal_side": "right", "node_id": "N2"},
            {"component_id": "V1", "terminal_side": "left", "node_id": "N2"},
            {"component_id": "C1", "terminal_side": "right", "node_id": "N3"},
            {"component_id": "R1", "terminal_side": "left", "node_id": "N3"},
            {"component_id": "V1", "terminal_side": "right", "node_id": "N4"},
            {"component_id": "R1", "terminal_side": "right", "node_id": "N4"},
        ],
    }


def test_node_star_keeps_complete_neighborhood():
    data = circuit()
    star = StructureIndex.from_context(data).node_star("N1")
    assert star is not None
    assert star["incident_component_ids"] == ["C1", "L1"]


def test_shortest_path_dag_keeps_equal_length_paths():
    data = circuit()
    result = assemble_query_conditioned_evidence(
        question="Along the shortest component path from the inductor in the diagram to the resistor in the diagram, how many intermediate components are traversed?",
        route_bundle="path_topology",
        compact_context=data,
        texts=[],
    )
    trace = result["audit_metadata"]["graph_query_trace"]
    assert trace["path_status"] == "found"
    assert trace["shortest_distance"] == 2
    assert trace["shortest_path_count"] == 2
    assert set(trace["dag_component_vertices"]) == {"L1", "C1", "V1", "R1"}

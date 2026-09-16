# QTER for CircuitVQA

Minimal implementation of question-conditioned topology-preserving evidence routing.

## Included

- circuit-structure indexing;
- entity grounding from question text;
- Node-Star retrieval for connectivity questions;
- shortest-path DAG retrieval for path questions;
- optional learned-router inference;
- one runnable example and two core tests.

Datasets, model weights, paper files, API keys, generated results, and third-party projects are intentionally excluded.

## Install

```bash
python -m pip install -e .
```

For learned-router inference:

```bash
python -m pip install -e ".[router]"
```

## Run

```bash
qter \
  --structure examples/structure.json \
  --question "Along the shortest component path from the inductor in the diagram to the resistor in the diagram, how many intermediate components are traversed?" \
  --route path_topology \
  --output evidence.json
```

Available routes:

- `component_inventory`
- `value_context`
- `component_geometry`
- `local_topology`
- `path_topology`

To use a trained router checkpoint instead of an explicit route:

```bash
qter --structure examples/structure.json --question "..." --router-model /path/to/router
```

The router directory must contain `best.pt` and the local sentence-encoder path recorded in that checkpoint.

## Input

The structure JSON contains four arrays:

```json
{
  "components": [],
  "nodes": [],
  "edges": [],
  "texts": []
}
```

See `examples/structure.json` for the minimal field set.

## Test

```bash
python -m pip install -e ".[test]"
pytest -q
```

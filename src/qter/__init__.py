"""Question-conditioned topology-preserving evidence routing."""

from .evidence import assemble_query_conditioned_evidence, ground_visible_entities
from .structure import PathDAGResult, StructureIndex

__all__ = [
    "PathDAGResult",
    "StructureIndex",
    "assemble_query_conditioned_evidence",
    "ground_visible_entities",
]

"""Associative vector indexing engine with strict cryptographic namespace segregation.
Implements Level 2 (segregated index structures) and Level 4 (partitioned cache keys) isolation.
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class SearchResult:
    """Represents a matched memory item from associative index lookup."""
    record_index: int
    segment_id: int
    similarity: float
    namespace_id: int


class NamespaceIndexPartition:
    """Dedicated vector index partition for a single cryptographic namespace.
    Contains only vectors belonging to this specific namespace.
    """

    def __init__(self, namespace_id: int, dim: int = 1536) -> None:
        self.namespace_id = namespace_id
        self.dim = dim
        self._keys_list: list[np.ndarray] = []
        self._keys_matrix: np.ndarray | None = None
        self._record_map: list[tuple[int, int]] = []  # (segment_id, record_index)
        self._is_normalized: bool = False

    def add(self, segment_id: int, record_index: int, key_vector: np.ndarray) -> None:
        """Adds a normalized key vector mapped to a storage coordinate."""
        vec = key_vector.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        else:
            vec = np.zeros_like(vec)

        self._keys_list.append(vec)
        self._record_map.append((segment_id, record_index))
        self._keys_matrix = None  # Invalidate cached matrix

    def _ensure_matrix(self) -> np.ndarray:
        if self._keys_matrix is None and len(self._keys_list) > 0:
            self._keys_matrix = np.vstack(self._keys_list)
        return self._keys_matrix if self._keys_matrix is not None else np.empty((0, self.dim), dtype=np.float32)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.75,
    ) -> list[SearchResult]:
        """Performs cosine similarity search against this namespace's vectors.
        Enforces confidence threshold: items with similarity < threshold are rejected.
        """
        matrix = self._ensure_matrix()
        if matrix.shape[0] == 0:
            return []

        q = query_vector.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-8:
            q = q / q_norm
        else:
            return []

        # Vectorized cosine similarity dot product: (N, D) @ (D,) -> (N,)
        similarities = matrix @ q

        # Filter by threshold
        valid_indices = np.where(similarities >= threshold)[0]
        if len(valid_indices) == 0:
            return []

        # Top-K sorting among valid indices
        valid_sims = similarities[valid_indices]
        if len(valid_indices) > top_k:
            top_k_sub = np.argpartition(valid_sims, -top_k)[-top_k:]
            sorted_indices = valid_indices[top_k_sub[np.argsort(-valid_sims[top_k_sub])]]
        else:
            sorted_indices = valid_indices[np.argsort(-valid_sims)]

        results: list[SearchResult] = []
        for idx in sorted_indices:
            seg_id, rec_idx = self._record_map[idx]
            sim = float(similarities[idx])
            results.append(
                SearchResult(
                    record_index=rec_idx,
                    segment_id=seg_id,
                    similarity=sim,
                    namespace_id=self.namespace_id,
                )
            )
        return results

    @property
    def size(self) -> int:
        return len(self._keys_list)


class PartitionedVectorIndex:
    """Root associative index coordinator enforcing strict multi-namespace isolation.
    Dispatches queries exclusively to the authorized namespace partition.
    """

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim
        self._partitions: dict[int, NamespaceIndexPartition] = {}
        # Level 4 Isolation: Cache key includes (namespace_id, query_hash)
        self._query_cache: dict[tuple[int, str], list[SearchResult]] = {}
        self._cache_capacity: int = 1000

    def get_or_create_partition(self, namespace_id: int) -> NamespaceIndexPartition:
        if namespace_id not in self._partitions:
            self._partitions[namespace_id] = NamespaceIndexPartition(namespace_id, dim=self.dim)
        return self._partitions[namespace_id]

    def add_record(self, namespace_id: int, segment_id: int, record_index: int, key_vector: np.ndarray) -> None:
        """Indexes a record strictly within its designated namespace partition."""
        partition = self.get_or_create_partition(namespace_id)
        partition.add(segment_id, record_index, key_vector)
        # Invalidate cache for this namespace
        self._query_cache = {k: v for k, v in self._query_cache.items() if k[0] != namespace_id}

    def search(
        self,
        namespace_id: int,
        query_vector: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.75,
        use_cache: bool = True,
    ) -> list[SearchResult]:
        """Executes associative search restricted strictly to the specified namespace_id."""
        if namespace_id not in self._partitions:
            return []

        # Check partitioned cache
        cache_key = None
        if use_cache:
            q_hash = hashlib.sha256(query_vector.astype(np.float32).tobytes()).hexdigest()
            cache_key = (namespace_id, q_hash)
            if cache_key in self._query_cache:
                return self._query_cache[cache_key]

        partition = self._partitions[namespace_id]
        results = partition.search(query_vector, top_k=top_k, threshold=threshold)

        if use_cache and cache_key is not None:
            if len(self._query_cache) >= self._cache_capacity:
                # Evict oldest entry
                first_key = next(iter(self._query_cache))
                del self._query_cache[first_key]
            self._query_cache[cache_key] = results

        return results

    def clear_cache(self, namespace_id: int | None = None) -> None:
        """Invalidates query cache entries for a specific namespace or globally."""
        if namespace_id is None:
            self._query_cache.clear()
        else:
            self._query_cache = {k: v for k, v in self._query_cache.items() if k[0] != namespace_id}

    def total_records(self) -> int:
        return sum(p.size for p in self._partitions.values())

    def partition_count(self) -> int:
        return len(self._partitions)

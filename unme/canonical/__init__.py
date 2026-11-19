"""Canonical Memory Representation (CMR) — Fase 3A.

This package implements the neutral dense representation layer that bridges
UNME persistent storage (Level 1 UMR) with LLM-specific hidden states (Level 3).

Modules:
    encoder  — ICanonicalEncoder ABC + MultilingualE5SmallEncoder (768-dim, fp16)
    store    — CanonicalMemoryRecord dataclass + NVMeCanonicalStore
"""

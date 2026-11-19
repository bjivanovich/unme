"""Neural adapter package for UNME — Fase 3B.

This package implements the micro-adapter that connects UNME canonical memory
to a frozen LLM via PyTorch Forward Hooks.  It operates entirely within the
< 256 MB VRAM budget and enforces the zero-stall policy during inference.

Modules:
    projector    — Linear projections between LLM hidden space and canonical space
    cross_attention — Cross-attention between LLM query and CMR value vectors
    gating       — Residual scalar gate alpha with learned activation threshold
    adapter      — IModelAdapter: unified container registering all sub-modules
"""

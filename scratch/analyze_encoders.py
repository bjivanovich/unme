"""Analysis of 768-dim encoder candidates for UNME."""

candidates = [
    {
        "name": "google/gemma-embedding-001 (EmbeddingGemma)",
        "dim": 768,
        "multilingual": True,
        "max_seq": 2048,
        "trust_remote_code": False,
        "needs_st": False,
        "ram_mb": 3800,
        "mrl": True,
        "gated": True,   # requires HF license acceptance
        "notes": (
            "MRL allows truncation to 512/256/128 dims without reindex. "
            "Gated repo: requires HF account + license accept before download. "
            "No offline download possible without prior authentication. "
            "Model weights ~3.8 GB (gemma-2 2B variant)."
        ),
    },
    {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "dim": 768,
        "multilingual": False,  # English-only
        "max_seq": 384,
        "trust_remote_code": False,
        "needs_st": False,
        "ram_mb": 420,
        "mrl": False,
        "gated": False,
        "notes": (
            "English-only. UMR records include Spanish content. "
            "Max 384 tokens (hard limit). Disqualified: not multilingual."
        ),
    },
    {
        "name": "google-bert/bert-base-multilingual-cased (mBERT)",
        "dim": 768,
        "multilingual": True,
        "max_seq": 512,
        "trust_remote_code": False,
        "needs_st": False,
        "ram_mb": 680,
        "mrl": False,
        "gated": False,
        "notes": (
            "NOT trained for semantic similarity. CLS pooling does not produce "
            "meaningful semantic embeddings. BEIR benchmarks show poor retrieval. "
            "Disqualified: inadequate semantic quality."
        ),
    },
    {
        "name": "intfloat/multilingual-e5-base",
        "dim": 768,
        "multilingual": True,
        "max_seq": 512,
        "trust_remote_code": False,
        "needs_st": False,
        "ram_mb": 1100,
        "mrl": False,
        "gated": False,
        "notes": (
            "Direct upgrade from multilingual-e5-small already in tests. "
            "Same query:/passage: prefix convention, same AutoModel API. "
            "Zero code change except model name and path. "
            "100+ languages, trained for semantic similarity (MTEB). "
            "~1.1 GB weights. No gating. No trust_remote_code."
        ),
    },
]

print("=" * 68)
print("UNME Candidate Encoder Analysis")
print("Requirements: 768-dim, multilingual (ES+EN), AutoModel API,")
print("              no gated repo, no sentence_transformers, CPU RAM <= 2 GB")
print("=" * 68)
print()

for c in candidates:
    fails = []
    if c["dim"] != 768:
        fails.append("dim mismatch")
    if not c["multilingual"]:
        fails.append("English-only (UNME needs ES+EN)")
    if c["needs_st"]:
        fails.append("requires sentence_transformers (not installed)")
    if c["gated"]:
        fails.append("gated repo (HF license + auth required)")
    if c["ram_mb"] > 2000:
        fails.append(f"RAM {c['ram_mb']} MB > 2000 MB budget")

    status = "DISQUALIFIED" if fails else "VIABLE"
    print(f"[{status}] {c['name']}")
    for f in fails:
        print(f"  FAIL: {f}")
    print(f"  RAM: ~{c['ram_mb']} MB | Max seq: {c['max_seq']} | "
          f"Gated: {c['gated']} | MRL: {c['mrl']}")
    print(f"  Notes: {c['notes']}")
    print()

print("=" * 68)
print("RECOMMENDATION: intfloat/multilingual-e5-base")
print("  - Passes all requirements")
print("  - Minimal code change (name + path only)")
print("  - Same semantic alignment as multilingual-e5-small tests")
print("  - 1.1 GB RAM, 100+ languages, no auth, no extra deps")
print("  - MTEB NDCG@10 multilingual retrieval: significantly better than small")
print()
print("SECONDARY NOTE on EmbeddingGemma:")
print("  MRL is architecturally interesting for UNME long term.")
print("  However: gated repo blocks offline/sandboxed download.")
print("  Can be revisited as Fase 4 optional upgrade if user accepts the size.")

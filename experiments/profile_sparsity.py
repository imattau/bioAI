"""Phase 1: Profile sparsity tolerance of VSA operations.

Measures:
1. Decoder similarity search: stores N vectors at given sparsity,
   queries with the EXACT same vector. At what density does it still
   find the right match via cosine similarity?
2. Noise tolerance: same but query is sparsified + sign-flip noise.
3. Signal-to-noise ratio: how does sparsity affect the separation
   between matching and non-matching vectors?
"""

import sys
import time
import statistics
import math
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.vsa import VSA, AssociativeStore


def _sparsify(vec: torch.Tensor, density: float) -> torch.Tensor:
    if density >= 1.0:
        return vec
    mask = torch.rand_like(vec, dtype=torch.float) > density
    r = vec.clone()
    r[mask] = 0.0
    return r


def _noise(vec: torch.Tensor, flip_pct: float) -> torch.Tensor:
    mask = torch.rand_like(vec, dtype=torch.float) < flip_pct
    r = vec.clone()
    r[mask & (vec != 0)] *= -1
    return r


@torch.no_grad()
def benchmark_sparsity(density: float, n_store: int = 2000) -> dict:
    vsa = VSA(dim=1000, device="cpu")
    store = AssociativeStore(dim=1000, capacity=n_store)
    stored_vecs = []

    for _ in range(n_store):
        hv = _sparsify(vsa.make_vector(), density)
        store.insert(hv, hv)
        stored_vecs.append(hv)

    # Test 1: exact match (same vector)
    exact_ok = 0
    for hv in stored_vecs[:500]:
        r = store.lookup(hv, k=1)
        if r and r[0][1] > 0.5:
            exact_ok += 1

    # Test 2: noise match (vector + 5% sign flips)
    noise_ok = 0
    for hv in stored_vecs[500:1000]:
        noisy = _noise(hv, 0.05)
        r = store.lookup(noisy, k=1)
        if r and r[0][1] > 0.3:
            noise_ok += 1

    # Test 3: separation (matching sim vs random sim)
    match_sims = []
    rand_sims = []
    for i in range(200):
        hv = stored_vecs[i]
        r = store.lookup(hv, k=2)
        if r:
            match_sims.append(r[0][1])
            if len(r) > 1:
                rand_sims.append(r[1][1])

    sep = (statistics.mean(match_sims) - statistics.mean(rand_sims)
           if match_sims and rand_sims else 0.0)

    # Test 4: proper cosine similarity (L2-normalized)
    n_keys = min(2000, len(stored_vecs))
    keys = torch.stack(stored_vecs[:n_keys])
    k_norm = keys / keys.norm(dim=1, keepdim=True).clamp(min=1e-8)
    q_norm = stored_vecs[0] / stored_vecs[0].norm().clamp(min=1e-8)
    t0 = time.perf_counter()
    for _ in range(50):
        _ = (q_norm.unsqueeze(0) @ k_norm.T)
    matmul_ms = (time.perf_counter() - t0) / 50 * 1000

    # Test 5: sparse matmul via torch.sparse if density < 1
    sparse_ms = 0.0
    if density < 1.0:
        try:
            s_keys = keys.to_sparse_csr()
            s_q = stored_vecs[0].to_sparse()
            t0 = time.perf_counter()
            for _ in range(50):
                _ = s_keys @ s_q
            sparse_ms = (time.perf_counter() - t0) / 50 * 1000
        except Exception:
            sparse_ms = -1.0

    return {
        "density": density,
        "active_dims": int(density * 1000),
        "exact_ok": exact_ok,
        "exact_pct": round(100 * exact_ok / 500, 1) if exact_ok else 0.0,
        "noise_ok": noise_ok,
        "noise_pct": round(100 * noise_ok / 500, 1) if noise_ok else 0.0,
        "separation": round(sep, 4),
        "mean_match_sim": round(statistics.mean(match_sims), 4) if match_sims else 0.0,
        "mean_rand_sim": round(statistics.mean(rand_sims), 4) if rand_sims else 0.0,
        "matmul_ms": round(matmul_ms, 3),
        "sparse_ms": round(sparse_ms, 3) if sparse_ms > 0 else 0.0,
    }


def main():
    densities = [1.0, 0.50, 0.25, 0.10, 0.05, 0.02, 0.01]
    n_store = 2000

    print("=" * 72)
    print("  Sparsity Profile")
    print("  Tests whether VSA operations work with sparse bipolar vectors.")
    print("=" * 72)
    print(f"\n  Store: {n_store} random vectors, 1000-dim")
    print(f"  Query: same vector (exact) + 5% sign-flip noise + random baseline")
    print()

    hdr = f"{'Density':>8} {'Active':>6} {'Exact%':>7} {'Noise%':>7} {'Match':>6} {'Rand':>6} "
    hdr += f"{'Sep':>6} {'Matmul':>7}"
    print(hdr)
    print("-" * 65)

    results = []
    for d in densities:
        r = benchmark_sparsity(d, n_store)
        results.append(r)
        flag = " ✓" if r["exact_pct"] >= 99 else " ✗"
        print(
            f"{r['density']:>7.1%} {r['active_dims']:>6} "
            f"{r['exact_pct']:>6.1f}% {r['noise_pct']:>6.1f}% "
            f"{r['mean_match_sim']:>6.3f} {r['mean_rand_sim']:>6.3f} "
            f"{r['separation']:>6.3f} {r['matmul_ms']:>6.2f}ms{flag} sparse={r.get('sparse_ms', 0):.2f}ms"
        )
        if r["exact_pct"] < 50:
            print(f"  → Stopping: exact-match search collapsed at {d:.1%}")
            break

    print()
    print("Summary:")
    viable = [r for r in results if r["exact_pct"] >= 99]
    if viable:
        best = max(viable, key=lambda x: x["density"])
        print(f"  Dense (100%): {results[0]['exact_pct']:.0f}% exact, "
              f"{results[0]['noise_pct']:.0f}% noise (baseline)")
        print(f"  Max speed (lowest viable density): "
              f"{viable[-1]['density']:.0%} = {viable[-1]['active_dims']} dims, "
              f"matmul {viable[-1]['matmul_ms']:.2f}ms vs "
              f"{results[0]['matmul_ms']:.2f}ms dense")
    collapsed = [r for r in results if r["exact_pct"] < 99]
    if collapsed:
        print(f"  First collapse: {collapsed[0]['density']:.1%} density "
              f"({collapsed[0]['exact_pct']:.1f}% exact)")


if __name__ == "__main__":
    main()

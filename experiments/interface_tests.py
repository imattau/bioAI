import torch
import sys
sys.path.insert(0, "src")

from vsa import VSA, AssociativeStore, HopfieldNet
from clonal import ClonalPool
from immune import SelfMonitor
from basal import GoNoGoActorCritic, MemoryRetrievalEnv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HD_DIM = 1000  # small for fast tests, but high enough for concentration of measure


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test(name: str, condition: bool, detail: str = ""):
    ok = "PASS" if condition else "FAIL"
    print(f"  [{ok}] {name}" + (f" — {detail}" if detail else ""))


def run_experiment_1_encoder_hopfield():
    """Test encoder → Hopfield: capacity, noise tolerance, basin separation."""
    section("Experiment 1: Encoder → Hopfield Store")

    vsa = VSA(dim=HD_DIM, device="cpu")
    hop = HopfieldNet(dim=HD_DIM)

    # Generate and store 10 sentence vectors
    sentences = [vsa.make_vector() for _ in range(10)]
    for s in sentences:
        hop.store(s)
    test("store 10 patterns", len(hop) == 10)

    # Perfect recall from clean cue
    recalled = hop.recall(sentences[0], steps=20)
    sim = vsa.similarity(sentences[0], recalled).item()
    test("clean recall similarity > 0.95", sim > 0.95, f"sim={sim:.4f}")

    # Recall from noisy cue
    noise = 0.3 * torch.randn(HD_DIM)
    recalled = hop.recall(sentences[0] + noise, steps=20)
    sim = vsa.similarity(sentences[0], recalled).item()
    test("noisy recall similarity > 0.9", sim > 0.9, f"sim={sim:.4f}")

    # Recall from very noisy cue (near chance)
    noise = 1.0 * torch.randn(HD_DIM)
    recalled = hop.recall(sentences[0] + noise, steps=20)
    sim = vsa.similarity(sentences[0], recalled).item()
    test("high-noise recall similarity > 0.7", sim > 0.7, f"sim={sim:.4f}")

    # Basin separation: cue near pattern 0 should NOT recall pattern 5
    recalled = hop.recall(sentences[0], steps=20)
    sim_to_5 = vsa.similarity(sentences[5], recalled).item()
    test("basin separation (pattern 0 falls in its own basin)",
         sim_to_5 < 0.5, f"sim to pattern 5={sim_to_5:.4f}")

    # Energy decreases during recall
    cue = sentences[0] + 0.2 * torch.randn(HD_DIM)
    e_before = hop.energy(cue.flatten()).item()
    recalled = hop.recall(cue, steps=20)
    e_after = hop.energy(recalled.flatten()).item()
    test("energy decreases during recall", e_after <= e_before + 1e-4,
         f"energy: {e_before:.2f} → {e_after:.2f}")

    # Capacity test: energy landscape has minima at all stored patterns
    energies = [hop.energy(p.flatten()).item() for p in sentences]
    max_e = max(energies)
    min_e = min(energies)
    test("energy landscape is non-trivial (min < max)", min_e < max_e - 0.01,
         f"energy range: [{min_e:.2f}, {max_e:.2f}]")


def run_experiment_2_hopfield_clonal():
    """Test Hopfield → Clonal: novelty detection, clone quality, forgetting."""
    section("Experiment 2: Hopfield → Clonal Memory")

    vsa = VSA(dim=HD_DIM, device="cpu")
    pool = ClonalPool(input_dim=HD_DIM, affinity_threshold=0.5, clone_margin=0.3, max_modules=10)

    # Store 5 distinct sentence vectors
    patterns = [vsa.make_vector() for _ in range(5)]
    for p in patterns:
        out, created = pool.process(p, lr=0.01)
    test("5 patterns create 5 modules", len(pool) == 5, f"modules={len(pool)}")
    assert len(pool) == 5

    # Novel pattern (different from all 5) should trigger clonal expansion
    novel = vsa.make_vector()
    out, created = pool.process(novel, lr=0.01)
    test("novel pattern triggers expansion", created, f"modules now={len(pool)}")
    assert len(pool) >= 5

    # Near-duplicate with very high noise (affinity ~0.5-0.7) should fine-tune, not clone
    # MAP vectors are highly concentrated; need large noise to reduce affinity below clone threshold
    near = patterns[0].sign() + 1.5 * torch.randn(HD_DIM)
    out, created = pool.process(near, lr=0.01)
    test("high-noise input does NOT trigger expansion (fine-tune path)",
         not created, f"modules={len(pool)}")

    # Measure forgetting: after processing many patterns, all originals still retrievable
    # Each pattern creates a module with a fixed receptor; affinity checks receptor match, not net output
    pool2 = ClonalPool(input_dim=HD_DIM, affinity_threshold=0.4, clone_margin=0.4, max_modules=25)
    originals = [vsa.make_vector() for _ in range(5)]
    for p in originals:
        pool2.process(p, lr=0.01)

    for _ in range(30):
        pool2.process(vsa.make_vector(), lr=0.01)

    # Affinity checks against module.receptor (always fixed), so surviving modules
    # with matching receptors should give perfect affinity
    affinities = []
    for p in originals:
        affs = [m.affinity(p) for m in pool2.modules]
        if affs:
            affinities.append(max(affs).item())
    avg_aff = sum(affinities) / len(affinities) if affinities else 0.0
    test("originals still retrievable via module receptors",
         avg_aff > 0.3, f"avg affinity={avg_aff:.4f}")

    # Pruning: pool should never exceed max_modules
    pool3 = ClonalPool(input_dim=HD_DIM, affinity_threshold=0.3, max_modules=5)
    for _ in range(50):
        pool3.process(vsa.make_vector(), lr=0.01)
    test("pool never exceeds max_modules", len(pool3) <= 5,
         f"modules={len(pool3)}")


def run_experiment_3_hopfield_immune():
    """Test Hopfield → Immune: can monitor distinguish OOD from normal noise."""
    section("Experiment 3: Hopfield → Immune Monitor")

    vsa = VSA(dim=128, device="cpu")
    hop = HopfieldNet(dim=128)
    monitor = SelfMonitor(hop, energy_threshold=3.0)

    normal_patterns = [vsa.make_vector() for _ in range(8)]
    for p in normal_patterns:
        hop.store(p)

    normal_recalls = []
    for p in normal_patterns:
        recalled = hop.recall(p + 0.1 * torch.randn(128), steps=10)
        normal_recalls.append(recalled)
    monitor.calibrate(normal_recalls)
    test("monitor calibrated", monitor.calibrated)

    clean_recall = hop.recall(normal_patterns[0] + 0.1 * torch.randn(128), steps=10)
    normal_score = monitor.score(clean_recall)
    test("normal recall is NOT anomalous", not normal_score["is_anomaly"],
         f"energy_z={normal_score['energy_z']:.2f}")

    ood = torch.randn(128)
    ood_score = monitor.score(ood)
    test("random vector IS anomalous", ood_score["is_anomaly"],
         f"energy_z={ood_score['energy_z']:.2f}")

    ood_flags = 0
    for _ in range(20):
        s = monitor.score(torch.randn(128))
        if s["is_anomaly"]:
            ood_flags += 1
    test("OOD detection rate > 80%", ood_flags >= 16,
         f"flagged {ood_flags}/20 OOD inputs")


def run_experiment_4_action_selection():
    """Test Basal → retrieval: Go/NoGo learns to select correct memory."""
    section("Experiment 4: Action Selection → Memory Retrieval")

    vsa = VSA(dim=64, device="cpu")
    store = AssociativeStore(dim=64, capacity=10)
    queries = [vsa.make_vector() for _ in range(5)]
    for q in queries:
        store.insert(q)

    # Create environment with known correct answers
    env = MemoryRetrievalEnv(store, queries, correct_indices=list(range(5)), dim=64)
    agent = GoNoGoActorCritic(input_dim=64, n_actions=5).to(DEVICE)
    optim = torch.optim.AdamW(agent.parameters(), lr=1e-3)

    # Train for a few episodes
    rewards = []
    for episode in range(50):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            state_t = torch.from_numpy(state).float().to(DEVICE)
            action, value = agent.act(state_t)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            loss = agent.compute_loss(
                state_t, action, reward,
                torch.from_numpy(next_state).float().to(DEVICE) if not done else None
            )
            optim.zero_grad()
            (loss * 0.01).backward()  # scale down for stability
            optim.step()
            total_reward += reward
            state = next_state
        rewards.append(total_reward)

    # Check if reward improved over training
    early = sum(rewards[:10]) / 10
    late = sum(rewards[-10:]) / 10
    test("reward improves with training", late > early,
         f"early={early:.2f}, late={late:.2f}")

    # Test deterministic policy on clean state
    state, _ = env.reset()
    state_t = torch.from_numpy(state).float().to(DEVICE)
    action, _ = agent.act(state_t, deterministic=True)
    test("deterministic action is valid", 0 <= action < 5,
         f"action={action}")


def run_experiment_5_decoder_adaln():
    """Placeholder: Decoder with adaLN conditioning. Not yet implemented."""
    section("Experiment 5: Decoder with adaLN Conditioning")
    print("  [SKIP] Requires adaLN implementation — not yet built.")


def run_experiment_6_full_pipeline():
    """End-to-end: encoder → store → retrieve → monitor → recall."""
    section("Experiment 6: Full Pipeline (all subsystems)")

    vsa = VSA(dim=256, device="cpu")
    store = AssociativeStore(dim=256, capacity=20)
    hop = HopfieldNet(dim=256)

    # Encode and store 5 sentences
    sentences = [vsa.make_vector() for _ in range(10)]
    for s in sentences[:5]:
        store.insert(s)
        hop.store(s)

    # Retrieve via Hopfield from noisy cue
    cue = sentences[0] + 0.2 * torch.randn(256)
    recalled = hop.recall(cue, steps=20)
    sim = vsa.similarity(sentences[0], recalled).item()
    test("full pipeline: retrieval from noisy cue", sim > 0.85, f"sim={sim:.4f}")

    # Store lookup (keyed retrieval)
    results = store.lookup(cue, k=1)
    test("full pipeline: store lookup returns result", len(results) > 0,
         f"top sim={results[0][1]:.4f}")

    # Novel input: not in store, Hopfield should still converge to nearest basin
    novel = vsa.make_vector()
    recalled = hop.recall(novel, steps=20)
    # The Hopfield should still converge to *some* basin (not necessarily the right one)
    # It should not be identical to the input
    diff = (recalled - novel).norm().item()
    test("full pipeline: Hopfield modifies novel input (attractor dynamics)",
         diff > 0.1, f"change={diff:.4f}")

    # Energy of stored patterns is lower than random vectors
    stored_energy = sum(hop.energy(s).item() for s in sentences[:5]) / 5
    random_energy = sum(hop.energy(vsa.make_vector()).item() for _ in range(5)) / 5
    test("full pipeline: stored patterns have lower energy than random",
         stored_energy < random_energy,
         f"stored={stored_energy:.2f}, random={random_energy:.2f}")


def print_summary(results: dict):
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    total = sum(len(v) for v in results.values())
    passed = sum(sum(1 for r in v if r) for v in results.values())
    print(f"  {passed}/{total} tests passed")
    for name, rs in results.items():
        p = sum(1 for r in rs if r)
        print(f"    {name}: {p}/{len(rs)}")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"HD dim: {HD_DIM}")

    results = {
        "Encoder → Hopfield": [],
        "Hopfield → Clonal": [],
        "Hopfield → Immune": [],
        "Action Selection": [],
        "Full Pipeline": [],
    }

    # Run experiment 1
    # We capture passes by printing, store bools manually
    try:
        run_experiment_1_encoder_hopfield()
    except Exception as e:
        print(f"  [ERROR] Experiment 1 failed: {e}")
    print()

    try:
        run_experiment_2_hopfield_clonal()
    except Exception as e:
        print(f"  [ERROR] Experiment 2 failed: {e}")
    print()

    try:
        run_experiment_3_hopfield_immune()
    except Exception as e:
        print(f"  [ERROR] Experiment 3 failed: {e}")
    print()

    try:
        run_experiment_4_action_selection()
    except Exception as e:
        print(f"  [ERROR] Experiment 4 failed: {e}")
    print()

    try:
        run_experiment_5_decoder_adaln()
    except Exception as e:
        print(f"  [ERROR] Experiment 5 failed: {e}")
    print()

    try:
        run_experiment_6_full_pipeline()
    except Exception as e:
        print(f"  [ERROR] Experiment 6 failed: {e}")
    print()

    print(f"\n{'='*60}")
    print(f"  Done. Each [PASS]/[FAIL] indicates interface viability.")
    print(f"  Design iterate on any [FAIL] before proceeding to implementation.")

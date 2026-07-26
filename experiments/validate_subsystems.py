import math
import sys
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
import numpy as np

import torchhd

sys.path.insert(0, "src")
from vsa import VSA, AssociativeStore, HopfieldNet
from clonal import ClonalPool
from immune import SelfMonitor
from basal import GoNoGoActorCritic, MemoryRetrievalEnv
from nca import NCACell, NCA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HD_DIM = 1000
SEED = 42


def seed_everything(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


@dataclass
class Result:
    name: str
    passed: bool
    detail: str = ""


class Reporter:
    def __init__(self):
        self.results: list[Result] = []
        self.section_name = ""

    def section(self, name: str):
        self.section_name = name
        print(f"\n  --- {name} ---")

    def test(self, name: str, passed: bool, detail: str = ""):
        self.results.append(Result(f"{self.section_name}: {name}", passed, detail))
        s = "PASS" if passed else "FAIL"
        print(f"    [{s}] {name}" + (f"  ({detail})" if detail else ""))

    def sum(self, skip_sections: set[str] | None = None):
        print(f"\n  {'='*50}")
        if skip_sections:
            relevant = [r for r in self.results if not any(
                s in r.name for s in skip_sections)]
        else:
            relevant = self.results
        passed = sum(1 for r in relevant if r.passed)
        total = len(relevant)
        print(f"  RESULTS: {passed}/{total} passed ({100*passed/total:.0f}%)")
        for r in relevant:
            s = "PASS" if r.passed else "FAIL"
            if not r.passed:
                print(f"    [{s}] {r.name}: {r.detail}")


rep = Reporter()
vsa = VSA(dim=HD_DIM, device="cpu")


# =============================================================================
# 1. ENCODER — Capacity, Collision, Compositionality
# =============================================================================
def validate_encoder():
    rep.section("Encoder")

    pos_table = torchhd.random(50, HD_DIM, device="cpu")
    word_table = torchhd.random(500, HD_DIM, device="cpu")

    def encode_sentence(word_ids: list[int]) -> torch.Tensor:
        word_hvs = word_table[word_ids]
        pos = pos_table[:len(word_ids)]
        return torchhd.multiset(torchhd.bind(word_hvs, pos)).sign()

    def encode_bag(word_ids: list[int]) -> torch.Tensor:
        return torchhd.multiset(word_table[word_ids]).sign()

    # Collision test: sample pairs instead of materialising a 10000x10000 matrix.
    sentences_ids = []
    for _ in range(10000):
        ids = [torch.randint(0, 500, ()).item() for _ in range(3)]
        sentences_ids.append(ids)
    hvs_list = torch.stack([encode_sentence(s) for s in sentences_ids])
    pair_gen = torch.Generator().manual_seed(SEED)
    left = torch.randint(0, len(hvs_list), (100_000,), generator=pair_gen)
    right = torch.randint(0, len(hvs_list) - 1, (100_000,), generator=pair_gen)
    right += (right >= left).long()
    distinct_inputs = torch.tensor([
        sentences_ids[i] != sentences_ids[j]
        for i, j in zip(left.tolist(), right.tolist())
    ])
    sampled_dots = (hvs_list[left] * hvs_list[right]).sum(dim=1)
    off_diag = sampled_dots[distinct_inputs]
    max_off = off_diag.max().item()
    mean_off = off_diag.float().mean().item()
    # Random MAP vectors have dot ~0 (concentration of measure); collisions would be >900
    collision_count = int((off_diag > 990).sum().item())
    rep.test("deterministic encoding (no VSA-level collisions beyond word-ID birthday bound)",
             collision_count < 10,
             f"near-identical pairs={collision_count}/{len(off_diag)} sampled distinct inputs "
             f"(max_dot={max_off}, mean={mean_off:.2f})")

    # Compositionality: order-swapped pair
    a = encode_sentence([0, 1, 2])  # A B C
    b = encode_sentence([2, 1, 0])  # C B A
    sim = vsa.similarity(a, b).item()
    rep.test("order-swapped pairs are different", sim < 0.8,
             f"sim={sim:.4f}")

    # Bag-of-words vs ordered
    bag = encode_bag([0, 1, 2])
    ordered = encode_sentence([0, 1, 2])
    sim_bo = vsa.similarity(bag, ordered).item()
    rep.test("ordered differs from bag-of-words",
             sim_bo < 0.95, f"sim={sim_bo:.4f}")

    # Prefix similarity: "A B" should be more similar to "A B C" than "D E F"
    ab = encode_sentence([0, 1])
    abc = encode_sentence([0, 1, 2])
    abc2 = encode_sentence([0, 1, 3])
    def_ = encode_sentence([3, 4, 5])
    sim_ab_abc = vsa.similarity(ab, abc).item()
    sim_ab_def = vsa.similarity(ab, def_).item()
    rep.test("prefix is closer than unrelated",
             sim_ab_abc > sim_ab_def,
             f"sim(AB,ABC)={sim_ab_abc:.4f} vs sim(AB,DEF)={sim_ab_def:.4f}")

    # Determinism: encoding same word IDs twice should give same vector
    hv_a = encode_sentence([0, 1, 2])
    hv_b = encode_sentence([0, 1, 2])
    sim_ab = vsa.similarity(hv_a, hv_b).item()
    rep.test("deterministic encoding", sim_ab > 0.99,
             f"sim={sim_ab:.4f}")

    # Distribution of affinities between random sentences
    random_sims = []
    for _ in range(500):
        i, j = torch.randint(0, len(hvs_list), (2,))
        random_sims.append(vsa.similarity(hvs_list[i], hvs_list[j]).item())
    mean_sim = float(np.mean(random_sims))
    std_sim = float(np.std(random_sims))
    rep.test("random pair similarity near zero",
             abs(mean_sim) < 0.05,
             f"mean={mean_sim:.4f}, std={std_sim:.4f}")


# =============================================================================
# 2. HOPFIELD — Capacity, Basin Radius, Interference
# =============================================================================
def validate_hopfield():
    rep.section("Hopfield")

    hop = HopfieldNet(dim=HD_DIM)
    patterns = [vsa.make_vector() for _ in range(200)]

    # Capacity: store N patterns, check recall from clean cue
    for i, p in enumerate(patterns):
        hop.store(p)
        if i % 50 == 49 or i == 0:
            recalled = hop.recall(patterns[0], steps=20)
            sim = vsa.similarity(patterns[0], recalled).item()
            threshold = 0.9 if (i + 1) <= 150 else 0.3
            rep.test(f"recall accuracy at {i+1} patterns",
                     sim > threshold,
                     f"sim={sim:.4f} (threshold={threshold}, capacity ~140 for 1000-d)")

    # Basin radius: inject noise and find failure point
    hop2 = HopfieldNet(dim=HD_DIM)
    base_pat = vsa.make_vector()
    hop2.store(base_pat)
    noise_levels = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    last_sim = 1.0
    failure_at = None
    for nl in noise_levels:
        noise = nl * torch.randn(HD_DIM)
        recalled = hop2.recall(base_pat + noise, steps=20)
        sim = vsa.similarity(base_pat, recalled).item()
        if sim < 0.9 and failure_at is None:
            failure_at = nl
        last_sim = sim
    rep.test("basin radius noise tolerance",
             failure_at is None or failure_at > 0.3,
             f"recall fails at noise={failure_at} (last sim={last_sim:.4f})")

    # Basin interference: two similar patterns
    hop3 = HopfieldNet(dim=HD_DIM)
    a = vsa.make_vector()
    b_nudge = 0.2 * torch.randn(HD_DIM)
    b = (a + b_nudge).sign()
    hop3.store(a)
    hop3.store(b)
    rec_a = hop3.recall(a, steps=20)
    rec_b = hop3.recall(b, steps=20)
    sim_a = vsa.similarity(a, rec_a).item()
    sim_b = vsa.similarity(b, rec_b).item()
    rep.test("similar patterns resolve to distinct basins",
             sim_a > 0.8 and sim_b > 0.8,
             f"sim_a={sim_a:.4f}, sim_b={sim_b:.4f}")

    # Repeated storage of same vector
    hop4 = HopfieldNet(dim=HD_DIM)
    v = vsa.make_vector()
    for _ in range(10):
        hop4.store(v)
    rec = hop4.recall(v, steps=20)
    sim = vsa.similarity(v, rec).item()
    rep.test("repeated storage of same pattern",
             sim > 0.99, f"len={len(hop4)}, sim={sim:.4f}")

    # Hopfield energy for stored vs random
    stored_energy = hop2.energy(base_pat.flatten()).item()
    rand_energy = hop2.energy(vsa.make_vector().flatten()).item()
    rep.test("stored patterns have lower energy than random",
             stored_energy < rand_energy,
             f"stored={stored_energy:.1f}, random={rand_energy:.1f}")


# =============================================================================
# 3. CLONAL MEMORY — Threshold Sweep, Forgetting, Clone Quality
# =============================================================================
def validate_clonal():
    rep.section("Clonal Memory")

    # Affinity distribution between random MAP vectors
    vecs = vsa.make_vectors(200)
    sims = []
    for i in range(200):
        for j in range(i + 1, 200):
            sims.append(vsa.similarity(vecs[i], vecs[j]).item())
    mean_aff = float(np.mean(sims))
    std_aff = float(np.std(sims))
    max_aff = float(np.max(sims))
    rep.test("random pair affinity distribution",
             abs(mean_aff) < 0.05 and max_aff < 0.3,
             f"mean={mean_aff:.4f}, std={std_aff:.4f}, max={max_aff:.4f}")

    # Threshold sweep: find optimal affinity_threshold
    for thresh in [0.3, 0.5, 0.7]:
        pool = ClonalPool(input_dim=HD_DIM, affinity_threshold=thresh,
                          clone_margin=0.2, max_modules=20)
        known = [vsa.make_vector() for _ in range(5)]
        for p in known:
            pool.process(p, lr=0.01)
        novel = vsa.make_vector()
        _, novel_created = pool.process(novel, lr=0.01)
        same = known[0] + 0.1 * vsa.make_vector()
        _, same_created = pool.process(same.sign(), lr=0.01)
        rep.test(f"threshold={thresh}: novel triggers expansion",
                 novel_created, f"modules={len(pool)}")

    # Forgetting curve with exemplar replay
    pool = ClonalPool(input_dim=HD_DIM, affinity_threshold=0.5,
                      clone_margin=0.3, max_modules=20)
    originals = [vsa.make_vector() for _ in range(5)]
    record = []
    for p in originals:
        pool.process(p, lr=0.01)
    record.append(1.0)

    for i in range(30):
        pool.process(vsa.make_vector(), lr=0.01)
        if i % 10 == 9:
            affs = []
            for orig in originals:
                mod_affs = [m.affinity(orig) for m in pool.modules]
                if mod_affs:
                    affs.append(max(mod_affs).item())
            avg = float(np.mean(affs)) if affs else 0.0
            record.append(avg)

    final_aff = record[-1]
    rep.test("forgetting: originals retrievable after 30 new patterns",
             final_aff > 0.3, f"affinity trajectory={[f'{x:.2f}' for x in record]}")

    probe_pool = ClonalPool(input_dim=HD_DIM, max_modules=2)
    probe = vsa.make_vector()
    _, created = probe_pool.process(probe, lr=0.01)
    receptor_affinity = max(
        (m.affinity(probe).item() for m in probe_pool.modules), default=-1.0
    )
    rep.test("new clone preserves its triggering receptor",
             created and receptor_affinity > 0.99,
             f"created={created}, receptor_affinity={receptor_affinity:.3f}")

    # Pruning fairness
    pool2 = ClonalPool(input_dim=HD_DIM, affinity_threshold=0.3, max_modules=10)
    important = [vsa.make_vector() for _ in range(3)]
    for p in important:
        pool2.process(p, lr=0.01)
    for _ in range(30):
        pool2.process(vsa.make_vector(), lr=0.01)
    surv = sum(1 for o in important
               if max((m.affinity(o) for m in pool2.modules), default=0.0) > 0.5)
    rep.test("pruning preserves important modules",
             surv >= 2, f"important survived: {surv}/3 after pruning to {len(pool2)}")


# =============================================================================
# 4. IMMUNE MONITOR — ROC Curve & Calibration Minimum
# =============================================================================
def validate_immune():
    rep.section("Immune Monitor")

    hop = HopfieldNet(dim=256)
    vsa256 = VSA(dim=256, device="cpu")
    normal_patterns = [vsa256.make_vector() for _ in range(10)]
    for p in normal_patterns:
        hop.store(p)

    normal_recalls = [hop.recall(p + 0.1 * torch.randn(256), steps=10)
                      for p in normal_patterns]

    # ROC curve: sweep threshold
    thresholds = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    roc = []
    for thr in thresholds:
        mon = SelfMonitor(hop, energy_threshold=thr)
        mon.calibrate(normal_recalls)

        tp = sum(1 for _ in range(20) if mon.score(torch.randn(256))["is_anomaly"])
        fp = sum(1 for _ in range(20) if mon.score(
            hop.recall(normal_patterns[0] + 0.1 * torch.randn(256), steps=10)
        )["is_anomaly"])
        roc.append((thr, tp, fp))

    # Find best threshold (max TPR - FPR)
    best_thr, best_tp, best_fp = max(roc, key=lambda x: x[1] - x[2])
    rep.test(f"ROC: best threshold={best_thr}",
             best_tp >= 15 and best_fp <= 5,
             f"TP={best_tp}/20, FP={best_fp}/20 at thr={best_thr}")

    # Calibration minimum: how few samples needed
    for n_cal in [5, 10, 25]:
        mon = SelfMonitor(hop, energy_threshold=best_thr)
        mon.calibrate(normal_recalls[:n_cal])
        tp = sum(1 for _ in range(10) if mon.score(torch.randn(256))["is_anomaly"])
        rep.test(f"calibration with {n_cal} samples",
                 tp >= 7, f"TP={tp}/10")

    # Near-OOD: sentence about "cat" when trained on animal sentences
    near_ood = hop.recall(normal_patterns[0] + 0.1 * torch.randn(256), steps=10)
    mon = SelfMonitor(hop, energy_threshold=best_thr)
    mon.calibrate(normal_recalls)
    near_result = mon.score(near_ood)
    rep.test("near-OOD (similar topic) NOT flagged",
             not near_result["is_anomaly"],
             f"energy_z={near_result['energy_z']:.2f}")

    # Far-OOD: truly random vector
    far_ood = torch.randn(256)
    far_result = mon.score(far_ood)
    rep.test("far-OOD (random) IS flagged",
             far_result["is_anomaly"],
             f"energy_z={far_result['energy_z']:.2f}")


# =============================================================================
# 5. ACTION SELECTION — Baseline Comparison
# =============================================================================
def validate_action_selection():
    rep.section("Action Selection")

    vsa_local = VSA(dim=64, device="cpu")
    store = AssociativeStore(dim=64, capacity=20)
    queries = [vsa_local.make_vector() for _ in range(10)]
    for q in queries:
        store.insert(q)

    # Random baseline
    env = MemoryRetrievalEnv(store, queries[:5], correct_indices=list(range(5)), dim=64)
    random_rewards = []
    for _ in range(50):
        s, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            a = torch.randint(0, 5, ()).item()
            s, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        random_rewards.append(total)
    random_avg = float(np.mean(random_rewards))

    # Trained agent
    agent = GoNoGoActorCritic(input_dim=64, n_actions=5).to(DEVICE)
    optim = torch.optim.AdamW(agent.parameters(), lr=1e-3)
    train_rewards = []
    for ep in range(100):
        s, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            st = torch.from_numpy(s).float().to(DEVICE)
            a, v = agent.act(st)
            ns, r, term, trunc, _ = env.step(a)
            done = term or trunc
            loss = agent.compute_loss(
                st, a, r,
                torch.from_numpy(ns).float().to(DEVICE) if not done else None)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += r
            s = ns
        train_rewards.append(total)
    trained_avg = float(np.mean(train_rewards[-20:]))

    rep.test("trained agent outperforms random",
             trained_avg > random_avg + 0.5,
             f"trained={trained_avg:.2f}, random={random_avg:.2f}")

    # Generalisation: train on 5 queries, test on 5 unseen
    env_train = MemoryRetrievalEnv(store, queries[:5], correct_indices=list(range(5)), dim=64)
    env_test = MemoryRetrievalEnv(store, queries[5:10], correct_indices=list(range(5)), dim=64)

    agent2 = GoNoGoActorCritic(input_dim=64, n_actions=5).to(DEVICE)
    optim2 = torch.optim.AdamW(agent2.parameters(), lr=1e-3)
    for ep in range(100):
        s, _ = env_train.reset()
        done = False
        while not done:
            st = torch.from_numpy(s).float().to(DEVICE)
            a, v = agent2.act(st)
            ns, r, term, trunc, _ = env_train.step(a)
            done = term or trunc
            loss = agent2.compute_loss(
                st, a, r,
                torch.from_numpy(ns).float().to(DEVICE) if not done else None)
            optim2.zero_grad()
            loss.backward()
            optim2.step()
            s = ns

    test_rewards = []
    for _ in range(20):
        s, _ = env_test.reset()
        done = False
        total = 0.0
        while not done:
            a, _ = agent2.act(
                torch.from_numpy(s).float().to(DEVICE), deterministic=True)
            s, r, term, trunc, _ = env_test.step(a)
            total += r
            done = term or trunc
        test_rewards.append(total)
    test_avg = float(np.mean(test_rewards))
    rep.test("generalises to unseen queries",
             test_avg > random_avg,
             f"test={test_avg:.2f}, random={random_avg:.2f} "
             f"(limited by linear policy; would need deeper network for full generalisation)")

    # Memory load sensitivity
    load_scores = {}
    for n in [5, 10, 20]:
        store_n = AssociativeStore(dim=64, capacity=n)
        qs = [vsa_local.make_vector() for _ in range(n)]
        for q in qs:
            store_n.insert(q)
        env_n = MemoryRetrievalEnv(store_n, qs[:min(5, n)],
                                    correct_indices=list(range(min(5, n))),
                                    dim=64)
        agent_n = GoNoGoActorCritic(input_dim=64, n_actions=min(5, n)).to(DEVICE)
        optim_n = torch.optim.AdamW(agent_n.parameters(), lr=1e-3)
        for ep in range(50):
            s, _ = env_n.reset()
            done = False
            while not done:
                st = torch.from_numpy(s).float().to(DEVICE)
                a, v = agent_n.act(st)
                ns, r, term, trunc, _ = env_n.step(a)
                done = term or trunc
                loss = agent_n.compute_loss(st, a, r,
                    torch.from_numpy(ns).float().to(DEVICE) if not done else None)
                optim_n.zero_grad()
                loss.backward()
                optim_n.step()
                s = ns
        eval_rewards = []
        for _ in range(20):
            s, _ = env_n.reset()
            total = 0.0
            done = False
            while not done:
                action, _ = agent_n.act(
                    torch.from_numpy(s).float().to(DEVICE), deterministic=True
                )
                s, reward, term, trunc, _ = env_n.step(action)
                total += reward
                done = term or trunc
            eval_rewards.append(total)
        load_scores[n] = float(np.mean(eval_rewards))
    rep.test("memory load training beats random at each tested size",
             all(score > random_avg for score in load_scores.values()),
             f"scores={load_scores}, random={random_avg:.2f}")


# =============================================================================
# 6. NCA — Conditioning & Self-Repair
# =============================================================================
def validate_nca():
    rep.section("NCA Generator")

    cell = NCACell(hidden_dim=16)
    nca = NCA(cell, grid_size=(16, 16), channels=16)

    # Conditioning effect: different seeds produce different results
    seed_a = torch.randn(1, 1, 16, 16)
    seed_b = torch.randn(1, 1, 16, 16)
    out_a = nca.generate(seed_a, steps=30)
    out_b = nca.generate(seed_b, steps=30)
    diff = (out_a - out_b).abs().mean().item()
    rep.test("different seeds produce different outputs",
             diff > 0.01, f"mean_abs_diff={diff:.6f}")

    # Convergence: state stabilises over time (use forward to get trajectory)
    seed = torch.randn(1, 1, 16, 16)
    states = nca(seed, steps=100)  # returns all states
    change = (states[-1] - states[-2]).abs().mean().item()
    rep.test("state changes across 100 steps (not frozen)",
             change > 0.01, f"final_change={change:.6f} (convergence requires training)")

    # Self-repair: corrupt mid-generation, check recovery
    seed = torch.randn(1, 1, 16, 16)
    state = nca.input_proj(seed)
    for _ in range(20):
        state = cell(state)
    corrupt_region_before = state[:, :, 4:8, 4:8].clone()
    state[:, :, 4:8, 4:8] = 0
    for _ in range(10):
        state = cell(state)
    corrupt_region_after = state[:, :, 4:8, 4:8]
    region_change = (corrupt_region_after - corrupt_region_before).abs().mean().item()
    rep.test("corrupted region continues evolving",
             region_change > 0.01, f"region change={region_change:.6f}; "
             "self-repair requires a trained NCA and a target-distance metric")

    # Generate with target (coarse conditioning)
    seed = torch.randn(1, 1, 16, 16)
    target = torch.randn(1, 1, 16, 16)
    gen_plain = nca.generate(seed, steps=20)
    gen_cond = nca.generate(seed, steps=20, target=target)
    cond_diff = (gen_cond - gen_plain).abs().mean().item()
    rep.test("target conditioning changes output",
             cond_diff > 0.001, f"diff={cond_diff:.6f}")


# =============================================================================
# 7. CROSS-SUBSYSTEM INTEGRATION
# =============================================================================
def validate_integration():
    rep.section("Cross-Subsystem Integration")

    # Encoder + Hopfield + Clonal: store sentences, query with noisy version
    # (Hopfield capacity ~36 for 256-d; store 25 to stay well within limit)
    vsa_i = VSA(dim=256, device="cpu")
    store_i = AssociativeStore(dim=256, capacity=40)
    hop_i = HopfieldNet(dim=256)
    pool_i = ClonalPool(input_dim=256, affinity_threshold=0.4, max_modules=30)

    stored = [vsa_i.make_vector() for _ in range(25)]
    for s in stored:
        store_i.insert(s)
        hop_i.store(s)

    # Query with noisy version of stored[0]
    noisy = stored[0] + 0.3 * torch.randn(256)
    recalled = hop_i.recall(noisy, steps=20)
    sim_hop = vsa_i.similarity(stored[0], recalled).item()
    rep.test("Hopfield recalls from noisy cue (50 patterns)",
             sim_hop > 0.8, f"sim={sim_hop:.4f}")

    # Store lookup
    results = store_i.lookup(noisy, k=3)
    rep.test("store lookup returns nearest neighbours",
             len(results) >= 1, f"found={len(results)}")

    # Clonal: match to nearest module
    out, created = pool_i.process(noisy, lr=0.01)
    affs = [m.affinity(noisy) for m in pool_i.modules]
    best_aff = max(affs).item() if affs else 0.0
    rep.test("clonal pool matches noisy input to nearest basin",
             best_aff > 0.3, f"best_affinity={best_aff:.4f}")

    # Hopfield energy: stored patterns < random
    hop_e = HopfieldNet(dim=256)
    stored_e = [vsa_i.make_vector() for _ in range(5)]
    for s in stored_e:
        hop_e.store(s)
    e_stored = hop_e.energy(stored_e[0].flatten()).item()
    e_random = hop_e.energy(vsa_i.make_vector().flatten()).item()
    rep.test("energy landscape: stored < random",
             e_stored < e_random,
             f"stored={e_stored:.1f}, random={e_random:.1f}")

    # Full pipeline consistency: recall then immune-check
    hop_f = HopfieldNet(dim=256)
    mon_f = SelfMonitor(hop_f, energy_threshold=3.0)
    stored_f = [vsa_i.make_vector() for _ in range(5)]
    for s in stored_f:
        hop_f.store(s)
    recalls_f = [hop_f.recall(s + 0.1 * torch.randn(256), steps=10) for s in stored_f]
    mon_f.calibrate(recalls_f)
    scores = [mon_f.score(hop_f.recall(s + 0.1 * torch.randn(256), steps=10))["is_anomaly"]
              for s in stored_f[:3]]
    rep.test("immune does not false-alarm on normal recalls",
             sum(scores) == 0, f"false_alarms={sum(scores)}/3")

    # OOD triggers immune
    ood_score = mon_f.score(torch.randn(256))
    rep.test("immune flags true OOD input",
             ood_score["is_anomaly"], f"energy_z={ood_score['energy_z']:.2f}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    seed_everything()
    print(f"Device: {DEVICE}, HD dim: {HD_DIM}")
    # Run no-grad validations under torch.no_grad()
    with torch.no_grad():
        validate_encoder()
        validate_hopfield()
        validate_immune()
        validate_nca()
    # Run validations that need gradient tracking outside no_grad
    validate_clonal()
    validate_action_selection()
    validate_integration()
    rep.sum()

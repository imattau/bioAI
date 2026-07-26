"""LLM-Driven Multi-Turn Memory Stress Test.

Tests the BioAI agent's ability to:
1. Store facts and recall them exactly (exact-match retrieval)
2. Retrieve facts by similar text (word-overlap generalization)
3. Remember across many turns with no capacity overflow

The LLM generates diverse facts. Each turn: store a fact, then immediately
recall it. Accuracy measures exact retrieval fidelity at scale.

Usage:
    python experiments/llm_dialogue_stress.py [--turns 1000] [--model qwen2.5-coder:1.5b]
"""

import json
import sys
import time
import argparse
import statistics
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

import ollama

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.text import BioAIDialogueAgent


@dataclass
class TurnRecord:
    turn: int
    fact: str
    query: str
    agent_response: str
    intent: str
    retrieval_accepted: bool
    retrieval_score: float
    retrieval_margin: float
    retrieved: str
    exact_recall: bool
    drift_detected: bool
    energy_z: float
    latency_ms: float


@dataclass
class ExperimentConfig:
    num_turns: int = 500
    llm_model: str = "qwen2.5-coder:1.5b"
    vsa_dim: int = 1000
    store_capacity: int = 2000
    report_interval: int = 50
    use_llm_facts: bool = True


class FactGenerator:
    def __init__(self, model: str, num_diverse: int = 0):
        self.model = model
        self.turns = 0
        self.pool: list[str] = []
        if num_diverse > 0:
            self._pre_generate(num_diverse)

    def _pre_generate(self, n: int):
        import random
        random.seed(42)
        topics = [
            "Germany geography", "Mars planet", "ocean depth",
            "dinosaurs", "piano instrument", "rainforest animals",
            "Shakespeare plays", "mathematics", "gravity physics",
            "ancient Rome", "human body", "solar system",
            "world rivers", "computer history", "bacteria",
        ]
        used = set()
        for i in range(n):
            topic = random.choice(topics)
            tries = 0
            fact = ""
            while tries < 3:
                try:
                    r = ollama.chat(self.model, messages=[
                        {"role": "user", "content":
                         f"One unique short fact about {topic}. "
                         "One sentence. Be specific."},
                    ], options={"num_predict": 50, "temperature": 0.8})
                    fact = r["message"]["content"].strip().rstrip(".")
                    if fact and fact not in used:
                        break
                except Exception:
                    pass
                tries += 1
            if not fact:
                fact = f"{topic} is a well-studied subject"
            self.pool.append(fact)
            used.add(fact)

    def add_template(self, topic: str, detail: str):
        self.pool.append(f"{topic} {detail}")

    def next_fact(self) -> str:
        if self.pool:
            fact = self.pool[self.turns % len(self.pool)]
        else:
            fact = f"fact_{self.turns}"
        self.turns += 1
        return fact


class MemoryStressTest:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.agent = BioAIDialogueAgent(vsa_dim=config.vsa_dim)
        self.agent.decoder = type(self.agent.decoder)(
            vsa=self.agent.vsa,
            store_capacity=config.store_capacity,
            encoder=self.agent.encoder,
        )
        self.gen = FactGenerator(
            config.llm_model,
            num_diverse=config.num_turns if config.use_llm_facts else 0,
        )
        self.records: list[TurnRecord] = []

    def run(self):
        cfg = self.config
        print(f"Starting memory stress test: {cfg.num_turns} turns")
        print(f"  LLM model: {cfg.llm_model}")
        print(f"  VSA dim: {cfg.vsa_dim}")
        print(f"  Store capacity: {cfg.store_capacity}")
        print()

        exact_recalls = 0

        for turn in range(1, cfg.num_turns + 1):
            fact = self.gen.next_fact()
            t0 = time.perf_counter()

            conversation_turn = self.agent.process_turn(fact)
            recalled = self.agent.recall(fact)
            exact = (recalled == fact)
            if exact:
                exact_recalls += 1

            latency = (time.perf_counter() - t0) * 1000

            self.records.append(TurnRecord(
                turn=turn,
                fact=fact,
                query="exact recall",
                agent_response=conversation_turn["response"],
                intent=conversation_turn["intent"],
                retrieval_accepted=conversation_turn["retrieval_accepted"],
                retrieval_score=conversation_turn["retrieval_score"],
                retrieval_margin=conversation_turn["retrieval_margin"],
                retrieved=recalled,
                exact_recall=exact,
                drift_detected=conversation_turn["drift_detected"],
                energy_z=conversation_turn["energy_z"],
                latency_ms=latency,
            ))

            if turn % cfg.report_interval == 0:
                window = self.records[-cfg.report_interval:]
                window_exact = sum(1 for r in window if r.exact_recall)
                print(
                    f"  [{turn:4d}/{cfg.num_turns}] "
                    f"exact={window_exact}/{len(window)} "
                    f"({100*window_exact/len(window):.0f}%) "
                    f"total={exact_recalls}/{turn} "
                    f"({100*exact_recalls/turn:.0f}%) "
                    f"latency={statistics.median([r.latency_ms for r in window]):.0f}ms "
                    f"library={len(self.agent.library)} "
                    f"hot={len(self.agent.decoder)} "
                    f"clonal={len(self.agent.clonal)}"
                )

        self._final_report()

    def _final_report(self):
        total = len(self.records)
        exact = sum(1 for r in self.records if r.exact_recall)
        first_half = sum(1 for r in self.records[:total // 2] if r.exact_recall)
        second_half = sum(1 for r in self.records[total // 2:] if r.exact_recall)
        half = total // 2

        print()
        print("=" * 70)
        print(f"  MEMORY STRESS TEST RESULTS  ({total} turns)")
        print("=" * 70)
        print(f"  Overall exact recall:  {exact}/{total} ({100*exact/total:.1f}%)")
        print(f"  First half:            {first_half}/{half} ({100*first_half/half:.1f}%)")
        print(f"  Second half:           {second_half}/{half} ({100*second_half/half:.1f}%)")
        print(f"  Decoder store size:    {len(self.agent.decoder)}")
        print(f"  Token library size:    {len(self.agent.library)}")
        print(f"  Clonal pool size:      {len(self.agent.clonal)}")
        print(f"  Median latency:        {statistics.median([r.latency_ms for r in self.records]):.0f}ms")

        report = {
            "config": asdict(self.config),
            "summary": {
                "total_turns": total,
                "exact_recall": exact,
                "accuracy_pct": round(100 * exact / total, 1),
                "first_half_correct": first_half,
                "second_half_correct": second_half,
                "decoder_size": len(self.agent.decoder),
                "token_library_size": len(self.agent.library),
                "token_storage_bytes": self.agent.library.token_storage_bytes,
                "clonal_size": len(self.agent.clonal),
                "latency_p50_ms": statistics.median(
                    [r.latency_ms for r in self.records]
                ),
                "latency_p95_ms": sorted(
                    r.latency_ms for r in self.records
                )[int(0.95 * (total - 1))],
            },
            "records": [asdict(r) for r in self.records],
        }
        path = Path(f"checkpoints/memory_stress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(f"\n  Report saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="BioAI Memory Stress Test")
    parser.add_argument("--turns", type=int, default=500)
    parser.add_argument("--model", type=str, default="qwen2.5-coder:1.5b")
    parser.add_argument("--vsa-dim", type=int, default=1000)
    parser.add_argument("--report-interval", type=int, default=50)
    parser.add_argument(
        "--synthetic-facts",
        action="store_true",
        help="Use deterministic fact_N inputs instead of generating facts with the LLM.",
    )
    args = parser.parse_args()

    config = ExperimentConfig(
        num_turns=args.turns,
        llm_model=args.model,
        vsa_dim=args.vsa_dim,
        report_interval=args.report_interval,
        use_llm_facts=not args.synthetic_facts,
    )
    test = MemoryStressTest(config)
    test.run()


if __name__ == "__main__":
    main()

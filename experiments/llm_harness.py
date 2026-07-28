import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import torchhd
import ollama

sys.path.insert(0, "src")
from vsa import VSA, AssociativeStore, HopfieldNet
from clonal import ClonalPool
from immune import SelfMonitor
from basal import GoNoGoActorCritic

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HD_DIM = 10000

LLM_MODEL = "gemma4:12b"


@dataclass
class TestResult:
    scenario: str
    pipeline_stage: str
    passed: bool
    details: str
    llm_evaluation: str = ""


class LLMGenerator:
    def __init__(self, model: str = LLM_MODEL):
        self.model = model

    def generate_sentence(self, prompt: str) -> str:
        r = ollama.chat(self.model, messages=[
            {"role": "system", "content": "You are a test data generator for a VSA-based AI system. "
             "Generate exactly one short sentence (3-5 words) using only simple English words. "
             "Return ONLY the sentence, no explanation."},
            {"role": "user", "content": prompt},
        ])
        return r["message"]["content"].strip().strip("\"'")

    def generate_order_pairs(self, prompt: str) -> tuple[str, str]:
        r = ollama.chat(self.model, messages=[
            {"role": "system", "content": "Generate exactly two sentences that use the same words "
             "but in different order, like 'dog bites man' and 'man bites dog'. "
             "Return as: FIRST | SECOND"},
            {"role": "user", "content": prompt},
        ])
        parts = r["message"]["content"].split("|")
        return parts[0].strip(), parts[1].strip()

    def generate_ood_input(self, known_topics: list[str]) -> str:
        r = ollama.chat(self.model, messages=[
            {"role": "system", "content": "Generate a short sentence about a topic completely "
             "unrelated to these topics: " + ", ".join(known_topics) + ". "
             "Return ONLY the sentence, no explanation."},
            {"role": "user", "content": "Generate something unrelated."},
        ])
        return r["message"]["content"].strip().strip("\"'")

    def evaluate_output(self, input_sentence: str, output_text: str, criterion: str) -> str:
        r = ollama.chat(self.model, messages=[
            {"role": "system", "content": "You are a test evaluator. Given an input, output, and "
             "evaluation criterion, determine if the test PASSED or FAILED. "
             "Return exactly: PASS or FAIL followed by a brief reason."},
            {"role": "user", "content": f"Input: {input_sentence}\nOutput: {output_text}\n"
             f"Criterion: {criterion}"},
        ])
        return r["message"]["content"].strip()


class PipelineRunner:
    def __init__(self):
        self.vsa = VSA(dim=HD_DIM, device="cpu")
        self.store = AssociativeStore(dim=HD_DIM, capacity=100)
        self.hop = HopfieldNet(dim=HD_DIM)
        self.vocab = {}
        self.vocab_vecs = {}
        self.pos_vectors = torchhd.random(32, HD_DIM, device="cpu")

    def _ensure_word(self, word: str) -> torch.Tensor:
        word = word.lower().strip(".,!?;:'\"")
        if word not in self.vocab:
            self.vocab[word] = len(self.vocab)
            self.vocab_vecs[word] = self.vsa.make_vector()
        return self.vocab_vecs[word]

    def encode_sentence(self, sentence: str) -> torch.Tensor:
        words = sentence.lower().strip(".,!?;:'\"").split()
        vecs = [self._ensure_word(w) for w in words]
        pos = self.pos_vectors[:len(vecs)]
        bound = torchhd.bind(torch.stack(vecs), pos)
        bundle = torchhd.multiset(bound)
        return bundle.sign()

    def store_sentence(self, sentence: str) -> torch.Tensor:
        hv = self.encode_sentence(sentence)
        self.store.insert(hv)
        self.hop.store(hv)
        return hv

    def retrieve(self, sentence: str, use_hopfield: bool = True) -> torch.Tensor:
        hv = self.encode_sentence(sentence)
        if use_hopfield:
            return self.hop.recall(hv, steps=20)
        return hv

    def lookup_store(self, sentence: str, k: int = 3) -> list:
        hv = self.encode_sentence(sentence)
        return self.store.lookup(hv, k=k)

    def check_novelty(self, sentence: str, pool: ClonalPool) -> bool:
        hv = self.encode_sentence(sentence)
        _, created = pool.process(hv)
        return created  # True = novel

    def check_ood(self, sentence: str, monitor: SelfMonitor) -> dict:
        hv = self.encode_sentence(sentence)
        return monitor.score(hv)


class CyclicHarness:
    def __init__(self):
        self.gen = LLMGenerator()
        self.pipeline = PipelineRunner()
        self.results: list[TestResult] = []
        self.max_tokens_per_word = 0
        self.vocab_set: set[str] = set()

    def add_result(self, scenario: str, stage: str, passed: bool, details: str,
                   llm_eval: str = ""):
        self.results.append(TestResult(scenario, stage, passed, details, llm_eval))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {stage}: {details}")

    def run_encoder_cycle(self):
        print("\n=== Cycle 1: Encoder → Store → Hopfield ===")
        sentence = self.gen.generate_sentence("Generate a short sentence about a cat.")
        print(f"  Input: {sentence}")
        hv = self.pipeline.store_sentence(sentence)

        hop_recalled = self.pipeline.hop.recall(hv, steps=20)
        sim = self.pipeline.vsa.similarity(hv, hop_recalled).item()
        self.add_result(sentence, "store-retrieve", sim > 0.9, f"sim={sim:.4f}")

        results = self.pipeline.lookup_store(sentence)
        self.add_result(sentence, "store-lookup", len(results) > 0,
                        f"found={len(results)}")

    def run_order_cycle(self):
        print("\n=== Cycle 2: Word Order Discrimination ===")
        pair = self.gen.generate_order_pairs("Generate a word-order pair with words: dog, cat, chase")
        s1, s2 = pair
        print(f"  Pair: {s1} | {s2}")
        hv1 = self.pipeline.encode_sentence(s1)
        hv2 = self.pipeline.encode_sentence(s2)
        sim = self.pipeline.vsa.similarity(hv1, hv2).item()
        self.add_result(f"{s1} vs {s2}", "order-discrimination", sim < 0.8,
                        f"sim={sim:.4f} (expected < 0.8 for different order)")

    def run_novelty_cycle(self):
        print("\n=== Cycle 3: Novelty Detection (Clonal) ===")
        known = ["the cat sat on mat", "dog bites man", "bird flies high"]
        for s in known:
            self.pipeline.store_sentence(s)
        pool = ClonalPool(input_dim=HD_DIM, affinity_threshold=0.4, max_modules=20)
        for s in known:
            pool.process(self.pipeline.encode_sentence(s))

        novel = self.gen.generate_sentence("Generate a sentence about a topic never mentioned before.")
        print(f"  Known: {known}")
        print(f"  Novel: {novel}")
        is_novel = self.pipeline.check_novelty(novel, pool)
        self.add_result(novel, "novelty-detection", is_novel,
                        f"novel={is_novel} (expected True for novel)")

    def run_ood_cycle(self):
        print("\n=== Cycle 4: OOD Detection (Immune) ===")
        known = ["the cat sat on mat", "dog bites man", "bird flies high"]
        for s in known:
            self.pipeline.store_sentence(s)
        hop = HopfieldNet(dim=HD_DIM)
        for s in known:
            hop.store(self.pipeline.encode_sentence(s))
        monitor = SelfMonitor(hop, energy_threshold=3.0)
        normal = [hop.recall(self.pipeline.encode_sentence(s), steps=10) for s in known]
        monitor.calibrate(normal)

        ood_input = self.gen.generate_ood_input(["cat", "dog", "bird", "mat"])
        print(f"  Normal: {'; '.join(known)}")
        print(f"  OOD:    {ood_input}")
        ood_score = self.pipeline.check_ood(ood_input, monitor)
        self.add_result(ood_input, "ood-detection", ood_score["is_anomaly"],
                        f"energy_z={ood_score['energy_z']:.2f} (expected anomalous)")

    def run_decoder_cycle(self):
        print("\n=== Cycle 5: Decoder Training Data Generation ===")
        prompt = "Generate 5 short sentences (3-4 words) for training a text decoder. "
        prompt += "Use format: sentence | sentence | sentence | sentence | sentence"
        r = ollama.chat(self.gen.model, messages=[
            {"role": "system", "content": "Generate exactly 5 short sentences separated by |"},
            {"role": "user", "content": prompt},
        ])
        sentences = [s.strip() for s in r["message"]["content"].split("|")][:5]
        print(f"  Generated {len(sentences)} sentences for decoder training data")

        words = set()
        for s in sentences:
            for w in s.lower().split():
                words.add(w.strip(".,!?;:'\""))
        self.vocab_set.update(words)
        self.add_result("|".join(sentences), "decoder-data-gen", len(sentences) >= 3,
                        f"sentences={len(sentences)}, unique_words={len(words)}")

    def run_full_pipeline_cycle(self):
        print("\n=== Cycle 6: Full Pipeline Integration ===")
        sentence = self.gen.generate_sentence("Generate a short sentence about a fish.")
        print(f"  Input: {sentence}")

        from vsa import HopfieldNet as HN
        local_hop = HN(dim=HD_DIM)
        hv = self.pipeline.encode_sentence(sentence)
        local_hop.store(hv)
        recalled = local_hop.recall(hv, steps=20)
        sim = self.pipeline.vsa.similarity(hv, recalled).item()
        self.add_result(sentence, "full-pipeline", sim > 0.85, f"sim={sim:.4f}")

    def run_all_cycles(self, iterations: int = 1):
        for i in range(iterations):
            print(f"\n{'#'*60}")
            print(f"  LLM Test Harness — Iteration {i+1}/{iterations}")
            print(f"{'#'*60}")
            print(f"  Model: {self.gen.model}")
            print(f"  VSA dim: {HD_DIM}")

            self.run_encoder_cycle()
            self.run_order_cycle()
            self.run_novelty_cycle()
            self.run_ood_cycle()
            self.run_decoder_cycle()
            self.run_full_pipeline_cycle()

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        print(f"\n{'='*60}")
        print(f"  RESULTS: {passed}/{total} passed ({100*passed/total:.0f}%)")
        print(f"{'='*60}")
        for r in self.results:
            s = "PASS" if r.passed else "FAIL"
            print(f"  [{s}] {r.pipeline_stage}: {r.scenario[:60]}")
            print(f"        {r.details}")

        report_path = Path("checkpoints") / "llm_harness_report.json"
        report_path.write_text(json.dumps(
            [asdict(r) for r in self.results], indent=2))
        print(f"\n  Report saved to {report_path}")


if __name__ == "__main__":
    harness = CyclicHarness()
    harness.run_all_cycles(iterations=1)

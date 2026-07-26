"""Controlled VSA-to-text generation benchmark.

Trains a compact VSA-conditioned DiT to reconstruct fully masked sentences,
then evaluates memorised sentences, held-out compositions, minimal order
pairs, and robustness to conditioning-vector noise.

Example:
    python -m experiments.generation_benchmark --epochs 200
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import torchhd

from src.decoder import VSAConditionedDiT


SUBJECTS = ("dog", "cat", "man", "woman")
VERBS = ("bites", "chases", "sees", "likes")
OBJECTS = ("mouse", "bird", "child", "fish")
TOKENS = SUBJECTS + VERBS + OBJECTS
TOKEN_TO_ID = {token: index for index, token in enumerate(TOKENS)}
MASK_ID = len(TOKENS)
PAD_ID = len(TOKENS) + 1
SEQUENCE_LENGTH = 3


@dataclass(frozen=True)
class GenerationConfig:
    vsa_dim: int = 256
    hidden_size: int = 64
    num_heads: int = 4
    num_layers: int = 2
    epochs: int = 200
    learning_rate: float = 1e-3
    batch_size: int = 16
    noise_levels: tuple[float, ...] = (0.0, 0.5, 1.0)
    seed: int = 42


@dataclass(frozen=True)
class GenerationMetrics:
    split: str
    noise: float
    examples: int
    exact_match: float
    token_accuracy: float
    mean_edit_distance: float
    order_accuracy: float
    latency_ms_per_example: float


class ControlledCorpus:
    def __init__(self, vsa_dim: int, seed: int):
        torch.manual_seed(seed)
        self.vsa_dim = vsa_dim
        self.word_vectors = torchhd.random(len(TOKENS), vsa_dim)
        self.position_vectors = torchhd.random(SEQUENCE_LENGTH, vsa_dim)

    def encode_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        words = self.word_vectors[token_ids]
        bound = torchhd.bind(words, self.position_vectors)
        return torchhd.multiset(bound).sign()

    def encode_batch(self, token_ids: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.encode_tokens(row) for row in token_ids])


def build_splits() -> tuple[torch.Tensor, torch.Tensor]:
    seen = []
    unseen = []
    for subject_index, subject in enumerate(SUBJECTS):
        for verb in VERBS:
            for object_index, object_ in enumerate(OBJECTS):
                ids = [
                    TOKEN_TO_ID[subject],
                    TOKEN_TO_ID[verb],
                    TOKEN_TO_ID[object_],
                ]
                # Every word occurs in training, but these subject/object
                # combinations are held out across every verb.
                target = unseen if subject_index == object_index else seen
                target.append(ids)
    return torch.tensor(seen), torch.tensor(unseen)


def levenshtein(left: list[int], right: list[int]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (left_item != right_item),
            ))
        previous = current
    return previous[-1]


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    split: str,
    noise: float,
    elapsed_seconds: float,
) -> GenerationMetrics:
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    examples = targets.shape[0]
    exact = (predictions == targets).all(dim=1).float().mean().item()
    token_accuracy = (predictions == targets).float().mean().item()
    edits = [
        levenshtein(pred.tolist(), target.tolist())
        for pred, target in zip(predictions, targets)
    ]
    # Order requires the subject and object to occupy their correct positions;
    # this specifically rejects reversed minimal pairs.
    order = (
        (predictions[:, 0] == targets[:, 0])
        & (predictions[:, 2] == targets[:, 2])
    ).float().mean().item()
    return GenerationMetrics(
        split=split,
        noise=noise,
        examples=examples,
        exact_match=exact,
        token_accuracy=token_accuracy,
        mean_edit_distance=sum(edits) / examples,
        order_accuracy=order,
        latency_ms_per_example=elapsed_seconds * 1000 / examples,
    )


def train_model(
    config: GenerationConfig,
    corpus: ControlledCorpus,
    train_tokens: torch.Tensor,
    device: str,
) -> tuple[VSAConditionedDiT, list[float]]:
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    model = VSAConditionedDiT(
        vocab_size=len(TOKENS),
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        cond_dim=config.hidden_size,
        vsa_dim=config.vsa_dim,
        max_seq_len=SEQUENCE_LENGTH,
        mask_token_id=MASK_ID,
    ).to(device)
    vectors = corpus.encode_batch(train_tokens).to(device)
    targets = train_tokens.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    losses = []

    for _ in range(config.epochs):
        permutation = torch.randperm(len(targets), generator=generator)
        epoch_loss = 0.0
        batches = 0
        model.train()
        for start in range(0, len(targets), config.batch_size):
            indices = permutation[start:start + config.batch_size].to(device)
            batch_targets = targets[indices]
            batch_vectors = vectors[indices]
            masked = torch.full_like(batch_targets, MASK_ID)
            logits = model(masked, batch_vectors)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                batch_targets.reshape(-1),
            )
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += loss.item()
            batches += 1
        losses.append(epoch_loss / batches)
    return model, losses


@torch.no_grad()
def evaluate_model(
    model: VSAConditionedDiT,
    corpus: ControlledCorpus,
    tokens: torch.Tensor,
    split: str,
    noise_levels: tuple[float, ...],
    device: str,
    seed: int,
) -> list[GenerationMetrics]:
    vectors = corpus.encode_batch(tokens).to(device)
    targets = tokens.to(device)
    masked = torch.full_like(targets, MASK_ID)
    model.eval()
    results = []
    for noise_index, noise in enumerate(noise_levels):
        generator = torch.Generator(device=device).manual_seed(seed + noise_index)
        noisy_vectors = vectors + noise * torch.randn(
            vectors.shape, generator=generator, device=device
        )
        started = time.perf_counter()
        predictions = model(masked, noisy_vectors).argmax(dim=-1)
        elapsed = time.perf_counter() - started
        results.append(compute_metrics(
            predictions.cpu(), tokens, split, noise, elapsed
        ))
    return results


def run_benchmark(
    config: GenerationConfig,
    device: str = "cpu",
) -> tuple[list[GenerationMetrics], list[float], VSAConditionedDiT]:
    corpus = ControlledCorpus(config.vsa_dim, config.seed)
    seen, unseen = build_splits()
    model, losses = train_model(config, corpus, seen, device)
    metrics = evaluate_model(
        model, corpus, seen, "seen", config.noise_levels, device, config.seed
    )
    metrics.extend(evaluate_model(
        model, corpus, unseen, "unseen_composition",
        config.noise_levels, device, config.seed + 10_000,
    ))
    return metrics, losses, model


def save_report(
    config: GenerationConfig,
    metrics: list[GenerationMetrics],
    losses: list[float],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"generation_benchmark_{stamp}.json"
    path.write_text(json.dumps({
        "benchmark": "controlled_vsa_text_generation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "training": {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
        },
        "metrics": [asdict(metric) for metric in metrics],
    }, indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--vsa-dim", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--noise-levels", default="0,0.5,1.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    config = GenerationConfig(
        vsa_dim=args.vsa_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        noise_levels=tuple(float(x) for x in args.noise_levels.split(",")),
        seed=args.seed,
    )
    metrics, losses, _ = run_benchmark(config, args.device)
    report = save_report(config, metrics, losses, args.output_dir)
    print(f"Training loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    for metric in metrics:
        print(
            f"{metric.split:18s} noise={metric.noise:g} "
            f"exact={metric.exact_match:.3f} token={metric.token_accuracy:.3f} "
            f"edit={metric.mean_edit_distance:.3f} order={metric.order_accuracy:.3f}"
        )
    print(f"Report: {report}")


if __name__ == "__main__":
    main()

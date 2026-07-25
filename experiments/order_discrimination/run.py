import itertools
import random
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torchhd

VOCAB = [
    "dog", "cat", "mouse", "man", "woman", "bird", "fish",
    "chases", "bites", "sees", "loves", "hates",
    "the", "a", "big", "small", "fast", "slow",
]

TEMPLATES = [
    "{subj} {verb} {obj}",
    "{subj} {verb} {adj} {obj}",
    "{adj} {subj} {verb} {obj}",
    "{subj} {verb} {obj} {prep} {prep_obj}",
]

SUBJECTS = ["dog", "cat", "mouse", "man", "woman", "bird", "fish"]
VERBS = ["chases", "bites", "sees", "loves", "hates"]
OBJECTS = ["dog", "cat", "mouse", "man", "woman", "bird", "fish", "the cat", "the dog", "the mouse"]
ADJECTIVES = ["big", "small", "fast", "slow"]
PREPOSITIONS = ["with", "near", "behind"]
PREP_OBJECTS = ["the dog", "the cat", "the mouse", "the man", "the woman", "a fish", "a bird"]

SEED = 42


def tokenize(sentence: str, vocab: list[str]) -> list[int]:
    words = sentence.split()
    return [vocab.index(w) for w in words if w in vocab]


def generate_sentences(num: int = 2000, seed: int = SEED) -> list[str]:
    rng = random.Random(seed)
    sentences: list[str] = []
    while len(sentences) < num:
        subj = rng.choice(SUBJECTS)
        verb = rng.choice(VERBS)
        obj = rng.choice(OBJECTS)
        adj = rng.choice(ADJECTIVES)

        variants = [
            f"{subj} {verb} {obj}",
            f"{adj} {subj} {verb} {obj}",
        ]

        for s in variants:
            if len(sentences) >= num:
                break
            if all(w in VOCAB for w in s.split()):
                sentences.append(s)

    return sentences


def generate_order_pairs(seed: int = SEED) -> list[tuple[str, str]]:
    pairs = []
    for subj in SUBJECTS:
        for verb in ["loves", "hates"]:
            for obj in OBJECTS:
                if subj == obj:
                    continue
                a = f"{subj} {verb} {obj}"
                b = f"{obj} {verb} {subj}"
                if all(w in VOCAB for w in a.split()) and all(w in VOCAB for w in b.split()):
                    pairs.append((a, b))
    return pairs[:100]


class SentenceEncoder:
    def __init__(self, vocab_size: int, hd_dim: int = 10000, device: str = "cpu"):
        self.hd_dim = hd_dim
        self.device = device
        self.vocab_size = vocab_size
        self.word_vectors = torchhd.random(vocab_size, hd_dim, device=device)
        self.position_vectors = torchhd.random(100, hd_dim, device=device)

    def encode(self, token_ids: list[int]) -> torch.Tensor:
        n = len(token_ids)
        word_hvs = self.word_vectors[token_ids]
        pos_hvs = self.position_vectors[:n]
        bound = torchhd.bind(word_hvs, pos_hvs)
        sentence_hv = torchhd.multiset(bound)
        return sentence_hv


class SmallLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 6, n_heads: int = 4,
                 max_len: int = 12, hd_dim: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size + 2, d_model, padding_idx=vocab_size + 1)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.vsa_proj = nn.Linear(hd_dim, d_model)
        self.mask_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size + 2)
        self.pad_id = vocab_size + 1

    def forward(self, token_ids: torch.Tensor, vsa_vector: torch.Tensor,
                mask_ratio: float = 0.15):
        bsz, seq_len = token_ids.shape
        emb = self.token_embed(token_ids) + self.pos_embed[:, :seq_len, :]

        vsa_cond = self.vsa_proj(vsa_vector).unsqueeze(1)
        emb = emb + vsa_cond

        if self.training and mask_ratio > 0:
            masked = token_ids.clone()
            labels = token_ids.clone()
            mask = torch.rand(bsz, seq_len, device=token_ids.device) < mask_ratio
            pad_mask = token_ids == self.pad_id
            mask = mask & ~pad_mask
            masked[mask] = vocab_size
            labels[~mask] = -100
            emb[mask] = self.mask_embed.unsqueeze(0)
            src_key_padding_mask = token_ids == self.pad_id
        else:
            masked = token_ids
            labels = None
            src_key_padding_mask = token_ids == self.pad_id

        out = self.transformer(emb, src_key_padding_mask=src_key_padding_mask)
        logits = self.lm_head(out)
        return logits, masked, labels

    def generate(self, vsa_vector: torch.Tensor, max_len: int = 12,
                 num_steps: int = 50, temp: float = 1.0,
                 mask_token_id: int | None = None) -> torch.Tensor:
        self.eval()
        bsz = vsa_vector.shape[0]
        mask_id = mask_token_id if mask_token_id is not None else self.token_embed.num_embeddings - 1
        tokens = torch.full((bsz, max_len), mask_id, device=vsa_vector.device)

        with torch.no_grad():
            for step in range(num_steps):
                emb = self.token_embed(tokens) + self.pos_embed[:, :max_len, :]
                vsa_cond = self.vsa_proj(vsa_vector).unsqueeze(1)
                emb = emb + vsa_cond
                out = self.transformer(emb)
                logits = self.lm_head(out) / temp
                probs = F.softmax(logits, dim=-1)
                preds = probs.argmax(dim=-1)

                confidence = probs.max(dim=-1).values
                frac = 0.3 * (1 - step / num_steps)
                n_mask = max(1, int(max_len * frac))
                _, idxs = confidence.sort(dim=1)
                to_mask = idxs[:, :n_mask]

                tokens = preds.clone()
                tokens.scatter_(1, to_mask, mask_id)
                if step == num_steps - 1:
                    tokens = preds

        return tokens


class SentenceDataset(Dataset):
    def __init__(self, sentences: list[str], encoder: SentenceEncoder, vocab: list[str],
                 max_len: int = 12):
        self.encoded = []
        self.token_ids = []
        self.max_len = max_len
        self.vocab_size = len(vocab)
        for s in sentences:
            ids = tokenize(s, vocab)
            if 3 <= len(ids) <= max_len:
                hv = encoder.encode(ids)
                pad_id = len(vocab) + 1
                padded = ids + [pad_id] * (max_len - len(ids))
                self.encoded.append(hv)
                self.token_ids.append(torch.tensor(padded[:max_len], dtype=torch.long))

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx], self.token_ids[idx]


def train(model, loader, optimizer, epochs: int = 100, device: str = "cpu"):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        steps = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for vsa_vec, token_ids in loop:
            vsa_vec = vsa_vec.to(device)
            token_ids = token_ids.to(device)
            logits, _, labels = model(token_ids, vsa_vec, mask_ratio=0.15)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                                   ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            loop.set_postfix(loss=loss.item())
        avg = total_loss / steps
        print(f"Epoch {epoch+1}: avg loss = {avg:.4f}")
        if avg < 0.1:
            print("Loss threshold reached, stopping early")
            break


@torch.no_grad()
def evaluate_exact_match(model, dataset, vocab_size: int, max_len: int, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    pad_id = vocab_size + 1
    mask_id = vocab_size
    for vsa_vec, token_ids in dataset:
        vsa_vec = vsa_vec.unsqueeze(0).to(device)
        preds = model.generate(vsa_vec, max_len=max_len, num_steps=50, mask_token_id=mask_id)
        target = token_ids[:token_ids.ne(pad_id).sum()].to(device)
        pred = preds[0, :len(target)]
        if torch.equal(pred, target):
            correct += 1
        total += 1
    return correct / total


@torch.no_grad()
def evaluate_order_discrimination(model, pairs: list[tuple[str, str]],
                                  encoder: SentenceEncoder, vocab: list[str],
                                  max_len: int, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    for s1, s2 in pairs:
        ids1 = tokenize(s1, vocab)
        ids2 = tokenize(s2, vocab)
        hv1 = encoder.encode(ids1).unsqueeze(0).to(device)
        hv2 = encoder.encode(ids2).unsqueeze(0).to(device)
        pred1 = model.generate(hv1, max_len=max_len, num_steps=50)[0]
        pred2 = model.generate(hv2, max_len=max_len, num_steps=50)[0]
        target1 = torch.tensor(ids1, device=device)
        target2 = torch.tensor(ids2, device=device)
        c1 = torch.equal(pred1[:len(ids1)], target1)
        c2 = torch.equal(pred2[:len(ids2)], target2)
        if c1 and c2:
            correct += 1
        total += 1
    return correct / total


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    hd_dim = 10000
    d_model = 256
    n_layers = 6
    n_heads = 4
    vocab_size = len(VOCAB)
    max_len = 12
    batch_size = 32
    epochs = 50

    print("Generating sentences...")
    sentences = generate_sentences(2000)
    order_pairs = generate_order_pairs()
    print(f"  {len(sentences)} total sentences")
    print(f"  {len(order_pairs)} order swap pairs")

    print("Initializing encoder...")
    encoder = SentenceEncoder(vocab_size, hd_dim=hd_dim, device="cpu")

    dataset = SentenceDataset(sentences, encoder, VOCAB, max_len=max_len)
    split = int(0.75 * len(dataset))
    train_ds, test_ds = torch.utils.data.random_split(
        dataset, [split, len(dataset) - split],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = SmallLM(vocab_size, d_model=d_model, n_layers=n_layers,
                    n_heads=n_heads, max_len=max_len, hd_dim=hd_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    print(f"Training {sum(p.numel() for p in model.parameters()):,} params...")
    train(model, train_loader, optimizer, epochs=epochs, device=device)

    print("\nEvaluating exact match on test set...")
    test_items = [test_ds[i] for i in range(min(100, len(test_ds)))]
    em = evaluate_exact_match(model, test_items, vocab_size, max_len, device=device)
    print(f"  Exact match accuracy: {em:.3f}")

    print("\nEvaluating order discrimination...")
    if len(order_pairs) > 0:
        od = evaluate_order_discrimination(model, order_pairs[:50], encoder,
                                            VOCAB, max_len, device=device)
        print(f"  Both correct: {od:.3f}")
    else:
        print("  No order pairs generated")

    print("\nDone.")

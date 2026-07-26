"""
DIFFUSION DECODER TRAINING — ARCHIVED

This script trains the VSAConditionedDiT (diffusion-based learned decoder).
The diffusion approach achieved near-zero training loss but only ~6% exact-match
accuracy (see checkpoints/decoder-training-autopsy.md).

Use LookupDecoder (src/decoder/lookup_decoder.py) for production text retrieval:
it uses HopfieldNet + AssociativeStore for deterministic exact-match retrieval
with no training required.

Kept for reproducibility of the diffusion decoder experiment.
"""

import itertools
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchhd
from tqdm import tqdm

sys.path.insert(0, "src")
from vsa import VSA
from decoder import VSAConditionedDiT, DiscreteDiffusion, iterative_unmasking_sample

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HD_DIM = 10000
MAX_SEQ_LEN = 12
BATCH_SIZE = 64
N_EPOCHS = 200
LR = 2e-4
HIDDEN_SIZE = 256
N_LAYERS = 6
N_HEADS = 4
OUT_DIR = Path("checkpoints")
OUT_DIR.mkdir(exist_ok=True)

WORDS = [
    "the", "a", "cat", "dog", "bird", "fish", "mouse", "man", "woman", "child",
    "big", "small", "fast", "slow", "red", "blue", "green", "old", "new", "young",
    "runs", "walks", "jumps", "flies", "swims", "sits", "sleeps", "eats", "drinks",
    "loves", "hates", "sees", "hears", "holds", "throws", "catches", "chases",
    "ball", "stick", "stone", "leaf", "tree", "flower", "house", "car", "boat", "plane",
    "on", "in", "under", "near", "with", "at", "to", "from", "over", "through",
    "and", "or", "but", "very", "quite", "too", "also", "always", "never", "often",
    "happy", "sad", "angry", "calm", "brave", "shy", "clever", "kind", "proud", "gentle",
    "today", "yesterday", "morning", "evening", "here", "there", "up", "down",
    "apple", "bread", "milk", "water", "table", "chair", "door", "window",
    "one", "two", "three", "four", "five", "some", "many", "every",
    "I", "you", "he", "she", "it", "we", "they",
    "ground", "behind",
]

VOCAB_SIZE = len(WORDS)


TEMPLATES = [
    "{subj} {verb} {obj}",
    "{subj} {verb} the {obj}",
    "{subj} {verb} {adj} {obj}",
    "{subj} {verb} {prep} {obj2}",
    "{adj} {subj} {verb} {obj}",
    "{subj} {verb} {obj} {prep} {obj2}",
    "{subj} {verb} {adj} {obj} {prep} {obj2}",
    "{subj} {verb} {det} {obj}",
    "{det} {adj} {subj} {verb} {obj}",
    "{subj} {verb} {det} {adj} {obj}",
]

SUBJECTS = ["the cat", "the dog", "the bird", "the fish", "the man", "the woman",
            "the child", "a cat", "a dog", "a bird", "a fish", "the mouse",
            "I", "you", "he", "she", "they"]
VERBS = ["runs", "walks", "jumps", "flies", "swims", "sits", "sleeps", "eats",
         "loves", "hates", "sees", "holds", "throws", "catches", "chases"]
OBJECTS = ["a ball", "a stick", "a stone", "a leaf", "a tree", "a flower",
           "the ball", "the stick", "the stone", "the tree",
           "a cat", "a dog", "a bird", "a fish", "the mouse", "a mouse"]
ADJECTIVES = ["big", "small", "fast", "slow", "red", "blue", "green",
              "old", "new", "young", "happy", "sad", "clever", "kind"]
PREPOSITIONS = ["on", "in", "under", "near", "with", "behind"]
PREP_OBJECTS = ["a table", "the table", "a chair", "the chair", "a door",
                "the door", "a house", "the house", "a car", "the car",
                "the ground", "the water", "a tree", "the tree"]
DETERMINERS = ["a", "the", "some", "many", "every"]
ADVERBS = ["very", "quite", "too", "always", "never", "often", "also"]


def tokenize(sentence: str) -> list[int]:
    words = sentence.lower().strip(".,!?;:'\"").split()
    ids = []
    for w in words:
        if w in WORDS:
            ids.append(WORDS.index(w))
    return ids


def generate_dataset(num_train: int = 8000, num_test: int = 2000):
    rng = random.Random(SEED)
    sentences = []
    while len(sentences) < num_train + num_test:
        subj = rng.choice(SUBJECTS)
        verb = rng.choice(VERBS)
        obj = rng.choice(OBJECTS)
        adj = rng.choice(ADJECTIVES)
        prep = rng.choice(PREPOSITIONS)
        prep_obj = rng.choice(PREP_OBJECTS)
        det = rng.choice(DETERMINERS)
        temp = rng.choice(TEMPLATES)
        sent = temp.format(subj=subj, verb=verb, obj=obj, adj=adj,
                           prep=prep, obj2=prep_obj, det=det)
        ids = tokenize(sent)
        if 3 <= len(ids) <= MAX_SEQ_LEN:
            sentences.append(sent)

    return sentences[:num_train], sentences[num_train:num_train + num_test]


def encode_dataset(sentences, vsa, pos_vectors, word_hvs):
    data = []
    for sent in tqdm(sentences, desc="Encoding"):
        ids = tokenize(sent)
        if not ids or len(ids) > MAX_SEQ_LEN:
            continue
        word_vecs = word_hvs[ids]
        pos = pos_vectors[:len(ids)]
        bound = torchhd.bind(word_vecs, pos)
        bundle = torchhd.multiset(bound).sign()
        padded = ids + [VOCAB_SIZE + 1] * (MAX_SEQ_LEN - len(ids))
        data.append((bundle.cpu(), torch.tensor(padded[:MAX_SEQ_LEN], dtype=torch.long)))
    return data


def collate_batch(batch):
    vsas, toks = zip(*batch)
    return torch.stack(vsas), torch.stack(toks)


@torch.no_grad()
def evaluate_accuracy(model, data, max_len: int, num_steps: int = 50,
                      hopfield=None):
    model.eval()
    correct = 0
    total = 0
    mask_id = VOCAB_SIZE
    for vsa_v, tok_ids in data:
        vsa_v = vsa_v.to(DEVICE)
        if hopfield is not None:
            vsa_v = hopfield.recall(vsa_v, steps=10)
        vsa_v = vsa_v.unsqueeze(0)
        preds = iterative_unmasking_sample(
            model, vsa_v, mask_token_id=mask_id,
            max_len=max_len, num_steps=num_steps, temp=0.5)
        target = tok_ids[:tok_ids.ne(VOCAB_SIZE + 1).sum()].to(DEVICE)
        pred = preds[0, :len(target)]
        if torch.equal(pred, target):
            correct += 1
        total += 1
    return correct / total


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Generating dataset...")
    train_sents, test_sents = generate_dataset(8000, 2000)
    print(f"  Train: {len(train_sents)} sentences")
    print(f"  Test:  {len(test_sents)} sentences")

    print("Initialising encoder...")
    vsa = VSA(dim=HD_DIM, device="cpu")
    pos_vectors = torchhd.random(MAX_SEQ_LEN, HD_DIM, device="cpu")
    word_hvs = torchhd.random(VOCAB_SIZE, HD_DIM, device="cpu")

    print("Encoding training data...")
    train_data = encode_dataset(train_sents, vsa, pos_vectors, word_hvs)
    test_data = encode_dataset(test_sents, vsa, pos_vectors, word_hvs)
    print(f"  Encoded: {len(train_data)} train, {len(test_data)} test")

    print("Initialising decoder...")
    model = VSAConditionedDiT(
        vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE,
        num_heads=N_HEADS, num_layers=N_LAYERS,
        cond_dim=HIDDEN_SIZE, vsa_dim=HD_DIM,
        max_seq_len=MAX_SEQ_LEN,
    ).to(DEVICE)
    params = sum(x.numel() for x in model.parameters())
    print(f"  Parameters: {params:,} ({params/1e6:.1f}M)")

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=N_EPOCHS)
    diffusion = DiscreteDiffusion(mask_token_id=VOCAB_SIZE, num_steps=1000)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_batch)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, collate_fn=collate_batch)

    best_acc = 0.0
    print(f"\nTraining {N_EPOCHS} epochs...")
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_loss = 0
        steps = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{N_EPOCHS}", leave=False)
        for vsa_v, tok_ids in loop:
            vsa_v, tok_ids = vsa_v.to(DEVICE), tok_ids.to(DEVICE)
            t = torch.randint(0, 1000, (vsa_v.shape[0],), device=DEVICE)
            corrupted = diffusion.corrupt(tok_ids, t)
            logits = model(corrupted, vsa_v)
            mask = (corrupted == VOCAB_SIZE) & (tok_ids != VOCAB_SIZE + 1)
            labels = tok_ids.clone()
            labels[~mask] = -100
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()
            steps += 1
            loop.set_postfix(loss=loss.item())
        scheduler.step()
        avg_loss = total_loss / steps

        if epoch % 5 == 0 or epoch == 1:
            acc = evaluate_accuracy(
                model, test_data[:200], MAX_SEQ_LEN, num_steps=50)
            print(f"  Epoch {epoch}: loss={avg_loss:.4f}, test_acc={acc:.3f}")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), OUT_DIR / "decoder_best.pt")
                print(f"  -> Saved best model (acc={acc:.3f})")
        if epoch % 50 == 0:
            torch.save(model.state_dict(), OUT_DIR / f"decoder_ep{epoch}.pt")

    print(f"\nBest test accuracy: {best_acc:.3f}")
    print(f"Saved to {OUT_DIR / 'decoder_best.pt'}")

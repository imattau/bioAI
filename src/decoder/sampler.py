import torch
import torch.nn.functional as F


@torch.no_grad()
def iterative_unmasking_sample(model, vsa_vector: torch.Tensor,
                                mask_token_id: int, max_len: int,
                                num_steps: int = 100, temp: float = 1.0,
                                re_mask_frac: float = 0.3) -> torch.Tensor:
    model.eval()
    bsz = vsa_vector.shape[0]
    tokens = torch.full((bsz, max_len), mask_token_id, device=vsa_vector.device)

    for step in range(num_steps):
        logits = model(tokens, vsa_vector)
        logits = logits / temp
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        if step < num_steps - 1:
            confidence = probs.max(dim=-1).values
            frac = re_mask_frac * (1 - step / num_steps)
            n_mask = max(1, int(max_len * frac))
            _, idxs = confidence.sort(dim=1)
            to_mask = idxs[:, :n_mask]
            tokens = preds.clone()
            tokens.scatter_(1, to_mask, mask_token_id)
        else:
            tokens = preds

    return tokens

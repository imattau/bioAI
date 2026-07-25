import torch
import torch.nn as nn


class ThreeFactorHebbian(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights

    def update(self, pre: torch.Tensor, post: torch.Tensor,
               reward_pe: torch.Tensor, lr: float = 0.001):
        delta = lr * reward_pe * (pre.unsqueeze(1) @ post.unsqueeze(0))
        self.weights.data += delta


class GoNoGoActorCritic(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.go = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.nogo = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        go_logits = self.go(state)
        nogo_logits = self.nogo(state)
        action_logits = go_logits - nogo_logits
        value = self.critic(state)
        return action_logits, value, go_logits, nogo_logits

    def act(self, state: torch.Tensor, deterministic: bool = False) -> tuple[int, torch.Tensor]:
        logits, value, _, _ = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        if deterministic:
            action = probs.argmax().item()
        else:
            action = torch.multinomial(probs, 1).item()
        return action, value

    def compute_loss(self, state: torch.Tensor, action: int,
                     reward: float, next_state: torch.Tensor | None,
                     gamma: float = 0.99) -> torch.Tensor:
        logits, value, go_logits, nogo_logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        log_prob = torch.log(probs[action] + 1e-8)

        if next_state is not None:
            _, next_value = self.act(next_state)
            target = reward + gamma * next_value
        else:
            target = reward

        advantage = target - value.squeeze(0)
        actor_loss = -log_prob * advantage.detach()
        critic_loss = advantage.pow(2)
        return (actor_loss + critic_loss).mean()

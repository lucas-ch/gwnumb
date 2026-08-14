from torch import nn
import torch

from numb_project.constants import BASE
import torch.nn.functional as F

class UnitOperationModule(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()

        self.transfo = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.transfo(x)

class SequentialOperationModule(nn.Module):
    def __init__(self, input_size, context_size, output_size=2, hidden_size=32, temperature=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_cell = nn.LSTMCell(input_size=input_size + context_size + 1, hidden_size=hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.temperature = temperature
        self.hard = True

    def init_cell(self, batch_size: int, device: str):
        h0 = torch.zeros(batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(batch_size, self.hidden_size, device=device)
        return (h0, c0)

    def forward(self, x: torch.Tensor, hc: tuple[torch.Tensor, torch.Tensor],
                step_embedding: torch.Tensor | None = None):
        if step_embedding is not None:
            x = torch.cat([x, step_embedding], dim=-1)

        h, c = self.memory_cell(x, hc)
        logits = self.output(h)  # (batch, 2) : [ne_pas_roll, roll]

        choice = torch.softmax(logits / self.temperature, dim=-1)

        a_roll = choice[:, 1:2]  # (batch, 1)

        return a_roll, (h, c)

def make_chain_task(
    digit_one_hot: torch.Tensor, base: int = BASE, start_chain = 0, end_chain = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = digit_one_hot.shape[0]
    device = digit_one_hot.device

    digit_idx = digit_one_hot.argmax(dim=1)
    right_addend_value = torch.randint(start_chain, end_chain, (batch_size,), device=device)

    target_idx = (digit_idx + right_addend_value) % base

    right_addend_onehot = F.one_hot(right_addend_value, num_classes=base).float()
    target_one_hot = F.one_hot(target_idx, num_classes=base).float()

    return right_addend_onehot, target_one_hot


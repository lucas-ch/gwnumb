
from shimmer import GWModuleBase, LatentsDomainGroupsT, LossOutput, RawDomainGroupsT
from torch import nn
import torch
import torch.nn.functional as F

from numb_project.data_module import rotate_item


class UnitaryOperationModule(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()

        self.transfo = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transfo(x)

    def loss(self, gw_mod: GWModuleBase,
            latent_domains: LatentsDomainGroupsT,) -> torch.Tensor:
        pass

class ChainOperationModule(nn.Module):
    def __init__(
        self,
        chain_length: int,
        start_task: int,
        end_task: int,
        base: int,
        gw_mod: GWModuleBase,
        operation_selection_module: "OperationSelectionModule",
        operation_module: nn.ModuleDict,
    ):
        super().__init__()
        self.chain_length = chain_length
        self.start_task = start_task
        self.end_task = end_task
        self.base = base
        self.gw_mod = gw_mod
        self.operation_selection_module =  operation_selection_module
        self.operation_module = operation_module

    def forward_chain(
        self,
        gw_state: torch.Tensor,
        task: torch.Tensor,
        digit_one_hot: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = gw_state.shape[0]
        self.operation_selection_module.reset_state(batch_size, gw_state.device)

        outputs = []
        roll_sequence = []
        for t in range(self.chain_length):
            operations_results = {
                f'{name}_to_gw': operation(gw_state)
                for name, operation in self.operation_module.items()
            }

            gw_state, roll = self.operation_selection_module.update_gw_state(gw_state, task, operations_results)
            digit_pred = self.gw_mod.gw_decoders["digit"](gw_state)

            if digit_one_hot is not None:
                # debug uniquement : suivi manuel (breakpoint/print) de la chaîne d'addition
                a_addl = torch.argmax(digit_one_hot, dim=1)
                a_addr = torch.argmax(task, dim=1)
                a_ground_truth = a_addl + a_addr
                a_gw_state = torch.argmax(digit_pred, dim=1)

            outputs.append(digit_pred)
            roll_sequence.append(roll)

        return torch.stack(outputs, dim=1), torch.stack(roll_sequence, dim=1)

    def create_chain_add_tasks(
        self, raw_group: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        digit_one_hot = raw_group["digit"]
        batch_size = digit_one_hot.shape[0]
        device = digit_one_hot.device

        digit_idx = digit_one_hot.argmax(dim=1)
        right_addend_value = torch.randint(self.start_task, self.end_task, (batch_size,), device=device)

        target_idx = (digit_idx + right_addend_value) % self.base

        right_addend_onehot = F.one_hot(right_addend_value, num_classes=self.base).float()
        target_one_hot = F.one_hot(target_idx, num_classes=self.base).float()

        return right_addend_onehot, right_addend_value, target_one_hot

    def sequence_loss(
        self, roll_sequence: torch.Tensor, task_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Supervise le choix roll/pas-roll à chaque pas.
        a_roll_sequence: (batch, chain_length, 1)
        right_addend_value: (batch,) -- nombre de +1 attendus
        Cible : rouler (1) tant que t < right_addend_value, sinon ne pas rouler (0).
        """
        batch_size, chain_length, num_operations = roll_sequence.shape
        device = roll_sequence.device

        t_range = torch.arange(chain_length, device=device).unsqueeze(0)
        is_adding = (t_range < task_targets.unsqueeze(1)).long()
        target = F.one_hot(is_adding, num_classes=num_operations).float()

        return F.binary_cross_entropy(roll_sequence, target)

    def final_prediction_loss(
        self, task_predicitions: torch.Tensor, task_targets: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(task_predicitions[:, -1, :], task_targets)

    def steps_loss(self, task_predicitions: torch.Tensor, task_targets: torch.Tensor) -> torch.Tensor:
        losses = torch.zeros(self.chain_length, device=task_predicitions.device)
        for i in range(self.chain_length):
            losses[i] = F.mse_loss(task_predicitions[:, i, :], task_targets)
        return torch.mean(losses[1:])

    def loss(
        self,
        task_predicitions: torch.Tensor,
        task_targets: torch.Tensor,
        roll_sequence: torch.Tensor,
        right_addend_value: torch.Tensor,
    ) -> LossOutput:
        metrics = {}
        sequence_loss = self.sequence_loss(roll_sequence, right_addend_value)
        final_prediction_loss = self.final_prediction_loss(task_predicitions, task_targets)

        metrics["sequence_loss"] = sequence_loss
        metrics["final_prediction_loss"] = final_prediction_loss

        total_loss = sequence_loss + final_prediction_loss

        return LossOutput(total_loss, metrics)

class OperationSelectionModule(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        batch_size: int,
        device: torch.device | str,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.temperature = temperature

        self.reset_state(batch_size, device)

    def reset_state(self, batch_size: int, device: torch.device | str) -> None:
        h0 = torch.zeros(batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(batch_size, self.hidden_size, device=device)
        self.hc = (h0, c0)

    def forward(
        self, x: torch.Tensor, hc: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h, c = self.memory_cell(x, hc)
        logits = self.output(h)

        output = torch.softmax(logits / self.temperature, dim=-1)

        return output, (h, c)

    def update_gw_state(
        self,
        gw_state: torch.Tensor,
        task: torch.Tensor,
        operations_results: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        operation_selection_vector, self.hc = self(task, self.hc)
        assert operation_selection_vector.shape[1] == 1 + len(operations_results), (
            "operation_selection_vector must carry one weight per operation plus 'stale'"
        )

        operation_selection_stale = operation_selection_vector[:, 0].unsqueeze(1)
        new_gw_state = operation_selection_stale * gw_state

        for i, operation_result in enumerate(operations_results.values(), start=1):
            operation_selection_weight = operation_selection_vector[:, i].unsqueeze(1)
            new_gw_state = new_gw_state + operation_selection_weight * operation_result

        return new_gw_state, operation_selection_vector


def get_input_target_shift_loss(
    gw_mod: GWModuleBase, latent_domains: LatentsDomainGroupsT, shift: int
) -> tuple[torch.Tensor, torch.Tensor]:
        digit_one_hot = latent_domains[frozenset({"digit", "image"})]['digit']
        target_one_hot = torch.roll(digit_one_hot, shifts=shift, dims=1)
        image = latent_domains[frozenset({"digit", "image"})]['image']

        with torch.no_grad():
            input = gw_mod.gw_encoders["image"](image)
            target = gw_mod.gw_encoders["digit"](target_one_hot)

        return input, target

def get_input_target_rotation_loss(
    gw_mod: GWModuleBase, raw_data: RawDomainGroupsT, degrees: float
) -> tuple[torch.Tensor, torch.Tensor]:
        # La rotation doit s'appliquer sur l'image brute (pixels) et non sur le
        # latent du domaine "image" (ex : mu d'un VAE), qui n'a pas de structure
        # spatiale.
        image = raw_data[frozenset({"digit", "image"})]['image']
        rotated_image = rotate_item({"image": image}, degrees)["image"]

        with torch.no_grad():
            image_latent = gw_mod.domain_mods["image"].encode(image)
            rotated_latent = gw_mod.domain_mods["image"].encode(rotated_image)
            input = gw_mod.gw_encoders["image"](image_latent)
            target = gw_mod.gw_encoders["image"](rotated_latent)

        return input, target

class RotateOperationModule(UnitaryOperationModule):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__(input_size, hidden_size, output_size)

    def loss(
        self,
        gw_mod: GWModuleBase,
        raw_data: RawDomainGroupsT,
    ) -> torch.Tensor:
        input, target = get_input_target_rotation_loss(gw_mod, raw_data, 10)
        pred = self(input)

        return F.mse_loss(pred, target)

class AddOperationModule(UnitaryOperationModule):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__(input_size, hidden_size, output_size)

    def loss(
        self,
        gw_mod: GWModuleBase,
        latent_domains: LatentsDomainGroupsT,
    ) -> torch.Tensor:
        input, target = get_input_target_shift_loss(gw_mod, latent_domains, 1)
        pred = self(input)

        return F.mse_loss(pred, target)

class SubOperationModule(UnitaryOperationModule):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__(input_size, hidden_size, output_size)

    def loss(
        self,
        gw_mod: GWModuleBase,
        latent_domains: LatentsDomainGroupsT,
    ) -> torch.Tensor:
        input, target = get_input_target_shift_loss(gw_mod, latent_domains, -1)
        pred = self(input)

        return F.mse_loss(pred, target)

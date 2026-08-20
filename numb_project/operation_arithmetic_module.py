from shimmer import GWModuleBase, LatentsDomainGroupsT
import torch
import torch.nn.functional as F

from numb_project.operation_module import UnitaryOperationModule

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


class AddOperationModule(UnitaryOperationModule):
    def loss(
        self,
        gw_mod: GWModuleBase,
        latent_domains: LatentsDomainGroupsT,
    ) -> torch.Tensor:
        input, target = get_input_target_shift_loss(gw_mod, latent_domains, 1)
        pred = self(input)

        return F.mse_loss(pred, target)

class SubOperationModule(UnitaryOperationModule):
    def loss(
        self,
        gw_mod: GWModuleBase,
        latent_domains: LatentsDomainGroupsT,
    ) -> torch.Tensor:
        input, target = get_input_target_shift_loss(gw_mod, latent_domains, -1)
        pred = self(input)

        return F.mse_loss(pred, target)

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
from torch import nn

from fairchem.core.common import distutils, gp_utils
from fairchem.core.common.registry import registry

from fairchem.core.common.logger import WandBSingletonLogger

class DDPMTLoss(nn.Module):
    """
    This class is a wrapper around a loss function that does a few things
    like handle nans and importantly ensures the reduction is done
    correctly for DDP. The main issue is that DDP averages gradients
    over replicas — this only works out of the box if the dimension
    you are averaging over is completely consistent across all replicas.
    In our case, that is not true for the number of atoms per batch and
    there are edge cases when the batch size differs between replicas
    e.g. if the dataset size is not divisible by the batch_size.

    Scalars are relatively straightforward to handle, but vectors and higher tensors
    are a bit trickier. Below are two examples of forces.

    Forces input: [Nx3] target: [Nx3]
    Forces are a vector of length 3 (x,y,z) for each atom.
    Number of atoms per batch (N) is different for each DDP replica.

    MSE example:
    #### Local loss computation ####
    local_loss = MSELoss(input, target) -> [Nx3]
    num_samples = local_loss.numel() -> [Nx3]
    local_loss = sum(local_loss [Nx3]) -> [1] sum reduces the loss to a scalar
    global_samples = all_reduce(num_samples) -> [N0x3 + N1x3 + N2x3 + ...] = [1] where N0 is the number of atoms on replica 0
    local_loss = local_loss * world_size / global_samples -> [1]
    #### Global loss computation ####
    global_loss = sum(local_loss / world_size) -> [1]
    == sum(local_loss / global_samples) # this is the desired corrected mean

    Norm example:
    #### Local loss computation ####
    local_loss = L2MAELoss(input, target) -> [N]
    num_samples = local_loss.numel() -> [N]
    local_loss = sum(local_loss [N]) -> [1] sum reduces the loss to a scalar
    global_samples = all_reduce(num_samples) -> [N0 + N1 + N2 + ...] = [1] where N0 is the number of atoms on replica 0
    local_loss = local_loss * world_size / global_samples -> [1]
    #### Global loss computation ####
    global_loss = sum(local_loss / world_size) -> [1]
    == sum(local_loss / global_samples) # this is the desired corrected mean
    """

    def __init__(
        self,
        loss_fn: torch.nn.Module,
        reduction: Literal["mean", "sum", "per_structure"] = "mean",
        coefficient: float = 1.0,
    ) -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.reduction = reduction
        self.reduction_map = {
            "mean": self.mean,
            "sum": self.sum,
            "per_structure": self.per_structure,
        }
        self.coefficient = coefficient
        assert self.reduction in list(
            self.reduction_map.keys()
        ), "Reduction must be one of: 'mean', 'sum', 'per_structure'"

    def sum(self, input, mult_mask, num_samples, loss, natoms):
        # this sum will reduce the loss down to a single scalar
        return torch.sum(loss)

    def _ddp_mean(self, num_samples, loss):
        # global_samples can be 0 if the head has no valid samples in the batch
        # protect against division by zero
        global_samples = max(distutils.all_reduce(num_samples, device=loss.device), 1)
        # Multiply by world size since gradients are averaged across DDP replicas
        # warning this is probably incorrect for any model parallel approach
        # Graph parallel note: numerator and denominator are inflated by the same
        # constant. # of processes in a single graph parallel group , which makes this
        # a strange way to implement the loss, but technically correct
        # however the gradient is not correct , please see comments at FixGPGrad()
        corrected_loss = loss * distutils.get_world_size() / global_samples

        if gp_utils.initialized():
            # make this explict so its easier to reason about loss here
            # calling fix_gp_grad in non-gp has no affect
            return gp_utils.scale_backward_grad(corrected_loss)
        return corrected_loss

    def mean(self, input, mult_mask, num_samples, loss, natoms):
        # this sum will reduce the loss down from num_sample -> 1
        loss = self.sum(input, mult_mask, num_samples, loss, natoms)
        return self._ddp_mean(num_samples, loss)

    def per_structure(self, input, mult_mask, num_samples, loss, natoms):
        struct_idx = torch.repeat_interleave(
            torch.arange(natoms.numel(), device=input.device), natoms
        )
        assert torch.unique(struct_idx).numel() == natoms.numel()
        per_struct_loss = torch.zeros(
            natoms.numel(), device=input.device
        ).scatter_reduce(0, struct_idx, loss, reduce="sum")

        # normalize by the number of free atoms in the structure
        free_natoms = torch.bincount(struct_idx[mult_mask], minlength=natoms.numel())
        zero_idx = torch.where(free_natoms == 0)[0]
        free_natoms[zero_idx] = natoms[zero_idx]
        assert torch.all(free_natoms > 0)
        assert torch.all(free_natoms <= natoms)
        per_struct_loss = per_struct_loss / free_natoms

        # takes the mean across all systems in the batch
        num_samples = torch.nonzero(per_struct_loss).numel()
        return self._ddp_mean(num_samples, per_struct_loss.sum())

    def _reduction(self, input, mult_mask, loss, natoms):
        num_samples = loss[mult_mask].numel()
        if self.reduction in self.reduction_map:
            return self.reduction_map[self.reduction](
                input, mult_mask, num_samples, loss, natoms
            )
        else:
            raise ValueError("Reduction must be one of: 'mean', 'sum'")

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mult_mask: torch.Tensor,
        natoms: torch.Tensor,
        step,
        batch
    ):
        # ensure torch doesn't do any unwanted broadcasting
        assert (
            input.shape[0] == target.shape[0] == mult_mask.shape[0]
        ), f"Mismatched shapes: {input.shape} and {target.shape} and {mult_mask.shape}"
        if hasattr(self.loss_fn, 'ignore_shape_check'):
            assert (
                input.shape[0] == target.shape[0] == mult_mask.shape[0]
            ), f"Mismatched shapes: {input.shape} and {target.shape} and {mult_mask.shape}"
            if input.numel() == mult_mask.numel():
                mult_mask = mult_mask.view(input.shape)
            loss = (
                self.loss_fn(
                    input.squeeze(), torch.nan_to_num(target, posinf=0.0, neginf=0.0), natoms, step, batch
                )
                * mult_mask
                )
            loss = self._reduction(input, mult_mask, loss, natoms)
        else:
        # Ensure torch doesn't do any unwanted broadcasting
            target = target.view(input.shape)
            if input.numel() == mult_mask.numel():
                mult_mask = mult_mask.view(input.shape)

            loss = (
                self.loss_fn(
                    input, torch.nan_to_num(target, posinf=0.0, neginf=0.0), natoms, step, batch
                )
                * mult_mask
            )
            loss = self._reduction(input, mult_mask, loss, natoms)

        # Zero out nans, if any
        found_nans_or_infs = not torch.all(loss.isfinite())
        if found_nans_or_infs is True:
            logging.warning("Found nans while computing loss")
            loss = torch.nan_to_num(loss, nan=0.0)

        return self.coefficient * loss


@registry.register_loss("mae")
class MAELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor
    ) -> torch.Tensor:
        return self.loss(pred, target)


@registry.register_loss("mse")
class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.MSELoss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor
    ) -> torch.Tensor:
        return self.loss(pred, target)


@registry.register_loss("per_atom_mae")
class PerAtomMAELoss(nn.Module):
    """
    Simply divide a loss by the number of atoms/nodes in the graph.
    Current this loss is intened to used with scalar values, not vectors or higher tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor, step=None, batch=None
    ) -> torch.Tensor:
        _natoms = torch.reshape(natoms, target.shape)
        # check if target is a scalar
        assert target.dim() == 1 or (target.dim() == 2 and target.shape[1] == 1)
        # check per_atom shape
        assert (target / _natoms).shape == target.shape
        return self.loss(pred / _natoms, target / _natoms)

# Assuming correct imports: torch, nn, registry, WandBSingletonLogger

@registry.register_loss("test_per_atom_mae")
class TestPerAtomMAELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Use reduction='none' to get element-wise loss before manual averaging
        self.loss_fn = nn.L1Loss(reduction="none")
        self.logger = WandBSingletonLogger.get_instance()
        self.mae_steps = 50000 # Define MAE steps threshold

    # Corrected signature matching compute_loss call
    def forward(
        self, pred_tensor: torch.Tensor, target_tensor: torch.Tensor, natoms: torch.Tensor, step: int = None, batch: dict = None
    ) -> torch.Tensor:

        # Ensure prediction and target are 1D and on same device
        predicted_energy = pred_tensor.squeeze().to(target_tensor.device)
        target_energy = target_tensor.squeeze() # Target comes from batch, should be correct device
        _natoms = natoms.view_as(predicted_energy).to(predicted_energy.device)
        # Avoid division by zero
        _natoms = torch.clamp(_natoms, min=1)

        is_mae_phase = batch.get("is_mae_phase", False) # Check flag from batch

        if is_mae_phase and step is not None and step < self.mae_steps:
            # --- MAE Phase ---
            mask = batch["mask"]
            original_energy = batch["original_energy"]

            # Validate MAE data is present
            if mask is None or original_energy is None:
                print(f"Warning: MAE phase (step {step}) but mask or original_energy missing. Calculating standard MAE.")
                is_mae_phase = False # Fallback to standard MAE
            else:
                 mask = mask.to(predicted_energy.device).squeeze()
                 original_energy = original_energy.to(predicted_energy.device).squeeze()

            # Proceed only if still in MAE phase after validation
            if is_mae_phase:
                # Handle empty mask case
                if mask.sum() == 0:
                    # Return zero loss tensor if nothing is masked
                    # Return shape expected by DDPLoss (scalar if it averages means, or element-wise zeros)
                    # Assuming scalar return:
                     return torch.tensor(0.0, device=predicted_energy.device, requires_grad=True)

                # Select masked elements
                pred_masked = predicted_energy[mask]
                orig_masked = original_energy[mask]
                natoms_masked = _natoms[mask]

                # Calculate element-wise loss ONLY on masked items
                loss_elements = self.loss_fn(pred_masked / natoms_masked, orig_masked / natoms_masked)

                # Calculate mean loss over masked items for logging and return
                final_loss = loss_elements.mean()


                log_dict = {
                        "l1_loss_terms/masked_per_atom_MAE": final_loss.item(),
                        "l1_loss_terms/mask_ratio_in_batch": mask.float().mean().item(),
                    }
                self.logger.log(log_dict, step=step, commit=False)

                # Return the scalar mean loss (assuming DDPLoss averages this)
                return final_loss

        # --- Standard L1 Phase (or fallback from MAE) ---
        # Calculate element-wise loss on ALL items
        loss_elements = self.loss_fn(predicted_energy / _natoms, target_energy / _natoms)

        # Calculate mean loss over all items
        final_loss = loss_elements.mean()

        # --- Logging for Standard MAE ---
        if self.logger.is_active:
             log_dict = {
                 "l1_loss_terms/standard_per_atom_MAE": final_loss.item(),
             }
             self.logger.log(log_dict, step=step, commit=False)

        # Return the scalar mean loss
        return final_loss

@registry.register_loss("hlgaussl1")
class HLGaussLossL1(nn.Module):
   """
   hl gauss loss
   """


   def __init__(self, min_value: float = -1.0, max_value: float = 1.0, num_bins: int = 200) -> None:
       super().__init__()
       #self.simpleMAEloss = nn.L1Loss()
       # reduction should be none as it is handled in DDPLoss
       # self.reduction = "none"
       self.ignore_shape_check = True
       self.min_value = min_value
       self.max_value = max_value
       self.num_bins = num_bins
       self.sigma = 0.75 * ((self.max_value-self.min_value)/self.num_bins)
       self.support = torch.linspace(
           min_value, max_value, num_bins + 1, dtype=torch.float32 
       )
       self.logger = (
           WandBSingletonLogger.get_instance()
       )


   def transform_to_probs(self, target):
       "convert scalar target to categorical distribution using Normal CDF"
       support = self.support.to(target.device)
       cdf_evals = torch.special.erf(
           (support - target.unsqueeze(-1))
           / (torch.sqrt(torch.tensor(2.0)) * self.sigma)
           )


       z = cdf_evals[..., -1] - cdf_evals[..., 0]
       bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
       return bin_probs / z.unsqueeze(-1)
      
   def transform_from_probs(self, probs):
       "prob to scalar"
       support = self.support.to(probs.device)
       centers = (support[:-1] + support[1:]) / 2
       return torch.sum(probs * centers, dim=-1)


   def forward(
       self, pred, target: torch.Tensor, natoms: torch.Tensor, step
   ) -> torch.Tensor:
       
       _natoms = torch.reshape(natoms, target.shape)
       target_per_atom = target / _natoms
       pred_per_atom = pred / _natoms
       return torch.nn.functional.l1_loss(target_per_atom, pred_per_atom)
       


@registry.register_loss("HLGaussLossHierarchal")
class HLGaussLossHierarchal(nn.Module):
   """
   hl gauss loss
   """


   def __init__(self) -> None:
       super().__init__()
       self.reduction = "none"
       self.ignore_shape_check = True
       self.lambda_fine = 0.5
       self.sigma_coarse = 0.5
       log_min, log_max = -2, 2
       num_points_per_side = 100
       positive_part = torch.logspace(log_min, log_max, num_points_per_side)
       negative_part = -torch.flip(positive_part, dims=[0])
       coarse_support = torch.cat((negative_part, torch.tensor([0.0]), positive_part))
       self.register_buffer("coarse_support", coarse_support)
       self.num_bins = len(self.coarse_support) - 1

       self.num_fine_bins = 20
       self.sigma_fine = 0.05
       fine_support = torch.linspace(0.0, 1.0, self.num_fine_bins + 1)
       self.register_buffer("fine_support", fine_support)
       self.logger = (
           WandBSingletonLogger.get_instance()
       )


   def transform_to_probs(self, target: torch.Tensor, support: torch.Tensor, sigma: float):
       "convert scalar target to categorical distribution using Normal CDF"
       support = support.to(target.device)
       target = target.unsqueeze(-1)

       cdf_evals = 0.5 * (1 + torch.special.erf(
        (support - target) / (sigma * (2**0.5))
    ))
       z = cdf_evals[..., -1] - cdf_evals[..., 0]
       bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
       return bin_probs / z.unsqueeze(-1)
      
   def transform_from_probs(self, probs):
       "prob to scalar"
       support = self.support.to(probs.device)
       centers = (support[:-1] + support[1:]) / 2
       return torch.sum(probs * centers, dim=-1)


   def forward(
       self, pred, target: torch.Tensor, natoms: torch.Tensor, step
   ) -> torch.Tensor:
       coarse_logits = pred[:, :self.num_bins]
       fine_logits = pred[:, self.num_bins:]
       self.coarse_support = self.coarse_support.to(target.device)
       self.fine_support = self.fine_support.to(target.device)
       
       target_per_atom = target / natoms
       min_val = self.coarse_support[0].to(target.device)
       max_val = self.coarse_support[-1].to(target.device)
       target_per_atom_clamped = torch.clamp(
           target_per_atom, min_val, max_val
        )
       coarse_target_probs = self.transform_to_probs(
           target_per_atom_clamped.squeeze(-1), self.coarse_support, self.sigma_coarse
           )
       
       # coarse loss
       log_coarse_probs = torch.nn.functional.log_softmax(coarse_logits, dim=-1)
       coarse_loss = torch.nn.functional.kl_div(log_coarse_probs, coarse_target_probs, reduction='none').sum(dim=1)

       bin_indices = torch.bucketize(
           target_per_atom.squeeze(-1), self.coarse_support[1:-1].to(target.device)
           )
       bin_starts = self.coarse_support.to(target.device)[bin_indices]
       bin_ends = self.coarse_support.to(target.device)[bin_indices + 1]
       bin_width = bin_ends - bin_starts + 1e-12
       fine_target_normalized = (target_per_atom.squeeze(-1) - bin_starts) / bin_width
       fine_target_normalized_clamped = torch.clamp(fine_target_normalized, 0.0, 1.0)
       
       fine_target_probs = self.transform_to_probs(
           fine_target_normalized_clamped, self.fine_support, self.sigma_fine
           )
       
       log_fine_probs = torch.nn.functional.log_softmax(fine_logits, dim=-1)
       fine_loss = torch.nn.functional.kl_div(log_fine_probs, fine_target_probs, reduction='none').sum(dim=1)
       
       total_loss = coarse_loss + self.lambda_fine * fine_loss
       
       coarse_mol_probs = torch.nn.functional.softmax(coarse_logits, dim=-1)
       fine_mol_probs = torch.nn.functional.softmax(fine_logits, dim=-1)
       coarse_bin_indices = torch.argmax(coarse_mol_probs, dim=-1)
       v_start = self.coarse_support[coarse_bin_indices]
       v_end = self.coarse_support[coarse_bin_indices + 1]
       bin_width = v_end - v_start
       fine_centers = (self.fine_support[:-1] + self.fine_support[1:]) / 2
       fine_scale_pos = torch.sum(fine_mol_probs * fine_centers.to(fine_mol_probs.device), dim=-1)
       scalar_energy_pred = v_start + (fine_scale_pos * bin_width)
       _natoms = torch.reshape(natoms, target.shape)
       mae = torch.abs((scalar_energy_pred.squeeze() / _natoms) - (target.squeeze() / _natoms)).mean()



       log_dict = {
                "hlgauss_loss_terms/total_loss": total_loss.mean().item(),
                "hlgauss_loss_terms/coarse_loss": coarse_loss.mean().item(),
                "hlgauss_loss_terms/fine_loss": fine_loss.mean().item(),
                "hlgauss_loss_terms/exact_mae": mae.item(), # Log the exact MAE
                "hlgauss_loss_terms/target_max": torch.max(target_per_atom).item(),
                "hlgauss_loss_terms/target_min": torch.min(target_per_atom).item(),
            }
       self.logger.log(log_dict, step=step, commit=False)  
       return total_loss.mean()

@registry.register_loss("HLGaussLossLinear")
class HLGaussLossLinear(nn.Module):
    def __init__(self, min_value: float = -10.0, max_value: float = 20.0, num_bins: int = 100) -> None:
        super().__init__()
        self.ignore_shape_check = True
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        
        # Sigma is calculated based on uniform bin width
        bin_width = (self.max_value - self.min_value) / self.num_bins
        self.sigma = 0.75 * bin_width
        
        # Support is created with evenly spaced points
        self.support = torch.linspace(
            min_value, max_value, num_bins + 1, dtype=torch.float32 
        )
        
        self.logger = (
            WandBSingletonLogger.get_instance()
            )
    def transform_to_probs(self, target, support):
        target = target.unsqueeze(-1)
        cdf_evals = 0.5 * (1 + torch.special.erf(
            (support - target) / (self.sigma * (2**0.5))
        ))
        z = cdf_evals[..., -1] - cdf_evals[..., 0] + 1e-12
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z.unsqueeze(-1)
      
    def transform_from_probs(self, probs, support):
        centers = (support[:-1] + support[1:]) / 2
        return torch.sum(probs * centers, dim=-1)

    def forward(self, pred, target: torch.Tensor, natoms: torch.Tensor, step):

        support = self.support.to(target.device)
        
        target_per_atom = target / natoms
        clamped_target = torch.clamp(target_per_atom, self.min_value, self.max_value)
        
        target_probs = self.transform_to_probs(clamped_target, support)
        pred_probs_normalized = pred / natoms.unsqueeze(1)

        target_probs_stable = target_probs + 1e-12
        pred_probs_stable = pred_probs_normalized + 1e-12
        log_pred_probs = torch.log(pred_probs_stable)
        
        loss_per_sample = torch.nn.functional.kl_div(
            log_pred_probs, target_probs_stable, reduction="none"
            ).sum(dim=1)
        
        pred_scalar = self.transform_from_probs(pred_probs_normalized, support)
        prediction_error = pred_scalar - target_per_atom
        mae = torch.abs(prediction_error).mean()
        mean_error_bias = prediction_error.mean() # New: Logs prediction bias
        pred_scalar_std = pred_scalar.std()        # New: Logs variance of predictions
        target_scalar_std = target_per_atom.std()  # New: Logs variance of targets
        target_entropy = torch.distributions.Categorical(probs=target_probs_stable).entropy().mean() # Requested
        pred_entropy = torch.distributions.Categorical(probs=pred_probs_stable).entropy().mean()   # New: Tracks model confidence

        #mae = torch.abs((self.transform_from_probs(pred / natoms.unsqueeze(1), support) - target_per_atom)).mean()

        log_dict = {
            "hlgauss_loss_terms/total_loss": loss_per_sample.mean().item(),
            "hlgauss_loss_terms/exact_mae": mae.item(),
            "hlgauss_loss_terms/target_max": torch.max(target_per_atom).item(),
            "hlgauss_loss_terms/target_min": torch.min(target_per_atom).item(),
            
            "hlgauss_diagnostics/target_entropy": target_entropy.item(),
            "hlgauss_diagnostics/pred_entropy": pred_entropy.item(),
            "hlgauss_diagnostics/mean_error_bias": mean_error_bias.item(),
            "hlgauss_diagnostics/target_scalar_std": target_scalar_std.item(),
            "hlgauss_diagnostics/pred_scalar_std": pred_scalar_std.item()
            }
        self.logger.log(log_dict, step=step, commit=False)
        if torch.isnan(loss_per_sample).any():
            print("NaN detected in loss_per_sample")

        return loss_per_sample

@registry.register_loss("HLGaussLossCE_Log")
class HLGaussLossCE_Log(nn.Module): # Renamed for clarity
    def __init__(self, start_exp: float = -4, end: float = 20, num_points_per_side: int = 16) -> None:
        super().__init__()
        self.ignore_shape_check = True
        
        # --- Create the symmetric log-spaced support tensor ---
        positive_part = torch.logspace(start_exp, torch.log10(torch.tensor(end)), num_points_per_side, base=2)
        negative_part = -torch.flip(positive_part, dims=[0])
        self.support = torch.cat((negative_part[:-1], torch.tensor([0.0]), positive_part))
        self.num_bins = len(self.support) - 1
        

        right_bin = self.support[1:]
        left_bin = self.support[:-1]
        bin_width = (right_bin - left_bin)
        self.sigma = 0.75 * torch.median(bin_width)
        
        self.logger = (WandBSingletonLogger.get_instance())

    def transform_to_probs(self, target):
        support = self.support.to(target.device)
        target = target.unsqueeze(-1)
        cdf_evals = 0.5 * (1 + torch.special.erf(
            (support - target) / (self.sigma * (2**0.5))
        ))
        z = cdf_evals[..., -1] - cdf_evals[..., 0] + 1e-12
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z.unsqueeze(-1)
      
    def transform_from_probs(self, probs):
        support = self.support.to(probs.device)
        centers = (support[:-1] + support[1:]) / 2
        return torch.sum(probs * centers, dim=-1)

    def forward(self, pred, target: torch.Tensor, natoms: torch.Tensor, step):
        
        
        target_per_atom = target / natoms
        # Clamp targets to the new dynamic log-spaced range
        self.support = self.support.to(target.device)
        min_val, max_val = self.support.min(), self.support.max()
        clamped_target = torch.clamp(target_per_atom, min_val, max_val)
        target_probs = self.transform_to_probs(clamped_target)
        
        log_pred_probs = torch.log(pred / natoms.unsqueeze(1))
        loss_per_sample = torch.nn.functional.kl_div(
            log_pred_probs, target_probs, reduction="none"
        ).sum(dim=1)
        # Return the average loss for the batch
        mae = torch.abs((self.transform_from_probs(pred / natoms.unsqueeze(1)) - target_per_atom)).mean()

        log_dict = {
                "hlgauss_loss_terms/total_loss": loss_per_sample.mean().item(),
                "hlgauss_loss_terms/exact_mae": mae.item(),
                "hlgauss_loss_terms/target_max": torch.max(target_per_atom).item(),
                "hlgauss_loss_terms/target_min": torch.min(target_per_atom).item(),
            }
        self.logger.log(log_dict, step=step, commit=False) 
        return loss_per_sample.mean()


@registry.register_loss("l2norm")
@registry.register_loss("l2mae")
class L2NormLoss(nn.Module):
    """
    Currently this loss is intened to used with vectors.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor, step=None, batch=None
    ) -> torch.Tensor:
        assert target.dim() == 2
        assert target.shape[1] != 1
        return torch.linalg.vector_norm(pred - target, ord=2, dim=-1)


class DDPLoss(nn.Module):
    """
    This class is a wrapper around a loss function that does a few things
    like handle nans and importantly ensures the reduction is done
    correctly for DDP. The main issue is that DDP averages gradients
    over replicas — this only works out of the box if the dimension
    you are averaging over is completely consistent across all replicas.
    In our case, that is not true for the number of atoms per batch and
    there are edge cases when the batch size differs between replicas
    e.g. if the dataset size is not divisible by the batch_size.

    Scalars are relatively straightforward to handle, but vectors and higher tensors
    are a bit trickier. Below are two examples of forces.

    Forces input: [Nx3] target: [Nx3]
    Forces are a vector of length 3 (x,y,z) for each atom.
    Number of atoms per batch (N) is different for each DDP replica.

    MSE example:
    #### Local loss computation ####
    local_loss = MSELoss(input, target) -> [Nx3]
    num_samples = local_loss.numel() -> [Nx3]
    local_loss = sum(local_loss [Nx3]) -> [1] sum reduces the loss to a scalar
    global_samples = all_reduce(num_samples) -> [N0x3 + N1x3 + N2x3 + ...] = [1] where N0 is the number of atoms on replica 0
    local_loss = local_loss * world_size / global_samples -> [1]
    #### Global loss computation ####
    global_loss = sum(local_loss / world_size) -> [1]
    == sum(local_loss / global_samples) # this is the desired corrected mean

    Norm example:
    #### Local loss computation ####
    local_loss = L2MAELoss(input, target) -> [N]
    num_samples = local_loss.numel() -> [N]
    local_loss = sum(local_loss [N]) -> [1] sum reduces the loss to a scalar
    global_samples = all_reduce(num_samples) -> [N0 + N1 + N2 + ...] = [1] where N0 is the number of atoms on replica 0
    local_loss = local_loss * world_size / global_samples -> [1]
    #### Global loss computation ####
    global_loss = sum(local_loss / world_size) -> [1]
    == sum(local_loss / global_samples) # this is the desired corrected mean
    """

    def __init__(
        self,
        loss_name,
        reduction: Literal["mean", "sum"],
    ) -> None:
        super().__init__()
        self.loss_fn = registry.get_loss_class(loss_name)()
        # default reduction is mean
        self.reduction = reduction if reduction is not None else "mean"
        self.reduction_map = {
            "mean": self.mean,
            "sum": self.sum,
        }
        assert self.reduction in list(
            self.reduction_map.keys()
        ), "Reduction must be one of: 'mean', 'sum'"

    def sum(self, input, loss, natoms):
        # this sum will reduce the loss down to a single scalar
        return torch.sum(loss)

    def _ddp_mean(self, num_samples, loss):
        global_samples = distutils.all_reduce(num_samples, device=loss.device)
        # Multiply by world size since gradients are averaged across DDP replicas
        # warning this is probably incorrect for any model parallel approach
        return loss * distutils.get_world_size() / global_samples

    def mean(self, input, loss, natoms):
        # total elements to take the mean over
        # could be batch_size, num_atoms, num_atomsx3, etc
        num_samples = loss.numel()
        # this sum will reduce the loss down from num_sample -> 1
        loss = self.sum(input, loss, natoms)
        return self._ddp_mean(num_samples, loss)

    def _reduction(self, input, loss, natoms):
        if self.reduction in self.reduction_map:
            return self.reduction_map[self.reduction](input, loss, natoms)
        else:
            raise ValueError("Reduction must be one of: 'mean', 'sum'")

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
    ):
        # ensure torch doesn't do any unwanted broadcasting
        assert (
            input.shape == target.shape
        ), f"Mismatched shapes: {input.shape} and {target.shape}"

        # zero out nans, if any
        found_nans_or_infs = not torch.all(input.isfinite())
        if found_nans_or_infs is True:
            logging.warning("Found nans while computing loss")
            input = torch.nan_to_num(input, nan=0.0)

        loss = self.loss_fn(input, target, natoms)
        return self._reduction(input, loss, natoms)

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import os
from math import prod
from typing import Dict
import random

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from scipy.spatial.transform import Rotation
from torch import Tensor
from einops import rearrange, repeat

from proteinfoundation.flow_matching.r3n_fm import R3NFlowMatcher
from proteinfoundation.nn.protein_transformer import ProteinTransformerAF3
from proteinfoundation.proteinflow.model_trainer_base import ModelTrainerBase, _extract_cath_code
from proteinfoundation.utils.align_utils.align_utils import kabsch_align
from proteinfoundation.utils.coors_utils import ang_to_nm, trans_nm_to_atom37
from proteinfoundation.utils.ff_utils.pdb_utils import extract_ca

@rank_zero_only
def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)


def sample_uniform_rotation(
    shape=tuple(), dtype=None, device=None
) -> Float[Tensor, "*batch 3 3"]:
    """
    Samples rotations distributed uniformly.

    Args:
        shape: tuple (if empty then samples single rotation)
        dtype: used for samples
        device: torch.device

    Returns:
        Uniformly samples rotation matrices [*shape, 3, 3]
    """
    return torch.tensor(
        Rotation.random(prod(shape)).as_matrix(),
        device=device,
        dtype=dtype,
    ).reshape(*shape, 3, 3)


class ProteinAE(ModelTrainerBase):
    def __init__(self, cfg_exp, store_dir=None):
        super(ProteinAE, self).__init__(cfg_exp=cfg_exp, store_dir=store_dir)
        self.save_hyperparameters()

        # Define flow matcher
       
        self.ca_only = cfg_exp.model.ca_only
        self.dim_latent = cfg_exp.model.ae.encoder.dim_latent
        self.fm = R3NFlowMatcher(
            zero_com=True,
            scale_ref=1.0,
            dim=3
        )  # Work in nm
        self.fm_z = R3NFlowMatcher(
            zero_com=False,
            scale_ref=1.0,
            dim=self.dim_latent
        )  # Work in latent space
        # Cache frequently used config values
        self.zaug_p = cfg_exp.training.get("zaug_p", 0.1)
        self.use_inv_folding_loss = cfg_exp.loss.get("use_inv_folding_loss", False) and \
            cfg_exp.model.get("apply_inv_folding", False)
        self.z_noise_scale = cfg_exp.training.get("z_noise_scale", 0.0)
        self.latent_add_place = cfg_exp.model.get("latent_add_place", "cond")
        self.self_cond = cfg_exp.training.get("self_cond", False)

        # Neural network
        self.encoder = ProteinTransformerAF3(
            **cfg_exp.model.ae.encoder,
            ca_only=self.ca_only,
            apply_inv_folding=self.use_inv_folding_loss
        )
        self.decoder = ProteinTransformerAF3(
            **cfg_exp.model.ae.decoder,
            ca_only=self.ca_only,
            apply_inv_folding=self.use_inv_folding_loss,
            latent_add_place=self.latent_add_place
        )

        self.nparams = sum(
            p.numel() 
            for p in \
            [
                *self.encoder.parameters(),
                *self.decoder.parameters()
            ] 
            if p.requires_grad
        )

    def align_wrapper(self, x_0, x_1, mask):
        """Performs Kabsch on the translation component of x_0 and x_1."""
        return kabsch_align(mobile=x_0, target=x_1, mask=mask)

    def predict_clean(self, batch: Dict):
        nn_out = self.decoder(batch)  # [*, n, 3]
        return (
            self._nn_out_to_x_clean(nn_out, batch),
            nn_out,  # [*, n, 3]
        )

    def predict_clean_n_v_w_guidance(
        self,
        batch: Dict,
        guidance_weight: float = 1.0,
        autoguidance_ratio: float = 0.0,
    ):
        if self.motif_conditioning and \
        (
            "fixed_structure_mask" not in batch \
                or "x_motif" not in batch
        ):
            batch.update(self.motif_factory(batch, zeroes = True))

        nn_out = self.decoder(batch)
        x_pred = self._nn_out_to_x_clean(nn_out, batch)

        if guidance_weight != 1.0:
            assert autoguidance_ratio >= 0.0 and autoguidance_ratio <= 1.0
            if autoguidance_ratio > 0.0:  # Use auto-guidance
                nn_out_ag = self.nn_ag(batch)
                x_pred_ag = self._nn_out_to_x_clean(nn_out_ag, batch)
            else:
                x_pred_ag = torch.zeros_like(x_pred)

            if autoguidance_ratio < 1.0:  # Use CFG
                assert (
                    "cath_code" in batch
                ), "Only support CFG when cath_code is provided"
                uncond_batch = batch.copy()
                uncond_batch.pop("cath_code")
                nn_out_uncond = self.decoder(uncond_batch)
                x_pred_uncond = self._nn_out_to_x_clean(nn_out_uncond, uncond_batch)
            else:
                x_pred_uncond = torch.zeros_like(x_pred)

            x_pred = guidance_weight * x_pred + (1 - guidance_weight) * (
                autoguidance_ratio * x_pred_ag
                + (1 - autoguidance_ratio) * x_pred_uncond
            )

        v = self.fm.xt_dot(x_pred, batch["x_t"], batch["t"], batch["coords_mask"])
        return x_pred, v

    def extract_clean_sample(self, batch):
        """
        Extracts clean sample, mask, batch size, protein length n, and dtype from the batch.
        Applies augmentations if those are required.

        Args:
            batch: batch from dataloader.

        Returns:
            Tuple (x_1, mask, batch_shape, n, dtype)
        """
        
        # Extract coordinates and masks based on ca_only mode
        x_1, mask, coords_mask = self._extract_coordinates_and_masks(batch)
        
        # Apply rotation augmentation if enabled
        if self.cfg_exp.model.augmentation.global_rotation:
            x_1, coords_mask, mask = \
            self._apply_rotation_augmentation(
                x_1,
                coords_mask,
                mask,
            )
        
        batch_shape = x_1.shape[:-2]
        n = x_1.shape[-2]
        
        return (
            ang_to_nm(x_1),
            mask,
            coords_mask,
            batch_shape,
            n,
            x_1.dtype,
        )  # Since we work in nm throughout

    def _extract_coordinates_and_masks(self, batch):
        """Extract coordinates and masks based on ca_only mode."""
        if self.ca_only:
            return self._extract_ca_coordinates(batch)
        else:
            return self._extract_backbone_coordinates(batch)

    def _extract_ca_coordinates(self, batch):
        """Extract CA-only coordinates and masks."""
        x_1 = batch["coords"][:, :, 1, :]  # [b, n, 3]
        coords_mask = batch["mask_dict"]["coords"][..., 0, 0]  # [b, n] boolean
        mask = coords_mask
        return x_1, mask, coords_mask

    def _extract_backbone_coordinates(self, batch):
        """Extract backbone coordinates (N, CA, C, O) and masks."""
        # Index of [N, CA, C, O] is [0, 1, 2, 4]
        BB_INDEX = [0, 1, 2, 4]
        x_1 = batch["coords"][:, :, BB_INDEX, :]  # [b, n, 4, 3]
        x_1 = rearrange(x_1, "b n c d -> b (n c) d")  # [b, 4*n, 3]
        coords_mask = batch["mask_dict"]["coords"][..., BB_INDEX, 0]  # [b, n, 4] boolean
        mask = coords_mask[..., 1]  # Use CA mask
        coords_mask = rearrange(coords_mask, "b n c -> b (n c)")  # [b, 4*n]
        return x_1, mask, coords_mask

    def _apply_rotation_augmentation(self, x_1, coords_mask, mask):
        """Apply rotation augmentation to coordinates and masks."""
        # CAREFUL: If naug_rot is > 1 this increases "batch size"
        x_1, coords_mask = self.apply_random_rotation(
            x_1, 
            coords_mask,
            naug=self.cfg_exp.model.augmentation.naug_rot
        )
        
        # Update mask for backbone mode after rotation
        if not self.ca_only:
            mask = rearrange(coords_mask, "b (n c) -> b n c", c=4)[..., 1]
        
        return x_1, coords_mask, mask

    def apply_random_rotation(self, x, mask, naug=1):
        """
        Applies random rotation augmentation. Each sample in the batch may receive more than one augmentation,
        specified by the parameters naug. If naug > 1 this is basically increaseing the batch size from b to
        naug * b. This should likely be implemented in the dataloaders.

        Args:
            - x: Data batch, shape [b, n, 3]
            - mask: Binary, shape [b, n]
            - naug: Number of augmentations to apply to each sample, effectively increasing batch size if >1.

        Returns:
            Augmented samples and mask, shapes [b * naug, n, 3] and [B * naug, n].
        """

        # Repeat for multiple augmentations per sample
        x = x.repeat([naug, 1, 1])  # [naug * b, n, 3]
        mask = mask.repeat([naug, 1])  # [naug * b, n]

        # Sample and apply rotations
        rots = sample_uniform_rotation(
            shape=x.shape[:-2], dtype=x.dtype, device=x.device
        )  # [naug * b, 3, 3]
        x_rot = torch.matmul(x, rots)
        return self.fm._mask_and_zero_com(x_rot, mask), mask
    
    def training_step(self, batch, batch_idx):
        """
        Computes training loss for batch of samples.

        Args:
            batch: Data batch.

        Returns:
            Training loss averaged over batches.
        """
        val_step = batch_idx == -1  # validation step is indicated with batch_idx -1
        log_prefix = "validation_loss" if val_step else "train"
        
        # Prepare batch data
        self._prepare_batch_data(batch)
        
        # Encode and prepare for flow matching
        single_repr = self._encode_and_prepare_flow_matching(batch)
        
        # Add noise to single representation (Optional)
        self._add_noise_to_single_repr(single_repr, batch)
        
        # Self-conditioning prediction
        self._apply_self_conditioning(batch)
        
        # Main prediction
        x_1_pred, nn_out = self.predict_clean(batch)

        # Compute and log losses
        train_loss = self._compute_all_losses(
            batch, x_1_pred, nn_out, log_prefix
        )
        
        # Logging
        self._log_training_metrics(train_loss, batch, log_prefix, val_step)

        return train_loss

    def _prepare_batch_data(self, batch):
        """Prepare batch data by extracting and processing coordinates."""
        x_1, mask, coords_mask, batch_shape, n, dtype = self.extract_clean_sample(batch)
        x_1 = self.fm._mask_and_zero_com(x_1, coords_mask)
        batch.update({
            "x_1": x_1,
            "mask": mask,
            "coords_mask": coords_mask,
            "batch_shape": batch_shape,
            "n": n,
            "dtype": dtype
        })

    def _encode_and_prepare_flow_matching(self, batch):
        """Encode input and prepare flow matching interpolation."""
        single_repr = self.encoder(batch).get("single_repr", None)
        
        # Sample time, reference and align reference to target
        t = self.sample_t(batch["batch_shape"])
        x_0 = self.fm.sample_reference(
            n=batch["n"],
            shape=batch["batch_shape"],
            device=self.device,
            dtype=batch["dtype"],
            mask=batch["coords_mask"],
        )
        
        # Interpolation
        x_t = self.fm.interpolate(x_0, batch["x_1"], t)
        
        batch.update({"t": t, "x_t": x_t})
        return single_repr

    def _add_noise_to_single_repr(self, single_repr, batch):
        """Add noise to single representation."""
        single_rep_noise = torch.randn_like(
            single_repr,
            device=single_repr.device,
        )
        single_repr = single_repr + \
            single_rep_noise * self.z_noise_scale
        batch["single_repr"] = single_repr

    def _apply_self_conditioning(self, batch):
        """Apply self-conditioning if enabled."""
        if random.random() > 0.5 and self.self_cond:
            x_pred_sc, _ = self.predict_clean(batch)
            if not self.ca_only:
                x_pred_sc = extract_ca(x_pred_sc)
            batch["x_sc"] = self.detach_gradients(x_pred_sc)

    def _compute_all_losses(self, batch, x_1_pred, nn_out, log_prefix):
        """Compute all loss components."""
        x_1, mask, coords_mask = batch["x_1"], batch["mask"], batch["coords_mask"]
        x_t, t = batch["x_t"], batch["t"]
        
        # Flow matching loss
        fm_loss = self.compute_fm_loss(
            x_1, x_1_pred, x_t, t, mask, coords_mask, log_prefix=log_prefix
        )
        train_loss = torch.mean(fm_loss)
        
        # Inverse folding loss
        if self.use_inv_folding_loss:
            inv_folding_loss = self.compute_inv_folding_loss(
                batch["residue_type"], nn_out["residue_type_logits"], log_prefix=log_prefix
            )
            train_loss = train_loss + inv_folding_loss
            
        return train_loss

    def _log_training_metrics(self, train_loss, batch, log_prefix, val_step):
        """Log training metrics and scaling information."""
        mask = batch["mask"]
        
        self.log(
            f"{log_prefix}/loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=mask.shape[0],
            sync_dist=True,
            add_dataloader_idx=False,
        )

        if not val_step:
            self.log(
                "train_loss",
                train_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=mask.shape[0],
                sync_dist=True,
                add_dataloader_idx=False,
            )
            self._log_scaling_metrics(mask)

    def _log_scaling_metrics(self, mask):
        """Log scaling law metrics."""
        b, n = mask.shape
        nflops_step = None  # TODO: Implement if needed
        
        if nflops_step is not None:
            self.nflops = self.nflops + \
            nflops_step * self.trainer.world_size
            self.log(
                "scaling/nflops",
                self.nflops * 1.0,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=1,
                sync_dist=True,
            )

        self.nsamples_processed = self.nsamples_processed + \
            b * self.trainer.world_size
        self.log(
            "scaling/nsamples_processed",
            self.nsamples_processed * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )

        self.log(
            "scaling/nparams",
            self.nparams * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )

    def compute_loss_weight(
        self, t: Float[Tensor, "*"], eps: float = 1e-3
    ) -> Float[Tensor, "*"]:
        t = t.clamp(min=eps, max=1.0 - eps)  # For safety
        return t / (
            1.0 - t
        )

    def compute_fm_loss(
        self,
        x_1: Float[Tensor, "* n 3"],
        x_1_pred: Float[Tensor, "* n 3"],
        x_t: Float[Tensor, "* n 3"],
        t: Float[Tensor, "*"],
        mask: Bool[Tensor, "* nres"],
        coords_mask: Bool[Tensor, "* nresx4"],
        log_prefix: str,
    ) -> Float[Tensor, "*"]:
        """
        Computes and logs flow matching loss.

        Args:
            x_1: True clean sample, shape [*, n, 3].
            x_1_pred: Predicted clean sample, shape [*, n, 3].
            x_t: Sample at interpolation time t (used as input to predict clean sample), shape [*, n, 3].
            t: Interpolation time, shape [*].
            mask: Boolean residue mask, shape [*, nres].

        Returns:
            Flow matching loss.
        """
        natoms = torch.sum(coords_mask, dim=-1) * 3  # [*]

        err = (x_1 - x_1_pred) * coords_mask[..., None]  # [*, n, 3]
        loss = torch.sum(err**2, dim=(-1, -2)) / natoms  # [*]

        total_loss_w = 1.0 / ((1.0 - t) ** 2 + 1e-5)

        loss = loss * total_loss_w  # [*]
        if log_prefix:
            self._log_metric(f"{log_prefix}/trans_loss", torch.mean(loss), mask.shape[0], prog_bar=True)
        return loss
    
    def compute_inv_folding_loss(
        self,
        gt_residue_type: Float[Tensor, "* n"],
        pred_residue_type_logits: Float[Tensor, "* n c"],
        ignore_index: int = -1,
        log_prefix: str = "",
    ) -> Float[Tensor, "*"]:
        """
        Computes inverse folding loss.
        """
        bs = gt_residue_type.shape[0]
        pred_residue_type_logits = rearrange(
            pred_residue_type_logits,
            "b n c -> b c n",
        )
        # cross entropy loss
        loss = F.cross_entropy(
            pred_residue_type_logits,
            gt_residue_type.long(),
            ignore_index=ignore_index,
        )
        self._log_metric(f"{log_prefix}/inv_folding_loss", torch.mean(loss), bs)
        return loss

    def _log_metric(
        self,
        name,
        value,
        batch_size,
        prog_bar=False,
        on_step=True,
        on_epoch=True,
    ):
        """Helper method for consistent metric logging."""
        self.log(
            name,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def detach_gradients(self, x):
        """Detaches gradients from sample x"""
        return x.detach()

    def samples_to_atom37(self, samples):
        """
        Transforms samples to atom37 representation.

        Args:
            samples: Tensor of shape [b, n, 3]

        Returns:
            Samples in atom37 representation, shape [b, n, 37, 3].
        """
        return trans_nm_to_atom37(samples, ca_only=self.ca_only)  # [b, n, 37, 3]
    
    def predict_step(self, batch, batch_idx):
        """
        Makes predictions. Should call set_inf_cfg before calling this.

        Args:
            batch: data batch, contains no data, but the info of the samples
                to generate (nsamples, nres, dt)

        Returns:
            Samples generated in atom 37 format.
        """
        ae_mode = self.inf_cfg.ae_mode
        sampling_args = self.inf_cfg.sampling_bbflow
        
        # Prepare inference parameters
        cath_code = _extract_cath_code(batch) if self.inf_cfg.get("fold_cond", False) else None
        guidance_weight = self.inf_cfg.get("guidance_weight", 1.0)
        autoguidance_ratio = self.inf_cfg.get("autoguidance_ratio", 0.0)
        dt = self.inf_cfg.get("dt", 0.0025)
        
        # Process based on mode
        if ae_mode == "autoencode":
            # Extract and encode input
            x_1, mask, coords_mask, batch_shape, n, dtype = self.extract_clean_sample(batch)
            x_1 = self.fm._mask_and_zero_com(x_1, coords_mask)
            
            batch.update({
                "x_1": x_1,
                "mask": mask,
                "coords_mask": coords_mask,
                "nsamples": 1,
                "nres": int(n // 4) if not self.ca_only else n,
            })
            
            single_repr = self.encoder(batch).get("single_repr", None)
            
        elif ae_mode == "decode":
            # Use provided single representation
            single_repr = batch["single_repr"]
            dtype = single_repr.dtype
            
            if "mask" not in batch:
                raise ValueError("Mask is required for decode mode")
            
            mask = batch["mask"]
            coords_mask = repeat(batch["mask"], "b n -> b (n c)", c=4)
            
            batch.update({
                "nsamples": 1,
                "nres": batch["mask"].shape[1],
            })
            
            x_1 = None  # No ground truth for decode mode
        else:
            raise ValueError(f"Sampling mode {ae_mode} not supported")
        
        # Generate samples
        batch["dt"] = torch.scalar_tensor(dt, dtype=single_repr.dtype)
        mask = batch.get("mask", torch.ones(1, batch["nres"], dtype=torch.bool, device=self.device))
        coords_mask = batch.get("coords_mask", repeat(mask, "b n -> b (n c)", c=4 if not self.ca_only else 1))
        
        x = self.generate(
            nsamples=batch["nsamples"],
            n=batch["nres"] if self.ca_only else batch["nres"] * 4,
            dt=batch["dt"],
            self_cond=self.inf_cfg.self_cond,
            cath_code=cath_code,
            guidance_weight=guidance_weight,
            autoguidance_ratio=autoguidance_ratio,
            dtype=single_repr.dtype,
            schedule_mode=self.inf_cfg.schedule.schedule_mode,
            schedule_p=self.inf_cfg.schedule.schedule_p,
            sampling_mode=sampling_args["sampling_mode"],
            sc_scale_noise=sampling_args["sc_scale_noise"],
            sc_scale_score=sampling_args["sc_scale_score"],
            gt_mode=sampling_args["gt_mode"],
            gt_p=sampling_args["gt_p"],
            gt_clamp_val=sampling_args["gt_clamp_val"],
            mask=mask,
            coords_mask=coords_mask,
            single_repr=single_repr,
        )
        
        # Predict residue types if needed
        pred_residue_type = None
        if self.inf_cfg.inv_folding:
            pred_residue_type = self.decoder.decode_residue_type(single_repr)
        
        # Format results
        results = {
            "id": batch.get("id", None),
            "pred_coords": self.samples_to_atom37(x),
            "single_repr": single_repr,
            "gt_residue_type": batch.get("residue_type", None),
            "pred_residue_type": pred_residue_type,
        }
        
        if ae_mode == "autoencode" and x_1 is not None:
            results["gt_coords"] = self.samples_to_atom37(x_1)
        else:
            results["gt_coords"] = None
            
        return results
    
    # #Debug functions
    # def on_after_backward(self) -> None:
    #     """
    #     Find unused parameters
    #     """
    #     for name, param in self.encoder.named_parameters():
    #         if param.grad is None:
    #             print(name)
    #     for name, param in self.decoder.named_parameters():
    #         if param.grad is None:
    #             print(name)
    #     return super().on_after_backward()
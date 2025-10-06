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
from typing import Dict, List, Literal
import random
from functools import partial
import math

import torch
from jaxtyping import Bool, Float
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from scipy.spatial.transform import Rotation
from torch import Tensor
from einops import rearrange, repeat

from proteinfoundation.flow_matching.r3n_fm import R3NFlowMatcher
from proteinfoundation.nn.protein_transformer import ProteinTransformerAF3, ProteinLatentTransformer
from proteinfoundation.proteinflow.model_trainer_base import ModelTrainerBase, _extract_cath_code
from proteinfoundation.utils.align_utils.align_utils import kabsch_align
from proteinfoundation.utils.coors_utils import ang_to_nm, trans_nm_to_atom37
from proteinfoundation.nn.motif_factory import SingleMotifFactory
from proteinfoundation.utils.ff_utils.pdb_utils import mask_cath_code_by_level
# from proteinfoundation.nn.dit import DiT_L_2  # Unused import

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


class ProteinLDM(ModelTrainerBase):
    """
    Protein Latent Diffusion Modeling (LDM) following the same structure as ProteinAE.
    """
    def __init__(self, cfg_exp, store_dir=None):
        super(ProteinLDM, self).__init__(cfg_exp=cfg_exp, store_dir=store_dir)
        self.save_hyperparameters()

        # Define flow matcher
       
        self.ca_only = cfg_exp.model.ca_only
        self.ae_token_dim = cfg_exp.model.ldm.encoder.get(
            "dim_latent",
            cfg_exp.model.ldm.encoder.token_dim  # No channel bottleneck
        )
        self.motif_conditioning = cfg_exp.training.get("motif_conditioning", False)
        self.fm = R3NFlowMatcher(zero_com= not self.motif_conditioning, scale_ref=1.0)  # Work in nm
        self.fm_latent = R3NFlowMatcher(
            zero_com=False,
            scale_ref=1.0,
            dim=self.ae_token_dim
        )  # Work in nm
        if self.motif_conditioning:
            self.motif_conditioning_sequence_rep = cfg_exp.training.get("motif_conditioning_sequence_rep", False)
            if self.motif_conditioning_sequence_rep:
                if "motif_sequence_mask" not in cfg_exp.model.nn.feats_init_seq:
                    cfg_exp.model.nn.feats_init_seq.append("motif_sequence_mask")
                if "motif_x1" not in cfg_exp.model.nn.feats_init_seq:
                    cfg_exp.model.nn.feats_init_seq.append("motif_x1")
                
            if "motif_structure_mask" not in cfg_exp.model.nn.feats_pair_repr:
                cfg_exp.model.nn.feats_pair_repr.append("motif_structure_mask")
            if "motif_x1_pair_dists" not in cfg_exp.model.nn.feats_pair_repr:
                cfg_exp.model.nn.feats_pair_repr.append("motif_x1_pair_dists")
            self.motif_factory = SingleMotifFactory(motif_prob=cfg_exp.training.get("motif_prob", 1.0))

        # Neural network
        self.encoder = ProteinTransformerAF3(
            **cfg_exp.model.ldm.encoder,
            ca_only=self.ca_only
        )
        # Align with original model
        self.decoder = ProteinTransformerAF3(
            **cfg_exp.model.ldm.decoder,
            ca_only=self.ca_only,
        )
        self.ldm = ProteinLatentTransformer(**cfg_exp.model.ldm.latent, ae_token_dim=self.ae_token_dim)
        
        # Calculate parameters
        self.nparams = self._calculate_model_parameters()

    def load_autoencoder_weights(self, cfg_exp):
        """Load pretrained weights for encoder and decoder."""
        assert cfg_exp.model.pretrained_weights is not None, "Pretrained weights are required for ProteinLDM"
        print(f"Loading pretrained weights from {cfg_exp.model.pretrained_weights}")
        
        state_dict = torch.load(
            cfg_exp.model.pretrained_weights,
            map_location="cpu",
            weights_only=False
        )["state_dict"]
        encoder_state_dict = {
            k[len("encoder."):]: v
            for k, v in state_dict.items() 
            if k.startswith("encoder.")
        }
        nn_state_dict = {
            k[len("decoder."):]: v 
            for k, v in state_dict.items() 
            if k.startswith("decoder.")
        }
        
        self.encoder.load_state_dict(encoder_state_dict)
        self.decoder.load_state_dict(nn_state_dict)
        
        # Set models to evaluation mode and freeze parameters
        self.encoder.eval()
        self.decoder.eval()
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

    def _calculate_model_parameters(self):
        """Calculate total number of trainable parameters."""
        return (
            sum(p.numel() for p in self.encoder.parameters() if p.requires_grad) +
            sum(p.numel() for p in self.decoder.parameters() if p.requires_grad) +
            sum(p.numel() for p in self.ldm.parameters() if p.requires_grad)
        )
        
    def _nn_out_to_x_clean_latent(self, nn_out, batch):
        """
        Transforms the output of the nn to a clean sample prediction. The transformation depends on the
        parameterization used. For now we admit x_1 or v.

        Args:
            nn_out: Dictionary, nerual network output
                - "coords_pred": Tensor of shape [b, n, 3], could be the clean sample or the velocity
                - "pair_pred" (Optional): Tensor of shape [b, n, n, num_buckets_predict_pair], could be the clean sample or the velocity
            batch: Dictionary, batch of data

        Returns:
            Clean sample prediction, tensor of shape [b, n, 3].
        """
        nn_pred = nn_out["single_repr_pred"]
        t = batch["t"]  # [*]
        t_ext = t[..., None, None]  # [*, 1, 1]
        x_t = batch["x_t"]  # [*, n, 3]
        if self.cfg_exp.model.target_pred == "x_1":
            x_1_pred = nn_pred
        elif self.cfg_exp.model.target_pred == "v":
            x_1_pred = x_t + (1.0 - t_ext) * nn_pred
        else:
            raise IOError(
                f"Wrong parameterization chosen: {self.cfg_exp.model.target_pred}"
            )
        return x_1_pred

    def predict_clean_latent(
        self,
        batch: Dict,
    ):
        """
        Predicts clean samples given noisy ones and time.

        Args:
            batch: a batch of data with some additions, including
                - "x_t": Type depends on the mode (see beluw, "returns" part)
                - "t": Time, shape [*]
                - "mask": Binary mask of shape [*, n]
                - "x_sc" (optional): Prediction for self-conditioning
                - Other features from the dataloader.

        Returns:
            Predicted clean sample, depends on the "modality" we're in.
                - For frameflow it returns a dictionary with keys "trans" and "rot", and values
                tensors of shape [*, n, 3] and [*, n, 3, 3] respectively,
                - For CAflow it returns a tensor of shape [*, n, 3].
            Other things predicted by nn (pair_pred for distogram loss)
        """
        nn_out = self.ldm(batch)
        # nn_out = self.latent(x=batch["x_t"], t=batch["t"], y=None)  # [*, n, 3]
        return self._nn_out_to_x_clean_latent(nn_out, batch), nn_out  # [*, n, 3]

    def predict_clean_n_v_w_guidance_latent(
        self,
        batch: Dict,
        guidance_weight: float = 1.0,
        autoguidance_ratio: float = 0.0,
    ):
        """
        Logic for CFG and autoguidance goes here. This computes a clean sample prediction (can be single thing, tuple, etc)
        and the corresponding vector field used to initialize.

        Here if we want to do the different self conditioning for cond / ucond, ag / no ag, we can just return tuples of x_pred and
        modify the batches accordingly every time we call predict clean.

        w: guidance weight
        alpha: autoguidance ratio
        x_pred = w * x_pred + (1 - alpha) * (1 - w) * x_pred_uncond + alpha * (1 - w) * x_pred_auto_guidance

        WARNING: The ag checkpoint needs to rely on the same parameterization of the main model. This can be changed after training
        so no big deal but just in case leaving a note.
        """
        if self.motif_conditioning and ("fixed_structure_mask" not in batch or "x_motif" not in batch):
            batch.update(self.motif_factory(batch, zeroes = True))  # for generation we have to pass conditioning info in. But for validation do the same as training

        nn_out = self.ldm(batch)
        # nn_out = self.latent(x=batch["x_t"], t=batch["t"], y=None)
        x_pred = self._nn_out_to_x_clean_latent(nn_out, batch)

        if guidance_weight != 1.0:
            assert autoguidance_ratio >= 0.0 and autoguidance_ratio <= 1.0
            if autoguidance_ratio > 0.0:  # Use auto-guidance
                nn_out_ag = self.nn_ag(batch)
                x_pred_ag = self._nn_out_to_x_clean_latent(nn_out_ag, batch)
            else:
                x_pred_ag = torch.zeros_like(x_pred)

            if autoguidance_ratio < 1.0:  # Use CFG
                assert (
                    "cath_code" in batch
                ), "Only support CFG when cath_code is provided"
                uncond_batch = batch.copy()
                uncond_batch.pop("cath_code")
                nn_out_uncond = self.ldm(uncond_batch)
                x_pred_uncond = self._nn_out_to_x_clean_latent(nn_out_uncond, uncond_batch)
            else:
                x_pred_uncond = torch.zeros_like(x_pred)

            x_pred = guidance_weight * x_pred + (1 - guidance_weight) * (
                autoguidance_ratio * x_pred_ag
                + (1 - autoguidance_ratio) * x_pred_uncond
            )

        v = self.fm_latent.xt_dot(x_pred, batch["x_t"], batch["t"], batch["mask"])
        return x_pred, v

    def align_wrapper(self, x_0, x_1, mask):
        """Performs Kabsch on the translation component of x_0 and x_1."""
        return kabsch_align(mobile=x_0, target=x_1, mask=mask)

    def extract_clean_sample(self, batch):
        """
        Extracts clean sample, mask, batch size, protein length n, and dtype from the batch.
        Applies augmentations if those are required.

        Args:
            batch: batch from dataloader.

        Returns:
            Tuple (x_1, mask, coords_mask, batch_shape, n, dtype)
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
        return self.fm_latent._mask_and_zero_com(x_rot, mask), mask
    
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
        self._prepare_batch_data_latent(batch)
        
        # Encode and prepare for flow matching
        single_repr = self._encode_and_prepare_flow_matching_latent(batch)
        
        # Apply motif conditioning if enabled
        self._apply_motif_conditioning(batch)
        
        # Apply fold conditional training
        self._apply_fold_conditional_training(batch)
        
        # Self-conditioning prediction
        self._apply_self_conditioning_latent(batch)
        
        # Main prediction
        single_repr_pred, nn_out = self.predict_clean_latent(batch)

        # Compute and log losses
        train_loss = self._compute_all_losses_latent(
            batch, single_repr, single_repr_pred, nn_out, log_prefix
        )
        
        # Logging
        self._log_training_metrics_latent(train_loss, batch, log_prefix, val_step)

        return train_loss

    def _prepare_batch_data_latent(self, batch):
        """Prepare batch data by extracting and processing coordinates."""
        x_1, mask, coords_mask, batch_shape, n, dtype = self.extract_clean_sample(batch)
        x_1 = self.fm_latent._mask_and_zero_com(x_1, coords_mask)
        batch.update({
            "x_1": x_1,
            "mask": mask,
            "coords_mask": coords_mask,
            "batch_shape": batch_shape,
            "n": n,
            "dtype": dtype
        })

    def _encode_and_prepare_flow_matching_latent(self, batch):
        """Encode input and prepare flow matching interpolation."""
        # Encode
        with torch.no_grad():
            single_repr = self.encoder(batch).get("single_repr", None)
            assert single_repr is not None, "Single rep is None"
        single_repr = self.detach_gradients(single_repr)
        batch["single_repr"] = single_repr
        
        # Sample time, reference and align reference to target
        t = self.sample_t(batch["batch_shape"])
        mask = batch["mask"]
        
        # Adjust mask if needed for latent space
        if single_repr.shape[1] != mask.shape[1]:
            mask = torch.ones(
                *batch["batch_shape"],
                single_repr.shape[1],
                device=self.device,
                dtype=torch.bool,
            )
            batch["mask"] = mask
            
        x_0 = self.fm_latent.sample_reference(
            n=single_repr.shape[1],
            shape=batch["batch_shape"],
            device=self.device,
            dtype=batch["dtype"],
            mask=mask,
        )
        
        # Interpolation
        x_t = self.fm_latent.interpolate(x_0, single_repr, t, mask)
        batch.update({"t": t, "x_t": x_t})
        return single_repr

    def _apply_motif_conditioning(self, batch):
        """Apply motif conditioning if enabled."""
        if self.motif_conditioning:
            batch.update(self.motif_factory(batch))
            # Update x_1 based on motif center changes
            batch["x_1"] = batch["x_1"]

    def _apply_fold_conditional_training(self, batch):
        """Apply fold conditional training."""
        if self.cfg_exp.training.fold_cond:
            bs = batch["x_1"].shape[0]
            cath_code_list = batch.cath_code
            for i in range(bs):
                # Progressively mask T, A, C levels
                cath_code_list[i] = mask_cath_code_by_level(
                    cath_code_list[i], level="H"
                )
                if random.random() < self.cfg_exp.training.mask_T_prob:
                    cath_code_list[i] = mask_cath_code_by_level(
                        cath_code_list[i], level="T"
                    )
                    if random.random() < self.cfg_exp.training.mask_A_prob:
                        cath_code_list[i] = mask_cath_code_by_level(
                            cath_code_list[i], level="A"
                        )
                        if random.random() < self.cfg_exp.training.mask_C_prob:
                            cath_code_list[i] = mask_cath_code_by_level(
                                cath_code_list[i], level="C"
                            )
            batch.cath_code = cath_code_list
        else:
            if "cath_code" in batch:
                batch.pop("cath_code")

    def _apply_self_conditioning_latent(self, batch):
        """Apply self-conditioning if enabled."""
        if random.random() > 0.5 and self.cfg_exp.training.self_cond:
            x_pred_sc_latent, _ = self.predict_clean_latent(batch)
            batch["x_sc_latent"] = self.detach_gradients(x_pred_sc_latent)

    def _compute_all_losses_latent(self, batch, single_repr, single_repr_pred, nn_out, log_prefix):
        """Compute all loss components."""
        x_t, t = batch["x_t"], batch["t"]
        mask = batch["mask"]
        
        # Flow matching loss
        fm_loss = self.compute_fm_loss_latent(
            single_repr, single_repr_pred, x_t, t, mask, log_prefix=log_prefix
        )
        train_loss = torch.mean(fm_loss)
        
        return train_loss

    def _log_training_metrics_latent(self, train_loss, batch, log_prefix, val_step):
        """Log training metrics and scaling information."""
        mask = batch["mask"]
        
        self._log_metric(f"{log_prefix}/loss", train_loss, mask.shape[0])

        if not val_step:
            self._log_metric("train_loss", train_loss, mask.shape[0], prog_bar=True)
            self._log_scaling_metrics_latent(mask)

    def _log_scaling_metrics_latent(self, mask):
        """Log scaling law metrics."""
        b, n = mask.shape
        nflops_step = None  # TODO: Implement if needed
        
        if nflops_step is not None:
            self.nflops = self.nflops + \
            nflops_step * self.trainer.world_size
            self._log_metric(
                "scaling/nflops",
                self.nflops * 1.0,
                batch_size=1,
                on_epoch=False
            )

        self.nsamples_processed = self.nsamples_processed + \
            b * self.trainer.world_size
        self._log_metric(
            "scaling/nsamples_processed",
            self.nsamples_processed * 1.0,
            batch_size=1,
            on_epoch=False
        )

        self._log_metric(
            "scaling/nparams",
            self.nparams * 1.0,
            batch_size=1,
            on_epoch=False
        )

    def compute_loss_weight(
        self, t: Float[Tensor, "*"], eps: float = 1e-3
    ) -> Float[Tensor, "*"]:
        t = t.clamp(min=eps, max=1.0 - eps)  # For safety
        return t / (
            1.0 - t
        )

    def compute_fm_loss_latent(
        self,
        single_repr: Float[Tensor, "* n token_dim"],
        single_repr_pred: Float[Tensor, "* n token_dim"],
        x_t: Float[Tensor, "* n token_dim"],
        t: Float[Tensor, "*"],
        mask: Bool[Tensor, "* nres"],
        log_prefix: str,
    ) -> Float[Tensor, "*"]:
        """
        Computes and logs flow matching loss.

        Args:
            x_1: True clean sample, shape [*, n, token_dim].
            x_1_pred: Predicted clean sample, shape [*, n, token_dim].
            x_t: Sample at interpolation time t (used as input to predict clean sample), shape [*, n, token_dim].
            t: Interpolation time, shape [*].
            mask: Boolean residue mask, shape [*, nres].

        Returns:
            Flow matching loss.
        """
        nlatent = torch.sum(mask, dim=-1) * self.ae_token_dim  # [*]
        assert single_repr.shape == single_repr_pred.shape == (
            *mask.shape,
            self.ae_token_dim,
        ), f"Shape mismatch: {single_repr.shape} != {single_repr_pred.shape} != {mask.shape}"

        err = (single_repr - single_repr_pred) * mask[..., None]  # [*, n, token_dim]
        loss = torch.sum(err**2, dim=(-1, -2)) / nlatent  # [*]

        total_loss_w = 1.0 / ((1.0 - t) ** 2 + 1e-5)

        loss = loss * total_loss_w  # [*]
        if log_prefix:
            self._log_metric(
                f"{log_prefix}/trans_loss",
                torch.mean(loss),
                mask.shape[0],
                prog_bar=True,
            )
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
    
    def generate_latent(
        self,
        nsamples: int,
        n: int,
        dt: float,
        self_cond: bool,
        cath_code: List[List[str]],
        guidance_weight: float = 1.0,
        autoguidance_ratio: float = 0.0,
        dtype: torch.dtype = None,
        schedule_mode: str = "uniform",
        schedule_p: float = 1.0,
        sampling_mode: str = "sc",
        sc_scale_noise: float = "1.0",
        sc_scale_score: float = "1.0",
        gt_mode: Literal["us", "tan"] = "us",
        gt_p: float = 1.0,
        gt_clamp_val: float = None,
        mask = None,
        coords_mask = None,
        x_motif = None,
        fixed_sequence_mask = None,
        fixed_structure_mask = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Generates samples by integrating ODE with learned vector field.
        """
        predict_clean_n_v_w_guidance_latent = partial(
            self.predict_clean_n_v_w_guidance_latent,
            guidance_weight=guidance_weight,
            autoguidance_ratio=autoguidance_ratio,
        )
        if mask is None:
            mask = torch.ones(nsamples, n).long().bool().to(self.device)
        if coords_mask is None:
            coords_mask = mask.clone()
        return self.fm_latent.full_simulation(
            predict_clean_n_v_w_guidance_latent,
            dt=dt,
            nsamples=nsamples,
            n=n,
            self_cond=self_cond,
            cath_code=cath_code,
            device=self.device,
            mask=mask,
            coords_mask=coords_mask,
            dtype=dtype,
            schedule_mode=schedule_mode,
            schedule_p=schedule_p,
            sampling_mode=sampling_mode,
            sc_scale_noise=sc_scale_noise,
            sc_scale_score=sc_scale_score,
            gt_mode=gt_mode,
            gt_p=gt_p,
            gt_clamp_val=gt_clamp_val,
            x_motif = x_motif,
            fixed_sequence_mask = fixed_sequence_mask,
            fixed_structure_mask = fixed_structure_mask,
            single_repr = None,
        )
    
    def predict_step(self, batch, batch_idx):
        """
        Makes predictions. Should call set_inf_cfg before calling this.

        Args:
            batch: data batch, contains no data, but the info of the samples
                to generate (nsamples, nres, dt)

        Returns:
            Samples generated in atom 37 format.
        """
        sampling_args_latent = self.inf_cfg.sampling_latent_flow

        cath_code = (
            _extract_cath_code(batch) if self.inf_cfg.get("fold_cond", False) else None
        )  # When using unconditional model, don't use cath_code
        guidance_weight = self.inf_cfg.get("guidance_weight", 1.0)
        autoguidance_ratio = self.inf_cfg.get("autoguidance_ratio", 0.0)
        
        mask = batch['mask'].squeeze(0) if 'mask' in batch else None
        if 'motif_seq_mask' in batch:
            fixed_sequence_mask = batch['motif_seq_mask'].squeeze(0).to(self.device)
            x_motif = batch['motif_structure'].squeeze(0).to(self.device)
            fixed_structure_mask = fixed_sequence_mask[:, :, None] * fixed_sequence_mask[:, None, :]
        else:
            fixed_sequence_mask, x_motif, fixed_structure_mask = None, None, None
            fixed_sequence_mask = None
        
        downsample_ratio = sampling_args_latent.get("downsample_ratio", 1)
        if downsample_ratio > 1:
            num_sampling_blocks = int(math.log(downsample_ratio, 2))
            downsample_conv = torch.nn.ModuleList()
            for _ in range(num_sampling_blocks):
                downsample_conv.append(
                    torch.nn.Conv1d(
                        in_channels=128,
                        out_channels=128,
                        kernel_size=3,
                        stride=2,
                    )
                )
            downsample_conv = torch.nn.Sequential(*downsample_conv)
            x_fake = torch.randn(batch["nsamples"], batch["nres"], 128)
            x_down = x_fake.mT
            for conv in downsample_conv:
                x_down = conv(x_down)
            x_down = x_down.mT
            n_latent = x_down.shape[1]
        
        single_repr = self.generate_latent(
            nsamples=batch["nsamples"],
            n=batch["nres"] if downsample_ratio == 1 else n_latent,
            dt=batch["dt_latent"].to(dtype=torch.float32),
            self_cond=self.inf_cfg.self_cond,
            cath_code=cath_code,
            guidance_weight=guidance_weight,
            autoguidance_ratio=autoguidance_ratio,
            dtype=torch.float32,
            schedule_mode=self.inf_cfg.schedule.schedule_mode,
            schedule_p=self.inf_cfg.schedule.schedule_p,
            sampling_mode=sampling_args_latent["sampling_mode"],
            sc_scale_noise=sampling_args_latent["sc_scale_noise"],
            sc_scale_score=sampling_args_latent["sc_scale_score"],
            gt_mode=sampling_args_latent["gt_mode"],
            gt_p=sampling_args_latent["gt_p"],
            gt_clamp_val=sampling_args_latent["gt_clamp_val"],
            mask=mask,
            x_motif=x_motif,
            fixed_sequence_mask=fixed_sequence_mask,
            fixed_structure_mask=fixed_structure_mask,
        )
        
        sampling_args = self.inf_cfg.sampling_bbflow
        
        x = self.generate(
            nsamples=batch["nsamples"],
            n=batch["nres"],
            dt=batch["dt"].to(dtype=torch.float32),
            self_cond=self.inf_cfg.self_cond,
            cath_code=cath_code,
            guidance_weight=guidance_weight,
            autoguidance_ratio=autoguidance_ratio,
            dtype=torch.float32,
            schedule_mode=self.inf_cfg.schedule.schedule_mode,
            schedule_p=self.inf_cfg.schedule.schedule_p,
            sampling_mode=sampling_args["sampling_mode"],
            sc_scale_noise=sampling_args["sc_scale_noise"],
            sc_scale_score=sampling_args["sc_scale_score"],
            gt_mode=sampling_args["gt_mode"],
            gt_p=sampling_args["gt_p"],
            gt_clamp_val=sampling_args["gt_clamp_val"],
            mask=mask,
            x_motif=x_motif,
            fixed_sequence_mask=fixed_sequence_mask,
            fixed_structure_mask=fixed_structure_mask,
            single_repr=single_repr,
        )

        return {
            "id": batch.get("id", None),
            "pred_coords": self.samples_to_atom37(x),  # [b, n, 37, 3]
            "single_repr": single_repr,
        }

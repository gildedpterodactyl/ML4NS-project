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
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm
import multiprocessing as mp

from openfold.np.residue_constants import resname_to_idx
from proteinfoundation.datasets.base_data import BaseLightningDataModule
from proteinfoundation.utils.cluster_utils import (
    cluster_sequences,
    df_to_fasta,
    expand_cluster_splits,
    fasta_to_df,
    read_cluster_tsv,
    setup_clustering_file_paths,
    split_dataframe,
)
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR

from graphein_utils.graphein_utils import (
    protein_to_pyg, 
    PDBManager,     
    download_pdb_multiprocessing,
)


class LatentDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        pdb_codes: Optional[List[str]] = None,
    ):
        self.data_dir = data_dir
        if pdb_codes is None:
            self.pdb_codes = [pdb_code.stem for pdb_code in Path(data_dir).glob("*.pt")][:2]
        else:
            self.pdb_codes = [pdb_code.lower() for pdb_code in pdb_codes]

    def __len__(self):
        return len(self.pdb_codes)

    def __getitem__(self, idx: int):
        pdb_code = self.pdb_codes[idx]
        latent = torch.load(self.data_dir / f"{pdb_code}.pt")
        mask = torch.ones(latent.shape[0], dtype=torch.bool)
        data = {
            "single_repr": latent,
            "mask": mask,
        }
        return data


class LatentDataModule(BaseLightningDataModule):
    def __init__(
        self,
        data_dir: str,
        pdb_codes: Optional[List[str]] = None,
        # arguments for BaseLightningDataModule
        batch_padding: bool = True,
        sampling_mode: Literal["random", "cluster-random", "cluster-reps"] = "random",
        transforms: Optional[List[Callable]] = None,
        pre_transforms: Optional[List[Callable]] = None,
        pre_filters: Optional[List[Callable]] = None,
        batch_size: int = 32,
        num_workers: int = 32,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__(
            batch_padding=batch_padding,
            sampling_mode=sampling_mode,
            transforms=transforms,
            pre_transforms=pre_transforms,
            pre_filters=pre_filters,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.data_dir = data_dir
        self.pdb_codes = pdb_codes
        
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_ds = self.train_dataset()
            self.val_ds = self.val_dataset()
        elif stage == "test":
            self.test_ds = self.test_dataset()

    def _get_dataset(self, split: Literal["train", "val", "test"]) -> LatentDataset:
        return LatentDataset(Path(self.data_dir) / split, self.pdb_codes)

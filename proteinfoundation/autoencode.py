"""
Protein Autoencoder - For protein structure encoding, decoding and autoencoding processes
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import lightning as L
from torch_geometric.data import Data
from einops import rearrange
from omegaconf import DictConfig
from loguru import logger
from dotenv import load_dotenv
import hydra
from openfold.np import residue_constants

# Project-specific imports
root = os.path.abspath(".")
sys.path.append(root)

from graphein_utils.graphein_utils import protein_to_pyg
from openfold.np.residue_constants import resname_to_idx
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR
from proteinfoundation.utils.dense_padding_data_loader import DensePaddingDataLoader
from proteinfoundation.proteinflow.proteinae import ProteinAE
from proteinfoundation.utils.coors_utils import ang_to_nm
from proteinfoundation.flow_matching.r3n_fm import R3NFlowMatcher
from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb
from proteinfoundation.metrics.designability import rmsd_metric, run_and_store_esm, load_pdb

class ProteinProcessor:
    """Protein structure processor"""
    
    # Backbone atom indices [N, CA, C, O]
    BACKBONE_ATOM_INDICES = [0, 1, 2, 4]
    
    def __init__(self, fill_value_coords: float = 1e-5):
        self.fill_value_coords = fill_value_coords
    
    def process_pdb(self, pdb_fpath: Path, chains: str = "all") -> Data:
        """
        Process PDB file and convert to PyTorch Geometric graph data
        
        Args:
            pdb_fpath: Path to PDB file
            chains: Chains to process, default is "all"
            
        Returns:
            Processed graph data
        """
        graph = protein_to_pyg(
            path=pdb_fpath,
            chain_selection=chains,
            keep_insertions=True,
            store_het=False,
            store_bfactor=True,
            fill_value_coords=self.fill_value_coords,
        )
        
        # Set basic properties
        graph.id = pdb_fpath.stem
        graph.database = "pdb"
        
        # Process coordinate mask
        coord_mask = graph.coords != self.fill_value_coords
        graph.coord_mask = coord_mask[..., 0]
        
        # Process residue types
        graph.residue_type = torch.tensor(
            [resname_to_idx[residue] for residue in graph.residues]
        ).long()
        
        # Process B-factors
        graph.bfactor_avg = torch.mean(graph.bfactor, dim=-1)
        
        # Process residue indices
        graph.residue_pdb_idx = torch.tensor(
            [int(s.split(":")[2]) for s in graph.residue_id], 
            dtype=torch.long
        )
        
        # Set sequence positions
        graph.seq_pos = torch.arange(graph.coords.shape[0]).unsqueeze(-1)
        
        # Reorder coordinates: convert from PDB format to OpenFold format
        graph.coords = graph.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
        graph.coord_mask = graph.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]
        
        return graph


class ProteinAutoEncoder:
    """Protein autoencoder wrapper class"""
    
    def __init__(self, model: ProteinAE, trainer: L.Trainer):
        self.model = model
        self.trainer = trainer
        self.processor = ProteinProcessor()
        self.device = next(model.parameters()).device
        
    @classmethod
    def from_checkpoint(cls, cfg: DictConfig, ckpt_file: Path) -> 'ProteinAutoEncoder':
        """Load model from checkpoint"""
        model = ProteinAE.load_from_checkpoint(str(ckpt_file), strict=True)
        trainer = L.Trainer(accelerator="gpu", devices=1, logger=False, enable_progress_bar=False)
        return cls(model, trainer)
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch data to device"""
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        return batch
    
    def _extract_backbone_coordinates(self, batch: Dict[str, Any]) -> Tuple:
        """Extract backbone coordinate information"""
        # Extract backbone atom coordinates [N, CA, C, O]
        x_1 = batch["coords"][:, :, self.processor.BACKBONE_ATOM_INDICES, :]  # [b, n, 4, 3]
        x_1 = rearrange(x_1, "b n c d -> b (n c) d")  # [b, 4*n, 3]
        
        # Process coordinate masks
        coords_mask = batch["mask_dict"]["coords"][..., self.processor.BACKBONE_ATOM_INDICES, 0]  # [b, n, 4]
        mask = coords_mask[..., 1]  # CA atom mask
        coords_mask = rearrange(coords_mask, "b n c -> b (n c)")  # [b, 4*n]
        
        batch_shape = x_1.shape[:-2]
        n = x_1.shape[-2]
        
        return ang_to_nm(x_1), mask, coords_mask, batch_shape, n, x_1.dtype
    
    def autoencode(self, pdb_fpath: Path, cfg: DictConfig) -> List[Dict]:
        """
        Autoencoding: input PDB file, output reconstructed structure
        
        Args:
            pdb_fpath: Path to PDB file
            cfg: Configuration parameters
            
        Returns:
            List of prediction results
        """
        cfg.ae_mode = "autoencode"
        self.model.configure_inference(cfg, nn_ag=None)
        
        item = self.processor.process_pdb(pdb_fpath)
        loader = DensePaddingDataLoader([item])
        predictions = self.trainer.predict(self.model, loader)
        
        return predictions
    
    def encode(self, pdb_fpath: Path) -> torch.Tensor:
        """
        Encoding: encode PDB file into latent representation
        
        Args:
            pdb_fpath: Path to PDB file
            
        Returns:
            Single representation tensor
        """
        # Initialize flow matcher
        fm = R3NFlowMatcher(zero_com=True, scale_ref=1.0)
        
        # Process input data
        item = self.processor.process_pdb(pdb_fpath)
        loader = DensePaddingDataLoader([item])
        batch = next(iter(loader))
        
        # Extract backbone coordinates
        x_1, mask, coords_mask, batch_shape, n, dtype = self._extract_backbone_coordinates(batch)
        
        # Center and apply mask
        x_1 = fm._mask_and_zero_com(x_1, coords_mask)
        
        # Prepare batch data
        batch.update({
            "x_1": x_1,
            "mask": mask,
            "coords_mask": coords_mask,
            "nsamples": 1,
            "nres": int(n // 4),
        })
        
        # Move to device and encode
        batch = self._move_to_device(batch)
        single_repr = self.model.encoder(batch).get("single_repr", None)
        
        return single_repr
    
    def decode(self, single_repr: torch.Tensor, cfg: DictConfig) -> List[Dict]:
        """
        Decoding: decode latent representation into protein structure
        
        Args:
            single_repr: Single representation tensor
            cfg: Configuration parameters
            
        Returns:
            List of prediction results
        """
        cfg.ae_mode = "decode"
        self.model.configure_inference(cfg, nn_ag=None)
        
        n_length = single_repr.shape[0]  # TODO: need to consider downsampling ratio
        
        item = {
            "single_repr": single_repr,
            "mask": torch.ones(n_length, dtype=torch.bool)
        }
        
        loader = DensePaddingDataLoader([item])
        predictions = self.trainer.predict(self.model, loader)
        
        return predictions
    
    def get_residue_type(self, predictions: List[Dict]) -> List[str]:
        """
        Get residue types from predictions
        """
        restypes = residue_constants.restypes + ["X"]
        return [restypes[r.item()] for r in predictions]


class OutputWriter:
    """Output file writer"""
    
    @staticmethod
    def write_predictions(predictions: List[Dict], save_dir: str) -> None:
        """
        Write prediction results to PDB files
        
        Args:
            predictions: List of prediction results
            save_dir: Save directory
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for i, pred in enumerate(predictions):
            pred_coords = pred["pred_coords"]  # [b, n, 37, 3]
            gt_coords = pred.get("gt_coords", None)
            
            # Save PDB file for each generated structure
            for j in range(pred_coords.shape[0]):
                # Save predicted structure
                pred_pdb_path = os.path.join(save_dir, "sample.pdb")
                write_prot_to_pdb(
                    pred_coords[j].numpy(),
                    pred_pdb_path,
                    overwrite=True,
                    no_indexing=True,
                )
                
                # Also save ground truth structure if available
                if gt_coords is not None:
                    gt_pdb_path = os.path.join(save_dir, "gt.pdb")
                    write_prot_to_pdb(
                        gt_coords[j].numpy(),
                        gt_pdb_path,
                        overwrite=True,
                        no_indexing=True,
                    )
                    # RMSD calculation
                    rmsd = rmsd_metric(pred_coords[j], gt_coords[j])
                    logger.info(f"RMSD: {rmsd}")


def setup_logging() -> None:
    """Setup logging configuration"""
    # Remove the default handler to avoid duplication
    logger.remove()
    
    # Add a sxingle handler with the desired format, preserving colorization
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
        colorize=True,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Protein Autoencoder - Encode and decode protein structures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input_pdb", 
        type=str, 
        required=True,
        help="Path to input PDB file"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Directory to save output files"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["encode", "decode", "autoencode"],
        default="autoencode",
        help="Processing mode: encode only, decode only, or full autoencoding"
    )
    
    parser.add_argument(
        "--chains",
        type=str,
        default="all",
        help="Protein chains to process (default: all chains)"
    )
    
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/shaoningli5190/ProteinAE/configs",
        help="Path to configuration directory"
    )
    
    parser.add_argument(
        "--config_name",
        type=str,
        default="inference_proteinae",
        help="Name of the configuration file"
    )
    
    return parser.parse_args()


def load_config(config_path: str, config_name: str) -> DictConfig:
    """Load configuration file"""
    version_base = hydra.__version__
    
    with hydra.initialize_config_dir(
        config_dir=f"{config_path}/experiment_config", 
        version_base=version_base
    ):
        cfg = hydra.compose(
            config_name=config_name,
            return_hydra_config=True,
        )
    return cfg


def calc_scRMSD(
    pred_seq: str,
    pdb_output_dir: str,
) -> float:
    """
    Calculate self-consistency RMSD
    """
    name = pdb_output_dir.split("/")[-1]
    logger.info(f"Running ESMFold for {name}")
    out_esm_paths = run_and_store_esm(name, [pred_seq], pdb_output_dir)
    gen_prot = load_pdb(os.path.join(pdb_output_dir, "sample.pdb"))
    results = []
    for out_esm in out_esm_paths:
        rec_prot_esm = load_pdb(out_esm)
        gen_coors = torch.Tensor(gen_prot.atom_positions)
        rec_coors = torch.Tensor(rec_prot_esm.atom_positions)
        results.append(rmsd_metric(gen_coors, rec_coors))
    return min(results)

def process_protein(
    autoencoder: ProteinAutoEncoder, 
    pdb_path: Path, 
    output_dir: Path, 
    cfg: DictConfig,
    mode: str = "autoencode",
    chains: str = "all"
) -> None:
    """
    Process protein structure based on specified mode
    
    Args:
        autoencoder: Protein autoencoder instance
        pdb_path: Path to input PDB file
        output_dir: Output directory
        cfg: Configuration parameters
        mode: Processing mode ('encode', 'decode', 'autoencode')
        chains: Protein chains to process
    """
    logger.info(f"Processing PDB file: {pdb_path}")
    logger.info(f"Processing mode: {mode}")
    logger.info(f"Processing chains: {chains}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if mode == "autoencode":
        # Full autoencoding pipeline
        predictions = autoencoder.autoencode(pdb_path, cfg)
        if cfg.inv_folding:
            pred_seq = "".join(autoencoder.get_residue_type(predictions[0]["pred_residue_type"][0].cpu()))
            gt_seq = "".join(autoencoder.get_residue_type(predictions[0]["gt_residue_type"][0].cpu()))
            logger.info(f"Predicted sequence: {pred_seq}")
            logger.info(f"Ground truth sequence: {gt_seq}")
        OutputWriter.write_predictions(predictions, str(output_dir))
        logger.info("Autoencoding completed")
        
        if cfg.inv_folding:
            scrmsd = calc_scRMSD(pred_seq, str(output_dir))
            logger.info(f"scRMSD: {scrmsd}")
        
    elif mode == "encode":
        # Encode only
        single_repr = autoencoder.encode(pdb_path).squeeze().cpu()
        logger.info(f"Encoding completed")
        
        # Save encoded representation
        repr_path = output_dir / "latent_repr.pt"
        torch.save(single_repr, repr_path)
        logger.info(f"Encoded representation saved to: {repr_path}")
        
    elif mode == "decode":
        # Decode mode requires a pre-encoded representation
        repr_path = pdb_path / "latent_repr.pt"
        if not repr_path.exists():
            raise FileNotFoundError(
                f"Encoded representation not found at {repr_path}. "
                "Please run encode mode first or provide the encoded representation file."
            )
        
        single_repr = torch.load(repr_path)
        logger.info(f"Loaded representation with shape: {single_repr.shape}")
        
        predictions = autoencoder.decode(single_repr, cfg)
        OutputWriter.write_predictions(predictions, str(output_dir))
        logger.info("Decoding completed")
    
    logger.info(f"Results saved to: {output_dir}")


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Environment setup
    load_dotenv()
    setup_logging()
    L.seed_everything(43)
    
    # Validate input file
    pdb_path = Path(args.input_pdb)
    if not pdb_path.exists():
        logger.error(f"Input PDB file not found: {pdb_path}")
        sys.exit(1)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    
    # Load configuration
    cfg = load_config(args.config_path, args.config_name)
    
    # Prepare model
    ckpt_file = Path(cfg.ckpt_path) / cfg.ckpt_name
    if not ckpt_file.exists():
        logger.error(f"Checkpoint file not found: {ckpt_file}")
        sys.exit(1)
    
    logger.info(f"Using checkpoint: {ckpt_file}")
    autoencoder = ProteinAutoEncoder.from_checkpoint(cfg, ckpt_file)
    
    # Process protein
    process_protein(
            autoencoder=autoencoder,
            pdb_path=pdb_path,
            output_dir=output_dir / pdb_path.stem,
            cfg=cfg,
            mode=args.mode,
            chains=args.chains
        )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
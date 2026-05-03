import argparse
import hashlib
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

SEQ2IND = {
    "I": 0,
    "L": 1,
    "V": 2,
    "F": 3,
    "M": 4,
    "C": 5,
    "A": 6,
    "G": 7,
    "P": 8,
    "T": 9,
    "S": 10,
    "Y": 11,
    "W": 12,
    "Q": 13,
    "N": 14,
    "H": 15,
    "E": 16,
    "D": 17,
    "K": 18,
    "R": 19,
    "X": 20,
    "J": 21,
    "*": 22,
    "-": 23,
}
IND2SEQ = {v: k for k, v in SEQ2IND.items()}


def seq_hash(seq: str) -> str:
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()[:12]


def seq_to_inds(seq: str, seq_len: int) -> torch.Tensor:
    clipped = seq[:seq_len]
    if len(clipped) < seq_len:
        clipped = clipped + ("J" * (seq_len - len(clipped)))
    return torch.tensor([SEQ2IND.get(ch, SEQ2IND["X"]) for ch in clipped], dtype=torch.long)


def inds_to_seq(indices: torch.Tensor) -> str:
    return "".join(IND2SEQ[int(i)] for i in indices.tolist())


def clean_seq(seq: str) -> str:
    return seq.replace("J", "").replace("*", "").replace("-", "")


def identity(seq1: str, seq2: str) -> float:
    n = min(len(seq1), len(seq2))
    if n == 0:
        return 0.0
    return float(sum(1 for a, b in zip(seq1[:n], seq2[:n]) if a == b) / n)


def hamming(seq1: str, seq2: str) -> int:
    n = min(len(seq1), len(seq2))
    return int(sum(1 for a, b in zip(seq1[:n], seq2[:n]) if a != b) + abs(len(seq1) - len(seq2)))


def infer_seq_col(df) -> str:
    for col in ["sequence_trimmed", "sequence", "pred_seq", "seq", "primary", "protein_sequence"]:
        if col in df.columns:
            return col
    raise ValueError("No sequence column found.")


def infer_fitness_col(df) -> Optional[str]:
    for col in ["fitness", "log_fluorescence", "tm", "enrichment"]:
        if col in df.columns:
            return col
    return None


def choose_wt(df) -> str:
    seq_col = infer_seq_col(df)
    if "label" in df.columns:
        wt = df[df["label"] == 0]
        if len(wt) > 0:
            return str(wt.iloc[0][seq_col])
    return str(df.iloc[0][seq_col])


def num_labels_for_dataset(name: str) -> int:
    if name in {"NESP", "ube4b"}:
        return 5
    if name in {"MSA", "MSA_RAW", "MDH"}:
        return 0
    return 8


def infer_dims_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    e = state.get("embed.weight")
    c0 = state.get("dec_conv_module.0.weight")
    c2 = state.get("dec_conv_module.2.weight")
    if e is None or c0 is None or c2 is None:
        return {"input_dim": 24, "embedding_dim": 100, "latent_dim": 64, "hidden_dim": 200, "seq_len": 95}
    hidden = int(c2.shape[1])
    return {
        "input_dim": int(e.shape[0]),
        "embedding_dim": int(e.shape[1]),
        "latent_dim": int(c0.shape[1]),
        "hidden_dim": hidden,
        "seq_len": int(c0.shape[0] // max(1, hidden // 2)),
    }


def build_hparams(dataset: str, dims: Dict[str, int], device: torch.device, batch_size: int = 16):
    return argparse.Namespace(
        dataset=dataset,
        num_labels=num_labels_for_dataset(dataset),
        input_dim=dims["input_dim"],
        latent_dim=dims["latent_dim"],
        hidden_dim=dims["hidden_dim"],
        embedding_dim=dims["embedding_dim"],
        kernel_size=4,
        layers=6,
        probs=0.2,
        batch_size=batch_size,
        lr=2e-5,
        alpha_val=1.0,
        gamma_val=1.0,
        sigma_val=1.5,
        eta_val=0.001,
        seq_len=dims["seq_len"],
        auxnetwork="dropout_reg",
        dif_T=500,
        dif_channel=128,
        dif_channel_mult=[1, 2, 2, 2],
        dif_res_blocks=2,
        dif_dropout=0.15,
        dif_beta_1=1e-4,
        dif_beta_T=0.028,
        device=str(device),
        dif_w=20.0,
    )


def normalize_ckpt_state(ckpt_obj: object) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        state = ckpt_obj["state_dict"]
    elif isinstance(ckpt_obj, dict):
        state = ckpt_obj
    else:
        raise ValueError("Unsupported checkpoint format")

    out = {}
    for k, v in state.items():
        nk = k
        for prefix in ("module.jtae.", "jtae.", "module."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        out[nk] = v
    return out


def filter_by_shape(src: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in src.items() if k in target and tuple(v.shape) == tuple(target[k].shape)}


class ESM2Scorer:
    def __init__(self, model_name: str, device: torch.device, head_path: Optional[str] = None):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self.head = None
        if head_path is not None and Path(head_path).exists():
            hidden = int(self.model.config.hidden_size)
            self.head = torch.nn.Linear(hidden, 1).to(device)
            ckpt = torch.load(head_path, map_location=device)
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            self.head.load_state_dict(state)
            self.head.eval()

    @torch.no_grad()
    def score_and_perplexity(self, seqs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        cleaned = [clean_seq(s).replace("X", "A") or "A" for s in seqs]
        tok = self.tokenizer(cleaned, return_tensors="pt", padding=True, truncation=True)
        tok = {k: v.to(self.device) for k, v in tok.items()}
        out = self.model(**tok, output_hidden_states=self.head is not None)

        logits = out.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        ids = tok["input_ids"]
        token_lp = torch.gather(log_probs, dim=-1, index=ids.unsqueeze(-1)).squeeze(-1)

        attn = tok["attention_mask"].bool()
        valid = attn.clone()
        for i in range(valid.shape[0]):
            pos = torch.where(attn[i])[0]
            if len(pos) >= 2:
                valid[i, pos[0]] = False
                valid[i, pos[-1]] = False
        mean_lp = (token_lp * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        perplexity = torch.exp(-mean_lp)

        if self.head is not None:
            hidden = out.hidden_states[-1]
            mask = tok["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            score = self.head(pooled).squeeze(-1)
        else:
            score = mean_lp

        return score, perplexity


def parse_ca_coords_from_pdb(path: Path) -> np.ndarray:
    try:
        from Bio.PDB import PDBParser
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Biopython is required for structure parsing. Install optional structural dependencies "
            "or run with structural scoring disabled."
        ) from exc
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(path))
    coords = []
    for atom in structure.get_atoms():
        if atom.get_name() == "CA":
            coords.append(atom.get_coord())
    if len(coords) == 0:
        raise ValueError(f"No CA atoms in {path}")
    return np.asarray(coords, dtype=np.float64)


def pdb_mean_plddt(pdb_text: str) -> float:
    vals = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM"):
            try:
                vals.append(float(line[60:66].strip()))
            except ValueError:
                continue
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    n = min(len(P), len(Q))
    if n == 0:
        return float("nan")
    P = P[:n]
    Q = Q[:n]
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    U = V @ D @ Wt
    P_aligned = Pc @ U
    return float(np.sqrt(np.mean(np.sum((P_aligned - Qc) ** 2, axis=1))))

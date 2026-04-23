import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
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
IND2SEQ = {ind: aa for aa, ind in SEQ2IND.items()}


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "t"}:
        return True
    if lowered in {"0", "false", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_sequence_for_esm(seq: str) -> str:
    cleaned = seq.replace("J", "X").replace("*", "").replace("-", "")
    if len(cleaned) == 0:
        return "A"
    return cleaned


def seq_to_inds(seq: str, seq_len: int) -> torch.Tensor:
    clipped = seq[:seq_len]
    if len(clipped) < seq_len:
        clipped = clipped + ("J" * (seq_len - len(clipped)))
    return torch.tensor([SEQ2IND.get(ch, SEQ2IND["X"]) for ch in clipped], dtype=torch.long)


def inds_to_seq(indices: torch.Tensor) -> str:
    return "".join(IND2SEQ[int(i)] for i in indices.tolist())


def infer_seq_column(frame: pd.DataFrame) -> str:
    for col in ["seq", "primary", "protein_sequence"]:
        if col in frame.columns:
            return col
    raise ValueError("Could not infer sequence column. Expected one of: seq, primary, protein_sequence")


def infer_fitness_column(frame: pd.DataFrame) -> Optional[str]:
    for col in ["fitness", "log_fluorescence", "tm", "enrichment"]:
        if col in frame.columns:
            return col
    return None


def choose_start_sequence(frame: pd.DataFrame, mode: str, start_index: int) -> str:
    seq_col = infer_seq_column(frame)
    if mode == "index":
        row_index = max(0, min(start_index, len(frame) - 1))
        return str(frame.iloc[row_index][seq_col])
    if mode == "random":
        return str(frame.sample(1).iloc[0][seq_col])
    if mode == "wildtype":
        if "label" in frame.columns:
            candidates = frame[frame["label"] == 0]
            if len(candidates) > 0:
                return str(candidates.iloc[0][seq_col])
        return str(frame.iloc[0][seq_col])
    if mode == "best":
        fit_col = infer_fitness_column(frame)
        if fit_col is None:
            return str(frame.iloc[0][seq_col])
        return str(frame.iloc[frame[fit_col].astype(float).idxmax()][seq_col])
    raise ValueError(f"Unsupported start mode: {mode}")


def get_jtae_hparams(
    seq_len: int,
    dataset: str,
    batch_size: int,
    latent_dim: int,
    input_dim: int,
    hidden_dim: int,
    embedding_dim: int,
    device: torch.device,
) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=dataset,
        num_labels=8,
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        kernel_size=4,
        layers=6,
        probs=0.2,
        batch_size=batch_size,
        lr=2e-5,
        alpha_val=1.0,
        gamma_val=1.0,
        sigma_val=1.5,
        eta_val=0.001,
        seq_len=seq_len,
        auxnetwork="dropout_reg",
        dif_T=500,
        dif_channel=128,
        dif_channel_mult=[1, 2, 2, 2],
        dif_res_blocks=2,
        dif_dropout=0.15,
        dif_beta_1=1e-4,
        dif_beta_T=0.028,
        device=str(device),
    )


def extract_ae_state_dict(ckpt_obj: object, model_state_keys: Sequence[str]) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        raw_state = ckpt_obj["state_dict"]
    elif isinstance(ckpt_obj, dict):
        raw_state = ckpt_obj
    else:
        raise ValueError("Unsupported checkpoint structure: expected dict or dict with key 'state_dict'.")

    model_key_set = set(model_state_keys)

    def keep_matching(source: Dict[str, torch.Tensor], strip_prefixes: Tuple[str, ...]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for key, val in source.items():
            normalized = key
            for prefix in strip_prefixes:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
            if normalized in model_key_set:
                out[normalized] = val
        return out

    candidates = [
        keep_matching(raw_state, ("jtae.",)),
        keep_matching(raw_state, ("module.jtae.",)),
        keep_matching(raw_state, ("module.",)),
        keep_matching(raw_state, tuple()),
    ]

    ae_state = max(candidates, key=lambda x: len(x))
    if len(ae_state) == 0:
        sample_keys = list(raw_state.keys())[:20]
        raise ValueError(
            "Unable to extract JTAE/AE weights from checkpoint. "
            f"Sample checkpoint keys: {sample_keys}"
        )
    return ae_state


def infer_model_dims_from_state_dict(raw_state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    embed_weight = raw_state.get("embed.weight")
    conv0_weight = raw_state.get("dec_conv_module.0.weight")
    conv2_weight = raw_state.get("dec_conv_module.2.weight")

    if embed_weight is None or conv0_weight is None or conv2_weight is None:
        return {
            "input_dim": 24,
            "embedding_dim": 100,
            "latent_dim": 64,
            "hidden_dim": 200,
            "seq_len": 95,
        }

    input_dim = int(embed_weight.shape[0])
    embedding_dim = int(embed_weight.shape[1])
    latent_dim = int(conv0_weight.shape[1])
    hidden_dim = int(conv2_weight.shape[1])

    half_hidden = max(1, hidden_dim // 2)
    seq_len = int(conv0_weight.shape[0] // half_hidden)
    if seq_len <= 0:
        seq_len = 95

    return {
        "input_dim": input_dim,
        "embedding_dim": embedding_dim,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "seq_len": seq_len,
    }


def filter_state_by_shape(
    state_dict: Dict[str, torch.Tensor], target_state: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    filtered: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape):
            filtered[key] = value
    return filtered


class ESM2Guidance:
    def __init__(self, model_name: str, device: torch.device, head_path: Optional[str] = None):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self.head = None

        if head_path is not None and Path(head_path).exists():
            hidden_size = int(self.model.config.hidden_size)
            self.head = torch.nn.Linear(hidden_size, 1).to(device)
            head_ckpt = torch.load(head_path, map_location=device)
            state = head_ckpt.get("state_dict", head_ckpt) if isinstance(head_ckpt, dict) else head_ckpt
            self.head.load_state_dict(state)
            self.head.eval()

    @torch.no_grad()
    def score(self, seqs: List[str]) -> torch.Tensor:
        cleaned = [clean_sequence_for_esm(s) for s in seqs]
        tok = self.tokenizer(cleaned, return_tensors="pt", padding=True, truncation=True)
        tok = {k: v.to(self.device) for k, v in tok.items()}
        out = self.model(**tok, output_hidden_states=self.head is not None)

        if self.head is not None:
            hidden = out.hidden_states[-1]
            mask = tok["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            return self.head(pooled).squeeze(-1)

        logits = out.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        input_ids = tok["input_ids"]
        gathered = torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

        attention_mask = tok["attention_mask"].bool()
        valid_mask = attention_mask.clone()
        for i in range(valid_mask.shape[0]):
            positions = torch.where(attention_mask[i])[0]
            if len(positions) >= 2:
                valid_mask[i, positions[0]] = False
                valid_mask[i, positions[-1]] = False
        summed = (gathered * valid_mask).sum(dim=1)
        counts = valid_mask.sum(dim=1).clamp_min(1)
        return summed / counts


@dataclass
class LikelihoodContext:
    model: torch.nn.Module
    guidance_mode: str
    device: torch.device
    esm2_guidance: Optional[ESM2Guidance]
    esm_weight: float
    reg_weight: float
    esm_mean: float
    esm_std: float
    reg_mean: float
    reg_std: float

    @torch.no_grad()
    def decode_to_sequence(self, z: torch.Tensor) -> List[str]:
        logits = self.model.decode(z)
        seq_indices = torch.argmax(logits, dim=1)
        sequences = [inds_to_seq(seq).replace("J", "") for seq in seq_indices]
        return sequences

    @torch.no_grad()
    def likelihood(self, z: torch.Tensor) -> torch.Tensor:
        if self.guidance_mode == "regressor":
            return self.model.regressor_module(z).squeeze(-1)
        if self.guidance_mode == "hybrid":
            if self.esm2_guidance is None:
                raise RuntimeError("Hybrid guidance requires ESM2 initialization.")
            reg_score = self.model.regressor_module(z).squeeze(-1)
            seqs = self.decode_to_sequence(z)
            esm_score = self.esm2_guidance.score(seqs)

            reg_z = (reg_score - self.reg_mean) / max(self.reg_std, 1e-8)
            esm_z = (esm_score - self.esm_mean) / max(self.esm_std, 1e-8)
            return (self.esm_weight * esm_z) + (self.reg_weight * reg_z)
        if self.esm2_guidance is None:
            raise RuntimeError("ESM2 guidance was requested but not initialized.")
        seqs = self.decode_to_sequence(z)
        return self.esm2_guidance.score(seqs)


@torch.no_grad()
def compute_guidance_calibration(
    model: torch.nn.Module,
    esm2_guidance: Optional[ESM2Guidance],
    seqs: List[str],
    seq_len: int,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    reg_scores: List[float] = []
    esm_scores: List[float] = []

    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i : i + batch_size]
        inds = torch.stack([seq_to_inds(s, seq_len) for s in chunk], dim=0).to(device)
        z = model.encode(inds).to(torch.float32)
        reg = model.regressor_module(z).squeeze(-1).detach().cpu().tolist()
        reg_scores.extend(reg)
        if esm2_guidance is not None:
            esm = esm2_guidance.score(chunk).detach().cpu().tolist()
            esm_scores.extend(esm)

    reg_t = torch.tensor(reg_scores, dtype=torch.float32)
    if len(esm_scores) > 0:
        esm_t = torch.tensor(esm_scores, dtype=torch.float32)
        esm_mean = float(esm_t.mean().item())
        esm_std = float(esm_t.std(unbiased=False).item())
    else:
        esm_mean = 0.0
        esm_std = 1.0

    return {
        "esm_mean": esm_mean,
        "esm_std": esm_std,
        "reg_mean": float(reg_t.mean().item()),
        "reg_std": float(reg_t.std(unbiased=False).item()),
    }


@torch.no_grad()
def elliptical_slice_step(
    z_current: torch.Tensor,
    log_like_current: torch.Tensor,
    likelihood_ctx: LikelihoodContext,
    center: Optional[torch.Tensor],
    delta: Optional[float],
    max_attempts: int,
) -> Tuple[torch.Tensor, torch.Tensor, bool, int]:
    nu = torch.randn_like(z_current)
    log_y = log_like_current + torch.log(torch.rand(1, device=z_current.device))

    theta = torch.rand(1, device=z_current.device) * (2.0 * math.pi)
    theta_min = theta - 2.0 * math.pi
    theta_max = theta.clone()

    for attempt in range(1, max_attempts + 1):
        proposal = z_current * torch.cos(theta) + nu * torch.sin(theta)
        if center is not None and delta is not None:
            if torch.norm(proposal - center, p=2).item() > float(delta):
                if theta.item() < 0:
                    theta_min = theta
                else:
                    theta_max = theta
                theta = torch.empty(1, device=z_current.device).uniform_(theta_min.item(), theta_max.item())
                continue

        proposal_ll = likelihood_ctx.likelihood(proposal.unsqueeze(0)).squeeze(0)
        if proposal_ll > log_y:
            return proposal, proposal_ll, True, attempt

        if theta.item() < 0:
            theta_min = theta
        else:
            theta_max = theta
        theta = torch.empty(1, device=z_current.device).uniform_(theta_min.item(), theta_max.item())

    return z_current, log_like_current, False, max_attempts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TFG generation with JTAE latent ESS/TESS.")
    parser.add_argument("--proldm-root", type=str, default="../PROLDM_OUTLIER")
    parser.add_argument("--checkpoint", type=str, default="train_logs/GFP/epoch_1000.pt")
    parser.add_argument("--input-csv", type=str, default="data/mut_data/GFP-train.csv")
    parser.add_argument("--dataset", type=str, default="GFP")
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--guidance", type=str, choices=["esm2", "regressor", "hybrid"], default="esm2")
    parser.add_argument("--esm2-model", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--esm2-head-path", type=str, default=None)
    parser.add_argument("--esm-weight", type=float, default=0.5)
    parser.add_argument("--reg-weight", type=float, default=0.5)
    parser.add_argument("--calibration-samples", type=int, default=2048)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-chains", type=int, default=8)
    parser.add_argument("--burnin", type=int, default=20)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--max-steps-per-chain", type=int, default=5000)
    parser.add_argument("--start-mode", type=str, choices=["wildtype", "best", "random", "index"], default="wildtype")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--use-tess", type=str2bool, default=False)
    parser.add_argument("--delta", type=float, default=12.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=str, default="generated_seq/tfg_generation.csv")
    parser.add_argument("--output-json", type=str, default="generated_seq/tfg_generation_meta.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    proldm_root = (script_dir / args.proldm_root).resolve()
    ckpt_path = (proldm_root / args.checkpoint).resolve()
    input_csv = (proldm_root / args.input_csv).resolve()

    if not proldm_root.exists():
        raise FileNotFoundError(f"PROLDM root not found: {proldm_root}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    sys.path.insert(0, str(proldm_root))
    from model.JTAE.models_condif_1d import jtae

    requested_device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    frame = pd.read_csv(input_csv)
    seq_col = infer_seq_column(frame)

    start_seq = choose_start_sequence(frame, args.start_mode, args.start_index)
    wt_seq = choose_start_sequence(frame, "wildtype", 0)

    ckpt_obj = torch.load(ckpt_path, map_location=requested_device)
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        ckpt_raw = ckpt_obj["state_dict"]
    elif isinstance(ckpt_obj, dict):
        ckpt_raw = ckpt_obj
    else:
        raise ValueError("Unsupported checkpoint format")

    normalized_raw = {}
    for k, v in ckpt_raw.items():
        nk = k
        for prefix in ("module.jtae.", "jtae.", "module."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        normalized_raw[nk] = v

    inferred_dims = infer_model_dims_from_state_dict(normalized_raw)
    seq_len = args.seq_len if args.seq_len > 0 else inferred_dims["seq_len"]
    latent_dim = args.latent_dim if args.latent_dim > 0 else inferred_dims["latent_dim"]

    hparams = get_jtae_hparams(
        seq_len=seq_len,
        dataset=args.dataset,
        batch_size=max(args.batch_size, 2),
        latent_dim=latent_dim,
        input_dim=inferred_dims["input_dim"],
        hidden_dim=inferred_dims["hidden_dim"],
        embedding_dim=inferred_dims["embedding_dim"],
        device=requested_device,
    )
    ae = jtae(hparams).to(requested_device)
    ae.eval()

    ae_state = extract_ae_state_dict(ckpt_obj, ae.state_dict().keys())
    ae_state = filter_state_by_shape(ae_state, ae.state_dict())
    missing, unexpected = ae.load_state_dict(ae_state, strict=False)

    if len(ae_state) == 0:
        raise RuntimeError("No JTAE weights were loaded from checkpoint.")

    esm2_guidance = None
    if args.guidance in {"esm2", "hybrid"}:
        esm2_guidance = ESM2Guidance(
            model_name=args.esm2_model,
            device=requested_device,
            head_path=args.esm2_head_path,
        )

    calib = {
        "esm_mean": 0.0,
        "esm_std": 1.0,
        "reg_mean": 0.0,
        "reg_std": 1.0,
    }
    if args.guidance == "hybrid":
        seqs_for_calib = frame[seq_col].astype(str).tolist()
        if len(seqs_for_calib) > args.calibration_samples:
            seqs_for_calib = frame.sample(args.calibration_samples, random_state=args.seed)[seq_col].astype(str).tolist()
        calib = compute_guidance_calibration(
            model=ae,
            esm2_guidance=esm2_guidance,
            seqs=seqs_for_calib,
            seq_len=seq_len,
            device=requested_device,
            batch_size=max(8, args.batch_size),
        )

    likelihood_ctx = LikelihoodContext(
        model=ae,
        guidance_mode=args.guidance,
        device=requested_device,
        esm2_guidance=esm2_guidance,
        esm_weight=float(args.esm_weight),
        reg_weight=float(args.reg_weight),
        esm_mean=calib["esm_mean"],
        esm_std=calib["esm_std"],
        reg_mean=calib["reg_mean"],
        reg_std=calib["reg_std"],
    )

    with torch.no_grad():
        start_inds = seq_to_inds(start_seq, seq_len).unsqueeze(0).to(requested_device)
        z_start = ae.encode(start_inds).squeeze(0).to(torch.float32)

        wt_inds = seq_to_inds(wt_seq, seq_len).unsqueeze(0).to(requested_device)
        z_wt = ae.encode(wt_inds).squeeze(0).to(torch.float32)

    samples_per_chain = math.ceil(args.num_samples / args.num_chains)
    rows: List[Dict[str, object]] = []

    for chain_id in range(args.num_chains):
        z_current = z_start.clone()
        if args.start_mode == "random":
            z_current = torch.randn_like(z_current)

        log_like_current = likelihood_ctx.likelihood(z_current.unsqueeze(0)).squeeze(0)
        saved = 0

        for step in range(args.max_steps_per_chain):
            center = z_wt if args.use_tess else None
            radius = args.delta if args.use_tess else None

            z_current, log_like_current, accepted, attempts = elliptical_slice_step(
                z_current=z_current,
                log_like_current=log_like_current,
                likelihood_ctx=likelihood_ctx,
                center=center,
                delta=radius,
                max_attempts=256,
            )

            if step < args.burnin:
                continue
            if ((step - args.burnin) % args.thin) != 0:
                continue

            seq = likelihood_ctx.decode_to_sequence(z_current.unsqueeze(0))[0]
            row = {
                "chain": chain_id,
                "step": step,
                "accepted": bool(accepted),
                "attempts": attempts,
                "score": float(log_like_current.item()),
                "sequence": seq,
                "sequence_trimmed": seq.replace("J", "").replace("*", "").replace("-", ""),
                "dist_to_wt_latent_l2": float(torch.norm(z_current - z_wt, p=2).item()),
                "start_mode": args.start_mode,
                "guidance": args.guidance,
                "source_seq_col": seq_col,
            }
            rows.append(row)
            saved += 1

            if saved >= samples_per_chain:
                break

        print(f"chain={chain_id} saved={saved} final_score={float(log_like_current.item()):.5f}")

    rows = rows[: args.num_samples]
    output_csv = (script_dir / args.output_csv).resolve()
    output_json = (script_dir / args.output_json).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)

    meta = {
        "proldm_root": str(proldm_root),
        "checkpoint": str(ckpt_path),
        "input_csv": str(input_csv),
        "num_rows": len(out_df),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "args": vars(args),
    }
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(f"Saved samples to: {output_csv}")
    print(f"Saved metadata to: {output_json}")


if __name__ == "__main__":
    main()
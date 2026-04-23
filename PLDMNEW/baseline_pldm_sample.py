import argparse
import sys
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline PRO-LDM sampler with CPU-safe checkpoint loading.")
    parser.add_argument("--proldm-root", type=str, default="../PROLDM_OUTLIER")
    parser.add_argument("--checkpoint", type=str, default="train_logs/GFP/dropout_tiny_epoch_1000.pt")
    parser.add_argument("--dataset", type=str, default="GFP")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--label", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--identity", action="store_true")
    parser.add_argument("--output-csv", type=str, default="generated_seq/raw_baseline_pldm.csv")
    return parser.parse_args()


def _num_labels_for_dataset(dataset: str) -> int:
    if dataset in {"NESP", "ube4b"}:
        return 5
    if dataset in {"MSA", "MSA_RAW", "MDH"}:
        return 0
    return 8


def inds_to_seq(seq_tensor: torch.Tensor) -> str:
    return "".join(IND2SEQ[int(i)] for i in seq_tensor.tolist())


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    proldm_root = (script_dir / args.proldm_root).resolve()
    ckpt_path = (proldm_root / args.checkpoint).resolve()
    out_path = (script_dir / args.output_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not proldm_root.exists():
        raise FileNotFoundError(f"PROLDM root not found: {proldm_root}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    sys.path.insert(0, str(proldm_root))
    from model.JTAE.models_condif_1d import jtae
    from model.ConDiff.DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionSampler

    requested_device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    ckpt = torch.load(ckpt_path, map_location=requested_device)

    embed_shape = ckpt["embed.weight"].shape
    conv0_shape = ckpt["dec_conv_module.0.weight"].shape
    conv2_shape = ckpt["dec_conv_module.2.weight"].shape

    input_dim = int(embed_shape[0])
    embedding_dim = int(embed_shape[1])
    latent_dim = int(conv0_shape[1])
    hidden_dim = int(conv2_shape[1])
    seq_len = int(conv0_shape[0] // max(1, hidden_dim // 2))

    model_args = argparse.Namespace(
        dataset=args.dataset,
        part="train",
        sav_dir="./train_logs/",
        input_dim=input_dim,
        batch_size=max(8, min(256, args.num_samples)),
        project_name="JT-AE",
        use_wandb=False,
        alpha_val=1.0,
        beta_val=0.0005,
        gamma_val=1.0,
        sigma_val=1.5,
        eta_val=0.001,
        reg_ramp=False,
        vae_ramp=True,
        wl2norm=False,
        lr=2e-5,
        n_epochs=500,
        dev=False,
        seq_len=seq_len,
        embedding_dim=embedding_dim,
        kernel_size=4,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        layers=6,
        probs=0.2,
        auxnetwork="dropout_reg",
        dif_T=500,
        dif_channel=128,
        dif_channel_mult=[1, 2, 2, 2],
        dif_res_blocks=2,
        dif_dropout=0.15,
        dif_multiplier=2.5,
        dif_beta_1=1e-4,
        dif_beta_T=0.028,
        device=str(requested_device),
        dif_w=1.8,
        num_labels=_num_labels_for_dataset(args.dataset),
        mode="sample",
        training_load_epoch=None,
        device_id=[0],
        multiplier=2.5,
        eval_load_epoch=None,
        multi_gpu=False,
        load_path=str((proldm_root / "train_logs").resolve()),
        dif_sample_size=args.num_samples,
        dif_sample_label=args.label,
        dif_sample_epoch=1000,
        dif_outlier_step=0,
        identity=args.identity,
    )

    model = jtae(model_args).to(requested_device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    with torch.no_grad():
        labels = (args.label * torch.ones(args.num_samples)).long().to(requested_device)
        sampler = GaussianDiffusionSampler(
            model.diff_model,
            model_args.dif_beta_1,
            model_args.dif_beta_T,
            model_args.dif_T,
            model_args,
        ).to(requested_device)

        noisy_z = torch.randn(size=[args.num_samples, 1, latent_dim], device=requested_device)
        sampled_z = sampler(noisy_z, labels)
        sampled_z = torch.squeeze(sampled_z, 1).to(torch.float32)

        x_hat = model.decode(sampled_z).argmax(dim=1)
        pred_fit = model.regressor_module(sampled_z).squeeze(dim=1).detach().cpu().numpy().tolist()

    pred_seq = []
    for seq in x_hat:
        s = inds_to_seq(seq)
        if args.identity:
            s = s.replace("J", "").replace("X", "").replace("*", "").replace("-", "")
        pred_seq.append(s)

    out_df = pd.DataFrame({"pred_seq": pred_seq, "pred_fit": pred_fit})
    out_df.to_csv(out_path, index=False)
    print(f"Saved baseline samples: {out_path}")


if __name__ == "__main__":
    main()

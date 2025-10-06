# ML4NS-project: ProteinAE

Diffusion Protein AutoEncoder for Structure Encoding and Generation.

## Setup
```
mamba activate proteinae
pip install -e .
```

## AutoEncoder

### Inference

```bash
python proteinfoundation/autoencode.py \
    --input_pdb example/T1133-D1.pdb \
    --output_dir output \
    --config /path/to/configs
```

### Training

```bash
python proteinfoundation/train_ae.py \
    --config_name training_ae_r1_d8
```


## LDM

### Training

```bash
python proteinfoundation/train_ldm.py \
    --config_name training_pldm_200M_afdb_512
```
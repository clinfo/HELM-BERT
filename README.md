# HELM-BERT

A peptide language model using **HELM (Hierarchical Editing Language for Macromolecules)** notation, compatible with Hugging Face Transformers.

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Flansma%2Fhelm--bert-blue)](https://huggingface.co/Flansma/helm-bert)

## Model Description

HELM-BERT is built upon the DeBERTa architecture, designed for peptide sequences in HELM notation:

- **Disentangled Attention**: Decomposes attention into content-content and content-position terms
- **Enhanced Mask Decoder (EMD)**: Injects absolute position embeddings at the decoder stage
- **Span Masking**: Contiguous token masking with geometric distribution
- **nGiE**: n-gram Induced Encoding layer (1D convolution, kernel size 3)

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Parameters | 54.8M |
| Hidden size | 768 |
| Layers | 6 |
| Attention heads | 12 |
| Vocab size | 78 |
| Max token length | 512 |

## Installation

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
```

## How to Use

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("Flansma/helm-bert", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Flansma/helm-bert", trust_remote_code=True)

# Cyclosporine A
inputs = tokenizer("PEPTIDE1{[Abu].[Sar].[meL].V.[meL].A.[dA].[meL].[meL].[meV].[Me_Bmt(E)]}$PEPTIDE1,PEPTIDE1,1:R1-11:R2$$$", return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state
```

## Training

### MLM Training

```bash
# Continue pre-training from Hub model (default)
python scripts/train_mlm.py

# Continue pre-training from local checkpoint
python scripts/train_mlm.py --pretrained_path ./my-checkpoint

# Train from scratch
python scripts/train_mlm.py --from_scratch

# From scratch with custom architecture
python scripts/train_mlm.py --from_scratch --num_hidden_layers 12 --hidden_size 768
```

| Option | Default | Description |
|--------|---------|-------------|
| `--pretrained_path` | Flansma/helm-bert | Model to continue pre-training from |
| `--from_scratch` | False | Train from scratch instead of continue pre-training |
| `--hidden_size` | 768 | Hidden dimension (only with --from_scratch) |
| `--num_hidden_layers` | 6 | Number of layers (only with --from_scratch) |
| `--num_attention_heads` | 12 | Number of attention heads (only with --from_scratch) |
| `--max_epochs` | 500 | Maximum training epochs |
| `--batch_size` | 64 | Batch size |
| `--learning_rate` | 1e-4 | Learning rate |
| `--mlm_probability` | 0.15 | Masking probability |
| `--early_stopping_patience` | 20 | Early stopping patience |
| `--precision` | 32-true | 32-true, 16-mixed, bf16-mixed |
| `--disable_wandb` | False | Disable WandB logging |

### Permeability Prediction

```bash
# Fine-tune all layers (default)
python scripts/train_permeability.py

# Freeze encoder (train head only)
python scripts/train_permeability.py --freeze_encoder --head_lr 1e-3

# Custom data
python scripts/train_permeability.py \
    --train_file ./data/custom_train.csv --test_file ./data/custom_test.csv \
    --helm_column HELM --target_column LogP
```

| Option | Default | Description |
|--------|---------|-------------|
| `--pretrained_path` | Flansma/helm-bert | Pretrained model path or Hub ID |
| `--freeze_encoder` | False | Freeze encoder weights |
| `--classifier_num_layers` | 2 | Number of MLP head layers |
| `--classifier_dropout` | 0.1 | Classifier dropout rate |
| `--max_epochs` | 200 | Maximum training epochs |
| `--batch_size` | 32 | Batch size |
| `--encoder_lr` | 3e-5 | Encoder learning rate |
| `--head_lr` | 1e-4 | Head learning rate |
| `--early_stopping_patience` | 20 | Early stopping patience |
| `--precision` | 32-true | 32-true, 16-mixed, bf16-mixed |
| `--disable_wandb` | False | Disable WandB logging |

### PPI Classification

```bash
# Default (both encoders frozen)
python scripts/train_ppi.py

# Fine-tune drug encoder
python scripts/train_ppi.py --finetune_drug_encoder

# Fine-tune both encoders
python scripts/train_ppi.py --finetune_drug_encoder --finetune_target_encoder --encoder_lr 1e-5

# Custom data
python scripts/train_ppi.py \
    --train_file ./data/custom_ppi_train.csv --test_file ./data/custom_ppi_test.csv \
    --drug_column Peptide_HELM --target_column Protein_Seq --label_column Binding
```

Uses HELM-BERT (peptide) + ESM-2 (protein) dual encoder.

| Option | Default | Description |
|--------|---------|-------------|
| `--pretrained_path` | Flansma/helm-bert | Drug encoder model |
| `--target_encoder` | facebook/esm2_t33_650M_UR50D | Target encoder model |
| `--finetune_drug_encoder` | - | Unfreeze drug encoder |
| `--finetune_target_encoder` | - | Unfreeze target encoder |
| `--max_epochs` | 200 | Maximum training epochs |
| `--batch_size` | 32 | Batch size |
| `--encoder_lr` | 3e-5 | Encoder learning rate |
| `--head_lr` | 1e-4 | Head learning rate |
| `--early_stopping_patience` | 20 | Early stopping patience |
| `--precision` | 32-true | 32-true, 16-mixed, bf16-mixed |
| `--disable_wandb` | False | Disable WandB logging |

## Citation

If you use HELM-BERT in your research, please cite:

```bibtex
@article{lee2025helmbert,
  title={HELM-BERT: A Transformer for Medium-sized Peptide Property Prediction},
  author={Seungeon Lee and Takuto Koyama and Itsuki Maeda and Shigeyuki Matsumoto and Yasushi Okuno},
  journal={arXiv preprint arXiv:2512.23175},
  year={2025},
  url={https://arxiv.org/abs/2512.23175}
}
```

## License

MIT License

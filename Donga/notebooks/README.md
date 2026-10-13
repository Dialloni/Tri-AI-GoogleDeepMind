# Colab notebooks

Fine-tuning needs a GPU, so these run in Colab (T4). Everything else in the
repo runs locally on CPU.

**Each notebook exists in two formats — they are the same code, not different
experiments.** The `.ipynb` uploads straight to Colab; the `.py` in
[scripts/](scripts/) is the same cell as plain text, for reading in a diff or
pasting into a fresh notebook. Rounds 1 and 2 were run by pasting the `.py`, so
they have no `.ipynb`; later rounds were run as notebooks.

## Which ones are in the submission

Only the three encoders below are loaded by `scripts/ltr.py`. Their exported
embeddings are committed to the repo root, so **the submission reproduces
without running any of these notebooks.**

| Round | Exports | Committed? | In the submission | Holdout nDCG@5 |
|---|---|---|---|---|
| [colab_finetune.py](scripts/colab_finetune.py) | `ft_embs.npz` | yes | **yes** | 0.8893 |
| [colab_finetune2.py](scripts/colab_finetune2.py) | `ft_embs2.npz` | yes | **yes** | 0.9225 |
| [colab_finetune4.py](scripts/colab_finetune4.py) · [.ipynb](colab_finetune4.ipynb) | `ft_embs4.npz` | yes | **yes** — helps the blend | 0.9165 |
| [colab_finetune3.py](scripts/colab_finetune3.py) · [.ipynb](colab_finetune3.ipynb) | `ft_embs3.npz` | no | no — diverged to NaN (fp16/Adam), kept for the fp32 fix | — |
| [colab_finetune5.py](scripts/colab_finetune5.py) · [.ipynb](colab_finetune5.ipynb) | `ft_embs5.npz` | no | no — curriculum variant, no gain | — |
| [colab_rerank.py](scripts/colab_rerank.py) | `ce_scores.npz` | no | no — round-3 cross-encoder, superseded | — |
| [colab_rerank2.py](scripts/colab_rerank2.py) · [.ipynb](colab_rerank2.ipynb) | `ce2_scores.npz` | yes | no — negative result | 0.9169 → 0.9578 |
| [colab_rerank2_export.py](scripts/colab_rerank2_export.py) | `ce2_scores.npz` | yes | no — produced the committed `ce2_scores.npz` | — |

The rounds that did *not* make the submission are kept on purpose: they are the
evidence behind the negative results written up in
[../doc/experiments.md](../doc/experiments.md).

## Running one

Upload the `.ipynb` to Colab (T4), or paste the matching `.py` into one cell
after `kagglehub.login()`. Each prints its topic-grouped holdout score, then
downloads its `.npz` — drop that in the repo root and `scripts/ltr.py` picks it
up automatically as an extra encoder.

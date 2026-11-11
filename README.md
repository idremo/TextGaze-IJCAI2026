# TextGaze-IJCAI2026

**Official PyTorch implementation of "TextGaze: Prompting Gaze Target Estimation with Textual Scene Cues" (IJCAI-ECAI 2026)**

[Paper (DOI)](https://doi.org/10.24963/ijcai.2026/177) · [arXiv:2607.10130](https://arxiv.org/abs/2607.10130) · [Pretrained checkpoints](https://github.com/idremo/TextGaze-IJCAI2026/releases/tag/checkpoints-v1.0)

> TextGaze injects textual scene cues — dense, automatically generated scene captions —
> into a gaze target estimation model, resolving ambiguity in scenarios where visual
> appearance alone is misleading (e.g., *is this person looking at their own screen,
> or at the person next to them?*)

<p align="center">
  <img src="assets/qualitative/sample_2_00003610_4_comparison.jpg" width="80%">
  <br>
  <em>Green box + dashed line: ground-truth gaze target. Textual scene cues let the model
  disambiguate who the person in the foreground is actually looking at.</em>
</p>

## 🔥 News

- **[2026-01]** TextGaze accepted to **IJCAI-ECAI 2026** (main track).
- **[2026-09]** Repository released: code, annotations, and text-cue annotations.
- **[2026-11]** Pretrained checkpoints released: [checkpoints-v1.0](https://github.com/idremo/TextGaze-IJCAI2026/releases/tag/checkpoints-v1.0).

> [!NOTE]
> This repository documents three corrections to the camera-ready version;
> see [ERRATA.md](ERRATA.md). The version of record is the IJCAI-ECAI 2026
> proceedings version ([DOI: 10.24963/ijcai.2026/177](https://doi.org/10.24963/ijcai.2026/177)).
> No model, experiment, result, or conclusion was changed.

## 🧩 Method

TextGaze builds on the GAZELLE gaze estimation architecture (DINOv2 backbone with a
vision-language fusion transformer) and adds a **textual cues branch**:

1. **Visual backbone**: DINOv2 (ViT-B / ViT-L) encodes the scene.
2. **Textual cues**: Qwen2.5-VL generates dense English scene captions; a BERT encoder
   embeds them (`*_Qwen_En` annotation variants carry these captions per head).
3. **Fusion**: a lightweight vision-language transformer fuses head features with text
   embeddings.
4. **Heads**: per-person gaze **heatmap** (64×64) + optional **in/out** of frame prediction.

The text branch requires no extra human annotation beyond the generated captions
and adds little inference overhead.

## 📊 Results (paper numbers)

| Benchmark | Model | AUC | Avg L2 | Min L2 / AP |
|---|---|---|---|---|
| GazeFollow | TextGaze ViT-L | **0.956** | 0.099 | 0.043 (MinL2) |
| GazeFollow | TextGaze ViT-B | 0.953 | 0.105 | 0.047 (MinL2) |
| VideoAttentionTarget | TextGaze ViT-L | 0.941 | 0.102 | 0.911 (AP) |
| VideoAttentionTarget | TextGaze ViT-B | 0.932 | 0.113 | 0.879 (AP) |
| ChildPlay | TextGaze ViT-L | 0.950 | 0.111 | 0.996 (AP) |
| ChildPlay | TextGaze ViT-B | 0.945 | 0.112 | 0.992 (AP) |

See the paper for the full comparison tables against prior work.

## 📁 Repository Structure

```
TextGaze-IJCAI2026/
├── gazefuze/                  # Core package (model, backbone, dataloader, utils)
├── data_prep/                 # Dataset preprocessing incl. Qwen2.5-VL caption generation
├── scripts/
│   ├── train_gazefollow.py    # Training on GazeFollow
│   ├── train_vat.py           # Training on VideoAttentionTarget
│   ├── eval_childplay.py      # Evaluation (ChildPlay / VAT-style JSON annotations)
│   ├── eval_gooreal.py        # Evaluation (GOO-Real / GazeFollow-format JSON)
│   └── pick_text.py           # Text-cue selection utilities
├── inference_demo.py          # Single-image quick-start demo
├── datasets/                  # Annotations (train/test where applicable; images not included)
├── models/                    # Checkpoint zoo (links)
├── experiments/               # Training-run layout reference
├── visualization_results/     # Additional qualitative heatmaps
└── assets/                    # Qualitative results used in this README
```

## ⚙️ Setup

```bash
git clone https://github.com/idremo/TextGaze-IJCAI2026.git
cd TextGaze-IJCAI2026
pip install -r requirements.txt
```

DINOv2 and `bert-base-uncased` weights are fetched automatically on first run
(DINOv2 via `torch.hub`; BERT via HuggingFace). For offline machines, pre-populate
the torch/HuggingFace caches, or set `DINOV2_HUB_DIR` to a local
`facebookresearch_dinov2_main` checkout.

The Qwen2.5-VL captioner is only needed when *regenerating* text cues — the released
`*_Qwen_En` annotations already contain them (`Qwen/Qwen2.5-VL-7B-Instruct`).

## 🏋️ Training

Training requires the benchmark images; this repository ships annotations only.
`--data_path` must point to a directory containing `train_preprocessed.json` and
the images referenced by it.

```bash
# GazeFollow (ViT-B)
python scripts/train_gazefollow.py \
    --model gazefuze_dinov2_vitb14 \
    --data_path /path/to/gazefollow_extended_Qwen_En \
    --ckpt_save_dir ./experiments --exp_name gf_vitb

# VAT (ViT-B), initialized from the released GazeFollow checkpoint.
# VAT uses text_supervision_weight=0.05; the GazeFollow stage uses the default 0.1.
python scripts/train_vat.py \
    --model gazefuze_dinov2_vitb14_inout \
    --init_ckpt /path/to/TextGaze_GF_ViT-B_epoch20.pt \
    --data_path /path/to/videoattentiontarget_Qwen_En \
    --text_supervision_weight 0.05 \
    --ckpt_save_dir ./experiments --exp_name vat_vitb
```

See `python scripts/train_gazefollow.py --help` for all options. Run outputs follow
the layout documented in [`experiments/README.md`](experiments/README.md).

## 📈 Evaluation

```bash
# ChildPlay (VAT-style JSON annotations)
python scripts/eval_childplay.py \
    --data_path  /path/to/childplay_Qwen_En/images \
    --model_name gazefuze_dinov2_vitl14_inout \
    --ckpt_path  /path/to/TextGaze_VAT_ViT-L_epoch7.pt \
    --use_text   True

# Ablation without textual cues
python scripts/eval_childplay.py \
    --data_path  /path/to/childplay_Qwen_En/images \
    --model_name gazefuze_dinov2_vitl14_inout \
    --ckpt_path  /path/to/TextGaze_VAT_ViT-L_epoch7.pt \
    --use_text   False

# GOO-Real / GazeFollow-format JSON
python scripts/eval_gooreal.py \
    --model      gazefuze_dinov2_vitl14_inout \
    --data_path  /path/to/gooreal_Qwen_En \
    --ckpt_path  /path/to/TextGaze_VAT_ViT-L_epoch7.pt
```

## 🚀 Quick Start (single image)

```python
# See inference_demo.py for a complete runnable example
import torch
from PIL import Image
from gazefuze.model import get_gazefuze_model

model, transform = get_gazefuze_model("gazefuze_dinov2_vitl14")
model.load_gazefuze_state_dict(torch.load("TextGaze_GF_ViT-L_epoch23.pt", map_location="cpu", weights_only=True))
model.eval()

pred = model.forward({
    "images": transform(Image.open("scene.jpg").convert("RGB")).unsqueeze(0),
    "bboxes": [[[0.42, 0.13, 0.50, 0.39]]],          # one head bbox (x1, y1, x2, y2), normalized
    "texts":  ["Two people at desks; the left person gazes toward the right person."],
    "num_ppl_per_img": [1],
})
heatmap = pred["heatmap"][0][0]   # 64x64 gaze heatmap for head #0
```

Or run the file directly:

```bash
python inference_demo.py --image scene.jpg --bbox 0.42 0.13 0.50 0.39 \
    --text "Two people at desks; the left person gazes toward the right person." \
    --ckpt TextGaze_GF_ViT-L_epoch23.pt \
    --arch gazefuze_dinov2_vitl14 --out overlay.jpg
```

For the VAT / ChildPlay checkpoints, use `--arch gazefuze_dinov2_vitl14_inout`
(or the ViT-B variant).

## 🗂️ Data & Annotations

The repository ships the annotations used to train and evaluate TextGaze.
Benchmark images are **not** redistributed. See
[`datasets/README.md`](datasets/README.md) for the full format specification.

| Directory | Content |
|---|---|
| `datasets/{childplay,childplay_Qwen_En}/` | ChildPlay test split, per-sequence CSV + `test_preprocessed.json` |
| `datasets/{gazefollow_extended,gazefollow_extended_Qwen_En}/` | GazeFollow(-extended) train + test release lists + preprocessed JSON |
| `datasets/{videoattentiontarget,videoattentiontarget_Qwen_En}/` | VAT train + test annotations (per-frame `.txt` + preprocessed JSON) |
| `datasets/{gooreal,gooreal_Qwen_En}/` | GOO-Real test split list |

`*_Qwen_En` variants carry the **generated textual scene cues** (per-head `text`
field in the JSON). CSV/TXT files do not include text cues. ChildPlay and
GOO-Real are used for zero-shot evaluation in the paper.

## 🖼️ Qualitative Results

Textual cues matter most when visual context alone is ambiguous — a person facing a
laptop may in fact be looking past it:

| Sample | w/o text cues (old model) | TextGaze (full model) |
|---|---|---|
| Classroom | peak on own screen ❌ | peak on the gazed-at person ✅ |

<p align="center">
  <img src="assets/qualitative/sample_1_00002049_2_old_model_pred.jpg" width="45%" title="without textual cues">
  <img src="assets/qualitative/sample_1_00002049_3_new_model_pred.jpg" width="45%" title="TextGaze">
  <br>
  <em>Left: baseline prediction. Right: TextGaze — the peak moves to the truly gazed-at person
  (green box = ground truth).</em>
</p>

More side-by-side comparisons: [`assets/gallery/`](assets/gallery/).

## 📌 Checkpoints

Pretrained models are available in [checkpoints-v1.0](https://github.com/idremo/TextGaze-IJCAI2026/releases/tag/checkpoints-v1.0)
(GazeFollow / VAT, ViT-B / ViT-L, with in/out variants). See
[`models/README.md`](models/README.md) for the full table, download links, and SHA-256 checksums.

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{she2026textgaze,
  title     = {TextGaze: Prompting Gaze Target Estimation with Textual Scene Cues},
  author    = {She, Junhui and Wang, Fei and Li, Kun and Nie, Yiqi and
               Liu, Yuxin and Duan, Zhangling and Yang, Xun},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on
               Artificial Intelligence (IJCAI-ECAI 2026)},
  pages     = {1586--1594},
  year      = {2026},
  doi       = {10.24963/ijcai.2026/177}
}
```

## 📄 License

This repository is released under the [Apache License 2.0](LICENSE).
Dataset annotations keep the licensing terms of their respective source benchmarks.

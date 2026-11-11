# Checkpoints

Pretrained TextGaze checkpoints (PyTorch, `load_gazelle_state_dict`-compatible),
released in [checkpoints-v1.0](https://github.com/idremo/TextGaze-IJCAI2026/releases/tag/checkpoints-v1.0).
SHA-256 checksums: [SHA256SUMS.txt](https://github.com/idremo/TextGaze-IJCAI2026/releases/download/checkpoints-v1.0/SHA256SUMS.txt).

| Model | Arch | Benchmark | AUC | Avg L2 | AP | Download |
|---|---|---|---|---|---|---|
| TextGaze GF ViT-L | `gazelle_dinov2_vitl14` | GazeFollow | 0.956 | 0.099 | — | [epoch_23.pt](https://github.com/idremo/TextGaze-IJCAI2026/releases/download/checkpoints-v1.0/TextGaze_GF_ViT-L_epoch23.pt) |
| TextGaze GF ViT-B | `gazelle_dinov2_vitb14` | GazeFollow | 0.953 | 0.105 | — | [epoch_20.pt](https://github.com/idremo/TextGaze-IJCAI2026/releases/download/checkpoints-v1.0/TextGaze_GF_ViT-B_epoch20.pt) |
| TextGaze VAT ViT-L | `gazelle_dinov2_vitl14_inout` | VAT / ChildPlay | 0.941 / 0.950 | 0.102 / 0.111 | 0.911 / 0.996 | [epoch_7.pt](https://github.com/idremo/TextGaze-IJCAI2026/releases/download/checkpoints-v1.0/TextGaze_VAT_ViT-L_epoch7.pt) |
| TextGaze VAT ViT-B | `gazelle_dinov2_vitb14_inout` | VAT / ChildPlay | 0.932 / 0.945 | 0.113 / 0.112 | 0.879 / 0.992 | [epoch_6.pt](https://github.com/idremo/TextGaze-IJCAI2026/releases/download/checkpoints-v1.0/TextGaze_VAT_ViT-B_epoch6.pt) |

> Archs with the `_inout` suffix additionally predict in/out-of-frame.
> The VAT checkpoints are also used for ChildPlay and GazeOutsideReal evaluation.

## Loading

```python
import torch
from gazefuze.model import get_gazelle_model

model, transform = get_gazelle_model("gazelle_dinov2_vitl14")
model.load_gazelle_state_dict(torch.load("TextGaze_GF_ViT-L_epoch23.pt", map_location="cpu", weights_only=True))
```

Each release asset is a plain PyTorch state dict (no wrapper).

# Checkpoints

TextGaze checkpoints (PyTorch, `load_gazelle_state_dict`-compatible) — release links
are being prepared:

| Model | Arch | Benchmark | AUC | Avg L2 | AP | Download |
|---|---|---|---|---|---|---|
| TextGaze GF ViT-L | `gazelle_dinov2_vitl14` | GazeFollow | 0.956 | 0.099 | — | *coming soon* |
| TextGaze GF ViT-B | `gazelle_dinov2_vitb14` | GazeFollow | 0.953 | 0.105 | — | *coming soon* |
| TextGaze VAT ViT-L | `gazelle_dinov2_vitl14_inout` | VAT / ChildPlay | 0.941 / 0.950 | 0.102 / 0.111 | 0.911 / 0.996 | *coming soon* |
| TextGaze VAT ViT-B | `gazelle_dinov2_vitb14_inout` | VAT / ChildPlay | 0.932 / 0.945 | 0.113 / 0.112 | 0.879 / 0.992 | *coming soon* |

> Archs with the `_inout` suffix additionally predict in/out-of-frame.

## Loading

```python
import torch
from gazefuze.model import get_gazelle_model

model, transform = get_gazelle_model("gazelle_dinov2_vitl14")
model.load_gazelle_state_dict(torch.load("ckpt.pt", map_location="cpu", weights_only=True))
```

If you need a specific checkpoint urgently, open an issue.

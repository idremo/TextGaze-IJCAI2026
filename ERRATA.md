# Corrections to the camera-ready version

The version of record is the IJCAI-ECAI 2026 proceedings version:
[DOI 10.24963/ijcai.2026/177](https://doi.org/10.24963/ijcai.2026/177).

The author's corrected version (arXiv v2) and this repository incorporate the
following corrections. No model, experiment, result, or conclusion was changed.

| # | Location | Camera-ready | Corrected |
|---|---|---|---|
| 1 | Training details: optimizer | AdamW (Loshchilov & Hutter) | Adam (Kingma & Ba, ICLR 2015) |
| 2 | Table 1, One Human, VAT AP | 0.093 | 0.925 |
| 3 | Token-order equation | `[in/out, visual, text]` | `[in/out, text, visual]` |

Notes:

- The optimizer correction matches the released training code
  (`torch.optim.Adam` in `scripts/train_gazefollow.py` and `scripts/train_vat.py`).
- The Table 1 entry was a decimal-point typo in the human reference row;
  TextGaze's own results are unchanged.
- The token-order correction matches the implementation and Fig. 2;
  the model and results are unchanged.

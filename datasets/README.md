# Datasets & Annotations

This directory carries the **test-set annotations** used to evaluate TextGaze, in two
variants per benchmark:

- `<benchmark>/` — original-style annotations (no textual cues)
- `<benchmark>_Qwen_En/` — same structure **plus per-head textual scene cues**
  (`text` field), generated with Qwen2.5-VL-7B-Instruct and used to train/evaluate
  TextGaze.

> **Images are not distributed here.** Download them from each benchmark's official
> source and place them according to the paths below.

## Included now

| Directory | Split | Files |
|---|---|---|
| `childplay/` , `childplay_Qwen_En/` | test | per-sequence CSVs + `test_preprocessed.json` |
| `gazefollow_extended/` , `gazefollow_extended_Qwen_En/` | train+test lists | `*_annotations_release.txt` + `*_preprocessed.json` |
| `videoattentiontarget/` , `videoattentiontarget_Qwen_En/` | test | per-sequence CSVs + JSONs |
| `gooreal/` , `gooreal_Qwen_En/` | split lists | `gooreal_test.txt` |

## Planned release (not yet uploaded)

The following may be released after further review — open an issue if you need them
sooner:

- [ ] ChildPlay **train** split annotations
- [ ] GazeOutsideReal full annotations
- [ ] The exact Qwen2.5-VL prompts & generation configs used for cue synthesis
      (a reference implementation is in `data_prep/preprocess_Qwen_*.py`)

## Annotation formats

### `*_preprocessed.json` (main format, used by `scripts/eval_*.py`)

```jsonc
[
  {
    "path": "relative/video/dir/",        // sequence dir (images inside)
    "width": 1920, "height": 1080,
    "frames": [
      {
        "path": "frame_000123.jpg",       // image file, relative to sequence dir
        "heads": [
          {
            "bbox_norm":  [x1, y1, x2, y2],   // head box, normalized to [0,1]
            "gazex_norm": [gx],               // gaze target x (normalized); -1 = out of frame
            "gazey_norm": [gy],               // gaze target y (normalized)
            "inout": 1,                       // 1 = in frame, 0 = out, 2 = not annotated
            "person_id": 1,
            "text": "..."                     // Qwen_En variants only: scene cue caption
          }
        ]
      }
    ]
  }
]
```

### Per-sequence CSVs

One CSV per video clip; each row is one annotated frame/head with the same fields as
above (kept for compatibility with the original VAT-style tooling). The JSON files
are what the evaluation scripts consume.

### GazeFollow release lists

`train/test_annotations_release.txt` follow the original GazeFollow annotation
format (one line per head: image path, bbox, gaze target, metadata).

## Image sources

| Benchmark | Source |
|---|---|
| GazeFollow | GazeFollow (Recasens et al., 2015) — official MIT distribution |
| VideoAttentionTarget | VAT (Chong et al., ECCV 2020) — request form from authors |
| ChildPlay / GazeOutsideReal | in-house benchmarks; see paper for collection protocol |

## Regenerating textual cues

`data_prep/preprocess_Qwen_*.py` scripts call a local
**Qwen2.5-VL-7B-Instruct** (vLLM or transformers) over each frame + head box and
write the `text` fields. See the script headers for the exact prompt templates.

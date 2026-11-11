# Datasets & Annotations

This directory carries the annotations used to train and evaluate TextGaze.
Benchmark images are **not** distributed here; download them from the original
benchmark distributions and place them according to the paths below.

Each benchmark has two variants:

- `<benchmark>/` — original-style annotations (no textual cues)
- `<benchmark>_Qwen_En/` — same structure plus per-head textual scene cues
  (`text` field), generated with Qwen2.5-VL-7B-Instruct

## Included

| Directory | Split | Files |
|---|---|---|
| `childplay/`, `childplay_Qwen_En/` | test | per-sequence CSVs + `test_preprocessed.json` |
| `gazefollow_extended/`, `gazefollow_extended_Qwen_En/` | train + test | `*_annotations_release.txt` + `*_preprocessed.json` |
| `videoattentiontarget/`, `videoattentiontarget_Qwen_En/` | train + test | per-frame `.txt` + `*_preprocessed.json` |
| `gooreal/`, `gooreal_Qwen_En/` | test | `gooreal_test.txt` |

ChildPlay and GOO-Real are used for **zero-shot evaluation** in the paper; no
training splits are provided for them.

## Annotation formats

### `*_preprocessed.json` (main format, used by `scripts/eval_*.py`)

```jsonc
[
  {
    "path": "relative/video/dir/",
    "width": 1920, "height": 1080,
    "frames": [
      {
        "path": "frame_000123.jpg",
        "heads": [
          {
            "bbox_norm":  [x1, y1, x2, y2],
            "gazex_norm": [gx],
            "gazey_norm": [gy],
            "inout": 1,
            "person_id": 1,
            "text": "..."
          }
        ]
      }
    ]
  }
]
```

The example is abridged: each head also carries absolute-pixel `bbox`, `gazex`,
`gazey`, and the `is_child` / `gaze_class` fields. The `text` field is present in
the `*_Qwen_En` variants only.

### Per-sequence CSVs (ChildPlay)

One CSV per clip, with columns:

```
clip,frame,person_id,bbox_x,bbox_y,bbox_width,bbox_height,gaze_class,gaze_x,gaze_y,is_child
```

Coordinates are absolute pixels. CSVs do not carry text cues; use the
`*_Qwen_En/test_preprocessed.json` file for `text`.

### VAT per-frame `.txt`

One line per annotated frame:

```
frame.jpg,x1,y1,x2,y2,gaze_x,gaze_y
```

Coordinates are absolute pixels.

### GazeFollow release lists

`train/test_annotations_release.txt` follow the original GazeFollow release
format (one line per annotated head).

### GOO-Real list

`gooreal_test.txt` follows the original GOO-Real list format (one line per head).

## Image sources

| Benchmark | Source |
|---|---|
| GazeFollow | GazeFollow (Recasens et al., 2015), official MIT distribution |
| VideoAttentionTarget | VAT (Chong et al., ECCV 2020), request form from authors |
| ChildPlay | ChildPlay (Tafasca et al., 2023), official benchmark distribution |
| GOO-Real | GOO-Real (Tomas et al., 2021), official benchmark distribution |

## Regenerating textual cues

`data_prep/preprocess_Qwen_*.py` contains the reference caption-generation
pipeline. The scripts use Qwen2.5-VL-7B-Instruct and write the `text` field into
the `*_Qwen_En` JSON files. Update the path constants at the top of each script
before running.

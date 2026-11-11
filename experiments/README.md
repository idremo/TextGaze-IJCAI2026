# Experiment layout

Training scripts write each run as:

```
experiments/
└── <exp_name>/
    └── <YYYY-MM-DD_HH-MM-SS>/
        ├── epoch_0.pt
        ├── ...
        └── training_log.txt
```

`--ckpt_save_dir` sets the root (default `./experiments`), `--exp_name` sets the
run family name, and the timestamp directory is created automatically.

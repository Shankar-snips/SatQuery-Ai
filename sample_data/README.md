# sample_data/

Drop dataset subsets here so `backend/training/finetune_bigearthnet.py` and the
evaluation notebooks can find them. Expected layout:

```
sample_data/
├── bigearthnet/                 # training / domain adaptation
│   ├── optical/                 # Sentinel-2 patches, <patch_id>.tif
│   ├── sar/                     # Sentinel-1 patches, <patch_id>.tif (same IDs)
│   └── labels.csv                # patch_id, caption_or_labels
├── vrsbench/                    # evaluation: single-image captioning + grounding
├── rsvqa/                       # evaluation: single-image VQA
└── cdvqa/                       # evaluation: multitemporal change-VQA
```

`labels.csv` example:

```csv
patch_id,caption_or_labels
S2A_MSIL2A_0001,"urban,water"
S2A_MSIL2A_0002,"A dense forest with a winding river through the center."
```

Download links (public, open-source, as referenced in the problem statement):
- BigEarthNet: https://arxiv.org/abs/2603.29630 (paper); the official BigEarthNet.txt
  release is distributed by the dataset's maintainers — place the extracted
  Sentinel-1/Sentinel-2 patch pairs + your text annotations in the layout above.
- VRSBench, RSVQA, CDVQA: use each benchmark's official release/test split for
  evaluation only — do **not** train on these, they are held out for judging.

None of these datasets are bundled with this repository (they are large and
license-gated); this folder just defines the contract the training/eval scripts
expect.

# `outputs/` — Generated artifacts

This directory holds artifacts produced by the training and evaluation pipeline. The **200 MB+ model weights are not tracked** (regenerable); the **small figure files are tracked** so the README's embedded images render correctly on GitHub.

| Subdirectory | Contents | Tracked? | Why |
|---|---|---|---|
| `cv_run_final/summary/` | One-page polished summary figure (PNG) | ✅ Yes | Referenced from README |
| `cv_run_final/eval/` | ROC, PR, confusion matrix, metrics.json, classification report | ✅ Yes | Referenced from README |
| `cv_run_final/gradcam/` | Grad-CAM visualization grid (PNG) | ✅ Yes | Referenced from README |
| `gradcam_trained/` | Earlier Grad-CAM from the 96×96 ResNet-18 model | ✅ Yes | Archived for comparison |
| `cv_run/` | Training-run checkpoints (best_model.pth, final_model.pth) | ❌ No (200 MB+) | Regenerable from `scripts/train.py` |
| `cv_run_final/eval/{targets,predictions,probabilities}.npy` | Raw eval arrays | ❌ No | Large, regenerable |
| `best_model.pth` | The current best model | ❌ No | 200 MB+ checkpoint |
| `eval_final/` | Eval artifacts from the 96×96 ResNet-18 model | ❌ No | Superseded by `cv_run_final/` |
| `pipeline/`, `pipeline_v*/` | Debug run directories | ❌ No | Scratch space |
| `*.log` | Training logs | ❌ No | Transitory |

## How to regenerate everything

```bash
# Train (4-6 hours on RX 7800 XT)
python scripts/train.py --config configs/cv_quality.yaml --device auto

# Evaluate + Grad-CAM + summary figure (~5 min on GPU)
python scripts/run_pipeline.py --config configs/cv_quality.yaml --skip-train \
    --checkpoint outputs/cv_run/best_model.pth --output-dir outputs/cv_run_final
```

This will re-create every file in `cv_run_final/` from the trained checkpoint.

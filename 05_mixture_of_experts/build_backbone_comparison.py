"""Build backbone_comparison_summary.csv from the three completed backbone runs.

One row per backbone: mean-across-folds fusion_val_acc, ensemble (acc) val acc,
and each of the 4 experts' mean validation accuracy.
"""
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = {
    "DenseNet121": "moe_cnn_fusion_densenet121_5fold_outputs",
    "MobileNetV2": "moe_cnn_fusion_mobilenetv2_5fold_outputs",
    "VGG11": "moe_cnn_fusion_vgg11_5fold_outputs",
}

rows = []
for backbone, out in RUNS.items():
    fm = HERE / out / "metrics" / "MixtureOfExpertsCNNFusion_fold_metrics.csv"
    if not fm.exists():
        print(f"[skip] {backbone}: {fm} missing")
        continue
    df = pd.read_csv(fm)
    per_fold = df[df["fold"].astype(str).str.isdigit()]
    rows.append(
        {
            "backbone": backbone,
            "n_folds": len(per_fold),
            "fusion_val_acc_mean": per_fold["fusion_val_acc"].mean(),
            "ensemble_val_acc_mean": per_fold["acc"].mean(),
            "expert_0_val_acc_mean": per_fold["expert_0_val_acc"].mean(),
            "expert_1_val_acc_mean": per_fold["expert_1_val_acc"].mean(),
            "expert_2_val_acc_mean": per_fold["expert_2_val_acc"].mean(),
            "expert_3_val_acc_mean": per_fold["expert_3_val_acc"].mean(),
        }
    )

out_df = pd.DataFrame(rows).sort_values("fusion_val_acc_mean", ascending=False)
out_path = HERE / "backbone_comparison_summary.csv"
out_df.to_csv(out_path, index=False)
print(out_df.to_string(index=False))
print(f"\nWritten: {out_path}")

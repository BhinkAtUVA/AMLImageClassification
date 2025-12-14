import os
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def analyze_validation(batch_size=BATCH_SIZE, top_misclassified_to_save=2):
    set_seed(42)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    MISCLASS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not VAL_SPLIT_CSV.exists():
        raise FileNotFoundError(f"Missing {VAL_SPLIT_CSV}")

    val_df = pd.read_csv(VAL_SPLIT_CSV).copy()
    val_df["true_label"] = val_df["label"].astype(int)  
    val_df["true0"] = val_df["true_label"] - 1         

    val_ds = BirdsDataset(VAL_SPLIT_CSV, TRAIN_IMAGES_DIR, val_transform, is_train=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = BirdResNet34(NUM_CLASSES, dropout_p=0.2)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_true0 = []
    all_pred0 = []
    all_conf  = []
    all_top5  = []

    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        for imgs, labels0 in val_dl:
            imgs = imgs.to(device, non_blocking=True)
            labels0 = labels0.to(device, non_blocking=True)

            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)

            pred0 = probs.argmax(dim=1)
            conf  = probs.max(dim=1).values
            top5  = probs.topk(5, dim=1).indices  # 0..199

            t_cpu = labels0.cpu().numpy()
            p_cpu = pred0.cpu().numpy()
            c_cpu = conf.cpu().numpy()
            top5_cpu = top5.cpu().numpy()

            for t, p in zip(t_cpu, p_cpu):
                conf_mat[t, p] += 1

            all_true0.extend(t_cpu.tolist())
            all_pred0.extend(p_cpu.tolist())
            all_conf.extend(c_cpu.tolist())
            all_top5.extend([",".join(map(str, (row + 1).tolist())) for row in top5_cpu])  # store as 1..200

    all_true0 = np.array(all_true0)
    all_pred0 = np.array(all_pred0)
    all_conf  = np.array(all_conf)

    correct = (all_true0 == all_pred0).astype(int)
    top1_acc = float(correct.mean())

    total_top5 = 0
    total_n = 0
    with torch.no_grad():
        for imgs, labels0 in val_dl:
            imgs = imgs.to(device, non_blocking=True)
            labels0 = labels0.to(device, non_blocking=True)
            logits = model(imgs)
            top5 = logits.topk(5, dim=1).indices
            total_top5 += top5.eq(labels0.unsqueeze(1)).any(dim=1).sum().item()
            total_n += labels0.size(0)
    top5_acc = float(total_top5 / total_n)

    val_out = val_df.copy()
    val_out["pred_label"] = (all_pred0 + 1).astype(int)   
    val_out["confidence"] = all_conf.astype(float)
    val_out["top5"] = all_top5
    val_out["correct"] = (val_out["true0"].values == all_pred0).astype(int)

    val_out[["image_path", "label", "true_label", "pred_label", "confidence", "top5", "correct"]].to_csv(
        VAL_PRED_CSV, index=False
    )

    per_class_total = np.zeros(NUM_CLASSES, dtype=int)
    per_class_correct = np.zeros(NUM_CLASSES, dtype=int)
    for t, p in zip(all_true0, all_pred0):
        per_class_total[t] += 1
        per_class_correct[t] += int(t == p)

    per_class_acc = per_class_correct / np.maximum(per_class_total, 1)
    per_df = pd.DataFrame({
        "class": np.arange(1, NUM_CLASSES+1),
        "support": per_class_total,
        "correct": per_class_correct,
        "accuracy": per_class_acc
    }).sort_values(["accuracy", "support"], ascending=[True, False])
    per_df.to_csv(PER_CLASS_METRICS_CSV, index=False)

    pairs = []
    for t in range(NUM_CLASSES):
        row = conf_mat[t].copy()
        row[t] = 0
        p = int(row.argmax())
        cnt = int(row[p])
        if cnt > 0:
            pairs.append({
                "true_class": t + 1,
                "most_confused_with": p + 1,
                "count": cnt,
                "true_support": int(conf_mat[t].sum())
            })
    pairs_df = pd.DataFrame(pairs).sort_values("count", ascending=False)
    pairs_df.to_csv(CONFUSED_PAIRS_CSV, index=False)

    plt.figure()
    plt.hist(all_conf[correct == 1], bins=20, alpha=0.7, label="correct")
    plt.hist(all_conf[correct == 0], bins=20, alpha=0.7, label="incorrect")
    plt.xlabel("Max softmax confidence")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CONF_HIST_PNG, dpi=200)
    plt.close()

    bins = np.linspace(0, 1, 11)
    bin_ids = np.digitize(all_conf, bins) - 1
    bin_acc = []
    bin_conf = []
    bin_count = []
    for b in range(10):
        idx = np.where(bin_ids == b)[0]
        if len(idx) == 0:
            bin_acc.append(np.nan); bin_conf.append(np.nan); bin_count.append(0)
        else:
            bin_acc.append(float(correct[idx].mean()))
            bin_conf.append(float(all_conf[idx].mean()))
            bin_count.append(int(len(idx)))

    calib_df = pd.DataFrame({
        "bin": [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(10)],
        "count": bin_count,
        "avg_conf": bin_conf,
        "acc": bin_acc,
    })
    calib_df.to_csv(CALIBRATION_BINS_CSV, index=False)

    plt.figure()
    x = np.array([v for v in bin_conf if not (v is None or (isinstance(v, float) and np.isnan(v)))], dtype=float)
    y = np.array([v for v in bin_acc  if not (v is None or (isinstance(v, float) and np.isnan(v)))], dtype=float)
    plt.plot([0,1], [0,1], linestyle="--", label="ideal")
    if len(x) > 0:
        plt.plot(x, y, marker="o", label="model")
    plt.xlabel("Avg confidence (bin)")
    plt.ylabel("Accuracy (bin)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CALIBRATION_PNG, dpi=200)
    plt.close()

    mistakes = val_out[val_out["correct"] == 0].sort_values("confidence", ascending=False).head(top_misclassified_to_save)
    font = _safe_font(20)

    for i, row in enumerate(mistakes.itertuples(index=False), start=1):
        filename = os.path.basename(row.image_path)
        img_path = TRAIN_IMAGES_DIR / filename
        img = Image.open(img_path).convert("RGB")

        draw = ImageDraw.Draw(img)
        text = f"WRONG #{i}\nTrue: {row.true_label}\nPred: {row.pred_label}\nConf: {row.confidence:.2f}"

        pad = 8
        bbox = draw.multiline_textbbox((0,0), text, font=font)
        box_w = bbox[2] - bbox[0] + 2*pad
        box_h = bbox[3] - bbox[1] + 2*pad
        draw.rectangle([0, 0, box_w, box_h], fill=(0,0,0))
        draw.multiline_text((pad, pad), text, fill=(255,255,255), font=font)

        out_path = MISCLASS_DIR / f"misclassified_{i}_true{row.true_label}_pred{row.pred_label}.png"
        img.save(out_path)

    print("\n[ANALYSIS DONE]")
    print(f"Top-1 acc (val): {top1_acc:.4f}")
    print(f"Top-5 acc (val): {top5_acc:.4f}")
    print(f"Saved files in: {ANALYSIS_DIR}")
    print(f"- {LEARNING_CURVES_CSV.name}, {LEARNING_CURVES_PNG.name}")
    print(f"- {VAL_PRED_CSV.name}, {PER_CLASS_METRICS_CSV.name}, {CONFUSED_PAIRS_CSV.name}")
    print(f"- {CONF_HIST_PNG.name}, {CALIBRATION_BINS_CSV.name}, {CALIBRATION_PNG.name}")
    print(f"- misclassified examples in: {MISCLASS_DIR}")

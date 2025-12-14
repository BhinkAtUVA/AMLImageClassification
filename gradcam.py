from pathlib import Path
import os, numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    VAL_SPLIT_CSV, TRAIN_IMAGES_DIR, ANALYSIS_DIR, BEST_MODEL_PATH,
    val_transform, NUM_CLASSES, BATCH_SIZE
)
from dataloader import BirdsDataset
from resnet_model import BirdResNet34

OUT_DIR = ANALYSIS_DIR / "gradcam_outputs"

def load_model(device):
    model = BirdResNet34(num_classes=NUM_CLASSES, dropout_p=0.2).to(device).eval()
    state = torch.load(BEST_MODEL_PATH, map_location=device)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.load_state_dict(state, strict=True)
    return model

class GradCAM:
    def __init__(self, model, target_module):
        self.model = model
        self.target_module = target_module
        self.activations = None
        self.gradients = None
        target_module.register_forward_hook(self._save_activation)
        # full_backward_hook works for nn.Sequential in PyTorch 2.x
        target_module.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, imgs, class_idx=None):
        # imgs: [N,3,H,W] already normalized
        logits = self.model(imgs)                           # [N,C]
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        one_hot = torch.zeros_like(logits).scatter_(1, class_idx.view(-1,1), 1.0)
        self.model.zero_grad(set_to_none=True)
        logits.backward(gradient=one_hot, retain_graph=True)

        acts = self.activations                              # [N,Ch,h,w]
        grads = self.gradients                               # [N,Ch,h,w]
        weights = grads.mean(dim=(2,3), keepdim=True)        # GAP over spatial dims
        cam = (weights * acts).sum(dim=1, keepdim=True)      # [N,1,h,w]
        cam = F.relu(cam)
        cam = cam - cam.amin(dim=(2,3), keepdim=True)
        cam = cam / (cam.amax(dim=(2,3), keepdim=True) + 1e-8)
        return logits.detach(), cam.detach().cpu()           # cam in [0,1]

def tensor_from_pil(pil):
    return val_transform(pil)                                # your eval preprocessing

def overlay(rgb_uint8, cam_01, alpha=0.35):
    H, W, _ = rgb_uint8.shape
    cam = Image.fromarray((cam_01 * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    cam = np.asarray(cam).astype(np.float32) / 255.0         # [H,W]
    heat = np.zeros_like(rgb_uint8).astype(np.float32) / 255.0
    heat[..., 0] = cam                                       # red channel as heat
    out = np.clip((1 - alpha) * (rgb_uint8 / 255.0) + alpha * heat, 0, 1)
    return (out * 255).astype(np.uint8)

def pick_two_misclassified(model, device, limit_imgs=512):
    """Return two tuples: (PIL image, tensor, true_idx, pred_idx, conf), one low-conf error and one high-conf error."""
    ds = BirdsDataset(VAL_SPLIT_CSV, TRAIN_IMAGES_DIR, transform=None, is_train=True)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    low_err, high_err = None, None
    seen = 0
    with torch.no_grad():
        for imgs_raw, labels0 in dl:
            # convert PIL later to keep originals for overlays
            pil_list = []
            ten_list = []
            for i in range(imgs_raw.size(0)):
                # reconstruct PIL from tensor without normalization, so reload from disk instead:
                # we have paths in ds.df, match by running index
                pass
            # simpler, re-open from disk for this batch
            start = seen
            end = start + imgs_raw.size(0)
            rows = ds.df.iloc[start:end]
            for rel in rows["image_path"].values:
                pil = Image.open(TRAIN_IMAGES_DIR / os.path.basename(rel)).convert("RGB")
                pil_list.append(pil)
                ten_list.append(tensor_from_pil(pil))
            x = torch.stack(ten_list).to(device, non_blocking=True)
            labels0 = labels0.to(device, non_blocking=True)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred0 = probs.max(dim=1)

            wrong = pred0.ne(labels0)
            if wrong.any():
                conf_wrong = conf[wrong].cpu().numpy()
                idxs = torch.where(wrong)[0].cpu().numpy()
                # high confidence error
                hi_local = idxs[int(conf_wrong.argmax())]
                # low confidence error
                lo_local = idxs[int(conf_wrong.argmin())]

                hi = (
                    pil_list[hi_local],
                    ten_list[hi_local],
                    int(labels0[hi_local].item()),
                    int(pred0[hi_local].item()),
                    float(conf[hi_local].item()),
                )
                lo = (
                    pil_list[lo_local],
                    ten_list[lo_local],
                    int(labels0[lo_local].item()),
                    int(pred0[lo_local].item()),
                    float(conf[lo_local].item()),
                )
                # keep best seen so far
                if high_err is None or hi[4] > high_err[4]:
                    high_err = hi
                if low_err is None or lo[4] < low_err[4]:
                    low_err = lo
            seen += imgs_raw.size(0)
            if seen >= limit_imgs and low_err is not None and high_err is not None:
                break
    return low_err, high_err

def save_panel(pil_img, cam_map, pred_idx, conf, out_path):
    rgb = np.array(pil_img)
    over = overlay(rgb, cam_map, alpha=0.35)
    plt.figure(figsize=(6.2, 3.2))
    plt.subplot(1,2,1); plt.imshow(rgb); plt.title("image"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(over); plt.title(f"Grad-CAM, pred {pred_idx+1}, conf {conf:.2f}"); plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    gradcam = GradCAM(model, model.layer4)                   # last conv stack in your ResNet-34

    # pick one low-confidence error and one high-confidence error from validation
    low_err, high_err = pick_two_misclassified(model, device, limit_imgs=512)

    for tag, pack in [("lowconf_error", low_err), ("highconf_error", high_err)]:
        if pack is None:
            print(f"[warn] no {tag} found in scanned subset, expand limit_imgs if needed")
            continue
        pil, ten, true_idx, pred_idx, conf = pack
        x = ten.unsqueeze(0).to(device)
        logits, cams = gradcam(x, class_idx=torch.tensor([pred_idx], device=device))
        cam = cams[0,0].numpy()
        out = OUT_DIR / f"{tag}_true{true_idx+1}_pred{pred_idx+1}.png"
        save_panel(pil, cam, pred_idx, conf, out)
        print(f"saved {out}")

if __name__ == "__main__":
    main()

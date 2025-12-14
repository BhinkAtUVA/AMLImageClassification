import math
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor, device, nn
from torch.utils.data import DataLoader, Dataset
from config import *
from dataloader import BirdsDataset
from transformers.modeling_outputs import ImageClassifierOutput
from resnet_model import BirdResNet34

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs, num_total_epochs):
    def lr_lambda(epoch):
        if epoch < num_warmup_epochs:
            return float(epoch + 1) / float(max(1, num_warmup_epochs))
        progress = (epoch - num_warmup_epochs) / float(max(1, num_total_epochs - num_warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



def mixup_data(x, y, alpha=MIXUP_ALPHA):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, preds, y_a, y_b, lam):
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)



class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert name in self.shadow
            new_avg = self.decay * self.shadow[name] + (1.0 - self.decay) * p.detach()
            self.shadow[name] = new_avg.clone()

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[name] = p.detach().clone()
            p.data.copy_(self.shadow[name].data)

    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p.data.copy_(self.backup[name].data)
        self.backup = {}


def make_splits():
    if TRAIN_SPLIT_CSV.exists() and VAL_SPLIT_CSV.exists():
        return

    df = pd.read_csv(TRAIN_CSV)
    tr, val = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    tr.to_csv(TRAIN_SPLIT_CSV, index=False)
    val.to_csv(VAL_SPLIT_CSV, index=False)


def train_model(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, dropout_p=0.2):
    set_seed(42)
    make_splits()

    train_ds = BirdsDataset(TRAIN_SPLIT_CSV, TRAIN_IMAGES_DIR, train_transform, True)
    val_ds   = BirdsDataset(VAL_SPLIT_CSV,   TRAIN_IMAGES_DIR, val_transform,   True)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BirdResNet34(NUM_CLASSES, dropout_p=dropout_p).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_EPOCHS, num_epochs)

    ema = EMA(model, decay=EMA_DECAY) if USE_EMA else None

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0

        for imgs, labels in train_dl:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            imgs_m, y_a, y_b, lam = mixup_data(imgs, labels, MIXUP_ALPHA)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs_m)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            if ema is not None and (epoch + 1) >= EMA_WARMUP_EPOCHS:
                ema.update(model)

            loss_sum += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = loss_sum / total
        train_acc = correct / total

        model.eval()
        if ema is not None:
            ema.apply_shadow(model)

        vtotal, vcorrect, vloss_sum = 0, 0, 0.0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(imgs)
                loss = criterion(logits, labels)

                vloss_sum += loss.item() * imgs.size(0)
                vcorrect += (logits.argmax(1) == labels).sum().item()
                vtotal += labels.size(0)

        val_loss = vloss_sum / vtotal
        val_acc = vcorrect / vtotal

        if ema is not None:
            ema.restore(model)

        scheduler.step()

        print(
            f"Epoch {epoch+1:03d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

            if ema is not None:
                ema.apply_shadow(model)
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                ema.restore(model)
            else:
                torch.save(model.state_dict(), BEST_MODEL_PATH)

    print(f"Best val accuracy: {best_val_acc:.6f} at epoch {best_epoch}")



def create_test_csv():
    sample = pd.read_csv(TEST_SAMPLE_CSV)
    paths  = pd.read_csv(TEST_PATHS_CSV)

    merged = sample[["id"]].merge(paths, on="id", how="left")
    out = BASE_DIR / "test_final_paths.csv"
    merged.to_csv(out, index=False)
    return out


def five_crop_tensor(img_tensor: torch.Tensor, crop_size=224):
    _, H, W = img_tensor.shape
    cs = crop_size
    assert H >= cs and W >= cs

    tl = img_tensor[:, 0:cs, 0:cs]
    tr = img_tensor[:, 0:cs, W-cs:W]
    bl = img_tensor[:, H-cs:H, 0:cs]
    br = img_tensor[:, H-cs:H, W-cs:W]
    cy = (H - cs) // 2
    cx = (W - cs) // 2
    cc = img_tensor[:, cy:cy+cs, cx:cx+cs]
    return [tl, tr, bl, br, cc]


def predict_test(batch_size=BATCH_SIZE):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_csv = create_test_csv()
    test_df = pd.read_csv(test_csv)

    model = BirdResNet34(NUM_CLASSES, dropout_p=0.2)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_ids = []
    all_preds = []

    for start in range(0, len(test_df), batch_size):
        batch = test_df.iloc[start:start+batch_size]
        ids = batch["id"].values.tolist()
        img_paths = batch["image_path"].values.tolist()

        imgs_256 = []
        for rel in img_paths:
            filename = os.path.basename(rel)
            img_path = TEST_IMAGES_DIR / filename
            img = Image.open(img_path).convert("RGB")
            img_t = tta_base(img)  # [3,256,256]
            imgs_256.append(img_t)

        views = []
        view_to_img = []
        for i, img_t in enumerate(imgs_256):
            crops = five_crop_tensor(img_t, crop_size=IMG_SIZE)
            for c in crops:
                views.append(c)
                view_to_img.append(i)
                views.append(torch.flip(c, dims=[2]))
                view_to_img.append(i)

        views_batch = torch.stack(views, dim=0).to(device, non_blocking=True)

        with torch.no_grad():
            logits = model(views_batch)
            probs = torch.softmax(logits, dim=1)

        probs = probs.cpu()
        agg = torch.zeros((len(imgs_256), NUM_CLASSES), dtype=probs.dtype)
        counts = torch.zeros((len(imgs_256),), dtype=torch.int32)

        for v_idx, img_idx in enumerate(view_to_img):
            agg[img_idx] += probs[v_idx]
            counts[img_idx] += 1

        agg = agg / counts.unsqueeze(1)
        preds = agg.argmax(dim=1).numpy() + 1

        all_ids.extend(ids)
        all_preds.extend(preds.tolist())

    sub = pd.DataFrame({"id": all_ids, "label": all_preds})
    sub.to_csv(SUBMISSION_PATH, index=False)

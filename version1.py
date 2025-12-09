import os
import math
import random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


BASE_DIR = Path("/content")

TRAIN_CSV        = BASE_DIR / "train_images.csv"
TEST_SAMPLE_CSV  = BASE_DIR / "test_images_sample.csv"
TEST_PATHS_CSV   = BASE_DIR / "test_images_path.csv"
TRAIN_IMAGES_DIR = BASE_DIR / "train_images" / "train_images"
TEST_IMAGES_DIR  = BASE_DIR / "test_images" / "test_images"
TRAIN_SPLIT_CSV  = BASE_DIR / "train_split.csv"
VAL_SPLIT_CSV    = BASE_DIR / "val_split.csv"

BEST_MODEL_PATH  = BASE_DIR / "result.pth"
SUBMISSION_PATH  = BASE_DIR / "result.csv"

NUM_CLASSES = 200
IMG_SIZE = 224

#Hyperparams
NUM_EPOCHS = 200
BATCH_SIZE = 64
BASE_LR = 3e-4
WEIGHT_DECAY = 1e-3
MIXUP_ALPHA = 0.4
WARMUP_EPOCHS = 50

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class BirdsDataset(Dataset):
    """
    Handles both train/val CSV (with labels)
    and test CSV (with id, no labels).
    """
    def __init__(self, csv_path, images_root, transform=None, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.images_root = images_root
        self.transform = transform
        self.is_train = is_train

        if self.is_train:
            self.image_paths = self.df["image_path"].values
            self.labels = self.df["label"].values - 1
        else:
            self.image_paths = self.df["image_path"].values
            self.ids = self.df["id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel = self.image_paths[idx]
        filename = os.path.basename(rel)
        img_path = self.images_root / filename

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.is_train:
            label = int(self.labels[idx])
            return img, label
        else:
            return img, int(self.ids[idx])

train_transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.4),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
])

val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])


class BasicModel(nn.Module): 
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.down = None

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        if self.down is not None:
            identity = self.down(x)

        out += identity
        out = F.relu(out, inplace=True)
        return out


class BetterBirdCNN(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),                
        )

        self.layer1 = nn.Sequential(
            BasicModel(64, 64, 1),
            BasicModel(64, 64, 1),
        )

        self.layer2 = nn.Sequential(
            BasicModel(64, 128, 2),
            BasicModel(128, 128, 1),
        )

        self.layer3 = nn.Sequential(
            BasicModel(128, 256, 2),
            BasicModel(256, 256, 1),
        )

        self.layer4 = nn.Sequential(
            BasicModel(256, 512, 2),
            BasicModel(512, 512, 1),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)      
        x = self.layer1(x)   
        x = self.layer2(x)    
        x = self.layer3(x)    
        x = self.layer4(x)    
        x = self.pool(x)      
        x = x.flatten(1)      
        x = self.drop(x)
        x = self.fc(x)
        return x

def get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs, num_total_epochs):
    def lr_lambda(epoch):
        if epoch < num_warmup_epochs:
            return float(epoch + 1) / float(max(1, num_warmup_epochs))
        progress = (epoch - num_warmup_epochs) / float(
            max(1, num_total_epochs - num_warmup_epochs)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def mixup_data(x, y, alpha=MIXUP_ALPHA):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, preds, y_a, y_b, lam):
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)



def make_splits():
    if TRAIN_SPLIT_CSV.exists() and VAL_SPLIT_CSV.exists():
        return

    df = pd.read_csv(TRAIN_CSV)
    tr, val = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )
    tr.to_csv(TRAIN_SPLIT_CSV, index=False)
    val.to_csv(VAL_SPLIT_CSV, index=False)


def train_model(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
    set_seed(42)
    make_splits()

    train_ds = BirdsDataset(TRAIN_SPLIT_CSV, TRAIN_IMAGES_DIR, train_transform, True)
    val_ds   = BirdsDataset(VAL_SPLIT_CSV,   TRAIN_IMAGES_DIR, val_transform, True)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BetterBirdCNN(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_epochs=WARMUP_EPOCHS,
        num_total_epochs=num_epochs,
    )

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0

        for imgs, labels in train_dl:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            imgs_m, targets_a, targets_b, lam = mixup_data(imgs, labels, MIXUP_ALPHA)

            optimizer.zero_grad()
            logits = model(imgs_m)
            loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            loss_sum += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = loss_sum / total
        train_acc = correct / total

        model.eval()
        vtotal, vcorrect, vloss_sum = 0, 0, 0.0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(imgs)
                loss = criterion(logits, labels)

                vloss_sum += loss.item() * imgs.size(0)
                vpreds = logits.argmax(1)
                vcorrect += (vpreds == labels).sum().item()
                vtotal += labels.size(0)

        val_loss = vloss_sum / vtotal
        val_acc = vcorrect / vtotal

        scheduler.step()

        print(
            f"Epoch {epoch+1:02d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    print(f"Best val accuracy: {best_val_acc:.6f} at epoch {best_epoch}")

def create_test_csv():
    sample = pd.read_csv(TEST_SAMPLE_CSV)
    paths  = pd.read_csv(TEST_PATHS_CSV)
    merged = sample[["id"]].merge(paths, on="id", how="left")
    out = BASE_DIR / "test_final_paths.csv"
    merged.to_csv(out, index=False)
    return out


def predict_test(batch_size=BATCH_SIZE):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_csv = create_test_csv()
    test_ds = BirdsDataset(test_csv, TEST_IMAGES_DIR, val_transform, False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = BetterBirdCNN(NUM_CLASSES)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    ids, preds = [], []

    with torch.no_grad():
        for imgs, im_ids in test_dl:
            imgs = imgs.to(device, non_blocking=True)

            logits1 = model(imgs)
            imgs_flipped = torch.flip(imgs, dims=[3])  
            logits2 = model(imgs_flipped)

            logits = (logits1 + logits2) / 2.0
            p = logits.argmax(1).cpu().numpy() + 1 

            preds.extend(p.tolist())
            ids.extend(im_ids.numpy().tolist())

    df = pd.DataFrame({"id": ids, "label": preds})
    df.to_csv(SUBMISSION_PATH, index=False)

if __name__ == "__main__":
    train_model(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    predict_test(batch_size=BATCH_SIZE)

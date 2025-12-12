import os
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
TRAIN_SPLIT_CSV  = BASE_DIR / "train_split_mnet.csv"
VAL_SPLIT_CSV    = BASE_DIR / "val_split_mnet.csv"
BEST_MODEL_PATH  = BASE_DIR / "mobilenetv2.pth"
SUBMISSION_PATH  = BASE_DIR / "mobilenetv2.csv"

NUM_CLASSES = 200
IMG_SIZE = 224

NUM_EPOCHS   = 500
BATCH_SIZE   = 64
BASE_LR      = 3e-4
WEIGHT_DECAY = 1e-4


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class BirdsDataset(Dataset):
    def __init__(self, csv_path, images_root, transform=None, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.images_root = images_root
        self.transform = transform
        self.is_train = is_train

        self.image_paths = self.df["image_path"].values
        if self.is_train:
            self.labels = self.df["label"].values - 1
        else:
            self.ids = self.df["id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.images_root / os.path.basename(self.image_paths[idx])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.is_train:
            return img, int(self.labels[idx])
        else:
            return img, int(self.ids[idx])


train_transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])



def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (self.stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            # pointwise
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        # depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])
        # pointwise-linear
        layers.extend([
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=NUM_CLASSES,
                 width_mult=1.0,
                 round_nearest=8):
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        # t, c, n, s (expansion, channels, num_blocks, stride)
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = [
            nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        ]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        features.extend([
            nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True),
        ])

        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_splits():
    if TRAIN_SPLIT_CSV.exists() and VAL_SPLIT_CSV.exists():
        return

    df = pd.read_csv(TRAIN_CSV)
    tr, val = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )
    tr.to_csv(TRAIN_SPLIT_CSV, index=False)
    val.to_csv(VAL_SPLIT_CSV, index=False)


def train_mobilenetv2(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
    set_seed(42)
    make_splits()

    train_ds = BirdsDataset(TRAIN_SPLIT_CSV, TRAIN_IMAGES_DIR, train_transform, True)
    val_ds   = BirdsDataset(VAL_SPLIT_CSV,   TRAIN_IMAGES_DIR, val_transform, True)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MobileNetV2(num_classes=NUM_CLASSES, width_mult=1.0).to(device)
    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0.0
    best_epoch   = 0

    for epoch in range(num_epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0

        for imgs, labels in train_dl:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        train_loss = loss_sum / total
        train_acc  = correct / total

        model.eval()
        vtotal, vcorrect, vloss_sum = 0, 0, 0.0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs   = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(imgs)
                loss   = criterion(logits, labels)

                vloss_sum += loss.item() * imgs.size(0)
                vpreds = logits.argmax(1)
                vcorrect += (vpreds == labels).sum().item()
                vtotal   += labels.size(0)

        val_loss = vloss_sum / vtotal
        val_acc  = vcorrect / vtotal

        scheduler.step()

        print(
            f"Epoch {epoch+1:03d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch + 1
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    print(f"Best val accuracy: {best_val_acc:.6f} at epoch {best_epoch}")



def create_test_csv():
    sample = pd.read_csv(TEST_SAMPLE_CSV)
    paths  = pd.read_csv(TEST_PATHS_CSV)
    merged = sample[["id"]].merge(paths, on="id", how="left")
    out = BASE_DIR / "test_final_paths_mnv2.csv"
    merged.to_csv(out, index=False)
    return out



def predict_test_mobilenet(model_class, weights_path, batch_size=64):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_csv = create_test_csv()
    test_ds = BirdsDataset(test_csv, TEST_IMAGES_DIR, val_transform, is_train=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = model_class(NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    all_ids = []
    all_preds = []

    with torch.no_grad():
        for imgs, ids in test_dl:
            imgs = imgs.to(device)


            logits1 = model(imgs)
            logits2 = model(torch.flip(imgs, dims=[3]))

            logits = (logits1 + logits2) / 2

            preds = logits.argmax(1).cpu().numpy() + 1

            all_ids.extend(ids.tolist())
            all_preds.extend(preds.tolist())

    df = pd.DataFrame({
        "id": all_ids,
        "label": all_preds
    })

    df.to_csv(SUBMISSION_PATH, index=False)



if __name__ == "__main__":
    train_mobilenetv2(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    predict_test_mobilenet(MobileNetV2, BEST_MODEL_PATH, batch_size=BATCH_SIZE)

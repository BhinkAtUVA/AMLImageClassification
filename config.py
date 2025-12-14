from pathlib import Path
import torch
import torchvision.transforms as T

BASE_DIR = Path("./content")

TRAIN_CSV        = BASE_DIR / "train_images.csv"
TEST_SAMPLE_CSV  = BASE_DIR / "test_images_sample.csv"
TEST_PATHS_CSV   = BASE_DIR / "test_images_path.csv"
TRAIN_IMAGES_DIR = BASE_DIR / "train_images" / "train_images"
TEST_IMAGES_DIR  = BASE_DIR / "test_images" / "test_images"
TRAIN_SPLIT_CSV  = BASE_DIR / "train_split.csv"
VAL_SPLIT_CSV    = BASE_DIR / "val_split.csv"
BEST_MODEL_PATH  = BASE_DIR / "result_v4.pth"
SUBMISSION_PATH  = BASE_DIR / "result_v4.csv"

BASELINE_DIR = BASE_DIR / "baseline_model"
BASELINE_PRED = BASELINE_DIR / "pred.csv"

NUM_CLASSES = 200
IMG_SIZE = 224

NUM_EPOCHS    = 1000
BATCH_SIZE    = 64
BASE_LR       = 3e-4
WEIGHT_DECAY  = 1e-3
MIXUP_ALPHA   = 0.4
WARMUP_EPOCHS = 50

# EMA hyperparams
USE_EMA = True
EMA_DECAY = 0.999
EMA_WARMUP_EPOCHS = 1

train_transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([T.RandomPerspective(distortion_scale=0.3, p=1.0)], p=0.3),
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

tta_base = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

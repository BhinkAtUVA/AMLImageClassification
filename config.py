from pathlib import Path
from torchvision.transforms import v2

BASE_DIR = Path("/content")

TRAIN_CSV        = BASE_DIR / "train_images.csv"
TEST_SAMPLE_CSV  = BASE_DIR / "test_images_sample.csv"
TEST_PATHS_CSV   = BASE_DIR / "test_images_path.csv"
TRAIN_IMAGES_DIR = BASE_DIR / "train_images" / "train_images"
TEST_IMAGES_DIR  = BASE_DIR / "test_images" / "test_images"
TRAIN_SPLIT_CSV  = BASE_DIR / "train_split.csv"
VAL_SPLIT_CSV    = BASE_DIR / "val_split.csv"
BEST_MODEL_PATH  = BASE_DIR / "result_v4.pth"
SUBMISSION_PATH  = BASE_DIR / "result_v4.csv"

IMG_SIZE = 224
NUM_CLASSES = 200
TRANSFORM_TRAIN = v2.Compose(
    [
        v2.Resize((256, 256)),
        v2.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply([v2.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.4),
        v2.RandomRotation(10),
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
        v2.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ]
)
TRANSFORM_VAL = v2.Compose(
    [
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.CenterCrop(IMG_SIZE),
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ]
)

# Hyperparams
NUM_EPOCHS = 200
BATCH_SIZE = 64
BASE_LR = 3e-4
WEIGHT_DECAY = 1e-3
MIXUP_ALPHA = 0.4
WARMUP_EPOCHS = 50
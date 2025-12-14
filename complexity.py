from fvcore.nn import FlopCountAnalysis
from transformers import AutoFeatureExtractor, SwinForImageClassification
from MobileNetV2 import MobileNetV2
from config import *
from resnet_model import BirdResNet34
from PIL import Image

feature_extractor = AutoFeatureExtractor.from_pretrained(BASELINE_DIR)
baseline = SwinForImageClassification.from_pretrained(BASELINE_DIR)

img = Image.open("content/train_images/train_images/1.jpg").convert("RGB")
baseline_input = feature_extractor(img, return_tensors="pt").pixel_values[0]
own_input = val_transform(img)

resnet_flops = FlopCountAnalysis(BirdResNet34(), (own_input[None, :])).total()
baseline_flops = FlopCountAnalysis(baseline, (baseline_input[None, :])).total()
mobilenet_flops = FlopCountAnalysis(MobileNetV2(), (own_input[None, :])).total()

print(f"Baseline FLOPS: {baseline_flops}") #  104080429056
print(f"Resnet FLOPS: {resnet_flops}") #        3682067456
print(f"Mobilenet FLOPS: {mobilenet_flops}") #   333203552
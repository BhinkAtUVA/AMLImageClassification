import torch
from torch import nn
from torch.utils.data import DataLoader
from config import *
import pandas as pd
from dataloader import BirdsDataset
from modelling import make_splits, predict_test
from transformers import SwinForImageClassification, AutoFeatureExtractor


feature_extractor = AutoFeatureExtractor.from_pretrained(BASELINE_DIR)
model = SwinForImageClassification.from_pretrained(BASELINE_DIR)
def curried_feature_extractor(input):
    return feature_extractor(input, return_tensors="pt").pixel_values.squeeze()

if __name__ == "__main__":
    make_splits()
    test_data = BirdsDataset(TEST_PATHS_CSV, TEST_IMAGES_DIR, curried_feature_extractor, False)
    predict_test(model, 16, None, BASELINE_PRED, test_data, True)


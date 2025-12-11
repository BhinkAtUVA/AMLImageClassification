from transformers import AutoFeatureExtractor, SwinForImageClassification
import safetensors

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-large-patch4-window12-384-in22k")
model = SwinForImageClassification.from_pretrained("microsoft/swin-large-patch4-window12-384-in22k")
model.load_state_dict()
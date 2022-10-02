import torch

from torch.utils.mobile_optimizer import optimize_for_mobile
from model import build_model

MODEL_VERSION = "v5"
LITE_VERSION = "v5.1"
IMAGE_SIZE = 256
DEVICE = 'cuda'

model = build_model(pretrained=False, fine_tune=False, num_classes=10)

checkpoint = torch.load(f"../outputs/{MODEL_VERSION}/model_pretrained_True.pth", map_location=DEVICE)

print("Loading trained model weights...")

model.load_state_dict(checkpoint['model_state_dict'], strict=False)

model.eval()

torchscript_model = torch.jit.script(model)
traced_script_module_optimized = optimize_for_mobile(torchscript_model)
traced_script_module_optimized._save_for_lite_interpreter(f"../outputs/app/model_{LITE_VERSION}.ptl")


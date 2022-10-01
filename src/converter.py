import torch

from torch.utils.mobile_optimizer import optimize_for_mobile
from model import build_model

VERSION = "v5"
IMAGE_SIZE = 256
DEVICE = 'cuda'

model = build_model(pretrained=False, fine_tune=False, num_classes=10)

checkpoint = torch.load(f"../outputs/{VERSION}/model_pretrained_True.pth", map_location=DEVICE)

print("Loading trained model weights...")

model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

dummy_input = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
torchscript_model = torch.jit.trace(model, dummy_input)
traced_script_module_optimized = optimize_for_mobile(torchscript_model)
traced_script_module_optimized._save_for_lite_interpreter("../outputs/app/model_v5.ptl")


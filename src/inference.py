import torch
import cv2
import numpy as np
import glob as glob
import os
from model import build_model
from torchvision import transforms
# Constants.
VERSION = "v5"
DATA_PATH = '../test_images'
IMAGE_SIZE = 256
DEVICE = 'cuda'
# Class names.
class_names = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
               'Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
               ]
# Load the trained model.
model = build_model(pretrained=False, fine_tune=False, num_classes=10)
checkpoint = torch.load(f'../outputs/{VERSION}/model_pretrained_True.pth', map_location=DEVICE)
print('Loading trained model weights...')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

# Get all the test image paths.
all_image_paths = glob.glob(f"{DATA_PATH}/*")
# Iterate over all the images and do forward pass.
for image_path in all_image_paths:
    # Get the ground truth class name from the image path.
    gt_class_name = image_path.split(os.path.sep)[-1].split('.')[0]
    # Read the image and create a copy.
    image = cv2.imread(image_path)
    orig_image = image.copy()

    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = transform(image)
    image = torch.unsqueeze(image, 0)
    image = image.to(DEVICE)
    model.to(DEVICE)

    with torch.no_grad():
        # Forward pass throught the image.
        outputs = model(image)

    print(outputs[0])
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    print(probabilities)
    # Check the top 5 categories that are predicted.
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        cv2.putText(orig_image, f"{top5_prob[i].item() * 100:.3f}%", (15, (i + 1) * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(orig_image, f"{class_names[top5_catid[i]]}", (100, (i + 1) * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1, cv2.LINE_AA)
        print(class_names[top5_catid[i]], top5_prob[i].item())

    print()
    cv2.imshow('Result', orig_image)
    cv2.waitKey(0)
    # outputs = outputs.detach().numpy()
    # pred_class_name = class_names[np.argmax(outputs[0])]
    # print(f"GT: {gt_class_name}, Pred: {pred_class_name.lower()}")
    # # Annotate the image with ground truth.
    # cv2.putText(
    #     orig_image, f"GT: {gt_class_name}",
    #     (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
    #     0.6, (0, 255, 0), 2, lineType=cv2.LINE_AA
    # )
    # # Annotate the image with prediction.
    # cv2.putText(
    #     orig_image, f"Pred: {pred_class_name.lower()}",
    #     (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
    #     0.6, (100, 100, 225), 2, lineType=cv2.LINE_AA
    # )
    # cv2.imshow('Result', orig_image)
    # cv2.waitKey(0)
    cv2.imwrite(f"../tests/{VERSION}/{gt_class_name}.png", orig_image)

import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import os
from torcheval.metrics import BinaryAccuracy, BinaryF1Score, BinaryConfusionMatrix, BinaryPrecisionRecallCurve

def clip_pred(imgs):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(
        text=["a synthetic image created by AI", "a real image taken by a human"],
        images=imgs,
        return_tensors="pt",
        padding=True
    )

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    prob = logits_per_image.softmax(dim=1)  
    return prob


def load_images_from_folder(folder):
    images = []
    
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
        
    
    labels = torch.zeros(len(images), dtype=torch.int)
    return images, labels

def metrics(ys, ts):
    acc = BinaryAccuracy()
    f1 = BinaryF1Score()
    cm = BinaryConfusionMatrix()
    acc.update(ys, ts)
    f1.update(ys, ts)
    cm.update(ys, ts)
    return acc.compute(), f1.compute(), cm.compute()


folder_path = r'/dtu/blackhole/18/160664/test/FAKE/'
images, labels = load_images_from_folder(folder_path)
probs = clip_pred(images)
metrics(probs[:,1], labels)
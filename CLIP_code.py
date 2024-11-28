import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import os
from torcheval.metrics import BinaryAccuracy, BinaryF1Score, BinaryConfusionMatrix, BinaryPrecisionRecallCurve

def clip_pred(imgs, imgs_class):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    logits = torch.empty(len(imgs),2)

    for i in range(len(imgs)):
        input = processor(
            text=["a synthetic image of a" + str(imgs_class[i]), "a real image of a" + str(imgs_class[i])],
            images=imgs[i],
            return_tensors="pt",
            padding=True
        )

        output = model(**input)
        logits.cat(logits, output.logits_per_image)  # this is the image-text similarity score
    prob = logits.softmax(dim=1)  
    return prob


def load_images_from_folder(folder):
    images = []
    class_types = {'(2)':'automobile','(3)': 'bird', '(4)': 'cat', '(5)': 'deer', '(6)': 'dog', '(7)': 'frog', '(8)': 'horse', '(9)': 'ship', '(10)': 'truck'}
    img_class = []

    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images.append(img)

            for c in class_types.keys():
                if c in filename:
                    img_class.append(class_types.get(c))
                else:
                    img_class.append('airplane')

            labels = torch.zeros(len(images), dtype=torch.int)
    return images, labels, imgs_class

def metrics(ys, ts):
    acc = BinaryAccuracy()
    f1 = BinaryF1Score()
    cm = BinaryConfusionMatrix()
    prc = BinaryPrecisionRecallCurve()
    acc.update(ys, ts)
    f1.update(ys, ts)
    cm.update(ys, ts)
    prc.update(ys, ts)

    return acc.compute(), f1.compute(), cm.compute(), prc.compute()


folder_path = r'/dtu/blackhole/18/160664/test/FAKE/'
images, labels, imgs_class = load_images_from_folder(folder_path)
probs = clip_pred(images, imgs_class)
metrics(probs[:,1], labels)
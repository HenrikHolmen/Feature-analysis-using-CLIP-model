from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "https://img.freepik.com/free-photo/view-3d-car-sketch-style_23-2151138903.jpg?t=st=1731338376~exp=1731341976~hmac=4b4d843cd016aa2e786e4c76adff61f713804b95e6a077d597b1c4a1049cc49d&w=740"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
    text=["a synthetic photo", "a real photo"],
    images=image,
    return_tensors="pt",
    padding=True
)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

print(probs)
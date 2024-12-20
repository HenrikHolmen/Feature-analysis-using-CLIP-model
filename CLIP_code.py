import torch
from torch import nn
import torch.optim as optim
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
from torcheval.metrics import BinaryAccuracy, BinaryF1Score, BinaryConfusionMatrix, BinaryAUROC
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def clip_pred(imgs_class, class_type, model, processor):
    """
    Predict probabilities and extract features using the CLIP model.
    """
    inputs = processor(
        text=['A human-made photo of a ' + str(class_type), 
              'A synthetic computer-generated photo of a ' + str(class_type)],
        images=imgs_class,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}  # Move inputs to GPU
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)
        img_features = model.get_image_features(pixel_values=inputs['pixel_values']).cpu()
    logits_per_image = outputs.logits_per_image  # Image-text similarity score
    prob = logits_per_image.softmax(dim=1)  # Probability over classes
    return prob[:, 1], img_features

def load_images_from_folder(fake_folder, real_folder):
    """
    Load images and classify them into real and fake.
    """
    images_class = {'airplane': [[], []], 'automobile': [[], []], 'bird': [[], []], 'cat': [[], []],
                    'deer': [[], []], 'dog': [[], []], 'frog': [[], []], 'horse': [[], []], 
                    'ship': [[], []], 'truck': [[], []]}
    class_types = {'(2)': 'automobile', '(3)': 'bird', '(4)': 'cat', '(5)': 'deer', '(6)': 'dog', 
                   '(7)': 'frog', '(8)': 'horse', '(9)': 'ship', '(10)': 'truck'}

    for filename in os.listdir(real_folder):
        img_path = os.path.join(real_folder, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path).convert("RGB")
            for c in class_types.keys():
                if c in filename:
                    images_class[class_types[c]][0].append(img)
                    images_class[class_types[c]][1].append(0)  # Real
                    break
            if all(c not in filename for c in class_types.keys()):
                images_class['airplane'][0].append(img)
                images_class['airplane'][1].append(0)  # Real

    for filename in os.listdir(fake_folder):
        img_path = os.path.join(fake_folder, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path).convert("RGB")
            for c in class_types.keys():
                if c in filename:
                    images_class[class_types[c]][0].append(img)
                    images_class[class_types[c]][1].append(1)  # Fake
                    break
            if all(c not in filename for c in class_types.keys()):
                images_class['airplane'][0].append(img)
                images_class['airplane'][1].append(1)  # Fake
    return images_class

def evaluate_model(imgs_class):
    """
    Evaluate the CLIP model using mini-batch processing and calculate metrics.
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    probs, labels, image_features = [], [], []

    for i, class_type in enumerate(imgs_class.keys()):
        print(f"Processing class {class_type} ({i + 1}/{len(imgs_class.keys())})")
        batch_probs, batch_features = clip_pred(imgs_class[class_type][0], class_type, model, processor)
        probs.extend(batch_probs.cpu().tolist())
        labels.extend(imgs_class[class_type][1])
        image_features.extend(batch_features.cpu())
        torch.cuda.empty_cache()

    probs = torch.tensor(probs)
    labels = torch.tensor(labels, dtype=torch.int)
    dataset = {'features': image_features, 'labels': labels}

    acc = BinaryAccuracy()
    f1 = BinaryF1Score()
    cm = BinaryConfusionMatrix()
    auroc = BinaryAUROC()

    acc.update(probs, labels)
    f1.update(probs, labels)
    cm.update(probs, labels)
    auroc.update(probs, labels)

    print(f"Accuracy: {acc.compute():.4f}")
    print(f"F1 Score: {f1.compute():.4f}")
    print(f"Confusion Matrix: \n{cm.compute()}")
    print(f"AUROC: {auroc.compute():.4f}")

    return dataset

def train_loop(train_loader, model, loss, optimizer, num_epochs):
    """
    Training loop for the feed-forward neural network.
    """
    model.train()
    acc = BinaryAccuracy()

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_acc = 0
        for batch_features, batch_labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            batch_features, batch_labels = batch_features.to("cuda"), batch_labels.to("cuda")

            optimizer.zero_grad()
            batch_output = model(batch_features)
            batch_loss = loss(batch_output.squeeze(), batch_labels)
            batch_loss.backward()
            optimizer.step()

            acc.update(batch_output.squeeze(), batch_labels)
            epoch_acc += acc.compute().item()
            epoch_loss += batch_loss.item()

        avg_loss = epoch_loss / len(train_loader)
        avg_acc = epoch_acc / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
        acc.reset()
        torch.cuda.empty_cache()

def test_network(test_loader, model, loss):
    """
    Testing loop for the feed-forward neural network.
    """
    model.eval()
    acc = BinaryAccuracy()
    auroc = BinaryAUROC()
    cm = BinaryConfusionMatrix()
    test_loss = 0

    with torch.no_grad():
        for batch_features, batch_labels in tqdm(test_loader, desc="Testing"):
            batch_features, batch_labels = batch_features.to("cuda"), batch_labels.to("cuda")
            
            batch_outputs = model(batch_features).squeeze()
            batch_loss = loss(batch_outputs, batch_labels)
            test_loss += batch_loss.item()

            acc.update(batch_outputs, batch_labels)
            cm.update(batch_outputs, batch_labels)
            auroc.update(batch_outputs, batch_labels)

    avg_loss = test_loss / len(test_loader)
    print(f"Average Test Loss: {avg_loss:.4f}")
    print(f"Accuracy: {acc.compute():.4f}")
    print(f"AUROC: {auroc.compute():.4f}")
    print(f"Confusion Matrix:\n{cm.compute()}")

class dict_to_data(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels.float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

class network(nn.Module):
    def __init__(self, num_features, num_hidden):
        super(network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# File paths
fake_train = r"/dtu/blackhole/18/160664/train/FAKE"
real_train = r"/dtu/blackhole/18/160664/train/REAL"

# Load training data
imgs_class = load_images_from_folder(fake_train, real_train)
feature_data_train = evaluate_model(imgs_class)
train_dataset = dict_to_data(feature_data_train['features'], feature_data_train['labels'])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)

# Train the network
num_features = len(train_dataset.features[0])
ffnn = network(num_features, 256).to("cuda")
train_loop(train_loader, ffnn, nn.BCELoss(), optim.Adam(ffnn.parameters(), lr=1e-4), num_epochs=70)

# Test the network
fake_test = r"/dtu/blackhole/18/160664/test/FAKE"
real_test = r"/dtu/blackhole/18/160664/test/REAL"
test_imgs_class = load_images_from_folder(fake_test, real_test)
feature_data_test = evaluate_model(test_imgs_class)
test_dataset = dict_to_data(feature_data_test['features'], feature_data_test['labels'])
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)

test_network(test_loader, ffnn, nn.BCELoss())
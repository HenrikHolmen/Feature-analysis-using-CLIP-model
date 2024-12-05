import torch
from torch import nn
import torch.optim as optim
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import os
from torcheval.metrics import BinaryAccuracy, BinaryF1Score, BinaryConfusionMatrix, BinaryAUROC
from torch.utils.data import DataLoader, Dataset

def clip_pred(imgs_class, class_type, model, processor):

    # Process all images and prompts in a single batch
    inputs = processor(
        text= ['A human-made photo of a' + str(class_type), 'A synthetic computer-generated photo of a' + str(class_type)],  # Prompts for each image
        images=imgs_class,  # Duplicate images to match the number of prompts
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.to("cpu") for k, v in inputs.items()}  # Move inputs to GPU
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)
        img_features = model.get_image_features(pixel_values=inputs['pixel_values'])
    logits_per_image = outputs.logits_per_image  # Image-text similarity score
    prob = logits_per_image.softmax(dim=1)  # Probability over classes
    return prob[:,1], img_features


def load_images_from_folder(fake_folder, real_folder):
    images_class = {'airplane': [[],[]], 'automobile': [[],[]], 'bird': [[],[]], 'cat': [[],[]], 'deer': [[],[]], 'dog': [[],[]], 'frog': [[],[]], 'horse': [[],[]], 'ship': [[],[]], 'truck': [[],[]]}
    class_types = {'(2)': 'automobile', '(3)': 'bird', '(4)': 'cat', '(5)': 'deer', '(6)': 'dog', '(7)': 'frog', '(8)': 'horse', '(9)': 'ship', '(10)': 'truck'}

    for filename in os.listdir(real_folder):
        img_path = os.path.join(real_folder, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path).convert("RGB")

            for c in class_types.keys():
                if c in filename:
                    images_class[class_types.get(c)][0].append(img)
                    images_class[class_types.get(c)][1].append(0)
                    break
            if all(c not in filename for c in class_types.keys()):
                images_class['airplane'][0].append(img)
                images_class['airplane'][1].append(0)

    for filename in os.listdir(fake_folder):
        img_path = os.path.join(fake_folder, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path).convert("RGB")

            for c in class_types.keys():
                if c in filename:
                    images_class[class_types.get(c)][0].append(img)
                    images_class[class_types.get(c)][1].append(1)
                    break
            if all(c not in filename for c in class_types.keys()):
                images_class['airplane'][0].append(img)
                images_class['airplane'][1].append(1)
    return images_class

def evaluate_model(imgs_class):
    """
    Evaluate the CLIP model using mini-batch processing and calculate metrics.
    """
    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cpu")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Process images in mini-batches
    probs = []
    labels = []
    image_features = []
    for i in imgs_class:
        batch_probs, batch_features = clip_pred(imgs_class[i][0], i,  model, processor)
        probs.extend(batch_probs)
        labels.extend(imgs_class[i][1])
        for f in batch_features:
            image_features.append(f)
    
    probs = torch.tensor(probs)
    labels = torch.tensor(labels, dtype=torch.int)
    dataset = {'features': image_features, 'labels': labels}
    
    # Calculate metrics
    acc = BinaryAccuracy()
    f1 = BinaryF1Score()
    cm = BinaryConfusionMatrix()
    auroc = BinaryAUROC()
    
    acc.update(probs, labels)
    f1.update(probs, labels)
    cm.update(probs, labels)
    auroc.update(probs, labels)
    
    accuracy = acc.compute()
    f1_score = f1.compute()
    confusion_matrix = cm.compute()
    auroc_score = auroc.compute()
    
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1_score}")
    print(f"Confusion Matrix: \n{confusion_matrix}")
    print(f"AUROC: {auroc_score}")

    return dataset

def train_loop(train_loader, model, loss, optimizer, num_epochs):
    model.train()
    acc = BinaryAccuracy()

    for epoch in range(num_epochs):
        for batch_features, batch_labels in train_loader:  
            batch_features, batch_labels = batch_features.to("cpu"), batch_labels.to("cpu")

            batch_output = model(batch_features)
            batch_loss = loss(batch_output.squeeze(), batch_labels)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            acc.update(batch_output.squeeze(), batch_labels)
            batch_accuracy = acc.compute()
            acc.reset()
            
        print(f"Epoch {epoch+1}, Loss: {batch_loss}, Accuracy: {batch_accuracy}")
    
def test_network(test_loader, model, loss):
    model.eval()
    test_loss = 0
    test_acc = 0
    acc = BinaryAccuracy()
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to('cpu'), batch_labels.to('cpu')

            # Forward pass
            batch_outputs = model(batch_features).squeeze()
            batch_loss = loss(batch_outputs, batch_labels)
            test_loss += batch_loss
            
            acc.update(batch_outputs, batch_labels)
            batch_accuracy = acc.compute()
            test_acc += batch_accuracy
            acc.reset()
            print(f'Test Loss: {batch_loss:.4f}, Test Accuracy: {batch_accuracy:.4f}')

    avg_loss = test_loss / len(test_loader)
    avg_acc =  test_acc / len(test_loader)

    print(f'Average test loss: {avg_loss:.4f}, Average test accuracy: {avg_acc:.4f}')

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
    

fake_train = r"/dtu/blackhole/18/160664/train/FAKE"
real_train = r"/dtu/blackhole/18/160664/train/REAL"

imgs_class = load_images_from_folder(fake_train, real_train)
# Evaluate the model
feature_data_train = evaluate_model(imgs_class)

train_dataset = dict_to_data(feature_data_train['features'], feature_data_train['labels'])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)

num_features = len(train_dataset.features[0])
num_hidden = 256
ffnn = network(num_features, num_hidden)
train_loop(train_loader, ffnn, nn.BCELoss(), optim.Adam(ffnn.parameters(), lr=0.0001), num_epochs = 70)

fake_test = r"/dtu/blackhole/18/160664/test/FAKE"
real_test = r"/dtu/blackhole/18/160664/test/REAL"
test_imgs_class = load_images_from_folder(fake_test, real_test)

feature_data_test = evaluate_model(test_imgs_class)
test_dataset = dict_to_data(feature_data_test['features'], feature_data_test['labels'])
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)
test_network(test_loader ,ffnn, nn.BCELoss())


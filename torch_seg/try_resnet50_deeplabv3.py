'''
backbone: resnet50
segmentation_head: DeepLabV3Head
'''
import torch
import torchvision
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np

# Load the pretrained DeepLabV3 model with a ResNet50 backbone
model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)

# Number of effective classes after mapping (19 classes + 1 background)
num_classes = 20

# Replace the classifier of the model
model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

# Mapping for reducing classes to 20 including background
mapping_20 = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 2, 9: 0,
    10: 0, 11: 3, 12: 4, 13: 5, 14: 0, 15: 0, 16: 0, 17: 6, 18: 0,
    19: 7, 20: 8, 21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14,
    27: 15, 28: 16, 29: 0, 30: 0, 31: 17, 32: 18, 33: 19, -1: 0
}

def encode_labels(mask):
    label_mask = np.zeros_like(mask)
    for k in mapping_20:
        label_mask[mask == k] = mapping_20[k]
    return label_mask

def transform_target(target):
    target = np.array(target)  # Convert PIL Image to numpy array
    target = encode_labels(target)  # Remap labels
    return torch.as_tensor(target, dtype=torch.int64)  # Convert numpy array to tensor

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_transform = transforms.Compose([
    transform_target
])

# Define the dataset with appropriate transforms for both images and targets
dataset_path = './small_gtFine_trainvaltest'
train_dataset = datasets.Cityscapes(root=dataset_path, split='train', mode='fine', target_type='semantic', transform=transform, target_transform=target_transform)
val_dataset = datasets.Cityscapes(root=dataset_path, split='val', mode='fine', target_type='semantic', transform=transform, target_transform=target_transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# Training loop
num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        print("label_max:", targets.max(), " label_min:", targets.min(), " label_median:", torch.median(targets),
              "label_shape:", targets.shape)

        optimizer.zero_grad()
        outputs = model(images)['out']

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print("loss:", loss.item())

    scheduler.step()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

# Evaluation
model.eval()
total = 0
correct = 0
with torch.no_grad():
    for images, targets in val_loader:
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)['out']
        _, predicted = torch.max(outputs, 1)
        total += targets.nelement()
        correct += (predicted == targets).sum().item()

print(f"Accuracy: {100 * correct / total}%")

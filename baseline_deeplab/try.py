import torch
import torchvision
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np

# Use updated parameter for pretrained models (adapted to warning messages)
model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
model.train()  # Set the model to training mode

# Define custom transformations for both images and targets
def transform_target(target):
    return torch.as_tensor(np.array(target), dtype=torch.int64)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_transform = transforms.Compose([
    transform_target
])

# Define the dataset with appropriate transforms for both images and targets
dataset_path = '../small_gtFine_trainvaltest'
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
        targets = targets.to(device)  # targets are already tensors

        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

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

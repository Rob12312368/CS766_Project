'''1.import pytorch segmentation library
2.define the pretrained model
3.define the dataset cityscapes/CamVid
4.run the training
5.run the evaluation'''
import torch
import torchvision
from torchvision import models, transforms
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# 1. Import the segmentation model from torchvision with pre-trained weights
model = models.segmentation.deeplabv3_resnet50(pretrained=True)
model.train()  # Set the model to training mode

# 2. Define the dataset
# Transformations for the input data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_path = './gtFine_trainvaltest'
# Initialize the Cityscapes dataset
train_dataset = Cityscapes(root=dataset_path, split='train', mode='fine', target_type='semantic', transform=transform)
val_dataset = Cityscapes(root=dataset_path, split='val', mode='fine', target_type='semantic', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 3. Set up the training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets['segmentation'].to(device)  # adjust depending on dataset structure

        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# 4. Evaluate the model
model.eval()
total = 0
correct = 0
with torch.no_grad():
    for images, targets in val_loader:
        images = images.to(device)
        targets = targets['segmentation'].to(device)
        outputs = model(images)['out']
        _, predicted = torch.max(outputs, 1)
        total += targets.nelement()
        correct += (predicted == targets).sum().item()

print(f"Accuracy: {100 * correct / total}%")

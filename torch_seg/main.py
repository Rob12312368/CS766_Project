'''
backbone: resnet50(pretrained: ImageNet)
segmentation_head: DeepLabV3Head
datapath: './gtFine_trainvaltest'(2975 training images, 500 validation images, 1525 test images)
num_epochs: 100
batch_size: 4
'''

import torch
import torchvision
from torchvision import models, datasets, transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time

class CustomDeepLabV3(torch.nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        # spatial dimensions of this map are smaller than the original input image due to the downsampling operations in the backbone.
        # We can upsample the output to the size of the input image using interpolation
        x = torch.nn.functional.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return {'out': x}

backbone = models.resnet50(pretrained=True)
backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))
# backbone.add_module('avgpool', torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))

num_classes = 20
# the segmentation head is responsible for making the final pixel-wise predictions
segmentation_head = DeepLabHead(2048, num_classes)

# Then use your custom model instead of the original one
model = CustomDeepLabV3(backbone, segmentation_head)



# Number of effective classes after mapping (19 classes + 1 background)

# Replace the classifier of the model

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
dataset_path = './gtFine_trainvaltest'
train_dataset = datasets.Cityscapes(root=dataset_path, split='train', mode='fine', target_type='semantic', transform=transform, target_transform=target_transform)
val_dataset = datasets.Cityscapes(root=dataset_path, split='val', mode='fine', target_type='semantic', transform=transform, target_transform=target_transform)
test_dataset = datasets.Cityscapes(root=dataset_path, split='test', mode='fine', target_type='semantic', transform=transform, target_transform=target_transform)

# batch size should be set to 4 or more on GPU for training
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("training on", device)
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# Training loop
num_epochs = 100 # 100-200 epochs typically
loss_values = [] # total loss
best_val_loss = float('inf')
for epoch in range(num_epochs):
    start_time = time.time()  # Start time measurement
    model.train()
    counter = 0
    running_loss = 0.0
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)['out']

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # print loss every 10 batches
        if counter % 10 == 0:
            end_time = time.time()
            duration = end_time - start_time
            print(f"Epoch {epoch + 1}, Batch {counter}, Loss: {loss.item()}, Epoch Time: {duration} s")
        counter += 1



    total_loss = running_loss / len(train_loader)
    loss_values.append(total_loss)

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    end_time = time.time()  # End time measurement
    epoch_duration = end_time - start_time

    print(f"Epoch {epoch + 1}, Training Loss: {total_loss}, Validation Loss: {val_loss}, Epoch Duration: {epoch_duration} s")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_weights.pth')


model.load_state_dict(torch.load('best_model_weights.pth'))
# Testing phase
model.eval()
total = 0
correct = 0
with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)['out']
        _, predicted = torch.max(outputs, 1)
        total += targets.nelement()
        correct += (predicted == targets).sum().item()
print(f"Test Accuracy: {100 * correct / total}%")

# show segmentation example
model.load_state_dict(torch.load('model_weights.pth'))
import matplotlib.pyplot as plt
model.eval()
with torch.no_grad():
    # Get a batch from the validation loader
    images, targets = next(iter(val_loader))
    images = images.to(device)
    targets = targets.to(device)

    # Get the model's prediction
    outputs = model(images)['out']
    _, preds = torch.max(outputs, 1)

    # Move images, targets and preds to cpu for visualization
    images = images.cpu().numpy()
    targets = targets.cpu().numpy()
    preds = preds.cpu().numpy()

    # Plot original image, true mask, and predicted mask
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(np.transpose(images[0], (1, 2, 0)))
    axs[0].set_title('Original Image')
    axs[1].imshow(targets[0])
    axs[1].set_title('True Mask')
    axs[2].imshow(preds[0])
    axs[2].set_title('Predicted Mask')
    plt.savefig('segmentation.png')


# show loss in plt
plt.figure()
plt.plot(loss_values)
plt.title('Training Loss')
plt.xlabel('images')
plt.ylabel('Loss')
plt.savefig('loss.png')

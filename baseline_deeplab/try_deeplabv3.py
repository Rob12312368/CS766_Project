'''
backbone: resnet50(pretrained: ImageNet)
segmentation_head: DeepLabV3Head
batch_size: 4

datapath: './gtFine_trainvaltest'(2975 training images, 500 validation images, 1525 test images)
num_epochs: 100

model structure: Resnet50 + Atrous Spatial Pyramid Pooling(ASPP)

learning rate: (1 - iter/total_iter) ** 0.9, which is rather high

crop size: 769, this helps the ASPP kernel to focus on image instead of padding

batch normalization: use imageNet mean and std instead of cityscapes

upsampling logits: upsampling the output of model instead of downsampling the target.
the downsampling is done through the backbone(size: 100 -> 4 or 769 -> 25)

modify the resnet model to reduce the downsample rate(typical 32 stride, H/32, W/32)
unable to modify the stride to a lower value(32 to 8)

No Data Augmentation
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
import json

with open('config.json') as config_file:
    config = json.load(config_file)

# Use the values from the configuration file
dataset_path = config['datapath']
num_epochs = config['num_epochs']
save_dir = config['save_dir']
continue_training = config['continue_training']

class PolyLearningRateScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iter, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = (1 - self.last_epoch / float(self.max_iter)) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]

class CustomDeepLabV3(torch.nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        # the classifier is responsible for making the final pixel-wise predictions

        self.deconv1 = torch.nn.ConvTranspose2d(
            in_channels=num_classes,
            out_channels=num_classes,
            kernel_size=3,
            stride=4,  # First stride of 4
            padding=1,
            output_padding=0  # Adjust if necessary
        )
        self.deconv2 =  torch.nn.ConvTranspose2d(
            in_channels=num_classes,
            out_channels=num_classes,
            kernel_size=3,
            stride=4,  # Second stride of 4
            padding=1,
            output_padding=0  # Adjust if necessary
        )
        self.deconv3 = torch.nn.ConvTranspose2d(
            in_channels=num_classes,
            out_channels=num_classes,
            kernel_size=3,
            stride=2,  # Third stride of 2
            padding=1,
            output_padding=0  # Adjust if necessary
        )

        self.refine_conv = torch.nn.Conv2d(
            in_channels=num_classes,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.downsample = None
        # modify the kernel size to a larger one to make the output looks smoother?

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        # spatial dimensions of this map are smaller than the original input image due to the downsampling operations in the backbone.
        # We can upsample the output to the size of the input image using interpolation
        # x = torch.nn.functional.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        x = self.deconv1(x)
        x = self.refine_conv(x)  # Refine features
        x = self.deconv2(x)
        x = self.refine_conv(x)  # Refine features
        x = self.deconv3(x)
        x = self.refine_conv(x) # exactly happens to be 769x769

        if self.downsample is None or self.downsample.output_size != input_shape:
            self.downsample = torch.nn.AdaptiveAvgPool2d(input_shape)

        # x = self.downsample(x)
        # why I dont need to downsample the output? isn't is should be 800x800?
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

def transform_target(target):
    target = np.array(target)  # Convert PIL Image to numpy array
    label_mask = np.zeros_like(target)
    for k, v in mapping_20.items():
        label_mask[target == k] = v
    return torch.as_tensor(label_mask, dtype=torch.int64)

def random_crop(img, target, crop_size=(769, 769)):
    i, j, h, w = transforms.RandomCrop.get_params(img, output_size=crop_size)
    img_cropped = transforms.functional.crop(img, i, j, h, w)
    target_cropped = transforms.functional.crop(target, i, j, h, w)
    return img_cropped, target_cropped

def joint_transform(img, target):
    img, target = random_crop(img, target)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    target = transform_target(target)
    return img, target

class CityscapesDataset(datasets.Cityscapes):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return joint_transform(img, target)

# Define the dataset with appropriate transforms for both images and targets
train_dataset = CityscapesDataset(root=dataset_path, split='train', mode='fine', target_type='semantic')
val_dataset = CityscapesDataset(root=dataset_path, split='val', mode='fine', target_type='semantic')


train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
#test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("training on", device)
model.to(device)
criterion = torch.nn.CrossEntropyLoss()

max_iter = num_epochs * len(train_loader)
learning_rate = 0.001  # Initial learning rate
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = PolyLearningRateScheduler(optimizer, max_iter=max_iter)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1) # learning rate decay，reduce the learning rate by a factor of 0.1 every 7 epochs

if continue_training:
    model.load_state_dict(torch.load(save_dir +'/best_model_weights.pth'))
# Training loop
train_loss_list = [] # total loss
val_loss_list = [] # total loss
best_val_loss = float('inf')
for epoch in range(num_epochs):
    start_time = time.time()  # Start time measurement
    model.train()
    counter = 0
    running_loss = 0.0

    print(f"Epoch {epoch + 1}, Training...")
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
            # epoch i / num_epochs, loss should only have 2 decimal places
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {counter}, Train Loss: {loss.item():.2f}, Epoch Time: {duration:.2f} s")
        counter += 1
    scheduler.step()
    train_loss = running_loss / len(train_loader)
    train_loss_list.append(train_loss)
    print(f"Epoch {epoch + 1}, Training Loss: {train_loss},Epoch Duration: {duration} s")

    # Validation phase
    model.eval()
    val_loss = 0.0
    counter = 0
    print(f"Epoch {epoch + 1}, Validating...")
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            if counter % 10 == 0:
                end_time = time.time()
                duration = end_time - start_time
                print(f"Epoch {epoch + 1}/{num_epochs}, Batch {counter}, Validation Loss: {loss.item():.2f}, Epoch Time: {duration:.2f} s")
            counter += 1
    val_loss /= len(val_loader)
    val_loss_list.append(val_loss)
    print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}, Epoch Duration: {duration} s")

    end_time = time.time()  # End time measurement
    epoch_duration = end_time - start_time

    print(f"Epoch {epoch + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss}, Epoch Duration: {epoch_duration} s")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_dir + '/best_model_weights.pth')

# Validation phase
model.load_state_dict(torch.load(save_dir +'/best_model_weights.pth'))
model.eval()
total = 0
correct = 0
counter = 0
total_mIou = []

print("Testing... ") #use the val set
start_time = time.time()

with torch.no_grad():
    for images, targets in val_loader:
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)['out']
        outputs = torch.nn.functional.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)
        #maybe can delete the above line

        _, predicted = torch.max(outputs, 1) # max is used to get the index of the class with the highest probability
        # evaluate the mIOU in the validation set
        total += targets.nelement() #
        correct += (predicted == targets).sum().item()

        iou_per_class = []
        # calculate MIoU here:
        for cls in range(num_classes):
            predicted_cls = predicted == cls
            target_cls = targets == cls
            intersection = (predicted_cls & target_cls).sum().item()
            union = (predicted_cls | target_cls).sum().item()
            if union == 0:
                iou_per_class.append(float('nan'))  # Avoid division by zero
            else:
                iou_per_class.append(intersection / union)

        mean_iou = np.nanmean(iou_per_class)
        total_mIou.append(mean_iou)

        if counter % 10 == 0:
            end_time = time.time()
            duration = end_time - start_time
            print(f"Batch {counter}, Test Accuracy: {100 * correct / total:.2f}%, Mean IoU: {100 * mean_iou:.2f}%, Test time: {duration:.2f} s")
        counter += 1

end_time = time.time()
duration = end_time - start_time
avg_mIou = np.nanmean(total_mIou)
print(f"Test Accuracy: {100 * correct / total:.2f}%, Mean IoU: {100 * avg_mIou:.2f}%, Test Duration: {duration:.2f} s")

# show segmentation example
import matplotlib.pyplot as plt
model.eval()
with torch.no_grad():
    # Get a batch from the val loader
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
    plt.savefig(save_dir + '/segmentation.png')


# show train loss in plt
plt.figure()
plt.plot(train_loss_list)
plt.title('Training Loss')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.savefig(save_dir + '/train_loss.png')

# show val loss in plt
plt.figure()
plt.plot(val_loss_list)
plt.title('Validation Loss')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.savefig(save_dir + '/val_loss.png')

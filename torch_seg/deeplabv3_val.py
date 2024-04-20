'''
backbone: resnet50(pretrained: ImageNet)
segmentation_head: DeepLabV3Head
datapath: './gtFine_trainvaltest'(2975 training images, 500 validation images, 1525 test images)
batch_size: 4
'''

import torch
import torchvision
from torchvision import models, datasets, transforms
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
import json
# import deeplabv3_resnet50
from torchvision.models.segmentation import deeplabv3_resnet50

with open('config.json') as config_file:
    config = json.load(config_file)

# Use the values from the configuration file
dataset_path = config['datapath']
save_dir = config['save_dir']
model_dir = config['save_dir']

num_classes = 20
model = deeplabv3_resnet50(weights=None, num_classes=20, aux_loss=True)
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

target_transform = transforms.Compose([
    transform_target
])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the dataset with appropriate transforms for both images and targets
train_dataset = datasets.Cityscapes(root=dataset_path, split='train', mode='fine', target_type='semantic',
                                    transform=transform, target_transform=target_transform)
val_dataset = datasets.Cityscapes(root=dataset_path, split='val', mode='fine', target_type='semantic',
                                  transform=transform, target_transform=target_transform)
# test_dataset = datasets.Cityscapes(root=dataset_path, split='test', mode='fine', target_type='semantic', transform=transform, target_transform=target_transform)

# batch size should be set to 4 or more on GPU for training
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

model.load_state_dict(torch.load(model_dir + '/best_model_weights.pth'))
model.eval()
total = 0
correct = 0
counter = 0
mIou_list = []
accuracy_list = []

best_accuracy = 0
best_iou = 0
best_images = None
best_targets = None
best_preds = None

print("Validating... ")  # use the val set
start_time = time.time()
with torch.no_grad():
    for images, targets in val_loader:
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)['out']

        _, predicted = torch.max(outputs, 1)  # max is used to get the index of the class with the highest probability
        # evaluate the mIOU in the validation set
        total = targets.nelement()  #
        correct = (predicted == targets).sum().item()
        accuracy = correct / total
        accuracy_list.append(accuracy)

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
        mIou_list.append(mean_iou)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            acu_images = images
            acu_targets = targets.cpu().numpy()
            acu_preds = predicted.cpu().numpy()

        if mean_iou > best_iou:
            best_iou = mean_iou
            mIou_images = images
            mIou_targets = targets.cpu().numpy()
            mIou_preds = predicted.cpu().numpy()

        if counter % 10 == 0:
            end_time = time.time()
            duration = end_time - start_time
            print(
                f"Batch {counter}, Test Accuracy: {100 * accuracy:.2f}%, Mean IoU: {100 * mean_iou:.2f}%, Test time: {duration:.2f} s")
        counter += 1
end_time = time.time()
duration = end_time - start_time
avg_mIou = np.nanmean(mIou_list)
avg_accuracy = np.mean(accuracy_list)

print(f"Test Accuracy: {100 * avg_accuracy:.2f}%, Mean IoU: {100 * avg_mIou:.2f}%, Test Duration: {duration:.2f} s")
print(f"Best Accuracy: {100 * best_accuracy:.2f}%, Best Mean IoU: {100 * best_iou:.2f}%")



normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
def denormalize(tensor):
    tensor = tensor.clone()  # Clone the tensor so as not to make changes to the original
    for t, m, s in zip(tensor, normalization.mean, normalization.std):
        t.mul_(s).add_(m)  # Multiply by std and add mean
    tensor = torch.clamp(tensor, 0, 1)  # Clamp values to the range [0, 1]
    return tensor

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(np.transpose(denormalize(acu_images[0]).cpu().numpy(), (1, 2, 0)))
axs[0].set_title('Original Image')
axs[1].imshow(acu_targets[0])
axs[1].set_title('True Mask')
axs[2].imshow(acu_preds[0])
axs[2].set_title('Predicted Mask')
plt.savefig(save_dir + '/segmentation_0.png')

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(np.transpose(denormalize(mIou_images[0]).cpu().numpy(), (1, 2, 0)))
axs[0].set_title('Original Image')
axs[1].imshow(mIou_targets[0])
axs[1].set_title('True Mask')
axs[2].imshow(mIou_preds[0])
axs[2].set_title('Predicted Mask')
plt.savefig(save_dir + '/segmentation_1.png')

# Plot original image, true mask, and predicted mask for the last batch
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(np.transpose(denormalize(images[0]).cpu().numpy(), (1, 2, 0)))
axs[0].set_title('Original Image')
axs[1].imshow(targets.cpu().numpy()[0])
axs[1].set_title('True Mask')
axs[2].imshow(predicted.cpu().numpy()[0])
axs[2].set_title('Predicted Mask')
plt.savefig(save_dir + '/segmentation_2.png')
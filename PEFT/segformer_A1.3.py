'''
backbone: transformer
Adapter: LoRA(attention,r = 512), linear probe
segmentation_head: segformer
batch_size: 2
more learnable param: 15380 by adding Linear probe, 6852608 by adding LoRA

datapath: './gtFine_trainvaltest'(2975 training images, 500 validation images, 1525 test images)
num_epochs: 5

backbone: Transformer
output size:
remove the last two layers(avgpool, fc):
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
import os
from datetime import datetime
import logging
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
import csv
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model
import csv
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torch.nn.functional as F
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors


model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
config = LoraConfig(
    r=512,
    lora_alpha=16,
    target_modules=["dense"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["fc"],
)
model = get_peft_model(model, config)
model.decode_head.classifier = torch.nn.Conv2d(256, 20, kernel_size=(1, 1), stride=(1, 1))
torch.nn.init.kaiming_normal_(model.decode_head.classifier.weight, mode='fan_out', nonlinearity='relu')
if model.decode_head.classifier.bias is not None:
    torch.nn.init.constant_(model.decode_head.classifier.bias, 0)

with open('config.json') as config_file:
    config = json.load(config_file)

# Use the values from the configuration file
dataset_path = config['data_path']
num_epochs = config['num_epochs']
save_dir = config['save_dir']
batch_size = config['batch_size']

# save log
save_dir = save_dir +'/' + datetime.now().strftime('%Y-%m-%d-%H%M')
os.mkdir(save_dir)
logger = logging.getLogger()
logging.basicConfig(filename=f'{save_dir}/run.log', encoding='utf-8', level=logging.INFO)

num_classes = 20

normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
def denormalize(tensor):
    tensor = tensor.clone()  # Clone the tensor so as not to make changes to the original
    for t, m, s in zip(tensor, normalization.mean, normalization.std):
        t.mul_(s).add_(m)  # Multiply by std and add mean
    tensor = torch.clamp(tensor, 0, 1)  # Clamp values to the range [0, 1]
    return tensor

def plot_segmentation(images, targets, preds, save_dir):
    images = denormalize(images)  # Denormalize the image
    images = images.cpu().numpy()  # Convert to numpy array
    targets = targets.cpu().numpy()  # Convert to numpy array
    preds = preds.cpu().numpy()  # Convert to numpy array
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    axs[0].imshow(np.transpose(images[0], (1, 2, 0)))  # Original image
    axs[0].set_title('Original Image')

    # Class names and their corresponding colors
    class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
                   'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                   'bicycle', 'other']
    colors = ['black', 'gray', 'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'brown', 'pink', 'cyan',
              'magenta', 'olive', 'lime', 'teal', 'coral', 'indigo', 'maroon', 'navy', 'white']

    cmap = mcolors.ListedColormap(colors)
    # Assign colors to each class index based on the specified 'colors'
    legend_handles = [mpatches.Patch(color=colors[i], label=class_names[i]) for i in range(len(colors))]

    # Create patches as legend handles

    #plot each class with a different color, values in preds are from 0 to 18
    # use colors to represent each class

    #0-19
    axs[1].imshow(targets[0], cmap=cmap, vmin=0, vmax=19)
    axs[1].set_title('Ground Truth')
    #0-18
    axs[2].imshow(preds[0], cmap=cmap, vmin=0, vmax=19)
    axs[2].set_title('Predicted Segmentation')
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', title="Classes")
    plt.tight_layout()
    plt.show()


    #add legend to the plot, representing each class with a different color
    class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    colors = ['black', 'gray', 'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'olive', 'lime', 'teal', 'coral', 'indigo', 'maroon', 'navy', 'white']

    plt.legend(handles=[mpatches.Patch(color=colors[i], label=class_names[i]) for i in range(19)], bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.savefig(save_dir + '/segmentation.png')
    # save the preds as image
    plt.imsave(save_dir + '/preds.png', preds[0])
    plt.imsave(save_dir + '/targets.png', targets[0])

mapping_20 = {
    0: 19, 1: 19, 2: 19, 3: 19, 4: 19, 5: 19, 6: 19, 7: 0, 8: 1, 9: 19,
    10: 19, 11: 2, 12: 3, 13: 4, 14: 19, 15: 19, 16: 19, 17: 5, 18: 19,
    19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13,
    27: 14, 28: 15, 29: 19, 30: 19, 31: 16, 32: 17, 33: 18, -1: 19}

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


class DynamicSquareSplitDataset(datasets.Cityscapes):
    def __init__(self, root, split, mode='fine', target_type='semantic', transform=None, target_transform=None, square_size=1024):
        super().__init__(root, split=split, mode=mode, target_type=target_type)
        self.transform = transform
        self.target_transform = target_transform
        self.square_size = square_size

    def __getitem__(self, index):
        image, target = super(DynamicSquareSplitDataset, self).__getitem__(index)

        # Convert PIL images to tensors for advanced tensor operations
        image = transforms.functional.to_tensor(image)
        target = torch.tensor(np.array(target), dtype=torch.int64).unsqueeze(0)  # Add a channel dimension to target

        # Calculate necessary padding for both dimensions
        height_pad = (self.square_size - image.shape[1] % self.square_size) % self.square_size
        width_pad = (self.square_size - image.shape[2] % self.square_size) % self.square_size

        # Pad the image and target tensors if necessary
        image = torch.nn.functional.pad(image, (0, width_pad, 0, height_pad), mode='constant', value=0)
        target = torch.nn.functional.pad(target, (0, width_pad, 0, height_pad), mode='constant', value=0)

        # Unfold the image and target to get squares of the specified size
        image_patches = image.unfold(1, self.square_size, self.square_size).unfold(2, self.square_size, self.square_size)
        target_patches = target.unfold(1, self.square_size, self.square_size).unfold(2, self.square_size, self.square_size)

        # Reshape the patches to flatten the batch of patches into a list of patches
        num_patches = image_patches.size(1) * image_patches.size(2)
        images = image_patches.permute(1, 2, 0, 3, 4).reshape(num_patches, 3, self.square_size, self.square_size)
        targets = target_patches.permute(1, 2, 0, 3, 4).reshape(num_patches, self.square_size, self.square_size)

        # Optionally apply transformations after obtaining the patches
        if self.transform:
            images = [self.transform(transforms.functional.to_pil_image(img)) for img in images]
        if self.target_transform:
            targets = [self.target_transform(torch.tensor(tgt, dtype=torch.int64)) for tgt in targets]

        return list(zip(images, targets))

    def __len__(self):
        return super().__len__() * (1024 // self.square_size) ** 2

# Define the dataset with appropriate transforms for both images and targets
square_size = 1024
train_dataset = DynamicSquareSplitDataset(root=dataset_path, split='train', square_size=square_size,transform=transform, target_transform=target_transform)
val_dataset = DynamicSquareSplitDataset(root=dataset_path, split='val', square_size=square_size, transform=transform, target_transform=target_transform)
#test_dataset = datasets.Cityscapes(root=dataset_path, split='test', mode='fine', target_type='semantic', transform=transform, target_transform=target_transform)

# batch size should be set to 4 or more on GPU for training
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"training on {device}")
logger.info(f"training on {device}")

model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1) # learning rate decay，reduce the learning rate by a factor of 0.1 every 7 epochs

# Training loop
train_loss_list = []
val_loss_list = [] # total loss
mIou_list = []
accuracy_list = []
best_val_loss = float('inf')
for epoch in range(num_epochs):
    start_time = time.time()  # Start time measurement
    model.train()
    counter = 0
    train_loss_total = 0
    val_loss_total = 0

    print(f"Epoch {epoch + 1}, Training...")
    logger.info(f"Epoch {epoch + 1}, Training...")
    for batch in train_loader:
        for images, masks in batch:
            images, masks = images.cuda(), masks.cuda().long()
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)


            logits = outputs.logits
            logits = F.interpolate(logits, size=(1024, 1024), mode='bilinear', align_corners=False)

            _, predicted = torch.max(logits,
                                     1)  # max is used to get the index of the class with the highest probability

            # Calculate loss
            loss = criterion(logits, masks)

            train_loss = criterion(logits, masks)
            train_loss.backward()
            optimizer.step()
            train_loss_total += train_loss.item()
            # print loss every 10 batches
        if counter % 10 == 0:
            end_time = time.time()
            duration = end_time - start_time
            # epoch i / num_epochs, loss should only have 2 decimal places
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {counter}, Train Loss: {train_loss.item():.2f}, Epoch Time: {duration:.2f} s")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Batch {counter}, Train Loss: {train_loss.item():.2f}, Epoch Time: {duration:.2f} s")
        counter += 1
    train_loss = train_loss_total / len(train_loader)
    train_loss_list.append(train_loss)

    # Validation phase
    model.eval()
    total = 0
    correct = 0
    counter = 0
    total_mIou = []
    print(f"Epoch {epoch + 1}, Validating...")
    logger.info(f"Epoch {epoch + 1}, Validating...")
    with torch.no_grad():
        for batch in val_loader:
            for images, masks in batch:
                images, masks = images.cuda(), masks.cuda().long()

                # Forward pass
                outputs = model(images)
                logits = outputs.logits
                logits = F.interpolate(logits, size=(1024, 1024), mode='bilinear', align_corners=False)
                targets = masks

                val_loss = criterion(logits, targets)
                val_loss_total += val_loss.item()

                _, predicted = torch.max(logits,
                                         1)  # max is used to get the index of the class with the highest probability
                total += targets.nelement()  #
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
                print(f"Batch {counter}, validation loss: {val_loss.item():.2f}, Mean IoU: {100 * mean_iou:.2f}%, Test time: {duration:.2f} s")
                logger.info( f"Batch {counter}, validation loss: {val_loss.item():.2f}, Mean IoU: {100 * mean_iou:.2f}%, Test time: {duration:.2f} s")
        counter += 1

    accuracy = correct / total
    accuracy_list.append(accuracy)

    avg_mIou = np.mean(total_mIou)
    mIou_list.append(avg_mIou)

    val_loss = val_loss_total / len(val_loader)
    val_loss_list.append(val_loss)

    end_time = time.time()  # End time measurement
    epoch_duration = end_time - start_time

    print(f"Epoch {epoch + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss}, accuracy: {accuracy*100:.2f}%, mIOU: {avg_mIou*100:.2f}%, Epoch Duration: {epoch_duration} s")
    logger.info(f"Epoch {epoch + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss}, accuracy: {accuracy*100:.2f}%, mIOU: {avg_mIou*100:.2f}%, Epoch Duration: {epoch_duration} s")
    if val_loss < best_val_loss:
        logger.info(f"Current best val loss: Epoch {epoch + 1}")
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_dir + '/best_model_weights.pth')

# Define path for the CSV file
csv_path = os.path.join(save_dir, 'metrics.csv')

# Write metrics to a CSV file
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the headers
    writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'mIOU'])

    # Write the data
    for epoch in range(num_epochs):
        writer.writerow([
            epoch + 1,
            train_loss_list[epoch],
            val_loss_list[epoch],
            mIou_list[epoch]
        ])

# show segmentation example

with torch.no_grad():
    # Get a batch from the val loader
    outputs = model(images)
    logits = outputs.logits
    logits = F.interpolate(logits, size=(1024, 1024), mode='bilinear', align_corners=False)
    _, preds = torch.max(logits, 1)
    plot_segmentation(images, masks, preds, save_dir)

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

plt.figure()
plt.plot(mIou_list)
plt.title('mIOU')
plt.xlabel('epochs')
plt.ylabel('mIOU')
plt.savefig(save_dir + '/mIOU.png')

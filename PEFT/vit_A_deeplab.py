'''
backbone: resnet50(pretrained: ImageNet)
segmentation_head: DeepLabV3Head
batch_size: 4

datapath: './gtFine_trainvaltest'(2975 training images, 500 validation images, 1525 test images)
num_epochs: 100

deeplabv3_2.2: No crop, use FCN + bilinear upsampling
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
from transformers import ViTModel, ViTConfig
from peft import LoraConfig, get_peft_model

# Load the pretrained Vision Transformer
model_checkpoint = "google/vit-base-patch16-224-in21k"
config = ViTConfig.from_pretrained(model_checkpoint)
vit_model = ViTModel.from_pretrained(model_checkpoint, add_pooling_layer=False)  # No pooling to maintain spatial dimensions

with open('config.json') as config_file:
    config = json.load(config_file)

# Use the values from the configuration file
dataset_path = config['data_path']
num_epochs = config['num_epochs']
save_dir = config['save_dir']
continue_training = config['continue_training']

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
lora_model = get_peft_model(vit_model, lora_config)

class CustomDeepLabV3(torch.nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        print("size after downsampling", features.shape)
        x = self.classifier(features)
        print("size after segmentation head", x.shape)
        # spatial dimensions of this map are smaller than the original input image due to the downsampling operations in the backbone.
        # We can upsample the output to the size of the input image using interpolation
        x = torch.nn.functional.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        print("size after upsampling", x.shape)
        return {'out': x}

class ViTBackbone(torch.nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model

    def forward(self, x):
        outputs = self.vit(x)
        # Extract the feature map from the tuple output
        feature_map = outputs.last_hidden_state  # Assuming the feature map is the last element of the tuple
        # Return the feature map as a dictionary with the key "out"
        return feature_map

# Update your model instantiation


# Replace the backbone in DeepLabV3

vit_backbone = ViTBackbone(lora_model)  # lora_model is your LoRA-enhanced ViT model
deeplab_head = DeepLabHead(768, 20)  # Number of classes is 20
model = CustomDeepLabV3(backbone=vit_backbone, classifier=deeplab_head)

# Number of effective classes after mapping (19 classes + 1 background)

# Replace the classifier of the model

# Mapping for reducing classes to 20 including background
mapping_20 = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 2, 9: 0,
    10: 0, 11: 3, 12: 4, 13: 5, 14: 0, 15: 0, 16: 0, 17: 6, 18: 0,
    19: 7, 20: 8, 21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14,
    27: 15, 28: 16, 29: 0, 30: 0, 31: 17, 32: 18, 33: 19, -1: 0
}
num_classes = 20
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



from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class CityscapesPatched(Dataset):
    def __init__(self, root, split, mode='fine', target_type='semantic', transform=None, target_transform=None):
        super().__init__()
        self.dataset = datasets.Cityscapes(root=root, split=split, mode=mode, target_type=target_type)
        self.transform = transform
        self.target_transform = target_transform
        self.patch_size = 224

    def __len__(self):
        num_images = len(self.dataset)
        return num_images * ((2048 // self.patch_size) * (1024 // self.patch_size))

    def __getitem__(self, idx):
        # Calculate which image and which patch this index corresponds to
        num_patches_per_image = (2048 // self.patch_size) * (1024 // self.patch_size)
        image_index = idx // num_patches_per_image
        patch_index = idx % num_patches_per_image

        image, target = self.dataset[image_index]
        row = (patch_index // (2048 // self.patch_size)) * self.patch_size
        col = (patch_index % (2048 // self.patch_size)) * self.patch_size

        image_patch = TF.crop(image, row, col, self.patch_size, self.patch_size)
        target_patch = TF.crop(target, row, col, self.patch_size, self.patch_size)

        if self.transform:
            image_patch = self.transform(image_patch)
        if self.target_transform:
            target_patch = self.target_transform(target_patch)

        return image_patch, target_patch


# Instantiate the training and validation datasets
train_dataset = CityscapesPatched(root=dataset_path, split='train', mode='fine', target_type='semantic',
                                  transform=transform, target_transform=target_transform)
val_dataset = CityscapesPatched(root=dataset_path, split='val', mode='fine', target_type='semantic',
                                transform=transform, target_transform=target_transform)

# Use DataLoader to load batches of data
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, drop_last=True)


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("training on", device)
model.to(device)
criterion = torch.nn.CrossEntropyLoss()

max_iter = num_epochs * len(train_loader)
learning_rate = 0.001  # Initial learning rate
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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

        outputs = model(images)

        main_output = outputs['out']

        loss = criterion(main_output, targets)

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()  # Add this line
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

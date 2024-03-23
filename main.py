from mobilenetv3 import mobilenetv3_large, mobilenetv3_small
import torch
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import urllib.request
class Mbnet():
    def __init__(self,choice,pretrained) -> None:
        if choice == 'large':
            self.model = mobilenetv3_large()
        elif choice == 'small':
            self.model = mobilenetv3_small()
        self.model.load_state_dict(torch.load(pretrained))
        self.preprocessor = self.build_preprocessor()
        self.get_img_num_to_label()
    def get_img_num_to_label(self):
        # URL to the ImageNet labels file
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        response = urllib.request.urlopen(url)
        labels_bytes = response.read()
        labels_text = labels_bytes.decode('utf-8')
        self.labels_list = labels_text.split('\n')
    def build_preprocessor(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to 224x224 (input size for MobileNetV3)
            transforms.ToTensor(),           # Convert image to tensor
            transforms.Normalize(            # Normalize pixel values
                mean=[0.485, 0.456, 0.406],  # Mean of ImageNet dataset
                std=[0.229, 0.224, 0.225]    # Standard deviation of ImageNet dataset
            ),
        ])
    def predict(self,dir_path):
        images = {}
        predicted_classes = {}
        self.model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image_paths = os.listdir(dir_path)
        for image_path in image_paths:
            image = Image.open(os.path.join(dir_path, image_path))
            input_tensor = self.preprocessor(image)
            input_batch = input_tensor.unsqueeze(0)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_batch = input_batch.to(device)
            with torch.no_grad():
                output = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_classes[image_path] = self.labels_list[torch.argmax(probabilities).item()]
            images[image_path] = image
        self.draw_graph(predicted_classes, images)
        return predicted_classes

    def draw_graph(self, predicted_classes, images):
        if len(predicted_classes) % 3:
            height =len(predicted_classes) // 3 + 1
        else:
            height =len(predicted_classes) // 3 
        fig, axs = plt.subplots(height, 3, figsize=(12, 9))
        row, col = 0, 0
        for key, value in images.items():
            axs[row, col].imshow(value)
            axs[row, col].set_title(predicted_classes[key])
            col = (col+1) % 3
            if col % 3 == 0:
                row += 1
        plt.show()

if __name__ == "__main__":
    model = Mbnet('small', 'pretrained/mobilenetv3-small-55df8e1f.pth')
    model.build_preprocessor()
    print(model.predict('./sample_images'))


'''
net_large = mobilenetv3_large()
net_small = mobilenetv3_small()

net_large.load_state_dict(torch.load('pretrained/mobilenetv3-large-1cd25616.pth'))
net_small.load_state_dict(torch.load('pretrained/mobilenetv3-small-55df8e1f.pth'))

# need to transform the input image and feed into the model using forward

# Define transformations to preprocess the image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224 (input size for MobileNetV3)
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize(            # Normalize pixel values
        mean=[0.485, 0.456, 0.406],  # Mean of ImageNet dataset
        std=[0.229, 0.224, 0.225]    # Standard deviation of ImageNet dataset
    ),
])

# Load the image
image_path = './flower.JPEG'
image = Image.open(image_path)

# Preprocess the image
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

# Move input tensor to the appropriate device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_batch = input_batch.to(device)

# Set the model to evaluation mode
net_small.eval()

# Forward pass
with torch.no_grad():
    output = net_small(input_batch)

# Get predicted class probabilities (assuming model was trained with softmax)
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Get the predicted class index
predicted_class = torch.argmax(probabilities).item()

print(predicted_class, probabilities)'''

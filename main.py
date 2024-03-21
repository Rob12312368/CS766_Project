from mobilenetv3 import mobilenetv3_large, mobilenetv3_small
import torch
from torchvision import transforms
from PIL import Image


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

print(predicted_class, probabilities)

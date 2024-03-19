from mobilenetv3 import mobilenetv3_large, mobilenetv3_small
import torch
net_large = mobilenetv3_large()
net_small = mobilenetv3_small()

net_large.load_state_dict(torch.load('pretrained/mobilenetv3-large-1cd25616.pth'))
net_small.load_state_dict(torch.load('pretrained/mobilenetv3-small-55df8e1f.pth'))

# need to transform the input image and feed into the model using forward
net_large.forward()

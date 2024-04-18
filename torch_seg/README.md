1. clone the repository to your local machine
command: !git clone https://github.com/Rob12312368/CS766_Project

2. download the cityscapes dataset
You need to download both the leftImg8bit and gtFine parts of the dataset.
link: https://www.cityscapes-dataset.com/downloads/

3. run a simple testing code without using the whole dataset to test the environment
It should run using the sample dataset that included in the repository.
The first run will download the pretrained model from the internet. The second run expects no error.
command: !python3 ~/CS766_Project/torch_seg/try_resnet50_imagenet_deeplabv3.py

4. run the training code
make sure the dataset is downloaded and the path is correct in the code. (Or you can change the dataset path in the code)
The file structure should looks like the follwing:
CS766_Project/torch_seg/gtFine_trainvaltest/gtFine
CS766_Project/torch_seg/gtFine_trainvaltest/leftImg8bit
both gtFine and leftImg8bit should have the same subfolders including train, val, and test.
command: !python3 ~/CS766_Project/torch_seg/main.py

dependencies:
!pip install torch torchvision numpy

resources:
use the following link to check the label of cityscapes dataset, the total number of labels is 34, we need to map the labels to 19 classes.
1.label of cityscapes: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
2.what is mIOU: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
3.state of the art models: https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes


# CS766_Project 
# dependencies:
jupyter notebook, torch, torchvision, numpy, matplotlib, peft
```bash
pip install jupyter torch torchvision numpy matplotlib git+https://github.com/huggingface/peft
```

# run code:
1. run demo at:
/CS766_Project/PEFT/demo.ipynb

2. clone the repository to your local machine
```bash
git clone https://github.com/Rob12312368/CS766_Project
```
3. download the cityscapes dataset
You need to download both the leftImg8bit and gtFine parts of the dataset.
link: https://www.cityscapes-dataset.com/downloads/

4. run the training code v1 main.py
make sure the dataset is downloaded and the path is correct in the code. (Or you can change the dataset path in the code)
The first run will download the pretrained model from the internet. The second run expects no error.
The file structure should looks like the following:
CS766_Project/gtFine_trainvaltest/gtFine
CS766_Project/gtFine_trainvaltest/leftImg8bit
both gtFine and leftImg8bit should have the same subfolders including train, val, and test.
```bash
python3 ~/CS766_Project/PEFT/segformer_A2.1.py
```
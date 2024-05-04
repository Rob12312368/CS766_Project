# CS766_Project 
# dependencies:
jupyter notebook, torch, torchvision, numpy, matplotlib, peft
```bash
pip install jupyter torch torchvision numpy matplotlib git+https://github.com/huggingface/peft
```

# run code:
1. **run demo at:**
/CS766_Project/PEFT/demo.ipynb

2. **clone the repository to your local machine**
```bash
git clone https://github.com/Rob12312368/CS766_Project
```
3. **download the cityscapes dataset:**
You need to download both the leftImg8bit and gtFine parts of the dataset.
link: https://www.cityscapes-dataset.com/downloads/

4. **download the pre-trained model:** 
The first run will download the pretrained model from the internet.
```bash
python3 ~/CS766_Project/PEFT/segformer_A2.1.py
```

5. **modify the configuration file:**
set parameters including data_path, save_dir, num_epochs, batch_size, num_workers, etc.

6. **execute the command in step 4 again:**
It will run the py file **segformer_A2.1.py** this time.


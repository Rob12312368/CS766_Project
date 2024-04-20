1. learning rate: 0.001 step decay (7epoch reduce 0.1 factor)
2. random crop iamge for training: 769 * 769
3. upsampling: deconvolution, 32 stride deconv(3layer,4*4*2), Not necessary for AdaptiveAvgPool2d
4. num_epches: 1
Test Accuracy: 57.90%, Mean IoU: 12.34%, Test Duration: 124.53 s

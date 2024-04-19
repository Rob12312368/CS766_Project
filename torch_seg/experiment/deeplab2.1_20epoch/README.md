1.learning rate: 0.001 step decay (7epoch reduce 0.1 factor)
2.random crop iamge for training: 769 * 769
3.upsampling: deconvolution(32 stride deconv + AdaptiveAvgPool2d)
4.num_epches: 20
Epoch 1, Training Loss: 1.0158066916209396, Validation Loss: 0.8475453269481659, Epoch Duration: 581.6772623062134 s
Epoch 2, Training Loss: 0.7478499949298879, Validation Loss: 0.8701184303760529, Epoch Duration: 579.0196607112885 s
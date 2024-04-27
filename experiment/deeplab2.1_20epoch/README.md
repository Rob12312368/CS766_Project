1.learning rate: 0.001 step decay (7epoch reduce 0.1 factor)
2.random crop iamge for training: 769 * 769
3.upsampling: deconvolution(32 stride deconv + AdaptiveAvgPool2d)
4.num_epches: 20
Epoch 1, Training Loss: 1.0158066916209396, Validation Loss: 0.8475453269481659, Epoch Duration: 581.6772623062134 s
Epoch 2, Training Loss: 0.7478499949298879, Validation Loss: 0.8701184303760529, Epoch Duration: 579.0196607112885 s
Epoch 3, Training Loss: 0.6951026229127761, Validation Loss: 0.6639328153133393, Epoch Duration: 579.8042645454407 s
Epoch 5, Training Loss: 0.5939242683591381, Validation Loss: 0.6709868257045746, Epoch Duration: 580.033132314682 s
Epoch 6, Training Loss: 0.5894430338214802, Validation Loss: 0.8620801148414612, Epoch Duration: 580.298641204834 s
Epoch 14, Training Loss: 0.43706872238106625, Validation Loss: 0.45013926243782043, Epoch Duration: 580.3874845504761 s
Epoch 20, Training Loss: 0.42830875712216543, Validation Loss: 0.46434865057468416, Epoch Duration: 579.9382929801941 s
Test Accuracy: 85.52%, Mean IoU: 35.09%, Test Duration: 62.73 s (cropped image)
Test Accuracy: 77.75%, Mean IoU: 29.87%, Test Duration: 124.66 s (original image)



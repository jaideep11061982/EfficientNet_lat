# EfficientNet_lat
This file extracts the lateral features of any efficient net model . THere are basically seven layers of features  of various sizes and depth depending upon configurations of Effnet model as per its block_Args .Please check out for config of effnet models here
https://github.com/lukemelas/EfficientNet-PyTorch
The way it works is Every block of Effnet starts from some depth say 16 after series of convolution it would end with same depth  depending upon number of repeats before new layer size would be starting (indicated by depth of last layers in MBblock) given by  depth coefficient * num repeat -  1.2*3=3.6 take the largest integer=4 afte decimal ,3.2 is also 4. 

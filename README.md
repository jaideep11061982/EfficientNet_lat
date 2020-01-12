# EfficientNet Lateral Feature Extraction
This file extracts the lateral features of any efficient net model . THere are basically seven layers of features  of various sizes and depth depending upon configurations of Effnet model as per its block_Args .Please check out for config of effnet models here
https://github.com/lukemelas/EfficientNet-PyTorch
The way it works is Every block of Effnet starts from some depth say 16 after series of convolution it would end with same depth  depending upon number of repeats before new layer size would be starting (indicated by depth of last layers in MBblock) given by  depth coefficient * num repeat -  1.2*3=3.6 take the largest integer=4 afte decimal ,3.2 is also 4. 


One Limitation this has got is  .This can work with image size multiple that is multiple of 64 or 128  any one of them.Use below script to check for the sizes . Upsample of uplayers should match with max pool of layers down
#Script
To validate he sizes



base_model = myEfficientNet.from_pretrained('efficientnet-b2') 
x=torch.randn(1,3,320,1280)
#x_center = x[:, :, :, 1536 // 12: -1536 // 12]
print(x_center.size())
x = base_model._swish(base_model._bn0(base_model._conv_stem(x_center)))
size1=0
feature=[]
for b in base_model._blocks:
    
    tmp=b(x)
    x=tmp
    #print(tmp.size())
    if size1!=tmp.size(1):
        feature.append(tmp.size(1))
        size1=tmp.size(1)
        print(tmp.size(),nn.MaxPool2d(2)(x).size())
        

feats= base_model._swish(base_model._bn1(base_model._conv_head(x)))  
device='cpu'
print(feats.size())
bg = torch.zeros([feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] // 8]).to(device)
feats = torch.cat([bg, feats, bg], 3)
feats.size()


Loaded pretrained weights for efficientnet-b2
torch.Size([1, 3, 320, 896])
torch.Size([1, 16, 160, 448]) torch.Size([1, 16, 80, 224])
torch.Size([1, 24, 80, 224]) torch.Size([1, 24, 40, 112])
torch.Size([1, 48, 40, 112]) torch.Size([1, 48, 20, 56])
torch.Size([1, 88, 20, 56]) torch.Size([1, 88, 10, 28])
torch.Size([1, 120, 20, 56]) torch.Size([1, 120, 10, 28])
torch.Size([1, 208, 10, 28]) torch.Size([1, 208, 5, 14])
torch.Size([1, 352, 10, 28]) torch.Size([1, 352, 5, 14])
torch.Size([1, 1408, 10, 28])

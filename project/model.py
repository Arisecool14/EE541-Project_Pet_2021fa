from torch.nn.modules.conv import Conv2d
from torchvision.models import resnet50
import torch.nn as nn
model1 = resnet50(pretrained=True)

for param in model1.parameters():
    param.requires_grad = False

model = model1




mymodel = nn.Sequential(
    nn.Conv2d(3,16,kernel_size=3),
    nn.ReLU(),
    nn.Conv2d(16,32,kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Conv2d(32,64,kernel_size=3),
    nn.ReLU(),
    nn.Conv2d(64,128,kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Conv2d(128,64,kernel_size=3),
    nn.ReLU(),
    nn.Conv2d(64,32,kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Flatten(),
    nn.Linear(32*24*24,2),
    nn.Linear(2,1),
    nn.Sigmoid()
)

# for param in model1.parameters():
#     param.requires_grad = False

# model1.fc = nn.Sequential(
#     nn.Linear(2048,2),
#     nn.ReLU(inplace=False),
#     nn.Linear(2,1),
#     nn.LogSoftmax(dim = 1)
# )


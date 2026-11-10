import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

# 先用 None 练习结构，不需要下载。
model = resnet18(weights=None)

# 要真正使用预训练权重，把上一行换成：
# model = resnet18(weights=ResNet18_Weights.DEFAULT)

# 1. 冻结当前已有的参数
for param in model.parameters():
    param.requires_grad = False

# 2. 换掉最后一层
print("旧的最后一层：", model.fc)
model.fc = nn.Linear(model.fc.in_features, 10)
print("新的最后一层：", model.fc)

# 3. 用两张假图片检查输出形状
model.eval()
with torch.no_grad():
    images = torch.randn(2, 3, 224, 224)
    scores = model(images)
    print(scores.shape)  # torch.Size([2, 10])
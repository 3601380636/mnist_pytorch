import torch
import numpy as np

x=np.array([[1,2],
            [3,4]])
y=torch.from_numpy(x)#numpy-->tensor
z=y.numpy()#tensor-->numpy
print(type(x),type(y),type(z))
print(y.shape)
#1.dataset（从MNIST数据集中取一张图）
from torchvision import datasets,transforms#导入数据集工具

train_dataset=datasets.MNIST(
    root="./data",
    train=True,#选训练集（False则选测试集）
    download=True,
    transform=transforms.ToTensor(),#照片转成Tensor
)
image,label=train_dataset[0]#取第一张图片

print("训练样本数：",len(train_dataset))
print("图片形状：",image.shape)
print("图片类型：",image.dtype)
print("标签：",label)
#2.dataloader:把多张图转成一批
from torch.utils.data import DataLoader#导入数据收集工具

train_loader = DataLoader(
    train_dataset,
    batch_size=32,#每批最多取32个样本
    shuffle=True,#表示每次重新遍历训练集时打乱样本的取出顺序，每张图仍然跟着自己的标签
)

data_iter = iter(train_loader)
images, labels = next(data_iter)

print("一批图片：", images.shape)   # torch.Size([32, 1, 28, 28])
print("一批标签：", labels.shape)   # torch.Size([32])
print("这批的第 1 张图：", images[0].shape)
print("这批的第 1 个标签：", labels[0].item())
#3.自己写一个dataset
import torch
from torch.utils.data import Dataset, DataLoader


class TinyDataset(Dataset):
    def __init__(self):
        self.features = torch.tensor([
            [1.0, 2.0],  # 第 1 条样本
            [3.0, 4.0],  # 第 2 条样本
            [5.0, 6.0],  # 第 3 条样本
        ])
        self.labels = torch.tensor([0, 1, 0])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        features = self.features[index]
        label = self.labels[index]
        return features, label


dataset = TinyDataset()
print("样本总数：", len(dataset))
print("第 2 条样本：", dataset[1])

loader = DataLoader(dataset, batch_size=2, shuffle=False)
for batch_no, (features, labels) in enumerate(loader, start=1):
    print(f"第 {batch_no} 批：", features.tolist(), labels.tolist())#tolist:将tensor转换成列表
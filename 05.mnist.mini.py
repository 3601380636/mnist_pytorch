import torch
import torch.nn as nn

from torchvision import datasets,transforms#导入数据集工具
from torch.utils.data import DataLoader
train_dataset=datasets.MNIST(
    root="./data",
    train=True,#选训练集（False则选测试集）
    download=True,
    transform=transforms.ToTensor(),#照片转成Tensor
)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,#每批最多取32个样本
    shuffle=True,#表示每次重新遍历训练集时打乱样本的取出顺序，每张图仍然跟着自己的标签
)

data_iter = iter(train_loader)
images, labels = next(data_iter)

print('一批图片：',images.shape)
print('一批标签：',labels.shape)
x=images.flatten(start_dim=1)#将images每一个都展开成一维，28*28=784
print('模型输入：',x.shape)#(32,784)
model=nn.Linear(784,10)
pred=model(x)
print('模型输出：',pred.shape)#(32,10)
criterion=nn.CrossEntropyLoss()
loss=criterion(pred,labels)
print('损失为：',loss.item())
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print('更新后的权重',model.weight)

with torch.no_grad():
    new_pred=model(x)
    new_loss=criterion(new_pred,labels)

print('更新后的损失：',new_loss.item())

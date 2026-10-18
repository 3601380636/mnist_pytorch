import torch
import torch.nn as nn

torch.manual_seed(0)

x=torch.randn(10,3)
y=torch.randn(10,2)
print(x)
print(y)

model=nn.Linear(3,2)

criterion=nn.MSELoss()#均方误差
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
print(criterion)
print(optimizer)

pred=model(x)#x通过模型得到的结果10*2
loss=criterion(pred,y)#定义损失函数:结果与实际作差，平方，取平均数

print('预测结果的形状：',pred.shape)
print('更新前的损失:',loss.item())

optimizer.zero_grad()#清空之前累积的梯度
loss.backward()#计算损失对各参数的梯度

print('权重的梯度:',model.weight.grad)

#optimizer.step()#根据梯度更新参数：model里的权重会变,x,y不变

with torch.no_grad():
    new_pred=model(x)
    new_loss=criterion(new_pred,y)

    print('更新后的损失:',new_loss.item())
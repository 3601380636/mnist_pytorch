import torch
import torch.nn as nn

torch.manual_seed(0)

x=torch.randn(10,3)
y=torch.randn(10,2)
print(x)
print(y)

model=nn.Linear(3,2)#三输入二输出，包括weight和bias
print(model.weight.shape)
print(model.bias.shape)

criterion=nn.MSELoss()#均方误差
optimizer=torch.optim.SGD(model.parameters(),lr=0.01) 
print(criterion)
print(optimizer)

pred=model(x)#x通过模型得到的预测值,形状为10*2 pred=X*W转置+bias
loss=criterion(pred,y)#定义损失函数:结果与实际作差，平方，取平均数

print('预测结果的形状：',pred.shape)
print('更新前的损失:',loss.item())
print('求梯度前的权重:',model.weight)

optimizer.zero_grad()#清空之前累积的梯度
loss.backward()#计算损失对各参数的梯度
print('求梯度后的权重:',model.weight)

print('权重的梯度:',model.weight.grad)

optimizer.step()#根据梯度更新参数：model里的weight=原来的-lr*model.weight.grad
print('更新后的权重:',model.weight)
with torch.no_grad():
    new_pred=model(x)
    new_loss=criterion(new_pred,y)

    print('更新后的损失:',new_loss.item())

# 1. 保存训练后的权重和偏置
torch.save(model.state_dict(), "linear_weights.pth")

# 2. 新建一个结构相同的线性层
restored = nn.Linear(3, 2)

# 3. 从文件读出参数，装进新模型
state = torch.load("linear_weights.pth", map_location="cpu", weights_only=True)
restored.load_state_dict(state)

# 4. 用同一个输入 x，比较两个模型的输出
model.eval()
restored.eval()

with torch.no_grad():
    before = model(x)#原模型
    after = restored(x)#新建的模型
    print("预测结果是否一致：", torch.allclose(before, after))  # True    
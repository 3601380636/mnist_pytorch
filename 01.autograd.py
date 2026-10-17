import torch

x=torch.tensor(4.,requires_grad=True)
w=torch.tensor(3.,requires_grad=True)
b=torch.tensor(2.,requires_grad=True)

y=w*x+b

print("y=",y.item())  #y最终转变为一个值

y.backward()#进行反向求导

print("x 的梯度 =", x.grad)
print("w 的梯度 =", w.grad)
print("b 的梯度 =", b.grad)
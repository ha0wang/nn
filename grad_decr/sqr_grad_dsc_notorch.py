import torch


# define input x
x = torch.rand([100,1])
lr = 0.015

# linear y = 3*x + 2
y_true = x**2

#print(y_true)

# define weight
w = torch.rand([1,10], requires_grad=True)
b = torch.rand([10], requires_grad=True)
#print(x*w + b)

for i in range(2000):
    y_relu = torch.nn.functional.relu(x*w + b)
    y_pred = y_relu.sum(dim=1)
    # print(f"y size is: {y_pred.size()}")
    # define loss
    loss = torch.mean((y_true - y_pred)**2)

    # backward 
    loss.backward()
    w.data = w.data - lr*w.grad
    b.data = b.data - lr*b.grad

    # clear grad
    w.grad = None
    b.grad = None

    if i % 50 == 0:
        print(f"w: {w}, b: {b}, loss: {loss}")

# draw the plot of y_pred and y_true
import matplotlib.pyplot as plt
plt.plot(x, y_true, label='y_true')

plt.scatter(x.numpy().reshape(-1),y_true.numpy().reshape(-1))
plt.scatter(x.numpy().reshape(-1),y_pred.detach().numpy().reshape(-1))
plt.show()
    

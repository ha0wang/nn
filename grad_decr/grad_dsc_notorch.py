import torch


# define input x
x = torch.rand([100,1])
lr = 0.01

# linear y = 3*x + 2
y_true = 3*x + 2

#print(y_true)

# define weight
w = torch.rand([1,1], requires_grad=True)
b = torch.rand([1], requires_grad=True)
#print(x*w + b)

for i in range(2000):
    y_pred = x*w + b
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
        print(f"w: {w.item()}, b: {b.item()}, loss: {loss}")
    

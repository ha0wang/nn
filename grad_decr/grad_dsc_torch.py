from torch import nn
import torch

device = torch.device("cuda")

class Lr(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

x = torch.rand([500,1]).to(device) 
model = Lr().to(device)
y_true = 5*x + 4
#y_true = y_true.to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.015)
for i in range(2000):
    y_pred = model(x)
    # define loss
    loss = loss_fn(y_true, y_pred)
    # backward
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (i+1) % 50 == 0:
        print(f"weight is: {model.linear.weight.item()}, bias is: {model.linear.bias.item()}, loss is: {loss.item()}")



import torch
# generate a 2 x 6 tensor with values at range 1 - 10
"""
print(torch.arange(1, 11))
a = torch.rand( 2, 6)
print(a)
b = a.view(4, 3)
print(b)
print(torch.empty(2, 3))
print("torch.randint(1, 10, (2, 6))")
a = torch.randint(1, 10, (2, 6))
print(a)
print(a.shape)
print(a.num)

print("a.view(2, 2, 3)")
print(a.view(2, 2, 3))
"""
x = torch.randint(1, 10, (10, 1))
print(f"x is: \n {x}")
w = torch.ones([1,1], requires_grad=True)
print(f"w is: \n {w}")
y = x*w
print(f"x*w is: \n {y}")
b = torch.ones([1], requires_grad=True)
print(f"b is: \n {b}")
print(f"x*b is: \n {x*b}")
import torch

x = torch.empty(3, 4)    # empty matrix
print(type(x))
print(x)

ones = torch.ones(2, 3)    # all-1 matrix
print(ones)

torch.manual_seed(1729)
random = torch.rand(2, 3)
print(random)

torch.manual_seed(1729)    # same seed, same output
random_1 = torch.rand(2, 3)
print(random_1)

random_1 = random_1.view(3, 2)    # change size
print(random_1)

random_1 = random_1.view(-1, 6)    # auto-detect size
print(random_1)

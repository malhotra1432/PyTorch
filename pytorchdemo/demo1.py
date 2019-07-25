import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)


input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)


output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)










# from __future__ import print_function
# import torch
#
#
# x = torch.ones(2, 2, requires_grad=True)
# print(x.grad)
#
#
# y = x + 2
# print(y)
#
# print(y.grad_fn)
# print("==-=-=-=-=-=-")
#
# z = y * y * 3
# out = z.mean()
#
# print(z, out)
#
# print("==-=-=-=-=-=-")
#
#
# # a = torch.randn(2, 2)
# # a = ((a * 3) / (a - 1))
# # print(a.requires_grad)
# # a.requires_grad_(True)
# # print(a.requires_grad)
# # b = (a * a).sum()
# # print(b.grad_fn)
#
#
# print(x.grad)
#
#
# out.backward()
#
# #
# # x = torch.empty(5, 3)
# # print(x)
# #
# # x = torch.zeros(5, 3, dtype=torch.long)
# # print(x)
# #
# # x = torch.tensor([5.5, 3])
# # print(x)
# #
# # x = x.new_ones(5, 3, dtype=torch.double)
# # print(x)
# #
# # x = torch.randn_like(x, dtype=torch.float)
# # print(x)
# # print(x.size())
# #
# # # let us run this cell only if CUDA is available
# # # We will use ``torch.device`` objects to move tensors in and out of GPU
# # if torch.cuda.is_available():
# #     device = torch.device("cuda")  # a CUDA device object
# #     y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
# #     x = x.to(device)  # or just use strings ``.to("cuda")``
# #     z = x + y
# #     print(z)
# #     print(z.to("cpu", torch.double))  # ``.to`` can also change dtype together!

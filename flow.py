from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

MNIST_ROOT = "/home/william/Datasets"

# torch.autograd.set_detect_anomaly(True)

## FLOW
#
# Given an input image, run a recurrent network to try to transform the image to the image
# class's exemplar. This is an interesting opportunity to test recurrent spatial networks.
#
# This process can be described with the following update rule.
#   X_0 = Input image
#   X_i = (1 - a) * Conv(Conv(X_0)) + a * X_0
#
# State = W
# (Exemplars) size = 10
# (Classifications) size = 10
# Layer input:  state, classifications, exemplars
# Layer output: state, classifications
#
# The task can be to either
#    1. Warp the input to the correct exemplar      (Builds something akin to a learned flow model)
#       - May be more generally useful
#       - May be harder to train
#       - Has a "canonical exemplar" problem
#         - Punt
# or 2. Write the correct pixels to classification layer      (Can be seen as a causal model)
#       - May be less generally useful
#       - May be easier to train.
#
# And the task can be done with or without the exemplars being passed in.
#
# This leaves us with hyperparameters
# Optimizer, Total Steps, Loss Function
#
# Optimizer can be SGD
#   LR = .001?
# For total steps we'll try 5, 10, 20, 50, 100.
#   This will be limited by update rule stability and memory of unrolled RNN.
# For 'a' we can try .999, .99, .9, .7, .5, 0
# The loss function might be the trickiest part to get right.
#   We'll start with just loss at final timestep and go from there.
# Conv architecture
#   We need to choose the depth and hidden layers

# Returns an exemplar from each class.
# Return format is (N, W, H)
def GetExemplars(loader):
    exemplars = torch.zeros(10, 28, 28)
    found = [False] * 10
    for index, batch in enumerate(loader):
        for (x, y) in zip(*batch):
            if found[y] == False:
                exemplars[y] = x
                found[y] = True
                if found == [True] * 10:
                    return exemplars

# Main:
#   For image batch
#   Run RNN
#   Get Loss
#   Backprop
#   Save trajectory of first batch

class RNN(nn.Module):
    pass


## MNIST VV

class ConvStack(nn.Module):
    def __init__(self, state, hidden):
        super(ConvStack, self).__init__()
        # 32x32 spatial
        # 32x32 depth
        # Relu
        # 32x128 depth

        # 128x128 spatial
        # 128x128 depth
        # Relu
        # 128x32 depth

        bias = True
        self.conv0 = nn.Conv2d(state, hidden, 3, padding = 1, groups = state, bias = bias)
        self.conv1 = nn.Conv2d(hidden, hidden, 1, bias = bias)
        self.conv2 = nn.Conv2d(hidden, state, 1, groups = state, bias = bias)
        # torch.nn.init.zeros_(self.conv2.weight)

    def forward(self, x):
        return self.conv2(F.elu(self.conv1(self.conv0(x))))

STATE = 64
HIDDEN = 64
CLASSES = 10
import numpy as np
class Net(nn.Module):
    def __init__(self, f, exemplars):
        super(Net, self).__init__()
        conv_params = (STATE, HIDDEN)
        self.W_update = ConvStack(*conv_params)
        self.W_attention = ConvStack(*conv_params)
        self.W_new_x = ConvStack(*conv_params)
        self.f = f

    def forward(self, x):
        # This is a GRU.
        # update = torch.sigmoid(self.W_update(x))
        # attention = torch.sigmoid(self.W_attention(x))
        new_x = torch.tanh(self.W_new_x(x))

        # keep = np.s_[:, 0 : CLASSES + 1, :, :]
        keep = np.s_[:, 0 : 1, :, :]
        # new_x[keep] = x[keep]
        return torch.tanh((x + .5 * new_x) / 3) * 3

STEPS = 8
def train(model, device, train_loader, optimizer, epoch, exemplars, f):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        x = data.expand(-1, STATE, -1, -1).clone()

        # Manual batch-norm in the classifcation grids prevents pixels from diverging to max color
        # (I think because high values are penalized less for relative uncertainty by cross entropy
        #  loss)
        batch_means = torch.mean(data, (2, 3))
        batch_means = torch.unsqueeze(batch_means, -1)
        batch_means = torch.unsqueeze(batch_means, -1)
        x -= batch_means
        x = x.to(device)

        # x[:, 1:11] = exemplars.to(device)
        for t in range(4 * STEPS):
            if t % STEPS == 0:
                x = x.detach()

            x = model(x)
            classification_pixels = x[:, 1 : 1 + CLASSES]
            background = torch.mean(classification_pixels, (1, 2, 3))
            background = torch.unsqueeze(background, -1)
            background = torch.unsqueeze(background, -1)
            background = torch.unsqueeze(background, -1)
            classification_pixels -= background

            if (t + 1) % STEPS == 0:
                classifications = F.log_softmax(torch.sum(classification_pixels, (-1, -2)) / 20)
                loss = F.nll_loss(classifications, target)
                loss.backward(retain_graph = True)

            if t % 4 == 0:
                Draw(f, x[0].detach() / 6 + .5, t)

        optimizer.step()

        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

import matplotlib.pyplot as plt
import math
# Images is a torch array n, h, w
def Draw(f, images, title):
    f.suptitle(title)

#    images -= torch.min(images)
#    images /= torch.max(images)

    # This is a subplot based solution. It's really slow.
    # num_images = int(images.shape[0])
    # grid_width = math.ceil(num_images ** .5)
    # grid_height = math.ceil(num_images / grid_width)
    # axes = f.subplots(grid_height, grid_width)

    # for i in range(num_images):
    #     x, y = i % grid_width, i // grid_width
    #     axes[y, x].imshow(images[i].numpy(), cmap='gray', interpolation='none')

    # for i in range(grid_width * grid_height):
    #     x, y = i % grid_width, i // grid_width
    #     axes[y, x].axis("off")

    # plt.draw()
    # plt.pause(.01)
    # f.clf()

    num_images = int(images.shape[0])
    grid_width = math.ceil(num_images ** .5)
    grid_height = math.ceil(num_images / grid_width)
    composite = torch.zeros(grid_height * 28, grid_width * 28)
    axes = f.subplots(1, 1)
    axes.axis("off")

    for i in range(num_images):
        x, y = i % grid_width, i // grid_width
        composite[28 * y : 28 * y + 28, 28 * x : 28 * x + 28] = images[i]
    axes.imshow(composite.numpy(), cmap='gray', interpolation='none')

    plt.draw()
    plt.pause(.001)
    f.clf()

def main():
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset1 = datasets.MNIST(MNIST_ROOT, train=True, download=True, transform = transforms.ToTensor())
    dataset2 = datasets.MNIST(MNIST_ROOT, train=False, transform = transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size = 64)
    test_loader = torch.utils.data.DataLoader(dataset2)

    exemplars = GetExemplars(train_loader).to(device)

    # inp = 1
    # exemplars = CLASSES
    # prediction = CLASSES
    # hidden_state = W

    f = plt.figure()
    model = Net(f, exemplars.to(device)).to(device)

    x = exemplars[0]


    optimizer = optim.SGD(model.parameters(), lr = .0001, momentum = .99)
    scheduler = StepLR(optimizer, step_size=1, gamma=.1)
    for epoch in range(1, 30):
        train(model, device, train_loader, optimizer, epoch, exemplars, f)
        # test(model, device, test_loader, exemplars)
        scheduler.step()

if __name__ == '__main__':
    main()

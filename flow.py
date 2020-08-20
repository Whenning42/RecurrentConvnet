from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

MNIST_ROOT = "/home/william/Datasets"

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
    def __init__(self, mutable_channels, constant_channels):
        super(ConvStack, self).__init__()
        self.conv0 = nn.Conv2d(mutable_channels + constant_channels, mutable_channels, 3, 1, padding = 1)
        self.conv1 = nn.Conv2d(mutable_channels + constant_channels, mutable_channels, 3, 1, padding = 1)
        self.constant_channels = constant_channels

    def forward(self, x):
        constant_input = x[:, 0:self.constant_channels]
        out_0 = F.relu(self.conv0(x))
        in_1 = torch.cat((constant_input, out_0), 1)
        out_1 = F.relu(self.conv1(in_1))
        return torch.cat((constant_input, out_1), 1)

W = 64
CLASSES = 10
import numpy as np
class Net(nn.Module):
    def __init__(self, exemplars):
        super(Net, self).__init__()
        conv_params = (W + CLASSES, CLASSES + 1)
        self.W_update = ConvStack(*conv_params)
        self.W_attention = ConvStack(*conv_params)
        self.W_new_x = ConvStack(*conv_params)

    def forward(self, x):
        # This is a GRU.
        update = torch.sigmoid(self.W_update(x))
        attention = torch.sigmoid(self.W_attention(x))

        # Need to double check this performs Hadamard(attention, x, ("width", "height"))
        new_x = torch.tanh(self.W_new_x(attention * x))
        keep = np.s_[:, 0 : CLASSES + 1, :, :]
        new_x[keep] = x[keep]
        return update * x + (1 - update) * new_x

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

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

    images -= torch.min(images)
    images /= torch.max(images)

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
    plt.pause(.2)
    f.clf()

def main():
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset1 = datasets.MNIST(MNIST_ROOT, train=True, download=True, transform = transforms.ToTensor())
    dataset2 = datasets.MNIST(MNIST_ROOT, train=False, transform = transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset1)
    test_loader = torch.utils.data.DataLoader(dataset2)

    exemplars = GetExemplars(train_loader).to(device)

    # inp = 1
    # exemplars = CLASSES
    # prediction = CLASSES
    # hidden_state = W

    model = Net(exemplars.to(device)).to(device)

    x = exemplars[0]

    f = plt.figure()

    optimizer = optim.SGD(model.parameters(), lr = .001, momentum = .9)

    for ex in range(10):

        x = torch.zeros(1, 1 + 2 * CLASSES + W, 28, 28).to(device)
        x[0][0] = exemplars[ex].to(device)
        x[0][1:11] = exemplars.to(device)

        for i in range(10):
            x = model(x).detach()
            classification_pixels = x[0][1 + CLASSES : 1 + CLASSES + CLASSES]
            for w in range(10):
                # Sum over the width and height pixels.
                # The constant 200 here is a softmax temperature used to normalize this.
                softmax_normalization = 1/200
                classifications = F.softmax(torch.sum(classification_pixels, (-1, -2)) * \
                                            softmax_normalization)
                print("Weight", w, classifications[w])
            Draw(f, x[0], i)

    scheduler = StepLR(optimizer, step_size=1, gamma=.1)
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()

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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        W = 16
        self.conv1 = nn.Conv2d(16, 16, 3, 1, padding = 1)
        self.conv2 = nn.Conv2d(16, 16, 3, 1, padding = 1)

    def forward(self, x):
        a = .99

        h = x
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        return a * x + (1 - a) * h


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
def Draw(v):
    fig = plt.figure()
    plt.imshow(v.numpy(), cmap='gray', interpolation='none')
    plt.show()

def main():
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset1 = datasets.MNIST(MNIST_ROOT, train=True, download=True, transform = transforms.ToTensor())
    dataset2 = datasets.MNIST(MNIST_ROOT, train=False, transform = transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset1)
    test_loader = torch.utils.data.DataLoader(dataset2)

    exemplars = GetExemplars(train_loader)
    model = Net().to(device)

    x = exemplars[0]
    Draw(x)

    optimizer = optim.SGD(model.parameters(), lr = .001, momentum = .9)
    upped = torch.zeros(1, 16, 28, 28)
    upped[0][0] = x
    out = model(upped.to(device))
    Draw(out[0][0].cpu().detach())
    return

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()

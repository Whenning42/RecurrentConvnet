# Test that I understand how unrolling RNN's works in Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# x_0 = k
# x_1 = f(x)
# x_2 = f(f(x))
# x_3 = f(f(x))
# f is a dense model with 1 input and 1 output.

# Y = 8k
# Check that the model learns f(x) = 2x

# 10 epochs
# Each epoch has 100 samples
#
# Optimizer is SGD, loss is MSE.
# Inputs are random decimals -10 to 10.

class RandomDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(RandomDataset, self).__init__()

    def __len__(self):
        return 1000

    def __getitem__(self, index):
        x = torch.rand(1, ) * 20 - 10
        return x, 8 * x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(1, 1)

    def single(self, x):
        return self.fc(x)

    def forward(self, x):
#        return self.fc(x)
        return self.fc(self.fc(self.fc(x)))

def train(model, device, train_loader, optimizer, epoch):
    log_interval = 10
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def main():
    device = torch.device("cuda")

    dataset = RandomDataset()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size = 10)

    model = Model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = .0001, momentum = 0.9)

    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, epoch)

        items = torch.tensor([[-1.0], [0.0], [1.0]])
        print(items, model.single(items.to(device)))

if __name__ == '__main__':
    main()

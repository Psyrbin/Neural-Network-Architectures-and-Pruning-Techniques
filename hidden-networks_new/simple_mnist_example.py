# General structure from https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd
import numpy as np

args = None



# class GetSubnet(autograd.Function):
#     @staticmethod
#     def forward(ctx, scores_r, scores_s, k):
#         # Get the supermask by sorting the scores and using the top k%
#         out_r = scores_r.clone()
#         _, idx = scores_r.flatten().sort()
#         j = int((1 - k) * scores_r.numel())

#         # flat_out and out access the same memory.
#         flat_out_r = out_r.flatten()
#         flat_out_r[idx[:j]] = 0
#         flat_out_r[idx[j:]] = 1

#         out_s = scores_s.clone()
#         _, idx = scores_s.flatten().sort()
#         j = int((1 - k) * scores_s.numel())

#         # flat_out and out access the same memory.
#         flat_out_s = out_s.flatten()
#         flat_out_s[idx[:j]] = 0
#         flat_out_s[idx[j:]] = 1

#         return out_r, out_s

#     @staticmethod
#     def backward(ctx, g1, g2):
#         # send the gradient g straight-through on the backward pass.
#         return g1, g2, None

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class SupermaskConv(nn.Conv2d):
    def __init__(self, *args, prune_rank1=True, prune=True, sparsity=0.5, **kwargs):
        super().__init__(*args, **kwargs)

        self.prune_rank1 = prune_rank1
        self.prune = prune
        self.sparsity = sparsity

        # initialize the scores

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # After multiplying W by rs^T the stds will be multiplies

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False

        if self.prune_rank1:
            # r.size() == [1, in_channels]
            # s.size() == [1, out_channels]
            self.r = nn.Parameter(torch.Tensor(self.weight.size()[1]))
            self.s = nn.Parameter(torch.Tensor(self.weight.size()[0]))


            self.scores_r = nn.Parameter(torch.Tensor(self.weight.size()[1]))
            self.scores_s = nn.Parameter(torch.Tensor(self.weight.size()[0]))

            nn.init.normal_(self.scores_r, 0, np.sqrt(self.weight.size()[1]))
            nn.init.normal_(self.scores_s, 0, np.sqrt(self.weight.size()[0]))

            nn.init.constant_(self.r, 1)
            nn.init.constant_(self.s, 1)

            self.r.requires_grad = False
            self.s.requires_grad = False
        else:
            self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def forward(self, x):
        if self.prune:
            if self.prune_rank1:
                # mask_r, mask_s = GetSubnet.apply(self.scores_r.abs(), self.scores_s.abs(), self.sparsity)
                # mask_r = GetSubnet.apply(self.scores_r.abs(), self.sparsity)
                mask_s = GetSubnet.apply(self.scores_s.abs(), self.sparsity)

                # create in_channels * out_channels rank 1 matrix, expand it to match shape of weights, elementwise multiply them
                # w = self.weight * torch.unsqueeze(torch.unsqueeze(torch.ger(self.s * mask_s, self.r * mask_r), 2), 3).expand(self.weight.size())
                # w = self.weight * torch.unsqueeze(torch.unsqueeze(torch.ger(self.s, self.r * mask_r), 2), 3).expand(self.weight.size())
                w = self.weight * torch.unsqueeze(torch.unsqueeze(torch.ger(self.s * mask_s, self.r), 2), 3).expand(self.weight.size())
                # print(np.sum(np.array(w.tolist()) == 0), np.prod(w.shape))
            else:
                subnet = GetSubnet.apply(self.scores.abs(), self.sparsity)
                w = self.weight * subnet
        else:
            w = self.weight

        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class SupermaskLinear(nn.Linear):
    def __init__(self, *args, prune_rank1=True, prune=True, sparsity=0.5, **kwargs):
        super().__init__(*args, **kwargs)

        self.prune_rank1 = prune_rank1
        self.prune = prune
        self.sparsity = sparsity

        # initialize the scores

        # NOTE: initialize the weights like this.

        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # After multiplying W by rs^T the stds will be multiplies

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False

        if self.prune_rank1:

            self.r = nn.Parameter(torch.Tensor(self.weight.size()[0]))
            self.s = nn.Parameter(torch.Tensor(self.weight.size()[1]))


            self.scores_r = nn.Parameter(torch.Tensor(self.weight.size()[0]))
            self.scores_s = nn.Parameter(torch.Tensor(self.weight.size()[1]))

            nn.init.normal_(self.scores_r, 0, np.sqrt(self.weight.size()[0]))
            nn.init.normal_(self.scores_s, 0, np.sqrt(self.weight.size()[1]))

            nn.init.constant_(self.r, 1)
            nn.init.constant_(self.s, 1)

            self.r.requires_grad = False
            self.s.requires_grad = False
        else:
            self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))


    def forward(self, x):
        if self.prune:
            if self.prune_rank1:
                # mask_r, mask_s = GetSubnet.apply(self.scores_r.abs(), self.scores_s.abs(), self.sparsity)

                # mask_r = GetSubnet.apply(self.scores_r.abs(), self.sparsity)
                mask_s = GetSubnet.apply(self.scores_s.abs(), self.sparsity)

                # w = self.weight * torch.ger(self.r * mask_r, self.s * mask_s)
                w = self.weight * torch.ger(self.r, self.s * mask_s)
                # w = self.weight * torch.ger(self.r * mask_r, self.s)
                # print(np.sum(np.array(w.tolist()) == 0), np.prod(w.shape))
            else:
                subnet = GetSubnet.apply(self.scores.abs(), self.sparsity)
                w = self.weight * subnet
        else:
            w = self.weight
        return F.linear(x, w, self.bias)
        return x

# NOTE: not used here but we use NON-AFFINE Normalization!
# So there is no learned parameters for your nomralization layer.
class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)

class Net(nn.Module):
    def __init__(self, sparsity=0.9):
        super(Net, self).__init__()
        self.conv1 = SupermaskConv(1, 32, 3, 1, prune_rank1=False, sparsity=0.55, bias=False)
        self.conv2 = SupermaskConv(32, 64, 3, 1, prune_rank1=False, prune=True, sparsity=0.45, bias=False)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)
        # self.fc1 = SupermaskLinear(9216, 128, bias=False)
        # self.fc2 = SupermaskLinear(128, 10, prune_rank1=False, bias=False)

        self.fc1 = SupermaskLinear(9216, 1024, prune_rank1=True, sparsity=0.09, bias=False)
        self.fc2 = SupermaskLinear(1024, 128, prune_rank1=False, prune=False, sparsity=0.31, bias=False)
        self.fc3 = SupermaskLinear(128, 32, prune_rank1=False, sparsity=0.49, bias=False)
        self.fc4 = SupermaskLinear(32, 10, prune_rank1=False, prune=True, sparsity=0.57, bias=False)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)

        x = self.fc3(x)
        x = self.fc4(x)

        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    global args
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=0.0005, metavar='M',
                        help='Weight decay (default: 0.0005)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--data', type=str, default='../data', help='Location to store data')
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='how sparse is each layer')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.join(args.data, 'mnist'), train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.join(args.data, 'mnist'), train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net(args.sparsity).to(device)
    # NOTE: only pass the parameters where p.requires_grad == True to the optimizer! Important!
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
    )
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, criterion, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()

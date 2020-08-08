import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import math

from args import args as parser_args


DenseConv = nn.Conv2d


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
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


# Not learning weights, finding subnet
class SubnetConv(nn.Conv2d):
    def __init__(self, *args, rank1=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank1 = rank1

        if self.rank1:
            self.r = nn.Parameter(torch.Tensor(self.weight.size()[1]))
            self.s = nn.Parameter(torch.Tensor(self.weight.size()[0]))


            #self.scores_r = nn.Parameter(torch.Tensor(self.weight.size()[1]))
            self.scores_s = nn.Parameter(torch.Tensor(self.weight.size()[0]))

            #nn.init.normal_(self.scores_r, 0, math.sqrt(self.weight.size()[1]))
            nn.init.normal_(self.scores_s, 0, math.sqrt(self.weight.size()[0]))

            nn.init.constant_(self.r, 1)
            nn.init.constant_(self.s, 1)

            self.r.requires_grad = False
            self.s.requires_grad = False
        else:
            self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        if self.rank1:
            return self.scores_s.abs()
        else:
            return self.scores.abs()

    def forward(self, x):
        if self.rank1:
            mask_s = GetSubnet.apply(self.clamped_scores, self.prune_rate)
            w = self.weight * torch.unsqueeze(torch.unsqueeze(torch.ger(self.s * mask_s, self.r), 2), 3).expand(self.weight.size())
        else:
            subnet = GetSubnet.apply(self.clamped_scores, self.prune_rate)
            w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class SubnetConv_rank1(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        #nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        self.r = nn.Parameter(torch.Tensor(self.weight.size()[1]))
        self.s = nn.Parameter(torch.Tensor(self.weight.size()[0]))


        #self.scores_r = nn.Parameter(torch.Tensor(self.weight.size()[1]))
        self.scores_s = nn.Parameter(torch.Tensor(self.weight.size()[0]))

        #nn.init.normal_(self.scores_r, 0, math.sqrt(self.weight.size()[1]))
        nn.init.normal_(self.scores_s, 0, math.sqrt(self.weight.size()[0]))

        nn.init.constant_(self.r, 1)
        nn.init.constant_(self.s, 1)

        self.r.requires_grad = False
        self.s.requires_grad = False

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores_s(self):
        return self.scores_s.abs()

    def forward(self, x):
        mask_s = GetSubnet.apply(self.clamped_scores_s, self.prune_rate)
        w = self.weight * torch.unsqueeze(torch.unsqueeze(torch.ger(self.s * mask_s, self.r), 2), 3).expand(self.weight.size())
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


"""
Sample Based Sparsification
"""


class StraightThroughBinomialSample(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        output = (torch.rand_like(scores) < scores).float()
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None


class BinomialSample(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        output = (torch.rand_like(scores) < scores).float()
        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        subnet, = ctx.saved_variables

        grad_inputs = grad_outputs.clone()
        grad_inputs[subnet == 0.0] = 0.0

        return grad_inputs, None


# Not learning weights, finding subnet
class SampleSubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        if parser_args.score_init_constant is not None:
            self.scores.data = (
                torch.ones_like(self.scores) * parser_args.score_init_constant
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    @property
    def clamped_scores(self):
        return torch.sigmoid(self.scores)

    def forward(self, x):
        subnet = StraightThroughBinomialSample.apply(self.clamped_scores)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return x


"""
Fixed subnets 
"""


class FixedSubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        print("prune_rate_{}".format(self.prune_rate))

    def set_subnet(self):
        output = self.clamped_scores().clone()
        _, idx = self.clamped_scores().flatten().abs().sort()
        p = int(self.prune_rate * self.clamped_scores().numel())
        flat_oup = output.flatten()
        flat_oup[idx[:p]] = 0
        flat_oup[idx[p:]] = 1
        self.scores = torch.nn.Parameter(output)
        self.scores.requires_grad = False

    def clamped_scores(self):
        return self.scores.abs()

    def get_subnet(self):
        return self.weight * self.scores

    def forward(self, x):
        w = self.get_subnet()
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


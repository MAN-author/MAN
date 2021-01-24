import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
class GraphConvolution(nn.Module):
    def __init__(self, bias=False):
        super(GraphConvolution, self).__init__()

        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.bias = None
    def set_w(self, n):
         ###初始化
        w = torch.Tensor(n,n)
        torch.nn.init.kaiming_normal(w, a=0, mode='fan_in')
        
        ###
        self.weight = Parameter(w)       


    def gen_adj(self, input):
        norm = F.normalize(input)
        norm2 = norm
        adj = norm.mm(norm2.t())
        assert adj.size(0) == input.size(0) and adj.size(1) == input.size(0)
        return adj

    def forward(self, input):
        self.A = self.gen_adj(input)
        #pdb.set_trace()
        AX = torch.mm(self.A, input)
        #pdb.set_trace()
        AXw = torch.mm(AX, self.weight)
        if self.bias is not None:
            return AXw + self.bias
        else:
            return AXw
            
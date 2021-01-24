from torch import nn
import torch
import pdb
class FC3(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(FC3, self).__init__()
        #self.features = features
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
        
    def forward(self, features):
        features = self.layer1(features)
        #pdb.set_trace()
        features = self.layer2(features)
        features = self.layer3(features)
        return features

    
class fc_cluster():
    def __init__(self,k):
        super(fc_cluster, self).__init__()
        self.k = k
    def run_cluster(self, x):
        #print(x.shape)
        #x=torch.from_numpy(x)
        #pdb.set_trace()
        #x = x.cuda()
        model = FC3(x.size(1), 512, 256, self.k)
        #pdb.set_trace()
        y = model(x)
        #print(y)
        y_label = torch.max(y,1)[1]
        #print(y_label)
        y_label = y_label.numpy().reshape(-1)
        return y_label
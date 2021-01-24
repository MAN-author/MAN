from torch import nn
import torch
import torch.nn.functional as F
from man.architectures.classifiers import classifier
from man.architectures.feature_extractors import convnet
from man.architectures.feature_extractors import convnet_pre_act
from man.architectures.feature_extractors import resnet_block
from man.architectures.feature_extractors import wrn_block


class ConvnetPlusClassifier(nn.Module):
    def __init__(self, opt):
        super().__init__()
        convnet_type = opt["convnet_type"]
        self.lamda = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        if convnet_type == "convnet":
            convolutional_net = convnet.create_model(opt["convnet_opt"])
        elif convnet_type == "convnet_pre_act":
            convolutional_net = convnet_pre_act.create_model(opt["convnet_opt"])
        elif convnet_type == "resnet_block":
            convolutional_net = resnet_block.create_model(opt["convnet_opt"])
        elif convnet_type == "wrn_block":
            convolutional_net = wrn_block.create_model(opt["convnet_opt"])
        else:
            raise KeyError(convnet_type)

        self.layers = nn.Sequential(
            convolutional_net,
            classifier.create_model(opt["classifier_opt"])
        )

    def forward(self, features,labels):
        classification_scores = self.layers(features)
        loss = F.cross_entropy(classification_scores, labels)
        loss = 2*loss * (torch.sigmoid(self.lamda)+1)
        #loss = loss * (torch.tanh(self.lamda)+1)
        return classification_scores,loss


def create_model(opt):
    return ConvnetPlusClassifier(opt)

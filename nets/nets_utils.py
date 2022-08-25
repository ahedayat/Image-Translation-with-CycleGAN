import os
import random
import torch
import torch.nn as nn
from torch.autograd import Variable


def c7s1_k(in_channel, out_channel, reflection_pad=False):
    """
        - `c7s1_k`:
            "
            Let c7s1-k denote a 7 × 7 Convolution-InstanceNorm-
            ReLU layer with k filters and stride 1
            "
    """
    block = []

    if reflection_pad:
        block.append(nn.ReflectionPad2d(in_channel))
    block += [
        nn.Conv2d(in_channel, out_channel, 7),
        nn.InstanceNorm2d(out_channel),
        nn.ReLU(inplace=True)
    ]
    # block = nn.Sequential(*block)

    return block


def d_k(in_channel, out_channel):
    """
        - `dk`:
        "
        dk denotes a 3 × 3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2.
        "
    """
    # block = nn.Sequential(
    #     nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=2),
    #     nn.InstanceNorm2d(out_channel),
    #     nn.ReLU(inplace=True)
    # )
    block = [
        nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=2),
        nn.InstanceNorm2d(out_channel),
        nn.ReLU(inplace=True)
    ]

    return block


def u_k(in_channel, out_channel):
    """
        - `uk`:
        "
        uk denotes a 3 × 3 fractional-strided-Convolution-InstanceNorm-ReLU 
        layer with k filters and stride 0.5(1/2) .
        "
    """
    # block = nn.Sequential(
    #     nn.Upsample(scale_factor=2),
    #     nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=0.5),
    #     nn.InstanceNorm2d(out_channel),
    #     nn.ReLU(inplace=True)
    # )

    block = [
        nn.Upsample(scale_factor=2),
        nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=1),
        nn.InstanceNorm2d(out_channel),
        nn.ReLU(inplace=True)
    ]

    return block


def c_k(in_channel, out_channel, instance_norm=True):
    """
        - `ck`:
        "
        Ck denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2.
        "
    """
    block = list()

    block.append(nn.Conv2d(in_channel, out_channel, 4, padding=1, stride=2))

    if instance_norm:
        block.append(nn.InstanceNorm2d(out_channel))

    block.append(nn.LeakyReLU(0.2, inplace=True))

    # return nn.Sequential(*block)
    return block


def init_conv_layer_normal(m):
    """
        - Reference: \
        https://github.com/rohan-paul/MachineLearning-DeepLearning-Code-for-my-YouTube-Channel.git
    """
    # classname = m.__class__.__name__
    # if classname.find("Conv") != -1:
    #     nn.init.normal_(m.weight.data, 0.0, 0.02)
    #     if hasattr(m, "bias") and m.bias is not None:
    #         nn.init.constant_(m.bias.data, 0.0)
    # elif classname.find("BatchNorm2d") != -1:
    #     nn.init.normal_(m.weight.data, 1.0, 0.02)
    #     nn.init.constant_(m.bias.data, 0.0)

    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ReplayBuffer:
    """
        Replay Buffer:
        ------------------------------------------
        - Paper: 
        "
        We keep an image buffer that stores the 50 previously created images.
        "
    """

    def __init__(self, max_size=50, buffer_prob=0.5):
        self.max_size = max_size
        self.buffer = list()
        self.buffer_prob = buffer_prob

    def push_pop(self, data):
        """
            This method add new data with probability of 'p' and otherwise,
            returns an older image
        """
        return_list = list()

        for d in data.data:
            if len(self.buffer) < self.max_size:
                self.buffer.append(d)
                return_list.append(d.unsqueeze(dim=0))

            else:
                if random.uniform(0, 1) < self.buffer_prob:
                    return_list.append(d.unsqueeze(dim=0))

                else:
                    ix = random.randint(0, self.max_size - 1)
                    old_data = self.buffer[ix]
                    self.buffer[ix] = d
                    return_list.append(old_data.unsqueeze(dim=0))

        return Variable(torch.cat(return_list))


def save_net(file_path, file_name, model, optimizer=None):
    """
        In this function, a model is saved.
        ------------------------------------------------
        Parameters:
            - file_path (str): saving path
            - file_name (str): saving name
            - model (torch.nn.Module)
            - optimizer (torch.optim)
    """
    state_dict = dict()

    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()

    torch.save(state_dict, os.path.join(file_path, file_name))


def load_net(ckpt_path, model, optimizer=None):
    """
        Loading Network
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model"])
    if ((optimizer != None) & ("optimizer" in checkpoint.keys())):
        optimizer.load_state_dict(checkpoint["optimizer"])

    return model, optimizer

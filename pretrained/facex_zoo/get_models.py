"""
Author: Huynh Van Thong
"""
import os.path

import torch

from .MobileFaceNets import MobileFaceNet
from .EfficientNets import EfficientNet, efficientnet

def load_model(model, path_weight):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(path_weight)['state_dict']
    new_pretrained_dict = {}
    for k in model_dict:
        new_pretrained_dict[k] = pretrained_dict['backbone.{}'.format(k)]

    model_dict.update(new_pretrained_dict)
    model.load_state_dict(model_dict)

    return model

def get_facex_zoo(model_name, root_weights):
    """

    :param model_name: MobileFaceNets, EfficientNets-B0
    :return:
    """

    if model_name == 'MobileFaceNet':
        backbone = MobileFaceNet(embedding_size=512, out_h=7, out_w=7)
        backbone = load_model(backbone, os.path.join(root_weights, 'MobileFaceNet.pt'))
    elif model_name == 'EfficientNets-B0':
        blocks_args, global_params = efficientnet(
            width_coefficient=1.0, depth_coefficient=1.0,
            dropout_rate=0.2, image_size=112)
        backbone = EfficientNet(out_h=7, out_w=7, feat_dim=512, blocks_args=blocks_args, global_params=global_params)
        backbone = load_model(backbone, os.path.join(root_weights, 'EfficientNet-B0.pt'))
    else:
        raise ValueError('Unknown backbone {} in FaceX-Zoo at this time.')

    return backbone

import torch
import argparse
import sys
sys.path.append('./RAFT/core')
from raft import RAFT
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
args, _ = parser.parse_known_args()


class EvalRAFT:
    def __init__(self, flow_threshold, ckpt_path='./RAFT/models/raft-sintel.pth', device='cuda'):
        model = RAFT(args) #torch.nn.DataParallel(RAFT(args))
        state_dict = torch.load(ckpt_path, map_location=device) #lambda storage, loc: storage)
        
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        model.eval()
        self.model = model.to(device)
        self.flow_threshold = flow_threshold

    def __call__(self, image0, image1):
        assert image0.dtype == torch.uint8, "Inputs must be integers, unnormalized in range [0, 10]"
        _, flow = self.model(image0, image1, iters=20, test_mode=True)
        magnitude = (flow ** 2).sum(1) ** 0.5
        motion_segment = (magnitude > self.flow_threshold).unsqueeze(1)
        return flow, magnitude, motion_segment

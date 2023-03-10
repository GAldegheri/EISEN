import torch
from core.eisen import EISEN
from core.raft import EvalRAFT
from core.datasets import fetch_dataloader
from types import SimpleNamespace
from collections import OrderedDict

if __name__ == "__main__":
    
    device = torch.device("cuda:1")
    
    args = SimpleNamespace(
        dataset='playroom',
        batch_size=1,
        num_workers=1,
        compute_flow=True,
        precompute_flow=False,
        flow_threshold=0.5
    )
    
    val_loader = fetch_dataloader(args)
    model = EISEN(device=device)
    raft_model = EvalRAFT(flow_threshold=args.flow_threshold, device=device)
    
    data_dict = next(iter(val_loader))
    data_dict = {k: v.to(device) for k, v in data_dict.items() if not k == 'file_name'}
    
    with torch.no_grad():
       _, _, segment_target = raft_model(data_dict['img1'], data_dict['img2'])
       
    _, loss, metric, seg_out = model(data_dict, segment_target.detach(),
                                     get_segments=True, vis_segments=False)
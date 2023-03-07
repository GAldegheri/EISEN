import torch
import torch.nn as nn
from core.datasets import fetch_dataloader
from lescroartdata import fetch_dataloader_lesc
import core.utils.sync_batchnorm as sync_batchnorm
from core.eisen import EISEN
from core.raft import EvalRAFT
from types import SimpleNamespace

if __name__=="__main__":
    
    args = SimpleNamespace(
        dataset='playroom',
        batch_size=8,
        num_workers=8,
        compute_flow=True,
        precompute_flow=False,
        flow_threshold=0.5
    )
    
    val_loader = fetch_dataloader(args)
    model = nn.DataParallel(EISEN())
    model = sync_batchnorm.convert_model(model)
    raft_model = EvalRAFT(flow_threshold=args.flow_threshold)
    
    state_dict = torch.load('pretrained/tdw_playroom_128x128_ckpt.pth')
    model.load_state_dict(state_dict)
    
    model.cuda()
    model.eval()
    #print(type(val_loader))
    
    data_dict = next(iter(val_loader))
    data_dict = {k: v.cuda() for k, v in data_dict.items() if not k == 'file_name'}
    
    # print('Fields in data dict:')
    # print(data_dict.keys())
    # print('Image shape:')
    # print(data_dict['img1'].shape)
    #print(data_dict['img2'].shape)
    #print(data_dict['gt_segment'].shape)
    
    bs = data_dict['img1'].shape[0]
    if bs < args.batch_size:
        pad_size = args.batch_size - bs
        for key in data_dict.keys():
            padding = torch.cat([torch.zeros_like(data_dict[key][0:1])] * pad_size, dim=0)
            data_dict[key] = torch.cat([data_dict[key], padding], dim=0)
    
    with torch.no_grad():
        _, _, segment_target = raft_model(data_dict['img1'], data_dict['img2'])
    
    _, loss, metric, seg_out = model(data_dict, segment_target.detach(),
                                         get_segments=True, vis_segments=False)
    
    
    print('Fields in output dict:')
    print(seg_out.keys())
    # lesc_loader = fetch_dataloader_lesc(args)
    
    # data_dict_lesc = next(iter(lesc_loader))
    
    # print('Lescroart - Image shape:')
    # print(data_dict_lesc['img1'].shape)
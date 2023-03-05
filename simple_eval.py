import torch
import torch.nn as nn
from core.datasets import fetch_dataloader
from core.eisen import EISEN
from core.raft import EvalRAFT
from types import SimpleNamespace
import core.utils.sync_batchnorm as sync_batchnorm
import numpy as np
import logging
import time
import datetime
# ---- Suppresses warnings: ------
import shutup
# --------------------------------
num_gpus = torch.cuda.device_count()

def evaluate_helper(args, dataloader, model, raft_model=None):
    shutup.please()
    model.eval()
    end = time.time()
    miou_list, loss_list, time_list = [], [], []
    
    for step, data_dict in enumerate(dataloader):
        start = time.time()
        data_time = time.time() - end
        data_dict = {k: v.cuda() for k, v in data_dict.items() if not k == 'file_name'}
        
        bs = data_dict['img1'].shape[0]
        if bs < args.batch_size:  # add padding if bs is smaller than batch size (required since drop_last is set to False)
            pad_size = args.batch_size - bs
            for key in data_dict.keys():
                padding = torch.cat([torch.zeros_like(data_dict[key][0:1])] * pad_size, dim=0)
                data_dict[key] = torch.cat([data_dict[key], padding], dim=0)
        
        if args.compute_flow:
            with torch.no_grad():
                _, _, segment_target = raft_model(data_dict['img1'], data_dict['img2'])
        else:
            segment_target = data_dict['segment_target']
            
        raft_time = time.time() - end
        _, loss, metric, segment = model(data_dict, segment_target.detach(),
                                         get_segments=True, vis_segments=False)
        mious = metric['metric_pred_segment_mean_ious']
        
        step_time = time.time() - end
        eta = int((len(dataloader) - step) * step_time)
        
        if num_gpus == 1:
            loss_list.append(loss.item())
            miou_list.append(mious.item())
        else:
            loss_list.extend([l.item() for l in list(loss)][0:bs])
            miou_list.extend([miou.item() for miou in list(mious)][0:bs])
            
        if (step + 1) % 10 == 0 or (step + 1) == len(dataloader):
            avg_miou = np.nanmean(miou_list)
            avg_loss = np.nanmean(loss_list)
            print(f"[val] iter: {step} avg_miou: {avg_miou:.3f} avg_loss: {avg_loss:.3f}")
            print(f"data_time: {data_time:.4f}  raft_time: {raft_time:.3f} step_time: {step_time:.3f}")
            print(f"eta: {str(datetime.timedelta(seconds=eta))}")
        end = time.time()
        time_list.append(end-start)
    
    return miou_list, avg_loss, time_list

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
    raft_model = EvalRAFT(flow_threshold=args.flow_threshold) if args.compute_flow else None
    
    state_dict = torch.load('pretrained/tdw_playroom_128x128_ckpt.pth')
    model.load_state_dict(state_dict)
    
    model.cuda()
    
    miou_list, avg_loss, time_list = evaluate_helper(args, val_loader, model, raft_model=raft_model)
    
    print('+--------------------+')
    print('| Validation summary |')
    print('+--------------------+')
    print(f'| Num. Imgs |  {len(miou_list)}   |')
    print(f'| Avg. mIoU |  {np.nanmean(miou_list):.3f} |')
    print(f'| Avg. time |  {np.mean(time_list[5:]):.3f} |') # the first 5 iterations are discarded
    print(f'| Avg. loss |  {np.nanmean(avg_loss):.3f} |')
    print('+--------------------+')
    
    
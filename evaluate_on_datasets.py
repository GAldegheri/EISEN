import torch
from torch.utils.data import DataLoader
from lescroartdata import LescroartDataset, BonnerDataset
import matplotlib.pyplot as plt
from core.eisen import EISEN
from core.raft import EvalRAFT
from collections import OrderedDict
import wandb
import shutup

def visualize_segments(seg_out, input):
    batch_size = len(seg_out[0])
    num_objects = len(seg_out[0][0])
    pred_obj_seg, gt_obj_seg, iou = seg_out
    fig, axs = plt.subplots(batch_size, num_objects + 1, 
                            figsize=(3*(num_objects + 1), 3*batch_size))
    
    for m in range(batch_size):
        axs[m, 0].imshow(input['img1'][m].permute(1, 2, 0).cpu())
        for n in range(num_objects):
            #if n + 1 <= len(pred_obj_seg[m]): 
            axs[m, n + 1].imshow(pred_obj_seg[m][n])
            
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].set_axis_off()
    
    #plt.show()
    #plt.close()
    return fig

def eval():
    
    shutup.please()
    
    device = torch.device("cuda:1")
    
    #dataset= LescroartDataset()
    dataset = BonnerDataset()
    
    val_loader = DataLoader(dataset, 
                            batch_size=2,
                            pin_memory=False,
                            shuffle=False)
    
    model = EISEN(device=device)
    state_dict = torch.load('pretrained/tdw_playroom_128x128_ckpt.pth', map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    
    wandb.init(project='Eisen_Bonner')
    
    # fake segment target:
    segment_target = torch.rand((2, 1, 512, 512), device=device) < 0.9  
    
    with torch.no_grad():
    
        for step, data_dict in enumerate(val_loader):
            
            data_dict = {k: v.to(device) for k, v in data_dict.items() if not k == 'file_name'}
            
            _, loss, metric, seg_out = model(data_dict, segment_target.detach(),
                                    get_segments=True, vis_segments=False)
            
            for x in seg_out['pred_segment']:
                print(len(x[0]))
            
            fig = visualize_segments(seg_out['pred_segment'], data_dict)
            
            wandb.log({"Predictions": fig})
            torch.cuda.empty_cache()
        
if __name__=="__main__":
    
    eval()
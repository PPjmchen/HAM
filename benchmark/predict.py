import os
import sys
import json
import pickle
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from lib.config import CONF
from lib.dataset import ScannetReferenceDataset
from lib.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from lib.loss_helper import get_loss
from lib.eval_helper import get_eval
from models.ham import HAM
from utils.box_util import get_3d_box
from data.scannet.model_util_scannet import ScannetDatasetConfig
from benchmark.eval import evaluate
from utils.pc_utils import write_ply_rgb


def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

def get_dataloader(args, scanrefer, scanrefer_new, all_scene_list, split, config):
    # TODO: Custom Dataset Params

    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer, 
        scanrefer_new=scanrefer_new,
        scanrefer_all_scene=all_scene_list, 
        split=split, 
        num_points=args.num_points, 
        use_height=(not args.no_height),
        use_color=args.use_color, 
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview,
        sent_len_max=args.sent_len_max,
        lang_num_max=args.lang_num_max,
        augment=False,
        sent_aug=False,
    )

    print("predict for {} samples".format(len(dataset)))

    dataloader = DataLoader(dataset, 
                                batch_size=args.batch_size, 
                                shuffle=False, 
                                num_workers=args.workers, 
                                # worker_init_fn=my_worker_init_fn,
                                pin_memory=True,
                                drop_last=False)
    return dataset, dataloader

def get_model(args, config):
    # load model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    model = HAM(
        num_class=config.num_class,
        num_heading_bin=config.num_heading_bin,
        num_size_cluster=config.num_size_cluster,
        mean_size_arr=config.mean_size_arr,
        num_proposal=args.num_proposals,
        sent_len_max=args.sent_len_max,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir,
        no_reference=args.no_reference,
        sampling='kps',
        num_decoder_layers=args.num_decoder_layers,
        fuse_with_query=args.fuse_with_query,
        fuse_with_key=args.fuse_with_key,
        backbone_width=args.backbone_width, 
        multi_window=args.multi_window,
        use_color = args.use_color,
        use_normal = args.use_normal,
        no_height = args.no_height,
        fps_method = args.fps_method,
        use_spa = args.use_spa,
    ).cuda()
    model = DDP(model, device_ids=[args.local_rank], broadcast_buffers=False, )
    
    model_name = "model.pth"
    path = os.path.join(CONF.PATH.OUTPUT, args.folder, model_name)
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    return model

def get_scanrefer(args):

    SCANREFER_TEST = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_%s.json" % args.pred_split)))

    scanrefer = SCANREFER_TEST
    scene_list = sorted(list(set([data["scene_id"] for data in scanrefer])))

    if args.pred_split == 'test':
        order_scanrefer = []
        all_data = {}
        for scene_id in scene_list:
            all_data[scene_id] = []
            for data in scanrefer:
                if data['scene_id'] == scene_id:
                    all_data[scene_id].append(data)
        
        for key, value in all_data.items():

            order_scanrefer += value
        
        scanrefer = order_scanrefer
    else:
        scanrefer = [data for data in scanrefer if data["scene_id"] in scene_list]

    new_scanrefer_val = scanrefer
    scanrefer_val_new = []
    scanrefer_val_new_scene = []
    scene_id = ""
    for data in scanrefer:
        if scene_id != data["scene_id"]:
            scene_id = data["scene_id"]
            if len(scanrefer_val_new_scene) > 0:
                scanrefer_val_new.append(scanrefer_val_new_scene)
            scanrefer_val_new_scene = []
        if len(scanrefer_val_new_scene) >= args.lang_num_max:
            scanrefer_val_new.append(scanrefer_val_new_scene)
            scanrefer_val_new_scene = []
        scanrefer_val_new_scene.append(data)
    if len(scanrefer_val_new_scene) > 0:
        scanrefer_val_new.append(scanrefer_val_new_scene)

    return scanrefer, scene_list, scanrefer_val_new

def get_random_state():
    python_random_state = random.getstate()
    np_random_state = np.random.get_state()
    torch_random_state = torch.get_rng_state()
    torch_cuda_random_state = torch.cuda.get_rng_state()
    torch_cuda_random_state_all = torch.cuda.get_rng_state_all()

    random_state = {'python_random_state': python_random_state, 
                    'np_random_state': np_random_state,
                    'torch_random_state': torch_random_state,
                    'torch_cuda_random_state': torch_cuda_random_state,
                    'torch_cuda_random_state_all': torch_cuda_random_state_all}
    return random_state

def load_random_state():
    random_state = torch.load('outputs/'+args.folder+'/random_state.pth')

    random.setstate(random_state['python_random_state'])
    np.random.set_state(random_state['np_random_state'])
    torch.set_rng_state(random_state['torch_random_state'])

    return random_state

def predict(args, model=None):
    # seed init

    if model == None:  # testing stage
        random_state = load_random_state()
    
    else:
        random_state = get_random_state()
        

        torch.save(random_state, os.path.join(CONF.PATH.OUTPUT, args.folder, "random_state.pth"))
    
    print("predict bounding boxes...")
    # constant
    DC = ScannetDatasetConfig()

    # init training dataset
    print("preparing data...")
    scanrefer, scene_list, scanrefer_val_new = get_scanrefer(args)

    # dataloader
    _, dataloader = get_dataloader(args, scanrefer, scanrefer_val_new, scene_list, args.pred_split, DC)


     # model
    if model == None:
        # For indenpendently testing
        model = get_model(args, DC)

    # predict
    print("predicting...")
    pred_bboxes = []
    pred_bboxes_upload = []

    tmp=0


    allref = 0
    confident =0
    for data_dict in tqdm(dataloader):
        for key in data_dict:
            data_dict[key] = data_dict[key].cuda()
        
        with torch.no_grad():
            data_dict = model(data_dict)

        tmp += 1
        objectness_preds_batch = torch.argmax(data_dict['last_objectness_scores'], 2).long()

        pred_ref = torch.argmax(data_dict['cluster_ref'], 1)  # (B,)
        pred_center = data_dict['center'] # (B,K,3)
        pred_heading_class = torch.argmax(data_dict['heading_scores'], -1) # B,num_proposal
        pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
        pred_heading_class = pred_heading_class # B,num_proposal
        pred_heading_residual = pred_heading_residual.squeeze(2) # B,num_proposal
        pred_size_class = torch.argmax(data_dict['size_scores'], -1) # B,num_proposal
        pred_size_residual = torch.gather(data_dict['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
        pred_size_class = pred_size_class
        pred_size_residual = pred_size_residual.squeeze(2) # B,num_proposal,3

        batch_size, lang_num_max = data_dict['ref_center_label_list'].shape[:2]
        pred_ref = pred_ref.reshape(batch_size, args.lang_num_max)

        
        for ref in data_dict['cluster_ref']:
            
            if ref.max() > 5:
                confident +=1
            allref+=1

        for i in range(pred_ref.shape[0]):
            # compute the iou
            for j in range(args.lang_num_max):
                if j < data_dict['lang_num'][i]:

                    pred_ref_idx = pred_ref[i][j]
                    
                    if args.pred_split == 'test':
                        pred_center_offset = pred_center[i, pred_ref_idx, 0:3].detach().cpu().numpy()
                        pred_center_offset[:2] += data_dict['point_offset'][i].cpu().numpy()

                        pred_obb = DC.param2obb(
                            pred_center_offset, 
                            pred_heading_class[i, pred_ref_idx].detach().cpu().numpy(), 
                            pred_heading_residual[i, pred_ref_idx].detach().cpu().numpy(),
                            pred_size_class[i, pred_ref_idx].detach().cpu().numpy(), 
                            pred_size_residual[i, pred_ref_idx].detach().cpu().numpy()
                        )
                    else:
                        pred_obb = DC.param2obb(
                            pred_center[i, pred_ref_idx, 0:3].detach().cpu().numpy(), 
                            pred_heading_class[i, pred_ref_idx].detach().cpu().numpy(), 
                            pred_heading_residual[i, pred_ref_idx].detach().cpu().numpy(),
                            pred_size_class[i, pred_ref_idx].detach().cpu().numpy(), 
                            pred_size_residual[i, pred_ref_idx].detach().cpu().numpy()
                        )
                    pred_bbox = get_3d_box(pred_obb[3:6], pred_obb[6], pred_obb[0:3])

                    # construct the multiple mask
                    multiple = data_dict["unique_multiple_list"][i][j].item()

                    # construct the others mask
                    others = 1 if data_dict["object_cat_list"][i][j] == 17 else 0

                    # store data
                    scanrefer_idx = data_dict["scan_idx"][i].item()


                    pred_data = {
                        "scene_id": scanrefer_val_new[scanrefer_idx][j]["scene_id"],
                        "object_id": scanrefer_val_new[scanrefer_idx][j]["object_id"],
                        "ann_id": scanrefer_val_new[scanrefer_idx][j]["ann_id"],
                        "bbox": pred_bbox.tolist(),
                        'pred_obb': pred_obb.tolist(),
                        "unique_multiple": multiple,
                        "others": others
                    }

                    pred_data_upload = {
                        "scene_id": scanrefer_val_new[scanrefer_idx][j]["scene_id"],
                        "object_id": scanrefer_val_new[scanrefer_idx][j]["object_id"],
                        "ann_id": scanrefer_val_new[scanrefer_idx][j]["ann_id"],
                        "bbox": pred_bbox.tolist(),
                        "unique_multiple": multiple,
                        "others": others
                    }
                    pred_bboxes.append(pred_data)

                    pred_bboxes_upload.append(pred_data_upload)

    # dump
    if args.pred_split == 'test':
        print("dumping...")
        pred_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "pred_test.json")
        with open(pred_path, "w") as f:
            json.dump(pred_bboxes, f, indent=4)

        pred_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "pred_upload.json")
        with open(pred_path, "w") as f:
            json.dump(pred_bboxes_upload, f, indent=4)

        print("done!")

    else:
        print("dumping...")
        pred_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "pred.json")
        with open(pred_path, "w") as f:
            json.dump(pred_bboxes, f, indent=4)

        print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder containing the model")
    parser.add_argument("--batch_size", type=int, help="batch size", default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--num_points", type=int, default=50000)
    parser.add_argument("--num_proposals", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_nms", action="store_true", help="do NOT use non-maximum suppression for post-processing.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")  # NEED
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.") # NEED
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--no_reference", action="store_true", help="Do NOT train the localization module.")
    parser.add_argument("--no_detection", action="store_true", help="Do NOT train the detection module.")

    parser.add_argument("--backbone_width", type=int, default=2)
    parser.add_argument("--num_decoder_layers", type=int, default=12)
    parser.add_argument("--fuse_with_query", action='store_true')
    parser.add_argument("--fuse_with_key", action='store_true') # NEED
    parser.add_argument("--multi_window", nargs='+', type=int, default=[4])
    parser.add_argument("--use_spa", action='store_true')
    parser.add_argument("--fps_method", type=str, choices=['F-FPS', 'D-FPS', 'D-F-FPS', 'CS'], default='CS')
    parser.add_argument("--lang_num_max", type=int, help="lang num max", default=8)
    parser.add_argument("--sent_len_max", type=int, help="max sentence length", default=200)

    parser.add_argument("--local_rank", type=int)

    parser.add_argument("--pred_split", type=str, choices=['val', 'test'],)

    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    
    seed_all(args.seed)

    
    predict(args)

    if args.pred_split == 'val':
        evaluate(args)
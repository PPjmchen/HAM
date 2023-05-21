from base64 import decode
from logging import critical
import os
from statistics import mode
import sys
import json
import argparse
from wsgiref.simple_server import demo_app
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import random
import time
import copy

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
from copy import deepcopy
from torch.profiler import record_function

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset import ScannetReferenceDataset
from lib.config import CONF
from lib.loss_helper import get_loss
from lib.eval_helper import get_eval
from utils import setup_logger, get_scheduler, decode_eta
from models.ham import HAM
from tqdm import tqdm
from benchmark.predict import predict, evaluate
 

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

# constants
DC = ScannetDatasetConfig()



def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list

# Intra-Sentence Ensemble
def sentence_augmentation(scanrefer_train, sent_aug_ratio=1):
    train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
    scene_packs = {k: [] for k in train_scene_list}
    
    min_aug_num, max_aug_num = args.sent_aug_num

    for data in scanrefer_train:
        scene_packs[data['scene_id']].append(data)
    
    scanrefer_train_sent_aug = []

    
    for scene_id, pack in scene_packs.items():
        obj_num = len(pack)
        for _ in range(int(len(pack) * sent_aug_ratio)):
            while True:
                sample_num = random.randint(min_aug_num, max_aug_num)
                sub_pack = random.sample(pack, sample_num)
                sub_pack_obj_ids = list(set([int(k['object_id']) for k in sub_pack]))
                if len(sub_pack_obj_ids) == len(sub_pack): # no repetitive sample
                    agg_sample = agg_multi_sample(sub_pack)
                    break
            scanrefer_train_sent_aug.append(agg_sample)
    return scanrefer_train_sent_aug

                
def agg_multi_sample(pack):
    token = copy.deepcopy(pack[0]['token'])
    for i in range(len(pack)-1):
        token.extend(pack[i+1]['token'])
    agg_sample = {'scene_id': pack[0]['scene_id'],
                  'object_id': '-'.join([k['object_id'] for k in pack]),
                  'object_name': '-'.join([k['object_name'] for k in pack]),
                  'ann_id': '-'.join([k['ann_id'] for k in pack]),
                  'description': ' '.join([k['description'] for k in pack]),
                  'token': token}
    return agg_sample
        

def get_scanrefer(scanrefer_train, scanrefer_val, num_scenes, lang_num_max, sent_aug=False, sent_aug_ratio=1):
    if args.no_reference:
        train_scene_list = get_scannet_scene_list("train")
        new_scanrefer_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_train.append(data)

        val_scene_list = get_scannet_scene_list("val")
        new_scanrefer_val = []
        for scene_id in val_scene_list:
            data = deepcopy(SCANREFER_VAL[0])
            data["scene_id"] = scene_id
            new_scanrefer_val.append(data)
    else:

        if sent_aug:
            scanrefer_train_sent_aug = sentence_augmentation(scanrefer_train, sent_aug_ratio=sent_aug_ratio)
            scanrefer_train = list(np.append(scanrefer_train, scanrefer_train_sent_aug))
            random.shuffle(scanrefer_train)
            scanrefer_train.sort(key=lambda x: x['scene_id'])
        
        # get initial scene list
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
        if num_scenes == -1: 
            num_scenes = len(train_scene_list)
        else:
            assert len(train_scene_list) >= num_scenes
        
        # slice train_scene_list
        train_scene_list = train_scene_list[:num_scenes]


        new_scanrefer_train = []
        scanrefer_train_new = []
        scanrefer_train_new_scene = []
        scene_id = ""
        for data in scanrefer_train:
            if data["scene_id"] in train_scene_list:
                new_scanrefer_train.append(data)
                if scene_id != data["scene_id"]:
                    scene_id = data["scene_id"]
                    if len(scanrefer_train_new_scene) > 0:
                        scanrefer_train_new.append(scanrefer_train_new_scene)
                    scanrefer_train_new_scene = []
                if len(scanrefer_train_new_scene) >= lang_num_max:
                    scanrefer_train_new.append(scanrefer_train_new_scene)
                    scanrefer_train_new_scene = []
                scanrefer_train_new_scene.append(data)

        scanrefer_train_new.append(scanrefer_train_new_scene)
        new_scanrefer_val = scanrefer_val
        scanrefer_val_new = []
        scanrefer_val_new_scene = []
        scene_id = ""
        for data in scanrefer_val:
            if scene_id != data["scene_id"]:
                scene_id = data["scene_id"]
                if len(scanrefer_val_new_scene) > 0:
                    scanrefer_val_new.append(scanrefer_val_new_scene)
                scanrefer_val_new_scene = []
            if len(scanrefer_val_new_scene) >= lang_num_max:
                scanrefer_val_new.append(scanrefer_val_new_scene)
                scanrefer_val_new_scene = []
            scanrefer_val_new_scene.append(data)
        scanrefer_val_new.append(scanrefer_val_new_scene)

    logger.info("scanrefer_train_new %s %s %s" % ( len(scanrefer_train_new), len(scanrefer_val_new), len(scanrefer_train_new[0])))  # 4819 1253 8
    sum = 0
    for i in range(len(scanrefer_train_new)):
        sum += len(scanrefer_train_new[i])
    logger.info("training sample numbers %s" % sum)  # 36665
    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    logger.info("train on %s samples and val on %s samples" %(len(new_scanrefer_train), len(new_scanrefer_val)))  # 36665 9508

    return new_scanrefer_train, new_scanrefer_val, all_scene_list, scanrefer_train_new, scanrefer_val_new

def get_dataloader(args, scanrefer, scanrefer_new, all_scene_list, split, config, augment, shuffle=True):
    # Init datasets and dataloaders

    if args.worker_seed == 'no_worker_seed':
        def my_worker_init_fn(worker_id):
            pass
    elif args.worker_seed == 'np':
        def my_worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id)
    elif args.worker_seed == 'np_rank':
        def my_worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id + torch.distributed.get_rank() * args.workers)
    
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer[split], 
        scanrefer_new=scanrefer_new[split],
        scanrefer_all_scene=all_scene_list, 
        split=split, 
        num_points=args.num_points, 
        use_height=(not args.no_height),
        use_color=args.use_color, 
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview,
        sent_len_max=args.sent_len_max,
        lang_num_max=args.lang_num_max,
        augment=augment,
        sent_aug=args.sent_aug
    )
    if split == 'train':
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(dataset, 
                                batch_size=args.batch_size, 
                                shuffle=False, 
                                num_workers=args.workers, 
                                sampler=train_sampler, 
                                pin_memory=True,
                                drop_last=True)

    else:
        dataloader = DataLoader(dataset, 
                                batch_size=args.batch_size, 
                                shuffle=False, 
                                num_workers=args.workers, 
                                pin_memory=True,
                                drop_last=False)

    return dataset, dataloader

def get_model(args):
    # initiate model
    model = HAM(
        num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
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
        

        use_color = args.use_color,
        use_normal = args.use_normal,
        use_multiview = args.use_multiview,
        no_height = args.no_height,
        fps_method = args.fps_method,

        use_spa = args.use_spa,
        multi_window=args.multi_window,

    )

    criterion = get_loss
    return model, criterion

def load_checkpoint(args, model, optimizer, scheduler):
    logger.info("=> loading checkpoint '{}'".format(args.checkpoint_path))

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    logger.info("=> loaded successfully '{}' (epoch {})".format(args.checkpoint_path, checkpoint['epoch']))

    del checkpoint
    torch.cuda.empty_cache()

def load_pretrained_model(args, model):
    logger.info("=> loading pretrained model weights {}".format(args.pretrained_model_path))

    model_dict = model.state_dict()

    pretrained_weights = torch.load(args.pretrained_model_path, map_location='cpu')
    pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in model_dict}
    
    model_dict.update(pretrained_weights)
    model.load_state_dict(model_dict)
    del pretrained_weights
    torch.cuda.empty_cache()

def load_groupfree(args, model):
    logger.info("=> loading pretrained weights of groupfree {}".format(args.pretrained_groupfree))

    model_dict = model.state_dict()

    pretrained_weights = torch.load(args.pretrained_groupfree, map_location='cpu')
    pretrained_weights = {k: v for k, v in pretrained_weights['model'].items() if k in model_dict}
    
    model_dict.update(pretrained_weights)
    model.load_state_dict(model_dict)
    del pretrained_weights
    torch.cuda.empty_cache()

def save_checkpoint(args, epoch, model, optimizer, scheduler, save_cur=False):
    logger.info('==> Saving...')
    state = {
        'config': args,
        'save_path': '',
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': None if scheduler is None else scheduler.state_dict(),
        'epoch': epoch,
    }

    if save_cur:
        state['save_path'] = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')
        torch.save(state, os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth'))
        logger.info("Saved in {}".format(os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')))
    elif epoch % args.save_freq == 0:
        state['save_path'] = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')
        torch.save(state, os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth'))
        logger.info("Saved in {}".format(os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')))
    else:
        print("not saving checkpoint")
        pass

def main(args):
    # init training dataset
    logger.info("preparing data...")
    scanrefer_train, scanrefer_val, all_scene_list, scanrefer_train_new, scanrefer_val_new = get_scanrefer(SCANREFER_TRAIN, SCANREFER_VAL, args.num_scenes, args.lang_num_max, args.sent_aug, args.sent_aug_ratio)
    scanrefer = {
        "train": scanrefer_train,
        "val": scanrefer_val
    }


    scanrefer_new = {
        "train": scanrefer_train_new,
        "val": scanrefer_val_new
    }

    # dataloader
    train_dataset, train_loader = get_dataloader(args, scanrefer, scanrefer_new, all_scene_list, "train", DC, True)
    val_dataset, val_loader = get_dataloader(args, scanrefer, scanrefer_new, all_scene_list, "val", DC, False)
    dataloader = {
        "train": train_loader,
        "val": val_loader
    }
    if dist.get_rank() == 0:
        logger.info(f"length of train set {len(train_loader)}")
        logger.info(f"length of val set {len(val_loader)}")
    
    model, criterion = get_model(args)
    

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "decoder" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "decoder" in n and p.requires_grad],
                "lr": args.decoder_lr,
            },
    ]
    optimizer = optim.Adam(param_dicts, lr=args.lr, weight_decay=args.wd)
    scheduler = get_scheduler(optimizer, len(train_loader), args)

    model = model.cuda()
    model = DDP(model, device_ids=[args.local_rank], broadcast_buffers=False, )

    if args.checkpoint_path:
        assert os.path.isfile(args.checkpoint_path)
        load_checkpoint(args, model, optimizer, scheduler)

    if args.pretrained_model_path:
        assert os.path.isfile(args.pretrained_model_path)
        load_pretrained_model(args, model)

    best_result = 0.
    for epoch in range(args.start_epoch, args.max_epoch):
        train_loader.sampler.set_epoch(epoch)

        tic = time.time()

        train_one_epoch(epoch, train_loader, val_loader, model, criterion, optimizer, scheduler, args)
        
        if scheduler is not None:
            scheduler.step()
        
        if (epoch + 1) % args.val_freq == 0:
            if dist.get_rank() == 0:
                results = evaluate_one_epoch(epoch, model, args, logger)
                if results['A50'] > best_result:
                    best_result = results['A50']
                    logger.info('New Best Result: Epoch {}.'.format(epoch))
                    torch.save(model.state_dict(), os.path.join(args.log_dir, "model.pth"))
                    torch.save(results['random_state'], os.path.join(CONF.PATH.OUTPUT, args.folder, "random_state.pth"))

        logger.info('Epoch {}, ETA time {:.2f} hours, lr {:.5f}, de_lr {:.5f}'.format(epoch, ((time.time() - tic) / 3600)*(args.max_epoch-epoch-1), optimizer.param_groups[0]['lr'],  optimizer.param_groups[1]['lr']))

        if dist.get_rank() == 0:
            save_checkpoint(args, epoch, model, optimizer, scheduler)

    return None

def evaluate_one_epoch(epoch, model, args, logger=None):
    
    logger.info('Evaluation on epoch %s' % epoch)

    model.eval()

    stat_dict = {'A25':[], 'A50':[], }

    random_state = predict(args=args, model=model)
    stat_dict['A25'], stat_dict['A50'] =evaluate(args, logger=logger)
    
    stat_dict['random_state'] = random_state

    logger.info("Evaluation Done: A25=%s\t A50=%s" % (stat_dict['A25'],stat_dict['A50']))
    return stat_dict

def train_one_epoch(epoch, train_loader, val_loader, model, criterion, optimizer, scheduler, config):
    stat_dict = {}  # collect statistics
    model.train()  # set model to training mode

    fetch_time = []
    forward_time = []
    backward_time = []
    iter_time = []
    eval_time = []

    fetch_start = time.time()
    for batch_idx, data_dict in enumerate(train_loader):
        for key in data_dict:
            data_dict[key] = data_dict[key].cuda(non_blocking=True)
        fetch_end = time.time()
        fetch_time.append(fetch_end-fetch_start)
        
        forward_start = time.time()

        data_dict = model(data_dict)
        forward_end = time.time()
        forward_time.append(forward_end-forward_start)
        
        backward_start = time.time()
        loss_start = time.time()


        _, data_dict = criterion(
            data_dict=data_dict, 
            config=dataset_config, 
            detection=not config.no_detection,
            reference=not config.no_reference, 
            use_lang_classifier=not config.no_lang_cls,
            num_decoder_layers=args.num_decoder_layers,
            detection_weight=args.detection_weight,
            sent_aug=args.sent_aug
        )
        loss_end = time.time()
        loss = data_dict['loss']

        optimizer.zero_grad()
        loss.backward()
        if config.clip_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
        optimizer.step()

        stat_dict['grad_norm'] = grad_total_norm
        for key in data_dict:
            if ('loss' in key and 'head_' not in key) or 'acc' in key or 'rate' in key:
                if key not in stat_dict: stat_dict[key] = 0
                if isinstance(data_dict[key], float):
                    stat_dict[key] += data_dict[key]
                else:    
                    stat_dict[key] += data_dict[key].item()

        backward_end = time.time()
        backward_time.append(backward_end-backward_start)

        if (batch_idx + 1) % config.print_freq == 0:
            
            mean_train_time = np.mean(fetch_time) + np.mean(forward_time) + np.mean(backward_time)

            eta_sec = mean_train_time * (len(train_loader) - batch_idx - 1 + len(train_loader) * (args.max_epoch - epoch - 1))
            eta = decode_eta(eta_sec)
            
            logger.info(f'\nTrain: [{epoch}][{batch_idx + 1}/{len(train_loader)}]  \n' + ''.join(
                [f'{key} {stat_dict[key] / config.print_freq:.4f} \n' for key in stat_dict.keys()]) + f'ETA: %sh %sm %ss \n'% (eta['h'], eta['m'], eta['s']))

            for key in sorted(stat_dict.keys()):
                stat_dict[key] = 0

        fetch_start = time.time()
    return None        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
    parser.add_argument("--local_rank", default=-1, type=int, help="For DDP Training")
    parser.add_argument("--batch_size", type=int, help="batch size", default=4)
    parser.add_argument("--workers", type=int, help="num of workers", default=8)
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to run [default: 1]')
    parser.add_argument("--max_epoch", type=int, help="number of epochs", default=150)
    parser.add_argument("--print_freq", type=int, help="iterations of showing verbose", default=50)
    parser.add_argument("--val_freq", type=int, help="epochs of validating", default=1)
    parser.add_argument("--save_freq", type=int, help="epochs of save checkpoint", default=20)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-2)
    parser.add_argument("--decoder_lr", type=float, default=5e-4)
    parser.add_argument("--lr_scheduler", type=str, default='step', help="learning rate scheduler")
    parser.add_argument('--lr_decay_epochs', type=int, default=[120], nargs='+', help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--clip_norm', default=100., type=float, help='gradient clipping max norm')
    parser.add_argument('--warmup-epoch', type=int, default=-1, help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)
    parser.add_argument("--num_points", type=int, default=50000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=512, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_augment", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_detection", action="store_true", help="Do NOT train the detection module.")
    parser.add_argument("--no_reference", action="store_true", help="Do NOT train the localization module.")
    parser.add_argument("--termcolor", action="store_true", help="Use color in terminal output")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--checkpoint_path", type=str, help="Specify the checkpoint root", default="")
    parser.add_argument("--pretrained_model_path", type=str, help="Specify the checkpoint root", default="")
    parser.add_argument("--pred_split", type=str, choices=['val', 'test'], default='val')

    parser.add_argument("--backbone_width", type=int, default=2)
    parser.add_argument("--num_decoder_layers", type=int, default=12)
    parser.add_argument("--detection_weight", type=float, help="the weight of detection loss.", default=16.)
    parser.add_argument("--fuse_with_query", action='store_true')
    parser.add_argument("--fuse_with_key", action='store_true')
    parser.add_argument("--multi_window", nargs='+', type=int, default=[4])
    parser.add_argument("--use_spa", action='store_true')
    parser.add_argument("--worker_seed", type=str, default='no_worker_seed')
    parser.add_argument("--fps_method", type=str, choices=['F-FPS', 'D-FPS', 'D-F-FPS', 'CS'], default='CS')

    # For Data Augmentation
    parser.add_argument("--sent_len_max", type=int, help="max sentence length", default=200)
    parser.add_argument("--lang_num_max", type=int, help="lang num max", default=8)
    parser.add_argument("--sent_aug", action="store_true")
    parser.add_argument("--sent_aug_ratio", type=float, default=1.)
    parser.add_argument("--sent_aug_num", nargs='+', type=int, default=[2, 6])

    args = parser.parse_args()
    args.seed += args.local_rank


    # reproducibility
    seed_all(args.seed)

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    dataset_config = ScannetDatasetConfig()

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    LOG_DIR = os.path.join(CONF.PATH.OUTPUT, '%s_%s'%(args.tag, stamp))

    args.log_dir = LOG_DIR
    args.folder = LOG_DIR
    os.makedirs(args.log_dir, exist_ok=True)

    logger = setup_logger(output=args.log_dir, color=args.termcolor, distributed_rank=dist.get_rank(), name='HAM')

    if dist.get_rank() == 0:
        path = os.path.join(args.log_dir, 'config.json')
        with open(path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))
        logger.info(str(vars(args)))
    
    main(args)
    

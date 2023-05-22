# HAM
This is the code for the paper "HAM: Hierarchical Attention Model with High Performance for 3D Visual Grounding". 

Our method is the 1st method on the ScanRefer benchmark (2022/10 - 2023/3) and is the winner of the ECCV 2022 2nd Workshop on Language for 3D Scenes.

## Data preparation
1. Download the [ScanRefer](https://github.com/daveredrum/ScanRefer) dataset and unzip it under `data/`.
2. Downloadand the preprocessed [GLoVE](http://kaldir.vc.in.tum.de/glove.p) embeddings and put them under `data/`.
3. Download the [ScanNetV2](https://github.com/ScanNet/ScanNet) dataset and put `scans/` under `data/scannet/scans/`.
4. Pre-process ScanNet data.
   ```
   cd data/scannet
   python batch_load_scannet_data.py
   ```

## Setup
```
pip install -r requirements.txt

cd lib/pointnet2
python setup.py install
```

Set the correct project path in `lib/config.py`.

## Quick Start
### Training
Using `--tag` to name your experiment, and the training snapshots and results will be put in `outputs/TAG_NAME_[timestamp]`
```
CUDA_VISIABLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
           --master_port 19999 --nproc_per_node 8 ./scripts/train_dist.py \
           --fuse_with_key  --use_spa  --sent_aug  \
           --use_color --use_normal \
           --tag TAG_NAME
```

### Evaluation
```
CUDA_VISIABLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch 
           --master_port 19998 --nproc_per_node 1 ./benchmark/predict.py  
           --fuse_with_key  --use_spa --use_color --use_normal 
           --no_nms --pred_split val --folder TAG_NAME
```

## Acknowledgements

We thank a lot for the codebases of [ScanRefer](https://github.com/daveredrum/ScanRefer),  [3DVG-Transformer](https://github.com/zlccccc/3DVG-Transformer),  [GroupFree](https://github.com/zeliu98/Group-Free-3D).


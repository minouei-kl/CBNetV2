#!/bin/bash
pip uninstall pycocotools -y
pip install mmpycocotools
pip install mmcv-full==1.3.14
pip install git+file:///netscratch/minouei/pubtabnet/src/CBNetV2
python -u tools/train.py configs/0pubtab/htc_cbv2_swin_large_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_pubtab.py  --launcher=slurm

# Repository for 2D Object Detection in mmdet Framework
# README will be updated SOON!!

## Hanhwa - IR Image Object Detection Challenge

# Writing Repository ..

This repository is deep learning template for 2D Object Detection, Using MMdetection
It supports MMdetection v2~ now, but will be upgraded to v3~ 

### Preparation
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v2.28.2
pip install -e .

My Repo
-mmdet (from mm-lab)
-projects
 |-configs
 |-mmdet_plugin
-tools
-data


RTX3060
docker pull pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
apt-get install libgl1-mesa-glx 

apt-get install libglib2.0-0

pip install wandb

for single gpu

tools/dist_train.sh projects/configs/baseline/baseline.py 1 --work-dir work_dirs/baseline/

for multi gpu (DDP)

tools/dist_train.sh projects/configs/baseline/baseline.py 4 --work-dir work_dirs/baseline/
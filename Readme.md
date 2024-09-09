# Repository for 2D Object Detection in mmdet Framework

### This is the repository I am working on.

## Introduction
This repository is deep learning template for 2D Object Detection, Using MMdetection like **Plugin**


It supports **MMdetection v2** now.

but will be upgraded to v3.

이 Repository는 MMdetection Repository에 나의 코드를 plugin 형식으로 붙여 간편하게 사용할 수 있도록 하기 위해 제작되었습니다.

기존 MMdetection을 사용하기 위해서는 MMdetection을 clone받아, mmdet 폴더 내부에 기존에 이미 존재하는 수많은 모델을 사이에 나의 모델을 구현하고, 수많은 데이터셋 코드 사이에 나의 데이터셋 코드를 구현해야하는 불편함이 있었지만, plugin형식으로 사용한다면 프로젝트 관리와 가독성이 좋아지는 효과를 가질 수 있습니다. 

또한, MMdetection을 입문 시, 도대체 뭐가 뭔지 모르겠는 상황이 생기는데 Plugin을 통해 자신의 모델을 구현하면서 감을 잡는데 많은 도움을 줄 수 있습니다.

사실 그냥 쓰기 편리해서 만들었습니다.

사용법은 코드를 보면 이해할 수도 있지만, 천천히 정리해서 올리겠습니다.

### Preparation

Now, This Repository support only single GPU Training.

Multi GPU Training will be supported.

지금 구현된 코드는 single GPU만 Training이 가능합니다.

multi GPU Training은 추후 업데이트 예정입니다.
* * *
I recommend using **Docker** to set up the environment.

Environment를 구성하기 위해선 도커를 사용하는 것을 추천드립니다.

Using Docker
-------------
1. Check if your GPU and the CUDA version it supports are compatible with mmcv-full version 1.6.0.
   
    자신의 GPU와 그 GPU가 지원하는 cuda 버전이 mmcv-full 1.6.0 버전에 존재하는지 확인합니다.

    you can find in here -> https://mmcv.readthedocs.io/en/v1.6.0/get_started/installation.html

    저의 경우 RTX3060을 사용하고 있고 11.3버전 이상의 cuda 버전이 지원되기 때문에 cuda 11.3에 torch1.11을 지원하는 mmcv-full 1.6.0 버전이 있다는 것을 확인했습니다.

2. Get Image from Docker Hub pytorch and Start Container

    Docker Hub에 들어가서 pytorch/pytorch에서 지원하는 위에서 찾은 이미지를 pull하여 start 합니다.

3. git clone this repo and install librarys
   
    ```
    apt-get install libgl1-mesa-glx 

    apt-get install libglib2.0-0

    pip install wandb

    #I assume you are working in ~/ws, but the workspace path can be set freely.

    cd ~/ws     

    git clone https://github.com/Shinsungjun/MMdet-ObjectDetection-Template.git

    # You need to modify the content inside the brackets {} to match your installed versions of PyTorch and CUDA. (That's why we checked if the version exists in mmcv-full 1.6.0 earlier.)
    
    pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/{cu113}/{torch1.11.0}/index.html
    
    pip install yapf==0.40.1

    cd MMdet-ObjectDetection-Template

    git clone https://github.com/open-mmlab/mmdetection.git

    cd mmdetection

    git checkout v2.28.2

    pip install -e .

    cd ~/ws/MMdet-ObjectDetection-Template

    mkdir data #Data for training can be linked in it
    ```

Using Conda
-------------


The first step and third step(Above) remain the same, but the second step(Above) should be done in Conda.

위의 과정 중 1, 3번은 동일하지만, 2번의 과정을 Conda에서 진행하면 됩니다.
  
Repository Directories Form
-------
MMdet-ObjectDetection-Template  
├── data  
├── debugging  
├── docs  
├── mmdetction (git clone from mm-lab)  
├── projects  
│   ├── configs  
│   └── mmdet_plugin  
└── tools

Training & Inference
---------
**for single gpu**


training

```
tools/single_train.sh projects/configs/baseline/baseline.py --work-dir work_dirs/baseline/
```
inference (test)
```
tools/single_test.sh projects/configs/baseline/baseline.py work_dirs/baseline/latest.pth --eval bbox 
```

args는 tools/train.py 나 tools/test.py를 보고 원하는 값을 넣으세요.

~~for multi gpu (DDP)~~ **Not Supported Yet**

tools/dist_train.sh projects/configs/baseline/baseline.py --work-dir work_dirs/baseline/
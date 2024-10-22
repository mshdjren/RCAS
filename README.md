# Target-class-Unlearning
Official Pytorch implementation of [Target Class Unlearning for Conditional Generative Adversarial Networks] (IPIU 2023).

Authors: Sanghyuk Moon, Je Hyeong Hong

<img src=https://github.com/mshdjren/Target-class-Unlearning/blob/main/assets/main_poster.png height="500" width="900"> 
<img src=https://github.com/mshdjren/Target-class-Unlearning/blob/main/assets/result_poster.png>

## Abstract
Machine unlearning 은 사전 학습된 네트워크에서 특정 데이터 또는 클래스에 대해 학습된 정보를 지우는 기법이다. 선행 연구들은 분류 기반 네트워크에서 정보 망각을 위한 기법에 집중하였지만, 이미지의 분포를 학습함에 따라 가상 이미지를 생성할 수 있어 사생활 보호 (privacy preserving) 측면에서 중요하게 고려되어야 할 적대적 생성 신경망 (generative adversarial networks, GAN) 기반의 망각 연구는 많이 수행되지 않았다. 본 논문은 GAN 에 machine unlearning 기법을 적용하는 초기 연구로써, 사전학습된 조건부 적대적 신경망 (conditional generative adversarial networks, CGAN)의 특정 클래스 정보를망각하기 위한 두가지 미세 조정 학습 방식을 제안한다. 첫째, 기존 데이터셋에서 망각할 클래스의 데이터셋을 제거한 후 미세 조정한다. 둘째, 첫번째 방법과 동일하되 기존의 클래스 정보를 망각할 클래스 정보로 대체한 데이터셋을 미세 조정한다. 미세 조정한 CGAN 을 통해 생성한 이미지를 다양한 관점 및 기법으로 분석하여 GAN 에서의 학습된 정보를 망각하는 machine unlearning 기법 적용 가능성을 제시한다.

## Requirements:
$ git clone https://github.com/eriklindernoren/PyTorch-GAN

$ cd PyTorch-GAN/

$ sudo pip3 install -r requirements.txt

````
torch>=0.4.0
torchvision
matplotlib
numpy
scipy
pillow
urllib3
scikit-image
````
This code has been tested with Ubuntu 20.04, A100 GPUs with CUDA 12.2, Python 3.8, Pytorch 1.10.

## How to run our code
Our code built upon the repository of PyTorch-GAN.

We borrowed most of the implementation of conditional generation framework from PyTorch-GAN repository.

Thanks to eriklindernoren for your contribution.

- **Training (Target-class unleraning for specific class)**
````
$ cd implementations/acgan/
$ python3 acgan.py
````

- **Testing (generating forgetting/remaining classes images)**
````
$ cd implementations/acgan/
$ python3 acgan.py
````

## Citation

## License
A patent application for XMP has been submitted and is under review for registration. XMP is licensed under the CC-BY-NC-SA-4.0 license limiting any commercial use.

## Acknowledgement
본 연구는 삼성전자 종합기술원 (Samgsung Advanced Institute of Technology)의 지원과 2023년도 정부 (과학기술정보통신부)의 재원으로 정보통신기획평과원의지원(No.2020-0-01373, 인공지능대학원지원(한양대학교))을 받아 수행된 연구임.

# DNML-HGG

## 1. About
This repository contains the implementation code of DNML-HGG.

## 2. Environment
- CPU: AMD EPYC 7452 (32 core) 2.35GHz
- OS: Ubuntu 18.04 LTS
- Memory: 512GB
- GPU: GeForce RTX 3090
- python: 3.6.9. with Anaconda.
The implementation assumes the availability of CUDA device.

## 3. How to Run
### Artificial Dataset
1. Run datasets.py to generate artificial datasets.

2. Execute `python experiment_lvm.py X Y`, where X \in {4, 8, 16} is the true dimensionality (HGG-X), and Y is the CUDA device in which the program runs.

### Real-world Dataset

1. Download the dataset from the URLs below. Then, put the txt files in `dataset/ca-AstroPh`, `dataset/ca-CondMat`, `dataset/ca-GrQc`, and `dataset/ca-HepPh`.
- AstroPh: https://snap.stanford.edu/data/ca-AstroPh.html
- CondMat: https://snap.stanford.edu/data/ca-CondMat.html
- GrQc: https://snap.stanford.edu/data/ca-GrQc.html
- HepPh: https://snap.stanford.edu/data/ca-HepPh.html

2. Execute `python experiment_realworld.py X Y Z`, where X \in {0, 1, 2, 3} is the id of the dataset (i.e, 0: AstroPh, 1:HepPh, 2: CondMat, and 3: GrQc), Y \in {2, 4, 8, 16, 32, 64} is the model dimensionality, and Z is the CUDA device in which the program runs. The combinations of X and Y are taken to be {0, 1, 2, 3}×{2, 4, 8, 16, 32, 64}.

3. run `MinGE.py`

### Metrics.py

1. Run `calc_metric.py`. For artificial dataset, selected dimensionality and benefit are shown in command line. For real-world datasets , selected dimensionality and AUC are shown in command line. At the same time, the figures of each criterion are generated in `results`.

## 4. Author & Mail address
Currently anonymous.

## 5. Requirements & License
### Requirements
- torch==1.8.1
- numpy
- scipy
- pandas
- matplotlib

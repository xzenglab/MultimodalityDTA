# Mdality-DTA: Multimodality fusion strategy for advanced drug–target affinity prediction

This is the implementation for the paper Modality-DTA: Multimodality fusion strategy for advanced drug–target affinity prediction

![Image](https://github.com/xzenglab/MultimodalityDTA/blob/main/image/overview.jpg)



# Dependecies

* Python>=3.6
* pythorch==1.9.0
* Tensorflow>=1.9.0
* sklearn>=0.24.2
* numpy>=1.19.5
* pandas>=1.1.5
# Data download

```
mkdir data
```
 [dataset](https://drive.google.com/drive/folders/1ViullcWrpfgSf1Uv7-nMhgCuHVS-xhGz?usp=sharing) download data

# Run

```
python main.py --lsd-dim 3000 --lst-dim 3000 --epochs_train 100 --lamb 1 --training_epochs 100
```


# Mdality-DTA: Multimodality fusion strategy for advanced drug–target affinity prediction

This is the implementation for the paper Modality-DTA: Multimodality fusion strategy for advanced drug–target affinity prediction

![Image](https://github.com/xzenglab/MultimodalityDTA/blob/main/image/overview.jpg)

# Abstract

**Motivation:** Prediction of the drug–target affinity (DTA) plays an important role in drug discovery. Existing
deep learning methods for DTA prediction typically leverage a single modality of SMILES (simplified
molecular input line entry specification) string or amino acid sequence to learn representations.
Some methods have been proposed to encode SMILES or amino acid sequences into the different
modalities. The prediction accuracy of fusing these multiple modalities is better than that of using a
single modality.

**Results:** We propose Modality-DTA, a novel deep learning method for DTA prediction that leverages multi-modalities of drugs and targets. A group of backward propagation neural networks (each for one
type of modality) is applied to ensure the completeness of the reconstruction process from the latent
feature representation to original multimodality data. Meanwhile, the tag between drug and target is
used to alleviate the noise information in latent representation from multimodality data. Experiments on
three benchmark data sets reveal that our modality-DTA outperforms existing state-of-the-art methods
in all metrics. Modality-DTA reduces the mean square error by over 10% and improves AUPR score by
more than 20%. We further find that the drug modality Morgan fingerprint and the target modality generated by one-hot-encoding play the most significant role. To the best of our knowledge, modality-DTA
is the first method to explore multi-modalities for DAT prediction.

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
down [ dataset ] { https://drive.google.com/drive/folders/1ViullcWrpfgSf1Uv7-nMhgCuHVS-xhGz?usp=sharing } and zip it in data

# Run

```
python main.py
```

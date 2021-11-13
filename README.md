# Mdality-DTA: Multimodality fusion strategy for advanced drug–target affinity prediction

This is the implementation for the paper Modality-DTA: Multimodality fusion strategy for advanced drug–target affinity prediction




# Dependecies

* Python>=3.6
* pythorch==1.9.0
* Tensorflow>=1.9.0
* sklearn>=0.24.2
* numpy>=1.19.5
* pandas>=1.1.5

# Data 
```
mkdir data
```

Please download dataset and tar them in data.  Code blocks for different dataset are defined in util/util.py 


# Run

```
python main.py --lsd-dim 3000 --lst-dim 3000 --epochs_train 100 --lamb 1 --training_epochs 100
```
# Result
```
Validation at Epoch 78 , MSE: 0.21899 , Pearson Correlation: 0.85297 with p-value: 0.0 , Concordance Index: 0.90382
```




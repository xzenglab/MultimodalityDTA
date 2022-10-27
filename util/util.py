import scipy.io as sio
import numpy as np
import math
from numpy.random import shuffle
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
class DataSet(object):

    def __init__(self, data, view_number, labels,classes):
        """
        Construct a DataSet.
        """
        self.data = dict()
        self._num_examples = data[0].shape[0]
        self._labels = labels
        self._classes = classes
        for v_num in range(view_number):
            self.data[str(v_num)] = data[v_num]

    @property
    def labels(self):
        return self._labels
    @property
    def classes(self):
        return self._classes

    @property
    def num_examples(self):
        return self._num_examples

def Normalize(data):
    """
    :param data:Input data
    :return:normalized data
    """
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return (data - m) / (mx - mn)

def scaler(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data)

def convert_y_unit(y, from_, to_):
    array_flag = False
    if isinstance(y, (int, float)):
        y = np.array([y])
        array_flag = True
    y = y.astype(float)    
    # basis as nM
    if from_ == 'nM':
        y = y
    elif from_ == 'p':
        y = 10**(-y) / 1e-9

    if to_ == 'p':
        zero_idxs = np.where(y == 0.)[0]
        y[zero_idxs] = 1e-10
        y = -np.log10(y*1e-9)
    elif to_ == 'nM':
        y = y
        
    if array_flag:
        return y[0]
    return y



def read_data():
    #ratio=0.8

### data load for Metz dataset

    #df = pd.read_csv('./data/metz/data.csv')
    #d, t, l = df['smiles'], df['sequence'], df['label']
    #X_drug, X_target, y  = np.array(d),np.array(t),np.array(l)
    #drug_encoding = np.load('./data/metz/drug_encoding.npy',allow_pickle=True).item()
    #target_encoding =np.load('./data/metz/target_encoding.npy',allow_pickle=True).item()
    #labels=list(y)
    #drug_class=np.load('./data/metz/drug_class.npy').tolist()
    #target_class=np.load('./data/metz/target_class.npy').tolist() 

### data load for KIBA dataset


    #X_drug=np.load('./data/davis/SMILES.npy')
    #X_target=np.load('./data/davis/Target_seq.npy')
    #y=np.load('./data/davis/y.npy')
    #drug_encoding = np.load('./data/kiba/drug_encoding.npy',allow_pickle=True).item()
    #target_encoding =np.load('./data/kiba/target_encoding.npy',allow_pickle=True).item()
    #labels=list(y)
    #target_class=np.load('./data/kiba/target_class.npy').tolist()
    #drug_class=np.load('./data/kiba/drug_class.npy').tolist()

### data load for Davis dataset


    X_drug=np.load('./data/davis/SMILES.npy')
    X_target=np.load('./data/davis/Target_seq.npy')
    y=np.load('./data/davis/y.npy')
    y=convert_y_unit(np.array(y), 'nM', 'p')
    drug_encoding = np.load('./data/davis/drug_encoding.npy',allow_pickle=True).item()
    target_encoding =np.load('./data/davis/target_encoding.npy',allow_pickle=True).item()
    labels=list(y)
    target_class=np.load('./data/davis/target_class.npy').tolist()
    drug_class=np.load('./data/davis/drug_class.npy').tolist()

    modalitytarget=['AAC', 'PseudoAAC', 'Conjoint_triad', 'Quasi-seq', 'ESPF', 'CNN']
    modalitydrug=['Morgan', 'Pubchem', 'Daylight', 'rdkit_2d_normalized', 'ESPF', 'CNN' ]
    #modalitytarget=['AAC', 'PseudoAAC']
    #modalitydrug=['Morgan', 'Pubchem']




    drug={}
    target={}


    for i in modalitydrug:
        d=[]
        for j in X_drug:
            d.append(drug_encoding[i][j])
        drug[i]=d

    for i in modalitytarget:
        t=[]
        for j in X_target:
            t.append(target_encoding[i][j])
        target[i]=t



    view_number_drug=len(list(drug.keys()))
    view_number_target=len(list(target.keys()))

    
    ratio=0.7   
    trainlen=int(ratio*len(labels))
    index = np.array([x for x in range(len(labels))])
    shuffle(index)
    index_train=index[:trainlen]
    index_test=index[trainlen:]
    
    
    drug_train=[]
    drug_test=[]
    for key,value in drug.items():
        drug_train.append(np.array(value)[index_train])
        drug_test.append(np.array(value)[index_test])
    

    label_train=[np.array(labels)[index_train]]
    label_test=[np.array(labels)[index_test]]

    Normal=1
    if (Normal == 1):
        for v_num in range(view_number_drug):
            drug_train[v_num] = scaler(drug_train[v_num])
            drug_test[v_num] = scaler(drug_test[v_num])

    traindrug = DataSet(drug_train, view_number_drug, np.array(label_train) ,np.array(drug_class)[index_train])
    testdrug = DataSet(drug_test, view_number_drug, np.array(label_test),np.array(drug_class)[index_test])

    target_train=[]
    target_test=[]
    for key,value in target.items():
        target_train.append(np.array(value)[index_train])
        target_test.append(np.array(value)[index_test])
    

    #label_train=[np.array(labels[:trainlen])]
    #label_test=[np.array(labels[trainlen:])]

    Normal=1
    if (Normal == 1):
        for v_num in range(view_number_target):
            target_train[v_num] = scaler(target_train[v_num])
            target_test[v_num] = scaler(target_test[v_num])

    traintarget = DataSet(target_train, view_number_target, np.array(label_train),np.array(target_class)[index_train])
    testtarget = DataSet(target_test, view_number_target, np.array(label_test),np.array(target_class)[index_test])
    
    return traindrug,testdrug,traintarget,testtarget,view_number_drug,view_number_target 

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

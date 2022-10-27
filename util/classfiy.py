import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, f1_score
import tensorflow as tf
from numpy.random import shuffle
from lifelines.utils import concordance_index

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import SequentialSampler
from torch import nn 

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, f1_score
from lifelines.utils import concordance_index
from scipy.stats import pearsonr
import pickle 
torch.manual_seed(2)    # reproducible torch:2 np:3
np.random.seed(3)
import copy
import os







class Classifier(nn.Sequential):
	def __init__(self,**config):
		super(Classifier, self).__init__()
		self.input_dim=config['input_dim']

		self.dropout = nn.Dropout(0.1)

		self.hidden_dims = config['hidden_dim']
		layer_size = len(self.hidden_dims) + 1
		dims = [self.input_dim] + self.hidden_dims + [1]
		
		self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])

	def forward(self, v_P):
		v_f = v_P
		for i, l in enumerate(self.predictor):
			if i==(len(self.predictor)-1):
				v_f = l(v_f)
			else:
				v_f = F.relu(self.dropout(l(v_f)))
		return v_f
    



class data_process_loader(data.Dataset):

	def __init__(self, list_IDs, labels, df):
		'Initialization'
		self.labels = labels
		self.list_IDs = list_IDs
		self.df = df

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.list_IDs)

	def __getitem__(self, index):
		'Generates one sample of data'
		#print(index)
		index =index
		#print(index)
		x = self.df[index]      
		y = self.labels[index]
		return x, y
def model_initialize(**config):
	model = EVA(**config)
	return model

class EVA:
    def __init__(self,**config):
        self.model=Classifier(**config)
        self.device=torch.device('cpu')
    def test_(self, data_generator, model):
        y_pred = []
        y_label = []
        model.eval()
        encoding=[]
        l=[]
        for i, (x, label) in enumerate(data_generator):
            x = x.float().to(self.device)
            score = self.model(x)
            logits = torch.squeeze(score).detach().cpu().numpy()
            label=torch.tensor([float(i) for  i in label])
            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()
            outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
        model.train()
        return mean_squared_error(y_label, y_pred), pearsonr(y_label, y_pred)[0], pearsonr(y_label, y_pred)[1], concordance_index(y_label, y_pred), y_pred
            
            
            #10820 43280 10820 43280 43280
        
    def evaluation(self, H_train,H_test,H_val,train_label,test_label,val_label,train_epochs,lr):
        params = {'batch_size': 64,
	    		'shuffle': True,
	    		'num_workers': 8,
	    		'drop_last': False}
        index_train=[i for i in range(len(H_train))]
        index_test=[i for i in range(len(H_test))]
        index_val=[i for i in range(len(H_val))]
        #print(len(H_train),len(H_test),len(index_train),len(index_test),len(train_label))
        training_generator = data.DataLoader(data_process_loader(index_train,train_label, H_train), **params)
        testting_generator = data.DataLoader(data_process_loader(index_test, test_label, H_test ), **params)
        valing_generator = data.DataLoader(data_process_loader(index_val, val_label, H_val ), **params)
        self.model = self.model.to(self.device)
        max_MSE = 10000
        valid_metric_record = []
        valid_metric_header = ["# epoch"] 
        t_start = time()
        loss_history=[]
        opt = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay = 0)
        e=0
        for epo in range(train_epochs):
            
            for i, (x, label) in enumerate(training_generator):
                

                x = x.float().to(self.device)                

                score = self.model(x)

                label=[float(i) for  i in label]
                label = Variable(torch.from_numpy(np.array(label)).float()).to(self.device)


                loss_fct = torch.nn.MSELoss()
                n = torch.squeeze(score, 1)
                loss = loss_fct(n, label)
                loss_history.append(loss.item())

                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.set_grad_enabled(False):
                mse, r2, p_val, CI, logits = self.test_(testting_generator, self.model)
                if mse < max_MSE:
                    model_max = copy.deepcopy(self.model)
                    self.model=model_max
                    max_MSE = mse
                    e=epo
                    print('Best mse')
                    print('Validation at Epoch '+ str(epo + 1) + ' , MSE: ' + str(mse)[:7] + ' , Pearson Correlation: '\
						 + str(r2)[:7] + ' with p-value: ' + str(p_val)[:7] +' , Concordance Index: '+str(CI)[:7])
                print('Validation at Epoch '+ str(epo + 1) + ' , MSE: ' + str(mse)[:7] + ' , Pearson Correlation: '\
						 + str(r2)[:7] + ' with p-value: ' + str(p_val)[:7] +' , Concordance Index: '+str(CI)[:7])
        mse, r2, p_val, CI, logits = self.test_(valing_generator, self.model)
        #logits=pd.DataFrame(logits).to_csv('pred.csv')
        print('Testing mse '+ str(e + 1) + ' , MSE: ' + str(mse)[:7] + ' , Pearson Correlation: '\
						 + str(r2)[:7] + ' with p-value: ' + str(p_val)[:7] +' , Concordance Index: '+str(CI)[:7])

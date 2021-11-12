import numpy as np
from util.util import read_data
from util.get_sn import get_sn
from util.model import CPMNets
import util.classfiy as classfiy
from sklearn.metrics import accuracy_score
import os
import warnings
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def scaler(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lsd-dim', type=int, default=256,
                        help='dimensionality of the latent space data: drug [default: 128]')
    parser.add_argument('--lst-dim', type=int, default=256,
                        help='dimensionality of the latent space data: target [default: 128]')
    parser.add_argument('--epochs_train', type=int, default=1, metavar='N',
                        help='number of epochs to train [default: 5]')
    parser.add_argument('--epochs-test', type=int, default=1, metavar='N',
                        help='number of epochs to test [default: 10]')
    parser.add_argument('--lamb', type=float, default=10,
                        help='trade off parameter [default: 1]')
    parser.add_argument('--missing-rate', type=float, default=0,
                        help='view missing rate [default: 0]')
    parser.add_argument('--training_epochs', type=int, default=1,
                        help='training epoch of prediction [default: 10]')
    args = parser.parse_args()
    import datetime
    a=datetime.datetime.now()
    print('Begging time:',a)
    # read data
    traindrug,testdrug,traintarget,testtarget,view_number_drug,view_number_target = read_data()
    
    outdim_size_drug = [traindrug.data[str(i)].shape[1] for i in range(view_number_drug)]
    outdim_size_target = [traintarget.data[str(i)].shape[1] for i in range(view_number_target)]
    
    #predict by no fusion
    config={'input_dim':sum(outdim_size_drug)+sum(outdim_size_target),
            'hidden_dim':[  int((sum(outdim_size_drug)+sum(outdim_size_target))/2),int((sum(outdim_size_drug)+sum(outdim_size_target))/4),
            int((sum(outdim_size_drug)+sum(outdim_size_target))/8),
            int((sum(outdim_size_drug)+sum(outdim_size_target))/16),
            2048,1024,512,256,128] 
        }
    models=classfiy.model_initialize(**config)
    lr=0.001
    
    drug_d=traindrug.data
    target_d=traintarget.data
    drug_data=np.hstack((drug_d['0'],drug_d['1']))
    target_data=np.hstack((target_d['0'],target_d['1']))
    for i in np.arange(2,view_number_drug):
        drug_data=np.hstack((drug_data,drug_d[str(i)]))
    for i in np.arange(2,view_number_target):    
        target_data=np.hstack((target_data,target_d[str(i)]))   
    data=np.hstack((drug_data,target_data))

    drug_d=testdrug.data
    target_d=testtarget.data
    drug_data=np.hstack((drug_d['0'],drug_d['1']))
    target_data=np.hstack((target_d['0'],target_d['1']))
    for i in np.arange(2,view_number_drug):
        drug_data=np.hstack((drug_data,drug_d[str(i)]))
    for i in np.arange(2,view_number_target):    
        target_data=np.hstack((target_data,target_d[str(i)]))   
    data1=np.hstack((drug_data,target_data))
    data=np.vstack((data,data1))


    d={}
    d['data']=list(data)
    d['label']=list(np.hstack((traindrug.labels.reshape(-1),testdrug.labels.reshape(-1))))
    df=pd.DataFrame(d)
    
    train,test=train_test_split(df,test_size=0.2, random_state = 5, shuffle=True)
    test,val=train_test_split(test,test_size=0.5, random_state = 5, shuffle=True)    

    
    models.evaluation( list(train.data),list(test.data),list(val.data), 
    list(train.label),list(test.label),list(val.label), args.training_epochs,lr)
       
    
    # set layer size
    layer_size_drug = [[int(args.lsd_dim*1.5), outdim_size_drug[i]] for i in range(view_number_drug)]
    layer_size_target = [[int(args.lst_dim*1.5), outdim_size_target[i]] for i in range(view_number_target)]

    # set parameter
    learning_rate = [0.001, 0.001]
    Sn_drug = get_sn(view_number_drug, traindrug.num_examples+testdrug.num_examples, args.missing_rate)
    Sn_train_drug = Sn_drug[np.arange(traindrug.num_examples)]
    Sn_test_drug = Sn_drug[np.arange(testdrug.num_examples)+traindrug.num_examples]
    
    Sn_target = get_sn(view_number_target, traintarget.num_examples+testtarget.num_examples, args.missing_rate)
    Sn_train_target = Sn_target[np.arange(traintarget.num_examples)]
    Sn_test_target = Sn_target[np.arange(testtarget.num_examples)+traintarget.num_examples]    
  
    

    model = CPMNets(view_number_drug,view_number_target, traindrug.num_examples,testdrug.num_examples,
                    layer_size_drug,layer_size_target, args.lsd_dim,args.lst_dim, learning_rate,
                    args.lamb)
    # train
    print('view number of drug and target:',view_number_drug,view_number_target)
    
    H_traindrug,H_traintarget=model.train(traindrug.data,traintarget.data, Sn_train_drug,Sn_train_target,
                traindrug.labels.reshape(traindrug.num_examples), args.epochs_train,
                traindrug.classes.reshape(traindrug.num_examples),
                traintarget.classes.reshape(traindrug.num_examples))
    

    
    H_traindrug,H_traintarget = model.get_h_train()

    
    H_train=np.hstack((H_traindrug,H_traintarget))
    label_train=list(np.array(traindrug.labels).T)    
    np.save("H_traindrug.npy",H_traindrug)
    np.save("H_traintarget.npy",H_traintarget)  
      
    drugdict=dict()
    targetdict=dict()

    for i in range(len(traindrug.classes)):
     classes=traindrug.classes[i]
     if classes in drugdict.keys():
             drugdict[classes].append(H_traindrug[i])
     else:
             drugdict[classes]=[]
    
    for i in range(len(traintarget.classes)):
      classes=traintarget.classes[i]
      if classes in targetdict.keys():
              targetdict[classes].append(H_traintarget[i])
      else:
              targetdict[classes]=[]
    
    for key,value in drugdict.items():
     drugdict[key]=sum(value)/len(value)

    for key,value in targetdict.items():
        targetdict[key]=sum(value)/len(value)
    
        
    H_testdrug=[]
    for i in testdrug.classes:
        H_testdrug.append(drugdict[i])
    H_testtarget=[]
    for i in testtarget.classes:
        H_testtarget.append(targetdict[i])
    
    H_testdrug=np.array(H_testdrug)
    H_testtarget=np.array(H_testtarget)


    len_test=int(0.5*len(label_train))
    
    H_test=list(H_train)[:len_test]
    label_test=label_train[:len_test]

    len_test=int(0.5*len(label_test))
    
    H_test1=list(H_test)[:len_test]
    label_test1=label_test[:len_test]
    
    H_val=list(H_test)[len_test:]
    label_val=label_test[len_test:]
    
    config={'input_dim':len(H_train[0]),
            'hidden_dim':[256,512,256,128] 
        }
    models=classfiy.model_initialize(**config)
    lr=0.0001
    #models.evaluation( list(train.data),list(test.data),list(val.data), 
    #list(train.label),list(test.label),list(val.label), args.training_epochs,lr)
    
    models.evaluation( list(H_train),list(H_test1),list(H_val), 
                      list(label_train),list(label_test1),list(label_val), args.training_epochs,lr)
    
    
    b=datetime.datetime.now()
    print('Finished, total time =:',b-a,'End time:',b)
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



import util.classfiy as classfiy
import tensorflow as tf
import numpy as np
from numpy.random import shuffle
from util.util import xavier_init
import torch

class CPMNets():
    """build model
    """
    def __init__(self, view_num_drug,view_num_target, trainLen,testLen, layer_size_drug,layer_size_target, lsd_dim=256,lst_dim=256, learning_rate=[0.001, 0.001], lamb=1):
        """
        :param learning_rate:learning rate of network and h
        :param view_num:view number
        :param layer_size:node of each net
        :param lsd_dim:latent space dimensionality drug
        :param lst_dim:latent space dimensionality target
        :param trainLen:training dataset samples
        :param testLen:testing dataset samples
        """
        # initialize parameter
        self.view_num_drug = view_num_drug
        self.view_num_target = view_num_target
        self.layer_size_drug = layer_size_drug
        self.layer_size_target=layer_size_target
        self.lsd_dim = lsd_dim
        self.lst_dim = lst_dim
        self.trainLen = trainLen
        self.testLen = testLen
        self.lamb = lamb
        # initialize latent space data
        self.h_traindrug, self.h_train_updatedrug = self.H_init('traindrug')
        self.h_testdrug, self.h_test_updatedrug = self.H_init('testdrug')
        self.hdrug = tf.concat([self.h_traindrug,self.h_testdrug],axis=0)
        self.h_traintarget, self.h_train_updatetarget = self.H_init('traintarget')
        self.h_testarget, self.h_test_updatetarget = self.H_init('testtarget')
        self.htarget = tf.concat([self.h_traintarget,self.h_testarget],axis=0)
        self.h_index = tf.placeholder(tf.int32, shape=[None, 1], name='h_index')
        self.Len=tf.placeholder(tf.int64)
        self.h_tempdrug = tf.gather_nd(self.hdrug, self.h_index)
        self.h_temptarget = tf.gather_nd(self.htarget, self.h_index)
        # initialize the input data
        self.inputdrug = dict()
        self.inputtarget = dict()
        self.sn_drug = dict()
        self.sn_target = dict()
        for v_num in range(self.view_num_drug):
            self.inputdrug[str(v_num)] = tf.placeholder(tf.float32, shape=[None, self.layer_size_drug[v_num][-1]],
                                                    name='inputdrug' + str(v_num))
            self.sn_drug[str(v_num)] = tf.placeholder(tf.float32, shape=[None, 1], name='sn_drug' + str(v_num))            
        for v_num in range(self.view_num_target):
            self.inputtarget[str(v_num)] = tf.placeholder(tf.float32, shape=[None, self.layer_size_target[v_num][-1]],
                                                    name='inputtarget' + str(v_num))
            self.sn_target[str(v_num)] = tf.placeholder(tf.float32, shape=[None, 1], name='sn_target' + str(v_num))
        # ground truth
        self.gt = tf.placeholder(tf.float32, shape=[None], name='gt')
        self.classes_drug = tf.placeholder(tf.int32, shape=[None], name='classes_drug')
        self.classes_target = tf.placeholder(tf.int32, shape=[None], name='classes_target')
        # bulid the model
        self.train_op, self.loss =self.bulid_model([self.h_train_updatedrug,self.h_train_updatetarget,
                                                    self.h_test_updatedrug,self.h_test_updatetarget],
                                                   learning_rate)
        # open session
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())
        

    
    
    def bulid_model(self, h_update, learning_rate):
        # initialize network
        netdrug = dict()
        nettarget=dict()
        for v_num in range(self.view_num_drug):
            netdrug[str(v_num)] = self.Encoding_net(self.h_tempdrug, v_num,self.layer_size_drug,'drug')
        for v_num in range(self.view_num_target):
            nettarget[str(v_num)] = self.Encoding_net(self.h_temptarget, v_num,self.layer_size_target,'target')
        #print(net,'net')
        # calculate reconstruction loss
        reco_loss = self.reconstruction_loss(netdrug,nettarget)

        # calculate regression loss
        mse_loss = self.regression_loss()
        mse_loss1 = self.regression_loss1()
        #calculate classification loss
        class_loss=self.classification_loss()
        class_loss1=self.classification_loss1()
        #calculate all loss
        all_loss = tf.add(reco_loss, self.lamb * class_loss)
        all_loss=tf.add(all_loss, self.lamb * mse_loss)
        all_loss1=tf.add(reco_loss, self.lamb * class_loss1)
        all_loss1=tf.add(reco_loss, self.lamb * mse_loss)
        # train net operator
        # train the network to minimize reconstruction loss
        train_net_op = tf.train.AdamOptimizer(learning_rate[0]) \
            .minimize(reco_loss, var_list=tf.get_collection('weight'))
        # train the latent space data to minimize reconstruction loss and classification loss
        train_hn_op = tf.train.AdamOptimizer(learning_rate[1]) \
            .minimize(all_loss, var_list=[h_update[0],h_update[1]])
        mse_train= tf.train.AdamOptimizer(learning_rate[0]) \
            .minimize(mse_loss,var_list=tf.get_collection('weight_mse')) 
        mse_hn_op = tf.train.AdamOptimizer(learning_rate[0]) \
            .minimize(mse_loss1,var_list=tf.get_collection('weight_mse1'))
        # adjust the latent space data
        test_hn_op = tf.train.AdamOptimizer(learning_rate[0]) \
            .minimize(reco_loss, var_list=[h_update[2],h_update[3]])
        #adjust the mse

     
        #print(reco_loss,class_loss,all_loss,train_net_op,train_hn_op,adj_hn_op,'loss and op in bulid model')
        return [train_net_op, train_hn_op, mse_train,mse_hn_op,test_hn_op], [reco_loss, mse_loss, all_loss,class_loss,all_loss1,class_loss1,mse_loss1]

    def H_init(self, a):
        with tf.variable_scope('H' + a):
            if a == 'traindrug':
                h = tf.Variable(xavier_init(self.trainLen, self.lsd_dim))
            if a == 'testdrug':
                h = tf.Variable(xavier_init(self.testLen, self.lsd_dim))
                
            elif a == 'traintarget':
                h = tf.Variable(xavier_init(self.trainLen, self.lst_dim))
            elif a == 'testtarget':
                h = tf.Variable(xavier_init(self.testLen, self.lst_dim))                
            h_update = tf.trainable_variables(scope='H' + a)
        #print(h,'h')
        #print(h_update,'h_update')
        return h, h_update

    def Encoding_net(self, h, v,layer_size,a):
        weight = self.initialize_weight(layer_size[v],a)
        #print(weight,'weight in Encoding_net')
        layer = tf.matmul(h, weight['w0']) + weight['b0']
        #print(layer,'layer in Encoding_net0')
        for num in range(1, len(layer_size[v])):
            layer = tf.nn.dropout(tf.matmul(layer, weight['w' + str(num)]) + weight['b' + str(num)], 0.9)
            #print(layer,'middle layer in Encoding_net')
        #print(layer,'finall layer in Encoding_net')
        return layer




    def initialize_weight(self, dims_net,a):
        all_weight = dict()
        with tf.variable_scope('weight'):
            if a=='drug':
                all_weight['w0'] = tf.Variable(xavier_init(self.lsd_dim, dims_net[0]))
            else:
                all_weight['w0'] = tf.Variable(xavier_init(self.lst_dim, dims_net[0]))
            all_weight['b0'] = tf.Variable(tf.zeros([dims_net[0]]))
            tf.add_to_collection("weight", all_weight['w' + str(0)])
            tf.add_to_collection("weight", all_weight['b' + str(0)])
            for num in range(1, len(dims_net)):
                all_weight['w' + str(num)] = tf.Variable(xavier_init(dims_net[num - 1], dims_net[num]))
                all_weight['b' + str(num)] = tf.Variable(tf.zeros([dims_net[num]]))
                tf.add_to_collection("weight", all_weight['w' + str(num)])
                tf.add_to_collection("weight", all_weight['b' + str(num)])
        #print(all_weight,'all_weight')
        return all_weight

    def classification_loss(self):
        
        F_h_h = tf.matmul(self.h_tempdrug, tf.transpose(self.h_tempdrug))
        #print("这是关于使用分类损失的函数：")
        #print(type(F_h_h),tf.shape(F_h_h))
        F_hn_hn = tf.diag_part(F_h_h)
        F_h_h = tf.subtract(F_h_h, tf.matrix_diag(F_hn_hn))
        classes = tf.reduce_max(self.classes_drug) - tf.reduce_min(self.classes_drug) + 1
        label_onehot = tf.one_hot(self.classes_drug - 1, classes)  # gt begin from 1
        label_num = tf.reduce_sum(label_onehot, 0, keep_dims=True)  # should sub 1.Avoid numerical errors
        F_h_h_sum = tf.matmul(F_h_h, label_onehot)
        label_num_broadcast = tf.tile(label_num, [self.trainLen, 1]) - label_onehot
        F_h_h_mean = tf.divide(F_h_h_sum, label_num_broadcast)
        classes_ = tf.cast(tf.argmax(F_h_h_mean, axis=1), tf.int32) + 1  # gt begin from 1
        F_h_h_mean_max = tf.reduce_max(F_h_h_mean, axis=1, keep_dims=False)
        theta = tf.cast(tf.not_equal(self.classes_drug, classes_), tf.float32)
        F_h_hn_mean_ = tf.multiply(F_h_h_mean, label_onehot)
        F_h_hn_mean = tf.reduce_sum(F_h_hn_mean_, axis=1, name='F_h_hn_mean')
        classes_loss_drug=tf.reduce_sum(tf.nn.relu(tf.add(theta, tf.subtract(F_h_h_mean_max, F_h_hn_mean))))
        
        F_h_h = tf.matmul(self.h_temptarget, tf.transpose(self.h_temptarget))
        F_hn_hn = tf.diag_part(F_h_h)
        F_h_h = tf.subtract(F_h_h, tf.matrix_diag(F_hn_hn))
        classes = tf.reduce_max(self.classes_target) - tf.reduce_min(self.classes_target) + 1
        label_onehot = tf.one_hot(self.classes_target - 1, classes)  # gt begin from 1
        label_num = tf.reduce_sum(label_onehot, 0, keep_dims=True)  # should sub 1.Avoid numerical errors
        F_h_h_sum = tf.matmul(F_h_h, label_onehot)
        label_num_broadcast = tf.tile(label_num, [self.trainLen, 1]) - label_onehot
        F_h_h_mean = tf.divide(F_h_h_sum, label_num_broadcast)
        classes_ = tf.cast(tf.argmax(F_h_h_mean, axis=1), tf.int32) + 1  # gt begin from 1
        F_h_h_mean_max = tf.reduce_max(F_h_h_mean, axis=1, keep_dims=False)
        theta = tf.cast(tf.not_equal(self.classes_target, classes_), tf.float32)
        F_h_hn_mean_ = tf.multiply(F_h_h_mean, label_onehot)
        F_h_hn_mean = tf.reduce_sum(F_h_hn_mean_, axis=1, name='F_h_hn_mean')
        classes_loss_target=tf.reduce_sum(tf.nn.relu(tf.add(theta, tf.subtract(F_h_h_mean_max, F_h_hn_mean))))       
        
        
        
        return classes_loss_drug+classes_loss_target

    def classification_loss1(self):
        
        F_h_h = tf.matmul(self.h_tempdrug, tf.transpose(self.h_tempdrug))
        #print("这是关于使用分类损失的函数：")
        #print(type(F_h_h),tf.shape(F_h_h))
        F_hn_hn = tf.diag_part(F_h_h)
        F_h_h = tf.subtract(F_h_h, tf.matrix_diag(F_hn_hn))
        
        classes = tf.reduce_max(self.classes_drug) - tf.reduce_min(self.classes_drug) + 1
        label_onehot = tf.one_hot(self.classes_drug - 1, classes)  # gt begin from 1
        label_num = tf.reduce_sum(label_onehot, 0, keep_dims=True)  # should sub 1.Avoid numerical errors
        F_h_h_sum = tf.matmul(F_h_h, label_onehot)
        label_num_broadcast = tf.tile(label_num, [np.array(self.testLen), 1]) - label_onehot
        F_h_h_mean = tf.divide(F_h_h_sum, label_num_broadcast)
        classes_ = tf.cast(tf.argmax(F_h_h_mean, axis=1), tf.int32) + 1  # gt begin from 1
        F_h_h_mean_max = tf.reduce_max(F_h_h_mean, axis=1, keep_dims=False)
        theta = tf.cast(tf.not_equal(self.classes_drug, classes_), tf.float32)
        F_h_hn_mean_ = tf.multiply(F_h_h_mean, label_onehot)
        F_h_hn_mean = tf.reduce_sum(F_h_hn_mean_, axis=1, name='F_h_hn_mean1')
        classes_loss_drug=tf.reduce_sum(tf.nn.relu(tf.add(theta, tf.subtract(F_h_h_mean_max, F_h_hn_mean))))
        
        F_h_h = tf.matmul(self.h_temptarget, tf.transpose(self.h_temptarget))
        F_hn_hn = tf.diag_part(F_h_h)
        F_h_h = tf.subtract(F_h_h, tf.matrix_diag(F_hn_hn))
        classes = tf.reduce_max(self.classes_target) - tf.reduce_min(self.classes_target) + 1
        label_onehot = tf.one_hot(self.classes_target - 1, classes)  # gt begin from 1
        label_num = tf.reduce_sum(label_onehot, 0, keep_dims=True)  # should sub 1.Avoid numerical errors
        F_h_h_sum = tf.matmul(F_h_h, label_onehot)
        
        label_num_broadcast = tf.tile(label_num, [np.array(self.testLen), 1]) - label_onehot
        F_h_h_mean = tf.divide(F_h_h_sum, label_num_broadcast)
        classes_ = tf.cast(tf.argmax(F_h_h_mean, axis=1), tf.int32) + 1  # gt begin from 1
        F_h_h_mean_max = tf.reduce_max(F_h_h_mean, axis=1, keep_dims=False)
        theta = tf.cast(tf.not_equal(self.classes_target, classes_), tf.float32)
        F_h_hn_mean_ = tf.multiply(F_h_h_mean, label_onehot)
        F_h_hn_mean = tf.reduce_sum(F_h_hn_mean_, axis=1, name='F_h_hn_mean1')
        classes_loss_target=tf.reduce_sum(tf.nn.relu(tf.add(theta, tf.subtract(F_h_h_mean_max, F_h_hn_mean))))       
        
        
        
        return classes_loss_drug+classes_loss_target


    def reconstruction_loss(self, netdrug,nettarget):
        loss = 0
        #print(self.sn,'sn')
        
        for num in range(self.view_num_drug):
            #print(self.view_num,net[str(num)],str(num),'view_num,net[str(num)],str(num)')
            loss = loss + tf.reduce_sum(
                tf.pow(tf.subtract(netdrug[str(num)], self.inputdrug[str(num)])
                       , 2.0)* self.sn_drug[str(num)]
            )
            
            
        for num in range(self.view_num_target):            
            loss = loss+tf.reduce_sum(
                tf.pow(tf.subtract(nettarget[str(num)], self.inputtarget[str(num)])
                       , 2.0)* self.sn_target[str(num)]
            )
        return loss



    def regression_loss(self):
        #修改classification loss为mse只要将self.h_temp的操作改为和标签的mse就可以的。
        #print(self.h_temp.shape)
        pred=self.multilayer_perceptron()
        #pred=tf.reshape(1,-1)
        gt=tf.squeeze(self.gt)
        #outputs = np.asarray([1 if i else 0 for i in (np.asarray(pred) >= 0.5)])
        #bce=tf.losses.BinaryCrossentropy()
        #return  tf.reduce_sum(tf.keras.losses.binary_crossentropy(gt,pred))
        return tf.reduce_sum(
                tf.pow(tf.subtract(gt, pred)
                       , 2.0))
    
    def multilayer_perceptron(self):

       #NetWork parameters
       n_hidden_1 = 512#1st layer num features
       n_hidden_2 = 1024#2nd layer num features
       n_hidden_3 = 512#2nd layer num features
       n_hidden_4 = 256#2nd layer num features
       #print(self.h_tempdrug.shape[1],self.h_temptarget.shape[1])
       #print(self.h_tempdrug,self.h_temptarget)
       n_input =int( self.h_tempdrug.shape[1]+self.h_temptarget.shape[1])
       #print(type(n_input))
       n_classses = 1 
       with tf.variable_scope('weight_mse'):
            weights = {
           'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
           'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
           'h3':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3])),           
           'h4':tf.Variable(tf.random_normal([n_hidden_3,n_hidden_4])),           
           'out':tf.Variable(tf.random_normal([n_input,n_classses]))
            }
            biases = {
           'b1':tf.Variable(tf.random_normal([n_hidden_1])),
           'b2':tf.Variable(tf.random_normal([n_hidden_2])),
           'b3':tf.Variable(tf.random_normal([n_hidden_3])),           
           'b4':tf.Variable(tf.random_normal([n_hidden_4])),           
           'out':tf.Variable(tf.random_normal([n_classses]))
           }
            tf.add_to_collection("weight_mse", weights)
            tf.add_to_collection("weight_mse", biases)
       #v=tf.concat([ self.h_tempdrug,self.h_temptarget],1)
       layer1 = tf.add(tf.matmul(tf.concat([ self.h_tempdrug,self.h_temptarget],1), weights['out']), biases['out'])
       layer1=tf.nn.sigmoid(layer1)
       #layer2 = tf.nn.relu(tf.add(tf.matmul(layer1,weights['h2']),biases['b2']))
       #layer3=tf.nn.relu(tf.add(tf.matmul(layer2,weights['h3']),biases['b3']))
       #layer4=tf.nn.relu(tf.add(tf.matmul(layer3,weights['h4']),biases['b4']))
       
       #pred=tf.add(tf.matmul(layer4,weights['out']),biases['out'])
      

       #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

       return layer1

    def regression_loss1(self):
        #修改classification loss为mse只要将self.h_temp的操作改为和标签的mse就可以的。
        pred=self.multilayer_perceptron1()
        gt=tf.squeeze(self.gt)
        return tf.reduce_sum(
                tf.pow(tf.subtract(gt, pred)
                       , 2.0))


    def multilayer_perceptron1(self):

       #NetWork parameters
       n_hidden_1 = 512#1st layer num features
       n_hidden_2 = 1024#2nd layer num features
       n_hidden_3 = 512#2nd layer num features
       n_hidden_4 = 256#2nd layer num features
       #print(self.h_tempdrug.shape[1],self.h_temptarget.shape[1])
       #print(self.h_tempdrug,self.h_temptarget)
       n_input =int( self.h_tempdrug.shape[1]+self.h_temptarget.shape[1])
       #print(type(n_input))
       n_classses = 1 
       with tf.variable_scope('weight_mse1'):
            weights = {
           'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
           'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
           'h3':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3])),           
           'h4':tf.Variable(tf.random_normal([n_hidden_3,n_hidden_4])),           
           'out':tf.Variable(tf.random_normal([n_hidden_4,n_classses]))
            }
            biases = {
           'b1':tf.Variable(tf.random_normal([n_hidden_1])),
           'b2':tf.Variable(tf.random_normal([n_hidden_2])),
           'b3':tf.Variable(tf.random_normal([n_hidden_3])),           
           'b4':tf.Variable(tf.random_normal([n_hidden_4])),           
           'out':tf.Variable(tf.random_normal([n_classses]))
           }
            tf.add_to_collection("weight_mse1", weights)
            tf.add_to_collection("weight_mse1", biases)
       #v=tf.concat([ self.h_tempdrug,self.h_temptarget],1)
       layer1 = tf.add(tf.matmul(tf.concat([ self.h_tempdrug,self.h_temptarget],1), weights['h1']), biases['b1'])
      
       layer2 = tf.add(tf.matmul(layer1,weights['h2']),biases['b2'])
       layer3=tf.add(tf.matmul(layer2,weights['h3']),biases['b3'])
       layer4=tf.add(tf.matmul(layer3,weights['h4']),biases['b4'])
       
       pred=tf.add(tf.matmul(layer4,weights['out']),biases['out'])
      

       #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

       return pred



    def train(self, datadrug,datatarget, sn_drug,sn_target, gt, epoch, drug_class,target_class,step=[5, 5]):
        global Reconstruction_LOSS
        index = np.array([x for x in range(self.trainLen)])
        shuffle(index)
        gt = gt[index]
        sn_drug = sn_drug[index]
        #sn_target = sn_target[index]
        #drug_class=drug_class[index]
        target_class=target_class[index]
        feed_dict = {self.inputdrug[str(v_num)]: datadrug[str(v_num)][index] for v_num in range(self.view_num_drug)}
        feed_dict.update( {self.inputtarget[str(v_num)]: datatarget[str(v_num)][index] for v_num in range(self.view_num_target)})

        feed_dict.update({self.sn_drug[str(i)]: sn_drug[:, i].reshape(self.trainLen, 1) for i in range(self.view_num_drug)})
        feed_dict.update({self.sn_target[str(i)]: sn_target[:, i].reshape(self.trainLen, 1) for i in range(self.view_num_target)})
        feed_dict.update({self.gt: gt})
        feed_dict.update({self.classes_drug: drug_class})
        feed_dict.update({self.classes_target: target_class})
        feed_dict.update({self.h_index: index.reshape((self.trainLen, 1))})
        feed_dict.update({self.Len: self.trainLen})
        min_mse_loss=float("inf")
        for iter in range(epoch):
            # update the reco network
            for i in range(step[0]):
                _, Reconstruction_LOSS, MSE_LOSS,all_loss,class_loss = self.sess.run(
                    [self.train_op[0], self.loss[0], self.loss[1],self.loss[2],self.loss[3]], feed_dict=feed_dict)
            #print(Reconstruction_LOSS, MSE_LOSS,all_loss,class_loss,'update reco net')
            # update the mse network
            for i in range(step[1]):
                _, Reconstruction_LOSS, MSE_LOSS,all_loss,class_loss = self.sess.run(
                    [self.train_op[2], self.loss[0], self.loss[1],self.loss[2],self.loss[3]], feed_dict=feed_dict)
             #   #print(Reconstruction_LOSS, MSE_LOSS,all_loss,class_loss,'update mse net')
            # update the h
            for i in range(step[1]):
                _, Reconstruction_LOSS, MSE_LOSS,all_loss,class_loss = self.sess.run(
                    [self.train_op[1], self.loss[0], self.loss[1],self.loss[2],self.loss[3]], 
                    feed_dict=feed_dict)
            #print(Reconstruction_LOSS, MSE_LOSS,all_loss,class_loss,'update h')   
                 
                
        #for iter in range(epoch):                
            #for i in range(step[1]):
              #  _, Reconstruction_LOSS, MSE_LOSS,all_loss,class_loss = self.sess.run(
                #    [self.train_op[2], self.loss[0], self.loss[6],self.loss[2],self.loss[3]], feed_dict=feed_dict)
                #print(Reconstruction_LOSS, MSE_LOSS,all_loss,class_loss,'update mse net')
        #for iter in range(epoch):                
            #for i in range(step[1]):
             #   _, Reconstruction_LOSS, MSE_LOSS,all_loss,class_loss = self.sess.run(
             #       [self.train_op[4], self.loss[0], self.loss[6],self.loss[2],self.loss[3]], feed_dict=feed_dict)
                #print(Reconstruction_LOSS, MSE_LOSS,all_loss,class_loss,'update mse net')
            output = "Epoch : {:.0f}  ===> Reconstruction Loss = {:.4f}, MSE Loss = {:.4f} " \
                .format((iter + 1), Reconstruction_LOSS, MSE_LOSS)
            print(output,'class loss:',class_loss,'all_loss:',all_loss)
            if MSE_LOSS<min_mse_loss:
                min_mse_loss=MSE_LOSS
                lsddrug = self.sess.run(self.hdrug)
                lsdtarget = self.sess.run(self.htarget)
        print(min_mse_loss)
        return lsddrug[0:self.trainLen],lsdtarget[0:self.trainLen]
                
                

    def test(self, datadrug,datatarget, sn_drug,sn_target, gt, epoch,drug_class,target_class):
        global Reconstruction_LOSS
        feed_dict = {self.inputdrug[str(v_num)]: datadrug[str(v_num)] for v_num in range(self.view_num_drug)}
        feed_dict.update({self.inputtarget[str(v_num)]: datatarget[str(v_num)] for v_num in range(self.view_num_target)})
    
        feed_dict.update({self.sn_drug[str(i)]: sn_drug[:, i].reshape(self.testLen, 1) for i in range(self.view_num_drug)})
        feed_dict.update({self.sn_target[str(i)]: sn_target[:, i].reshape(self.testLen, 1) for i in range(self.view_num_target)})

        feed_dict.update({self.gt: gt})
        #drug_class=drug_class[np.array([x for x in range(self.testLen)]).reshape(self.testLen, 1) + self.trainLen]
        #target_class=target_class[np.array([x for x in range(self.testLen)]).reshape(self.testLen, 1) + self.trainLen]
        feed_dict.update({self.classes_drug: drug_class})
        feed_dict.update({self.classes_target: target_class})
        feed_dict.update({self.h_index:
                              np.array([x for x in range(self.testLen)]).reshape(self.testLen, 1) + self.trainLen})
        feed_dict.update({self.Len: self.testLen})
        
        max_mse=float("inf")
        for iter in range(epoch*2):
            # update the h
            for i in range(5):
                _, Reconstruction_LOSS, MSE_LOSS,all_loss,class_loss= self.sess.run(
                    [self.train_op[4], self.loss[0], self.loss[1],self.loss[4],self.loss[5]], feed_dict=feed_dict)
                
               # _, Reconstruction_LOSS, MSE_LOSS,all_loss,class_loss= self.sess.run(
                #    [self.train_op[3], self.loss[0], self.loss[6],self.loss[4],self.loss[5]], feed_dict=feed_dict)
                
                                
                
                #print(Reconstruction_LOSS,MSE_LOSS,all_loss,class_loss,'update h')
                #Reconstruction_LOSS, MSE_LOSS= self.sess.run(
                    #[ self.loss[0], self.loss[1]], feed_dict=feed_dict)
                                
                #print(Reconstruction_LOSS,MSE_LOSS)
                
                if MSE_LOSS<max_mse:
                    max_mse=MSE_LOSS
            output = "Epoch : {:.0f}  ===> Reconstruction Loss = {:.4f}, MSE Loss = {:.4f} " \
                .format((iter + 1), Reconstruction_LOSS, MSE_LOSS)
            print(output,'all loss: ',all_loss,'class_loss: ',class_loss)
            #print(output1)
            # update the h
        print('MSE Finally:',max_mse)




    def get_h_train(self):
        lsddrug = self.sess.run(self.hdrug)
        lsdtarget = self.sess.run(self.htarget)
        #print(lsd)
        return lsddrug[0:self.trainLen],lsdtarget[0:self.trainLen]

    def get_h_test(self):
        lsddrug = self.sess.run(self.hdrug)
        lsdtarget = self.sess.run(self.htarget)
        #print(lsd)
        return lsddrug[self.trainLen:],lsdtarget[self.trainLen:]

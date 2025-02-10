# 2024.8.2
# need to run the following lines before running
# CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
import gc
import math
import json
import numpy as np
import pandas as pd
from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
from scipy.stats import bernoulli
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,roc_auc_score,confusion_matrix,roc_curve
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import time
from tqdm import tqdm
import random
import re

from load_data_fusion import data_loader

TRAINING_SET = 'mix'#'balabit', 'sapimouse'
SAMPLE_SELECTION = True
DYNAMIC_AUTH = 7

def data_augmentation(input,cut_points):
    p = 1
    res = input.copy()
    segments = []
    if bernoulli.rvs(p):
        start = [0] + cut_points[:-1]
        for i,cut_point in enumerate(cut_points):
            segments.append(res[start[i]:cut_point,])
        if cut_points[-1] < res.shape[0]:
            segments.append(res[cut_points[-1]:,])
        random.shuffle(segments)
        res = np.concatenate(segments)

    return res

class CustomCallback(keras.callbacks.Callback):
    def __init__(self,x_valid,y_valid,n_test_instances,fold):
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.n_test_instances = n_test_instances
        self.best_res = 1
        self.fold = fold
        #self.best_res = 0
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            '''
            y_pred = self.model.predict(self.x_valid,verbose=0)
            y_valid_pred,y_valid = [],[]
            for i in range(int(len(y_pred)/self.n_test_instances)):
                y_valid_pred.append(round(np.mean(y_pred[i*self.n_test_instances:(i+1)*self.n_test_instances])))
                y_valid.append(self.y_valid[self.n_test_instances*i])
            y_valid = np.array(y_valid).reshape(-1)
            acc,recall,precision = accuracy_score(y_valid,y_valid_pred),recall_score(y_valid,y_valid_pred),precision_score(y_valid,y_valid_pred)
            print('epoch:{}, validation: acc:{}, recall:{}, precision:{}'.format(epoch,acc,recall,precision))
            if acc > self.best_res:
                self.best_res = acc
                self.model.save('./mouse/trained_models/authentication_shen.h5',save_format='h5')
            gc.collect()
            tf.keras.backend.clear_session()
            '''

            #'''
            valid_res = self.model.evaluate(self.x_valid,self.y_valid,verbose=0)
            loss,acc,recall,precision = tuple(valid_res)
            print('epoch:{}, validation: loss:{}, acc:{}, recall:{}, precision:{}'.format(epoch,loss,acc,recall,precision))
            if loss < self.best_res:
                self.best_res = loss
                self.model.save('./mouse/trained_models/authentication_fusion_'+TRAINING_SET+'_{}.h5'.format(self.fold),save_format='h5')
            #'''

class mouse_classifier():
    def __init__(self):
        '''
        gpus = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[-4], True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        '''
        self.data_shape = (256,4)
        self.n_features = 36
        self.n_test_instances = DYNAMIC_AUTH
        self.best_sample = SAMPLE_SELECTION
        self.fold = 0
        self.t_auc,self.t_far,self.t_frr,self.n_instances = [],[],[],[]

        loader = data_loader(self.data_shape)
        self.segments,self.users_list = loader.load()
        self.shuffled_list = np.random.permutation(np.arange(1,131))


    def update_fold(self,fold):
        self.fold = fold

        if TRAINING_SET == 'mix' or TRAINING_SET == 'balabit':
            self.test_users = self.shuffled_list[26*fold:26*(fold+1)]
            self.train_users = []
            for i in self.shuffled_list:
                if not i in self.test_users:
                    self.train_users.append(i)
            self.train_users = np.array(self.train_users)
            

        elif TRAINING_SET == 'sapimouse':
            self.test_users = np.arange(1,121)[24*fold:24*(fold+1)]
            self.test_users = np.concatenate((self.test_users,np.arange(121,131)))
            self.train_users = []
            for i in np.arange(1,121):
                if not i in self.test_users:
                    self.train_users.append(i)
            self.train_users = np.array(self.train_users)
        #elif TRAINING_SET == 'balabit':
        #    self.test_users = np.arange(121,131)[2*fold:2*(fold+1)]
        #    self.test_users = np.concatenate((self.test_users,np.arange(1,121)))
        #    self.train_users = []
        #    for i in np.arange(121,131):
        #        if not i in self.test_users:
        #            self.train_users.append(i)
        #    self.train_users = np.array(self.train_users)

        
        self.train_segments, self.valid_segments, self.test_segments = [],[],[]
        for i in self.train_users:
            t1 = self.users_list.index(i)
            t2 = self.segments[t1]
            self.train_segments.append((t2[0][:int(len(t2[0])*6/10)],t2[1][:int(len(t2[0])*6/10)]))
            self.valid_segments.append((t2[0][int(len(t2[0])*6/10):int(len(t2[0])*8/10)],t2[1][int(len(t2[0])*6/10):int(len(t2[0])*8/10)]))
            self.test_segments.append((t2[0][int(len(t2[0])*8/10):],t2[1][int(len(t2[0])*8/10):]))
        #self.train_segments = [(self.segments[self.users_list.index(i)][0][:int(len(self.segments[self.users_list.index(i)][0])*6/10)],self.segments[self.users_list.index(i)][1][:int(len(self.segments[self.users_list.index(i)][0])*6/10)]) for i in self.train_users]
        #self.valid_segments = [(self.segments[self.users_list.index(i)][0][int(len(self.segments[self.users_list.index(i)][0])*6/10):int(len(self.segments[self.users_list.index(i)][0])*8/10)],self.segments[self.users_list.index(i)][1][int(len(self.segments[self.users_list.index(i)][0])*6/10):int(len(self.segments[self.users_list.index(i)][0])*8/10)]) for i in self.train_users]
        #self.test_segments = [(self.segments[self.users_list.index(i)][0][int(len(self.segments[self.users_list.index(i)][0])*8/10):],self.segments[self.users_list.index(i)][1][int(len(self.segments[self.users_list.index(i)][0])*8/10):]) for i in self.train_users]
        for i in self.test_users:
            t1 = self.users_list.index(i)
            t2 = self.segments[t1]
            self.test_segments.append((t2[0],t2[1]))
        self.test_users_list = np.concatenate((self.train_users,self.test_users)).tolist()

        self.x1,self.x2,self.y,self.users = self.generate_instances(self.train_segments,self.train_users.tolist(),augmentation=True)
        self.x1 = (np.array([i[0] for i in self.x1],dtype=np.float32),np.array([i[1] for i in self.x1],dtype=np.float32))
        self.x2 = (np.array([i[0] for i in self.x2],dtype=np.float32),np.array([i[1] for i in self.x2],dtype=np.float32))
        self.y = np.array(self.y,dtype=np.float32)
        self.users = np.array(self.users,dtype=np.float32)

        p = np.random.permutation(len(self.y))
        self.x1 = (self.x1[0][p],self.x1[1][p])#normalize(self.x1[1][p],axis=0,norm='max')
        self.x2 = (self.x2[0][p],self.x2[1][p])#normalize(self.x2[1][p],axis=0,norm='max')
        self.y = self.y[p]
        self.users = self.users[p]
            
        self.x_train1 = self.x1
        self.x_train2 = self.x2
        self.y_train = self.y.reshape((-1,1))
        print('generated training instances')

        #'''
        self.x1,self.x2,self.y,self.users = self.generate_valid_instances(self.valid_segments,self.train_segments,self.train_users.tolist(),augmentation=False)
        self.x1 = (np.array([i[0] for i in self.x1],dtype=np.float32),np.array([i[1] for i in self.x1],dtype=np.float32))#normalize(np.array([i[1] for i in self.x1],dtype=np.float32),axis=0,norm='max')
        self.x2 = (np.array([i[0] for i in self.x2],dtype=np.float32),np.array([i[1] for i in self.x2],dtype=np.float32))#normalize(np.array([i[1] for i in self.x2],dtype=np.float32),axis=0,norm='max')
        self.y = np.array(self.y,dtype=np.float32)
        self.users = np.array(self.users,dtype=np.float32)

        self.x_valid1 = self.x1
        self.x_valid2 = self.x2
        self.y_valid = self.y.reshape((-1,1))
        self.valid_users = self.users
        print('generated validation instances')
        #'''

        self.nb_filters = 128
        self.depth = 6
        optimizer = Adam(learning_rate= 1e-5)

        
        self.classifier = self.build_classifier()
        self.classifier.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy',tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

    def generate_instances(self,segments,users_list,augmentation=False):
        '''
        segments: segments set for generating instances
        users_list: list of user index corresponding to segments
        user_index: list of user for who generate instances
        augmentation: whether to apply data augmentation
        '''
        x1,x2,y,users = [],[],[],[]
        for user in users_list:
            user_segments,cut_points = segments[users_list.index(user)]
            for i in range(int(len(user_segments)/2)):
                #record from the same user
                if augmentation:
                    x1.append((data_augmentation(user_segments[i][0],cut_points[i]),user_segments[i][1]))
                    x2.append((data_augmentation(user_segments[i+int(len(user_segments)/2)][0],cut_points[i+int(len(user_segments)/2)]),user_segments[i+int(len(user_segments)/2)][1]))
                else:
                    x1.append(user_segments[i])
                    x2.append(user_segments[i+int(len(user_segments)/2)])
                y.append(1)
                users.append((user,user))

                #for k in range(3):
                random_user = np.random.choice(users_list)#add random record from different users to keep the dataset balance
                while random_user == user:
                    random_user = np.random.choice(users_list)
                random_segments,random_cut_points = segments[users_list.index(random_user)]
                random_segment_i = np.random.randint(0,len(random_segments))


                if augmentation:
                    x1.append((data_augmentation(user_segments[i][0],cut_points[i]),user_segments[i][1]))
                    x2.append((data_augmentation(random_segments[random_segment_i][0],random_cut_points[random_segment_i]),random_segments[random_segment_i][1]))
                else:
                    x1.append(user_segments[i])
                    x2.append(random_segments[random_segment_i])
                y.append(0)
                users.append([user,random_user])
        return x1,x2,y,users

    def generate_valid_instances(self,valid_segments,train_segments,users_list,augmentation=False):
        x1,x2,y,users = [],[],[],[]
        for user in users_list:
            user_segments,cut_points = valid_segments[users_list.index(user)]
            same_user_segments,same_cut_points = train_segments[users_list.index(user)]
            for i in range(len(user_segments)):
                #record from the same user
                #for j in range(self.n_test_instances):
                random_segment_i = np.random.randint(0,len(same_user_segments))
                if augmentation:
                    x2.append((data_augmentation(user_segments[i][0],cut_points[i]),user_segments[i][1]))
                    x1.append((data_augmentation(same_user_segments[random_segment_i][0],same_cut_points[random_segment_i]),same_user_segments[random_segment_i][1]))
                else:
                    x2.append(user_segments[i])
                    x1.append(same_user_segments[random_segment_i])
                y.append(1)
                users.append((user,user))

                
                random_user = np.random.choice(users_list)
                while random_user == user:
                    random_user = np.random.choice(users_list)
                random_segments,random_cut_points = valid_segments[users_list.index(random_user)]

                #for j in range(self.n_test_instances):
                random_segment_i = np.random.randint(0,len(random_segments))
                if augmentation:
                    x2.append((data_augmentation(user_segments[i][0],cut_points[i]),user_segments[i][1]))
                    x1.append((data_augmentation(random_segments[random_segment_i][0],random_cut_points[random_segment_i]),random_segments[random_segment_i][1]))
                else:
                    x2.append(user_segments[i])
                    x1.append(random_segments[random_segment_i])
                y.append(0)
                users.append([user,random_user])
        return x1,x2,y,users

    def generate_test_instances(self,test_segments,train_segments,best_samples,augmentation=False):
        x1,x2,y,users = [],[],[],[]
        '''
        for user in range(1,131):
            user_segments,cut_points = test_segments[users_list.index(user)]
            t1,t2 = [],[]
            i = 0
            while i < len(user_segments):
                t1.append(user_segments[i])
                t2.append(cut_points[i])
                i += math.ceil(len(cut_points[i])/2)
            #print(len(t1))
            test_segments[users_list.index(user)] = (t1,t2)
        '''
        for user in self.train_users:
            user_segments,cut_points = test_segments[self.test_users_list.index(user)]
            same_user_segments,same_cut_points = train_segments[self.train_users.tolist().index(user)]
            for i in range(len(user_segments)-self.n_test_instances+1):
                #record from the same user
                random_segment_i = np.random.randint(0,len(same_user_segments)-self.n_test_instances+1)
                for j,sample_index in enumerate(best_samples[user]):
                    if augmentation:
                        x2.append((data_augmentation(user_segments[i+j][0],cut_points[i+j]),user_segments[i+j][1]))
                        x1.append((data_augmentation(same_user_segments[sample_index][0],same_cut_points[sample_index]),same_user_segments[sample_index][1]))
                    else:
                        x2.append(user_segments[i+j])
                        if self.best_sample:
                            x1.append(same_user_segments[sample_index])
                        else:
                            x1.append(same_user_segments[random_segment_i+j])
                    y.append(1)
                    users.append((user,user))

                
                random_user = np.random.choice(self.test_users)
                while random_user == user:
                    random_user = np.random.choice(self.test_users)
                random_segments,random_cut_points = test_segments[self.test_users_list.index(random_user)]
                random_segment_j = np.random.randint(0,len(random_segments)-self.n_test_instances+1)

                for j,sample_index in enumerate(best_samples[user]):
                    if augmentation:
                        x2.append((data_augmentation(random_segments[random_segment_i][0],random_cut_points[random_segment_i]),random_segments[random_segment_i][1]))
                        x1.append((data_augmentation(same_user_segments[sample_index][0],same_cut_points[sample_index]),same_user_segments[sample_index][1]))
                    else:
                        x2.append(random_segments[random_segment_j+j])
                        if self.best_sample:
                            x1.append(same_user_segments[sample_index])
                        else:
                            x1.append(same_user_segments[random_segment_i+j])
                    y.append(0)
                    users.append([user,random_user])
        return x1,x2,y,users


    def training_sample_generalization_test(self,valid_segments,train_segments):
        users_list = self.train_users.tolist()
        best_samples = {i:[] for i in users_list}
        for user in users_list:
            print(users_list.index(user))
            loss_list = []
            user_train_segments,_ = train_segments[users_list.index(user)]
            user_valid_segments,_ = valid_segments[users_list.index(user)]

            user_valid_features = np.array([i[1] for i in user_valid_segments])
            user_valid_segments = np.array([i[0] for i in user_valid_segments])
            p = np.random.permutation(len(user_valid_segments))
            user_valid_segments = user_valid_segments[p]
            user_valid_segments = user_valid_segments[:20]
            user_valid_features = user_valid_features[p]
            user_valid_features = user_valid_features[:20]

            negative_valid_segments = []
            negative_valid_features = []
            for i in range(20):
                random_user = np.random.choice(users_list)
                while random_user == user:
                    random_user = np.random.choice(users_list)
                random_segments,_ = valid_segments[users_list.index(random_user)]
                random_segment_i = np.random.randint(0,len(random_segments))
                negative_valid_segments.append(random_segments[random_segment_i][0])
                negative_valid_features.append(random_segments[random_segment_i][1])

            x2 = (np.concatenate([user_valid_segments,negative_valid_segments]),np.concatenate([user_valid_features,negative_valid_features]))
            y = [1] * 20 + [0] * 20
            y = np.array(y,dtype=np.float32).reshape((-1,1))

            user_train_features = np.array([i[1] for i in user_train_segments])
            user_train_segments = np.array([i[0] for i in user_train_segments])
            p = np.random.permutation(len(user_train_segments))
            user_train_segments = user_train_segments[p]
            user_train_features = user_train_features[p]

            for i in range(user_train_segments.shape[0]):
                if len(loss_list) > 200:
                    break
                x1 = (np.array([user_train_segments[i]] * 40), np.array([user_train_features[i]] * 40))
                test_res = self.classifier.evaluate([x1[0],x2[0]],y,verbose=0)
                loss,acc,recall,precision = tuple(test_res)
                loss_list.append(loss)

            index = np.argpartition(loss_list,np.arange(self.n_test_instances*5))
            j = -1
            while len(best_samples[user]) < self.n_test_instances:
                j += 1
                i = index[j]
                neighbor_segment = False
                for t in best_samples[user]:
                    if abs(int(p[i]) - t) < 5:
                        neighbor_segment = True
                        continue
                if not neighbor_segment:
                    best_samples[user].append(int(p[i]))
            #for i in index[:self.n_test_instances]:
            #    best_samples[user].append(int(p[i]))

        return best_samples
            
            



    def build_classifier(self):
        data1 = keras.Input(shape=self.data_shape)
        data2 = keras.Input(shape=self.data_shape)

        #'''
        conv1 = layers.Conv1D(128, 8, padding='same', name='conv1')
        conv2 = layers.Conv1D(256, 5, padding='same', name='conv2')
        conv3 = layers.Conv1D(128, 3, padding='same', name='conv3')
        lstm1 = layers.LSTM(128, return_sequences=True)
        lstm2 = layers.LSTM(128)
        

        cnn_output1 = conv1(data1)
        cnn_output1 = layers.BatchNormalization()(cnn_output1)
        cnn_output1 = keras.layers.Activation('relu')(cnn_output1)
        cnn_output1 = conv2(cnn_output1)
        cnn_output1 = layers.BatchNormalization()(cnn_output1)
        cnn_output1 = keras.layers.Activation('relu')(cnn_output1)
        cnn_output1 = conv3(cnn_output1)
        cnn_output1 = layers.BatchNormalization()(cnn_output1)
        cnn_output1 = keras.layers.Activation('relu')(cnn_output1)
        cnn_output1 = keras.layers.GlobalAveragePooling1D()(cnn_output1)

        cnn_output2 = conv1(data2)
        cnn_output2 = layers.BatchNormalization()(cnn_output2)
        cnn_output2 = keras.layers.Activation('relu')(cnn_output2)
        cnn_output2 = conv2(cnn_output2)
        cnn_output2 = layers.BatchNormalization()(cnn_output2)
        cnn_output2 = keras.layers.Activation('relu')(cnn_output2)
        cnn_output2 = conv3(cnn_output2)
        cnn_output2 = layers.BatchNormalization()(cnn_output2)
        cnn_output2 = keras.layers.Activation('relu')(cnn_output2)
        cnn_output2 = keras.layers.GlobalAveragePooling1D()(cnn_output2)

        lstm_output1 = lstm1(data1)
        lstm_output1 = lstm2(lstm_output1)

        lstm_output2 = lstm1(data2)
        lstm_output2 = lstm2(lstm_output2)

        combined_input = layers.concatenate([cnn_output1,cnn_output2,lstm_output1,lstm_output2])
        output = layers.Dense(512, activation='relu',name='dense1')(combined_input)
        output = layers.Dropout(0.5)(output)
        output = layers.Dense(512, activation='relu',name='dense2')(output)
        output = layers.Dropout(0.5)(output)
        output = layers.Dense(512, activation='relu',name='dense3')(output)
        output = layers.Dropout(0.5)(output)
        output = layers.Dense(1, activation='sigmoid',name='dense4')(output)
        #'''

        return Model(inputs=[data1,data2],outputs=output)

    def train(self, epochs, batch_size=128):
        p = np.random.permutation(len(self.y_train))#shuffle
        self.x_train1 = (self.x_train1[0][p],self.x_train1[1][p])
        self.x_train2 = (self.x_train2[0][p],self.x_train2[1][p])
        self.y_train = self.y_train[p]

        #'''
        model_path = './mouse/trained_models/authentication_fusion_'+TRAINING_SET+'_{}.h5'.format(self.fold)
        if TRAINING_SET == 'balabit':
            model_path = './mouse/trained_models/authentication_fusion_'+'mix'+'_{}.h5'.format(self.fold)
        if not os.path.exists(model_path):
            start = time.time()
            mycallback = CustomCallback([self.x_valid1[0],self.x_valid2[0],],self.y_valid,self.n_test_instances,self.fold)
            self.classifier.fit([self.x_train1[0],self.x_train2[0]],self.y_train,epochs=epochs,verbose=0,batch_size=batch_size,callbacks = [mycallback])#
            print('training time:{}'.format(time.time()-start))
        self.classifier = keras.models.load_model(model_path)
        #'''
        
        x_train = self.x_train1[1] - self.x_train2[1]
        self.forest_classifier = RandomForestClassifier()
        self.forest_classifier.fit(x_train,self.y_train.reshape(-1,))

        samples_path = './mouse/trained_models/authentication_fusion_'+TRAINING_SET+'_{}.dict'.format(self.fold)
        if TRAINING_SET == 'balabit':
            samples_path = './mouse/trained_models/authentication_fusion_'+'mix'+'_{}.dict'.format(self.fold)
        if not os.path.exists(samples_path):
            best_samples = self.training_sample_generalization_test(self.valid_segments,self.train_segments)
            json.dump(best_samples,open(samples_path,'w'))
        else:
            with open(samples_path, 'r') as f:
                best_samples = json.load(f)

        #'''

        t = {}
        for key in best_samples.keys():
            t[int(key)] = best_samples[key][:self.n_test_instances]
        best_samples = t
        self.x1,self.x2,self.y,self.users = self.generate_test_instances(self.test_segments,self.train_segments,best_samples,augmentation=False)
        #self.x1,self.x2,self.y,self.users = self.generate_valid_instances(self.test_segments,self.train_segments,self.users_list,np.arange(1,len(self.users_list)+1),augmentation=False)
        self.x1 = (np.array([i[0] for i in self.x1],dtype=np.float32),np.array([i[1] for i in self.x1],dtype=np.float32))
        self.x2 = (np.array([i[0] for i in self.x2],dtype=np.float32),np.array([i[1] for i in self.x2],dtype=np.float32))
        self.y = np.array(self.y,dtype=np.float32)
        self.users = np.array(self.users,dtype=np.int32)

        self.x_test1 = self.x1
        self.x_test2 = self.x2
        self.y_test = self.y.reshape((-1,1))
        self.test_users = self.users
        #'''

        #'''
        forest_y_pred = self.forest_classifier.predict_proba(self.x_test1[1] - self.x_test2[1])
        dl_y_pred = self.classifier.predict([self.x_test1[0],self.x_test2[0]],verbose=0)#[self.x_test1[0],self.x_test1[1],self.x_test2[0],self.x_test2[1]]
        y_pred = (forest_y_pred[:,1] + dl_y_pred.reshape(-1,))/2
        #y_pred = dl_y_pred.reshape(-1,)
        
        if TRAINING_SET == 'balabit':
            balabit_index = []
            for i in range(self.test_users.shape[0]):
                if self.test_users[i,0] in np.arange(121,131):
                    balabit_index.append(True)
                else:
                    balabit_index.append(False)
            balabit_index = np.array(balabit_index)
            self.y_test = self.y_test[balabit_index]
            y_pred = y_pred[balabit_index]
            self.test_users = self.test_users[balabit_index]
        

        y_test_pred,y_test,test_users = [],[],[]
        y_auc = []
        user_errors = {i:[0,0] for i in range(1,131)}
        

        for i in range(int(len(y_pred)/self.n_test_instances)):
            y_auc.append(np.mean(y_pred[i*self.n_test_instances:(i+1)*self.n_test_instances]))
            test_users.append(self.test_users[i*self.n_test_instances][0])
            if np.mean(y_pred[i*self.n_test_instances:(i+1)*self.n_test_instances]) > 0.5:
                y_test_pred.append(1)
            else:
                y_test_pred.append(0)
            y_test.append(self.y_test[self.n_test_instances*i])
            if y_test[-1] != y_test_pred[-1]:
                user_errors[self.test_users[i*self.n_test_instances][0]][0] += 1
            user_errors[self.test_users[i*self.n_test_instances][0]][1] += 1
        y_auc = np.array(y_auc)
        
        y_test = np.array(y_test).reshape(-1)
        user_auc = {}
        for i in np.unique(test_users):
            user_auc[i] = roc_auc_score(y_test[test_users == i],y_auc[test_users == i])
        #print(user_auc)
        

        fpr, tpr, threshold = roc_curve(y_test, y_auc, pos_label=1)
        fnr = 1 - tpr
        #print([roc_auc_score(y_test,y_auc),fpr[np.nanargmin(np.absolute((fnr - fpr)))],len(y_test)])
        print('AUC:{}, EER:{}'.format(roc_auc_score(y_test,y_auc),fpr[np.nanargmin(np.absolute((fnr - fpr)))]))
        frr,far = [],[]
        for i in range(21):
            t = []
            for j in y_auc:
                if j >= i*0.05:
                    t.append(1)
                else:
                    t.append(0)
            tn, fp, fn, tp = confusion_matrix(y_test,t).ravel()
            frr.append(fp/(fp+tn))
            far.append(fn/(fn+tp))
        #print(frr,far,sep='\n')
        print('FAR: ', far)
        print('FRR: ', frr)
        self.t_auc.append(roc_auc_score(y_test,y_auc))
        self.t_far.append(far[10])
        self.t_frr.append(frr[10])
        self.n_instances.append(len(y_auc))
        #'''



if __name__ == '__main__':
    np.random.seed(1)
    random.seed(1)
    tf.random.set_seed(1)
    classifier = mouse_classifier()
    for fold in range(5):
        classifier.update_fold(fold)
        classifier.train(epochs=500,batch_size=256)




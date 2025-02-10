from gc import callbacks
import numpy as np
import pandas as pd
from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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

PRE = True
RF = False
DL = False
FUSION = True

def data_augmentation(input,cut_points):
    p = 1
    res = input.copy()
    #permutation
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
    def __init__(self,x_valid,y_valid,fold,pre):
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.fold = fold
        self.pre = pre
        self.best_res = 0
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            valid_res = self.model.evaluate(self.x_valid,self.y_valid,verbose=0)
            loss1,acc,recall,precision = tuple(valid_res)
            print('epoch:{}, validation: loss:{}, acc:{}, recall:{}, precision:{}'.format(epoch,loss1,acc,recall,precision))
            if acc > self.best_res:
                self.best_res = acc
                self.model.save('./mouse/trained_models/embedding_{}_{}.h5'.format(self.fold,int(self.pre)),save_format='h5')


class mouse_classifier():
    def __init__(self):
        #gpus = tf.config.list_physical_devices('GPU')
        #tf.config.set_visible_devices(gpus[-4], 'GPU')
        #tf.config.experimental.set_memory_growth(gpus[-4], True)
        #logical_gpus = tf.config.list_logical_devices('GPU')
        #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

        
        self.data_shape = (256,4)
        loader = loader = data_loader(self.data_shape,pre=PRE)
        self.segments,self.users_list = loader.load()
            

        self.user_array = np.arange(1,len(self.users_list)+1)
        np.random.shuffle(self.user_array)



    def data_split(self,fold):
        l = int(len(self.user_array)/5)
        user_blocks = [self.user_array[i*l:(i+1)*l] for i in range(5)]
        
        self.test_users_index = user_blocks[fold%5]
        self.train_users_index = np.concatenate([user_blocks[(fold+1)%5],user_blocks[(fold+2)%5],user_blocks[(fold+3)%5]])
        self.valid_users_index = user_blocks[(fold+4)%5]


        self.x_train1,self.x_train2,self.y_train,self.train_users = self.generate_instances(self.segments,self.users_list,self.train_users_index,augmentation=False)
        self.x_train1 = (np.array([i[0] for i in self.x_train1],dtype=np.float32),np.array([i[1] for i in self.x_train1],dtype=np.float32))
        self.x_train2 = (np.array([i[0] for i in self.x_train2],dtype=np.float32),np.array([i[1] for i in self.x_train2],dtype=np.float32))
        self.y_train = np.array(self.y_train,dtype=np.float32).reshape((-1,1))
        self.train_users = np.array(self.train_users)

        self.x_valid1,self.x_valid2,self.y_valid,self.valid_users = self.generate_instances(self.segments,self.users_list,self.valid_users_index,augmentation=False)
        self.x_valid1 = (np.array([i[0] for i in self.x_valid1],dtype=np.float32),np.array([i[1] for i in self.x_valid1],dtype=np.float32))
        self.x_valid2 = (np.array([i[0] for i in self.x_valid2],dtype=np.float32),np.array([i[1] for i in self.x_valid2],dtype=np.float32))
        self.y_valid = np.array(self.y_valid,dtype=np.float32).reshape((-1,1))
        self.valid_users = np.array(self.valid_users)

        self.x_test1,self.x_test2,self.y_test,self.test_users = self.generate_instances(self.segments,self.users_list,self.test_users_index,augmentation=False)
        self.x_test1 = (np.array([i[0] for i in self.x_test1],dtype=np.float32),np.array([i[1] for i in self.x_test1],dtype=np.float32))
        self.x_test2 = (np.array([i[0] for i in self.x_test2],dtype=np.float32),np.array([i[1] for i in self.x_test2],dtype=np.float32))
        self.y_test = np.array(self.y_test,dtype=np.float32).reshape((-1,1))
        self.test_users = np.array(self.test_users)


    def generate_instances(self,segments,users_list,user_index,augmentation=False):
        x1,x2,y,users = [],[],[],[]
        for user in user_index:
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
                random_user = np.random.choice(user_index)#add random record from different users to keep the dataset balance
                while random_user == user:
                    random_user = np.random.choice(user_index)
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

    def build_classifier(self):
        data1 = keras.Input(shape=self.data_shape)
        data2 = keras.Input(shape=self.data_shape)

        #'''
        conv1 = layers.Conv1D(128, 8, padding='same', name='conv1')
        conv2 = layers.Conv1D(256, 5, padding='same', name='conv2')
        conv3 = layers.Conv1D(128, 3, padding='same', name='conv3')
        lstm1 = layers.LSTM(128, return_sequences=True)
        lstm2 = layers.LSTM(128)
        
        #'''
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
        #'''

        #'''
        lstm_output1 = lstm1(data1)
        lstm_output1 = lstm2(lstm_output1)

        lstm_output2 = lstm1(data2)
        lstm_output2 = lstm2(lstm_output2)
        #'''

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
        far_dic,frr_dic,user_auc = {},{},{}
        training_time,n_instances = 0,0
        for fold in range(5):
            self.data_split(fold)
            optimizer = Adam(learning_rate= 1e-5)
            self.classifier = self.build_classifier()
            self.classifier.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy',tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
                
            p = np.random.permutation(len(self.y_train))#shuffle
            self.x_train1 = (self.x_train1[0][p],self.x_train1[1][p])
            self.x_train2 = (self.x_train2[0][p],self.x_train2[1][p])
            self.y_train = self.y_train[p]

            x_train = self.x_train1[1] - self.x_train2[1]#np.concatenate((self.x_train1[1],self.x_train2[1]),axis=1)
            self.forest_classifier = RandomForestClassifier()
            self.forest_classifier.fit(x_train,self.y_train.reshape(-1,))

            #'''
            #start = time.time()
            model_path = './mouse/trained_models/embedding_{}_{}.h5'.format(fold,int(PRE))
            if not os.path.exists(model_path):
                mycallback = CustomCallback([self.x_valid1[0],self.x_valid2[0]],self.y_valid,fold,PRE)
                self.classifier.fit([self.x_train1[0],self.x_train2[0]],self.y_train,epochs=epochs,verbose=0,batch_size=batch_size,callbacks = [mycallback])
            #print('training time:{}'.format(time.time()-start))
            self.classifier = keras.models.load_model(model_path)
            start = time.time()
            forest_y_pred = self.forest_classifier.predict_proba(self.x_test1[1] - self.x_test2[1])
            dl_y_pred = self.classifier.predict([self.x_test1[0],self.x_test2[0]],verbose=0)#[self.x_test1[0],self.x_test1[1],self.x_test2[0],self.x_test2[1]]
            if RF:
                y_pred = forest_y_pred[:,1]
            elif DL:
                y_pred = dl_y_pred.reshape(-1,)
            elif FUSION:
                y_pred = (forest_y_pred[:,1] + dl_y_pred.reshape(-1,))/2
            training_time += time.time()-start
            n_instances += self.x_test1[0].shape[0]

            far_dic[fold] = []
            frr_dic[fold] = []
            for i in range(21):
                t = []
                for j in y_pred:
                    if j >= i*0.05:
                        t.append(1)
                    else:
                        t.append(0)

                tn, fp, fn, tp = confusion_matrix(self.y_test,t).ravel()
                frr_dic[fold].append(fp/(fp+tn))
                far_dic[fold].append(fn/(fn+tp))
            fpr, tpr, threshold = roc_curve(self.y_test,y_pred, pos_label=1)
            fnr = 1 - tpr
            print('AUC:{}, EER:{}'.format(roc_auc_score(self.y_test,y_pred),fpr[np.nanargmin(np.absolute((fnr - fpr)))]))
        print('FAR of 5 folds: ', far_dic)
        print('FRR of 5 folds: ', frr_dic)
        print('verification time: ', training_time/n_instances)
        #print(far_dic,frr_dic,training_time/n_instances,sep='\n')
            #'''


if __name__ == '__main__':

    np.random.seed(0)
    random.seed(0)
    tf.random.set_seed(0)
    classifier = mouse_classifier()
    classifier.train(epochs=200,batch_size=256)

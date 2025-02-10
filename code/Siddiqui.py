import numpy as np
import pandas as pd
from pathlib import Path
import random
import re
from sklearn.metrics import accuracy_score,recall_score,precision_score,roc_auc_score,confusion_matrix,roc_curve
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import time




def set_seeds(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=0):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    #torch.manual_seed(seed)

class mouse_classifier():
    def __init__(self):
        self.load_data()
        self.data_shape = (9,2)
        self.user_array = np.arange(0,len(self.users))
        np.random.shuffle(self.user_array)



    def split(self,fold):
        l = int(len(self.user_array)/5)
        user_blocks = [self.user_array[i*l:(i+1)*l] for i in range(5)]
        
        self.test_users_index = user_blocks[fold%5]
        self.train_users_index = np.concatenate([user_blocks[(fold+1)%5],user_blocks[(fold+2)%5],user_blocks[(fold+3)%5],user_blocks[(fold+4)%5]])


    def generate_instances(self,test_user):
        seg = self.segments[test_user]
        dur = self.durations[test_user]
        self.x_train,self.y_train = [],[]
        i,t = 0,0
        while t < 30:
            self.x_train.append(seg[i])
            self.y_train.append(1)
            t += dur[i]
            i += 1

            random_user = np.random.choice(self.train_users_index)
            random_segment_i = np.random.randint(0,len(self.segments[random_user]))
            self.x_train.append(self.segments[random_user][random_segment_i])
            self.y_train.append(0)
        
        self.x_test,self.y_test = [],[]
        t = 0
        while t < 30:
            self.x_test.append(seg[i])
            self.y_test.append(1)
            t += dur[i]
            i += 1

            random_user = np.random.choice(self.test_users_index)
            while random_user == test_user:
                random_user = np.random.choice(self.test_users_index)
            random_segment_i = np.random.randint(0,len(self.segments[random_user]))
            self.x_test.append(self.segments[random_user][random_segment_i])
            if type(self.segments[random_user][random_segment_i]) == list:
                print('test')
            self.y_test.append(0)

        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)
        self.x_test = np.array(self.x_test)
        self.y_test = np.array(self.y_test)


    def build_classifier(self):
        data1 = keras.Input(shape=self.data_shape)

        conv1 = layers.Conv1D(128, 1, name='conv1')
        conv2 = layers.Conv1D(128, 1, name='conv2')
        
        cnn_output1 = conv1(data1)
        cnn_output1 = conv2(cnn_output1)
        cnn_output1 = keras.layers.GlobalMaxPooling1D()(cnn_output1)
        cnn_output1 = keras.layers.Flatten()(cnn_output1)

        output = layers.Dense(60, activation='relu',name='dense1')(cnn_output1)
        output = layers.Dense(1, activation='sigmoid',name='dense2')(output)

        return Model(inputs=data1,outputs=output)

    def train(self, epochs, batch_size=128):
        far_dic,frr_dic = {},{}
        auc,eer = [],[]
        training_time = 0
        for fold in range(5):
            self.split(fold)
            pred, test = [],[]
            for user in self.test_users_index:
                optimizer = Adam(learning_rate= 1e-5)
                self.classifier = self.build_classifier()
                self.classifier.compile(loss='binary_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy',tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

                self.generate_instances(user)
                
                p = np.random.permutation(len(self.y_train))#shuffle
                self.x_train = self.x_train[p]
                self.y_train = self.y_train[p]

                #'''
                start = time.time()
                #mycallback = CustomCallback([self.x_valid1,self.x_valid2],self.y_valid,fold)
                self.classifier.fit(self.x_train,self.y_train.astype(np.float64),epochs=epochs,verbose=0,batch_size=batch_size)
                #print('training time:{}'.format(time.time()-start))
                #test_res = self.classifier.evaluate(self.x_test,self.y_test,verbose=0)
                #loss,acc,recall,precision = tuple(test_res)
                y_pred = self.classifier.predict(self.x_test,verbose=0)
                training_time += time.time()-start

                pred += y_pred.reshape((-1)).tolist()
                test += self.y_test.tolist()


            far_dic[fold] = []
            frr_dic[fold] = []
            for i in range(21):
                t = []
                for j in pred:
                    if j >= i*0.05:
                        t.append(1)
                    else:
                        t.append(0)

                tn, fp, fn, tp = confusion_matrix(test,t).ravel()
                frr_dic[fold].append(fp/(fp+tn))
                far_dic[fold].append(fn/(fn+tp))
            auc.append(roc_auc_score(test,pred))

            fpr, tpr, threshold = roc_curve(test, pred, pos_label=1)
            fnr = 1 - tpr
            eer.append(fpr[np.nanargmin(np.absolute((fnr - fpr)))])

        print('AUC:{}, EER:{}'.format(np.mean(auc),np.mean(eer)))
        print(far_dic,frr_dic,training_time)

    def segmentation(self,dataframe):
            dataframe.sort_values(by=['client timestamp'],inplace=True)
            dataframe = dataframe.loc[dataframe['button']!='Scroll']
            dataframe = dataframe.drop_duplicates('client timestamp')

            x_list = dataframe['x'].to_list()
            y_list = dataframe['y'].to_list()
            x_max,x_min = max(x_list),min(x_list)
            x_list = [(float(x)-x_min)/(x_max-x_min) for x in x_list]
            y_max,y_min = max(y_list),min(y_list)
            y_list = [(float(y)-y_min)/(y_max-y_min) for y in y_list]
            time_list = dataframe['client timestamp'].to_list()
            if type(time_list[0]) == int:#timestamp in ms
                time_list = [time_list[i]/1000  for i in range(0,len(time_list))]

            i = 1
            segments,durations = [],[]
            tx, ty, tt = [x_list[0]],[y_list[0]],[time_list[0]]
            while i < len(x_list):
                if len(tx) == 10:
                    ax = [(tx[i+1] - tx[i])/(tt[i+1] - tt[i]) for i in range(9)]
                    ay = [(ty[i+1] - ty[i])/(tt[i+1] - tt[i]) for i in range(9)]
                    segments.append(np.array([np.array(ax),np.array(ay)]).T)
                    durations.append(tt[-1]-tt[0])
                    tx, ty, tt = [tx[-1]],[ty[-1]],[tt[-1]]
                if time_list[i] - tt[-1] > 10:
                    tx, ty, tt = [],[],[]
                tx.append(x_list[i])
                ty.append(y_list[i])
                tt.append(time_list[i])
                i += 1

            return segments,durations

    def load_data(self):
        self.segments,self.durations,self.users = [],[],[]

        pathlist = Path('./mouse/balabit').glob('**/session_*')
        user_dic = {7:121, 9:122, 12:123, 15:124, 16:125, 20:126, 21:127, 23:128, 29:129, 35:130}
        for path in pathlist:
            path_in_str = str(path)
            user_index = int(re.findall(r'\d+',path_in_str)[0])
            #if user_dic[user_index] in [123,124,125,127,128,129,130]:
            #    continue
            #if user_index != 12:
            #    continue
            t = pd.read_csv(path_in_str)
            seg,dur = self.segmentation(t)
            if not user_dic[user_index] in self.users:
                self.segments.append(seg)
                self.durations.append(dur)
                self.users.append(user_dic[user_index])
            else:
                self.segments[self.users.index(user_dic[user_index])] += seg
                self.durations[self.users.index(user_dic[user_index])] += dur

        pathlist = Path('./mouse/sapimouse').glob('**/*.csv')
        for path in pathlist:
            path_in_str = str(path)
            user_index = int(re.findall(r'\d+',path_in_str)[0])
            t = pd.read_csv(path_in_str)
            seg,dur = self.segmentation(t)
            if not user_index in self.users:
                self.segments.append(seg)
                self.durations.append(dur)
                self.users.append(user_index)
            else:
                self.segments[self.users.index(user_index)] += seg
                self.durations[self.users.index(user_index)] += dur


        print('loading completed')




if __name__ == '__main__':
    set_global_determinism(0)
    classifier = mouse_classifier()
    classifier.train(epochs=100,batch_size=8)
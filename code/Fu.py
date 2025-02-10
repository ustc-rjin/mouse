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
        self.data_shape = (128,4)
        self.load_data()
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
            self.y_train.append([0,1])
            t += dur[i]
            i += 1
        self.x_train = self.x_train[:8*int(len(self.x_train)/8)]
        self.y_train = self.y_train[:8*int(len(self.x_train)/8)]

        for j in range(int(len(self.x_train)/8)*14):
            random_user = np.random.choice(self.train_users_index)
            while random_user == test_user:
                random_user = np.random.choice(self.train_users_index)
            random_segment_i = np.random.randint(0,len(self.segments[random_user])-8)
            self.x_train += self.segments[random_user][random_segment_i:random_segment_i+8]
            self.y_train += [[1,0]] * 8
        
        self.x_test,self.y_test = [],[]
        t = 0
        while t < 30:
            self.x_test.append(seg[i])
            self.y_test.append([0,1])
            t += dur[i]
            i += 1
        self.x_test = self.x_test[:8*int(len(self.x_test)/8)]
        self.y_test = self.y_test[:8*int(len(self.x_test)/8)]

        for i in range(int(len(self.x_test)/8)):
            random_user = np.random.choice(self.test_users_index)
            while random_user == test_user:
                random_user = np.random.choice(self.test_users_index)
            random_segment_i = np.random.randint(0,len(self.segments[random_user])-8)
            self.x_test += self.segments[random_user][random_segment_i:random_segment_i+8]
            self.y_test += [[1,0]] * 8

        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)
        self.x_test = np.array(self.x_test)
        self.y_test = np.array(self.y_test)


    def build_classifier(self):
        data1 = keras.Input(shape=self.data_shape)

        conv1 = layers.Conv1D(32, 8, 1, name='conv1')
        conv2 = layers.Conv1D(48, 6, 1, name='conv2')
        conv3 = layers.Conv1D(64, 4, 1, name='conv3')

        rnn1 = layers.Bidirectional(layers.LSTM(256, return_sequences=True))
        rnn2 = layers.Bidirectional(layers.LSTM(256))
        
        cnn_output1 = conv1(data1)
        cnn_output1 = conv2(cnn_output1)
        cnn_output1 = conv3(cnn_output1)
        #cnn_output1 = keras.layers.Flatten()(cnn_output1)

        rnn_output = rnn1(cnn_output1)
        rnn_output = rnn2(rnn_output)

        output = layers.Dense(2, activation='softmax',name='dense1')(rnn_output)

        return Model(inputs=data1,outputs=output)

    def train(self, epochs, batch_size=128):
        far_dic,frr_dic = {},{}
        auc,eer = [],[]
        training_time = 0
        for fold in range(5):
            self.split(fold)
            pred, test = [],[]
            for user in self.test_users_index:
                print(user)
                self.generate_instances(user)
                t_pred = np.zeros(int(self.x_test.shape[0]/8))
                for n_classifier in range(8):
                    optimizer = Adam()
                    self.classifier = self.build_classifier()
                    self.classifier.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy',tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

                    
                    train_index = [i*8+n_classifier for i in range(int(self.x_train.shape[0]/8))]
                    test_index = [i*8+n_classifier for i in range(int(self.x_test.shape[0]/8))]

                    #'''
                    start = time.time()
                    #mycallback = CustomCallback([self.x_valid1,self.x_valid2],self.y_valid,fold)
                    self.classifier.fit(self.x_train[train_index],self.y_train[train_index].astype(np.float64),epochs=epochs,verbose=0,batch_size=batch_size)
                    training_time += time.time()-start
                    #print('training time:{}'.format(time.time()-start))
                    y_pred = self.classifier(self.x_test[test_index],training=False)

                    t_pred += y_pred.numpy().T[1]

                t_pred /= 8
                pred += t_pred.tolist()
                test += self.y_test[test_index].T[1].tolist()


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
        print(far_dic,frr_dic,training_time/len(self.users))

    def segmentation(self,dataframe):
            dataframe.sort_values(by=['client timestamp'],inplace=True)
            dataframe = dataframe.loc[dataframe['button']!='Scroll']
            dataframe = dataframe.drop_duplicates('client timestamp')


            segments,durations = [],[]
            start,end = [0],[0]
            time_list = dataframe['client timestamp'].to_list()
            time_list = time_list[1:]
            time_list = [(time_list[i] - time_list[i-1]) for i in range(1,len(time_list))]
            t = np.array(time_list[:100].copy())
            if np.sum(t > 1) > np.sum(t < 1):#timestamp in ms
                time_list = [time_list[i]/1000  for i in range(0,len(time_list))]
            state_list = dataframe['state'].to_list()#the value can be Move, Pressed, Drag, or Released
            state_list = state_list[1:]
            x_list = dataframe['x'].to_list()
            x_list = x_list[1:]
            x_max,x_min = max(x_list),min(x_list)
            x_list = [(float(x)-x_min)/(x_max-x_min) for x in x_list]
            x_cors = [(float(x)-x_min)/(x_max-x_min) for x in x_list]
            y_list = dataframe['y'].to_list()
            y_list = y_list[1:]
            y_max,y_min = max(y_list),min(y_list)
            y_list = [(float(y)-y_min)/(y_max-y_min) for y in y_list]
            y_cors = [(float(y)-y_min)/(y_max-y_min) for y in y_list]
            x_list = [x_list[i] - x_list[i-1] for i in range(1,len(x_list))]
            y_list = [y_list[i] - y_list[i-1] for i in range(1,len(y_list))]
            x_speed_list = [x_list[i]/(time_list[i]+1) for i in range(0,len(x_list))]
            y_speed_list = [y_list[i]/(time_list[i]+1) for i in range(0,len(y_list))]

            event_index = 0
            while True:
                try:
                    current_state = state_list[start[event_index]]

                    if current_state in ['Released','Drag']:#if button was pressed before recording started, or drag is not between pressed and released, discard the record.
                        start[event_index] += 1
                        end[event_index] += 1
                        continue

                    if current_state == 'Move':
                        while state_list[end[event_index]+1] == 'Move' and time_list[end[event_index]] < 0.3:
                            end[event_index] += 1
                        start.append(end[event_index]+1)
                        end.append(end[event_index]+1)

                    if current_state == 'Pressed':
                        while state_list[end[event_index]] != 'Released':
                            end[event_index] += 1
                            if state_list[end[event_index]] == 'Pressed':#both buttons have been pressed
                                while state_list[end[event_index]] != 'Released':
                                    end[event_index] += 1
                                end[event_index] += 1
                        end[event_index] += 1
                        start.append(end[event_index])
                        end.append(end[event_index])
                    event_index += 1
                except IndexError:
                    break

            start = start[:-1]
            end = end[:-1]

            #filter meaningless events
            i = 0
            while i < len(start):
                if end[i] - start[i] < 5 and state_list[start[i]] != 'Pressed':
                    start.pop(i)
                    end.pop(i)
                    continue
                if np.sum(time_list[start[i]:end[i]]) > 10:
                    start.pop(i)
                    end.pop(i)
                    continue
                i += 1

            event_index = 0
            while True:
                if event_index == len(start):
                    break
                buf = np.zeros((self.data_shape[1],self.data_shape[0]))
                dx = x_list[start[event_index]:end[event_index]]
                dy = y_list[start[event_index]:end[event_index]]
                vx = x_speed_list[start[event_index]:end[event_index]]
                vy = y_speed_list[start[event_index]:end[event_index]]
                length = len(dx)
                if length > self.data_shape[0]:
                    event_index += 1
                    continue
                buf[0,:length] = dx
                buf[1,:length] = dy
                buf[2,:length] = vx
                buf[3,:length] = vy
                segments.append(buf.T)
                durations.append(np.sum(time_list[start[event_index]:end[event_index]]))
                event_index += 1
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
            #if user_index == 84:
            #    print('test')
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
    classifier.train(epochs=100,batch_size=10)
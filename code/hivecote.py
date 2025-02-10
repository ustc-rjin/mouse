from numba import config
config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY = True
from aeon.classification.hybrid import HIVECOTEV2
import numpy as np
from sklearn.metrics import accuracy_score,recall_score,precision_score,roc_auc_score,confusion_matrix
import time
import tensorflow as tf
import random

from load_data import data_loader
class mouse_classifier():
    def __init__(self):
        gpus = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(gpus[-3], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[-3], True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

        self.data_shape = (256,4)
        loader = loader = data_loader(self.data_shape)
        self.segments,self.users_list = loader.load()

        self.user_array = np.arange(1,len(self.users_list)+1)
        np.random.shuffle(self.user_array)
            

    def generate_instances(self,segments,users_list,user_index):
        x1,x2,y,users = [],[],[],[]
        for user in user_index:
            user_segments,cut_points = segments[users_list.index(user)]
            for i in range(int(len(user_segments)/2)):
                #record from the same user
                x1.append(user_segments[i].T)
                x2.append(user_segments[i+int(len(user_segments)/2)].T)
                y.append(1)
                users.append((user,user))

                #for k in range(3):
                random_user = np.random.choice(user_index)#add random record from different users to keep the dataset balance
                while random_user == user:
                    random_user = np.random.choice(user_index)
                random_segments,random_cut_points = segments[users_list.index(random_user)]
                random_segment_i = np.random.randint(0,len(random_segments))

                x1.append(user_segments[i].T)
                x2.append(random_segments[random_segment_i].T)
                y.append(0)
                users.append([user,random_user])
        return x1,x2,y,users

    def data_split(self,fold):
        #train test split
        '''
        array = np.arange(1,len(self.users_list)+1)
        np.random.shuffle(array)
        self.train_users_index = array[int(len(array)/5):]
        self.test_users_index = array[:int(len(array)/5)]
        '''

        l = int(len(self.user_array)/5)
        user_blocks = [self.user_array[i*l:(i+1)*l] for i in range(5)]
        
        self.test_users_index = user_blocks[fold%5]
        self.train_users_index = np.concatenate([user_blocks[(fold+1)%5],user_blocks[(fold+2)%5],user_blocks[(fold+3)%5],user_blocks[(fold+4)%5]])


        self.x_train1,self.x_train2,self.y_train,self.train_users = self.generate_instances(self.segments,self.users_list,self.train_users_index)#[],[],[],[]
        '''
        for user in self.train_users_index:#range(1,97):
            user_segments,cut_points = train_segments[train_users.index(user)]
            for i in range(int(len(user_segments)/2)):
                #record from the same user
                #self.x_train1.append(user_segments[i])
                #self.x_train2.append(user_segments[i+int(len(user_segments)/2)])
                self.x_train1.append(user_segments[i].T)
                self.x_train2.append(user_segments[i+int(len(user_segments)/2)].T)
                #self.y_train.append(np.array([0,1]))
                self.y_train.append(1)
                self.train_users.append((user,user))

                #for k in range(3):
                random_user = np.random.choice(self.train_users_index)#add random record from different users to keep the dataset balance
                while random_user == user:
                    random_user = np.random.choice(self.train_users_index)
                random_segments,random_cut_points = train_segments[train_users.index(random_user)]
                random_segment_i = np.random.randint(0,len(random_segments))

                #self.x_train1.append(user_segments[i])
                #self.x_train2.append(random_segments[random_segment_i])
                self.x_train1.append(user_segments[i].T)
                self.x_train2.append(random_segments[random_segment_i].T)
                #self.y_train.append(np.array([1,0]))
                self.y_train.append(0)
                self.train_users.append([user,random_user])
        '''
        self.x_train1 = np.array(self.x_train1,dtype=np.float32)
        self.x_train2 = np.array(self.x_train2,dtype=np.float32)
        self.y_train = np.array(self.y_train,dtype=np.float32)
        self.train_users = np.array(self.train_users)

        self.x_test1,self.x_test2,self.y_test,self.test_users = self.generate_instances(self.segments,self.users_list,self.test_users_index)#[],[],[],[]
        '''
        for user in self.test_users_index:#range(97,121):
            user_segments,_ = train_segments[train_users.index(user)]
            for i in range(int(len(user_segments)/2)):
                self.x_test1.append(user_segments[i].T)#record from the same user
                self.x_test2.append(user_segments[i+int(len(user_segments)/2)].T)
                #self.y_test.append(np.array([0,1]))
                self.y_test.append(1)
                self.test_users.append((user,user))

                #add random record from different users to keep the dataset balance
                random_user = np.random.choice(self.test_users_index)#np.random.randint(97,121)
                while random_user == user:
                    random_user = np.random.choice(self.test_users_index)#np.random.randint(97,121)
                random_segments,_ = train_segments[train_users.index(random_user)]
                random_segment = random_segments[np.random.randint(0,len(random_segments))]
                self.x_test1.append(user_segments[i].T)
                self.x_test2.append(random_segment.T)
                #self.y_test.append(np.array([1,0]))
                self.y_test.append(0)
                self.test_users.append([user,random_user])
        '''
        self.x_test1 = np.array(self.x_test1,dtype=np.float32)
        self.x_test2 = np.array(self.x_test2,dtype=np.float32)
        self.y_test = np.array(self.y_test,dtype=np.float32)
        self.test_users = np.array(self.test_users)


    def train(self):
        far_dic,frr_dic = {},{}
        for fold in range(5):
            self.data_split(fold)
            p = np.random.permutation(len(self.y_train))#shuffle
            self.x_train1 = self.x_train1[p]
            self.x_train2 = self.x_train2[p]
            self.y_train = self.y_train[p]
            self.x_train = np.concatenate([self.x_train1,self.x_train2],axis=1,dtype=np.float64)
            self.x_train = self.x_train[:10000,:,:]
            self.y_train = self.y_train[:10000]

            p = np.random.permutation(len(self.y_test))
            self.x_test1 = self.x_test1[p]
            self.x_test2 = self.x_test2[p]
            self.y_test = self.y_test[p]
            self.x_test = np.concatenate([self.x_test1,self.x_test2],axis=1,dtype=np.float64)
            self.x_test = self.x_test[:1000,:,:]
            self.y_test = self.y_test[:1000]

            #'''
            start = time.time()
            self.classifier = HIVECOTEV2(verbose=1,random_state=0,n_jobs=1,time_limit_in_minutes=720)
            self.classifier.fit(self.x_train,self.y_train)
            print('training time:{}'.format(time.time()-start))
            start = time.time()
            y_test_pred_prob = self.classifier.predict_proba(self.x_test)
            y_test_pred,t = [],[]
            print(y_test_pred_prob[0])
            for y in y_test_pred_prob:
                t.append(y[1])
                if y[1] > 0.5:
                    y_test_pred.append(1)
                else:
                    y_test_pred.append(0)
            y_test_pred_prob = t

            far_dic[fold] = []
            frr_dic[fold] = []
            for i in range(21):
                t = []
                for j in y_test_pred_prob:
                    if j >= i*0.05:
                        t.append(1)
                    else:
                        t.append(0)
                tn, fp, fn, tp = confusion_matrix(self.y_test,t).ravel()
                frr_dic[fold].append(fp/(fp+tn))
                far_dic[fold].append(fn/(fn+tp))

            test_eval = [accuracy_score(self.y_test,y_test_pred),recall_score(self.y_test,y_test_pred),precision_score(self.y_test,y_test_pred),roc_auc_score(self.y_test,y_test_pred_prob)]
            print('evaluation:{}'.format(test_eval))
            #'''
        print(far_dic,frr_dic)


if __name__ == '__main__':

    np.random.seed(0)
    random.seed(0)
    classifier = mouse_classifier()
    classifier.train()




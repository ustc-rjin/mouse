import math
import numpy as np
import pandas as pd
from pathlib import Path
import random
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,roc_auc_score,confusion_matrix,roc_curve
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

def set_seeds(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=0):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    #torch.manual_seed(seed)

class mouse_classifier():
    def __init__(self):
        self.data_shape = (1,39)
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
        while t < 20:
            self.x_train.append(seg[i])
            self.y_train.append(1)
            t += dur[i]
            i += 1

            random_user = np.random.choice(self.train_users_index)
            while random_user == test_user:
                random_user = np.random.choice(self.train_users_index)
            random_segment_i = np.random.randint(0,len(self.segments[random_user]))
            self.x_train.append(self.segments[random_user][random_segment_i])
            self.y_train += [0]
        
        self.x_test,self.y_test = [],[]
        t = 0
        while t < 20:
            self.x_test.append(seg[i])
            self.y_test.append(1)
            t += dur[i]
            i += 1

        random_user = np.random.choice(self.test_users_index)
        while random_user == test_user:
            random_user = np.random.choice(self.test_users_index)
        random_segment_i = np.random.randint(0,len(self.segments[random_user])-len(self.x_test))
        self.y_test += [0] * len(self.x_test)
        self.x_test += self.segments[random_user][random_segment_i:random_segment_i+len(self.x_test)]

        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train).T
        self.x_test = np.array(self.x_test)
        self.y_test = np.array(self.y_test).T


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
                start = time.time()
                self.classifier = RandomForestClassifier()
                self.classifier.fit(self.x_train,self.y_train.T)
                y_pred = self.classifier.predict_proba(self.x_test).T[1]
                training_time += time.time()-start

                pred.append(np.mean(y_pred[self.y_test==1]))
                test.append(1)
                pred.append(np.mean(y_pred[self.y_test==0]))
                test.append(0)
                


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


            segments,durations,types = [],[],[]
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


            event_index = 0
            while True:
                try:
                    current_state = state_list[start[event_index]]

                    if current_state in ['Released','Drag']:#if button was pressed before recording started, or drag is not between pressed and released, discard the record.
                        start[event_index] += 1
                        end[event_index] += 1
                        continue

                    if current_state == 'Move':
                        while state_list[end[event_index]+1] == 'Move' and time_list[end[event_index]] < 10:
                            end[event_index] += 1
                        if state_list[end[event_index]+1] == 'Pressed' and state_list[end[event_index]+2] == 'Released' and time_list[end[event_index]] < 10:
                            end[event_index] += 2
                            types.append(1)
                        else:
                            types.append(0)
                        start.append(end[event_index]+1)
                        end.append(end[event_index]+1)

                    if current_state == 'Pressed':
                        drag = 1
                        while state_list[end[event_index]] != 'Released':
                            if state_list[end[event_index]] != 'Released' == 'Drug':
                                drag = 2
                            end[event_index] += 1
                            if state_list[end[event_index]] == 'Pressed':#both buttons have been pressed
                                while state_list[end[event_index]] != 'Released':
                                    end[event_index] += 1
                                end[event_index] += 1
                        end[event_index] += 1
                        start.append(end[event_index])
                        end.append(end[event_index])
                        types.append(drag)
                    event_index += 1
                except IndexError:
                    break

            start = start[:-1]
            end = end[:-1]

            #filter meaningless events
            i = 0
            while i < len(start):
                if end[i] - start[i] < 5:
                    start.pop(i)
                    end.pop(i)
                    types.pop(i)
                    continue
                i += 1

            event_index = 0
            while True:
                if event_index == len(start):
                    break
                dx = x_list[start[event_index]:end[event_index]]
                dy = y_list[start[event_index]:end[event_index]]
                dt = time_list[start[event_index]:end[event_index]]
                buf = np.array(self.cal_features(dx,dy,dt,types[event_index]))
                segments.append(buf)
                durations.append(np.sum(time_list[start[event_index]:end[event_index]]))
                event_index += 1
            return segments,durations

    def cal_features(self,dx,dy,dt,action_type):
        res = []
        angle, vx, vy, s, v, curvature = [],[],[],[],[],[]
        acc, jerk, angle_v = [],[],[]
        for i in range(len(dx)):
            angle.append(math.atan2(dy[i],dx[i]))
            vx.append(dx[i]/dt[i])
            vy.append(dy[i]/dt[i])
            s.append(math.sqrt(dx[i]*dx[i] + dy[i]*dy[i]))
            v.append(s[i]/dt[i])
            if s[i] != 0:
                curvature.append(angle[i]/s[i])
            else:
                curvature.append(0)
        for i in range(len(v)-1):
            acc.append((v[i+1]-v[i])*2/(dt[i]+dt[i+1]))
            angle_v.append(angle[i]*2/(dt[i]+dt[i+1]))
        for i in range(len(acc)-1):
            jerk.append((acc[i+1]-acc[i])/dt[i+1])
        
        for time_sery in [vx,vy,v,acc,jerk,angle_v,curvature]:
            res.append(np.mean(time_sery))
            res.append(np.std(time_sery))
            res.append(min(time_sery))
            res.append(max(time_sery))
        res.append(action_type)
        res.append(np.sum(dt))
        res.append(np.sum(s))
        tx, ty = np.sum(dx), np.sum(dy)
        dist = math.sqrt(tx**2 + ty**2)
        res.append(dist)

        if abs(tx) >= abs(ty) and tx >= 0 and ty >= 0:
            res.append(1)
        elif abs(tx) <= abs(ty) and tx >= 0 and ty >= 0:
            res.append(2)
        elif abs(tx) <= abs(ty) and tx <= 0 and ty >= 0:
            res.append(3)
        elif abs(tx) >= abs(ty) and tx <= 0 and ty >= 0:
            res.append(4)
        elif abs(tx) >= abs(ty) and tx <= 0 and ty <= 0:
            res.append(5)
        elif abs(tx) <= abs(ty) and tx <= 0 and ty <= 0:
            res.append(6)
        elif abs(tx) <= abs(ty) and tx >= 0 and ty <= 0:
            res.append(7)
        elif abs(tx) >= abs(ty) and tx >= 0 and ty <= 0:
            res.append(8)

        res.append(dist/(np.sum(s)+1))
        res.append(len(dx))
        res.append(np.sum(angle))
        
        p1 = np.array([0,0])
        p2 = np.array([tx,ty])
        res.append(0)
        for i in range(len(dx)):
            p3 = np.array([np.sum(dx[:i]),np.sum(dy[:i])])
            d = np.linalg.norm(np.abs(np.cross(p2-p1, p1-p3)))/(np.linalg.norm(p2-p1)+1)
            if d > res[-1]:
                res[-1] = d

        res.append(np.sum(np.array(curvature) < 0.0005))
        acc_t = 0
        for i in range(len(acc)):
            if acc[i] > 0:
                acc_t += (dt[i]+dt[i+1])/2
        res.append(acc_t)

        return np.array(res)

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
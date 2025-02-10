import numpy as np
import math
#import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import re
import struct
import time


class data_loader():
    def __init__(self,data_shape=(256,4),pre=True):
        self.data_shape = data_shape
        self.event_lengths = {}
        self.mouse_movement_time = {}
        self.session_len = {}
        self.pre = pre
        

    def load(self):
        segments,users = [],[]
        

        #'''
        pathlist = Path('./mouse/balabit').glob('**/session_*')
        #user_dic = {7:1, 9:2, 12:3, 15:4, 16:5, 20:6, 21:7, 23:8, 29:9, 35:10}
        user_dic = {7:121, 9:122, 12:123, 15:124, 16:125, 20:126, 21:127, 23:128, 29:129, 35:130}
        for path in pathlist:
            path_in_str = str(path)
            user_index = int(re.findall(r'\d+',path_in_str)[0])
            #if user_dic[user_index] in [123,124,125,127,128,129,130]:
            #    continue
            #if user_index != 12:
            #    continue
            t = pd.read_csv(path_in_str)
            if not user_dic[user_index] in users:
                segments.append(self.segmentation(t,self.data_shape[0],user_dic[user_index]))
                users.append(user_dic[user_index])
            else:
                t3 = self.segmentation(t,self.data_shape[0],user_dic[user_index])
                t1 = segments[users.index(user_dic[user_index])][0] + t3[0]
                t2 = segments[users.index(user_dic[user_index])][1] + t3[1]
                segments[users.index(user_dic[user_index])] = (t1,t2)
        #'''

        #'''
        pathlist = Path('./mouse/sapimouse').glob('**/*.csv')
        for path in pathlist:
            path_in_str = str(path)
            user_index = int(re.findall(r'\d+',path_in_str)[0])
            t = pd.read_csv(path_in_str)
            if not user_index in users:
                segments.append(self.segmentation(t,self.data_shape[0],user_index))
                users.append(user_index)
            else:
                t3 = self.segmentation(t,self.data_shape[0],user_index)
                t1 = segments[users.index(user_index)][0] + t3[0]
                t2 = segments[users.index(user_index)][1] + t3[1]
                segments[users.index(user_index)] = (t1,t2)
        #'''

        print('loading completed')

        n_samples1,n_samples2 = [],[]
        for i,user in enumerate(users):
            if user < 121:
                n_samples1.append(np.sum(self.mouse_movement_time[user]))
            else:
                n_samples2.append(np.sum(self.mouse_movement_time[user]))
        n_samples1,n_samples2 = [],[]
        for i,user in enumerate(users):
            if user < 121:
                n_samples1.append(np.sum(self.session_len[user]))
            else:
                n_samples2.append(np.sum(self.session_len[user]))
        return segments,users
        
    def up_sampling(self,dataframe):
        time_list = dataframe['client timestamp'].to_list()
        button_list = dataframe['button'].to_list()
        state_list = dataframe['state'].to_list()
        for i in range(dataframe.shape[0]):
            pass

    def log_smooth(self,input):
        return input
        if input >= 0:
            return math.log10(input+1)
        else:
            return -math.log10(abs(input-1))
        

    def segmentation(self,dataframe,duration=256,user_id=0):#input dataframe and duration
        dataframe.sort_values(by=['client timestamp'],inplace=True)
        dataframe = dataframe.loc[dataframe['button']!='Scroll']
        dataframe = dataframe.drop_duplicates('client timestamp')
        #dataframe.drop_duplicates(inplace=True)
        #dataframe = self.up_sampling(dataframe)
        '''
        duplicates = (dataframe.iloc[:,:4].shift() != dataframe.iloc[:,:4]).values
        duplicates = np.array(np.sum(duplicates,axis=1),dtype=bool)
        dataframe = dataframe.loc[duplicates]

        x_list = dataframe['x'].to_list()
        y_list = dataframe['y'].to_list()
        state_list = dataframe['state'].to_list()
        duplicates = [True]
        for i in range(1,len(x_list)):
            if (abs(x_list[i]-x_list[i-1]) + abs(y_list[i]-y_list[i-1])) < 3 and state_list[i] == 'Move':
                duplicates.append(False)
            else:
                duplicates.append(True)
        dataframe = dataframe.loc[duplicates]

        subsampling = [True]
        time_list = dataframe['client timestamp'].to_list()
        if time_list[-1] > 1000:#timestamp in ms
            time_list = [time_list[i]/1000  for i in range(0,len(time_list))]
        i = 1
        while i < len(time_list):
            if time_list[i] - time_list[i-1] < 0.01  and state_list[i] == 'Move':
                subsampling += [False,True]
                i += 2
            else:
                subsampling.append(True)
                i += 1
        subsampling = subsampling[:dataframe.shape[0]]
        dataframe = dataframe.loc[subsampling]
        '''


        segments,durations,types = [],[],[]
        start,end = [0],[0]
        time_list = dataframe['client timestamp'].to_list()
        time_list = time_list[1:]
        time_list = [(time_list[i] - time_list[i-1]) for i in range(1,len(time_list))]
        t = np.array(time_list[:100].copy())
        if np.sum(t > 1) > np.sum(t < 1):#timestamp in ms
            time_list = [time_list[i]/1000  for i in range(0,len(time_list))]
        if not user_id in self.session_len.keys():
            self.session_len[user_id] = [np.sum(time_list)]
        else:
            self.session_len[user_id].append(np.sum(time_list))
        #print(np.sum(time_list))
        button_list = dataframe['button'].to_list()#clicked button, the value can be NoButton, Left, or Right
        state_list = dataframe['state'].to_list()#the value can be Move, Pressed, Drag, or Released
        state_list = state_list[1:]
        x_list = dataframe['x'].to_list()
        x_list = x_list[1:]
        x_max,x_min = max(x_list),min(x_list)
        x_list = [(float(x)-x_min)/(x_max-x_min) for x in x_list]
        x_cors = x_list.copy()
        y_list = dataframe['y'].to_list()
        y_list = y_list[1:]
        y_max,y_min = max(y_list),min(y_list)
        y_list = [(float(y)-y_min)/(y_max-y_min) for y in y_list]
        y_cors = y_list.copy()
        x_list = [x_list[i] - x_list[i-1] for i in range(1,len(x_list))]
        y_list = [y_list[i] - y_list[i-1] for i in range(1,len(y_list))]
        x_speed_list = [x_list[i]/(time_list[i]+1) for i in range(0,len(x_list))]
        y_speed_list = [y_list[i]/(time_list[i]+1) for i in range(0,len(y_list))]
        x_acc_list = [(x_speed_list[i] - x_speed_list[i-1])/(time_list[i]+1) for i in range(1,len(x_speed_list))]
        x_acc_list = [0] + x_acc_list
        y_acc_list = [(y_speed_list[i] - y_speed_list[i-1])/(time_list[i]+1) for i in range(1,len(y_speed_list))]
        y_acc_list = [0] + y_acc_list



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
                    if state_list[end[event_index]+1] == 'Pressed' and state_list[end[event_index]+2] == 'Released' and time_list[end[event_index]] < 0.3:
                        end[event_index] += 2
                        types.append(1)
                    else:
                        types.append(0)
                    start.append(end[event_index]+1)
                    end.append(end[event_index]+1)

                if current_state == 'Pressed':
                    drag = 1
                    while state_list[end[event_index]] != 'Released':
                        if state_list[end[event_index]] == 'Drug':
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

        if self.pre:
            #filter meaningless events
            i = 0
            while i < len(start):
                if end[i] - start[i] < 5:
                    start.pop(i)
                    end.pop(i)
                    continue
                if np.max(x_cors[start[i]:end[i]]) - np.min(x_cors[start[i]:end[i]]) < 0.05 and np.max(y_cors[start[i]:end[i]]) - np.min(y_cors[start[i]:end[i]]) < 0.05:
                    start.pop(i)
                    end.pop(i)
                    continue
                i += 1

            #for i in range(len(start)):
            #    length = end[i] - start[i]
            #    if not length in self.event_lengths.keys():
            #        self.event_lengths[length] = 1
            #    else:
            #        self.event_lengths[length] += 1

            #concatenate to generate samples
            segment = []
            event_index = 0
            while True:
                if event_index == len(start):
                    break
                if end[event_index] - start[event_index] > duration:
                    event_index += 1
                    segment = []
                    continue
                t = segment.copy()
                segment.append(event_index)
                length = 0
                for event in segment:
                    length += end[event] - start[event]
                if length > duration:
                    segments.append(t)
                    segment = []
                    if not len(t) == 1:
                        event_index = t[1]
                    else:
                        event_index = t[0] + 1
                else:
                    event_index += 1

        else:
            segment = []
            event_index = 0
            while True:
                if event_index == len(start):
                    break
                if end[event_index] - start[event_index] > duration:
                    event_index += 1
                    segment = []
                    continue
                t = segment.copy()
                segment.append(event_index)
                length = 0
                for event in segment:
                    length += end[event] - start[event]
                if length > duration:
                    #exclude concatenated segments
                    concatenated = False
                    for event in t[1:]:
                        if np.sum(time_list[end[event-1]:start[event]]) > 10:
                            concatenated = True
                    if not concatenated:
                        segments.append(t)
                    segment = []
                    event_index = t[-1] + 1
                else:
                    event_index += 1

        res = []
        cut_points = []
        for segment in segments:
            buf = np.zeros((self.data_shape[1],self.data_shape[0]))
            tx,ty,tt,cut_point = [],[],[],[]
            tx_acc,ty_acc = [],[]
            for i in segment:
                tx += x_list[start[i]:end[i]]
                ty += y_list[start[i]:end[i]]
                tt += time_list[start[i]:end[i]]
                tx_acc += x_speed_list[start[i]:end[i]]
                ty_acc += y_speed_list[start[i]:end[i]]
                cut_point.append(len(tx))
            length = len(tx)
            buf[0,:length] = tx
            buf[1,:length] = ty
            buf[2,:length] = tx_acc
            buf[3,:length] = ty_acc
            cut_points.append(cut_point)
            res.append((buf.T,self.cal_features(tx,ty,tt,len(segment))))

        t_movement = 0
        for i in range(len(start)):
            t_movement += np.sum(time_list[start[i]:end[i]])
        if not user_id in self.mouse_movement_time.keys():
            self.mouse_movement_time[user_id] = [t_movement]
        else:
            self.mouse_movement_time[user_id].append(t_movement)

        return res,cut_points

    def cal_features(self,dx,dy,dt,n_events):
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
            res.append(self.log_smooth(np.mean(time_sery)))
            res.append(self.log_smooth(np.std(time_sery)))
            res.append(self.log_smooth(min(time_sery)))
            res.append(self.log_smooth(max(time_sery)))
        res.append(self.log_smooth(np.sum(dt)))#28
        res.append(self.log_smooth(np.sum(s)))#29
        tx, ty = np.sum(dx), np.sum(dy)
        dist = math.sqrt(tx**2 + ty**2)
        res.append(self.log_smooth(dist))#30
        res.append(self.log_smooth(dist/(np.sum(s)+1)))#31
        res.append(self.log_smooth(n_events))#32
        res.append(self.log_smooth(np.sum(angle)))#33
        res.append(self.log_smooth(np.sum(np.array(curvature) < 0.0005)))#34
        acc_t = 0
        for i in range(len(acc)):
            if acc[i] > 0:
                acc_t += (dt[i]+dt[i+1])/2
        res.append(self.log_smooth(acc_t))#35

        return np.array(res)

if __name__ == '__main__':
    test = data_loader((256,4))
    train_segments,train_users = test.load()

    auth_t = []
    for i,user_segments in enumerate(train_segments):
        segments,_ = user_segments
        for segment in segments:
            t = 0
            for j in range(segment.shape[0]):
                if not segment[j,0] == 0:
                    t += segment[j,0]/segment[j,2] - 1
                elif not segment[j,1] == 0:
                    t += segment[j,1]/segment[j,3] - 1
            auth_t.append(t)




    '''
    print(test.event_lengths)
    key_list = list(test.event_lengths.keys())
    for key in key_list:
        if key > 128:
            test.event_lengths.pop(key, None)
    plt.bar(list(test.event_lengths.keys()), test.event_lengths.values(), color='g')
    plt.savefig("/data/rjin/mouse/figs/output.jpg")
    '''
from PIL import Image
import os
import numpy as np
import torch.utils.data as data
import cv2
import gc
import torch
import pandas as pd
import operator
from glob import glob

def read_dataset(anno_path):
    anno_dict = dict()
    max_len = dict()
    # Saliency4ASD has 300 images
    for i in range(1,301):
        img = cv2.imread(os.path.join(anno_path,'Images',str(i)+'.png'))
        y_lim, x_lim, _ = img.shape
        anno_dict[i] = dict()
        anno_dict[i]['img_size'] = [y_lim,x_lim]
        asd = pd.read_csv(os.path.join(anno_path,'ASD','ASD_scanpath_'+str(i)+'.txt'))
        ctrl = pd.read_csv(os.path.join(anno_path,'TD','TD_scanpath_'+str(i)+'.txt'))
        group_name = ['ctrl','asd']
        for flag, group in enumerate([ctrl, asd]):
            anno_dict[i][group_name[flag]] = dict()
            anno_dict[i][group_name[flag]]['fixation'] = []
            anno_dict[i][group_name[flag]]['duration'] = []
            cur_idx = list(group['Idx'])
            cur_x = list(group[' x'])
            cur_y = list(group[' y'])
            cur_dur = list(group[' duration'])
            tmp_fix = []
            tmp_dur = []
            for j in range(len(cur_idx)):
                # finish loading data for one subject
                if cur_idx[j] == 0  and j != 0:
                    anno_dict[i][group_name[flag]]['fixation'].append(tmp_fix)
                    anno_dict[i][group_name[flag]]['duration'].append(tmp_dur)
                    tmp_fix = []
                    tmp_dur = []
                tmp_fix.append([cur_y[j],cur_x[j]])
                tmp_dur.append(cur_dur[j])
            # save data of the last subject
            anno_dict[i][group_name[flag]]['fixation'].append(tmp_fix)
            anno_dict[i][group_name[flag]]['duration'].append(tmp_dur)

    return anno_dict

def loo_split(anno_dict,subj_id):
    train_dict = dict()
    val_dict = dict()
    # Saliency4ASD dataset has 14 Controls and 14 ASDs
    if subj_id >=14: # ctrl
        subj_id -= 14
        cur_group = 1
    else:
        cur_group = 0 # asd

    group_name = ['asd','ctrl']

    for k in anno_dict.keys():
        if subj_id+1 > len(anno_dict[k][group_name[cur_group]]['fixation']):
            train_dict[k] = anno_dict[k] # current image does not have enough data, thus skipped for splitting
        else:
            train_dict[k] = dict()
            val_dict[k] = dict()
            # contructing data for the opposite group (no need for splitting)
            train_dict[k]['img_size'] = val_dict[k]['img_size'] = anno_dict[k]['img_size']
            train_dict[k][group_name[1-cur_group]] = anno_dict[k][group_name[1-cur_group]]
            val_dict[k][group_name[1-cur_group]] = dict()
            val_dict[k][group_name[1-cur_group]]['fixation'] = []
            val_dict[k][group_name[1-cur_group]]['duration'] = []

            # constructing data for the current group (split into train/val for leave-one-out validation)
            train_dict[k][group_name[cur_group]] = dict()
            val_dict[k][group_name[cur_group]] = dict()

            # splitting based on the relative position of the hold-out subjects
            if subj_id+1 == len(anno_dict[k][group_name[cur_group]]['fixation']):
                train_dict[k][group_name[cur_group]]['fixation'] = anno_dict[k][group_name[cur_group]]['fixation'][:subj_id]
                train_dict[k][group_name[cur_group]]['duration'] = anno_dict[k][group_name[cur_group]]['duration'][:subj_id]
            else:
                left_fix = anno_dict[k][group_name[cur_group]]['fixation'][:subj_id]
                right_fix = anno_dict[k][group_name[cur_group]]['fixation'][(subj_id+1):]
                if len(left_fix)>0 and not isinstance(left_fix[0],list):
                    left_fix = [left_fix]
                if len(right_fix)>0 and not isinstance(right_fix[0],list):
                    right_fix = [right_fix]

                left_dur = anno_dict[k][group_name[cur_group]]['duration'][:subj_id]
                right_dur = anno_dict[k][group_name[cur_group]]['duration'][(subj_id+1):]
                if len(left_dur)>0 and not isinstance(left_dur[0],list):
                    left_dur = [left_dur]
                if len(right_dur)>0 and not isinstance(right_dur[0],list):
                    right_dur = [right_dur]

                train_dict[k][group_name[cur_group]]['fixation'] = left_fix + right_fix
                train_dict[k][group_name[cur_group]]['duration'] = left_dur + right_dur

            val_dict[k][group_name[cur_group]]['fixation'] = [anno_dict[k][group_name[cur_group]]['fixation'][subj_id]]
            val_dict[k][group_name[cur_group]]['duration'] = [anno_dict[k][group_name[cur_group]]['duration'][subj_id]]

    return train_dict,val_dict



def image_selection(train_set,select_number=100):
    fisher_score = dict()
    for img in train_set.keys():
        asd_fix = train_set[img]['asd']['fixation']
        asd_dur = train_set[img]['asd']['duration']
        ctrl_fix = train_set[img]['ctrl']['fixation']
        ctrl_dur = train_set[img]['ctrl']['duration']
        img_size = train_set[img]['img_size']
        stat = [[] for _ in range(2)]

        # calculate the fisher score to select discriminative images
        for group, data in enumerate([(asd_fix,asd_dur),(ctrl_fix,ctrl_dur)]):
            cur_fix, cur_dur = data
            for i in range(len(cur_fix)):
                for j in range(len(cur_fix[i])):
                    y, x = cur_fix[i][j]
                    dist = np.sqrt((y-img_size[0]/2)**2 + (x-img_size[1]/2)**2)
                    dur = cur_dur[i][j]
                    stat[group].append([y,x,dist,dur])
        pos = np.array(stat[0])
        neg = np.array(stat[1])
        fisher = (np.mean(pos,axis=0)-np.mean(neg,axis=0))**2 / (np.std(pos,axis=0)**2 + np.std(pos,axis=0)**2) # fisher score
        fisher_score[img] = np.mean(fisher)

    # selecting the images by fisher score
    sorted_score = sorted(fisher_score.items(),key=operator.itemgetter(1))
    sorted_score.reverse()
    selected_img = []
    for i in range(select_number):
        selected_img.append(sorted_score[i][0])

    return selected_img

class Dataset(data.Dataset):
    def __init__(self,img_dir,data,valid_id,max_len,img_height,img_width,transform):
        self.img_dir = img_dir
        self.initial_dataset(data,valid_id)
        self.max_len = max_len
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform

    def initial_dataset(self,data,valid_id):
        self.fixation = []
        self.label = []
        self.img_id = []
        self.img_size = []

        for img_id in data.keys():
            if not img_id in valid_id:
                continue
            for group_label, group in enumerate(['ctrl','asd']):
                self.fixation.extend(data[img_id][group]['fixation'])
                self.img_id.extend([os.path.join(self.img_dir,str(img_id)+'.png')]*len(data[img_id][group]['fixation']))
                self.label.extend([group_label]*len(data[img_id][group]['fixation']))
                self.img_size.extend([data[img_id]['img_size']]*len(data[img_id][group]['fixation']))

    def get_fix(self,idx):
        fixs = self.fixation[idx]
        y_lim, x_lim = self.img_size[idx]
        fixation = []
        invalid = 0
        # only consider the first k fixations
        for i in range(self.max_len):
            if i+1 <= len(fixs):
                y_fix, x_fix = fixs[i]
                x_fix = int(x_fix*(self.img_width/float(x_lim)))/32
                y_fix = int(y_fix*(self.img_height/float(y_lim)))/33
                if x_fix >=0 and y_fix>=0:
                    fixation.append(y_fix*25 + x_fix) # get the corresponding index of fixation on the downsampled feature map
                else:
                    invalid += 1
            else:
                fixation.append(0) # pad if necessary
        for i in range(invalid):
            fixation.append(0)
        fixation = torch.from_numpy(np.array(fixation).astype('int'))
        return fixation

    def __getitem__(self,index):
        img = Image.open(self.img_id[index])
        if self.transform is not None:
            img = self.transform(img)
        label = torch.FloatTensor([self.label[index]])
        fixation = self.get_fix(index)
        return img, label, fixation

    def __len__(self,):
        return len(self.fixation)

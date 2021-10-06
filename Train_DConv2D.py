#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 5/8/2020 12:04 AM
# @Author : Zhicheng Zhang
# @E-mail : zhicheng0623@gmail.com
# @Site :
# @File : Train_main.py
# @Software: PyCharm
########################################################################################################################
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'utils'))

import warnings

warnings.filterwarnings("ignore")

import datetime, time
import utils.ckpt as ckpt
import random
import copy
from ops import *

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# from imblearn.pipeline import make_pipeline
# from imblearn.over_sampling import SMOTE
# from imblearn.combine import SMOTEENN, SMOTETomek


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'


from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

Ix = Iy = 160

aug = 6


class model(object):
    def __init__(self):
        # self.wint = tf.random_normal_initializer(mean=0.0, stddev=0.005, seed=2)
        self.wint = tf.contrib.layers.xavier_initializer()  #

        self.param = {}
        self.param['retrain'] = True

        self.param['model_save_path'] = os.path.join('./Results', os.path.basename(__file__).split('.')[0], 'model')
        self.param['tensorboard_save_logs'] = os.path.join('./logs', os.path.basename(__file__).split('.')[0])

    def _l2normalize(self, v, eps=1e-12):
        with tf.name_scope('l2normalize'):
            return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

    def global_avg_pooling(self, x):
        gap = tf.reduce_mean(x, axis=[1, 2])
        return gap

    def DynamicConv2D(self, x, out_plane=32, k=8, name='DConv2D'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            att = self.global_avg_pooling(x)
            att = tf.layers.dense(att, 2 * k, activation=tf.nn.leaky_relu)
            att = tf.layers.dense(att, k)

            att = tf.nn.softmax(att/0.7)


            feature = att[:, 0] * tf.transpose(tf.layers.conv2d(x, filters=out_plane, kernel_size=[3, 3], padding='SAME'), perm=[1, 2, 3, 0])
            for i in range(1, k):
                feature += att[:, i] * tf.transpose(tf.layers.conv2d(x, filters=out_plane, kernel_size=[3, 3], padding='SAME'), perm=[1, 2, 3, 0])

            feature = tf.transpose(feature, perm=[3, 0, 1, 2])

        return feature

    def swish(self, x):
        return x * tf.nn.sigmoid(x)



    def cox_loss(self, score, time_value, event):
        '''
        Args
            score: 		predicted survival time_value, tf tensor of shape (None, 1)
            time_value:		true survival time_value, tf tensor of shape (None, )
            event:		event, tf tensor of shape (None, )
        Return
            loss:		partial likelihood of cox regression
        '''

        ## cox regression computes the risk score, we want the opposite
        score   = -score

        ## find index i satisfying event[i]==1
        ix      = tf.where(tf.cast(event, tf.bool))  # shape of ix is [None, 1]

        ## sel_mat is a matrix where sel_mat[i,j]==1 where time_value[i]<=time_value[j]
        sel_mat = tf.cast(tf.gather(time_value, ix) <= time_value, tf.float32)

        ## formula: \sum_i[s_i-\log(\sum_j{e^{s_j}})] where time_value[i]<=time_value[j] and event[i]==1
        p_lik   = tf.gather(score, ix) - tf.log(tf.reduce_sum(sel_mat * tf.transpose(tf.exp(score)), axis=-1))
        # p_lik = tf.gather(score, ix) - tf.log(tf.reduce_sum(tf.transpose(tf.exp(score)), axis=-1))
        loss    = -tf.reduce_mean(p_lik)

        return loss

    def hinge_loss(self, score, time_value, event):
        '''
        Args
        score:	 	predicted score, tf tensor of shape (None, 1)
        time_value:		true survival time_value, tf tensor of shape (None, )
        event:		event, tf tensor of shape (None, )
        '''
        ## find index pairs (i,j) satisfying time_value[i]<time_value[j] and event[i]==1
        ix = tf.where(tf.logical_and(tf.expand_dims(time_value, axis=-1) < time_value,
                                     tf.expand_dims(tf.cast(event, tf.bool), axis=-1)), name='ix')
        ## if score[i]>score[j], incur hinge loss
        s1 = tf.gather(score, ix[:, 0])
        s2 = tf.gather(score, ix[:, 1])
        loss = tf.reduce_mean(tf.maximum(1 + s1 - s2, 0.0), name='loss')

        return loss

    def log_loss(self, score, time_value, event):
        '''
        Args
        score: 	predicted survival time_value, tf tensor of shape (None, 1)
        time_value:		true survival time_value, tf tensor of shape (None, )
        event:		event, tf tensor of shape (None, )
        '''
        ## find index pairs (i,j) satisfying time_value[i]<time_value[j] and event[i]==1
        ix = tf.where(tf.logical_and(tf.expand_dims(time_value, axis=-1) < time_value,
                                     tf.expand_dims(tf.cast(event, tf.bool), axis=-1)), name='ix')
        ## if score[i]>score[j], incur log loss
        s1 = tf.gather(score, ix[:, 0])
        s2 = tf.gather(score, ix[:, 1])
        loss = tf.reduce_mean(tf.log(1 + tf.exp(s1 - s2)), name='loss')
        return loss

    def __concordance_index(self, score, time_value, event):
        '''
        Args
            score: 		predicted score, tf tensor of shape (None, )
            time_value:		true survival time_value, tf tensor of shape (None, )
            event:		event, tf tensor of shape (None, )
        '''

        ## find index pairs (i,j) satisfying time_value[i]<time_value[j] and event[i]==1
        ix = tf.where(tf.logical_and(tf.expand_dims(time_value, axis=-1) < time_value,
                                     tf.expand_dims(tf.cast(event, tf.bool), axis=-1)), name='ix')

        ## count how many score[i]<score[j]
        s1 = tf.gather(score, ix[:, 0])
        s2 = tf.gather(score, ix[:, 1])
        ci = tf.reduce_mean(tf.cast(s1 < s2, tf.float32), name='c_index')

        return ci

    def model(self, x, name='model'):
        k = 16
        # b,w,h,c=x.get_shape().as_list()
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            conv    = tf.layers.conv2d(x, filters=k, kernel_size=[3, 3], padding='SAME',activation=tf.nn.leaky_relu)
            conv1    = self.DynamicConv2D(conv, out_plane=k, k=8, name='DConv2D_10')

            x_2     = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2], padding='SAME')
            conv    = tf.layers.conv2d(x_2, filters=conv1.get_shape().as_list()[-1], kernel_size=[3, 3], padding='SAME',activation=tf.nn.leaky_relu)
            q       = tf.concat([conv, tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=[2, 2], padding='SAME')],axis=-1)
            conv2    = self.DynamicConv2D(q, out_plane=q.get_shape().as_list()[-1], k=8, name='DConv2D_20')

            x_3     = tf.layers.max_pooling2d(x, pool_size=[4, 4], strides=[4, 4], padding='SAME')
            conv    = tf.layers.conv2d(x_3, filters=conv2.get_shape().as_list()[-1], kernel_size=[3, 3], padding='SAME',activation=tf.nn.leaky_relu)
            q       = tf.concat([conv, tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=[2, 2], padding='SAME')],axis=-1)
            conv3    = self.DynamicConv2D(q, out_plane=q.get_shape().as_list()[-1], k=8, name='DConv2D_30')


            x_4     = tf.layers.max_pooling2d(x, pool_size=[8, 8], strides=[8, 8], padding='SAME')
            conv    = tf.layers.conv2d(x_4, filters=conv3.get_shape().as_list()[-1], kernel_size=[3, 3], padding='SAME', activation=tf.nn.leaky_relu)
            q       = tf.concat([conv, tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=[2, 2], padding='SAME')],axis=-1)
            conv4    = self.DynamicConv2D(q, out_plane=q.get_shape().as_list()[-1], k=8, name='DConv2D_40')


            x_5     = tf.layers.max_pooling2d(x, pool_size=[16, 16], strides=[16, 16], padding='SAME')
            conv    = tf.layers.conv2d(x_5, filters=conv4.get_shape().as_list()[-1], kernel_size=[3, 3], padding='SAME',activation=tf.nn.leaky_relu)
            q       = tf.concat([conv, tf.layers.max_pooling2d(conv4, pool_size=[2, 2], strides=[2, 2], padding='SAME')],axis=-1)
            conv5    = self.DynamicConv2D(q, out_plane=q.get_shape().as_list()[-1], k=8, name='DConv2D_50')


            p1 = self.global_avg_pooling(conv1)  # k 
            p2 = self.global_avg_pooling(conv2)  #  2k
            p3 = self.global_avg_pooling(conv3)  #  4k
            p4 = self.global_avg_pooling(conv4)  #  8k
            p5 = self.global_avg_pooling(conv5)  #  16k

            p = tf.concat([p1, p2, p3, p4, p5], axis=-1)
            feature = tf.layers.flatten(p)

        with tf.variable_scope('PM', reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(feature, 128, kernel_initializer=self.wint)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.dropout(x, rate=self.dopro)
            logit = tf.layers.dense(x, 1, activation=tf.nn.sigmoid, kernel_initializer=self.wint)
        with tf.variable_scope('survival', reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(feature, 128, kernel_initializer=self.wint)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.dropout(x, rate=self.dopro)
            s = tf.layers.dense(x, 1, activation=None, kernel_initializer=self.wint)
            return feature, logit, s

    def train(self):
        n_epochs = 20000
        checkpointdir = self.param['model_save_path']
        ####################################################################################
        if not os.path.exists(checkpointdir):
            checkpoints_dir = checkpointdir
            os.makedirs(checkpoints_dir)
        else:
            current_time    = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            checkpoints_dir = checkpointdir.format(current_time)
        ####################################################################################
        # Read the filename of Training dataset
        data_zd   = np.load('./DataSet/ROI_Zhongda_160re.npy',   allow_pickle=True)
        data_nf_1 = np.load('./DataSet/ROI_nanfang_1_160re.npy', allow_pickle=True)  
        data_nf_0 = np.load('./DataSet/ROI_nanfang_0_160re.npy', allow_pickle=True)  

        train_data_0 = data_nf_0[0:np.uint16(1. * len(data_nf_0))]
        train_data_1 = data_nf_1[0:np.uint16(1. * len(data_nf_1))]

        self.param['batchsize'] = 128  # np.uint16(len(train_data_0))
        self.param['n_positve'] = 1

        ##################################################################
        train_img_0_WO_aug  = np.empty([np.uint16(len(train_data_0)), 160, 160, 3], dtype=np.float32)
        train_PM_0_WO_aug   = np.empty([np.uint16(len(train_data_0)), 1], dtype=np.float32)
        train_DFS_0_WO_aug  = np.empty([np.uint16(len(train_data_0))], dtype=np.float32)
        train_st_0_WO_aug   = np.empty([np.uint16(len(train_data_0))], dtype=np.float32)

        train_img_1_WO_aug  = np.empty([np.uint16(len(train_data_1)), 160, 160, 3], dtype=np.float32)
        train_PM_1_WO_aug   = np.empty([np.uint16(len(train_data_1)), 1], dtype=np.float32)
        train_DFS_1_WO_aug  = np.empty([np.uint16(len(train_data_1))], dtype=np.float32)
        train_st_1_WO_aug   = np.empty([np.uint16(len(train_data_1))], dtype=np.float32)

        for i in range(np.uint16(len(train_data_0))):
            train_img_0_WO_aug[i, :, :, :]  = train_data_0[i]['img']
            train_PM_0_WO_aug[i, :]         = train_data_0[i]['PM'][0]
            train_DFS_0_WO_aug[i]           = train_data_0[i]['DFS'][0]
            train_st_0_WO_aug[i]            = train_data_0[i]['st'][0]
        for i in range(np.uint16(len(train_data_1))):
            train_img_1_WO_aug[i, :, :, :]  = train_data_1[i]['img']
            train_PM_1_WO_aug[i, :]         = train_data_1[i]['PM'][0]
            train_DFS_1_WO_aug[i]           = train_data_1[i]['DFS'][0]
            train_st_1_WO_aug[i]            = train_data_1[i]['st'][0]

        train_img   = np.concatenate([train_img_0_WO_aug, train_img_1_WO_aug], axis=0)
        train_PM    = np.concatenate([train_PM_0_WO_aug, train_PM_1_WO_aug], axis=0)
        train_DFS   = np.concatenate([train_DFS_0_WO_aug, train_DFS_1_WO_aug], axis=0)
        train_st    = np.concatenate([train_st_0_WO_aug, train_st_1_WO_aug], axis=0)

        ##################################################################
        valid_data  = data_zd  
        valid_img   = np.empty([np.uint16(len(valid_data)), 160, 160, 3], dtype=np.float32)
        valid_PM    = np.empty([np.uint16(len(valid_data)), 1], dtype=np.float32)
        valid_DFS   = np.empty([np.uint16(len(valid_data))], dtype=np.float32)
        valid_st    = np.empty([np.uint16(len(valid_data))], dtype=np.float32)

        for i in range(np.uint16(len(valid_data))):
            valid_img[i, :, :, :]   = valid_data[i]['img']
            valid_PM[i, :]          = valid_data[i]['PM'][0]
            valid_DFS[i]            = valid_data[i]['DFS'][0]
            valid_st[i]             = valid_data[i]['st'][0]

        #####################################################################################

        graph = tf.Graph()
        with graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            ####################################################################################
            self.x_1   = tf.placeholder(tf.float32, [None, 160, 160, 3])
            self.PM_1  = tf.placeholder(tf.float32, [None, 1])
            self.DFS_1 = tf.placeholder(tf.float32, [None])
            self.st_1  = tf.placeholder(tf.float32, [None])

            self.x_1_n   = tf.placeholder(tf.float32, [self.param['n_positve'], 160, 160, 3])
            self.PM_1_1  = tf.placeholder(tf.float32, [None, 1])
            self.DFS_1_1 = tf.placeholder(tf.float32, [None])
            self.st_1_1  = tf.placeholder(tf.float32, [None])

            self.x_0   = tf.placeholder(tf.float32, [self.param['batchsize'], 160, 160, 3])
            self.PM_0  = tf.placeholder(tf.float32, [self.param['batchsize'], 1])
            self.DFS_0 = tf.placeholder(tf.float32, [self.param['batchsize']])
            self.st_0  = tf.placeholder(tf.float32, [self.param['batchsize']])

            self.is_training = tf.placeholder(tf.bool)
            self.dopro       = tf.placeholder(tf.float32)

            ####################################################################################

            features_1,   logit_1,   s_1   = self.model(self.x_1, name='model')
            features_0,   logit_0,   s_0   = self.model(self.x_0, name='model')
            features_1_1, logit_1_1, s_1_1 = self.model(self.x_1_n, name='model')

            ####################################################################################

            features_1_1 = features_1_1 / tf.tile(tf.expand_dims(tf.reduce_sum(features_1_1 ** 2, axis=1) ** 0.5 + 1e-6, axis=-1),[1, features_1_1.get_shape().as_list()[-1]])
            features_1   = features_1   / tf.tile(tf.expand_dims(tf.reduce_sum(features_1 ** 2,   axis=1) ** 0.5 + 1e-6, axis=-1),[1, features_1.get_shape().as_list()[-1]])
            features_0   = features_0   / tf.tile(tf.expand_dims(tf.reduce_sum(features_0 ** 2,   axis=1) ** 0.5 + 1e-6, axis=-1),[1, features_0.get_shape().as_list()[-1]])

            cos_1_1   = tf.matmul(features_1, features_1_1, transpose_b=True)
            cos_1_0   = tf.matmul(features_1, features_0, transpose_b=True)
            cos_1_1_0 = tf.matmul(features_1_1, features_0, transpose_b=True)

            feature_loss = -tf.log(tf.reduce_sum(tf.exp(cos_1_1 / 0.7)) / (tf.reduce_sum(tf.exp(cos_1_1 / 0.7)) + tf.reduce_sum(tf.exp(cos_1_0 / 0.7))))

            logit_all    = tf.concat([logit_1, logit_0, logit_1_1], axis=0)
            PM_all       = tf.concat([self.PM_1, self.PM_0, self.PM_1_1], axis=0)

            disc_loss    = tf.reduce_sum(-PM_all * tf.log(logit_all + 0.0001) / self.param['n_positve'] - (1 - PM_all) * tf.log(1 - logit_all + 0.0001) / self.param['batchsize'])

            score        = tf.concat([s_1, s_0, s_1_1], axis=0)
            time_value   = tf.concat([self.DFS_1, self.DFS_0, self.DFS_1_1], axis=0)
            event        = tf.concat([self.st_1, self.st_0, self.st_1_1], axis=0)

            # hinge_loss     log_loss   cox_loss
            cox_value    = self.cox_loss(score, time_value, event)

            loss         = feature_loss + disc_loss + 0.2 * cox_value

            ci           = self.__concordance_index(s_1, self.DFS_1, self.st_1)
            ####################################################################################

            global_step  = tf.Variable(0)

            self.ADAM_opt = tf.train.AdamOptimizer(0.00001).minimize(loss, global_step=global_step)

            ####################################################################################


            ####################################################################################
            with tf.Session(config=config, graph=graph) as sess:
                sess.run(tf.global_variables_initializer())
                ####################################################################################
                # Read the existed model
                if not self.param['retrain']:
                    checkpoint      = tf.train.get_checkpoint_state(checkpoints_dir)
                    meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
                    ckpt.load_ckpt(sess=sess, save_dir=checkpoints_dir, is_latest=True)
                    epoch_pre       = int(meta_graph_path.split("-")[1].split(".")[0])
                else:
                    sess.run(tf.global_variables_initializer())
                    epoch_pre       = 0
                ####################################################################################

                train_taget_all = np.zeros((self.param['batchsize'] * (np.shape(train_img)[0] // self.param['batchsize']), 1),dtype=np.float32)
                train_pprob_all = np.zeros((self.param['batchsize'] * (np.shape(train_img)[0] // self.param['batchsize']), 1),dtype=np.float32)

                train_final_input_n = np.empty(shape=(self.param['n_positve'], 160, 160, 3), dtype=np.float32)
                train_final_PM_1_n  = np.empty(shape=(self.param['n_positve'], 1), dtype=np.float32)
                train_final_DFS_1_n = np.empty(shape=(self.param['n_positve']), dtype=np.float32)
                train_final_st_1_n  = np.empty(shape=(self.param['n_positve']), dtype=np.float32)

                ####################################################################################
                for epoch in range(epoch_pre, n_epochs):
                    ####################################################################################
                    #
                    #  Training stage
                    #
                    ####################################################################################
                    ploss_all, pfea, pdis, pcox_value_all = 0, 0, 0, 0
                    for iq in range(np.shape(train_img_1_WO_aug)[0]):
                        Train_Num = [i for i in range(np.shape(train_img_0_WO_aug)[0])]
                        random.shuffle(Train_Num)

                        Train_Num1 = [i for i in range(np.shape(train_img_1_WO_aug)[0])]
                        random.shuffle(Train_Num1)

                        train_final_input_1 = np.expand_dims(train_img_1_WO_aug[iq, :, :, :], axis=0)
                        train_final_PM_1    = np.expand_dims(train_PM_1_WO_aug[iq, :], axis=0)
                        train_final_DFS_1   = np.expand_dims(train_DFS_1_WO_aug[iq], axis=0)
                        train_final_st_1    = np.expand_dims(train_st_1_WO_aug[iq], axis=0)

                        train_final_input_0 = train_img_0_WO_aug[Train_Num[0:self.param['batchsize']], :, :, :]
                        train_final_PM_0    = train_PM_0_WO_aug[Train_Num[0:self.param['batchsize']], :]
                        train_final_DFS_0   = train_DFS_0_WO_aug[Train_Num[0:self.param['batchsize']]]
                        train_final_st_0    = train_st_0_WO_aug[Train_Num[0:self.param['batchsize']]]

                        train_final_input_n = train_img_1_WO_aug[Train_Num1[0:np.shape(train_final_input_n)[0]], :, :,:]
                        train_final_PM_1_n  = train_PM_1_WO_aug[Train_Num1[0:np.shape(train_final_input_n)[0]], :]
                        train_final_DFS_1_n = train_DFS_1_WO_aug[Train_Num1[0:np.shape(train_final_input_n)[0]]]
                        train_final_st_1_n  = train_st_1_WO_aug[Train_Num1[0:np.shape(train_final_input_n)[0]]]

                        feed_dict = {self.x_1: train_final_input_1, self.PM_1: train_final_PM_1,
                                     self.DFS_1: train_final_DFS_1, self.st_1: train_final_st_1,
                                     self.x_1_n: train_final_input_n, self.PM_1_1: train_final_PM_1_n,
                                     self.DFS_1_1: train_final_DFS_1_n, self.st_1_1: train_final_st_1_n,
                                     self.x_0: train_final_input_0, self.PM_0: train_final_PM_0,
                                     self.DFS_0: train_final_DFS_0, self.st_0: train_final_st_0,
                                     self.dopro: 0.5}

                    if (epoch + 1) % 1 == 0:
                        var_list = [var for var in tf.global_variables()]
                        print(time.asctime(time.localtime(time.time())),
                              'the %d-th iterations. Saving Models...' % epoch)
                        ckpt.save_ckpt(sess=sess, mode_name='model.ckpt', save_dir=checkpoints_dir, global_step=epoch,
                                       var_list=var_list)
                        print("[*] Saving checkpoints SUCCESS! ")


if __name__ == '__main__':
    srcnn = model()
    srcnn.train()


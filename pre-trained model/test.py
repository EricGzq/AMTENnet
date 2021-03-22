# -*- coding: utf-8 -*-
import os,math
import sys
import caffe
import cv2
import numpy as np
from numpy import *
import lmdb
from sklearn import svm

caffe_root = '/home/caffegpu/caffe-master'

MODEL_FILE = '/home/caffegpu/Desktop/pre-trained model/deploy_AMTENnet.prototxt'
PRETRAINED = '/home/caffegpu/Desktop/pre-trained model/c23.caffemodel'

net = caffe.Net(MODEL_FILE, PRETRAINED,caffe.TEST)
caffe.set_device(0)
caffe.set_mode_gpu()


lmdb_env = lmdb.open('/home/newdisk2/caffegpu/p2-CVIU/Binary_classification/FF++/c23/test_lmdb/')

lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
count = 0
correct = 0

p_y = []
t_y = []
feat_tt = []

fp = []
tp = []

for key, value in lmdb_cursor:
        print "Count:"
        print count
        count = count + 1
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        t_y.append(int(datum.label))
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)
        im = image
        out = net.forward_all(data=np.asarray([im]))
        p_y.append(out['prob'][0].argmax(axis=0))
        feat_tt.append(net.blobs['fc_net_2'].data[0].tolist())
        print("Label is class " + str(int(datum.label)) + ", predicted class is " + str(out['prob'][0].argmax(axis=0)))
        # prepare for auc
        fp.append(int(datum.label))
        tp.append(out['prob'][0].argmax(axis=0))


# Softmax confusion matrix and testing accuracy
from sklearn.metrics import confusion_matrix

cmat = confusion_matrix(t_y, p_y)
nbr = cmat.sum(1)
nbr = np.array(nbr, dtype = 'f')
M = cmat/nbr
np.set_printoptions(suppress=True)
M = np.around(M*100, decimals=2)
print M
binary = [t_y[i]==p_y[i] for i in range(len(p_y))]
acc = binary.count(True)/float(count)

print 'The testing accuracy is ' + str(acc)

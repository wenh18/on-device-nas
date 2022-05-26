import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
import csv
import pickle
import sys
import torch

def batch_read(filename, batch_size=1024):
    # open the data stream
    features = torch.load(filename).cpu()
    datalen = features.shape[0]
    # reset the batch
    # 充当数据缓存的角色，每一批缓存batch_size条数据
    batch_img_output = []
    batch_output  = []
    # iterate over the file
    for (i) in range(datalen):
      # if the batch is of the right size
      # 当填满一个batch时,yield这个batch
      if i > 0 and i % batch_size == 0:
        # yield back the batch as an ndarray
        # 填满batch后才yield
        yield(batch_output,np.array(batch_img_output))
        # 重置batch
        batch_img_output = list() 
        batch_output = list()                 
      batch_img_output.append(np.array(features[i]))
      batch_output.append(i) 
        # when the loop is over, yield what's left
        # 在最后yield剩余的数据
        # 只有在没有填满一个batch的情况下才会执行此条语句
    yield(batch_output,np.array(batch_img_output))

def batch_img_read(filename, batch_size=1024):
    # open the data stream
    features = torch.load(filename).cpu()
    datalen = features.shape[0]
    # reset the batch
    # 充当数据缓存的角色，每一批缓存batch_size条数据
    batch_img_output = []
    # iterate over the file
    for (i) in range(datalen):
      # if the batch is of the right size
      # 当填满一个batch时,yield这个batch
      if i > 0 and i % batch_size == 0:
        # yield back the batch as an ndarray
        # 填满batch后才yield
        yield(np.array(batch_img_output))
        # 重置batch
        batch_img_output = list()                  
      batch_img_output.append(np.array(features[i]))  
        # when the loop is over, yield what's left
        # 在最后yield剩余的数据
        # 只有在没有填满一个batch的情况下才会执行此条语句
    yield(np.array(batch_img_output))

def ipca(input_data_file,save_data_file,train=False):
  print("ipca start...")
  ##分批读入训练数据
  batch_size = 1000
  fit_iter = 0
  if train:
    ipca = IncrementalPCA(n_components=256, batch_size=batch_size)
    for batch_img_output in batch_img_read(input_data_file, batch_size=batch_size):
      if (batch_img_output.shape[0] == batch_size):
          ipca.partial_fit(batch_img_output)
          fit_iter += 1
          if (fit_iter % 1 == 0):
              print('IPCA fit_iter:{}'.format(fit_iter))
    ##把PCA参数进行保存
    with open('/root/resnet20/tmp/pca.pkl', 'wb') as pickle_file:
        pickle.dump(ipca, pickle_file)


  ##分批读入数据进行转换
  dest = open(save_data_file, "w")
  with open('/root/resnet20/tmp/pca.pkl', 'rb') as pickle_file:
      pca = pickle.load(pickle_file)
      trans_iter=0
      for batch_output,batch_img_output in batch_read(input_data_file, batch_size=1000):
          X_r = pca.transform(batch_img_output)
          X_l = X_r.tolist()
          for i,x in enumerate(batch_output):
              dest.write(' '.join(str(j) for j in X_l[i] ) + '\n')  
          trans_iter +=1 
          if (trans_iter % 1 == 0):
            print('IPCA trans_iter:{}'.format(trans_iter))
  dest.close()
  print("ipca done.")


# input_data_file="/root/resnet20/tools/features_test.pth"
# save_data_file="/root/resnet20/tools/pca_test.tsv"
# ipca(input_data_file,save_data_file,train=True)
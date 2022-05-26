from traceback import print_tb
import numpy as np
import sys
import matplotlib.pyplot as plt

def generate_idxs(batchsize,num_class=10,train=False):
    train_list = [0,5500, 11000, 15500, 18000, 20500, 24500, 32000, 39000, 44500, 50000]
    test_list = [0,1000, 2100, 2700, 3200, 3900, 5100, 6400, 7900, 8800, 10000]
    # train_list = [i*500 for i in range(num_class+1)]
    # test_list = [i*100 for i in range(num_class+1)]
    data_idxs = []
    k = 1
    if train:              
        index = 0
        while(index < train_list[-1]):
            if index + batchsize >= train_list[-1]:
                    batchsize = train_list[-1] - index
            #k =  np.random.randint(1,100)/10
            alpha = [k] * num_class
            label_p = np.random.dirichlet(alpha, 1)[0]
            #print('train p:{}'.format(label_p))
            for i in range(num_class):
                size = int(label_p[i] * batchsize + 1)
                index += size
                low = train_list[i]
                up = train_list[i+1]
                #print('low:{} up:{}'.format(low,up))
                for j in range(size):
                    data_idxs.append(np.random.randint(low,up))
        while(len(data_idxs) > train_list[-1]):
            index = np.random.randint(0,train_list[-1])
            del data_idxs[index]
    else:
        index = 0
        while(index < test_list[-1]):
            if index + batchsize >= test_list[-1]:
                    batchsize = test_list[-1] - index
            alpha = [k] * num_class
            label_p = np.random.dirichlet(alpha, 1)[0]
            # print('test p:{}'.format(label_p))
            for i in range(num_class):
                size = int(label_p[i] * batchsize + 1)
                index += size
                low = test_list[i]
                up = test_list[i+1]
                #print('low:{} up:{}'.format(low,up))
                for j in range(size):
                    data_idxs.append(np.random.randint(low,up))
        while(len(data_idxs) > test_list[-1]):
            index = np.random.randint(0,test_list[-1])
            del data_idxs[index]
    return data_idxs

if __name__ == '__main__':
    for i in range(2):
        data_idxs= generate_idxs(train=False,batchsize=10000,num_class=10)
        print(data_idxs,end=',\n')
        print()
        print()
        print()

    # data_idxs,p= generate_idxs(train=False,batchsize=10000,num_class=100)
    # p = abs(np.sort(-p))
    # x = [i for i in range(100)]
    # plt.plot(x,p)
    # plt.show()
    
    
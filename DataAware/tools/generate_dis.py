import numpy as np
import sys
import matplotlib.pyplot as plt

def generate_idxs(train=False,batchsize=512,num_class=100):
    # train_list = [0, 5000, 8000, 19500, 23000, 25500, 30500, 36000, 40500, 44500, 50000]
    # test_list = [0, 1000, 1600, 4000, 4700, 5200, 6100, 7200, 8100, 8900, 10000]
    train_list = [i*500 for i in range(num_class+1)]
    test_list = [i*100 for i in range(num_class+1)]
    data_idxs = []
    if train:
        index = 0
        while(index < train_list[-1]):
            if index + batchsize >= train_list[-1]:
                    batchsize = train_list[-1] - index
            alpha = [1] * num_class
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
        batchsize = 10000
        index = 0
        while(index < test_list[-1]):
            if index + batchsize >= test_list[-1]:
                    batchsize = test_list[-1] - index
            alpha = [1] * num_class
            label_p = np.random.dirichlet(alpha, 1)[0]
            #print('test p:{}'.format(label_p))
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
    data_idxs,p = generate_idxs(train=True)
    p = abs(np.sort(-p))
    x = [i for i in range(100)]
    plt.plot(x,p)
    sys.exit(0)
    
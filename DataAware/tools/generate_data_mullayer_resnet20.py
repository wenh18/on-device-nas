from cProfile import label
import os
import sys
from grpc import protos
from sklearn.pipeline import FeatureUnion
import torchvision.datasets as datasets
from skimage import io
import torchvision as tv
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import os, sys
import pickle
import torchvision.models as model
sys.path.append(os.getcwd())
from mean_shift import mean_shift
from GaussianMixture import GM
from PCA import ipca
from kmeans import k_means
from network.resnet20_ori import cifar100_resnet20_ori
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


def get_vgg19_output(x,model):
    x = model.features(x)
    x = torch.flatten(x, 1)
    return x

def get_resnet_output(x, model):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.layer1(x)
    x = model.layer2(x)
    #x = model.layer3(x)
    x = x.view(x.size(0), -1)
    return x

def write_to_txt(content,file):
    f = open(file, "w")
    for ct in content:
        f.write(str(ct)+'\n')
    f.close()

def read_pca(file):
    print('read_pca...')
    data = []
    with open(file, 'r') as f: 
     for line in f:
        tmp = [float(x) for x in line.split()]
        data.append(tmp)
    data = np.array(data)
    return data


class SaveFeatureOutput:
    def __init__(self):
        self.outputs = []
 
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
 
    def clear(self):
        self.outputs = []

save_feature = SaveFeatureOutput()

def get_mullayer_feature(input,model):
    save_feature.clear()
    model(input)
    print('get_mullayer_feature')
    print(len(save_feature.outputs))
    return save_feature.outputs[0]


def load_data():
    normalize = transforms.Normalize(mean=[0.507, 0.4865, 0.4409],
                                     std=[0.2673, 0.2564, 0.2761])
    train_set = datasets.CIFAR100(root=r'/root/resnet20/cifar-100-python', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), download=True)
    test_set = datasets.CIFAR100(root=r'/root/resnet20/cifar-100-python', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), download=True)
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for i, (X, Y) in enumerate(train_set):  # 将train_set的数据和label读入列表
        train_data.append(np.array(X))
        train_label.append(np.array(Y))
    for i, (X, Y) in enumerate(test_set):  # 将train_set的数据和label读入列表
        test_data.append(np.array(X))
        test_label.append(np.array(Y))
    train_data = torch.from_numpy(np.array(train_data))
    train_label = torch.from_numpy(np.array(train_label))
    test_data = torch.from_numpy(np.array(test_data))
    test_label = torch.from_numpy(np.array(test_label))
    print('train:{}'.format(train_data.shape))
    print('train_label:{}'.format(train_label.shape))
    print('test:{}'.format(test_data.shape))
    print('test_label:{}'.format(test_label.shape))

    train_data = train_data.to(device)
    train_label = train_label.to(device)
    test_data = test_data.to(device)
    test_label = test_label.to(device)
    return train_data, train_label, test_data, test_label

def get_feature(data, model, batch_size = 10000):
    print('get_feature start...')

    data_len = data.shape[0]
    features = []
    index = 0
    while(index + batch_size <= data_len):
        feature = get_mullayer_feature(data[index:index+batch_size,:,:,:],model)
        features.append(feature.detach().cpu())
        save_feature.clear()
        index += batch_size
        del feature
    if index < data_len:
        feature = get_mullayer_feature(data[index:data_len,:,:,:],model)
        features.append(feature.detach().cpu())
        save_feature.clear()
        del feature
    features = torch.stack(features, dim=0)
    print('features tensor size:{}'.format(features.size()))
    #features = features.reshape(-1,features.size()[2] * features.size()[3] * features.size()[4])
    features = features.reshape(-1,4096)
    print('features shape:{}'.format(features.shape))
    print('get_feature done.')
    return features

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def eval_model(data,target,model,batch_size=10000):
    print('eval_model start...')
    data_len = data.shape[0]
    index = 0
    while(index + batch_size <= data_len):
        feature = model(data[index:index+batch_size,:,:,:])
        print(accuracy(feature,target[index:index+batch_size]),feature.shape[0])
        index += batch_size
        del feature
    if index < data_len:
        feature = model(data[index:data_len,:,:,:])
        print(accuracy(feature,target[index:data_len]),feature.shape[0])
        del feature
    print('eval_model done.')


def sort_by_kmeans(target,expert_label,expert_num=10):
    print('sort_by_kmeans start...')
    data_size = len(target)
    cnt = np.zeros((100,expert_num),dtype=int)
    for i in range(data_size):
        cnt[target[i]][expert_label[i]] += 1
    index = np.argmax(cnt, axis=1)

    print('cnt:{}'.format(cnt))
    ans = [0 for i in range(expert_num)]
    for i in range(100):
        ans[index[i]] += 1
    print(ans)

    label = np.zeros(data_size,dtype=int)
    for i in range(data_size):
        label[i] = index[target[i]]
    return label


def classify_cifar100():
    
    # pre_model = cifar100_resnet20_ori()
    # pre_dict = torch.load('/root/resnet20/model/premodel/cifar100_resnet20-23dac2f1.pth')
    # pre_model.load_state_dict(pre_dict)
    # pre_model.eval()
    # pre_model.to(device)
    # trarin_data, train_label, test_data,test_label = load_data()
    # eval_model(test_data,test_label,pre_model)
    # #获取feature特征
    # featur_layers = [
    #                 'layer3.0.downsample.1',
    #                 #'layer3.1.conv1',
    #                  #'layer3.1.conv2',
    #                   #'layer3.2.conv1',
    #                  #  'layer3.2.conv2',
    #                 ]

    # for name,layer in pre_model.named_modules():
    #     if name in featur_layers:
    #         featur_layer = name
    #         print(featur_layer)
    #         layer.register_forward_hook(save_feature)
    #         features_test = get_feature(test_data,pre_model)
    #         features_train = get_feature(trarin_data,pre_model)
    #         label = k_means(features_train,train=True)
    #         label = sort_by_kmeans(train_label,label,expert_num=10)
    #         ans = [0 for i in range(10)]
    #         for v in label:
    #             ans[v] += 1
    #         print('train_label:{}'.format(ans))
    #         #sys.exit(0)
    #         train_feature_label_path = '/root/resnet20/tmp_resnet20/train_' + featur_layer + '_label.txt'
    #         print('train_feature_label_path:{}'.format(train_feature_label_path))
    #         write_to_txt(label,train_feature_label_path)
    #         label = k_means(features_test,train=False)
    #         label = sort_by_kmeans(test_label,label,expert_num=10)
    #         ans = [0 for i in range(10)]
    #         for v in label:
    #             ans[v] += 1
    #         print('val_label:{}'.format(ans))
    #         test_feature_label_path = '/root/resnet20/tmp_resnet20/test_' + featur_layer + '_label.txt'
    #         print('test_feature_label_path:{}'.format(test_feature_label_path))
    #         write_to_txt(label,test_feature_label_path)
    # sys.exit()

    #将每张图片所属的簇添加到数据集  
    data_file = '/root/resnet20/cifar-100-python/cifar-100-python/train'
    save_file = '/root/resnet20/tmp_resnet20/expert_train'
    add_expert_label_to_dataset(data_file,save_file,train=True)
    data_file = '/root/resnet20/cifar-100-python/cifar-100-python/test'
    save_file = '/root/resnet20/tmp_resnet20/expert_test'
    add_expert_label_to_dataset(data_file,save_file,train=False)

    # root = '/root/resnet20/cifar-100-python'
    # cifar100_to_img_by_expertlabel(root)
    
    
def test():
    train_dis =[5500, 5500, 4500, 2500, 2500, 4000, 7500, 7000, 5500, 5500]
    #train_dis = [i*500 for i in range(100)]
    test_dis =[1000, 1100, 600, 500, 700, 1200, 1300, 1500, 900, 1200]
    #test_dis = [i*100 for i in range(100)]
    ans_train = 0
    ans_test = 0
    for i in range(10):
        ans_train += train_dis[i]
        train_dis[i] = ans_train
        ans_test += test_dis[i]
        test_dis[i] = ans_test
    print(train_dis)
    print(test_dis) 
    path = '/root/resnet20/tmp_resnet20/expert_test'
    with open(path, 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
    for k,v in entry.items():
        print(k)
    for i in test_dis:
        print('{} label:{}'.format(i,entry['layer32conv2'][i-1]))
    path = '/root/resnet20/tmp_resnet20/expert_train'
    with open(path, 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
    for k,v in entry.items():
        print(k)
    for i in train_dis:
        print('{} label:{}'.format(i,entry['layer32conv2'][i-1]))
    sys.exit(0)
    ans = [0 for i in range(10)]
    with open('/root/resnet20/tmp_resnet20/gs_train_label.txt', "r") as f:  # 打开文件
        data = f.readlines()  # 读取文件
        for d in data:
            d = d.split()[0]
            ans[(int(d))] += 1
    print(ans)
    ans = [0 for i in range(10)]
    with open('/root/resnet20/tmp_resnet20/gs_test_label.txt', "r") as f:  # 打开文件
        data = f.readlines()  # 读取文件
        for d in data:
            d = d.split()[0]
            ans[(int(d))] += 1
    print(ans)

def cifar100_to_img_by_expertlabel(root):

    character_train = [[] for i in range(100)]
    character_test = [[] for i in range(100)]

    train_set = tv.datasets.CIFAR100(root, train=True, download=True)
    test_set = tv.datasets.CIFAR100(root, train=False, download=True)
    expert_train_label = []
    expert_test_label = []
    trainset = []
    testset = []
    with open("/root/resnet20/tmp_resnet20/feature_train_label.txt", "r") as f:  
        data = f.readlines()  
        for d in data:
            d = d.split()[0]
            expert_train_label.append(int(d[1]))
    with open("/root/resnet20/tmp_resnet20/feature_test_label.txt", "r") as f:  
        data = f.readlines()  
        for d in data:
            d = d.split()[0]
            expert_test_label.append(int(d[1]))

    for i, (X, Y) in enumerate(train_set):  # 将train_set的数据和label读入列表
        trainset.append(list((np.array(X), np.array(Y), np.array(expert_train_label[i]))))
    for i, (X, Y) in enumerate(test_set):  # 将test_set的数据和label读入列表
        testset.append(list((np.array(X), np.array(Y), np.array(expert_test_label[i]))))
    for X, Y, l in trainset:
        character_train[Y].append(list((X, l)))  # 32*32*3
    for X, Y, l in testset:
        character_test[Y].append(list((X, l)))  # 32*32*3

    root = '/root/resnet20/tmp_resnet20/'
    os.mkdir(os.path.join(root, 'train_expert'))
    os.mkdir(os.path.join(root, 'test_expert'))
    for i in range(10):
        path = os.path.join(root, 'train_expert', str(i))
        os.mkdir(path)
    for i in range(10):
        path = os.path.join(root, 'test_expert', str(i))
        os.mkdir(path)
    for i, per_class in enumerate(character_train):
        for j, img in enumerate(per_class):
            character_path = os.path.join(root, 'train_expert', str(img[1]), str(i))
            if not os.path.exists(character_path):
                os.mkdir(character_path)
            img_path = character_path + '/' + str(j) + ".png"
            io.imsave(img_path, img[0])
    for i, per_class in enumerate(character_test):
        character_path = os.path.join(root, 'test_expert', str(i))
        os.mkdir(character_path)
        for j, img in enumerate(per_class):
            img_path = character_path + '/' + str(j) + ".png"
            io.imsave(img_path, img)
def get_expert_label(datafile):
    expert_label = []
    print(datafile)
    with open(datafile, "r") as f:  # 打开文件
        data = f.readlines()  # 读取文件
        for d in data:
            d = d.split()[0]
            expert_label.append(int(d))
    return expert_label
    
def add_expert_label_to_dataset(data_file,save_file,train=False):
    print('add_expert_label_to_dataset start...{}'.format(data_file))
    with open(data_file, 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
    if train:
        filename = 'train'
    else:
        filename = 'test'
    layer30downsample1 = get_expert_label(datafile='/root/resnet20/tmp_resnet20/'+ filename + '_layer3.0.downsample.1_label.txt')
    layer31conv1 = get_expert_label(datafile='/root/resnet20/tmp_resnet20/'+ filename + '_layer3.1.conv1_label.txt')
    layer31conv2 = get_expert_label(datafile='/root/resnet20/tmp_resnet20/'+ filename + '_layer3.1.conv2_label.txt')
    layer32conv1 = get_expert_label(datafile='/root/resnet20/tmp_resnet20/'+ filename + '_layer3.2.conv1_label.txt')
    layer32conv2 = get_expert_label(datafile='/root/resnet20/tmp_resnet20/'+ filename + '_layer3.2.conv2_label.txt')
    data = []
    data_size = len(entry['fine_labels'])
    for i in range(data_size):
        data.append(list((entry['coarse_labels'][i],entry['fine_labels'][i],entry['data'][i],layer30downsample1[i],
                                layer31conv1[i],layer31conv2[i],layer32conv1[i],layer32conv2[i])))

    if train:
        data.sort(key=lambda x:x[7])
    else:
        print('fine_labels')
        data.sort(key=lambda x:x[7])

    coarse_labels=[]
    fine_labels = []
    img = []
    layer30downsample1 = []
    layer31conv1 = []
    layer31conv2 = []
    layer32conv1 = []
    layer32conv2 = []

    for i in range(data_size):
        coarse_labels.append(data[i][0])
        fine_labels.append(data[i][1])
        img.append(data[i][2])
        layer30downsample1.append(data[i][3])
        layer31conv1.append(data[i][4])
        layer31conv2.append(data[i][5])
        layer32conv1.append(data[i][6])
        layer32conv2.append(data[i][7])

    img = np.array(img)
    entry['coarse_labels'] = coarse_labels
    entry['fine_labels'] = fine_labels
    entry['data'] = img
    entry['layer30downsample1'] = layer30downsample1
    entry['layer31conv1'] = layer31conv1
    entry['layer31conv2'] = layer31conv2
    entry['layer32conv1'] = layer32conv1
    entry['layer32conv2'] = layer32conv2
    with open(save_file, "wb") as f:  # 打开文件
        pickle.dump(entry, f)
    
    print('add_expert_label_to_dataset done.')
 


if __name__ == '__main__':
    # 将cifar100根据feature聚类, 使用resnet20 pretrain model
    classify_cifar100()
    test()

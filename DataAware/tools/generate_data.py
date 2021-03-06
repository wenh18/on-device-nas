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
import torchvision.models as models
sys.path.append(os.getcwd())
from mean_shift import mean_shift
from GaussianMixture import GM
from PCA import ipca
from kmeans import k_means
from network import resnet20_ori
from network.resnet50_ori import cifar100_resnet50_ori
from network import vgg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_vgg19_output(x,model):
    x = model.features(x)
    x = torch.flatten(x, 1)
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

features = SaveFeatureOutput()

def get_mullayer_feature(input,model):
    features.clear()
    print('get layer...')
    for name, layer in model.named_modules():
        if (('4.1' in name or '4.2' in name ) and 'conv' in name) or '4.0.conv3' in name:
            layer.register_forward_hook(features)
            print(name)
    model(input)
    
    return features.outputs


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
    for i, (X, Y) in enumerate(train_set):  # ???train_set????????????label????????????
        train_data.append(np.array(X))
        train_label.append(np.array(Y))
    for i, (X, Y) in enumerate(test_set):  # ???train_set????????????label????????????
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

def get_feature(data, model, batch_size = 500,layer_num=7):
    print('get_feature start...')
    data_len = data.shape[0]
    features = []
    index = 0
    while(index + batch_size <= data_len):
        feature = get_vgg19_output(data[index:index+batch_size,:,:,:],model)
        features.append(feature.detach().cpu())
        index += batch_size
        del feature
    if index < data_len:
        feature = get_vgg19_output(data[index:data_len,:,:,:],model)
        for i in range(layer_num):
            features[i].append(feature.detach().cpu())
        del feature

    features = torch.stack(features, dim=0)
    print('features tensor size:{}'.format(features.size()))
    features = features.reshape(-1,512)
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

def eval_model(data,target,model,batch_size=500):
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
    print(index)
    ans = [0 for i in range(expert_num)]
    for i in range(100):
        ans[index[i]] += 1
    print(ans)

    label = np.zeros((data_size),dtype=int)
    for i in range(data_size):
        label[i] = index[target[i]]
    return label

def get_resnet20_model():
    model = resnet20_ori.cifar100_resnet20_ori(pre_model=True)
    return model

def get_vgg19_model():
    model = getattr(vgg, f"{'cifar100'}_{'vgg19_bn'}")()
    model.load_state_dict(torch.load('/root/resnet20/model/premodel/cifar100_vgg19_bn-b98f7bd7.pt'))
    return model
def get_resnet50_model():
    models.resnet50

def classify_cifar100():
    
    pre_model = get_vgg19_model()
    pre_model.eval()
    pre_model.to(device)
    trarin_data, train_label, test_data,test_label = load_data()
    #eval_model(test_data,test_label,pre_model)
    #??????feature??????
    features_train = get_feature(trarin_data,pre_model)
    features_test = get_feature(test_data,pre_model)
    # torch.save(features_train,'/root/resnet20/tmp/features_train.pth')
    # torch.save(features_test,'/root/resnet20/tmp/features_test.pth')

    # input_data_file="/root/resnet20/tmp/features_train.pth"
    # save_data_file="/root/resnet20/tmp/pca_train_result.tsv"
    # ipca(input_data_file,save_data_file,train=True)
    # input_data_file="/root/resnet20/tmp/features_test.pth"
    # save_data_file="/root/resnet20/tmp/pca_test_result.tsv"
    # ipca(input_data_file,save_data_file,train=False)

    # pca_result_file="/root/resnet20/tmp/pca_train_result.tsv"
    # pca_result = read_pca(pca_result_file)

    #???????????????
    label = k_means(features_train,train=True)
    ans = [0 for i in range(10)]
    #cluster,labels = mean_shift(pca_result)
    #print('cluster:{}'.format(cluster))
    #label = GM(features_train,n_components=10,train=True)
    sort_by_kmeans(train_label,label,expert_num=10)
    for v in label:
        ans[v] += 1
    print('train_label:{}'.format(ans))
    write_to_txt(label,'/root/resnet20/vgg19_label/feature_train_label.txt')
    #pca_result_file="/root/resnet20/tmp/pca_test_result.tsv"
    #pca_result = read_pca(pca_result_file)
    label = k_means(features_test,train=False)
    #label = GM(features_test,n_components=10,train=False)
    sort_by_kmeans(test_label,label,expert_num=10)
    ans = [0 for i in range(10)]
    for v in label:
        ans[v] += 1
    print('val_label:{}'.format(ans))
    write_to_txt(label,'/root/resnet20/vgg19_label/feature_test_label.txt')

    data_file = '/root/resnet20/cifar-100-python/cifar-100-python/train'
    expert_label_file = '/root/resnet20/vgg19_label/feature_train_label.txt'
    save_file = '/root/resnet20/vgg19_label/expert_train'

    # ?????????????????????????????????????????????
    add_expert_label_to_dataset(data_file,expert_label_file,save_file,train=True)
    data_file = '/root/resnet20/cifar-100-python/cifar-100-python/test'
    expert_label_file = '/root/resnet20/vgg19_label/feature_test_label.txt'
    save_file = '/root/resnet20/vgg19_label/expert_test'
    add_expert_label_to_dataset(data_file,expert_label_file,save_file,train=False)

    # root = '/root/resnet20/cifar-100-python'
    # cifar100_to_img_by_expertlabel(root)
    
    
def test():
    #train_dis =[4000, 5500, 4500, 5500, 6500, 3000, 6000, 4500, 4500, 6000]
    train_dis = [i*500 for i in range(100)]
    #test_dis = [800, 1100, 900, 1100, 1300, 600, 1200, 900, 900, 1200]
    test_dis = [i*100 for i in range(100)]
    ans_train = 0
    ans_test = 0
    # for i in range(10):
    #     ans_train += train_dis[i]
    #     train_dis[i] = ans_train
    #     ans_test += test_dis[i]
    #     test_dis[i] = ans_test
    print(train_dis)
    print(test_dis) 
    path = '/root/resnet20/vgg19_label/expert_test'
    with open(path, 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
    for k,v in entry.items():
        print(k)
    ans = 0
    for i in test_dis:
        print('{} label:{}'.format(i,entry['fine_labels'][i]))
    path = '/root/resnet20/vgg19_label/expert_train'
    with open(path, 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
    for k,v in entry.items():
        print(k)
    ans = 0
    for i in train_dis:
        print('{} label:{}'.format(i,entry['fine_labels'][i]))
    sys.exit(0)
    ans = [0 for i in range(10)]
    with open('/root/resnet20/tmp/gs_train_label.txt', "r") as f:  # ????????????
        data = f.readlines()  # ????????????
        for d in data:
            d = d.split()[0]
            ans[(int(d))] += 1
    print(ans)
    ans = [0 for i in range(10)]
    with open('/root/resnet20/tmp/gs_test_label.txt', "r") as f:  # ????????????
        data = f.readlines()  # ????????????
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
    with open("/root/resnet20/tmp/feature_train_label.txt", "r") as f:  
        data = f.readlines()  
        for d in data:
            d = d.split()[0]
            expert_train_label.append(int(d[1]))
    with open("/root/resnet20/tmp/feature_test_label.txt", "r") as f:  
        data = f.readlines()  
        for d in data:
            d = d.split()[0]
            expert_test_label.append(int(d[1]))

    for i, (X, Y) in enumerate(train_set):  # ???train_set????????????label????????????
        trainset.append(list((np.array(X), np.array(Y), np.array(expert_train_label[i]))))
    for i, (X, Y) in enumerate(test_set):  # ???test_set????????????label????????????
        testset.append(list((np.array(X), np.array(Y), np.array(expert_test_label[i]))))
    for X, Y, l in trainset:
        character_train[Y].append(list((X, l)))  # 32*32*3
    for X, Y, l in testset:
        character_test[Y].append(list((X, l)))  # 32*32*3

    root = '/root/resnet20/tmp/'
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

def add_expert_label_to_dataset(data_file,expert_label_file,save_file,train=False):
    train = False
    print('add_expert_label_to_dataset start...')
    with open(data_file, 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
    expert_label = []
    with open(expert_label_file, "r") as f:  # ????????????
        data = f.readlines()  # ????????????
        for d in data:
            d = d.split()[0]
            expert_label.append(int(d))
    data = []
    data_size = len(entry['fine_labels'])
    for i in range(data_size):
        data.append(list((entry['coarse_labels'][i],entry['fine_labels'][i],entry['data'][i],expert_label[i])))
    if train:
        data.sort(key=lambda x:x[1])
    else:
        print('fine_labels')
        data.sort(key=lambda x:x[1])
    coarse_labels=[]
    fine_labels = []
    img = []
    expert_label = []
    for i in range(data_size):
        coarse_labels.append(data[i][0])
        fine_labels.append(data[i][1])
        img.append(data[i][2])
        expert_label.append(data[i][3])
    img = np.array(img)
    entry['coarse_labels'] = coarse_labels
    entry['fine_labels'] = fine_labels
    entry['data'] = img
    entry['vgg_label'] = expert_label
    with open(save_file, "wb") as f:  # ????????????
        pickle.dump(entry, f)
    
    print('add_expert_label_to_dataset done.')
 

def test_get_feature():
    model = models.resnet50()
    for name, layer in model.named_modules():
        if (('4.1' in name or '4.2' in name ) and 'conv' in name) or '4.0.conv3' in name:
            layer.register_forward_hook(features)
            print(name)
    x = torch.Tensor(1,3,64,64)
    out = model(x)
    

if __name__ == '__main__':
    # ???cifar100??????feature?????? ??????vgg19 pretrain model 
    classify_cifar100()
    test()
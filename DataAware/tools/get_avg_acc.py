import re
import matplotlib.pyplot as plt
from numpy import double
x = []
val_acc = []
ori_acc = []
re_acc = []
file = '/root/resnet20/tmp/k-menas_expert10_layer3.txt'
with open(file, "r") as f:  # 打开文件
    data = f.readlines()  # 读取文件
    for d in data:
        if 'epoch' in d:
            d = re.split(':| |\n',d)
            val_acc.append(double(d[5]))
            ori_acc.append(double(d[7]))
            re_acc.append(double(d[5])-double(d[7]))
            x.append(int(d[1]))
            #print(d)
ans = 0
start_epoch = 100
end_epoch = 200
for i in range(start_epoch,end_epoch):
    ans += re_acc[i]
print('avg:{}'.format(ans/(end_epoch - start_epoch)))
plt.plot(x,re_acc)

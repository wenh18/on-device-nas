import random
import matplotlib
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import re


def draw_pic_test():
    x = [i for i in range(10)]
    y1 = [0.1224, 0.0071, 0.1303, 0.0002, 0.0001, 0.3955, 0.0291, 0.3137, 0.0007, 0.0009] 
    y2 = [0.013, 0.0179, 0.0001, 0.0622, 0.1155, 0.0106, 0.7718, 0.0002, 0.0001, 0.0086] 
    y3 = [0.2266, 0.0001, 0.3036, 0.0001, 0.0001, 0.3288, 0.0001, 0.028, 0.1125, 0.0001]  
    y4 = [0.0803, 0.0006, 0.2412, 0.0545, 0.1471, 0.0001, 0.0002, 0.0096, 0.0004, 0.466] 
    y5 = [0.0333, 0.0001, 0.2305, 0.0008, 0.0117, 0.0011, 0.0064, 0.0363, 0.0272, 0.6526] 
    y6 = [0.0163, 0.7939, 0.0087, 0.0005, 0.0003, 0.0511, 0.0945, 0.0244, 0.0046, 0.0057] 
    # y1.sort(reverse=True)
    # y2.sort(reverse=True)
    # y3.sort(reverse=True)
    # y4.sort(reverse=True)
    # y5.sort(reverse=True)
    # y6.sort(reverse=True)
    plt.title('Label distribution')  # 折线图标题
    plt.xlabel('expert label')  # x轴标题
    plt.ylabel('number')  # y轴标题
    plt.plot(x, y1, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    plt.plot(x, y2, marker='o', markersize=3)
    plt.plot(x, y3, marker='o', markersize=3)
    plt.plot(x, y4, marker='o', markersize=3)
    plt.plot(x, y5, marker='o', markersize=3)
    plt.plot(x, y6, marker='o', markersize=3)

    # for a, b in zip(x, y1):
    #     plt.text(a, b, ha='center', va='bottom', fontsize=10)  # 设置数据标签位置及大小
    # for a, b in zip(x, y2):
    #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    # for a, b in zip(x, y3):
    #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    # for a, b in zip(x, y4):
    #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    # for a, b in zip(x, y5):
    #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    # for a, b in zip(x, y6):
    #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    
    plt.legend(['dis1', 'dis2', 'dis3', 'dis4', 'dis5','dis6'])  # 设置折线名称
    
    plt.show()  # 显示折线图

def histogram():
    
    N = 13
    S = (52, 49, 48, 47, 44, 43, 41, 41, 40, 38, 36, 31, 29)
    C = (38, 40, 45, 42, 48, 51, 53, 54, 57, 59, 57, 64, 62)
    
    d=[]
    for i in range(0,len(S)):
        sum = S[i] + C[i]
        d.append(sum)
    M = (10, 11, 7, 11, 8, 6, 6, 5, 3, 3, 7, 5, 9)
    #menStd = (2, 3, 4, 1, 2)
    #womenStd = (3, 5, 2, 3, 3)
    ind = np.arange(N)    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence
    
    p1 = plt.bar(ind, S, width, color='#d62728')#, yerr=menStd)
    p2 = plt.bar(ind, C, width, bottom=S)#, yerr=womenStd)
    p3 = plt.bar(ind, M, width, bottom=d)
    
    plt.ylabel('Scores')
    plt.title('Scores by group and gender')
    plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5','G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13'))
    plt.yticks(np.arange(0, 81, 20))
    plt.legend((p1[0], p2[0], p3[0]), ('S', 'C', 'M'))
    
    plt.show()


if __name__ == '__main__':
    draw_pic_test()
import torch
from torch.cuda import synchronize 
import torchvision
from torch2trt import torch2trt
from torch2trt import TRTModule
import time
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import copy
from torch.nn.parameter import Parameter
import cv2

def get_img_np_nchw(filename):
    image = cv2.imread(filename)
    image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_cv = cv2.resize(image_cv, (224, 224))
    miu = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = np.array(image_cv, dtype=float) / 255.
    r = (img_np[:, :, 0] - miu[0]) / std[0]
    g = (img_np[:, :, 1] - miu[1]) / std[1]
    b = (img_np[:, :, 2] - miu[2]) / std[2]
    img_np_t = np.array([r, g, b])
    img_np_nchw = np.expand_dims(img_np_t, axis=0)
    return img_np_nchw

# class new_model(nn.Module):
#     def __init__(self, ModuleList):
#         super(new_model, self).__init__()
#         self.ModuleList = ModuleList
    
#     def forward(self, x):
#         for j in range(len(self.ModuleList) - 1):
#             x = self.ModuleList[j](x)
#         x = self.ModuleList[-1](x.view(-1, x.size()[1]))
#         return x

def get_modulelist(model):
    net_structure = list(model.children())
    new_module = nn.ModuleList()
    for i in range(len(net_structure)):
        if isinstance(net_structure[i], nn.ModuleList):
            for j in net_structure[i]:
                new_module.append(j)
            continue
        new_module.append(net_structure[i])
    return new_module


def save_trt(new_module, batch_size, ks, e):
    input_data = torch.rand((batch_size, 3, 224, 224), dtype=torch.float).cuda()
    for i in range(len(new_module)):
        print("------------", i, "------------")
        new_module.cuda()
        if i == (len(new_module) - 1):
            # input_data = input
            model_trt_int8 = torch2trt(new_module[i].cuda().eval(), [input_data.view(-1, input_data.size()[1])], max_batch_size=batch_size, fp16_mode=True, int8_mode=True)
            torch.save(model_trt_int8.state_dict(), './torch2trt_blocks/block{}ks{}e{}.pth'.format(i, ks, e))
        # input_data = new_module[i](input_data)
        else:
            model_trt_int8 = torch2trt(new_module[i].cuda().eval(), [input_data], max_batch_size=batch_size, fp16_mode=True, int8_mode=True)
            torch.save(model_trt_int8.state_dict(), './torch2trt_blocks/block{}ks{}e{}.pth'.format(i, ks, e))
            input_data = new_module[i](input_data)

def OFANets2Blocks():
    for ks in [3, 5, 7]:
        for e in [3, 4, 6]:
            if ks == 3 and e != 6:
                continue
            net = torch.load("./torch2trt_blocks/OFANets/ks{}e{}.pkl".format(ks, e))
            new_module = get_modulelist(net)
            save_trt(new_module, batch_size=1, ks=ks, e=e)

# OFANets2Blocks()

def get_inference_time(ks_list, ex_list, x, SameBlocks=2, TIMES=1000):
    total_time = 0
    # same blocks
    model_trt = TRTModule()
    for blockidx in range(SameBlocks):
        
        model_trt.load_state_dict(torch.load('./torch2trt_blocks/block{}ks{}e{}.pth'.format(blockidx, 3, 3)))
        average_time = 0
        for _ in range(TIMES):
            with torch.no_grad():
                output_data, time_block = model_trt(x)
            average_time += time_block
        with torch.no_grad():
            x, _ = model_trt(x)
        average_time /= 1000
        total_time += average_time
        # very important, otherwise TRTModules will run out of memory
        model_trt.context.__del__()
        model_trt.engine.__del__()
        # del model_trt
            
    for blockidx in range(len(ks_list)):
        # print(blockidx)
        if ks_list[blockidx] == 0:
            continue
        else:
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load('./torch2trt_blocks/block{}ks{}e{}.pth'.format(blockidx, ks_list[blockidx], ex_list[blockidx])))
            average_time = 0
            for _ in range(TIMES):
                with torch.no_grad():
                    output_data, time_block = model_trt(x)
                average_time += time_block
            with torch.no_grad():
                x, _ = model_trt(x)
            average_time /= 1000
            total_time += average_time
            model_trt.context.__del__()
            model_trt.engine.__del__()
            # del model_trt
    
    for blockidx in range(22, 25):
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load('./torch2trt_blocks/block{}ks{}e{}.pth'.format(blockidx, 3, 3)))
        average_time = 0
        for _ in range(TIMES):
            with torch.no_grad():
                output_data, time_block = model_trt(x)
            average_time += time_block
        with torch.no_grad():
            x, _ = model_trt(x)
        average_time /= 1000
        total_time += average_time
        # del model_trt
        model_trt.context.__del__()
        model_trt.engine.__del__()

    blockidx = 25
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load('./torch2trt_blocks/block{}ks{}e{}.pth'.format(blockidx, 3, 3)))
    average_time = 0
    for _ in range(TIMES):
        with torch.no_grad():
            output_data, time_block = model_trt(x.view(-1, x.size()[1]))
        average_time += time_block
    average_time /= 1000
    total_time += average_time
    # del model_trt
    torch.cuda.empty_cache()
    # import pdb;pdb.set_trace()
    # model_trt.__del__()
    model_trt.context.__del__()
    model_trt.engine.__del__()
    # model_trt.Run
    return total_time

# while True:
#     print("one turn")
#     # import pdb;pdb.set_trace()
#     ks = np.random.randint(0, 3, 20)
#     for i in range(len(ks)):
#         ks[i] = ks[i]*2 + 3
#     e = np.random.randint(0, 3, 20)
#     for i in range(20):
#         e[i] = e[i] + 3
#         if e[i] == 5:
#             e[i] = 6
#     x = torch.rand((1, 3, 224, 224), dtype=torch.float).cuda()
#     _ = get_inference_time(ks, e, x)
#     print(_)
#     # del x
#     torch.cuda.empty_cache()

# t1 = time.time()
# x = get_img_np_nchw('test.JPEG')
# x = x.astype(dtype=np.float32)
# x = Variable(torch.from_numpy(x)).cuda()
# # ks_list = [7 for i in range(20)]
# ks_list = [7, 7, 0, 0, 5, 5, 0, 0, 7, 7, 0, 0, 3, 3, 0, 0]
# e_list = [6, 6, 0, 0, 6, 6, 0, 0, 6, 6, 0, 0, 6, 6, 0, 0]
# print(get_inference_time(ks_list, e_list, x))
# print(time.time() - t1)

## export resnet50  input: batchsize 3 224 224
# x = get_img_np_nchw('test.JPEG')
# x = x.astype(dtype=np.float32)
# x = Variable(torch.from_numpy(x))
# net = torch.load("full_subnet.pkl")
# y = net(x)
# # import pdb;pdb.set_trace()
# print(y.argmax(dim=1))
# new_module = get_modulelist(net)
# # print(len(new_module))
# batch_size = 1
# save_trt(new_module, batch_size)
# import pdb;pdb.set_trace()

# t_start = time.time()
# x = get_img_np_nchw('test.JPEG')
# x = x.astype(dtype=np.float32)
# x = Variable(torch.from_numpy(x)).cuda()
# # input_data = torch.rand((batch_size, 3, 224, 224), dtype=torch.float).cuda()
# # model_trt_int8 = torch2trt(net.cuda().eval(), [input_data], max_batch_size=batch_size, fp16_mode=True, int8_mode=True)
# # torch.save(model_trt_int8.state_dict(), 'full_subnet.pth')
# model_trt = TRTModule()
# model_trt.load_state_dict(torch.load('full_subnet.pth'))
# average_time = 0
# out_trt = None
# for _ in range(1000):
#     # t1 = time.time()
#     out_trt, t_used = model_trt(x, True)
#     # torch.cuda.synchronize()
#     # average_time = average_time + time.time() - t1
#     average_time += t_used
# average_time /= 1000
# print("class:", out_trt.argmax(dim=1))
# print(average_time)
# print("total time:", time.time() - t_start)


# # input_data = torch.rand((batch_size, 3, 224, 224), dtype=torch.float).cuda()
# # total_time = 0
# # block_list = nn.ModuleList()
# # for i in range(len(new_module)):
# #     model_trt = TRTModule()
# #     model_trt.load_state_dict(torch.load('./torch2trt_blocks/block{}ks7channel6.pth'.format(i)))
# #     block_list.append(model_trt)
# # torch.save(block_list.state_dict(), './torch2trt_blocks/test_net.pth')

# # average_time = 0

# # for _ in range(5000):
# #     x = torch.rand((batch_size, 3, 224, 224), dtype=torch.float).cuda()
# #     # t1 = time.time()
# #     t_used = 0
# #     for i in range(len(block_list) - 1):
# #         x, t_block = block_list[i](x)
# #         t_used += t_block
# #     x, t_block = block_list[-1](x.view(-1, x.size()[1]))
# #     t_used += t_block
# #     average_time += t_used
# # print(average_time / 5000)
# t_start = time.time()
# # input_data = torch.rand((batch_size, 3, 224, 224), dtype=torch.float).cuda()
# x = get_img_np_nchw('test.JPEG')
# x = x.astype(dtype=np.float32)
# x = Variable(torch.from_numpy(x)).cuda()
# total_time = 0
# block_time = []

# for i in range(len(new_module) - 1):
#     model_trt = TRTModule()
#     model_trt.load_state_dict(torch.load('./torch2trt_blocks/block{}ks7channel6.pth'.format(i)))
    
#     average_time = 0
#     for _ in range(1000):
#         # t1 = time.time()
#         output_data, time_block = model_trt(x)
#         average_time = average_time + time_block
#     x, _ = model_trt(x)
#     average_time /= 1000
#     block_time.append(average_time)
#     total_time += average_time
# model_trt = TRTModule()
# i = len(new_module) - 1
# model_trt.load_state_dict(torch.load('./torch2trt_blocks/block{}ks7channel6.pth'.format(i)))
# average_time = 0
# # print(average_time)
# output_data = None
# for _ in range(1000):
#     # t1 = time.time()
#     output_data, time_block = model_trt(x.view(-1, x.size()[1]), True)
#     average_time = average_time + time_block
# print("class:", output_data.argmax(dim=1))
# # print(output)
# average_time /= 1000
# block_time.append(average_time)
# total_time += average_time
# print(total_time)
# print(block_time)
# print("used:", time.time() - t_start)




########################## OLD BACKUP ############################
#     average_time = 0
#     for _ in range(10000):
#         t1 = time.time()
#         output_data = model_trt(input_data)
#         average_time = average_time + time.time() - t1
#     input_data = model_trt(input_data)
#     average_time /= 10000
#     total_time += average_time
# model_trt = TRTModule()
# i = len(new_module) - 1
# model_trt.load_state_dict(torch.load('./torch2trt_blocks/block{}ks7channel6.pth'.format(i)))
# average_time = 0
# # print(average_time)
# for _ in range(10000):
#     t1 = time.time()
#     output_data = model_trt(input_data.view(-1, input_data.size()[1]))
#     average_time = average_time + time.time() - t1
# average_time /= 10000
# total_time += average_time
# print(total_time)


# t1 = time.time()
# out = net(input_data)
# torch.cuda.synchronize()
# print(time.time() - t1)



# import torch 
# import torchvision
# from torch2trt import torch2trt
# from torch2trt import TRTModule
# import time
# import torch.nn as nn
# import numpy as np
# import torch.optim as optim
# from torch.autograd import Variable
# import torch.nn.functional as F
# import torch
# import copy
# from torch.nn.parameter import Parameter
# import cv2

# help(TRTModule)
# import pdb;pdb.set_trace()

# def get_modulelist(model):
#     net_structure = list(model.children())
#     new_module = nn.ModuleList()
#     for i in range(len(net_structure)):
#         if isinstance(net_structure[i], nn.ModuleList):
#             for j in net_structure[i]:
#                 new_module.append(j)
#             continue
#         new_module.append(net_structure[i])
#     return new_module


# def save_trt(new_module, batch_size):
#     input_data = torch.rand((batch_size, 3, 224, 224), dtype=torch.float).cuda()
#     for i in range(len(new_module)):
#         print("------------", i, "------------")
#         new_module.cuda()
#         if i == (len(new_module) - 1):
#             # input_data = input
#             model_trt_int8 = torch2trt(new_module[i].cuda().eval(), [input_data.view(-1, input_data.size()[1])], max_batch_size=batch_size, fp16_mode=True, int8_mode=True)
#             torch.save(model_trt_int8.state_dict(), './torch2trt_blocks/block{}ks7channel6.pth'.format(i))
#         # input_data = new_module[i](input_data)
#         else:
#             # model_trt_int8 = torch2trt(new_module[i].cuda().eval(), [input_data], max_batch_size=batch_size, fp16_mode=True, int8_mode=True)
#             # torch.save(model_trt_int8.state_dict(), './torch2trt_blocks/block{}ks7channel6.pth'.format(i))
#             input_data = new_module[i](input_data)

# ## export resnet50  input: batchsize 3 224 224
# net = torch.load("full_subnet.pkl")
# new_module = get_modulelist(net)
# print(len(new_module))
# batch_size = 1
# # save_trt(new_module, batch_size)


# input_data = torch.rand((batch_size, 3, 224, 224), dtype=torch.float).cuda()
# # model_trt_int8 = torch2trt(net.eval(), [input_data], max_batch_size=batch_size, fp16_mode=True, int8_mode=True)
# # torch.save(model_trt_int8.state_dict(), 'full_subnet.pth')
# model_trt = TRTModule()
# model_trt.load_state_dict(torch.load('full_subnet.pth'))
# average_time = 0
# for _ in range(10000):
#     t1 = time.time()
#     out_trt = model_trt(input_data)
#     # torch.cuda.synchronize()
#     average_time = average_time + time.time() - t1
# average_time /= 10000
# # import pdb;pdb.set_trace()
# print(average_time)


# input_data = torch.rand((batch_size, 3, 224, 224), dtype=torch.float).cuda()
# total_time = 0
# block_list = []
# for i in range(len(new_module) - 1):
#     model_trt = TRTModule()
#     model_trt.load_state_dict(torch.load('./torch2trt_blocks/block{}ks7channel6.pth'.format(i)))
    
#     average_time = 0
#     for _ in range(10000):
#         t1 = time.time()
#         output_data = model_trt(input_data)
#         average_time = average_time + time.time() - t1
#     input_data = model_trt(input_data)
#     average_time /= 10000
#     total_time += average_time
# model_trt = TRTModule()
# i = len(new_module) - 1
# model_trt.load_state_dict(torch.load('./torch2trt_blocks/block{}ks7channel6.pth'.format(i)))
# average_time = 0
# # print(average_time)
# for _ in range(10000):
#     t1 = time.time()
#     output_data = model_trt(input_data.view(-1, input_data.size()[1]))
#     average_time = average_time + time.time() - t1
# average_time /= 10000
# total_time += average_time
# print(total_time)

import argparse
import os
from random import randrange
import sys
import time
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from copy import deepcopy
from tensorboardX import SummaryWriter

from network.resnet50_ori import cifar100_resnet50_ori
from tools.frozen_layers import frozen_layers_resnet20
sys.path.append(os.getcwd())
from network.resnet20_train import cifar100_resnet20
from network.resnet20_ori import cifar100_resnet20_ori
from tools.load_data import load_dis_data,load_data
import global_var

parser = argparse.ArgumentParser(description='resnet on cifar100')
parser.add_argument('-j', '--workers', default=14, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--num_experts', default=11, type=int, metavar='N',
                    help='number of experts')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print_freq', '-p', default=3, type=int,
                    metavar='N', help='print frequency (default: 20)')
                    #/root/resnet20/model/trainmodel/
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='use cpu')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default=r'/root/resnet20/model/trainmodel', type=str)

global_var._init() 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer_loss_train = SummaryWriter(logdir=r'/root/tf-logs/log/resnet20/loss/train')
writer_loss_val = SummaryWriter(logdir=r'/root/tf-logs/log/resnet20/loss/val')
writer_acc_train = SummaryWriter(logdir=r'/root/tf-logs/log/resnet20/acc/train')
writer_acc_val = SummaryWriter(logdir=r'/root/tf-logs/log/resnet20/acc/val')
writer_acc_ori = SummaryWriter(logdir=r'/root/tf-logs/log/resnet20/acc/ori')
writer_acc_pc = SummaryWriter(logdir=r'/root/tf-logs/log/resnet20/acc/pc')
writer_acc_comp = SummaryWriter(logdir=r'/root/tf-logs/log/resnet20/compare/compare')
writer_acc_pc_comp = SummaryWriter(logdir=r'/root/tf-logs/log/resnet20/compare/pc_compare')

def main():
    global args
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = cifar100_resnet20()
    model_dict = model.state_dict()
    for k,v in model_dict.items():
        print(k,v.shape)
    ori_model = cifar100_resnet20_ori(pre_model=True)

    model_dict = model.state_dict()
    pretrained_dict = torch.load('/root/resnet20/model/premodel/cifar100_resnet20-23dac2f1.pth')
    # pre_dict_tmp = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pre_dict_tmp)
    for k, v in pretrained_dict.items():
        if 'layer3' in k and 'layer3.0' not in k and 'conv' in k:
            model_dict[k] = pretrained_dict[k].expand_as(model_dict[k])
            model_dict[k[:14]+'_pre'+k[14:]] = pretrained_dict[k]
        elif 'fc' in k:
            #pass Unexpected key(s) in state_dict: "layer3.1.conv1.weight_pre", "layer3.1.conv2.weight_pre", "layer3.2.conv1.weight_pre", "layer3.2.conv2.weight_pre", "fc.weight_pre", "fc.bias_pre".
            model_dict[k] = pretrained_dict[k].expand_as(model_dict[k])
            model_dict[k[0:2]+'_pre'+k[2:]] = pretrained_dict[k]
        else:
            model_dict[k] = pretrained_dict[k]
        # if 'layer3' in k and 'layer3.0' not in k and 'bn' in k:
        #     for i in range(args.num_experts):
        #         model_k = '{}conv{}_bn.{}{}'.format(k[:9],k[11],i,k[12:])
        #         model_dict[model_k] = pretrained_dict[k]
    model.load_state_dict(model_dict)
    model = frozen_layers_resnet20(model)
    for name,v,in model.named_parameters():
        print(name,v.requires_grad)
    if args.cpu:
        model.cpu()
    else:
        model.to(device)
        ori_model.to(device)
        cudnn.benchmark = True


    criterion = torch.nn.CrossEntropyLoss()

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(params, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    val_epoch = 5
    cnt_ori_prec1 = 0.
    val_ori_acc = []
    for i in range(val_epoch):
        val_loader = load_dis_data(train=False,index=i)
        ori_prec1, _ = validate(val_loader, ori_model, criterion)
        cnt_ori_prec1 += ori_prec1
        print('val_epoch:{} {}'.format(i,ori_prec1))
        val_ori_acc.append(ori_prec1)
    cnt_ori_prec1 /= val_epoch
    for epoch in range(args.start_epoch, args.epochs):

        train_loader = load_dis_data(train=True)
        train_prec1 = train(train_loader, model, criterion, optimizer, epoch)   
        adjust_learning_rate(optimizer, epoch)

        if epoch % args.print_freq == 0:
            cnt_val_prec1 = 0.
            cnt_val_loss = 0.
            cnt_pc_prec1 = 0.
            #cnt_ori_prec1 = 0.
            for i in range(val_epoch):
                val_loader = load_dis_data(train=False,index=i)

                val_prec1,val_loss = validate(val_loader, model, criterion)
                #pc_prec1, _ =  validate(val_loader, model, criterion,pc=True)
                #ori_prec1, _ = validate(val_loader, ori_model, criterion)
                #cnt_ori_prec1 += ori_prec1
                cnt_val_prec1 += val_prec1
                cnt_val_loss += val_loss
                #cnt_pc_prec1 += pc_prec1
                print('val_epoch:{} {}'.format(i,val_prec1 - val_ori_acc[i]))

            cnt_val_prec1 /= val_epoch
            cnt_val_loss /= val_epoch
            #cnt_pc_prec1 /= val_epoch
            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'state_dict': model.state_dict(),
            #     'optimizer': optimizer.state_dict(),
            # }, filename=os.path.join(args.save_dir, 'checkpoint_{}_acc_{:.2f}.tar'.format(epoch, cnt_val_prec1)))

            print("epoch:{} train:{} val:{} pc_acc:{} ori_acc:{} ".format(epoch, train_prec1, cnt_val_prec1, cnt_pc_prec1, cnt_ori_prec1))
            writer_loss_val.add_scalar('loss',cnt_val_loss,epoch)
            writer_acc_val.add_scalar('acc',cnt_val_prec1,epoch)
            writer_acc_ori.add_scalar('acc',ori_prec1,epoch)
            #writer_acc_pc.add_scalar('acc',pc_prec1,epoch)
            writer_acc_comp.add_scalar('compare',cnt_val_prec1-cnt_ori_prec1,epoch)
            #writer_acc_pc_comp.add_scalar('compare',cnt_pc_prec1-cnt_ori_prec1,epoch)
        else:
            print("epoch:{} train:{}".format(epoch, train_prec1))


def get_batch_dis(target,class_num=10):
    layer_num = 5
    cls_cnt = [[0 for j in range(class_num)] for i in range(layer_num)]
    batch_size = len(target[0]) * 1.0
    for i in range(layer_num):
        for t in target[i]:
            cls_cnt[i][int(t)] += 1

    for i in range(layer_num):    
        for j in range(class_num):
            cls_cnt[i][j] = cls_cnt[i][j] / batch_size
    # for i in range(layer_num-1):
    #     cls_cnt[i] = cls_cnt[layer_num-1]
    return cls_cnt

def get_batch_class_num(target,class_num=100):
    batch_size = len(target) * 1.0
    cls_cnt = [0 for _ in range(class_num)]
    for t in target:
        cls_cnt[t] += 1
    return cls_cnt

def train(train_loader, model, criterion, optimizer, epoch):
    """ 
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to train mode
    model.train()
    model._hook_before_iter()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        expert_label = target[1:]
        target = target[0]
        if args.cpu == False:
            input = input.to(device)
            target = target.to(device)

        if args.half:
            input = input.half()

        optimizer.zero_grad()
        alpha = get_batch_dis(expert_label)
        # clsnum = get_batch_class_num(target)
        # print('train aplha:{}'.format(alpha))
        # print('train cls_num:{}'.format(clsnum))
        #alphas = [alpha for i in range(7)]
        global_var.set_value('alpha', alpha)
        global_var.set_value('moe_index',0)
        output = model(input)
        loss = criterion(output,target)
        #loss.requires_grad_(True) 
        loss.backward()
        optimizer.step()
        output = output.float()
        loss = loss.float()
        prec1 = accuracy(output.data, target)[0].data.item()
        top1.update(prec1, input.size(0))
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        # if i % args.print_freq == 0:
        #     print('loss:{}'.format(loss))
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        #         epoch, i, len(train_loader), batch_time=batch_time,
        #         data_time=data_time, loss=losses, top1=top1))
        
    writer_acc_train.add_scalar('acc', top1.avg, epoch)
    writer_loss_train.add_scalar('loss', losses.avg, epoch)
    return top1.avg

def validate(val_loader, model, criterion, pc=False):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    end = time.time()
    for input, target in (val_loader):
        expert_label = target[1:]
        target = target[0]
        if args.cpu == False:
            input = input.to(device)
            target = target.to(device)
        if args.half:
            input = input.half()
        with torch.no_grad():
            alpha = get_batch_dis(expert_label)
            global_var.set_value('alpha', alpha)
            global_var.set_value('moe_index', 0)
            output = model(input)
            if pc:        
                _, pred = output.topk(1, 1, True, True)
                pred = pred.t()[0]
                label_dis = [0.0 for i in range(100)]
                for v in pred:
                    label_dis[v] += 1
                data_size = len(pred)
                for i in range(100):
                    label_dis[i] /= data_size
                label_dis = torch.tensor(label_dis).cuda()
                re = torch.log(label_dis)
                for j in range(output.shape[0]):
                    output[j] = output[j] + re
            
            # loss = criterion(output_logits=output, target=target, extra_info=extra_info)
            loss = criterion(output, target)
        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0].data.item()
        top1.update(prec1, input.size(0))
        losses.update(loss.item(), input.size(0))
    return float(top1.avg), float(losses.avg)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  


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


if __name__ == '__main__':
    main()
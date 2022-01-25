import sys
import logging
import argparse
sys.path.append("..")
from nets.mobilenet_v3 import MobileNetV3, TeacherModel
from utils import classification_utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# logging.basicConfig(filename='logger1024_batchsize_shuffleisfalse.log', level=logging.INFO)

parser = argparse.ArgumentParser("multi_branch_network")
parser.add_argument('--learning_rate', type=float, default=0.025, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight-decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=20, help='batch size')
# parser.add_argument('--final', type=int, default=20, help='batch size')
# parser.add_argument()
args = parser.parse_args()
args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
args.print_freq = 100
args.batchsize = 256
args.save_freq = 5
args.num_workers = 8
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logging.info(args)

def validate(args, val_loader, model, criterion, model_type='teacher', stage=0, teacher_model=None):
    model.eval()
    model = model.to(args.device)
    val_loss = utils.AverageMeter()
    val_acc = utils.AverageMeter()
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_loader):
            # print(step)
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            if model_type == 'teacher':
                outputs = model(inputs, train_stage=3)
            else:
                subnet, subnet_choice = model.generate_random_subnet(stage)
                outputs = model(inputs, subnet, subnet_choice, train_stage=stage)
            if model_type == 'teacher' or stage == 3:
                loss = criterion(outputs, targets)
                prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
                n = inputs.size(0)
                val_loss.update(loss.item(), n)
                val_acc.update(prec1.item(), n)
            else:
                n = inputs.size(0)
                teacher_outputs = teacher_model(inputs, stage).detach()
                loss = torch.norm(teacher_outputs - outputs, p=2) / args.batchsize
                del teacher_outputs
                val_loss.update(loss.item(), n)
    return val_loss.avg, val_acc.avg

def train(args, epoch, train_loader, model, optimizer, teacher_model=None, train_stage=0, temp=5., alpha=.7):
    model.train()
    lr = optimizer.param_groups[0]["lr"]
    train_acc = utils.AverageMeter()
    train_loss = utils.AverageMeter()
    steps_per_epoch = len(train_loader)
    for step, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()
        subnet, subnet_choice = model.generate_random_subnet(train_stage)
        outputs = model(inputs, subnet, subnet_choice, train_stage)
        teacher_outputs = teacher_model(inputs, train_stage).detach()
        if train_stage < 3:
            # loss = criterion(outputs, teacher_outputs)
            loss = torch.norm(teacher_outputs - outputs, p=2) / args.batchsize
        else:
            soft_loss = nn.KLDivLoss()(F.log_softmax(outputs / temp, dim=1), F.softmax(teacher_outputs / temp, dim=1)) * temp * temp * 2.0 * alpha
            stu_loss = F.cross_entropy(outputs, targets) * (1-alpha)
            loss = soft_loss + stu_loss
        del teacher_outputs
        loss.backward()
        optimizer.step()
        if train_stage == 3:
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            train_loss.update(loss.item(), n)
            train_acc.update(prec1.item(), n)
        else:
            prec1 = 0
            n = inputs.size(0)
            train_loss.update(loss.item(), n)
        if step % args.print_freq == 0 or step == len(train_loader) - 1:
                logging.info(
                    '[Supernet Training] lr: %.5f epoch: %03d/%03d, step: %03d/%03d, '
                    'train_loss: %.3f(%.3f), train_acc: %.3f(%.3f)'
                    % (lr, epoch+1, args.epochs, step+1, steps_per_epoch,
                    loss.item(), train_loss.avg, prec1, train_acc.avg)
                )
    return train_loss.avg, train_acc.avg

teacher_model = TeacherModel()
teacher_model = utils.load_model(teacher_model, "../model_data/mobilenetv3-large.pth")
teacher_model = teacher_model.to(args.device)
model = MobileNetV3()
model = utils.load_to_MultiModel(model, "../model_data/mobilenetv3-large.pth")
model = model.to(args.device)
# logging.info(model)
print("model loaded")
train_dataset, train_loader, val_dataset, val_loader = utils.build_dataloader(args.batchsize, args.num_workers)
print("data loaded")
for stage in range(4):
    t1 = time.time()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss().to(args.device)
    if stage < 3:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        for epoch in range(args.epochs):
            train_loss, train_acc = train(args, epoch, train_loader, model, optimizer, teacher_model, stage)
            scheduler.step()
            logging.info(
                    '[Supernet Training] stage: %d, epoch: %03d, train_loss: %.3f, train_acc: %.3f' %
                    (stage, epoch + 1, train_loss, train_acc)
                )
            val_loss, val_acc = validate(args, val_loader, model, criterion, model_type='student', stage=stage, teacher_model=teacher_model)
            logging.info(
                    '[Supernet val] stage: %d, epoch: %03d, val_loss: %.3f, val_acc: %.3f' %
                    (stage, epoch + 1, val_loss, val_acc)
                )
            if epoch % args.save_freq == 0:
                model_name = "sage%d_epoch%d_numworker8loss%.5f.pth" % (stage, epoch, val_loss)
                torch.save(model.state_dict(), model_name)
            print("epoch time:", time.time() - t1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)
        for epoch in range(50):
            train_loss, train_acc = train(args, epoch, train_loader, model, optimizer, teacher_model, stage)
            scheduler.step()
            logging.info(
                    '[Supernet Training] stage: %d, epoch: %03d, train_loss: %.3f, train_acc: %.3f' %
                    (stage, epoch + 1, train_loss, train_acc)
                )
            val_loss, val_acc = validate(args, val_loader, model, criterion, model_type='student', stage=stage, teacher_model=teacher_model)
            logging.info(
                    '[Supernet val] stage: %d, epoch: %03d, val_loss: %.5f, val_acc: %.3f' %
                    (stage, epoch + 1, val_loss, val_acc)
                )
            if epoch % args.save_freq == 0:
                model_name = "sage%d_epoch%d_numworker8loss%.5f.pth" % (stage, epoch, val_loss)
                torch.save(model.state_dict(), model_name)
            print("epoch time:", time.time() - t1)

# 14:46
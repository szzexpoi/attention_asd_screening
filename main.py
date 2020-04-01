import sys
sys.path.append('./util')
sys.path.append('./model')
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import cv2
import argparse
import os
import tensorflow as tf
from model import Sal_seq
from data_processing import Dataset, read_dataset, image_selection, loo_split
from glob import glob

parser = argparse.ArgumentParser(description='Autism screening based on eye-tracking data')
parser.add_argument('--img_dir', type=str, default='./saliency4asd/TrainingDataset/TrainingData/Images', help='Directory to images')
parser.add_argument('--anno_dir', type=str, default='./saliency4asd/TrainingDataset/TrainingData', help='Directory to annotation files')
parser.add_argument('--backend', type=str, default='resnet', help='Backend for visual encoder')
parser.add_argument('--lr',type=float,default=1e-4,help='specify learning rate')
parser.add_argument('--checkpoint_path',type=str,default=None,help='Directory for saving checkpoints')
parser.add_argument('--epoch',type=int,default=10,help='Specify maximum number of epoch')
parser.add_argument('--batch_size',type=int,default=12,help='Batch size')
parser.add_argument('--max_len',type=int,default=14,help='Maximum number of fixations for an image')
parser.add_argument('--hidden_size',type=int,default=512,help='Hidden size for RNN')
parser.add_argument('--clip',type=float,default=10,help='Gradient clipping')
parser.add_argument('--select_number',type=int,default=100,help='Number of images selected based on fisher score')
parser.add_argument('--n_fold',type=int,default=28,help='Number of folds used in the K-fold validation, default 28 for leave-one-out')
parser.add_argument('--img_height',type=int,default=600,help='Image Height')
parser.add_argument('--img_width',type=int,default=800,help='Image Width')

args = parser.parse_args()

transform = transforms.Compose([transforms.Resize((args.img_height,args.img_width)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_lr(optimizer, epoch):
    "adatively adjust lr based on epoch"
    if epoch <= 0 :
        lr = args.lr
    else :
        lr = args.lr * (0.5 ** (float(epoch) / 2))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    tf_summary_writer = tf.summary.create_file_writer(args.checkpoint_path)
    anno = read_dataset(args.anno_dir)
    overall_acc = []
    for fold in range(args.n_fold):
        train_data, val_data = loo_split(anno,fold)
        valid_id = image_selection(train_data, args.select_number)
        train_set = Dataset(args.img_dir,train_data,valid_id,args.max_len,args.img_height,args.img_width,transform)
        val_set = Dataset(args.img_dir,val_data,valid_id,args.max_len,args.img_height,args.img_width,transform)
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
        valloader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

        model = Sal_seq(backend=args.backend,seq_len=args.max_len,hidden_size=args.hidden_size)
        model = nn.DataParallel(model) # adding multiple GPU support
        model = model.cuda()

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5) # 5e-4

        def train(iteration):
            model.train()
            avg_loss = 0
            for j, (img,target,fix) in enumerate(trainloader):
                if len(img) < args.batch_size:
                    continue
                img, target, fix = Variable(img), Variable(target.type(torch.FloatTensor)), Variable(fix,requires_grad=False)
                img, target, fix = img.cuda(), target.cuda(), fix.cuda()
                optimizer.zero_grad()

                pred = model(img,fix)
                loss = F.binary_cross_entropy(pred,target)
                loss.backward()
                if args.clip != -1:
                    clip_gradient(optimizer,args.clip)
                optimizer.step()
                avg_loss = (avg_loss*np.maximum(0,j) + loss.data.cpu().numpy())/(j+1)

                if j%25 == 0:
                    with tf_summary_writer.as_default():
                        tf.summary.scalar('training loss',avg_loss,step=iteration)
                iteration += 1

            return iteration

        def validation_loo(epoch):
            model.eval()
            avg_pred = []

            for _, (img,target,fix) in enumerate(valloader):
                img, target, fix = Variable(img), Variable(target.type(torch.FloatTensor)), Variable(fix,requires_grad=False)
                img, target, fix = img.cuda(), target.cuda(), fix.cuda()

                pred = model(img,fix)
                pred = pred.data.cpu().numpy()
                target = target.data.cpu().numpy()[0,0]
                avg_pred.extend(pred)

            # average voting
            avg_pred = np.mean(avg_pred)

            if not target:
                avg_pred = 1-avg_pred
            label = 'asd' if target else 'ctrl'
            with tf_summary_writer.as_default():
                # print confidence of the correct prediction
                tf.summary.scalar('validation_acc_subject_' + label + '_' + str(fold+1), avg_pred, step=epoch+1)
            return avg_pred

        print('Start %d-fold validation for fold %d' %(args.n_fold,fold+1))
        iteration = 0
        best_acc = 0
        for epoch in range(args.epoch):
            # adjust_lr(optimizer,epoch)
            iteration = train(iteration)
            acc = validation_loo(epoch)
            if acc > best_acc:
                torch.save(model.state_dict(),os.path.join(args.checkpoint_path,'best_model_subj_'+str(fold+1)+'.pth'))
main()

import os
import sys
import argparse
import datetime

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from dataset import PolypDataset
from model import BFPS
from utils import AverageMeter, Logger
from utils import dice_coefficient, iou_coefficient
from utils import StructureLoss


def train_step(model, train_loader, criterion, optimizer, scheduler, epoch):
    print("----------")
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    print(f'Current lr: {lr} | Train epoch: {epoch}')
    losses = AverageMeter()
    model.train()
    scheduler.step()
    for i, (image, mask) in enumerate(tqdm(iter(train_loader), total=len(train_loader))):
        image = image.float().cuda()                                   
        label = mask.unsqueeze(1).float().cuda()                            

        output = model(image)
        
        loss = criterion(output, label)
        losses.update(loss.data, image.size(0))

        optimizer.zero_grad()
        loss.backward()
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.dtype.is_complex:
                    p.data = p.data.float()
        optimizer.step()
    print(f'Loss: {losses.avg:.4f}')


def validate(model, val_loader, best_score, exp_folder):
    print("----------")
    print('Validating')
    ious = []
    dices = []
    score=0
    model.eval()
    for (image, mask) in tqdm(iter(val_loader), total=len(val_loader)):
        image = image.float().cuda()
        label = mask.float().cuda()

        output = model(image)
        pred = output.squeeze(0).squeeze(0) > 0
        gt = label.squeeze(0)
    
        ious.append(iou_coefficient(gt, pred))
        dices.append(dice_coefficient(gt, pred))

    mean_iou = np.average(ious)
    mean_dice = np.average(dices)

    print(f'Mean IoU: {mean_iou: .4f} | Mean dice: {mean_dice: .4f}')
    score += mean_iou

    if score > best_score:
        best_score = score
        ckpt_file_name = os.path.join(exp_folder, 'best.pth')
        print(f'Save best checkpoint at {ckpt_file_name}')
        torch.save(model.state_dict(), ckpt_file_name)
    print('Validating done')

    return best_score


def test(model, test_loader):
    print("----------")
    print('Testing')
    ious = []
    dices = []
    model.eval()
    for (image, mask) in tqdm(iter(test_loader), total=len(test_loader)):
        image = image.float().cuda()
        label = mask.float().cuda()

        output = model(image)
        pred = output.squeeze(0).squeeze(0) > 0
        gt = label.squeeze(0)

        ious.append(iou_coefficient(gt, pred))
        dices.append(dice_coefficient(gt, pred))

    mean_iou = np.average(ious)
    mean_dice = np.average(dices)

    print(f'Mean IoU: {mean_iou: .4f} | Mean dice: {mean_dice: .4f}')
    print('Testing done')


def train(model, args, exp_folder):
    train_set = PolypDataset(args.data_folder, image_size=args.image_size)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    val_set = PolypDataset(args.data_folder, image_size=args.image_size, val_dataset='CVC-ClinicDB', train=False)
    val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    criterion = StructureLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_rate, weight_decay=args.weight_decay)    
    start_epoch = 0
    if args.resume_file:
        print("----------")
        print(f'Try to load resume file from {args.resume_file}')
        if os.path.isfile(args.resume_file):
            checkpoint = torch.load(args.resume_file, map_location=torch.device('cpu'))
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['opt_dict'])
            print(f'Checkpoint loaded (epoch {checkpoint["epoch"]})')
        else:
            print(f'No checkpoint found at {args.resume_file}')
    lambda_poly = lambda i: (1 - (i + start_epoch) / args.epochs) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_poly)

    print("----------")
    print(f'Start training from epoch {start_epoch}')
    best_score = 0
    for epoch in range(start_epoch, args.epochs):
        train_step(model, train_loader, criterion, optimizer, scheduler, epoch)
        if epoch % args.val_epoch == 0 and epoch > 0:
            best_score = validate(model, val_loader, best_score, exp_folder)
        if epoch % args.save_epoch == 0 and epoch > 0:
            resume_file_name = os.path.join(exp_folder, f'resume_{epoch}.pth')
            print(f'Save resume file at {resume_file_name}')
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
            torch.save(state, resume_file_name)
    print('Training done')

    checkpoint_name = os.path.join(exp_folder, 'best.pth')
    checkpoint = torch.load(checkpoint_name)
    model.load_state_dict(checkpoint, map_location=torch.device('cpu'))
    for name in ['Kvasir', 'CVC-ClinicDB', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
        test_set = PolypDataset(args.data_folder, image_size=args.image_size, val_dataset=name, train=False)
        test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers)
        print(f'Testing dataset {name}')
        test(model, test_loader)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comprehensive attention network for biomedical Dataset')
    parser.add_argument('--cuda_device', type=int, default=0, help='set the cuda_device')
    parser.add_argument('--version', default='b4', help='version of model')
    parser.add_argument('--exp_name', default='', help='name of experiment, if not set, use version and time stamp')
    parser.add_argument('--data_folder', default='data/polyp/', help='data folder for a specific dataset')
    parser.add_argument('--image_size', default=(352, 352), help='the input and output image size')
    parser.add_argument('--num_workers', type=int, default=6, help='number of workers used in dataloading')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 50)')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
    parser.add_argument('--lr_rate', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--in_channels', default=3, type=int, help='number of input channels')
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weights regularizer')
    parser.add_argument('--val_epoch', default=1, type=int, help='validate per how many epochs')
    parser.add_argument('--save_epoch', default=10, type=int, help='save per how many epochs')
    parser.add_argument('--resume_file', default='', help='the checkpoint that resumes from')
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    args = parser.parse_args()
    
    torch.cuda.set_device(args.cuda_device)
    torch.cuda.manual_seed(args.seed)
    if not args.exp_name:
        now = datetime.datetime.now()
        time_stamp = now.strftime("%Y%m%d_%H%M%S_%f")
        args.exp_name = f"polyp_{args.version}_{time_stamp}"
    exp_folder = os.path.join('experiment/', args.exp_name)
    os.makedirs(exp_folder, exist_ok=True)
    logfile = os.path.join(exp_folder, 'train.log')
    sys.stdout = Logger(logfile)
    print(f'Experiment will be saved at {exp_folder}')
    
    model = BFPS(in_channels=args.in_channels, num_classes=args.num_classes, backbone=args.version)
    model = model.cuda()

    # print("----------")
    # print("Network Architecture:")
    # print(model)
    print("----------")
    import thop
    input_tensor = torch.randn(1, 3, args.image_size[0], args.image_size[1]).cuda()
    flops, params = thop.profile(model, inputs=(input_tensor,), verbose=False)
    print(f'FLOPS: {flops / 1e9}G')
    print(f'Params: {params / 1e6}M')

    train(model, args, exp_folder)

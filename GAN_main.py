import numpy as np
import argparse
import os
import json
import torch.utils.data as data
from torchvision import transforms
import random
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
import math

from face_dataset import faceDataset
from GAN import Generator, Discriminator
from util import averageMeter, lr_decay
        
def main():
    global save_dir, logger
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # setup directory to save logfiles, checkpoints, and output csv
    save_dir = args.save_dir
    if args.phase == 'train' and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # setup logger
    logger = None
    if args.phase == 'train':
        logger = open(os.path.join(save_dir, 'train.log'), 'a')
        logfile = os.path.join(save_dir, 'training_log.json')
        log = {'train': []}
        logger.write('{}\n'.format(args))
    
        # setup data loader for training images
        trans_train = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset_train = faceDataset(os.path.join(args.data_root, 'face/train'), trans_train)
        train_loader = data.DataLoader(dataset_train, shuffle=True, drop_last=False, pin_memory=True, batch_size=args.batch_size)
        print('train: {}'.format(dataset_train.__len__()))
        logger.write('train: {}\n'.format(dataset_train.__len__()))
    
    # setup model
    G = Generator(in_dim=args.z_dim).cuda()
    D = Discriminator(in_dim=3).cuda()
    
    if args.phase == 'train':
        logger.write('Generator:{}\n'.format(G))
        logger.write('Discriminator:{}\n'.format(D))
   
    # setup optimizer
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.5, 0.999))
    
    if args.phase == 'train':
        logger.write('opt_G:{}\n'.format(opt_G))
        logger.write('opt_D:{}\n'.format(opt_D))
        
    # setup loss function
    criterion = nn.BCELoss()
    
    # load checkpoint
    start_ep = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        G.load_state_dict(checkpoint['G_state'])
        D.load_state_dict(checkpoint['D_state'])
        opt_G.load_state_dict(checkpoint['opt_G_state'])
        opt_D.load_state_dict(checkpoint['opt_D_state'])
        start_ep = checkpoint['epoch']
        print("Loaded checkpoint '{}' (epoch: {})".format(args.checkpoint, start_ep))
        if args.phase == 'train':
            logger.write("Loaded checkpoint '{}' (epoch: {})\n".format(args.checkpoint, start_ep))
            if os.path.isfile(logfile):
                log = json.load(open(logfile, 'r'))
    
    if args.phase == 'train':
        
        # start training
        print('Start training from epoch {}'.format(start_ep))
        logger.write('Start training from epoch {}\n'.format(start_ep))        
        for epoch in range(start_ep, args.epochs):
            loss_D, loss_G = train(train_loader, G, D, opt_G, opt_D, epoch, criterion)
            log['train'].append([epoch + 1, loss_D, loss_G])
                
            if (epoch + 1) % 10 == 0:
                # generate sampled images
                with torch.no_grad():
                    val(G, epoch, n_sample=64)
                
                # save checkpoint
                state = {
                    'epoch': epoch + 1,
                    'opt_G_state': opt_G.state_dict(),
                    'opt_D_state': opt_D.state_dict(),
                    'loss_G': loss_G,
                    'loss_D': loss_D,
                    'G_state': G.state_dict(),
                    'D_state': D.state_dict(),
                }
                checkpoint = os.path.join(save_dir, 'ep-{}.pkl'.format(epoch + 1))
                torch.save(state, checkpoint)
                print('[Checkpoint] {} is saved.'.format(checkpoint))
                logger.write('[Checkpoint] {} is saved.\n'.format(checkpoint))
                json.dump(log, open(logfile, 'w'))

            if (epoch + 1) % args.step == 0:
                lr_decay(opt_G, decay_rate=args.gamma)
                lr_decay(opt_D, decay_rate=args.gamma)
               
        # save last model
        state = {
            'epoch': epoch + 1,
            'opt_G_state': opt_G.state_dict(),
            'opt_D_state': opt_D.state_dict(),
            'loss_G': loss_G,
            'loss_D': loss_D,
            'G_state': G.state_dict(),
            'D_state': D.state_dict(),
        }
        checkpoint = os.path.join(save_dir, 'last_checkpoint.pkl')
        torch.save(state, checkpoint)
        print('[Checkpoint] {} is saved.'.format(checkpoint))
        logger.write('[Checkpoint] {} is saved.\n'.format(checkpoint))
        print('Training is done.')
        logger.write('Training is done.\n')
        logger.close()
        
    else:
        # generate images with the generator
        with torch.no_grad():
            val(G, n_sample=32)
            print('Testing is done.')

            
def train(data_loader, G, D, opt_G, opt_D, epoch, criterion):

    g_losses = averageMeter()
    d_losses = averageMeter()
    
    # setup training mode
    G.train()
    D.train()
    
    for (step, value) in enumerate(data_loader):
        image = value[0].cuda()
        bs = image.size(0)
        
        """ Train D """
        r_imgs = image.cuda()
        
        # generate fake images
        z = torch.randn(bs, args.z_dim).cuda()
        f_imgs = G(z)
            
        # label (1: real, 0: fake)  
        r_label = torch.ones((bs)).cuda()
        f_label = torch.zeros((bs)).cuda()
            
        # compute discriminator score
        r_logit = D(r_imgs)
        f_logit = D(f_imgs.detach())
            
        # compute loss
        r_loss = torch.mean(criterion(r_logit, r_label).squeeze())
        f_loss = torch.mean(criterion(f_logit, f_label).squeeze())
        loss_d = (r_loss + f_loss) / 2
        d_losses.update(loss_d.item(), bs)
            
        # backward
        D.zero_grad()
        loss_d.backward()
        opt_D.step()
    
        """ Train G """
        # generate fake images
        z = torch.randn(bs, args.z_dim).cuda()
        f_imgs = G(z)
        
        # compute discriminator score
        f_logit = D(f_imgs)
        
        # compute loss
        loss_g = torch.mean(criterion(f_logit, r_label).squeeze())
        g_losses.update(loss_g.item(), bs)
            
        # backward
        G.zero_grad()
        loss_g.backward()
        opt_G.step()
            
    # logging
    curr_lr_g = opt_G.param_groups[0]['lr']
    curr_lr_d = opt_D.param_groups[0]['lr']
    print('Epoch: [{}/{}]\t' \
        'LR_D: [{:.6g}]\t' \
        'LR_G: [{:.6g}]\t' \
        'Loss_D {d_loss.avg:.4f}\t' \
        'Loss_G {g_loss.avg:.4f}\n'.format(epoch + 1, args.epochs, curr_lr_d, curr_lr_g, d_loss=d_losses, g_loss=g_losses)
    )
    logger.write('Epoch: [{}/{}]\t' \
        'LR_D: [{:.6g}]\t' \
        'LR_G: [{:.6g}]\t' \
        'Loss_D {d_loss.avg:.4f}\t' \
        'Loss_G {g_loss.avg:.4f}\n'.format(epoch + 1, args.epochs, curr_lr_d, curr_lr_g, d_loss=d_losses, g_loss=g_losses)
    )
    
    return d_losses.avg, g_losses.avg

def val(G, epoch=None, n_sample=32):
    
    # setup evaluation mode
    G.eval()
    
    # generate fake images
    z = torch.randn(n_sample, args.z_dim).cuda()
    f_imgs_sample = (G(z).data + 1) / 2.0
    if epoch is not None:
        filename = os.path.join(save_dir, 'Epoch-{}.jpg'.format(epoch + 1))
    else:
        if args.out_img:
            filename = args.out_img
        else:
            filename = os.path.join(save_dir, 'fig2_2.png')
    save_image(f_imgs_sample, filename, nrow=8)
    print('Samples saved to {}.'.format(filename))

if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=5e-4, help='base learning rate (default: 5e-4)')
    parser.add_argument('--step', type=int, default=10, help='learning rate decay step (default: 10)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate step gamma (default: 0.1)')
    parser.add_argument('--z_dim', type=int, default=1024, help='dimension of latent vector (default: 1024)')
    parser.add_argument('--n_sample', type=int, default=32, help='number of samples (default: 32)')
    parser.add_argument('--test_dir', type=str, default='', help='testing image directory')
    parser.add_argument('--checkpoint', type=str, default='', help='pretrained model')
    parser.add_argument('--save_dir', type=str, default='checkpoint/GAN', help='directory to save logfile, checkpoint and output csv')
    parser.add_argument('--out_img', type=str, default='', help='path to output image')
    parser.add_argument('--data_root', type=str, default='face_data', help='data root')
    parser.add_argument('--phase', type=str, default='train', help='phase (train/test)')
    
    args = parser.parse_args()
    print(args)
    
    main()
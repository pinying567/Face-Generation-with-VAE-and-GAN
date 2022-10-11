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
from PIL import Image

from face_dataset import faceDataset
from VAE import VAE
from util import averageMeter, lr_decay, cvt_image
import math
        
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
        log = {'train': [], 'val': []}
        logger.write('{}\n'.format(args))
    
    # setup data loader for training images
    if args.phase == 'train':
        trans_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset_train = faceDataset(os.path.join(args.data_root, 'face/train'), trans_train)
        train_loader = data.DataLoader(dataset_train, shuffle=True, drop_last=False, pin_memory=True, batch_size=args.batch_size)
        print('train: {}'.format(dataset_train.__len__()))
        logger.write('train: {}\n'.format(dataset_train.__len__()))
        
    # setup data loader for validation/testing images
    trans_val = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if args.test_dir:
        dataset_val = faceDataset(args.test_dir, trans_val)
    else:
        dataset_val = faceDataset(os.path.join(args.data_root, 'face/test'), trans_val)
    
    print('val/test: {}'.format(dataset_val.__len__()))
    if args.phase == 'train':
        logger.write('val/test: {}\n'.format(dataset_val.__len__()))
        
    val_loader = data.DataLoader(dataset_val, shuffle=False, drop_last=False, batch_size=args.batch_size)
    
    # setup model
    model = VAE(nz=args.z_dim).cuda()
    if args.phase == 'train':
        logger.write('{}\n'.format(model))
   
    # setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    if args.phase == 'train':
        logger.write('{}\n'.format(optimizer))   
    
    # load checkpoint
    start_ep = 0
    best_mse = math.inf
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['opt_state'])
        start_ep = checkpoint['epoch']
        best_mse = checkpoint['mse']
        print("Loaded checkpoint '{}' (epoch: {}, mse: {:.4f})".format(args.checkpoint, start_ep, best_mse))
        
        if args.phase == 'train':
            logger.write("Loaded checkpoint '{}' (epoch: {}, mse: {:.4f})\n".format(args.checkpoint, start_ep, best_mse))
            if os.path.isfile(logfile):
                log = json.load(open(logfile, 'r'))

    if args.phase == 'train':
        
        # start training
        print('Start training from epoch {}'.format(start_ep))
        logger.write('Start training from epoch {}\n'.format(start_ep))
        for epoch in range(start_ep, args.epochs):
            
            mse_train, kld_train = train(train_loader, model, optimizer, epoch)
            log['train'].append([epoch + 1, mse_train, kld_train])
            
            if (epoch + 1) % args.val_ep == 0:
                with torch.no_grad():
                    mse_val, kld_val = val(val_loader, model)
                log['val'].append([epoch + 1, mse_val, kld_val])
                
                if mse_val < best_mse:
                    # save checkpoint
                    state = {
                        'epoch': epoch + 1,
                        'mse': mse_val,
                        'model_state': model.state_dict(),
                        'opt_state': optimizer.state_dict()
                    }
                    checkpoint = os.path.join(save_dir, 'best_model.pkl')
                    torch.save(state, checkpoint)
                    print('[Checkpoint] {} is saved.'.format(checkpoint))
                    logger.write('[Checkpoint] {} is saved.\n'.format(checkpoint))
                    json.dump(log, open(logfile, 'w'))
                    best_mse = mse_val
                    
            if (epoch + 1) % args.step == 0:
                lr_decay(optimizer, decay_rate=args.gamma)
        
        # save last model
        state = {
            'epoch': epoch + 1,
            'mse': mse_val,
            'model_state': model.state_dict(),
            'opt_state': optimizer.state_dict()
        }
        checkpoint = os.path.join(save_dir, 'last_checkpoint.pkl')
        torch.save(state, checkpoint)
        print('[Checkpoint] {} is saved.'.format(checkpoint))
        logger.write('[Checkpoint] {} is saved.\n'.format(checkpoint))
        print('Training is done.')
        logger.write('Training is done.\n')
        logger.close()
        
    else:
        with torch.no_grad():
            mse_val, kld_val = val(val_loader, model, save_result=args.save_feat)
            
            # generate images with the decoder
            model.eval()
            sample = torch.randn(32, args.z_dim).cuda()
            sample = model.decode(sample).data.cpu() * 0.5 + 0.5
            if args.out_img:
                save_image(sample, nrow=8, fp=args.out_img)
            else:
                save_image(sample, nrow=8, fp=os.path.join(save_dir, 'fig1_4.png'))
            
        print('Testing is done.')

            
def train(data_loader, model, optimizer, epoch):

    losses = averageMeter()
    MSE = averageMeter()
    KLD = averageMeter()
    
    # setup training mode
    model.train()
    
    for (step, value) in enumerate(data_loader):

        image = value[0].cuda()
        
        # forward
        recon_batch, mu, logvar = model(image)
        
        # compute loss
        mse = F.mse_loss(recon_batch, image, reduction='mean')
        kld = torch.mean(-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
        loss = mse + args.lamda * kld
        MSE.update(mse.item(), image.size(0))
        KLD.update(kld.item(), image.size(0))
        losses.update(loss.item(), image.size(0))
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # logging
    curr_lr = optimizer.param_groups[0]['lr']
    print('Epoch: [{}/{}]\t' \
        'LR: [{:.6g}]\t' \
        'Loss {loss.avg:.4f}\t' \
        'KLD {KLD.avg:.4f}\t' \
        'MSE {MSE.avg:.4f}'.format(epoch + 1, args.epochs, curr_lr, loss=losses, KLD=KLD, MSE=MSE)
    )
    logger.write('Epoch: [{}/{}]\t' \
        'LR: [{:.6g}]\t' \
        'Loss {loss.avg:.4f}\t' \
        'KLD {KLD.avg:.4f}\t'
        'MSE {MSE.avg:.4f}\n'.format(epoch + 1, args.epochs, curr_lr, loss=losses, KLD=KLD, MSE=MSE)
    )
    
    return MSE.avg, KLD.avg

    
def val(data_loader, model, save_result=False):

    losses = averageMeter()
    MSE = averageMeter()
    KLD = averageMeter()
       
    if args.save_feat:
        all_feat = []
        all_fname = []

    # setup evaluation mode
    model.eval()
    
    for (step, value) in enumerate(data_loader):

        image = value[0].cuda()
        
        # forward
        recon_batch, mu, logvar, latent_z = model(image, feat=True)
                   
        if args.save_feat:
            latent_z = latent_z.data.cpu().numpy()
            all_feat.extend(latent_z)
            all_fname.extend(value[1])
            
        # compute loss
        mse = F.mse_loss(recon_batch, image, reduction='none')
        kld = torch.mean(-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
        loss = torch.mean(mse) + args.lamda * kld
        MSE.update(torch.mean(mse).item(), image.size(0))
        KLD.update(kld.item(), image.size(0))
        losses.update(loss.item(), image.size(0))
        
        if save_result:
            if step == 0:
                f = open(os.path.join(save_dir, 'sample.csv'), 'w')
                f.write('sample_id,image_name,MSE\n')
                for i in range(args.n_sample):
                    sample_img = cvt_image(recon_batch[i, :, :, :].data.cpu())
                    fname = os.path.join(save_dir, 'sample_{}.png'.format(i))
                    Image.fromarray(sample_img).save(fname)
                    f.write('{},{},{}\n'.format(i, value[1][i], mse[i].mean().item()))
                f.close()

    # logging                        
    print('[Val] Loss {loss.avg:.4f}\tKLD {KLD.avg:.3f}\tMSE {MSE.avg:.3f}'.format(loss=losses, KLD=KLD, MSE=MSE))
    if args.phase == 'train':
        logger.write('[Val] Loss {loss.avg:.4f}\tKLD {KLD.avg:.3f}\tMSE {MSE.avg:.3f}\n'.format(loss=losses, KLD=KLD, MSE=MSE))

    if args.save_feat:
        np.save(os.path.join(save_dir, 'val_feat.npy'), np.asarray(all_feat))
        np.save(os.path.join(save_dir, 'val_fname.npy'), np.asarray(all_fname))
    
    return MSE.avg, KLD.avg


if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=5e-4, help='base learning rate (default: 5e-4)')
    parser.add_argument('--step', type=int, default=10, help='learning rate decay step (default: 10)')
    parser.add_argument('--val_ep', type=int, default=5, help='validation period (default: 5)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate step gamma (default: 0.1)')
    parser.add_argument('--lamda', type=float, default=1, help='weight for kld loss (default: 1)')
    parser.add_argument('--z_dim', type=int, default=1024, help='dimension of latent vector (default: 1024)')
    parser.add_argument('--n_sample', type=int, default=10, help='number of samples (default: 10)')
    parser.add_argument('--test_dir', type=str, default='', help='testing image directory')
    parser.add_argument('--checkpoint', type=str, default='', help='pretrained model')
    parser.add_argument('--save_dir', type=str, default='checkpoint/VAE', help='directory to save logfile, checkpoint and output csv')
    parser.add_argument('--out_img', type=str, default='', help='path to output image')
    parser.add_argument('--data_root', type=str, default='face_data', help='data root')
    parser.add_argument('--save_feat', type=bool, default=False, help='save features and corresponding labels for TSNE plot')
    parser.add_argument('--phase', type=str, default='train', help='phase (train/test)')
    
    args = parser.parse_args()
    print(args)
    
    main()
import argparse
import logging
from datetime import datetime
import pickle
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from network import MushroomNet
import matplotlib.pyplot as plt
from dataset import MushroomDataset

plt.ion()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 


if __name__ == '__main__':

    # Add Parameters from argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_path', type=str, default='resnet_result')
    parser.add_argument('--batch_size', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--has_checkpoint', action='store_true', default=False)
    parser.add_argument('--epoch_checkpoint', type=int, default=0)
    args = parser.parse_args()

    save_path = os.path.join(ROOT_DIR, args.save_path)
    # Create a directory if not exist.
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    dataset_dir = 'Mushrooms'
    train_csv_path = os.path.join(dataset_dir, 'train_test_split', 'train.csv')
    test_csv_path = os.path.join(dataset_dir, 'train_test_split', 'test.csv')
    
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds= [0.229, 0.224, 0.225]

    T = transforms.Compose([
                               transforms.Resize(500),
                               transforms.RandomRotation(20),
                               transforms.RandomHorizontalFlip(0.5),
                               transforms.RandomCrop(400, padding = 10),
                               transforms.ToTensor(),
                               transforms.Normalize(mean = pretrained_means, 
                                                    std = pretrained_stds)
                           ])

    train_dataset = MushroomDataset(csv_file=train_csv_path, root_dir=ROOT_DIR, transform=T)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_workers)

    test_dataset = MushroomDataset(csv_file=test_csv_path, root_dir=ROOT_DIR, transform=T)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_workers)

    print('training on dataset with %d classes'%(train_dataset.num_classes))
    net = MushroomNet(train_dataset.num_classes).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(args.device)

    # Read models from checkpoint
    fname_checkpoint = os.path.join(save_path, 'chckpt_%i.pt' % args.epoch_checkpoint)
    if args.epoch_checkpoint > 1:
        print("Loading checkpoint from path: " + fname_checkpoint)
        checkpoint = torch.load(fname_checkpoint, map_location=args.device)
        net.load_state_dict(checkpoint['state_dict'])

        loss_train = checkpoint['loss_train']
        loss_test = checkpoint['loss_test']
        epoch_checkpoint = checkpoint['epoch']
    else:
        loss_train = []
        loss_test = []
        epoch_checkpoint = 1

    viz_data = False
    if viz_data:
        fig, ax = plt.subplots(1, args.batch_size)

    loss_fig, loss_ax = plt.subplots(1,1)

    num_batch_train = train_dataset.__len__() / args.batch_size
    num_batch_test = test_dataset.__len__() / args.batch_size

    for epoch in range(epoch_checkpoint, epoch_checkpoint + args.num_epochs):
        train_stat, test_stat = np.zeros(2).astype('float'), np.zeros(2).astype('float')

        net.train()
        # train cycle
        epoch_loss = []
        for i, sample in enumerate(train_data_loader):
            data = sample['data'].to(args.device)
            label = sample['label'].to(args.device)

            # visualize dataset
            if viz_data:
                for i in range(args.batch_size):
                    ax[i].imshow(data[i].detach().cpu().numpy().transpose(1,2,0))
                    ax[i].xaxis.set_visible(False)
                    ax[i].yaxis.set_visible(False)
                    ax[i].set_title(sample['name'][i])
                plt.pause(0.01)

            optim.zero_grad()
            pred = net(data)
            loss = criterion(pred, label)
            loss.backward()

            for param in net.parameters():
                param.grad.data.clamp_(-.1, .1)
            optim.step()

            # update the accuracy
            class_pred = np.argmax(pred.detach().cpu().numpy(), axis=1)
            result = class_pred == label.detach().cpu().numpy()
            train_stat[0] += np.sum(result)
            train_stat[1] += result.shape[0]
            # record the loss
            epoch_loss.append(loss.item())
            print('[%6d: %6d/%6d] train loss: %f, training accuracy: %f' % \
                    (epoch, i, num_batch_train, loss.item(), train_stat[0]/train_stat[1]))


        loss_train.append(np.average(epoch_loss))

        net.eval()
        # test cycle
        epoch_loss = []
        for i, sample in enumerate(test_data_loader):
            data = sample['data'].to(args.device)
            label = sample['label'].to(args.device)

            pred = net(data)
            loss = criterion(pred, label)

            # update the accuracy
            class_pred = np.argmax(pred.detach().cpu().numpy(), axis=1)
            result = class_pred == label.detach().cpu().numpy()
            test_stat[0] += np.sum(result)
            test_stat[1] += result.shape[0]
            # record the loss
            epoch_loss.append(loss.item())
            print('[%6d: %6d/%6d] test loss: %f, testing accuracy: %f' % \
                    (epoch, i, num_batch_train, loss.item(), test_stat[0]/test_stat[1]))

        loss_test.append(np.average(epoch_loss))

        # plot result
        loss_ax.plot(loss_train, 'r')
        loss_ax.plot(loss_test, 'b')
        plt.pause(0.01)

        if (epoch) % args.save_every == 0:
            torch.save({    
                    'epoch': epoch,
                    'state_dict': net.state_dict(),
                    'loss_train': loss_train,
                    'loss_test': loss_test
                }, '%s/chckpt_%i.pt' % (save_path, epoch))
            loss_fig.savefig('%s/loss_%d.png'% (save_path, epoch))

        
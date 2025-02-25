# -*- coding: utf-8 -*-
"""
========================================================================
ForkNet for DoFP sensor to reconstruct s0, dolp and aop, Version 1.0
Copyright(c) 2020 Xianglong Zeng, Yuan Luo, Xiaojing Zhao, Wenbin Ye
All Rights Reserved.
----------------------------------------------------------------------
Permission to use, copy, or modify this software and its documentation
for educational and research purposes only and without fee is here
granted, provided that this copyright notice and the original authors'
names appear on all copies and supporting documentation. This program
shall not be used, rewritten, or adapted as the basis of a commercial
software or hardware product without first obtaining permission of the
authors. The authors make no representations about the suitability of
this software for any purpose. It is provided "as is" without express
or implied warranty.
----------------------------------------------------------------------
Please cite the following paper when you use it:

Xianglong Zeng, Yuan Luo, Xiaojing Zhao, and Wenbin Ye, "An end-to-end 
fully-convolutional neural network for division of focal plane sensors 
to reconstruct S0, DoLP, and AoP," Opt. Express 27, 8566-8577 (2019)
========================================================================
"""

import numpy as np
import h5py
from model import MSE_LOSS, LOSS, smooth_loss
from utils.batch_generator import patch_batch_generator
from utils.utils import dolp, psnr, normalize, aop, gs_rand_choice
import matplotlib.pyplot as plt 
import os
import math
import csv
from tqdm import tqdm

import torch
import torch.optim as optim
from models import *
# from skimage.measure import compare_ssim

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
#FINE_TUNE = False
LEARNING_RATE = 0.001
LEARNING_RATE_DECAY_STEPS = 600
LEARNING_RATE_DECAY_RATE = 0.988
IMG_NUM = 110
EPOCH_NUM = 300
BATCH_SIZE = 16
PATCH_WIDTH = 40
PATCH_HEIGHT = 40
# GPUS = "2"
DSP_ITV = 6
metrics = 'training loss'
save_best = True
early_stop = False
patient = 5
train_img_index_path = './list/train_image_index.list'
val_img_index_path = './list/val_image_index.list'
Y_path = './data/training_set/Y.h5'
labels_path = './data/training_set/Labels.h5'
BIC_path = './data/training_set/BIC.h5'
ckpt_path = './best_model/model_1/model_1.ckpt'
csv_path = './list/psnr_record_1.csv'

#------------------------------------------------------------------------------
def load_data(batch_size = BATCH_SIZE, train_img_index_path = train_img_index_path,
              val_img_index_path = val_img_index_path, Y_path = Y_path, labels_path = labels_path):
    '''
    Divide the training set and validation set.
    Return two generator to generate batches of data.
    '''
    # read data from h5 file
    with h5py.File(Y_path, 'r') as h1:
        Y = np.array(h1.get('inputs'))
    print("XXXX1")
    with h5py.File(labels_path, 'r') as h2:
        label = np.array(h2.get('labels'))
    print("XXXX2")

    with h5py.File(BIC_path, 'r') as h3:
        bic = np.array(h3.get('bic'))
    print("XXXX3")

    Y = Y[:1000]
    label = label[:1000]
    bic = bic[:1000]
    Input = np.concatenate((Y, bic), axis=-1)

    patch_num = Y.shape[0]
    patch_num_per_img = patch_num // IMG_NUM

    train_img_index_str = open(train_img_index_path).read()
    train_img_index = [int(idx) for idx in train_img_index_str.split(',') if int(idx) < IMG_NUM]
    # train_img_num = len(train_img_index)

    val_img_index_str = open(val_img_index_path).read()
    val_img_index = [int(idx) for idx in val_img_index_str.split(',') if int(idx) < IMG_NUM]
    # val_img_num = len(val_img_index)

    patch_index_train = np.concatenate(
        [np.arange(i * patch_num_per_img, (i + 1) * patch_num_per_img) for i in train_img_index])
    patch_index_val = np.concatenate(
        [np.arange(i * patch_num_per_img, (i + 1) * patch_num_per_img) for i in val_img_index])

    # patch_index_train = np.concatenate(
    #     [gs_rand_choice(i * patch_num_per_img, (i + 1) * patch_num_per_img, 192) for i in train_img_index])
    # patch_index_val = np.concatenate(
    #     [gs_rand_choice(i * patch_num_per_img, (i + 1) * patch_num_per_img, 192) for i in val_img_index])

    # patch_index_train = np.concatenate(
    #     [np.random.choice(np.arange(i * patch_num_per_img, (i + 1) * patch_num_per_img), 192) for i in train_img_index])
    # patch_index_val = np.concatenate(
    #     [np.random.choice(np.arange(i * patch_num_per_img, (i + 1) * patch_num_per_img), 192) for i in val_img_index])

    patch_num_train = len(patch_index_train)
    # training steps in one epoch
    train_steps = int(np.ceil(patch_num_train * 1. / batch_size))
    print('# Training Patches: {}.'.format(patch_num_train))

    patch_num_val = len(patch_index_val)
    # validation steps in one epoch
    val_steps = int(np.ceil(patch_num_val * 1. / batch_size))
    print('# Validation Patches: {}.'.format(patch_num_val))

    train_Y = Input[patch_index_train]
    train_label = label[patch_index_train]
    val_Y = Input[patch_index_val]
    val_para = label[patch_index_val]
    # val_bic = bic[patch_index_val]

    return train_steps, train_Y, train_label, val_steps, val_Y, val_para

#------------------------------------------------------------------------------
def train(device, patch_width = PATCH_WIDTH, patch_height = PATCH_HEIGHT, epoch_num = EPOCH_NUM, batch_size = BATCH_SIZE,  learning_rate = LEARNING_RATE,
          learning_rate_decay_steps = LEARNING_RATE_DECAY_STEPS, learning_rate_decay_rate = LEARNING_RATE_DECAY_RATE,
          dsp_itv = DSP_ITV, ckpt_path = ckpt_path, save_best = save_best, early_stop = early_stop):
    '''
    Difine the tensorflow graph, execute training and validation.
    '''
    
    
    train_steps, train_Y, train_label, val_steps, val_Y, val_label = load_data()
    # val_s0 = val_para[:, :, :, :1]
    # val_dolp = val_para[:, :, :, 1:2]
    # val_aop = val_para[:, :, :, 2:]
#    print(np.max(val_Y))
    model = Model()
    model.to(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                            betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch_num, eta_min=1e-6)

    # with tf.Session() as sess:
    if True:
        print('\nStart Training!')
        # sess.run(init)
        min_loss = np.inf
        wait = 0
        psnr_record = []
        total_S0_PSNR_BIC = 0
        total_DoLP_PSNR_BIC = 0
        total_AoP_PSNR_BIC = 0

        for epoch in range(epoch_num):          
            #training set batch generator   
            train_generator = patch_batch_generator(train_Y, train_label, batch_size, patch_width, patch_height, random_shuffle = True)

             #test set batch generator   
            val_generator = patch_batch_generator(val_Y, val_label, batch_size, patch_width, patch_height, random_shuffle=False, augment=False)
            
            print('=======================================Epoch:{}/{}======================================='.format(epoch, epoch_num))
            # training
            total_train_loss = 0
            # Create a tqdm progress bar
            model.train()
            with tqdm(range(train_steps), desc="Training", ncols=100) as pbar:
                for step in pbar:
                    (Input_batch_train, Para_batch_train) = next(train_generator)
                    Input_batch_train = Input_batch_train.to(device)
                    Para_batch_train = Para_batch_train.to(device)

                    Y_batch_train = Input_batch_train[:, :, :, :1]
                    Y_batch_train = torch.cat([Y_batch_train] * 3, dim=-1)
                    Y_batch_train = Y_batch_train.permute(0, -1, 1, 2).contiguous()
                    S0_batch_train = Para_batch_train[:,:,:,:1]
                    DoLP_batch_train = Para_batch_train[:,:,:,1:2]
                    AoP_batch_train = Para_batch_train[:,:,:,2:]

                    S0_hat, DoLP_hat, AoP_hat = model(Y_batch_train)
                    S0_hat = S0_hat.permute(0, 2, 3, 1).max(dim=-1, keepdim=True).values
                    DoLP_hat = DoLP_hat.permute(0, 2, 3, 1).max(dim=-1, keepdim=True).values
                    AoP_hat = AoP_hat.permute(0, 2, 3, 1).max(dim=-1, keepdim=True).values

                    # sess.run(train_step, feed_dict={Y:Y_batch_train, S0:S0_batch_train, DoLP:DoLP_batch_train, AoP:AoP_batch_train})
                    # train_loss = sess.run(loss, feed_dict={Y:Y_batch_train, S0:S0_batch_train, DoLP:DoLP_batch_train, AoP:AoP_batch_train})
                    train_loss = LOSS(S0_hat, S0_batch_train, DoLP_hat, DoLP_batch_train, AoP_hat, AoP_batch_train)
                    total_train_loss += train_loss

                    # backward
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()

                    pbar.set_postfix(loss=train_loss)

            scheduler.step()
                    
            # validation

            total_val_loss = 0
            total_S0_PSNR = 0
            total_DoLP_PSNR = 0
            total_AoP_PSNR = 0

            model.eval()
            for step in range(val_steps):
                (Input_batch_val, Para_batch_val) = next(val_generator)
                Input_batch_val = Input_batch_val.to(device)
                Para_batch_val = Para_batch_val.to(device)

                Y_batch_val = Input_batch_val[:, :, :, :1]
                Y_batch_val = torch.cat([Y_batch_val] * 3, dim=-1)
                Y_batch_val = Y_batch_val.permute(0, -1, 1, 2).contiguous()
                BIC_batch_val = Input_batch_val[:, :, :, 1:]
                S0_batch_val = Para_batch_val[:, :, :, :1]
                DoLP_batch_val = Para_batch_val[:, :, :, 1:2]
                AoP_batch_val = Para_batch_val[:, :, :, 2:]

                S0_hat_val, DoLP_hat_val, AoP_hat_val = model(Y_batch_val)
                S0_hat_val = S0_hat_val.permute(0, 2, 3, 1).max(dim=-1, keepdim=True).values
                DoLP_hat_val = DoLP_hat_val.permute(0, 2, 3, 1).max(dim=-1, keepdim=True).values
                AoP_hat_val = AoP_hat_val.permute(0, 2, 3, 1).max(dim=-1, keepdim=True).values
                # total_val_loss += sess.run(loss, feed_dict={Y: Y_batch_val, S0:S0_batch_val, DoLP:DoLP_batch_val, AoP:AoP_batch_val})
                # S0_hat_val, DoLP_hat_val, AoP_hat_val = sess.run([S0_hat, DoLP_hat, AoP_hat], feed_dict={Y:Y_batch_val})
                #limit the value
                val_loss = LOSS(S0_hat_val, S0_batch_val, DoLP_hat_val, DoLP_batch_val, AoP_hat_val, AoP_batch_val)
                total_val_loss += val_loss

                S0_hat_val = S0_hat_val.cpu().detach().numpy()
                S0_hat_val = np.clip(S0_hat_val, 0, 2)
                DoLP_hat_val = DoLP_hat_val.cpu().detach().numpy()
                DoLP_hat_val = np.clip(DoLP_hat_val, 0, 1)
                AoP_hat_val = AoP_hat_val.cpu().detach().numpy()
                # AoP_hat_val = np.clip(AoP_hat_val, 0, math.pi)
                # DoLP_hat_val = Normalize(DoLP_hat_val, 0, 1)
                S0_batch_val = S0_batch_val.cpu().detach().numpy()
                DoLP_batch_val = DoLP_batch_val.cpu().detach().numpy()
                AoP_batch_val = AoP_batch_val.cpu().detach().numpy()
                total_S0_PSNR += psnr(S0_batch_val[:, :, :, 0], S0_hat_val[:, :, :, 0], 2)
                total_DoLP_PSNR += psnr(DoLP_batch_val[:, :, :, 0], DoLP_hat_val[:, :, :, 0], 1)
                total_AoP_PSNR += psnr(AoP_batch_val[:, :, :, 0], AoP_hat_val[:, :, :, 0], math.pi / 2.)
                # for b in range(AoP_batch_val.shape[0]):
                #     total_AoP_PSNR += compare_ssim(np.float32(AoP_batch_val[b, :, :, 0]), np.float32(AoP_hat_val[b, :, :, 0]), data_range=math.pi / 2.)

                BIC_batch_val = BIC_batch_val.cpu().detach().numpy()
                if epoch == 0:
    #                print('max:', max(val_bic[0,6:-6,6:-6,0]))
                    S0_BIC = (BIC_batch_val[:,:,:,0] + BIC_batch_val[:,:,:,1] + BIC_batch_val[:,:,:,2] + BIC_batch_val[:,:,:,3]) / 2.
                    DoLP_BIC = dolp(BIC_batch_val[:,:,:,0], BIC_batch_val[:,:,:,1], BIC_batch_val[:,:,:,2], BIC_batch_val[:,:,:,3])
                    AoP_BIC = aop(BIC_batch_val[:,:,:,0], BIC_batch_val[:,:,:,1], BIC_batch_val[:,:,:,2], BIC_batch_val[:,:,:,3]) + math.pi / 4.  #avoid the minus number
                    total_S0_PSNR_BIC += psnr(S0_batch_val[:, :, :, 0], S0_BIC, 2)
                    total_DoLP_PSNR_BIC += psnr(DoLP_batch_val[:, :, :, 0], DoLP_BIC, 1)
                    total_AoP_PSNR_BIC += psnr(AoP_batch_val[:, :, :, 0], AoP_BIC, math.pi / 2.)

                    # for b in range(AoP_batch_val.shape[0]):
                    #     total_AoP_PSNR_BIC += compare_ssim(np.float32(AoP_batch_val[b, :, :, 0]), np.float32(AoP_BIC[b]), data_range=math.pi / 2.)
                
            print('========================================Validation=======================================' +
                  '\nTraining loss: %.5f' % (total_train_loss/train_steps) + 
                  '\nValidation loss: %.5f' % (total_val_loss/val_steps) +
                  '\n ————————————————————————————————————————————————————————————————————————————————' + 
#                  '\n| PSNR of I_0 using PDCNN: %.5f    |   PSNR of I_0 using BICUBIC: %.5f   |' % (total_X_0_PSNR/val_steps, 32.872) + 
#                  '\n| PSNR of I_45 using PDCNN: %.5f   |   PSNR of I_45 using BICUBIC: %.5f  |' % (total_X_45_PSNR/val_steps, 32.973) + 
#                  '\n| PSNR of I_90 using PDCNN: %.5f   |   PSNR of I_90 using BICUBIC: %.5f  |' % (total_X_90_PSNR/val_steps, 33.008) + 
#                  '\n| PSNR of I_135 using PDCNN: %.5f  |   PSNR of I_135 using BICUBIC: %.5f |' % (total_X_135_PSNR/val_steps, 32.923) + 
                  '\n| PSNR of S_0 using SRCNN: %.5f    |   PSNR of S_0 using BICUBIC: %.5f   |' % (total_S0_PSNR/val_steps, total_S0_PSNR_BIC/val_steps) +
                  '\n| PSNR of DoLP using SRCNN: %.5f   |   PSNR of DoLP using BICUBIC: %.5f  |' % (total_DoLP_PSNR/val_steps, total_DoLP_PSNR_BIC/val_steps) +
                  '\n| PSNR of AoP using SRCNN: %.5f    |   PSNR of AoP using BICUBIC: %.5f   |' % (total_AoP_PSNR/val_steps, total_AoP_PSNR_BIC/val_steps) +
                  '\n ————————————————————————————————————————————————————————————————————————————————')

            psnr_record.append([total_S0_PSNR / val_steps, total_DoLP_PSNR / val_steps, total_AoP_PSNR / val_steps])
            
            if save_best or early_stop:
                if metrics == 'validation loss':
                    current_loss = total_val_loss/val_steps
                elif metrics == 'training loss':
                    current_loss = total_train_loss/train_steps
                if current_loss < min_loss:      
                    print('Validation loss decreased from %.5f to %.5f' % (min_loss, current_loss))
                    min_loss = current_loss
                    if save_best:
                        torch.save(model, ckpt_path)
                        print("Model saved in file: %s" % ckpt_path)
                    if early_stop:   
                        wait = 0
                else:
                    print('Validation loss did not decreased.')
                    if early_stop:   
                        wait += 1
                        if wait > patient:
                            print('Early stop!')
                            break                         
        if not save_best:
            torch.save(model, ckpt_path)
            print("Model saved in file: %s" % ckpt_path)

        with open(csv_path,'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(psnr_record)
                       
if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = GPUS
    # tf.reset_default_graph()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train(device=device)
    

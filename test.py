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
import random
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from model import ForkNet, LOSS
from utils.utils import dolp, psnr, normalize, view_bar, aop, pad_shift, count_para, plot_feature_map, fig2array
import imageio as imgio
import os
import math
import time
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.measure import compare_ssim
from newton_polynomial_inteprolation import interpolate as NPI

IMG_ALL = 10
IMG_NUM = 10
IMG_WIDTH = 1280
IMG_HEIGHT = 960
output_images=[]
vis_feature_map = False
hot_map = True
plot_dir = './images/feature_maps/'
test_img_path = './data/test_set'
model_path = './best_model/model_1/model_1.ckpt'
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

newton_img = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH, 4], np.float32)
bic_s0 = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)
origin_s0 = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)
pred_s0 = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)
bic_dolp = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)
origin_dolp = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)
pred_dolp = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)
bic_aop = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)
origin_aop = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)
pred_aop = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)

# define the index for downsampling
mns = [(0, 0), (0, 1), (1, 1), (1, 0)]

tf.reset_default_graph()

Y = tf.placeholder(tf.float32, [None, None, None, 1], name='Y')
S0 = tf.placeholder(tf.float32, [None, None, None, 1], name='S0')
DoLP = tf.placeholder(tf.float32, [None, None, None, 1], name='DoLP')
AoP = tf.placeholder(tf.float32, [None, None, None, 1], name='AoP')
#    Para = tf.placeholder(tf.float32, [None, None, None, 3])#define tensors of input and label

# DoLP_hat= srcnn_ete(Y)
S0_hat, DoLP_hat, AoP_hat = ForkNet(Y, padding='SAME')

with tf.Session() as sess:
    saver = tf.train.Saver(var_list=tf.global_variables())
    saver.restore(sess,model_path)

    count_para()

    total_S0_PSNR = np.zeros((IMG_NUM))
    total_S0_PSNR_BIC = np.zeros((IMG_NUM))
    total_DoLP_PSNR = np.zeros((IMG_NUM))
    total_DoLP_PSNR_BIC = np.zeros((IMG_NUM))
    total_AoP_PSNR = np.zeros((IMG_NUM))
    total_AoP_PSNR_BIC = np.zeros((IMG_NUM))
    total_S0_PSNR_NEWTON = np.zeros((IMG_NUM))
    total_DoLP_PSNR_NEWTON = np.zeros((IMG_NUM))
    total_AoP_PSNR_NEWTON = np.zeros((IMG_NUM))
    total_time = 0
    random.seed(100)
    numbers = random.sample(range(IMG_ALL), IMG_NUM)
    for i in range(0, IMG_NUM):
        si = numbers[i]
        tic = time.time()
        for j in range(0, 4):
            path_origin = test_img_path + '/image_{}_{}.bmp'.format(si + 1, j * 45)
            if not os.path.exists(path_origin):
                path_origin = test_img_path + '/image_{}_{}.jpg'.format(si + 1, j * 45)
            print("=======================")
            print(path_origin)
            img = np.array(Image.open(path_origin).convert('L'), np.float32) / 255.
            IMG_HEIGHT, IMG_WIDTH = img.shape
            if j == 0:
                msc_img = np.zeros([IMG_HEIGHT, IMG_WIDTH, 1], np.float32)
                bic_img = np.zeros([IMG_HEIGHT, IMG_WIDTH, 4], np.float32)
                origin_img = np.zeros([IMG_HEIGHT, IMG_WIDTH, 4], np.float32)
            downimg = img[mns[j][0]::2, mns[j][1]::2]
            msc_img[mns[j][0]::2, mns[j][1]::2, 0] = downimg
            bic_img[..., j] = cv2.resize(downimg, (IMG_WIDTH, IMG_HEIGHT), cv2.INTER_CUBIC)
            origin_img[..., j] = img

        bic_img=pad_shift(bic_img)
        # newton_img[i] = NPI(msc_img[i, ..., 0])

        start = time.time()
        S0_hat_test, DoLP_hat_test, AoP_hat_test = sess.run([S0_hat, DoLP_hat, AoP_hat], feed_dict={Y: msc_img[np.newaxis]})
        end = time.time()
        total_time += end - start

        # the s0, dolp and aop of predict images
        S0_hat_test = np.clip(S0_hat_test[0, :, :, 0], 0, 2)
        DoLP_hat_test = np.clip(DoLP_hat_test[0, :, :, 0], 0, 1)
        # AoP_hat_test = AoP_hat_test[0, :, :, 0]
        # print(np.max(AoP_hat_test), np.min(AoP_hat_test))
        AoP_hat_test = np.clip(AoP_hat_test[0, :, :, 0], 0, math.pi/2)
        # AoP_hat_test = Normalize(AoP_hat_test[0, :, :, 0], math.pi / 2., 0)

        #the s0, dolp and aop of original images
        S0_true = 0.5 * (origin_img[..., 0] + origin_img[..., 1] + origin_img[..., 2] + origin_img[..., 3])
        DoLP_true = dolp(origin_img[..., 0], origin_img[..., 1], origin_img[..., 2], origin_img[..., 3])
        AoP_true = aop(origin_img[..., 0], origin_img[..., 1], origin_img[..., 2], origin_img[..., 3]) + math.pi/4.

        #the dolp of bic images
        S0_BIC = 1 / 2 * (bic_img[..., 0] + bic_img[..., 1] + bic_img[..., 2] + bic_img[..., 3])
        DoLP_BIC = dolp(bic_img[..., 0], bic_img[..., 1], bic_img[..., 2], bic_img[..., 3])
        AoP_BIC = aop(bic_img[..., 0], bic_img[..., 1], bic_img[..., 2], bic_img[..., 3]) + math.pi / 4.

        # #the dolp of newton images
        # S0_NEWTON = 1 / 2 * (newton_img[i, :, :, 0] + newton_img[i, :, :, 1] + newton_img[i, :, :, 2] + newton_img[i,:, :, 3])
        # DoLP_NEWTON = dolp(newton_img[i, :, :, 0], newton_img[i, :, :, 1], newton_img[i, :, :, 2], newton_img[i, :, :, 3])
        # AoP_NEWTON = aop(newton_img[i, :, :, 0], newton_img[i, :, :, 1], newton_img[i, :, :, 2], newton_img[i, :, :, 3]) + math.pi / 4.

        # Calculate the PSNR of S0, DoLP and AoP obtained through PDCNN method
        total_S0_PSNR[i]=  psnr(S0_true, S0_hat_test, 2)
        total_DoLP_PSNR[i] = psnr(DoLP_true, DoLP_hat_test, 1)
        total_AoP_PSNR[i] = psnr(AoP_true, AoP_hat_test, math.pi/2.)

        # Calculate the PSNR of S0, DoLP and AoP obtained through BICUBIC method
        total_S0_PSNR_BIC[i] = psnr(S0_true, S0_BIC, 2)
        total_DoLP_PSNR_BIC[i] = psnr(DoLP_true, DoLP_BIC, 1)
        total_AoP_PSNR_BIC[i] = psnr(AoP_true, AoP_BIC, math.pi/2.)

        # # Calculate the PSNR of S0, DoLP and AoP obtained through NEWTON method
        # total_S0_PSNR_NEWTON[i] = psnr(S0_true, S0_NEWTON, 2)
        # total_DoLP_PSNR_NEWTON[i] = psnr(DoLP_true, DoLP_NEWTON, 1)
        # total_AoP_PSNR_NEWTON[i] = psnr(AoP_true, AoP_NEWTON, math.pi/2.)

        # show the progress bar
        view_bar(i, IMG_NUM)
        toc = time.time()
        print("\nTIME SPEND", toc - tic)

    print('\n========================================Testing=======================================' +
          '\n ————————————————————————————————————————————————————————————————————————————————' +
          '\n| PSNR of S_0 using SRCNN: %.5f    |   PSNR of S_0 using BICUBIC: %.5f   |' % (np.mean(total_S0_PSNR), np.mean(total_S0_PSNR_BIC)) +
          '\n| PSNR of DoLP using PDCNN: %.5f   |   PSNR of DoLP using BICUBIC: %.5f  |' % (np.mean(total_DoLP_PSNR), np.mean(total_DoLP_PSNR_BIC)) +
          '\n| PSNR of AoP using SRCNN: %.5f    |   PSNR of AoP using BICUBIC: %.5f   |' % (np.mean(total_AoP_PSNR), np.mean(total_AoP_PSNR_BIC)) +
          '\n ————————————————————————————————————————————————————————————————————————————————')

    # print('\n| PSNR of S_0 using NEWTON: %.5f   |' % (np.mean(total_S0_PSNR_NEWTON)) +
    #       '\n| PSNR of DoLP using NEWTON: %.5f  |' % (np.mean(total_DoLP_PSNR_NEWTON)) +
    #       '\n| PSNR of AoP using NEWTON: %.5f   |' % (np.mean(total_AoP_PSNR_NEWTON)) +
    #       '\n ————————————————————————————————————————————————————————————————————————————————')

    print('\nSRCNN time: {} sec'.format(total_time / IMG_NUM))

    for j in output_images:

        imgio.imsave("./images/bic_s0_{}_{}.jpg".format(j, total_S0_PSNR_BIC[j-1]), bic_s0[j-1, 390:-470, 690:-490])
        imgio.imsave("./images/pred_s0_3_path_{}_{}.jpg".format(j, total_S0_PSNR[j-1]), pred_s0[j-1, 200:-80, 325:-275])
        imgio.imsave("./images/org_s0_{}.jpg".format(j), origin_s0[j-1, 390:-470, 690:-490])

        imgio.imsave("./images/bic_dolp_{}_{}.jpg".format(j, total_DoLP_PSNR_BIC[j - 1]), bic_dolp[j - 1, 285:-575, 865:-315])
        imgio.imsave("./images/pred_dolp_3_path_{}_{}.jpg".format(j, total_DoLP_PSNR[j - 1]), pred_dolp[j - 1, 200:-80, 325:-275])
        imgio.imsave("./images/org_dolp_{}.jpg".format(j), origin_dolp[j - 1, 285:-575, 865:-315])

        imgio.imsave("./images/bic_aop_{}_{}.jpg".format(j, total_AoP_PSNR_BIC[j - 1]), bic_aop[j - 1, 200:-80, 325:-275])
        imgio.imsave("./images/pred_aop_3_path_{}_{}.jpg".format(j, total_AoP_PSNR[j - 1]), pred_aop[j - 1, 200:-80, 325:-275])
        imgio.imsave("./images/org_aop_{}.jpg".format(j), origin_aop[j - 1, 200:-80, 325:-275])

        if hot_map:
            plt.axis('off')
            fig = plt.figure()
            fig = plt.gcf()
            height, width = bic_aop[j - 1, 200:-80, 325:-275].shape
            fig.set_size_inches(width/300, height/300)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)

            plt.imshow(bic_aop[j - 1, 200:-80, 325:-275], cmap=cm.jet)
            plt.savefig('./images/bic_aop.jpg', dpi=300)

            plt.imshow(pred_aop[j - 1, 200:-80, 325:-275],  cmap=cm.jet)
            plt.savefig('./images/pred_aop_0.05_0.0180_log.jpg', dpi=300)

            plt.imshow(origin_aop[j - 1, 200:-80, 325:-275], cmap=cm.jet)
            plt.savefig('./images/org_aop.jpg', dpi=300)

            # imgio.imsave("./images/msc_img_{}.jpg".format(j), msc_img[j - 1, 320:-320, 410:-410, :])

        if vis_feature_map:
            visualize_layers = ['x_1', 'x_2', 'x_3_1', 'x_3_2', 'x_3_3']
            conv_out = sess.run(tf.get_collection('feature_maps'), feed_dict={Y: msc_img[j-1:j]})
            for m, layer in enumerate(visualize_layers):
                if not os.path.exists(plot_dir + layer):
                    os.mkdir(plot_dir + layer)
                plot_feature_map(conv_out[m][:, 320:-320, 410:-410], plot_dir + layer, maps_all=True)


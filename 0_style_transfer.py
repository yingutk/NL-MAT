from tensorflow.python.client import device_lib
from utils_rgb import *
from pbAuto_transfer_one_structure import*
# from pbAuto_transfer_decoder import*

# from pbAuto_transfer_stack_structure import*

import time
import os
import argparse

# Written by Ying Qu <yqu3@vols.utk.edu>
# This code is a demo code for our paper
# “Non-local Representation based Mutual Affine-Transfer Network for Photorealistic Stylization”, TPAMI 2021
# The code is for research purpose only
# All Rights Reserved

# If you use the code, please cite the following paper
# Ying Qu, Zhenzhou Shao and Hairong Qi.
# “Non-local Representation based Mutual Affine-Transfer Network for Photorealistic Stylization”,
# IEEE Transactions on Pattern Analysis and Machine Intelligence，July 2021.

parser = argparse.ArgumentParser(description='NL-MAT for photorealistic sytle transfer')
parser.add_argument('--cuda', default='0', help='Choose GPU.')
parser.add_argument('--content', default='data/content1/in', help='Content Path.')
parser.add_argument('--style', default='data/style1/in', help='Style Path.')
parser.add_argument('--datanum', default='data', help='data Name.')
parser.add_argument('--format', default='.png', help='Image format.')
parser.add_argument('--filenum', type=int, default=10, help='Image Name.')
parser.add_argument('--load_path', default='_single_', help='Model Path.')
parser.add_argument('--output_path', default='output_single_data/')
parser.add_argument('--nhlayer', type=int, default=20, help='First hidden layer')
parser.add_argument('--nh2layer', type=int, default=10, help='Second hidden layer')
parser.add_argument('--mu', type=float32, default=0.1, help='mutual constraint')
parser.add_argument('--sp', type=float32, default=0.001, help='sparse constraint')
parser.add_argument('--sr', type=int, default=8, help='Downsample factor')
parser.add_argument('--lrate', type=float32, default=0.001, help='learning rate')
parser.add_argument('--epoch', type=int, default=8000, help='Maximum epoch')
parser.add_argument('--tol', type=float32, default=1.0, help='Stop criterion')
parser.add_argument('--training', type=int, default=0, help='Training 1 or Transfer 0')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']= args.cuda
tf.logging.set_verbosity(tf.logging.ERROR)

def main():
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    local_device_protos = device_lib.list_local_devices()
    print(local_device_protos)
    loadLRonly = True
    loadLRonly = False

    content_lrate = args.lrate
    style_rate = 1
    maxiter = args.epoch
    tol = args.tol
    vol_r = 0.0001
    mu_r = args.mu
    sp_r = args.sp
    num_h1 = args.nhlayer
    num_h2 = args.nh2layer
    sr = args.sr
    ly = 3

    nNetLevel = [ly,ly,ly,ly,ly,ly,ly,ly,ly]

    file_content = args.content +str(args.filenum) + args.format
    file_style = args.style +str(args.filenum) + args.format

    print(file_content)
    print(file_style)
    load_path = args.datanum + '_' + str(args.filenum) + args.load_path + str(num_h1) +'_' + str(num_h2)  + '_m' + str(mu_r) + '_s'+str(sp_r)+ '_sr'+str(args.sr)+'/'
    save_path = args.datanum + '_' + str(args.filenum) + args.load_path + str(num_h1) +'_' + str(num_h2)  + '_m' + str(mu_r) + '_s'+str(sp_r)+ '_sr'+str(args.sr)+'/'
    img_dir = args.output_path

    print('image pair '+str(args.filenum) + ' is processing')

    data = readData(file_content, file_style, args.sr)
    data.mark = args.load_path

    # betapan(input data,rate for content, rate for style,
    # network level,
    # maximum epoch, is_adam,
    # volume rate, entropy rate,
    # number of hidden layer 1, number of hidden layer 2, downsmaple sacle, configuration)
    auto = betapan(data, content_lrate, style_rate,
                   nNetLevel,maxiter, True,
                   vol_r,mu_r,sp_r,
                   num_h1,num_h2,sr,config)

    start_time = time.time()
    # network training and image transfer
    if args.training:
        path = auto.train(load_path, save_path, img_dir,loadLRonly, tol,args.filenum)
    else:
        # load a model and generate transfer images
        auto.transfer(save_path, load_path, img_dir, args.filenum)

    print("--- %s seconds ---" % (time.time() - start_time))
    print('image pair '+str(args.filenum) + ' is done')

if __name__ == "__main__":
    # define main use two __, if use only one_, it will not debug
    main()

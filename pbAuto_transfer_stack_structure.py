import tensorflow as tf
import numpy as np
import math
# import matplotlib.pyplot as plt
import scipy.io as sio
import random
import scipy.misc
import os
from tensorflow.python.training import saver
import tensorflow.contrib.layers as ly
from os.path import join as pjoin
from numpy import *
import numpy.matlib
import scipy.ndimage
import csv
import cv2

# Written by Ying Qu <yqu3@vols.utk.edu>
# This code is a demo code for our paper
# “Non-local Representation based Mutual Affine-Transfer Network for Photorealistic Stylization”, TPAMI 2021
# The code is for research purpose only
# All Rights Reserved


class betapan(object):
    def __init__(self, input, lr_rate, p_rate, nNetLevel, epoch, is_adam,
                 vol_r, mu_r, sp_r, num_h1, num_h2, sr, config):
        # initialize the input and weights matrices
        self.input = input
        self.mark = input.mark
        self.initlrate = lr_rate
        self.initprate = p_rate
        self.epoch = epoch
        self.nNetLevel = nNetLevel
        self.num_h1 = num_h1
        self.num_h2 = num_h2
        self.is_adam = is_adam
        self.vol_r = vol_r
        self.mu_r = mu_r
        self.sp_r = sp_r
        self.input_content = input.content_reduced_scaled
        self.input_style = input.style_reduced_scaled
        self.meanc = input.meanc_scaled
        self.means = input.means_scaled
        self.dimc = input.dimc_scaled
        self.dims = input.dims_scaled
        self.col_content = input.col_content_scaled
        self.col_style = input.col_style_scaled
        self.sr = sr

        with tf.name_scope('inputs'):
            self.content = tf.placeholder(tf.float32, [None, input.dimc[2]], name='content_input')
            self.style = tf.placeholder(tf.float32, [None, input.dims[2]], name='style_input')

        self.sess = tf.Session(config=config)

        with tf.variable_scope('content_decoder') as scope:
            self.wCdecoder = {
                'content_decoder_w1': tf.Variable(tf.truncated_normal([self.num_h1, self.num_h1], stddev=0.1)),
                'content_decoder_w2': tf.Variable(tf.truncated_normal([1, self.dimc[2]], stddev=0.1)),
            }

        with tf.variable_scope('style_decoder') as scope:
            self.wSdecoder = {
                'style_decoder_w1': tf.Variable(tf.truncated_normal([self.num_h1, self.num_h1], stddev=0.1)),
                'style_decoder_w2': tf.Variable(tf.truncated_normal([1, self.dims[2]], stddev=0.1)),
            }

        with tf.variable_scope('basic_decoder') as scope:
            self.wCSdecoder = {
                'basic_decoder_w1': tf.Variable(tf.truncated_normal([self.num_h1, self.dimc[2]], stddev=0.1)),
            }

        with tf.variable_scope('shared_hidden_decoder') as scope:
            self.wCSdecoder_h = {
                'shared_hidden_decoder_w1': tf.Variable(tf.truncated_normal([self.num_h2, self.num_h1], stddev=0.1)),
            }


    def compute_latent_vars_break(self, i, remaining_stick, v_samples):
        # compute stick segment
        stick_segment = v_samples[:, i] * remaining_stick
        remaining_stick *= (1 - v_samples[:, i])
        return (stick_segment, remaining_stick)

    def variable_summaries(self,var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    # difference from tf 1.3 version to 0.9 version. the tf.layers.dense  --> tf.contrib.layers.fully_connected
    # tf.concat([],1) --> tf.concat(1,[])

    def wct_tf(self,content, style, alpha=1):
        content_t = tf.transpose(tf.squeeze(content,axis=0), (2, 0, 1))
        style_t = tf.transpose(tf.squeeze(style,axis=0), (2, 0, 1))
        [Cc, Hc, Wc] = content_t.shape
        [Cs, Hs, Ws] = style_t.shape

        # CxHxW -> CxH*W
        content_flat = tf.reshape(content_t, (Cc, Hc * Wc))
        style_flat = tf.reshape(style_t, (Cs, Hs * Ws))
        # Content covariance
        mc = tf.reduce_mean(content_flat, axis=1, keep_dims=True)
        fc = content_flat - mc
        eps = 1e-8
        fcfc = tf.matmul(fc, fc, transpose_b=True) / (tf.cast(Hc * Wc, tf.float32) - 1.) + tf.eye(int(Cc)) * eps

        # Style covariance
        ms = tf.reduce_mean(style_flat, axis=1, keep_dims=True)
        fs = style_flat - ms
        fsfs = tf.matmul(fs, fs, transpose_b=True) / (tf.cast(Hs * Ws, tf.float32) - 1.) + tf.eye(int(Cs)) * eps

        # tf.svd is slower on GPU, see https://github.com/tensorflow/tensorflow/issues/13603
        with tf.device('/cpu:0'):
            Sc, Uc, _ = tf.svd(fcfc)
            Ss, Us, _ = tf.svd(fsfs)
        # Filter small singular values
        k_c = tf.reduce_sum(tf.cast(tf.greater(Sc, 1e-5), tf.int32))
        k_s = tf.reduce_sum(tf.cast(tf.greater(Ss, 1e-5), tf.int32))

        # Whiten content feature
        Dc = tf.diag(tf.pow(Sc[:k_c], -0.5))
        fc_hat = tf.matmul(tf.matmul(tf.matmul(Uc[:, :k_c], Dc), Uc[:, :k_c], transpose_b=True), fc)

        # Color content with style
        Ds = tf.diag(tf.pow(Ss[:k_s], 0.5))
        fcs_hat = tf.matmul(tf.matmul(tf.matmul(Us[:, :k_s], Ds), Us[:, :k_s], transpose_b=True), fc_hat)

        # Re-center with mean of style
        fcs_hat = fcs_hat + ms

        # Blend whiten-colored feature with original content feature
        blended = alpha * fcs_hat + (1 - alpha) * (fc + mc)
        # CxH*W -> CxHxW
        blended = tf.reshape(blended, (Cc, Hc, Wc))
        # CxHxW -> 1xHxWxC
        blended = tf.expand_dims(tf.transpose(blended, (1, 2, 0)), 0)

        return blended

    def next_feed(self):
        feed_dict = {self.style:self.input_style, self.content:self.input_content}
        return feed_dict

    def construct_stick_break(self,vsample, dim, stick_size):
        size = dim[0]*dim[1]
        size = int(size)
        remaining_stick = tf.ones(size, )
        for i in range(stick_size):
            [stick_segment, remaining_stick] = self.compute_latent_vars_break(i, remaining_stick, vsample)
            if i == 0:
                stick_segment_sum_lr = tf.expand_dims(stick_segment, 1)
            else:
                stick_segment_sum_lr = tf.concat([stick_segment_sum_lr, tf.expand_dims(stick_segment, 1)],1)
        return stick_segment_sum_lr

    def construct_vsamples(self,uniform,wb,hsize):
        concat_wb = wb
        for iter in range(hsize - 1):
            concat_wb = tf.concat([concat_wb, wb], 1)
        v_samples = 1 - (1-uniform) ** (1.0 / concat_wb)
        return v_samples

    def encoder_uniform_h1(self,x,reuse=False):
        with tf.variable_scope('encoder_uniform_h1') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            layer_11 = tf.contrib.layers.fully_connected(x, self.nNetLevel[0], activation_fn=None)
            stack_layer_11 = tf.concat([layer_11, x], 1)
            layer_12 = tf.contrib.layers.fully_connected(stack_layer_11, self.nNetLevel[1], activation_fn=None)
            stack_layer_12 = tf.concat([layer_12, stack_layer_11], 1)
            layer_13 = tf.contrib.layers.fully_connected(stack_layer_12, self.nNetLevel[2], activation_fn=None)
            stack_layer_13 = tf.concat([layer_13, stack_layer_12], 1)
            layer_14 = tf.contrib.layers.fully_connected(stack_layer_13, self.nNetLevel[2], activation_fn=None)
            stack_layer_14 = tf.concat([layer_14, stack_layer_13], 1)
            layer_15 = tf.contrib.layers.fully_connected(stack_layer_14, self.nNetLevel[2], activation_fn=None)
            stack_layer_15 = tf.concat([layer_15, stack_layer_14], 1)
            uniform = tf.contrib.layers.fully_connected(stack_layer_15, self.num_h1, activation_fn=None)
        return stack_layer_12, uniform

    def encoder_beta_h1(self, x, reuse=False):
        with tf.variable_scope('encoder_beta_h1') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            layer_14 = tf.contrib.layers.fully_connected(x, self.nNetLevel[3], activation_fn=None)
            stack_layer_14 = tf.concat([layer_14,x], 1)
            layer_15 = tf.contrib.layers.fully_connected(stack_layer_14, self.num_h1, activation_fn=None)
            stack_layer_15 = tf.concat([layer_15,stack_layer_14], 1)
            wb = tf.contrib.layers.fully_connected(stack_layer_15, 1, activation_fn=None)
        return wb

    def encoder_vsamples_h1(self, x, hsize, reuse=False):
            stack_layer_12, uniform = self.encoder_uniform_h1(x,reuse)
            wb = self.encoder_beta_h1(stack_layer_12,reuse)

            uniform_sig = tf.nn.sigmoid(uniform)
            wb_sp = tf.nn.softplus(wb)
            v_samples = self.construct_vsamples(uniform_sig,wb_sp,hsize)
            return v_samples, uniform, wb

    def encoder_hidden_h2(self, x, uniform_h1, beta_h1, hsize, dim, reuse=False):
        with tf.variable_scope('encoder_hidden_h2') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            stacked_uniform_h1 = x
            layer_11 = tf.contrib.layers.fully_connected(stacked_uniform_h1, self.nNetLevel[4], activation_fn=None)
            stack_layer_11 = tf.concat([layer_11, stacked_uniform_h1], 1)
            layer_12 = tf.contrib.layers.fully_connected(stack_layer_11, self.nNetLevel[5], activation_fn=None)
            stack_layer_12 = tf.concat([layer_12, stack_layer_11], 1)
            layer_13 = tf.contrib.layers.fully_connected(stack_layer_12, self.nNetLevel[6], activation_fn=None)
            stack_layer_13 = tf.concat([layer_13, stack_layer_12], 1)
            layer_14 = tf.contrib.layers.fully_connected(stack_layer_13, self.nNetLevel[6], activation_fn=None)
            stack_layer_14 = tf.concat([layer_14, stack_layer_13], 1)

            uniform = tf.contrib.layers.fully_connected(stack_layer_14, self.num_h2, activation_fn=tf.nn.sigmoid)

            stacked_beta_h1 = stack_layer_12
            wb_11 = tf.contrib.layers.fully_connected(stacked_beta_h1, self.nNetLevel[7], activation_fn=None)
            stacked_beta_h2 = tf.concat([wb_11,stacked_beta_h1],1)
            wb_12 = tf.contrib.layers.fully_connected(wb_11, self.nNetLevel[7], activation_fn=None)
            stacked_beta_h3 = tf.concat([wb_12,stacked_beta_h2],1)
            wb = tf.contrib.layers.fully_connected(stacked_beta_h3, 1, activation_fn= tf.nn.softplus)

        v_samples = self.construct_vsamples(uniform,wb,hsize)
        stick_segment_sum_stacked = self.construct_stick_break(v_samples,dim,hsize)
        return stick_segment_sum_stacked

    def encoder_content_h1(self, x, reuse=False):
        v_samples, uniform, wb = self.encoder_vsamples_h1(x, self.num_h1, reuse)
        stick_content_h1 = self.construct_stick_break(v_samples, self.dimc, self.num_h1)
        return stick_content_h1,uniform, wb

    def encoder_content_h2(self, x, uniform, wb, reuse=False):
        stick_content_h2 = self.encoder_hidden_h2(x, uniform, wb, self.num_h2, self.dimc, reuse)
        return stick_content_h2

    def encoder_style_h1(self, x, reuse=False):
        v_samples, uniform, wb = self.encoder_vsamples_h1(x, self.num_h1, reuse)
        stick_content_h1 = self.construct_stick_break(v_samples, self.dims, self.num_h1)
        return stick_content_h1,uniform, wb

    def encoder_style_h2(self, x, uniform, wb, reuse=False):
        stick_content_h2 = self.encoder_hidden_h2(x, uniform, wb, self.num_h2, self.dims, reuse)
        return stick_content_h2

    def decoder_content(self, x):
        layer_1 = tf.matmul(x, self.wCdecoder['content_decoder_w1'])
        layer_2 = tf.matmul(layer_1, self.wCSdecoder['basic_decoder_w1'])
        layer_4 = tf.add(layer_2, self.wCdecoder['content_decoder_w2'])
        return layer_4

    def decoder_style(self, x):
        layer_1 = tf.matmul(x, self.wSdecoder['style_decoder_w1'])
        layer_2 = tf.matmul(layer_1, self.wCSdecoder['basic_decoder_w1'])
        layer_4 = tf.add(layer_2, self.wSdecoder['style_decoder_w2'])
        return layer_4

    def decoder_hidden_content(self, x):
        layer_2 = tf.matmul(x, self.wCSdecoder_h['shared_hidden_decoder_w1'])
        return layer_2

    def decoder_hidden_style(self, x):
        layer_2 = tf.matmul(x, self.wCSdecoder_h['shared_hidden_decoder_w1'])
        return layer_2

    def t_mi_h1(self, x, reuse=False):
        h_size = x.get_shape().as_list()
        with tf.variable_scope('t_rmi_h1') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            layer1 = tf.layers.dense(x, h_size[3], activation=None, use_bias=True)
            layer = tf.layers.dense(layer1, 1, activation=tf.nn.sigmoid, use_bias=False)
        return layer

    def t_mi_h2(self, x, reuse=False):
        h_size = x.get_shape().as_list()
        with tf.variable_scope('t_rmi_h2') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            layer1 = tf.layers.dense(x, h_size[3], activation=None, use_bias=True)
            layer = tf.layers.dense(layer1, 1, activation=tf.nn.sigmoid, use_bias=False)
        return layer

    def gen_content(self, x, reuse=False):
        encoder_lr_op, uniform, wb = self.encoder_content_h1(x, reuse)
        encoder_hidden = self.encoder_content_h2(encoder_lr_op, uniform, wb, reuse)
        decoder_hidden = self.decoder_hidden_content(encoder_hidden)
        decoder_lr_op = self.decoder_content(decoder_hidden)
        return decoder_lr_op

    def gen_style(self, x, reuse=False):
        encoder_lr_op, uniform, wb = self.encoder_style_h1(x, reuse)
        encoder_hidden = self.encoder_style_h2(encoder_lr_op, uniform, wb, reuse)
        decoder_hidden = self.decoder_hidden_style(encoder_hidden)
        decoder_lr_op = self.decoder_style(decoder_hidden)
        return decoder_lr_op

    def gen_content_h2(self, x, reuse=False):
        encoder_lr_op, uniform, wb = self.encoder_content_h1(x, reuse)
        encoder_hidden = self.encoder_content_h2(encoder_lr_op, uniform, wb, reuse)
        decoder_hidden = self.decoder_hidden_content(encoder_hidden)
        content_h2_loss = decoder_hidden - encoder_lr_op
        return content_h2_loss

    def gen_style_h2(self, x, reuse=False):
        encoder_lr_op, uniform, wb = self.encoder_style_h1(x, reuse)
        encoder_hidden = self.encoder_style_h2(encoder_lr_op, uniform, wb, reuse)
        decoder_hidden = self.decoder_hidden_style(encoder_hidden)
        style_h2_loss = decoder_hidden - encoder_lr_op
        return style_h2_loss


    def gen_hidden_transfer_h1(self, reuse=False):
        content_h1, uniform_c, wb_c = self.encoder_content_h1(self.content,reuse)
        style_h1, uniform_s, wb_s= self.encoder_style_h1(self.style, reuse)

        content_s1 = tf.reshape(content_h1, [1, self.dimc[0], self.dimc[1], self.num_h1])
        style_s1 = tf.reshape(style_h1, [1, self.dims[0], self.dims[1], self.num_h1])

        cont_sty1 = (self.wct_tf(content_s1,style_s1))
        cont_sty1 = tf.reshape(cont_sty1, [self.dimc[0] * self.dimc[1], self.num_h1])

        content_h2 = self.encoder_content_h2(cont_sty1, uniform_c, wb_c, reuse)
        decoder_hidden = self.decoder_hidden_style(content_h2)
        out = self.decoder_style(decoder_hidden)
        return out

    def gen_hidden_transfer_h2(self, reuse=False):
        content_h1, uniform_c, wb_c = self.encoder_content_h1(self.content,reuse)
        content_h2 = self.encoder_content_h2(content_h1, uniform_c, wb_c, reuse)
        style_h1,uniform_s, wb_s = self.encoder_style_h1(self.style, reuse)
        style_h2 = self.encoder_style_h2(style_h1, uniform_s, wb_s, reuse)

        content_s = tf.reshape(content_h2, [1, self.dimc[0], self.dimc[1], self.num_h2])
        style_s = tf.reshape(style_h2, [1, self.dims[0], self.dims[1], self.num_h2])
        cont_sty = (self.wct_tf(content_s,style_s))
        cont_sty = tf.reshape(cont_sty, [self.dimc[0] * self.dimc[1], self.num_h2])
        decoder_hidden = self.decoder_hidden_style(cont_sty)
        out = self.decoder_style(decoder_hidden)

        return out

    def gen_hidden_transfer(self, reuse=False):
        content_h1, uniform_c, wb_c = self.encoder_content_h1(self.content,reuse)
        style_h1, uniform_s, wb_s= self.encoder_style_h1(self.style, reuse)

        content_s1 = tf.reshape(content_h1, [1, self.dimc[0], self.dimc[1], self.num_h1])
        style_s1 = tf.reshape(style_h1, [1, self.dims[0], self.dims[1], self.num_h1])
        cont_sty1 = (self.wct_tf(content_s1,style_s1))
        cont_sty1 = tf.reshape(cont_sty1, [self.dimc[0] * self.dimc[1], self.num_h1])

        content_h2 = self.encoder_content_h2(cont_sty1, uniform_c, wb_c, reuse)
        style_h2 = self.encoder_style_h2(style_h1, uniform_s, wb_s, reuse)

        content_s = tf.reshape(content_h2, [1, self.dimc[0], self.dimc[1], self.num_h2])
        style_s = tf.reshape(style_h2, [1, self.dims[0], self.dims[1], self.num_h2])
        cont_sty = (self.wct_tf(content_s,style_s))
        cont_sty = tf.reshape(cont_sty, [self.dimc[0] * self.dimc[1], self.num_h2])
        decoder_hidden = self.decoder_hidden_style(cont_sty)
        out = self.decoder_style(decoder_hidden)

        return out

    def gen_color_transfer(self, reuse=False):
        content_h1, uniform_c, wb_c = self.encoder_content_h1(self.content,reuse)
        content_h2 = self.encoder_content_h2(content_h1, uniform_c, wb_c, reuse)
        decoder_hidden = self.decoder_hidden_style(content_h2)
        out = self.decoder_style(decoder_hidden)
        return out

    def build_model(self):
        ## Reconstruction error for content image
        y_pred_content = self.gen_content(self.content,False)
        y_true_content = self.content
        error_content = y_pred_content - y_true_content

        error_content_h2 = self.gen_content_h2(self.content,True)
        content_loss_euc_h2 = tf.reduce_mean(tf.reduce_sum(tf.pow(error_content_h2, 2)))
        content_loss_euc = tf.reduce_mean(tf.reduce_sum(tf.pow(error_content, 2))) + content_loss_euc_h2

        decoder_ch2_op11 = tf.matmul(self.wCSdecoder_h['shared_hidden_decoder_w1'],self.wCdecoder['content_decoder_w1'])
        decoder_ch2_op2 = tf.matmul(decoder_ch2_op11,self.wCSdecoder['basic_decoder_w1'])
        decoder_ch2_op3 = tf.add(decoder_ch2_op2,self.wCdecoder['content_decoder_w2'])
        volume_c = tf.reduce_mean(tf.matmul(tf.transpose(decoder_ch2_op3),decoder_ch2_op3))
        content_volume_loss = volume_c

        ## mutual information for hidden layer h1 and h2
        content_h1, uniform_c, wb_c = self.encoder_content_h1(self.content, reuse=True)
        content_h2 = self.encoder_content_h2(content_h1, uniform_c, wb_c, reuse=True)
        content_shuffle = tf.random_shuffle(self.content)

        content_h1_img = tf.reshape(content_h1, [1, self.dimc[0], self.dimc[1], self.num_h1])
        content_h2_img = tf.reshape(content_h2, [1, self.dimc[0], self.dimc[1], self.num_h2])
        content_img = tf.reshape(self.content, [1, self.dimc[0], self.dimc[1], self.dimc[2]])
        content_shuffle_img = tf.reshape(content_shuffle, [1, self.dimc[0], self.dimc[1], self.dimc[2]])

        positive_samples_ch1 = tf.concat([content_img, content_h1_img], -1)
        positive_samples_ch2 = tf.concat([content_img, content_h2_img], -1)
        negative_samples_ch1 = tf.concat([content_shuffle_img, content_h1_img], 3)
        negative_samples_ch2 = tf.concat([content_shuffle_img, content_h2_img], 3)


        positive_scores_ch1 = self.t_mi_h1(positive_samples_ch1)
        negative_scores_ch1 = self.t_mi_h1(negative_samples_ch1, reuse=True)

        positive_scores_ch2 = self.t_mi_h2(positive_samples_ch2)
        negative_scores_ch2 = self.t_mi_h2(negative_samples_ch2, reuse=True)


        eps = 0.00000001
        content_loss_mi_h1 = -(tf.reduce_mean(-tf.nn.softplus(-positive_scores_ch1))
                             -tf.reduce_mean(tf.nn.softplus(negative_scores_ch1)))
        content_loss_mi_h2 = -(tf.reduce_mean(-tf.nn.softplus(-positive_scores_ch2))
                             -tf.reduce_mean(tf.nn.softplus(negative_scores_ch2)))

        content_loss_mi = content_loss_mi_h1 + content_loss_mi_h2

        # spatial sparse constraint for content image h1 and h2
        con_top_h1 = content_h1
        con_top_h2 = content_h2
        con_base_norm_h1 = tf.reduce_sum(con_top_h1, 1, keepdims=True)
        con_base_norm_h2 = tf.reduce_sum(con_top_h2, 1, keepdims=True)
        con_base_norm_h1 = tf.clip_by_value(con_base_norm_h1,eps,tf.reduce_max(con_base_norm_h1))
        con_base_norm_h2 = tf.clip_by_value(con_base_norm_h2,eps,tf.reduce_max(con_base_norm_h2))

        con_sparse_h1 = tf.div(con_top_h1, (con_base_norm_h1))
        con_sparse_h2 = tf.div(con_top_h2, (con_base_norm_h2))
        con_loss_sparse_h1 = tf.reduce_mean(-tf.multiply(con_sparse_h1, tf.log(tf.clip_by_value(con_sparse_h1,eps,tf.reduce_max(con_sparse_h1)))))
        con_loss_sparse_h2 = tf.reduce_mean(-tf.multiply(con_sparse_h2, tf.log(tf.clip_by_value(con_sparse_h2,eps,tf.reduce_max(con_sparse_h2)))))
        # con_loss_sparse = con_loss_sparse_h1 + 0*(self.num_h2/self.num_h1)*con_loss_sparse_h2
        con_loss_sparse = con_loss_sparse_h1 + con_loss_sparse_h2

        # content total loss
        content_loss = content_loss_euc + self.vol_r * content_volume_loss \
                        + self.sp_r * con_loss_sparse + self.mu_r * content_loss_mi

        # updated parameters for the content image
        theta_encoder_uniform_h1 = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder_uniform_h1')
        theta_encoder_beta_h1 = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder_beta_h1')
        theta_encoder_h2 = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder_hidden_h2')
        theta_share_decoder_h2 = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='shared_hidden_decoder')
        theta_content_decoder = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='content_decoder')
        theta_share_decoder_h1 = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='basic_decoder')
        theta_rmi_h1 = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='t_rmi_h1')
        theta_rmi_h2 = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='t_rmi_h2')

        counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        opt_c = ly.optimize_loss(loss=content_loss, learning_rate=self.initlrate,
                                 optimizer=tf.train.AdamOptimizer if self.is_adam is True else tf.train.RMSPropOptimizer,
                                 variables=theta_encoder_uniform_h1+theta_encoder_beta_h1+theta_encoder_h2
                                           +theta_share_decoder_h2+theta_content_decoder+theta_share_decoder_h1+theta_rmi_h1+theta_rmi_h2
                                           ,
                                 global_step=counter_c)

        ######################
        #### Style image  ####
        ######################

        ## Reconstruction error for content image
        x_pred_s = self.gen_style(self.style, True)
        x_true_s = self.style
        error_s = x_pred_s - x_true_s

        error_s_h2 = self.gen_style_h2(self.style,True)
        style_loss_euc_h2 = tf.reduce_mean(tf.reduce_sum(tf.pow(error_s_h2, 2)))
        style_loss_euc = tf.reduce_mean(tf.reduce_sum(tf.pow(error_s, 2))) + style_loss_euc_h2

        decoder_sh2_op11 = tf.matmul(self.wCSdecoder_h['shared_hidden_decoder_w1'],self.wSdecoder['style_decoder_w1'])
        decoder_sh2_op2 = tf.matmul(decoder_sh2_op11,self.wCSdecoder['basic_decoder_w1'])
        decoder_sh2_op3 = tf.add(decoder_sh2_op2,self.wSdecoder['style_decoder_w2'])
        volume_s = tf.reduce_mean(tf.matmul(tf.transpose(decoder_sh2_op3),decoder_sh2_op3))
        style_volume_loss = volume_s


        # mutual information for hidden layer h1 and h2
        style_h1, uniform_s, wb_s = self.encoder_style_h1(self.style, reuse=True)
        style_h2 = self.encoder_style_h2(style_h1, uniform_s, wb_s, reuse=True)
        style_shuffle = tf.random_shuffle(self.style)

        style_h1_img = tf.reshape(style_h1, [1, self.dims[0], self.dims[1], self.num_h1])
        style_h2_img = tf.reshape(style_h2, [1, self.dims[0], self.dims[1], self.num_h2])
        style_img = tf.reshape(self.style, [1, self.dims[0], self.dims[1], self.dims[2]])
        style_shuffle_img = tf.reshape(style_shuffle, [1, self.dims[0], self.dims[1], self.dims[2]])


        positive_samples_sh1 = tf.concat([style_img, style_h1_img], -1)
        positive_samples_sh2 = tf.concat([style_img, style_h2_img], -1)
        negative_samples_sh1 = tf.concat([style_shuffle_img, style_h1_img], 3)
        negative_samples_sh2 = tf.concat([style_shuffle_img, style_h2_img], 3)


        positive_scores_sh1 = self.t_mi_h1(positive_samples_sh1, reuse=True)
        negative_scores_sh1 = self.t_mi_h1(negative_samples_sh1, reuse=True)
        positive_scores_sh2 = self.t_mi_h2(positive_samples_sh2, reuse=True)
        negative_scores_sh2 = self.t_mi_h2(negative_samples_sh2, reuse=True)


        style_loss_mi1 = -(tf.reduce_mean(-tf.nn.softplus(-positive_scores_sh1))
                             -tf.reduce_mean(tf.nn.softplus(negative_scores_sh1)))
        style_loss_mi2 = -(tf.reduce_mean(-tf.nn.softplus(-positive_scores_sh2))
                             -tf.reduce_mean(tf.nn.softplus(negative_scores_sh2)))
        style_loss_mi = style_loss_mi1 + style_loss_mi2

        # spatial sparse constrint for style h1 h2
        sty_top_h1 = style_h1
        sty_top_h2 = style_h2
        sty_base_norm_h1 = tf.reduce_sum(sty_top_h1, 1, keepdims=True)
        sty_base_norm_h2 = tf.reduce_sum(sty_top_h2, 1, keepdims=True)

        sty_base_norm_h1 = tf.clip_by_value(sty_base_norm_h1,eps,tf.reduce_max(sty_base_norm_h1))
        sty_base_norm_h2 = tf.clip_by_value(sty_base_norm_h2,eps,tf.reduce_max(sty_base_norm_h2))

        sty_sparse_h1 = tf.div(sty_top_h1, sty_base_norm_h1)
        sty_sparse_h2 = tf.div(sty_top_h2, sty_base_norm_h2)
        sty_loss_sparse_h1 = tf.reduce_mean(-tf.multiply(sty_sparse_h1, tf.log(tf.clip_by_value(sty_sparse_h1,eps,tf.reduce_max(sty_sparse_h1)))))
        sty_loss_sparse_h2 = tf.reduce_mean(-tf.multiply(sty_sparse_h2, tf.log(tf.clip_by_value(sty_sparse_h2,eps,tf.reduce_max(sty_sparse_h2)))))
        # sty_loss_sparse = sty_loss_sparse_h1 + 0*(self.num_h2/self.num_h1)*sty_loss_sparse_h2
        sty_loss_sparse = sty_loss_sparse_h1 + sty_loss_sparse_h2

        style_loss = style_loss_euc + self.vol_r * style_volume_loss \
                     + self.sp_r * sty_loss_sparse + self.mu_r * style_loss_mi

        theta_style_decoder = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='style_decoder')

        counter_s = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        opt_s = ly.optimize_loss(loss=style_loss, learning_rate=self.initlrate,
                               optimizer=tf.train.AdamOptimizer if self.is_adam is True else tf.train.RMSPropOptimizer,
                               variables= theta_encoder_uniform_h1+theta_encoder_beta_h1+theta_encoder_h2
                                            +theta_share_decoder_h2+theta_style_decoder+theta_share_decoder_h1 +theta_rmi_h1+theta_rmi_h2,
                               global_step=counter_s)
        total_loss  = content_loss + style_loss
        opt_total  = opt_c + opt_s

        return content_loss, opt_c, style_loss, opt_s, content_volume_loss, content_loss_mi, style_loss_mi, total_loss, opt_total

    def init_test_image(self):
        self.input_content = self.input.content_reduced
        self.input_style = self.input.style_reduced
        self.meanc = self.input.meanc
        self.means = self.input.means
        self.dimc = self.input.dimc
        self.dims = self.input.dims
        self.col_content = self.input.col_content
        self.col_style = self.input.col_style

    def train(self, load_Path, save_dir, img_dir, loadLRonly, tol, index):
        content_loss, opt_c, style_loss, opt_s, content_volume_loss, content_loss_entropy, style_loss_entropy, total_loss, opt_total = self.build_model()

        self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if os.path.exists(load_Path):
            if loadLRonly:
                # load part of the variables
                vars = tf.contrib.slim.get_variables_to_restore()
                variables_to_restore = [v for v in vars if v.name.startswith('encoder_uniform_h1/')] \
                                        + [v for v in vars if v.name.startswith('encoder_beta_h1/')] \
                                        + [v for v in vars if v.name.startswith('encoder_hidden_h2/')] \
                                        + [v for v in vars if v.name.startswith('content_decoder_h/')] \
                                        + [v for v in vars if v.name.startswith('shared_hidden_decoder/')] \
                                        + [v for v in vars if v.name.startswith('content_decoder/')] \
                                        + [v for v in vars if v.name.startswith('basic_decoder/')] \
                                        + [v for v in vars if v.name.startswith('style_decoder_h/')] \
                                        + [v for v in vars if v.name.startswith('style_decoder/')] \
                                        + [v for v in vars if v.name.startswith('t_rmi_h1/')] \
                                        + [v for v in vars if v.name.startswith('t_rmi_h2/')]

                saver = tf.train.Saver(variables_to_restore)
                load_file = tf.train.latest_checkpoint(load_Path)
                if load_file==None:
                    print('No checkpoint was saved.')
                else:
                    saver.restore(self.sess,load_file)
            else:
                # load all the variables
                saver = tf.train.Saver(max_to_keep=1)
                load_file = tf.train.latest_checkpoint(load_Path)
                if load_file==None:
                    print('No checkpoint was saved.')
                else:
                    saver.restore(self.sess, load_file)
        else:
            saver = tf.train.Saver(max_to_keep=1)

        results_file_name = pjoin(save_dir,"sb_" + "lrate_" + str(self.initlrate)+ ".txt")
        results_file = open(results_file_name, 'a')
        feed_dict = self.next_feed()

        sam_style = 10
        sam_content = 10
        rmse_total = zeros(self.epoch+1)
        rmse_total[0] = 1
        for epoch in range(self.epoch):
            _, tloss = self.sess.run([opt_total,total_loss], feed_dict=feed_dict)
            self.initlrate = self.initlrate * 0.9995
            self.vol_r = self.vol_r * 0.9995
            sloss = self.sess.run(style_loss, feed_dict=feed_dict)
            closs = self.sess.run(content_loss, feed_dict=feed_dict)

            if (epoch + 1) % 60 == 0:
                # Report and save progress.
                results = "epoch {}: total loss {:.12f} learing_rate {:.9f}"
                results = results.format(epoch, tloss, self.initlrate)
                print (results)
                print ("\n")
                results_file.write(results + "\n\n")
                results_file.flush()

                results = "epoch {}: content loss {:.12f} learing_rate {:.9f}"
                results = results.format(epoch, closs, self.initlrate)
                print (results)
                print ("\n")
                results_file.write(results + "\n\n")
                results_file.flush()

                results = "epoch {}: style loss {:.12f} learing_rate {:.9f}"
                results = results.format(epoch, sloss, self.initlrate)
                print (results)
                print ("\n")
                results_file.write(results + "\n\n")
                results_file.flush()

                lr_v_loss = self.sess.run(content_volume_loss, feed_dict=feed_dict)
                volume = "volume of the decoder: {:.12f}"
                volume = volume.format(lr_v_loss)
                print (volume)
                results_file.write(volume + "\n")
                results_file.flush()

                lr_en_loss = self.sess.run(content_loss_entropy, feed_dict=feed_dict)
                results = "epoch {}: lr en loss {:.12f} learing_rate {:.9f}"
                results = results.format(epoch, lr_en_loss, self.initlrate)
                print (results)
                print ("\n")
                results_file.write(results + "\n\n")
                results_file.flush()


                p_en_loss = self.sess.run(style_loss_entropy, feed_dict=feed_dict)
                results = "epoch {}: pan en loss {:.12f} learing_rate {:.9f}"
                results = results.format(epoch, p_en_loss, self.initprate)
                print (results)
                print ('\n')
                results_file.write(results + "\n\n")
                results_file.flush()

                img_content = self.sess.run(self.gen_content(self.content, reuse=True), feed_dict=feed_dict) + self.meanc
                sam_content = self.evaluation(img_content,self.col_content,'Content',epoch,results_file)

                img_style = self.sess.run(self.gen_style(self.style, reuse=True), feed_dict=feed_dict) + self.means
                sam_style = self.evaluation(img_style,self.col_style,'Style',epoch,results_file)

            if (epoch+1)%500==0:
                # saver = tf.train.Saver()
                results_ckpt_name = pjoin(save_dir, "epoch_" + str(epoch) + "_sam_" + str(round(sam_style,3)) + ".ckpt")
                save_path = saver.save(self.sess,results_ckpt_name)

                results = "weights saved at epoch {}"
                results = results.format(epoch)
                print (results)
                print ('\n')

            if ((sam_style>tol) or (sam_content>tol)):
                results = "epoch {}: total loss {:.12f} learing_rate {:.9f}"
                results = results.format(epoch, tloss, self.initlrate)
                print (results)
                print ("\n")
                results_file.write(results + "\n\n")
                results_file.flush()

            # elif ((sam_style<tol) and (sam_content<tol) or (epoch==self.epoch-1)):
            elif ((sam_style < tol) or (epoch == self.epoch - 1)):
                # saver = tf.train.Saver()
                results_ckpt_name = pjoin(save_dir, "epoch_" + str(epoch) + "_sam_" + str(round(sam_style,3)) + ".ckpt")
                save_path = saver.save(self.sess, results_ckpt_name)
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)

                self.init_test_image()
                feed_dict = self.next_feed()
                # name = save_dir[:save_dir.find('_')]
                name_init = save_dir[:save_dir.find('_')]
                name = name_init + self.mark + str(index)

                print('training is done')
                break;
        return save_path

    def evaluation(self,img_hr,img_tar,name,epoch,results_file):
        # evalute the results
        ref = img_tar*255.0
        tar = img_hr*255.0
        lr_flags = tar<0
        tar[lr_flags]=0
        hr_flags = tar>255.0
        tar[hr_flags] = 255.0

        diff = ref - tar;
        size = ref.shape
        rmse = np.sqrt( np.sum(np.sum(np.power(diff,2))) / (size[0]*size[1]));

        results = name + " epoch {}: RMSE  {:.12f} "
        results = results.format(epoch,  rmse)
        print (results)
        results_file.write(results + "\n")
        results_file.flush()

        # spectral loss
        nom_top = np.sum(np.multiply(ref, tar),0)
        nom_pred = np.sqrt(np.sum(np.power(ref, 2),0))
        nom_true = np.sqrt(np.sum(np.power(tar, 2),0))
        nom_base = np.multiply(nom_pred, nom_true)
        angle = np.arccos(np.divide(nom_top, (nom_base)))
        angle = np.nan_to_num(angle)
        sam = np.mean(angle)*180.0/3.14159
        results = name + " epoch {}: SAM  {:.12f} "
        results = results.format(epoch,  sam)
        print (results)
        print ("\n")
        results_file.write(results + "\n")
        results_file.flush()
        return sam

    def postprocess(self,img):
        img = img*255.0;
        img = np.clip(img, 0, 255).astype('uint8')
        # rgb to bgr
        img = img[..., ::-1]
        return img

    def transfer(self, save_dir, filename,img_dir,index):
        self.init_test_image()
        feed_dict = self.next_feed()
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        gen_content = self.gen_content(self.content,reuse=False)
        gen_style = self.gen_style(self.style,reuse=True)

        gen_content_h1,uniform_c, wb_c  = self.encoder_content_h1(self.content,reuse=True)
        gen_content_h2 = self.encoder_content_h2(gen_content_h1,uniform_c, wb_c , reuse=True)

        gen_style_h1,uniform_s, wb_s  = self.encoder_style_h1(self.style, reuse=True)
        gen_style_h2 = self.encoder_style_h2(gen_style_h1, uniform_s, wb_s, reuse=True)
        saver = tf.train.Saver()
        save_path = tf.train.latest_checkpoint(filename)
        print(save_path)
        if save_path == None:
            print('No checkpoint was saved.')
        else:
            saver.restore(self.sess, save_path)
            print(save_path + '  is loaded.')
        name_init = save_dir[:save_dir.find('_')]
        name= name_init + self.mark +str(index)

        # save color transfer only
        color_transfered = self.gen_color_transfer(reuse=True)
        img_color = self.sess.run(color_transfered, feed_dict=feed_dict) + self.means
        image_array_color = img_color.reshape((self.dimc[0], self.dimc[1], self.dimc[2]))
        image_array_color = self.postprocess(image_array_color)
        cv2.imwrite(img_dir + name + '_color_' + str(self.num_h1) + '_' + str(self.num_h2) + '_m' + str(
            self.mu_r) + 's' + str(self.sp_r) + 'sr' + str(self.sr) + '.png', image_array_color)

        # save wct on h
        hidden_transfered = self.gen_hidden_transfer(True)
        img_wct_h = self.sess.run(hidden_transfered,feed_dict=feed_dict) + self.means
        image_array_wct_h = img_wct_h.reshape((self.dimc[0],self.dimc[1],self.dimc[2]))
        image_array_wct_h = self.postprocess(image_array_wct_h)
        cv2.imwrite(img_dir + name + '_wct_h_' + str(self.num_h1) + '_' + str(self.num_h2) + '_m' + str(self.mu_r) + 's' + str(self.sp_r) + 'sr'+ str(self.sr) + '.png', image_array_wct_h)

        hidden_transfered_h1 = self.gen_hidden_transfer_h1(True)
        img_wct_h_all = self.sess.run(hidden_transfered_h1,feed_dict=feed_dict) + self.means
        image_array_wct_h1 = img_wct_h_all.reshape((self.dimc[0],self.dimc[1],self.dimc[2]))
        image_array_wct_h1 = self.postprocess(image_array_wct_h1)
        cv2.imwrite(img_dir + name + '_wct_h1_' + str(self.num_h1) + '_' + str(self.num_h2) + '_m' + str(self.mu_r) + 's' + str(self.sp_r) + 'sr'+ str(self.sr) + '.png', image_array_wct_h1)

        hidden_transfered_h2 = self.gen_hidden_transfer_h2(True)
        img_wct_h_first = self.sess.run(hidden_transfered_h2,feed_dict=feed_dict) + self.means
        image_array_wct_h2 = img_wct_h_first.reshape((self.dimc[0],self.dimc[1],self.dimc[2]))
        image_array_wct_h2 = self.postprocess(image_array_wct_h2)
        cv2.imwrite(img_dir + name + '_wct_h2_' + str(self.num_h1) + '_' + str(self.num_h2) + '_m' + str(self.mu_r) + 's' + str(self.sp_r) + 'sr'+ str(self.sr) + '.png', image_array_wct_h2)


        img_content =  self.sess.run(gen_content,feed_dict=feed_dict) + self.meanc
        image_array_content = img_content.reshape((self.dimc[0],self.dimc[1],self.dimc[2]))
        image_array_content = self.postprocess(image_array_content)
        cv2.imwrite(img_dir + name + '_content_' + str(self.num_h1) + '_' + str(self.num_h2) + '_m' + str(self.mu_r) + 's' + str(self.sp_r) + 'sr'+ str(self.sr) + '.png', image_array_content)


        img_style =  self.sess.run(gen_style,feed_dict=feed_dict) + self.means
        image_array_style = img_style.reshape((self.dims[0],self.dims[1],self.dims[2]))
        image_array_style = self.postprocess(image_array_style)
        cv2.imwrite(img_dir + name + '_style_' + str(self.num_h1) + '_' + str(self.num_h2) + '_m' + str(self.mu_r) + 's' + str(self.sp_r) + 'sr'+ str(self.sr) + '.png', image_array_style)


        # # # # save hidden layers
        hidden_content1 = self.sess.run(gen_content_h1, feed_dict=feed_dict)
        hidden_content1_cube = np.reshape(hidden_content1,[self.dimc[0],self.dimc[1],self.num_h1])
        hidden_style1 = self.sess.run(gen_style_h1, feed_dict=feed_dict)
        hidden_style1_cube = np.reshape(hidden_style1,[self.dims[0],self.dims[1],self.num_h1])
        hidden_content2 = self.sess.run(gen_content_h2, feed_dict=feed_dict)
        hidden_content2_cube = np.reshape(hidden_content2,[self.dimc[0],self.dimc[1],self.num_h2])
        hidden_style2 = self.sess.run(gen_style_h2, feed_dict=feed_dict)
        hidden_style2_cube = np.reshape(hidden_style2,[self.dims[0],self.dims[1],self.num_h2])

        result = {'hidden_content1': hidden_content1_cube,
                  'hidden_style1': hidden_style1_cube,
                  'hidden_content2': hidden_content2_cube,
                  'hidden_style2': hidden_style2_cube}
        sio.savemat(save_dir + "/rep_out.mat", result)


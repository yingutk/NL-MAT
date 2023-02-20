import numpy as np
import collections
import cv2

# Written by Ying Qu <yqu3@vols.utk.edu>
# This code is a demo code for our paper
# “Non-local Representation based Mutual Affine-Transfer Network for Photorealistic Stylization”, TPAMI 2021
# The code is for research purpose only
# All Rights Reserved

def preprocess(img):
    # bgr to rgb
    img = img[..., ::-1]
    if np.max(img)>1:
        img = img/255.0
    return img

def readData(file_content,file_style,scale=4):
    data = collections.namedtuple('data', ['content', 'content_scaled','style','style_scaled'
                                           'dimc', 'dimc_sacaled','dims_scaled','dims',
                                           'content_reduced', 'content_reduced_scaled'
                                           'style_reduced','style_reduced_scaled','num'])

    scale_factor = 1.0/scale

    content_bgr = cv2.imread(file_content, cv2.IMREAD_COLOR)
    data.content = preprocess(content_bgr)
    data.content = data.content.astype(np.float32)
    data.dimc = data.content.shape

    target_h_c = int(data.dimc[0] * scale_factor)
    target_w_c = int(data.dimc[1] * scale_factor)
    content_bgr_scaled = cv2.resize(content_bgr.copy(), (target_w_c, target_h_c))
    data.content_scaled = preprocess(content_bgr_scaled)
    data.content_scaled = data.content_scaled.astype(np.float32)
    data.dimc_scaled = data.content_scaled.shape

    style_bgr = cv2.imread(file_style, cv2.IMREAD_COLOR)
    data.style = preprocess(style_bgr)
    data.style = data.style.astype(np.float32)
    data.dims = data.style.shape

    target_h_s = int(data.dims[0] * scale_factor)
    target_w_s = int(data.dims[1] * scale_factor)


    style_bgr_scaled = cv2.resize(style_bgr.copy(), (target_w_s, target_h_s))
    data.style_scaled = preprocess(style_bgr_scaled)
    data.style_scaled = data.style_scaled.astype(np.float32)
    data.dims_scaled = data.style_scaled.shape

    data.col_content = np.reshape(data.content,[data.dimc[0]*data.dimc[1],data.dimc[2]])
    data.meanc = np.mean(data.col_content,axis=0,keepdims=True)
    data.content_reduced = np.subtract(data.col_content,data.meanc)
    data.content_reduced_img = np.reshape(data.content_reduced,[data.dimc[0],data.dimc[1],data.dimc[2]])

    data.col_style = np.reshape(data.style,[data.dims[0]*data.dims[1],data.dims[2]])
    data.means = np.mean(data.col_style,axis=0,keepdims=True)
    data.style_reduced = np.subtract(data.col_style,data.means)
    data.style_reduced_img = np.reshape(data.style_reduced, [data.dims[0],data.dims[1],data.dims[2]])

    data.col_content_scaled = np.reshape(data.content_scaled,[data.dimc_scaled[0]*data.dimc_scaled[1],data.dimc_scaled[2]])
    data.meanc_scaled = np.mean(data.col_content_scaled,axis=0,keepdims=True)
    data.content_reduced_scaled = np.subtract(data.col_content_scaled,data.meanc_scaled)
    data.content_reduced_img_scaled = np.reshape(data.content_reduced_scaled,[data.dimc_scaled[0],data.dimc_scaled[1],data.dimc_scaled[2]])

    data.uu,data.ss,data.vv = np.linalg.svd(data.content_reduced_scaled.T,full_matrices=False)
    data.projU = data.uu


    data.col_style_scaled = np.reshape(data.style_scaled,[data.dims_scaled[0]*data.dims_scaled[1],data.dims_scaled[2]])
    data.means_scaled = np.mean(data.col_style_scaled,axis=0,keepdims=True)
    data.style_reduced_scaled = np.subtract(data.col_style_scaled,data.means_scaled)
    data.style_reduced_img_scaled = np.reshape(data.style_reduced_scaled, [data.dims_scaled[0],data.dims_scaled[1],data.dims_scaled[2]])


    return data

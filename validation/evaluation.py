#!/usr/bin/env python
import sys
import os
import os.path
import random
import numpy as np

from PIL import Image
import scipy.misc

# import numpy as np
# from PIL import Image
from scipy.signal import convolve2d

from skimage.metrics import structural_similarity as ssim
# import cv2
#from myssim import compare_ssim as ssim

#SCALE = 8 
SCALE = 4

def _convert_input_type_range(img):
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError(f'The img type should be np.float32 or np.uint8, but got {img_type}')
    return img

def _convert_output_type_range(img, dst_type):
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(f'The dst_type should be np.float32 or np.uint8, but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)

def bgr2ycbcr(img, y_only=False):
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img

def to_y_channel(img):
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.

def calculate_psnr(img, img2, test_y_channel=True,):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    mse = np.mean((img - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):

    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return np.mean(np.mean(ssim_map))

def calculate_ssim(img, img2, test_y_channel=True):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(compute_ssim(img[..., i], img2[..., i]))
    return np.array(ssims).mean()


def _open_img(img_p):
    #F = scipy.misc.fromimage(Image.open(img_p)).astype(float)/255.0
    # F = np.asarray(Image.open(img_p)).astype(float)/255.0
    F = np.asarray(Image.open(img_p)).astype(np.float64)[..., ::-1]
    h, w, c = F.shape
    F = F[:h-h%SCALE, :w-w%SCALE, :]
    boundarypixels = SCALE
    F = F[boundarypixels:-boundarypixels,boundarypixels:-boundarypixels,:]
    return F

def compute_psnr(ref_im, res_im):
    return calculate_psnr(
        _open_img(os.path.join(input_dir,'ref',ref_im)),
        _open_img(os.path.join(input_dir,'res',res_im))
        )

def compute_mssim(ref_im, res_im):
    return calculate_ssim(
        _open_img(os.path.join(input_dir,'ref',ref_im)),
        _open_img(os.path.join(input_dir,'res',res_im))
        )

def _open_img_ssim(img_p):
    #F = scipy.misc.fromimage(Image.open(img_p))#.astype(float)
    F = np.asarray(Image.open(img_p))[..., ::-1]#.astype(float)
    h, w, c = F.shape
    F = F[:h-h%SCALE, :w-w%SCALE, :]
    boundarypixels = SCALE 
    F = F[boundarypixels:-boundarypixels,boundarypixels:-boundarypixels,:]
    return F

def compute_mssim_skimage(ref_im, res_im):
    ref_img = _open_img(os.path.join(input_dir,'ref',ref_im))
    res_img = _open_img(os.path.join(input_dir,'res',res_im))
    ref_img = to_y_channel(ref_img)
    res_img = to_y_channel(res_img)
    ssims = []
    for i in range(ref_img.shape[2]):
        ssims.append(ssim(ref_img[..., i], res_img[..., i], 
                          data_range=res_img[..., i].max() - res_img[..., i].min()))
    return np.array(ssims).mean()
    # return ssim(ref_img,res_img, data_range=res_img.max() - res_img.min())
    # channels = []
    # for i in range(3):
    #     channels.append(ssim(ref_img[:,:,i],res_img[:,:,i],
    #     data_range=res_img[:,:,i].max() - res_img[:,:,i].min()))
    # return np.mean(channels)

# as per the metadata file, input and output directories are the arguments
[_, input_dir, output_dir] = sys.argv

res_dir = os.path.join(input_dir, 'res/')
ref_dir = os.path.join(input_dir, 'ref/')
#print("REF DIR")
#print(ref_dir)


runtime = -1
cpu = -1
data = -1
other = ""
readme_fnames = [p for p in os.listdir(res_dir) if p.lower().startswith('readme')]
try:
    readme_fname = readme_fnames[0]
    print("Parsing extra information from %s"%readme_fname)
    with open(os.path.join(input_dir, 'res', readme_fname)) as readme_file:
        readme = readme_file.readlines()
        lines = [l.strip() for l in readme if l.find(":")>=0]
        runtime = float(":".join(lines[0].split(":")[1:]))
        cpu = int(":".join(lines[1].split(":")[1:]))
        data = int(":".join(lines[2].split(":")[1:]))
        other = ":".join(lines[3].split(":")[1:])
except:
    print("Error occured while parsing readme.txt")
    print("Please make sure you have a line for runtime, cpu/gpu, extra data and other (4 lines in total).")
print("Parsed information:")
print("Runtime: %f"%runtime)
print("CPU/GPU: %d"%cpu)
print("Data: %d"%data)
print("Other: %s"%other)





ref_pngs = sorted([p for p in os.listdir(ref_dir) if p.lower().endswith('png')])
res_pngs = sorted([p for p in os.listdir(res_dir) if p.lower().endswith('png')])
if not (len(ref_pngs)==len(res_pngs)):
    raise Exception('Expected %d .png images'%len(ref_pngs))




scores = []
for (ref_im, res_im) in zip(ref_pngs, res_pngs):
    print(ref_im,res_im)
    scores.append(
        compute_psnr(ref_im,res_im)
    )
    #print(scores[-1])
psnr = np.mean(scores)


scores_ssim = []
for (ref_im, res_im) in zip(ref_pngs, res_pngs):
    print(ref_im,res_im)
    scores_ssim.append(
        #compute_mssim(ref_im,res_im) # this function can produce same results as main papers, but runs very slowly.
        compute_mssim_skimage(ref_im,res_im)
        )
    #print(scores_ssim[-1])
mssim = np.mean(scores_ssim)



# the scores for the leaderboard must be in a file named "scores.txt"
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
    output_file.write("PSNR:%f\n"%psnr)
    output_file.write("SSIM:%f\n"%mssim)
    output_file.write("ExtraRuntime:%f\n"%runtime)
    output_file.write("ExtraPlatform:%d\n"%cpu)
    output_file.write("ExtraData:%d\n"%data)


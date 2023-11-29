# ------------------------------------------------------------------------
# Modified from Grounded-SAM (https://github.com/IDEA-Research/Grounded-Segment-Anything)
# ------------------------------------------------------------------------
import os
import random
import cv2
from scipy import ndimage

import argparse

import numpy as np
import torch
from torch.nn import functional as F
import torchvision
import networks
import utils

import sys

# SAM
sys.path.insert(0, './segment-anything')
from segment_anything.utils.transforms import ResizeLongestSide

# SD
#from diffusers import StableDiffusionPipeline

transform = ResizeLongestSide(1024)
# Green Screen
PALETTE_back = (51, 255, 146)

mam_checkpoint="checkpoints/mam_vitb.pth"
device="cuda"
background_list = os.listdir('assets/backgrounds')

# initialize MAM
mam_model = networks.get_generator_m2m(seg='sam_vit_b', m2m='sam_decoder_deep')
mam_model.to(device)
checkpoint = torch.load(mam_checkpoint, map_location=device)
mam_model.m2m.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)
mam_model = mam_model.eval()

def run_grounded_sam(input_image, in_bbox, output_dir, task_type="scribble_box", background_type="real_world_sample", scribble_mode="", guidance_mode="merge"):

    global sam_predictor

    # make dir
    os.makedirs(output_dir, exist_ok=True)

     # load image
    loaded_img = cv2.imread(input_image)
    image_ori = loaded_img
    #scribble = input_image["mask"]
    original_size = image_ori.shape[:2]

    image = transform.apply_image(image_ori)
    image = torch.as_tensor(image).to(device)
    image = image.permute(2, 0, 1).contiguous()

    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3,1,1).to(device)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3,1,1).to(device)

    image = (image - pixel_mean) / pixel_std

    h, w = image.shape[-2:]
    pad_size = image.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    image = F.pad(image, (0, padw, 0, padh))

    if task_type == 'scribble_point':
        scribble = scribble.transpose(2, 1, 0)[0]

        labeled_array, num_features = ndimage.label(scribble >= 255)

        centers = ndimage.center_of_mass(scribble, labeled_array, range(1, num_features+1))
        centers = np.array(centers)
        ### (x,y)
        centers = transform.apply_coords(centers, original_size)
        point_coords = torch.from_numpy(centers).to(device)
        point_coords = point_coords.unsqueeze(0).to(device)
        point_labels = torch.from_numpy(np.array([1] * len(centers))).unsqueeze(0).to(device)
        if scribble_mode == 'split':
            point_coords = point_coords.permute(1, 0, 2)
            point_labels = point_labels.permute(1, 0)
            
        sample = {'image': image.unsqueeze(0), 'point': point_coords, 'label': point_labels, 'ori_shape': original_size, 'pad_shape': pad_size}
    elif task_type == 'scribble_box':
        ### (x1, y1, x2, y2)
        x_min = in_bbox[0]
        x_max = in_bbox[2]
        y_min = in_bbox[1]
        y_max = in_bbox[3]
        bbox = np.array([x_min, y_min, x_max, y_max])
        bbox = transform.apply_boxes(bbox, original_size)
        bbox = torch.as_tensor(bbox, dtype=torch.float).to(device)

        sample = {'image': image.unsqueeze(0), 'bbox': bbox.unsqueeze(0), 'ori_shape': original_size, 'pad_shape': pad_size}
    elif task_type == 'text':
        sample = {'image': image.unsqueeze(0), 'bbox': bbox.unsqueeze(0), 'ori_shape': original_size, 'pad_shape': pad_size}
    else:
        print("task_type:{} error!".format(task_type))

    with torch.no_grad():
        feas, pred, post_mask = mam_model.forward_inference(sample)

        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
        alpha_pred_os8 = alpha_pred_os8[..., : sample['pad_shape'][0], : sample['pad_shape'][1]]
        alpha_pred_os4 = alpha_pred_os4[..., : sample['pad_shape'][0], : sample['pad_shape'][1]]
        alpha_pred_os1 = alpha_pred_os1[..., : sample['pad_shape'][0], : sample['pad_shape'][1]]

        alpha_pred_os8 = F.interpolate(alpha_pred_os8, sample['ori_shape'], mode="bilinear", align_corners=False)
        alpha_pred_os4 = F.interpolate(alpha_pred_os4, sample['ori_shape'], mode="bilinear", align_corners=False)
        alpha_pred_os1 = F.interpolate(alpha_pred_os1, sample['ori_shape'], mode="bilinear", align_corners=False)
        
        if guidance_mode == 'mask':
            weight_os8 = utils.get_unknown_tensor_from_mask_oneside(post_mask, rand_width=10, train_mode=False)
            post_mask[weight_os8>0] = alpha_pred_os8[weight_os8>0]
            alpha_pred = post_mask.clone().detach()
        else:
            weight_os8 = utils.get_unknown_box_from_mask(post_mask)
            alpha_pred_os8[weight_os8>0] = post_mask[weight_os8>0]
            alpha_pred = alpha_pred_os8.clone().detach()


        weight_os4 = utils.get_unknown_tensor_from_pred_oneside(alpha_pred, rand_width=20, train_mode=False)
        alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4>0]
        
        weight_os1 = utils.get_unknown_tensor_from_pred_oneside(alpha_pred, rand_width=10, train_mode=False)
        alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1>0]
       
        alpha_pred = alpha_pred[0][0].cpu().numpy()

    #### draw
    ### alpha matte
    alpha_rgb = cv2.cvtColor(np.uint8(alpha_pred*255), cv2.COLOR_GRAY2RGB)
    ### com img with background
    if background_type == 'real_world_sample':
        background_img_file = os.path.join('assets/backgrounds', random.choice(background_list))
        background_img = cv2.imread(background_img_file)
        background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
        background_img = cv2.resize(background_img, (image_ori.shape[1], image_ori.shape[0]))
        com_img = alpha_pred[..., None] * image_ori + (1 - alpha_pred[..., None]) * np.uint8(background_img)
        com_img = np.uint8(com_img)
    # else:
    #     if background_prompt is None:
    #         print('Please input non-empty background prompt')
#        else:
#            background_img = generator(background_prompt).images[0]
#            background_img = np.array(background_img)
#            background_img = cv2.resize(background_img, (image_ori.shape[1], image_ori.shape[0]))
#            com_img = alpha_pred[..., None] * image_ori + (1 - alpha_pred[..., None]) * np.uint8(background_img)
#            com_img = np.uint8(com_img)
    ### com img with green screen
    green_img = alpha_pred[..., None] * image_ori + (1 - alpha_pred[..., None]) * np.array([PALETTE_back], dtype='uint8')
    green_img = np.uint8(green_img)

    ### write images to disk
    filename = str.split(os.path.basename(input_image),".")[-2]

    cv2.imwrite(os.path.join(output_dir,filename+"_con.png"),com_img)
    cv2.imwrite(os.path.join(output_dir,filename+"_green.png"),green_img)
    cv2.imwrite(os.path.join(output_dir,filename+"_alpha.png"),alpha_rgb)
    
    return [(com_img, 'composite with background'), (green_img, 'green screen'), (alpha_rgb, 'alpha matte')]

if __name__ == "__main__":
    parser = argparse.ArgumentParser("MAM Input", add_help=True)
    parser.add_argument("--image", type=str, help="Image path")
    parser.add_argument("--in_bbox", type=int, nargs="+", help="Bbox coordinates: x1, y1, x2, y2")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Final output folder")
    #parser.add_argument("--output", type=str, help="Image output path")
    args = parser.parse_args()

    run_grounded_sam(args.image, args.in_bbox, args.output_dir)
"""
Script based on "infer_wild.py" from MotionBERT folder for processing input json data from AlphaPose
"""

import os
import numpy as np
import argparse
import time
from tqdm import tqdm
import imageio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_wild import WildDetDataset
from lib.utils.vismo import render_and_save

from image_share_service.service_server import ServiceServer


def parse_args():
    """
    Function for parsing input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/pose3d/MB_ft_h36m_global_lite.yaml', help="Path to the config file.")
    parser.add_argument('-e', '--evaluate', default='checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-o', '--out_path', type=str, default="/home/guest/image_recog_results/" ,help='output path')
    parser.add_argument('--focus', type=int, default=None, help='target person id')
    parser.add_argument('--clip_len', type=int, default=1, help='clip length for network input')
    parser.add_argument('--debug', type=bool, default=False, help='enables saving of partial results from MotionBERT')
    opts = parser.parse_args()
    return opts


def callback_image_server(received_data:dict):
    """
    Callback functions receiveing dictionary from client with json data
    Input:
        - received_data: dictionary with data to process from client
    Output:
        - received_data: dictionary with results data
    """
    print("MotionBERT image recognition started...")
    print("Main sends: Hello " + received_data["Hello"])
    received_data["Hello"] = "ROS"
    loaded_json = received_data["json_results"]

    fps_in = 1.0 # .jpg image -> for visualization
   
    wild_dataset = WildDetDataset(loaded_json, clip_len=opts.clip_len, scale_range=[1,1], focus=opts.focus)

    test_loader = DataLoader(wild_dataset, **testloader_params)

    results_all = []
    with torch.no_grad():
        for batch_input in tqdm(test_loader):
            N, T = batch_input.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if args.flip:    
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip) # Flip back
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
            else:
                predicted_3d_pos = model_pos(batch_input)
            if args.rootrel:
                predicted_3d_pos[:,:,0,:]=0                    # [N,T,17,3]
            else:
                predicted_3d_pos[:,0,0,2]=0
                pass
            if args.gt_2d:
                predicted_3d_pos[...,:2] = batch_input[...,:2]
            results_all.append(predicted_3d_pos.cpu().numpy())

    results_all = np.hstack(results_all)
    results_all = np.concatenate(results_all)

    # Optional saving of results to files and rendering visualization
    if opts.debug:
        render_and_save(results_all, '%s/human.mp4' % (opts.out_path), keep_imgs=True, fps=fps_in, draw_face=True)
        np.save('%s/human.npy' % (opts.out_path), results_all)

    received_data["human_joints_positions"] = results_all
    print("Results saved, MotionBERT waiting for another picture...")

    return received_data

if __name__ == "__main__":

    opts = parse_args()
    print(opts)

    # Load models and checkpoints before processing.
    args = get_config(opts.config)

    model_backbone = load_backbone(args)
    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    print('Loading checkpoint', opts.evaluate)
    checkpoint = torch.load(opts.evaluate, map_location=lambda storage, loc: storage)

    # Code for renaming parts loaded from checkpoint, used when error occures with missing keys in checkpoint
    """ 
    from collections import OrderedDict
    new_checkpoint = OrderedDict()
    for k, v in checkpoint['model_pos'].items():
        name = k[7:] # remove module.
        new_checkpoint[name] = v

    model_backbone.load_state_dict(new_checkpoint, strict=True)
    """

    model_backbone.load_state_dict(checkpoint["model_pos"], strict=True)
    model_pos = model_backbone
    model_pos.eval()
    testloader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 4,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': True,
            'drop_last': False
    }
    os.makedirs(opts.out_path, exist_ok=True)

    image_service_server = ServiceServer(callback_image_server, port=242426)
    print("MotionBERT image server running, waiting for new image...")

    # Keep the script alive and waiting for client request.
    try:
        while(True):
            time.sleep(1)
    except KeyboardInterrupt as e:
        print(e)

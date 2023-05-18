"""Script for single-gpu/multi-gpu demo."""
import argparse
import os
import platform
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
import natsort

from image_share_service.service_server import ServiceServer

from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.file_detector import FileDetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter

"""----------------------------- Demo options -----------------------------"""

args = argparse.Namespace(
    cfg='configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml',
    checkpoint='pretrained_models/halpe26_fast_res50_256x192.pth',
    debug=False,
    detbatch=5,
    detector='yolo',
    detfile='',
    eval=False,
    flip=False,
    format=None,
    gpus='0',
    inputimg='examples/my_demo/human.jpg', #human_4m.jpg'
    inputlist='',
    inputpath='',
    min_box_area=0,
    outputpath='examples/res/',
    pose_flow=False,
    pose_track=False,
    posebatch=64,
    profile=True, 
    qsize=1024,
    save_img=True,
    save_video=False,
    showbox=False,
    sp=False,
    video='',
    vis=False,
    vis_fast=False,
    webcam=-1,
)

cfg = update_config(args.cfg)

if platform.system() == 'Windows':
    args.sp = True

args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.detbatch = args.detbatch * len(args.gpus)
args.posebatch = args.posebatch * len(args.gpus)
args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'

if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

print("args = ")
print(args)

def check_input():
    # for images
    if len(args.inputimg):
        inputimg = args.inputimg

        if len(inputimg):
            args.inputpath = os.path.split(inputimg)[0]
            im_names = [os.path.split(inputimg)[1]]

        return 'image', im_names

    else:
        raise NotImplementedError


def print_finish_info():
    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')

def loop():
    n = 0
    while True:
        yield n
        n += 1

def callback_image_server(stuff:dict):
    """
    Funkce ktera zpracuje obraz na vyzadani od klienta a odpovi az ho zpracuje a zapise do slozky
    """
    print(stuff)
    stuff["Hello"] = "ROS"

    
    det_loader = DetectionLoader(input_source, get_detector(args), cfg, args, batchSize=args.detbatch, mode=mode, queueSize=args.qsize)
    det_worker = det_loader.start()

    

    data_len = det_loader.length
    im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

    try:
        for i in im_names_desc:
            start_time = getTime()
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                if orig_img is None:
                    break
                if boxes is None or boxes.nelement() == 0:
                    writer.save(None, None, None, None, None, orig_img, im_name)
                    continue
                if args.profile:
                    ckpt_time, det_time = getTime(start_time)
                    runtime_profile['dt'].append(det_time)
                # Pose Estimation
                inps = inps.to(args.device)
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                    hm_j = pose_model(inps_j)
                    hm.append(hm_j)
                hm = torch.cat(hm)
                if args.profile:
                    ckpt_time, pose_time = getTime(ckpt_time)
                    runtime_profile['pt'].append(pose_time)
                hm = hm.cpu()
                writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
                if args.profile:
                    ckpt_time, post_time = getTime(ckpt_time)
                    runtime_profile['pn'].append(post_time)

            if args.profile:
                # TQDM
                im_names_desc.set_description(
                    'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                        dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
                )
        print_finish_info()
        while(writer.running()):
            time.sleep(1)
            print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...', end='\r')
        #writer.stop()
        det_loader.stop()
        print("Image recognition with AlphaPose finished.")
    except Exception as e:
        print(repr(e))
        print('An error as above occurs when processing the images, please check it')
        pass
    except KeyboardInterrupt:
        print_finish_info()
        # Thread won't be killed when press Ctrl+C
        if args.sp:
            det_loader.terminate()
            while(writer.running()):
                time.sleep(1)
                print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...', end='\r')
            writer.stop()
        else:
            # subprocesses are killed, manually clear queues

            det_loader.terminate()
            writer.terminate()
            writer.clear_queues()
            det_loader.clear_queues()

    stuff["AlphaPose"] = "finished"
    return stuff




if __name__ == "__main__":
    mode, input_source = check_input()

    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)

    # Load pose model
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print('Loading pose model from %s...' % (args.checkpoint,))
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
    
    if len(args.gpus) > 1:
        pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(args.device)
    else:
        pose_model.to(args.device)
    pose_model.eval()
    
    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }
    # Init data writer
    queueSize = args.qsize
    batchSize = args.posebatch

    # init data writer
    writer = DataWriter(cfg, args, save_video=False, queueSize=queueSize).start()

    image_processing_server = ServiceServer(callback_image_server, port=242425)
    print("Image processing server with AplhaPose is ready for file...")

    try:
        while(True):
            time.sleep(1)
    except KeyboardInterrupt as e:
        print(e)



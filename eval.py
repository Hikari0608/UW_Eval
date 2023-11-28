'''
eval metrics for two folder (input, gt) or single folder (only input)
'''

import torch
from pathlib import Path
import sys
import os
import argparse
import platform
from core.monitor import Monitor
from core.Losses import (Metrics, PSNR, SSIM, NegMetric)
from torchvision.transforms import ToTensor, Resize

from PIL import Image
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # get root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))


def parse_opt(known=False):
    path_dict = {
        'input': r'/share/zhangdan2013/p/增强test/增强test/output',
        'gt': r'/share/zhangdan2013/code/datasets/UIEB_end/val/target'
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--input', default=path_dict['input'], help='input folder')
    parser.add_argument('--gt', default=path_dict['gt'], help='gt folder')
    parser.add_argument('--resize', default=0, help='save result to path')
    parser.add_argument('--save_dir', default='./res/', help='save result to path')
    
    return parser.parse_known_args()[0] if known else parser.parse_args()

def channelsAndstd(image):
    color_index = np.sum(np.sum(image, axis=0), axis=0)/ image.shape[0]/ image.shape[1]
    color_std = np.std(color_index, axis=0)
    return color_index, color_std

def cast(image, resize):
    #array = np.array(image).astype(np.float32)/255.
    resize = int(resize)
    array = ToTensor()(np.array(image))
    if resize != 0:
        array = Resize((resize, resize))(array)
    array = array.unsqueeze(0)
    return array

def main(opt, device):  # hyp is path/to/hyp.yaml or hyp dictionary
    save_dir = Path(opt.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log = Monitor(save_dir)

    input_path = Path(opt.input)
    gt_path = Path(opt.gt)

    metrics = Metrics()
    metrics.add([PSNR(), SSIM(), NegMetric()])
    
    filenames = os.listdir(opt.input)
    filenames = [x for x in filenames if x.endswith(('jpg', 'png', 'jpeg'))]
    check_ref = os.path.isfile(gt_path / filenames[0])

    gt_path = input_path if not check_ref else gt_path

    metrics.clear()
    for _, file in enumerate(filenames):
        print(file)

        input, gt = Image.open(input_path / file),  Image.open(gt_path / file) # Image.open(gt_path / str(file.split('_')[0] + '.png'))
        input = cast(input, opt.resize)
        gt = cast(gt, opt.resize)
        metrics(input, gt)
        #print(channelsAndstd(input))
        log.metricsWriter(_, metrics.back())
        print(metrics.back())
    metric = metrics.output(len(filenames))

    print('avg metric:{}'.format(metric))

if __name__ == '__main__':  
    opt = parse_opt()
    main(opt, torch.device('cuda:'+ opt.device))
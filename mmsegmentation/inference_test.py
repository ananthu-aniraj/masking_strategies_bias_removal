from mmseg.apis import init_model, inference_model, show_result_pyplot
import cv2
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference for mmseg model'
    )
    parser.add_argument('--data_path',
                        help='directory that contains cub files, must'
                             'contain folder "./images"')
    parser.add_argument('--config_file',
                        help='path to config file')
    parser.add_argument('--checkpoint_file',
                        help='path to checkpoint file')
    parser.add_argument('--save_path', help='path to save directory')
    parser.add_argument('--save_seg_masks', help='save segmentation masks', action='store_true')
    parser.add_argument('--alpha', default=0.5, type=float)
    args = parser.parse_args()
    return args


def inference_and_save(args):
    # Initialize paths
    config_file = args.config_file
    checkpoint_file = args.checkpoint_file
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    alpha = args.alpha
    beta = 1 - alpha
    base_img_path = args.data_path
    img_paths = glob.glob(base_img_path + "/*.jpg")
    save_path_base = args.save_path

    # build the model from a config file and a checkpoint file
    model = init_model(config_file, checkpoint_file, device=device)

    for img_path in tqdm(img_paths, desc="Processing images"):
        img = cv2.imread(img_path)
        with torch.inference_mode():
            result = inference_model(model, img)
        if not args.save_seg_masks:
            result_gray = np.moveaxis(result.pred_sem_seg.data.cpu().numpy(), 0, -1) * 255
            result_color = cv2.applyColorMap(result_gray.astype(np.uint8), cv2.COLORMAP_JET)
            vis_image = cv2.addWeighted(img, alpha, result_color, beta, 0)
        else:
            vis_image = np.moveaxis(result.pred_sem_seg.data.cpu().numpy(), 0, -1).squeeze(axis=-1)

        class_name = img_path.split("/")[-2]
        save_path_folder = os.path.join(save_path_base, class_name)
        Path(save_path_folder).mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(save_path_folder, img_path.split("/")[-1])

        cv2.imwrite(save_path, vis_image)


if __name__ == "__main__":
    arguments = parse_args()
    inference_and_save(arguments)

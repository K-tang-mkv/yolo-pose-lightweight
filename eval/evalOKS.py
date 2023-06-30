import os.path

import torch
import numpy as np
import argparse
from pathlib import Path


def oks(pred: str, gt: str):
    """
    pred: prediction labels
    gt: ground truth labels
    """
    pred = Path(pred).joinpath('labels/') if 'labels' not in pred else Path(pred)
    gt = Path(gt).joinpath('labels') if 'labels' not in gt else Path(gt)

    assert os.path.exists(pred) and os.path.exists(gt), f'{str(pred)} or {str(gt)} not exist'

    results = []
    count = 0
    for gt_label in os.listdir(gt):
        if not gt_label.endswith(".txt"):
            continue
        count += 1
        print(gt_label)
        pred_label = pred.joinpath(gt_label)
        gt_label = gt.joinpath(gt_label)

        result = one_img_oks(pred_label, gt_label)
        results.append(result)

    return sum(results) / count


def one_img_oks(pred_label, gt_label):
    '''
    for each image, we can calculate its oks
    '''
    if not os.path.exists(pred_label):
        return 0.

    sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0

    pred = []
    gt = []
    with open(pred_label, 'r') as f:
        for line in f.readlines():
            pred.append([float(i) for i in line.split(' ')])
    with open(gt_label, 'r') as f:
        for line in f.readlines():
            gt.append([float(i) for i in line.split(' ')])

    result = []
    for i in range(len(gt)):
        gta = np.array(gt[i])
        if i+1 > len(pred):
            break
        preda = np.array(pred[i])
        s = np.prod(gta[3:5])
        d = (preda[5::3]-gta[5::3])**2 + (preda[6::3]-gta[6::3])**2
        kpt_mask = gta[7::3] > 0
        oks = np.sum(np.exp(-d/(2*s*4*sigmas**2))*kpt_mask)/np.sum(kpt_mask)
        result.append(oks)
    result = sum(result) / len(gt)

    return result

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", default="../best.pt", help="model")
    parser.add_argument("--pred_dir", default="../output/", help="prediction files")
    parser.add_argument("--gt_dir", default="../test_img/", help="target files")


if __name__ == "__main__":
    result = one_img_oks('../runs/detect/exp45/labels/000000000785.txt', '../coco_kpts/labels/val2017/000000000785.txt')
    print(result)
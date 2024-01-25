import os
import os.path as osp
import time
import math
import random
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset, generate_score_geo_maps
from dataset import SceneTextDataset
from model import EAST
from detect import get_bboxes, detect
from deteval import calc_deteval_metrics

import wandb
import glob
import re
import numpy as np
from pathlib import Path


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../data/medical'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--ignore_tags', type=list, default=['masked', 'excluded-region', 'maintable', 'stamp'])

    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--run_name', type=str, default='baseline_fold0')

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


def increment_path(path, exist_ok=False):
    """Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, ignore_tags, seed, run_name):

    ckpt_fpath = increment_path(os.path.join(model_dir, run_name))
    if not osp.exists(ckpt_fpath):
                os.makedirs(ckpt_fpath)
                
    max_hmean = 0.0

    wandb.init(
        project='ocr',
        entity='cv2-ocr',
        name=run_name,
    )
    
    dataset_train = SceneTextDataset(
        data_dir,
        split='0_train',
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags,
        color_jitter=True
    )
    dataset_valid = SceneTextDataset(
        data_dir,
        split='0_valid',
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags,
        color_jitter=False
    )
    dataset_train = EASTDataset(dataset_train)
    dataset_valid = EASTDataset(dataset_valid)

    num_batches = math.ceil(len(dataset_train) / batch_size)
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    valid_loader = DataLoader(
        dataset_valid,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)
    
    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)

                # wandb logging(batch)
                wandb.log({
                    'Cls loss': extra_info['cls_loss'], 
                    'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                })

        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        # valid
        valid_cls, valid_angle, valid_iou = 0.0, 0.0, 0.0
        valid_precision, valid_recall, valid_hmean = 0.0, 0.0, 0.0

        gt_bboxes = {}
        pred_bboxes = {}

        model.eval()
        with torch.no_grad():
            with tqdm(total=len(dataset_valid)) as pbar:
                pbar.set_description('[Epoch {} valid]'.format(epoch + 1))
                for img, gt_score_map, gt_geo_map, roi_mask in valid_loader:
                    # loss
                    _, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                    valid_cls += extra_info['cls_loss']
                    valid_angle += extra_info['angle_loss']
                    valid_iou += extra_info['iou_loss'] 

                    # deteval(f1, precision, recall)
                    pred_score, pred_geo = model(img.to(device))

                    pred_bboxes = get_bboxes(pred_score.cpu().numpy()[0], pred_geo.cpu().numpy()[0])
                    gt_bboxes = get_bboxes(gt_score_map.numpy()[0], gt_geo_map.numpy()[0])

                    pred_bboxes = {0: np.expand_dims(pred_bboxes, axis=0)}
                    gt_bboxes = {0: np.expand_dims(gt_bboxes, axis=0)}

                    metric = calc_deteval_metrics(pred_bboxes, gt_bboxes, transcriptions_dict={0: [""]*len(gt_bboxes[0][0])})
                    valid_precision += metric['total']['precision']
                    valid_recall += metric['total']['recall']
                    valid_hmean += metric['total']['hmean']

                    pbar.update(1)

            wandb.log({
                'valid_cls': valid_cls/len(dataset_valid),
                'valid_angle': valid_angle/len(dataset_valid),
                'valid_iou': valid_iou/len(dataset_valid),
                'valid_precision': valid_precision/len(dataset_valid),
                'valid_recall': valid_recall/len(dataset_valid),
                'valid_hmean': valid_hmean/len(dataset_valid),
            })
        model.train()

        # model save(last)
        path = osp.join(ckpt_fpath, 'latest.pth')
        torch.save(model.state_dict(), path)
        
        # model save(best)
        current_hmean = valid_hmean/len(dataset_valid)
        if current_hmean > max_hmean:
            print(f'New best model for hmean: {current_hmean:4.4}! saving the best model')
            max_hmean = current_hmean
            path = osp.join(ckpt_fpath, 'best_hmean.pth')
            torch.save(model.state_dict(), path)
        
        print(
                f"[Val] hmean: {current_hmean:4.4} || "
                f"best hmean: {max_hmean:4.4}"
            )

        # wandb logging(epoch)
        wandb.log({
            'Mean loss': epoch_loss / num_batches,
        })
    
    wandb.finish()


def main(args):
    seed_everything(args.seed)
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)

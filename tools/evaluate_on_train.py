#!/usr/bin/env python
"""
Evaluate a trained HACDR-Net model on the TRAINING set.

Usage:
    python evaluate_on_train.py configs/HACDR_idrid.py \
        save_dir/HACDRNet_idrid/iter_40000.pth

This will compute IoU, Dice, AUPR, etc. on the training data.
"""

import argparse
import os
import os.path as osp

import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate HACDR-Net on training set')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='directory to save results',
        default='./eval_train_results')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default=['mDice', 'mAUPR'],
        help='evaluation metrics')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='GPU id to use')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Load config
    cfg = mmcv.Config.fromfile(args.config)
    
    # Create work directory
    mmcv.mkdir_or_exist(osp.abspath(args.work_dir))

    # Build the model
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # Load checkpoint
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    # Move model to GPU
    model = MMDataParallel(model, device_ids=[args.gpu_id])
    model.eval()

    # ===================================================================
    # KEY PART: Build TRAINING dataset with TEST-TIME pipeline
    # ===================================================================
    # We use the training data but with the TEST pipeline (no augmentation)
    
    train_dataset_cfg = cfg.data.train.copy()
    
    # Replace training pipeline with test pipeline (no augmentation)
    train_dataset_cfg.pipeline = cfg.data.test.pipeline
    
    # Build dataset
    dataset = build_dataset(train_dataset_cfg)
    
    # Build dataloader
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    print(f'\n{"="*70}')
    print(f'Evaluating on TRAINING set')
    print(f'{"="*70}')
    print(f'Config: {args.config}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Number of training images: {len(dataset)}')
    print(f'Metrics: {args.eval}')
    print(f'{"="*70}\n')

    # Run evaluation
    results = single_gpu_test(model, data_loader, show=False)

    # Compute metrics
    metric_results = dataset.evaluate(
        results,
        metric=args.eval,
        logger='print')

    # Print results
    print(f'\n{"="*70}')
    print(f'TRAINING SET RESULTS')
    print(f'{"="*70}')
    
    # Format and print
    for key, value in metric_results.items():
        if isinstance(value, (list, tuple)):
            print(f'{key}:')
            for i, v in enumerate(value):
                print(f'  Class {i}: {v:.4f}')
        else:
            print(f'{key}: {value:.4f}')

    # Save results to file
    result_file = osp.join(args.work_dir, 'train_metrics.json')
    mmcv.dump(metric_results, result_file, indent=4)
    print(f'\n✅ Results saved to: {result_file}')
    print(f'{"="*70}\n')


if __name__ == '__main__':
    main()

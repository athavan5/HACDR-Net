model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='HACDR',
        embed_dims=[32, 64, 160, 256],
        DFFN_ratios=[8, 8, 4, 4],
        drop_rate=0.1,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 2],
        attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
        attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
        option_k=7,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='BN', requires_grad=True)),
    decode_head=dict(
        type='UnetHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=32,
        last_channels=32,
        dropout_ratio=0.2,
        num_classes=5,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            #dict(
                #type='CrossEntropyLoss',
                #loss_name='loss_1',
                #use_sigmoid=False,
                #loss_weight=1.0,
                #class_weight=[0.5, 0.7, 1.5, 0.7, 1.1],
            #),
            dict(type='NALoss',
                 loss_name='loss_2',
                 thres=0.5,
                 loss_weight=0.2,
                 cls_num_list=[102380331, 502150, 20735, 84662, 692122],
                 class_weight=[0.5, 0.7, 1.5, 0.7, 1.1],
                 sigma=4, ),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4, #originally 4
    train=dict(
            type='FGADRDataset',
            data_root='./data/eyepacs_1440x960/',
            img_dir='splits/train/images',
            ann_dir='splits/train/masks',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(type='Resize', img_scale=(960, 1440), ratio_range=(0.5, 2.0)),
                dict(type='RandomCrop', crop_size=(960, 1440), cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(type='PhotoMetricDistortion'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size=(960, 1440), pad_val=0, seg_pad_val=255),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ],
            split='train.txt'),
    val=dict(
            type='FGADRDataset',
            data_root='./data/eyepacs_1440x960/',
            img_dir='splits/val/images',
            ann_dir='splits/val/masks',
            split='val.txt',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='MultiScaleFlipAug',
                    img_scale=(960, 1440),
                    flip=False,
                    transforms=[
                        dict(type='Resize', keep_ratio=True),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ])
            ]),
    test=dict(
            type='FGADRDataset',
            data_root='./data/eyepacs_1440x960/',
            img_dir='splits/test/images',
            ann_dir='splits/test/masks',
            split='test.txt',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='MultiScaleFlipAug',
                    img_scale=(960, 1440),
                    flip=False,
                    transforms=[
                        dict(type='Resize', keep_ratio=True),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ])
            ])
    )
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05, #lr initially 6e-05
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=15.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0, #min_lr initially 0.0
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mDice', pre_eval=True)
work_dir = './save_dir/HACDRNet_idrid/'
gpu_ids = [5]
auto_resume = False

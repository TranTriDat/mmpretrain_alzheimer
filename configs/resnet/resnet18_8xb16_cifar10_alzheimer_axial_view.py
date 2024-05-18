_base_ = [
    '../_base_/models/resnet18.py',
    '../_base_/schedules/imagenet_bs1024_adamw_conformer.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier', 
    backbone=dict(
        type='ResNet_CIFAR',
        init_cfg = dict(
            type='Pretrained', 
            #type='ResNet', 
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth', 
            prefix='backbone')
    ),
    head=dict(
        num_classes=3,
        topk = (1, ),
    ))

dataset_type = 'CustomDataset'
data_preprocessor = dict(
     mean=[124.508, 116.050, 106.438],
     std=[58.577, 57.310, 57.437],
     to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),     # read image
    dict(type='RandomResizedCrop', scale=100),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),   # random horizontal flip
    dict(type='PackInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),     # read image
    dict(type='ResizeEdge', scale=100),  # Scale the short side to 256
    dict(type='CenterCrop', crop_size=100),
    dict(type='PackInputs'),    
]

train_dataloader = dict(
    batch_size=16,
    num_workers=1,
    dataset=dict(
        type='CustomDataset',
        data_prefix='data/YoriDataset_vgg/train',
        classes='data/classes.txt',
        # ann_file='data/train_ann.txt',
        # with_label=True,
        pipeline=train_pipeline
    ),
    persistent_workers=True
)

val_dataloader = dict(
    batch_size=32,
    num_workers=1,
    dataset=dict(
        type='CustomDataset',
        data_prefix='data/YoriDataset_vgg/validation',
        classes='data/classes.txt',
        # ann_file='data/val_ann.txt',
        # with_label=True,
        pipeline=test_pipeline
    ),
    persistent_workers=True
)

test_dataloader = dict(
    batch_size=32,
    num_workers=1,
    dataset=dict(
        type='CustomDataset',
        data_prefix='data/YoriDataset_vgg/test',
        classes='data/classes.txt',
        # ann_file='data/test_ann.txt',
        # with_label=True,
        pipeline=test_pipeline
    ),    
    persistent_workers=True
)

val_evaluator = dict(type='Accuracy', topk=(1, ))
test_evaluator = val_evaluator 

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        # for batch in each gpu is 128, 8 gpu
        # lr = 5e-4 * 128 * 8 / 512 = 0.001
        lr=5e-4 * 128 * 8 / 512,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.cls_token': dict(decay_mult=0.0),
        }),
)

# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=295,
        eta_min=1e-5,
        by_epoch=True,
        begin=5,
        end=300)
]

train_cfg = dict(by_epoch=True, max_epochs=3, val_interval=1)
val_cfg = dict()
test_cfg = dict()

auto_scale_lr = dict(base_batch_size=128)

default_scope = 'mmpretrain'

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1),

    # set sampler seed in a distributed environment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)

env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi-process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
vis_backends = [dict(type='LocalVisBackend')]  # use local HDD backend
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends, name='visualizer')

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

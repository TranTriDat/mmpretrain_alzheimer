_base_ = [
    '../_base_/models/swin_transformer/tiny_224.py', 
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

pretrained='https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer', 
        arch='tiny',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained,
            prefix='backbone'
        ),
    ),
    head=dict(
        num_classes=8,
        topk=(1, 5),
    ))

dataset_type = 'CustomDataset'
# dataset_type = 'ImageNet'
data_preprocessor = dict(
    num_classes=8,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

train_pipeline = [
    dict(type='LoadImageFromFile'),     # read image
    # dict(type='ResizeEdge', scale=256),  # Scale the short side to 256
    dict(type='RandomResizedCrop', scale=224, backend='pillow', interpolation='bicubic'), # 224x224
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),   # random horizontal flip
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='PackInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),     # read image
    # dict(type='ResizeEdge', scale=256,edge='short',backend='pillow',interpolation='bicubic'),  # Scale the short side to 256
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),    
]

train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='../Dat/Vit/lora',
        data_prefix='ISIC2019_train/Image',
        classes='data/classes_ISIC_2019.txt',
        with_label=True,
        split='train',
        pipeline=train_pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True
)

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='../Dat/Vit/lora',
        data_prefix='ISIC2019_train/Image',
        classes='data/classes_ISIC_2019.txt',
        with_label=True,
        pipeline=test_pipeline,
        split='val'
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True
)

test_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='../Dat/Vit/lora',
        data_prefix='ISIC2019_train/Image',
        classes='data/classes_ISIC_2019.txt',
        with_label=True,
        pipeline=test_pipeline,
        split='val'
    ),    
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True
)

val_evaluator = dict(type='Accuracy', topk=(1, 5))
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

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
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

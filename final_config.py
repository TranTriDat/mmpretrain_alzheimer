auto_scale_lr = [1;35mdict[0m[1m([0m[33mbase_batch_size[0m=[1;36m1024[0m[1m)[0m
data = [1;35mdict[0m[1m([0m
    [33msamples_per_gpu[0m=[1;36m32[0m,
    [33mtest[0m=[1;35mdict[0m[1m([0m
        [33mclasses[0m=[32m'/content/drive/MyDrive/YZU/YoriDataset/classes.txt'[0m,
        [33mdata_prefix[0m=[32m'/content/drive/MyDrive/YZU/YoriDataset/test/AXIAL'[0m,
        [33mpipeline[0m=[1m[[0m
            [1;35mdict[0m[1m([0m[33mtype[0m=[32m'LoadImageFromFile'[0m[1m)[0m,
            [1;35mdict[0m[1m([0m[33mbackend[0m=[32m'pillow'[0m, [33msize[0m=[1m([0m
                [1;36m100[0m,
                [1;36m-1[0m,
            [1m)[0m, [33mtype[0m=[32m'Resize'[0m[1m)[0m,
            [1;35mdict[0m[1m([0m[33mcrop_size[0m=[1;36m100[0m, [33mtype[0m=[32m'CenterCrop'[0m[1m)[0m,
            [1;35mdict[0m[1m([0m
                [33mmean[0m=[1m[[0m
                    [1;36m124.508[0m,
                    [1;36m116.05[0m,
                    [1;36m106.438[0m,
                [1m][0m,
                [33mstd[0m=[1m[[0m
                    [1;36m58.577[0m,
                    [1;36m57.31[0m,
                    [1;36m57.437[0m,
                [1m][0m,
                [33mto_rgb[0m=[3;91mFalse[0m,
                [33mtype[0m=[32m'Normalize'[0m[1m)[0m,
            [1;35mdict[0m[1m([0m[33mkeys[0m=[1m[[0m
                [32m'img'[0m,
            [1m][0m, [33mtype[0m=[32m'ImageToTensor'[0m[1m)[0m,
            [1;35mdict[0m[1m([0m[33mkeys[0m=[1m[[0m
                [32m'img'[0m,
            [1m][0m, [33mtype[0m=[32m'Collect'[0m[1m)[0m,
        [1m][0m,
        [33mtype[0m=[32m'CustomDataset'[0m[1m)[0m,
    [33mtrain[0m=[1;35mdict[0m[1m([0m
        [33mclasses[0m=[32m'/content/drive/MyDrive/YZU/YoriDataset/classes.txt'[0m,
        [33mdata_prefix[0m=[32m'/content/drive/MyDrive/YZU/YoriDataset/train/AXIAL'[0m,
        [33mpipeline[0m=[1m[[0m
            [1;35mdict[0m[1m([0m[33mtype[0m=[32m'LoadImageFromFile'[0m[1m)[0m,
            [1;35mdict[0m[1m([0m[33mbackend[0m=[32m'pillow'[0m, [33msize[0m=[1;36m100[0m, [33mtype[0m=[32m'RandomResizedCrop'[0m[1m)[0m,
            [1;35mdict[0m[1m([0m
                [33mmean[0m=[1m[[0m
                    [1;36m124.508[0m,
                    [1;36m116.05[0m,
                    [1;36m106.438[0m,
                [1m][0m,
                [33mstd[0m=[1m[[0m
                    [1;36m58.577[0m,
                    [1;36m57.31[0m,
                    [1;36m57.437[0m,
                [1m][0m,
                [33mto_rgb[0m=[3;91mFalse[0m,
                [33mtype[0m=[32m'Normalize'[0m[1m)[0m,
            [1;35mdict[0m[1m([0m[33mkeys[0m=[1m[[0m
                [32m'img'[0m,
            [1m][0m, [33mtype[0m=[32m'ImageToTensor'[0m[1m)[0m,
            [1;35mdict[0m[1m([0m[33mkeys[0m=[1m[[0m
                [32m'gt_label'[0m,
            [1m][0m, [33mtype[0m=[32m'ToTensor'[0m[1m)[0m,
            [1;35mdict[0m[1m([0m[33mkeys[0m=[1m[[0m
                [32m'img'[0m,
                [32m'gt_label'[0m,
            [1m][0m, [33mtype[0m=[32m'Collect'[0m[1m)[0m,
        [1m][0m,
        [33mtype[0m=[32m'CustomDataset'[0m[1m)[0m,
    [33mval[0m=[1;35mdict[0m[1m([0m
        [33mclasses[0m=[32m'/content/drive/MyDrive/YZU/YoriDataset/classes.txt'[0m,
        [33mdata_prefix[0m=[32m'/content/drive/MyDrive/YZU/YoriDataset/validation/AXIAL'[0m,
        [33mpipeline[0m=[1m[[0m
            [1;35mdict[0m[1m([0m[33mtype[0m=[32m'LoadImageFromFile'[0m[1m)[0m,
            [1;35mdict[0m[1m([0m[33mbackend[0m=[32m'pillow'[0m, [33msize[0m=[1m([0m
                [1;36m100[0m,
                [1;36m-1[0m,
            [1m)[0m, [33mtype[0m=[32m'Resize'[0m[1m)[0m,
            [1;35mdict[0m[1m([0m[33mcrop_size[0m=[1;36m100[0m, [33mtype[0m=[32m'CenterCrop'[0m[1m)[0m,
            [1;35mdict[0m[1m([0m
                [33mmean[0m=[1m[[0m
                    [1;36m124.508[0m,
                    [1;36m116.05[0m,
                    [1;36m106.438[0m,
                [1m][0m,
                [33mstd[0m=[1m[[0m
                    [1;36m58.577[0m,
                    [1;36m57.31[0m,
                    [1;36m57.437[0m,
                [1m][0m,
                [33mto_rgb[0m=[3;91mFalse[0m,
                [33mtype[0m=[32m'Normalize'[0m[1m)[0m,
            [1;35mdict[0m[1m([0m[33mkeys[0m=[1m[[0m
                [32m'img'[0m,
            [1m][0m, [33mtype[0m=[32m'ImageToTensor'[0m[1m)[0m,
            [1;35mdict[0m[1m([0m[33mkeys[0m=[1m[[0m
                [32m'img'[0m,
            [1m][0m, [33mtype[0m=[32m'Collect'[0m[1m)[0m,
        [1m][0m,
        [33mtype[0m=[32m'CustomDataset'[0m[1m)[0m,
    [33mworkers_per_gpu[0m=[1;36m2[0m[1m)[0m
dataset_type = [32m'CustomDataset'[0m
default_hooks = [1;35mdict[0m[1m([0m
    [33mcheckpoint[0m=[1;35mdict[0m[1m([0m[33minterval[0m=[1;36m1[0m, [33mtype[0m=[32m'CheckpointHook'[0m[1m)[0m,
    [33mlogger[0m=[1;35mdict[0m[1m([0m[33minterval[0m=[1;36m100[0m, [33mtype[0m=[32m'LoggerHook'[0m[1m)[0m,
    [33mparam_scheduler[0m=[1;35mdict[0m[1m([0m[33mtype[0m=[32m'ParamSchedulerHook'[0m[1m)[0m,
    [33msampler_seed[0m=[1;35mdict[0m[1m([0m[33mtype[0m=[32m'DistSamplerSeedHook'[0m[1m)[0m,
    [33mtimer[0m=[1;35mdict[0m[1m([0m[33mtype[0m=[32m'IterTimerHook'[0m[1m)[0m,
    [33mvisualization[0m=[1;35mdict[0m[1m([0m[33menable[0m=[3;91mFalse[0m, [33mtype[0m=[32m'VisualizationHook'[0m[1m)[0m[1m)[0m
default_scope = [32m'mmpretrain'[0m
env_cfg = [1;35mdict[0m[1m([0m
    [33mcudnn_benchmark[0m=[3;91mFalse[0m,
    [33mdist_cfg[0m=[1;35mdict[0m[1m([0m[33mbackend[0m=[32m'nccl'[0m[1m)[0m,
    [33mmp_cfg[0m=[1;35mdict[0m[1m([0m[33mmp_start_method[0m=[32m'fork'[0m, [33mopencv_num_threads[0m=[1;36m0[0m[1m)[0m[1m)[0m
evaluation = [1;35mdict[0m[1m([0m[33mmetric[0m=[32m'accuracy'[0m, [33mmetric_options[0m=[1;35mdict[0m[1m([0m[33mtopk[0m=[1m([0m[1;36m1[0m, [1m)[0m[1m)[0m[1m)[0m
img_norm_cfg = [1;35mdict[0m[1m([0m
    [33mmean[0m=[1m[[0m
        [1;36m124.508[0m,
        [1;36m116.05[0m,
        [1;36m106.438[0m,
    [1m][0m,
    [33mstd[0m=[1m[[0m
        [1;36m58.577[0m,
        [1;36m57.31[0m,
        [1;36m57.437[0m,
    [1m][0m,
    [33mto_rgb[0m=[3;91mFalse[0m[1m)[0m
load_from = [3;35mNone[0m
log_config = [1;35mdict[0m[1m([0m[33minterval[0m=[1;36m10[0m[1m)[0m
log_level = [32m'INFO'[0m
lr_config = [1;35mdict[0m[1m([0m
    [33mby_epoch[0m=[3;91mFalse[0m,
    [33mmin_lr_ratio[0m=[1;36m0[0m[1;36m.01[0m,
    [33mpolicy[0m=[32m'CosineAnnealing'[0m,
    [33mwarmup[0m=[32m'linear'[0m,
    [33mwarmup_by_epoch[0m=[3;91mFalse[0m,
    [33mwarmup_iters[0m=[1;36m6260[0m,
    [33mwarmup_ratio[0m=[1;36m0[0m[1;36m.001[0m[1m)[0m
model = [1;35mdict[0m[1m([0m
    [33mbackbone[0m=[1;35mdict[0m[1m([0m
        [33mdepth[0m=[1;36m18[0m,
        [33minit_cfg[0m=[1;35mdict[0m[1m([0m
            [33mcheckpoint[0m=
            [32m'https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_[0m
[32m8xb32_in1k_20210831-fbbb1da6.pth'[0m,
            [33mprefix[0m=[32m'backbone'[0m,
            [33mtype[0m=[32m'Pretrained'[0m[1m)[0m,
        [33mnum_stages[0m=[1;36m4[0m,
        [33mout_indices[0m=[1m([0m[1;36m3[0m, [1m)[0m,
        [33mstyle[0m=[32m'pytorch'[0m,
        [33mtype[0m=[32m'ResNet'[0m[1m)[0m,
    [33mhead[0m=[1;35mdict[0m[1m([0m
        [33mcal_acc[0m=[3;92mTrue[0m,
        [33min_channels[0m=[1;36m512[0m,
        [33mloss[0m=[1;35mdict[0m[1m([0m[33mloss_weight[0m=[1;36m1[0m[1;36m.0[0m, [33mtype[0m=[32m'CrossEntropyLoss'[0m[1m)[0m,
        [33mnum_classes[0m=[1;36m3[0m,
        [33mtopk[0m=[1m([0m[1;36m1[0m, [1m)[0m,
        [33mtype[0m=[32m'LinearClsHead'[0m[1m)[0m,
    [33mneck[0m=[1;35mdict[0m[1m([0m[33mtype[0m=[32m'GlobalAveragePooling'[0m[1m)[0m,
    [33mtype[0m=[32m'ImageClassifier'[0m[1m)[0m
optim_wrapper = [1;35mdict[0m[1m([0m
    [33moptimizer[0m=[1;35mdict[0m[1m([0m
        [33mbetas[0m=[1m([0m
            [1;36m0.9[0m,
            [1;36m0.999[0m,
        [1m)[0m,
        [33meps[0m=[1;36m1e[0m[1;36m-08[0m,
        [33mlr[0m=[1;36m0[0m[1;36m.001[0m,
        [33mtype[0m=[32m'AdamW'[0m,
        [33mweight_decay[0m=[1;36m0[0m[1;36m.05[0m[1m)[0m,
    [33mparamwise_cfg[0m=[1;35mdict[0m[1m([0m
        [33mbias_decay_mult[0m=[1;36m0[0m[1;36m.0[0m,
        [33mcustom_keys[0m=[1;35mdict[0m[1m([0m[1m{[0m[32m'.cls_token'[0m: [1;35mdict[0m[1m([0m[33mdecay_mult[0m=[1;36m0[0m[1;36m.0[0m[1m)[0m[1m}[0m[1m)[0m,
        [33mnorm_decay_mult[0m=[1;36m0[0m[1;36m.0[0m[1m)[0m[1m)[0m
optimizer = [1;35mdict[0m[1m([0m
    [33mbetas[0m=[1m([0m
        [1;36m0.9[0m,
        [1;36m0.999[0m,
    [1m)[0m,
    [33meps[0m=[1;36m1e[0m[1;36m-08[0m,
    [33mlr[0m=[1;36m0[0m[1;36m.001[0m,
    [33mparamwise_cfg[0m=[1;35mdict[0m[1m([0m
        [33mbias_decay_mult[0m=[1;36m0[0m[1;36m.0[0m,
        [33mcustom_keys[0m=[1;35mdict[0m[1m([0m[1m{[0m[32m'.cls_token'[0m: [1;35mdict[0m[1m([0m[33mdecay_mult[0m=[1;36m0[0m[1;36m.0[0m[1m)[0m[1m}[0m[1m)[0m,
        [33mnorm_decay_mult[0m=[1;36m0[0m[1;36m.0[0m[1m)[0m,
    [33mtype[0m=[32m'AdamW'[0m,
    [33mweight_decay[0m=[1;36m0[0m[1;36m.05[0m[1m)[0m
optimizer_config = [1;35mdict[0m[1m([0m[33mgrad_clip[0m=[3;35mNone[0m[1m)[0m
param_scheduler = [1m[[0m
    [1;35mdict[0m[1m([0m
        [33mbegin[0m=[1;36m0[0m,
        [33mby_epoch[0m=[3;92mTrue[0m,
        [33mconvert_to_iter_based[0m=[3;92mTrue[0m,
        [33mend[0m=[1;36m5[0m,
        [33mstart_factor[0m=[1;36m0[0m[1;36m.001[0m,
        [33mtype[0m=[32m'LinearLR'[0m[1m)[0m,
    [1;35mdict[0m[1m([0m
        [33mT_max[0m=[1;36m295[0m,
        [33mbegin[0m=[1;36m5[0m,
        [33mby_epoch[0m=[3;92mTrue[0m,
        [33mend[0m=[1;36m300[0m,
        [33meta_min[0m=[1;36m1e[0m[1;36m-05[0m,
        [33mtype[0m=[32m'CosineAnnealingLR'[0m[1m)[0m,
[1m][0m
paramwise_cfg = [1;35mdict[0m[1m([0m
    [33mbias_decay_mult[0m=[1;36m0[0m[1;36m.0[0m,
    [33mcustom_keys[0m=[1;35mdict[0m[1m([0m[1m{[0m[32m'.cls_token'[0m: [1;35mdict[0m[1m([0m[33mdecay_mult[0m=[1;36m0[0m[1;36m.0[0m[1m)[0m[1m}[0m[1m)[0m,
    [33mnorm_decay_mult[0m=[1;36m0[0m[1;36m.0[0m[1m)[0m
randomness = [1;35mdict[0m[1m([0m[33mdeterministic[0m=[3;91mFalse[0m, [33mseed[0m=[3;35mNone[0m[1m)[0m
resume = [3;91mFalse[0m
runner = [1;35mdict[0m[1m([0m[33mmax_epochs[0m=[1;36m3[0m, [33mtype[0m=[32m'EpochBasedRunner'[0m[1m)[0m
test_cfg = [1;35mdict[0m[1m([0m[1m)[0m
test_pipeline = [1m[[0m
    [1;35mdict[0m[1m([0m[33mtype[0m=[32m'LoadImageFromFile'[0m[1m)[0m,
    [1;35mdict[0m[1m([0m[33mbackend[0m=[32m'pillow'[0m, [33msize[0m=[1m([0m
        [1;36m100[0m,
        [1;36m-1[0m,
    [1m)[0m, [33mtype[0m=[32m'Resize'[0m[1m)[0m,
    [1;35mdict[0m[1m([0m[33mcrop_size[0m=[1;36m100[0m, [33mtype[0m=[32m'CenterCrop'[0m[1m)[0m,
    [1;35mdict[0m[1m([0m
        [33mmean[0m=[1m[[0m
            [1;36m124.508[0m,
            [1;36m116.05[0m,
            [1;36m106.438[0m,
        [1m][0m,
        [33mstd[0m=[1m[[0m
            [1;36m58.577[0m,
            [1;36m57.31[0m,
            [1;36m57.437[0m,
        [1m][0m,
        [33mto_rgb[0m=[3;91mFalse[0m,
        [33mtype[0m=[32m'Normalize'[0m[1m)[0m,
    [1;35mdict[0m[1m([0m[33mkeys[0m=[1m[[0m
        [32m'img'[0m,
    [1m][0m, [33mtype[0m=[32m'ImageToTensor'[0m[1m)[0m,
    [1;35mdict[0m[1m([0m[33mkeys[0m=[1m[[0m
        [32m'img'[0m,
    [1m][0m, [33mtype[0m=[32m'Collect'[0m[1m)[0m,
[1m][0m
train_cfg = [1;35mdict[0m[1m([0m[33mby_epoch[0m=[3;92mTrue[0m, [33mmax_epochs[0m=[1;36m300[0m, [33mval_interval[0m=[1;36m1[0m[1m)[0m
train_pipeline = [1m[[0m
    [1;35mdict[0m[1m([0m[33mtype[0m=[32m'LoadImageFromFile'[0m[1m)[0m,
    [1;35mdict[0m[1m([0m[33mbackend[0m=[32m'pillow'[0m, [33msize[0m=[1;36m100[0m, [33mtype[0m=[32m'RandomResizedCrop'[0m[1m)[0m,
    [1;35mdict[0m[1m([0m
        [33mmean[0m=[1m[[0m
            [1;36m124.508[0m,
            [1;36m116.05[0m,
            [1;36m106.438[0m,
        [1m][0m,
        [33mstd[0m=[1m[[0m
            [1;36m58.577[0m,
            [1;36m57.31[0m,
            [1;36m57.437[0m,
        [1m][0m,
        [33mto_rgb[0m=[3;91mFalse[0m,
        [33mtype[0m=[32m'Normalize'[0m[1m)[0m,
    [1;35mdict[0m[1m([0m[33mkeys[0m=[1m[[0m
        [32m'img'[0m,
    [1m][0m, [33mtype[0m=[32m'ImageToTensor'[0m[1m)[0m,
    [1;35mdict[0m[1m([0m[33mkeys[0m=[1m[[0m
        [32m'gt_label'[0m,
    [1m][0m, [33mtype[0m=[32m'ToTensor'[0m[1m)[0m,
    [1;35mdict[0m[1m([0m[33mkeys[0m=[1m[[0m
        [32m'img'[0m,
        [32m'gt_label'[0m,
    [1m][0m, [33mtype[0m=[32m'Collect'[0m[1m)[0m,
[1m][0m
val_cfg = [1;35mdict[0m[1m([0m[1m)[0m
vis_backends = [1m[[0m
    [1;35mdict[0m[1m([0m[33mtype[0m=[32m'LocalVisBackend'[0m[1m)[0m,
[1m][0m
visualizer = [1;35mdict[0m[1m([0m
    [33mtype[0m=[32m'UniversalVisualizer'[0m, [33mvis_backends[0m=[1m[[0m
        [1;35mdict[0m[1m([0m[33mtype[0m=[32m'LocalVisBackend'[0m[1m)[0m,
    [1m][0m[1m)[0m
workflow = [1m[[0m
    [1m([0m
        [32m'train'[0m,
        [1;36m1[0m,
    [1m)[0m,
    [1m([0m
        [32m'val'[0m,
        [1;36m1[0m,
    [1m)[0m,
[1m][0m


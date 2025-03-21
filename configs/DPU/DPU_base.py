from Train_video import opt

_base_ = [
    "../_base_/six_gray_sim_data.py",
    # "../_base_/middle_scale.py",
    "../_base_/davis.py",
    "../_base_/default_runtime.py"
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
)

# resize_h, resize_w = 16, 16
# resize_h, resize_w = 32, 32
# resize_h, resize_w = 64, 64
resize_h, resize_w = 128, 128
# resize_h, resize_w = 256, 256
train_pipeline = [
    dict(type='RandomResize'),
    dict(type='RandomCrop', crop_h=resize_h, crop_w=resize_w, random_size=True),
    dict(type='Flip', direction='horizontal', flip_ratio=0.5, ),
    dict(type='Flip', direction='diagonal', flip_ratio=0.5, ),
    dict(type='Resize', resize_h=resize_h, resize_w=resize_w),
]
train_data = dict(
    # mask_path = None,# 随机0，1矩阵
    mask_path="test_datasets/mask/efficientsci_mask.mat",
    # mask_path="test_datasets/mask/random_mask.mat",
    # mask_path="test_datasets/mask/mask.mat",
    # mask_path="test_datasets/mask/mask_DAUHST.mat",

    mask_shape=(resize_h, resize_w, 8),
    # mask_shape=None,
    pipeline=train_pipeline
)
test_data = dict(
    mask_path="test_datasets/mask/efficientsci_mask.mat",
    # mask_path="test_datasets/mask/random_mask.mat",
    # mask_path="test_datasets/mask/mask.mat",
    # mask_path="test_datasets/mask/mask_DAUHST.mat",
    # mask_shape=(256, 256, 8),
    # mask_shape=(resize_h, resize_w, 8),
    # mask_shape=(128, 128, 8),
)

model = dict(
    # type='NetVideo_conv3d',
    # type='NetVideo_conv3d_TSAB',
    # type='NetVideo_conv3d_TSAB_shareBody',
    # type='NetVideo_conv3d_TSAB_shareBody_action',
    # type='NetVideo_conv3d_TSAB_shareBody_action_changeSample',
    # type='NetVideo_conv3d_TSAB_shareBody_action_changeSample_seriesConnection',
    # type='NetVideo_conv3d_TSAB_shareBody_action_changeSample_seriesConnection_withoutDP',
    # type='NetVideo_conv3d_TSAB_shareBody_action_changeSample_seriesConnection_withoutDP_deepcache',
    # type='NetVideo_conv3d_TSAB_shareBody_action_changeSample_seriesConnection_withoutDP_deepcache_keepy',
    # type='NetVideo_conv3d_TSAB_shareBody_action_changeSample_seriesConnection_withoutDP_deepcache_keepy_reuse',
    # type='NetVideo_conv3d_TSAB_shareBody_action_changeSample_seriesConnection_withoutDP_deepcache_keepy_reuse_merge',
    # type='NetVideo_conv3d_TSAB_shareBody_action_changeSample_seriesConnection_withoutDP_deepcache_keepy_reuse_merge_plus',
    # type='NetVideo_conv3d_TSAB_shareBody_action_changeSample_seriesConnection_withoutDP_deepcache_keepy_FAB_TSAB',
    # type='NetVideo_conv3d_TSAB_shareBody_action_changeSample_seriesConnection_withoutDP_deepcache_keepy_withoutT',
    # type='NetVideo_conv3d_TSAB_shareBody_action_changeSample_seriesConnection_withoutDP_deepcache_keepy_lightweight',
    # type='NetVideo_conv3d_TSAB_shareBody_action_changeSample_seriesConnection_withoutDP_deepcache_keepy_lightweight_changeConv3d',
    # type='NetVideo_conv3d_TSAB_shareBody_action_changeSample_seriesConnection_withoutDP_deepcache_cal',
    # type='NetVideo_conv3d_TSAB_shareBody_action_changeSample_seriesConnection_withoutDP_keepy',
    # type='NetVideo_conv3d_TSAB_shareBody_action_changeSample_seriesConnection_withoutDP_keep_y',  # 11177MiB
    # type='NetVideo_conv3d_TSAB_shareBody_action_changeSample_seriesConnection_withoutDP_keep_y_newdeepcache',
    # type='NetVideo_conv3d_TSAB_shareBody_action_changeSample_seriesConnection_withoutDP_keep_y_reduceGPU',
    # type='NetVideo_conv3d_TSAB_shareBody_action_changeSample_seriesConnection_withoutDP_keep_y_ffn',
    # type='NetVideo_conv3d_TSAB_shareBody_action_changeSample_seriesConnection_withoutDP_calssim',
    # type='NetVideo_conv3d_TSAB_shareBody_action_changeSample_seriesConnection_withoutDP_MA',
    # type='NetVideo_conv3d_TSAB_shareBody_action_changeSample_seriesConnection_withoutDP_changeIPB',
    # type='NetVideo_conv3d_TSAB_shareBody_action_changeSample_resFemVrm',
    # type='NetVideo_conv3d_TSAB_shareBody_deepcache',
    # type='NetVideo_conv3d_TSAB_shareBody_action_deepcache',

    # type='NetVideo_conv3d_FTSAB_shareBody_action_changeInput',
    # type='NetVideo_conv3d_FTSAB',
    # type='NetVideo_conv3d_SCB',
    # type='NetVideo',


    # type='NetVideo_base',
    # type='NetVideo_base_noStageInteraction',
    # type='NetVideo_base_noStageInteraction_normalunfolding',
    # type='NetVideo_base_noStageInteraction_normalunfolding_1',
    # type='NetVideo_base_noStageInteraction_deepcache',
    # type='NetVideo_base_noStageInteraction_deepcache_action',
    # type='NetVideo_base_noStageInteraction_deepcache_ffnlw',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new',


    # type='NetVideo_base_reuse',
    # type='NetVideo_base_reuse_1',
    # type='NetVideo_base_reuse_2',

    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_1',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_2',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_3',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_4',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_5',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_6',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_7',
    #

    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_2',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_4',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_5',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_3',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_6',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalusnfolding_cross_8',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cros s_9',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_11',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_12',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_13',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_14',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_15',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_16',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_17',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_18',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_19',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_20',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_21',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_22',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_23',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_24',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_25',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_26',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_27',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_28',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_29',
    type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_29_noin',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_29_test',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_30',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_31',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_32',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_33',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_34',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_35',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_36',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_37',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_38',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_39',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_40',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_40_test',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_41',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_42',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_43',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_44',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_45',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_46',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_47',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_48',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_49',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_50',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_51',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_52',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_53',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_54',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_55',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_56',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_qkv',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_cross_dw',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_ema',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_ema_both',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_ema_both_C2RV',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_ema_both_C2RV_divide',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_ema_both_DBCA',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_ema_both_C2RV_position',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_ema_spatial',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_fourier',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace_new_normalunfolding_stconv',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw_single_multireuse_multimain_replace',
    # type='NetVideo_base_noStageInteraction_deepcache_action_ffnlw',
    # type='NetVideo_base_pixelattention',
    # type='NetVideo_base_Tspatialattention',
    # type='NetVideo_base_lngn',
    # type='NetVideo_base_Tposition',
    # type='NetVideo_base_Tspatialattention',
    # type='NetVideo_base_poolattention',
    # type='NetVideo_base_ffnchange',
    # type='NetVideo_base_ffnchange2',
    # type='NetVideo_base_Tenhance',
    # type='NetVideo_base_removeDS',
    # type='NetVideo_base_resnet',
    # type='NetVideo_base_resnet_GDFN',
    # type='NetVideo_base_resnet_ffn',
    # type='NetVideo_base_resnet_repffn',
    # type='NetVideo_base_resnet_testFAB_TSAB',
    # type='NetVideo_FA',
    opt=opt
    # opt=arg
)

eval = dict(
    flag=True,
    interval=1
)
#
# checkpoints="/home/yychen/zhangmuyuan/DPU/work_dirs/DPU_base/checkpoints/NetVideo_base_noStageInteraction_normalunfolding/epoch_93.pth"
# checkpoints = None

# Explanation of generator_args is in sam/segment_anything/automatic_mask_generator.py: SamAutomaticMaskGenerator
sam_args = {
    'sam_checkpoint': "ckpt/sam_vit_b_01ec64.pth",
    'model_type': "vit_b",
    'generator_args':{
        'points_per_side': 16,
        'pred_iou_thresh': 0.8,
        'stability_score_thresh': 0.9,
        'crop_n_layers': 1,
        'crop_n_points_downscale_factor': 2,
        'min_mask_region_area': 200,
    },
    'gpu_id': 0,
}
aot_args = {
    'phase': 'PRE_YTB_DAV',
    'model': 'r50_deaotl',
    'model_path': 'ckpt/R50_DeAOTL_PRE_YTB_DAV.pth',
    'long_term_mem_gap': 9999,
    'max_len_long_term': 9999,
    'gpu_id': 0,
}
segtracker_args = {
    'sam_gap': 10, # the interval to run sam to segment new objects
    'min_area': 200, # minimal mask area to add a new mask as a new object
    'max_obj_num': 255, # maximal object number to track in a video
    'min_new_obj_iou': 0.8, # the background area ratio of a new object should > 80% 
}

# Map supported SAM checkpoints to their model_type
SAM_CKPT_TO_TYPE = {
    "ckpt/sam_vit_h_4b8939.pth": "vit_h",
    "ckpt/sam_vit_l_0b3195.pth": "vit_l",
    "ckpt/sam_vit_b_01ec64.pth": "vit_b",
}

# Map supported GroundingDINO checkpoints to config files
GD_CKPT_TO_CONFIG = {
    "ckpt/groundingdino_swint_ogc.pth": "config/GroundingDINO_SwinT_OGC.py",
    # For SwinB, place this file under ./config or rely on the package fallback (see Detector below)
    "ckpt/groundingdino_swinb_cogcoor.pth": "config/GroundingDINO_SwinB_cfg.py",
}

# Default GroundingDINO args; the app will override these at runtime
gd_args = {
    'ckpt_path': "ckpt/groundingdino_swint_ogc.pth",
    'config_file': GD_CKPT_TO_CONFIG["ckpt/groundingdino_swint_ogc.pth"],
    'gpu_id': 0,
}
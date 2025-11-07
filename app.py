import os
import gradio as gr
from model_args import segtracker_args as default_segtracker_args, sam_args as default_sam_args, aot_args as default_aot_args, gd_args as default_gd_args, SAM_CKPT_TO_TYPE, GD_CKPT_TO_CONFIG
import copy
import tempfile
from SegTracker import SegTracker
from tool.transfer_tools import draw_outline, draw_points
import cv2
from PIL import Image
import torch
import math
from seg_track_anything import aot_model2ckpt, tracking_objects_in_video, draw_mask, TRACKING_RESULTS_DIR
import gc
import numpy as np
from tool.transfer_tools import mask2bbox
import zipfile
import shutil

# Common image filename extensions accepted by the app.
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')

def clean():
    """Reset UI state for a new input."""
    # Ensure all components are reset, including the caption
    # Outputs map to: input_video, input_img_seq, Seg_Tracker, input_first_frame, origin_frame, drawing_board, click_stack, grounding_caption
    return None, None, None, None, None, None, [[], []], ""

def _safe_extract_images(zip_path: str, dest_dir: str) -> None:
    """
    Safely extract only image files from a zip into dest_dir.
    Prevents Zip Slip by validating real paths.
    """
    if not os.path.exists(zip_path):
        return
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for m in zf.infolist():
                # Skip directories and non-images early
                name = m.filename
                if name.endswith("/") or not name.lower().endswith(IMAGE_EXTS):
                    continue
                
                # Resolve final path and ensure it's inside dest_dir
                target_path = os.path.join(dest_dir, name)
                real_target = os.path.realpath(target_path)
                real_base = os.path.realpath(dest_dir)
                
                # Use os.path.commonprefix or startswith to ensure path is safe
                if os.path.commonprefix([real_target, real_base]) != real_base:
                    # Malicious path; skip
                    continue
                
                # Ensure the directory structure exists before extraction
                os.makedirs(os.path.dirname(real_target), exist_ok=True)
                
                # Extract the file
                with zf.open(m) as src, open(real_target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    except zipfile.BadZipFile:
        print(f"Error: Invalid or corrupt ZIP file: {zip_path}")
    except Exception as e:
        print(f"An error occurred during extraction: {e}")

                
def _infer_sam_model_type(ckpt_path: str) -> str:
    import os
    base = os.path.basename(ckpt_path)
    for known, mtype in SAM_CKPT_TO_TYPE.items():
        if base == os.path.basename(known):
            return mtype
    # Heuristic fallback
    if "vit_h" in base: return "vit_h"
    if "vit_l" in base: return "vit_l"
    return "vit_b"

def _gd_config_for_ckpt(ckpt_path: str) -> str:
    import os
    base = os.path.basename(ckpt_path)
    for known, cfg in GD_CKPT_TO_CONFIG.items():
        if base == os.path.basename(known):
            return cfg
    # Default to SwinT OGC
    return "config/GroundingDINO_SwinT_OGC.py"

def get_click_prompt(click_stack, point):

    click_stack[0].append(point["coord"])
    click_stack[1].append(point["mode"]
    )
    
    prompt = {
        "points_coord": click_stack[0],
        "points_mode":  click_stack[1],
        "multimask":    True,  # bool, not string
    }

    return prompt

def get_meta_from_video(input_video):
    if input_video is None:
        return None, None, None, ""

    print("get meta information of input video")
    cap = cv2.VideoCapture(input_video)
    
    try:
        ok, first_frame = cap.read()
    finally:
        cap.release()
    if not ok or first_frame is None:
        return None, None, None, ""
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    # Returns: visualization, origin_frame_state, drawing_board_reset, caption_reset
    return first_frame, first_frame, first_frame, ""

def get_meta_from_img_seq(input_img_seq):
    if input_img_seq is None or not hasattr(input_img_seq, 'name'):
        return None, None, None, ""

    os.makedirs(ASSETS_DIR, exist_ok=True)
    print("get meta information of img seq")
    # Prepare extraction dir
    try:
        file_name = os.path.splitext(os.path.basename(input_img_seq.name))[0]
    except Exception:
         return None, None, None, ""

    file_path = os.path.join(ASSETS_DIR, file_name)
    if os.path.isdir(file_path):
        shutil.rmtree(file_path)
    os.makedirs(file_path, exist_ok=True)
    # Safely extract only images (prevents Zip Slip)
    _safe_extract_images(input_img_seq.name, file_path)

    imgs_path = sorted(
        os.path.join(file_path, n)
        for n in os.listdir(file_path)
        if n.lower().endswith(IMAGE_EXTS)
    )
    if not imgs_path:
        return None, None, None, ""

    first_frame = cv2.imread(imgs_path[0])
    if first_frame is None:
         return None, None, None, ""
         
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    # Returns: visualization, origin_frame_state, drawing_board_reset, caption_reset
    return first_frame, first_frame, first_frame, ""

def SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask):
    # Use autocast on CUDA if available for speed, otherwise no-op.
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Enable autocast only on CUDA to avoid CPU bfloat16 surprises
    with torch.amp.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
        # Reset the first frame's mask
        frame_idx = 0
        Seg_Tracker.restart_tracker()
        Seg_Tracker.add_reference(origin_frame, predicted_mask, frame_idx)
        Seg_Tracker.first_frame_mask = predicted_mask

    return Seg_Tracker

def _prepare_args(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, sam_ckpt, gd_ckpt):
    """Prepare argument dictionaries for SegTracker initialization by copying defaults and applying overrides."""
    # Use deepcopy to ensure nested dictionaries (like generator_args) are independent and global defaults are not mutated.
    segtracker_args = copy.deepcopy(default_segtracker_args)
    sam_args = copy.deepcopy(default_sam_args)
    aot_args = copy.deepcopy(default_aot_args)
    gd_args = copy.deepcopy(default_gd_args)

    # Apply AOT args overrides
    if aot_model:
        aot_args["model"] = aot_model
        # Use .get() for safer access in case the model name is invalid
        aot_args["model_path"] = aot_model2ckpt.get(aot_model, aot_args["model_path"])

    aot_args["long_term_mem_gap"] = long_term_mem
    aot_args["max_len_long_term"] = max_len_long_term
    
    # Apply SegTracker args overrides
    segtracker_args["sam_gap"] = sam_gap
    segtracker_args["max_obj_num"] = max_obj_num
    
    # Apply SAM args overrides
    sam_args["generator_args"]["points_per_side"] = points_per_side
    if sam_ckpt is not None:
        sam_args["sam_checkpoint"] = sam_ckpt
        sam_args["model_type"] = _infer_sam_model_type(sam_ckpt)
    
    # Apply GroundingDINO args overrides
    _gd_cfg = _gd_config_for_ckpt(gd_ckpt) if gd_ckpt is not None else gd_args["config_file"]
    gd_args["ckpt_path"] = gd_ckpt or gd_args["ckpt_path"]
    gd_args["config_file"] = _gd_cfg

    return segtracker_args, sam_args, aot_args, gd_args

def init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, sam_ckpt, gd_ckpt, origin_frame):
    
    # Initializes the tracker and returns the core state: [Seg_Tracker, input_first_frame_vis, click_stack]

    if origin_frame is None:
        return None, None, [[], []]

    segtracker_args, sam_args, aot_args, gd_args = _prepare_args(
        aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, sam_ckpt, gd_ckpt
    )
    # Initialize SegTracker
    try:
        Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args, gd_args)
        Seg_Tracker.restart_tracker()
    except Exception as e:
        # Handle initialization failure gracefully (e.g., missing checkpoints, CUDA errors)
        print(f"Error initializing SegTracker: {e}")
        # Return origin_frame to keep visualization, but no tracker.
        return None, origin_frame, [[], []]
        
    return Seg_Tracker, origin_frame, [[], []]

def init_for_stroke_tab(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, sam_ckpt, gd_ckpt, origin_frame):
    tracker, frame_vis, stack = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, sam_ckpt, gd_ckpt, origin_frame)
    # Returns: [Tracker, FrameVis, Stack, DrawingBoard (reset to frame)]
    return tracker, frame_vis, stack, frame_vis

def init_for_text_tab(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, sam_ckpt, gd_ckpt, origin_frame):
    tracker, frame_vis, stack = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, sam_ckpt, gd_ckpt, origin_frame)
    # Returns: [Tracker, FrameVis, Stack, Caption (cleared)]
    return tracker, frame_vis, stack, ""

def reset_app_state(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, sam_ckpt, gd_ckpt, origin_frame):
    # Used by the Reset button for a full reset.
    tracker, frame_vis, stack = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, sam_ckpt, gd_ckpt, origin_frame)    # Returns: [Tracker, FrameVis, Stack, Caption (cleared), DrawingBoard (reset to frame)]
    return tracker, frame_vis, stack, "", frame_vis


# Simplified handlers by removing unused configuration arguments.

def undo_click_stack_and_refine_seg(Seg_Tracker, origin_frame, click_stack):    
    if Seg_Tracker is None:
        # If tracker is missing, we can't undo, just reset the view.
        return None, origin_frame, [[], []]

    print("Undo!")
    # Remove the last click
    if len(click_stack[0]) > 0:
        click_stack[0] = click_stack[0][: -1]
        click_stack[1] = click_stack[1][: -1]
    
    if len(click_stack[0]) > 0:
        prompt = {
            "points_coord":click_stack[0],
            "points_mode":click_stack[1],
            "multimask": True,
        }

        masked_frame = seg_acc_click(Seg_Tracker, prompt, origin_frame)
        return Seg_Tracker, masked_frame, click_stack
    else:
        # If no clicks remain, revert the visualization to the state before this object was started.
        # This relies on Seg_Tracker.origin_merged_mask holding the previous state.
        if Seg_Tracker.origin_merged_mask is not None:
            masked_frame = draw_mask(origin_frame.copy(), Seg_Tracker.origin_merged_mask)
            # Update the current working mask (first_frame_mask) back to the origin state
            Seg_Tracker.first_frame_mask = Seg_Tracker.origin_merged_mask.copy()
        else:
            masked_frame = origin_frame
            Seg_Tracker.first_frame_mask = None

        return Seg_Tracker, masked_frame, [[], []]

def roll_back_undo_click_stack_and_refine_seg(Seg_Tracker, origin_frame, click_stack, input_video, input_img_seq, frame_num, refine_idx):
    if Seg_Tracker is None:
        return None, origin_frame, [[], []]

    print("Undo!")
    if len(click_stack[0]) > 0:
        click_stack[0] = click_stack[0][: -1]
        click_stack[1] = click_stack[1][: -1]
    
    if len(click_stack[0]) > 0:
        prompt = {
            "points_coord":click_stack[0],
            "points_mode":click_stack[1],
            "multimask": True,
        }

        # We need the current overall mask state for this frame to merge the results
        _, curr_mask, _ = res_by_num(input_video, input_img_seq, frame_num)
        if curr_mask is None:
            print("Error: Failed to load mask for rollback undo.")
            return Seg_Tracker, origin_frame, click_stack
        Seg_Tracker.curr_idx = refine_idx
        predicted_mask, _ = Seg_Tracker.seg_acc_click( 
                                                    origin_frame=origin_frame, 
                                                    coords=np.array(prompt["points_coord"]),
                                                    modes=np.array(prompt["points_mode"]),
                                                    multimask=prompt["multimask"],
                                                    )
        # Ensure we don't clear the mask if the index is invalid (e.g. background)
        if refine_idx is not None and refine_idx != 0:
             curr_mask[curr_mask == refine_idx]  = 0
             
        curr_mask[predicted_mask != 0]  = refine_idx
        merged_mask = curr_mask
        
        # Update the tracker state with the new merged mask
        Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, merged_mask)
        # Redraw the frame visualization based on the final merged mask
        masked_frame = draw_mask(origin_frame.copy(), merged_mask)

        return Seg_Tracker, masked_frame, click_stack
    else:
        # No clicks left, remove the object being refined from the mask
        _, curr_mask, _ = res_by_num(input_video, input_img_seq, frame_num)
        if curr_mask is not None:
            if refine_idx is not None and refine_idx != 0:
                curr_mask[curr_mask == refine_idx] = 0
            merged_mask = curr_mask
            Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, merged_mask)
            masked_frame = draw_mask(origin_frame.copy(), merged_mask)
            return Seg_Tracker, masked_frame, [[], []]
        
        # Fallback if mask loading failed
        return Seg_Tracker, origin_frame, [[], []]

def seg_acc_click(Seg_Tracker, prompt, origin_frame):
    # seg acc to click
    predicted_mask, masked_frame = Seg_Tracker.seg_acc_click( 
                                                        origin_frame=origin_frame, 
                                                        coords=np.array(prompt["points_coord"]),
                                                        modes=np.array(prompt["points_mode"]),
                                                        multimask=prompt["multimask"],
                                                    )

    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)

    return masked_frame

def _ensure_tracker(Seg_Tracker, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, sam_ckpt, gd_ckpt, origin_frame):
    """Helper to ensure Seg_Tracker is initialized, initializing it if necessary."""
    if Seg_Tracker is not None:
        return Seg_Tracker
        
    if origin_frame is None:
        print("Warning: Cannot initialize tracker without an input frame.")
        return None

    print("Initializing SegTracker on demand...")
    # Initialize using the standard initialization function
    # We only care about the tracker instance itself here.
    tracker, _, _ = init_SegTracker(
        aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num,
        points_per_side, sam_ckpt, gd_ckpt, origin_frame
    )
    return tracker

def sam_click(Seg_Tracker, origin_frame, point_mode, click_stack, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, sam_ckpt, gd_ckpt, evt:gr.SelectData):
    """
    Args:
        origin_frame: nd.array
        click_stack: [[coordinate], [point_mode]]
    """

    print("Click")

    # FIX 2: Safety check for evt is None
    if evt is None or evt.index is None:
        print("Warning: Click event data (evt) or index is missing.")
        # Return current state. We use origin_frame as a fallback visualization.
        return Seg_Tracker, origin_frame, click_stack

    if point_mode == "Positive":
        point = {"coord": [evt.index[0], evt.index[1]], "mode": 1}
    else:
        point = {"coord": [evt.index[0], evt.index[1]], "mode": 0}

    Seg_Tracker = _ensure_tracker(
        aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num,
        points_per_side, sam_ckpt, gd_ckpt, origin_frame
    )
    if Seg_Tracker is None:
        return None, origin_frame, click_stack

    # get click prompts for sam to predict mask
    click_prompt = get_click_prompt(click_stack, point)

    # Refine acc to prompt
    masked_frame = seg_acc_click(Seg_Tracker, click_prompt, origin_frame)

    return Seg_Tracker, masked_frame, click_stack

# FIX 2: Removed default values (=None) for sam_ckpt, gd_ckpt, and evt.
def roll_back_sam_click(Seg_Tracker, origin_frame, point_mode, click_stack, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, sam_ckpt, gd_ckpt, input_video, input_img_seq, frame_num, refine_idx, evt:gr.SelectData):
    """
    Args:
        origin_frame: nd.array
        click_stack: [[coordinate], [point_mode]]
    """

    print("Click")

    # FIX 2: Safety check for evt is None
    if evt is None or evt.index is None:
        print("Warning: Click event data (evt) or index is missing during rollback.")
        # Return current state. We use origin_frame as a fallback visualization.
        return Seg_Tracker, origin_frame, click_stack

    if point_mode == "Positive":
        point = {"coord": [evt.index[0], evt.index[1]], "mode": 1}
    else:
        point = {"coord": [evt.index[0], evt.index[1]], "mode": 0}
        
    # In rollback mode, Seg_Tracker should ideally be present, but we ensure it exists.
    Seg_Tracker = _ensure_tracker(
        Seg_Tracker, aot_model, long_term_mem, max_len_long_term,
        sam_gap, max_obj_num, points_per_side, sam_ckpt, gd_ckpt, origin_frame
    )
    if Seg_Tracker is None:
        return None, origin_frame, click_stack

    # get click prompts for sam to predict mask
    prompt = get_click_prompt(click_stack, point)

    # We need the current overall mask state for this frame to merge the results
    _, curr_mask, _ = res_by_num(input_video, input_img_seq, frame_num)
    if curr_mask is None:
        print("Error: Failed to load mask for rollback click.")
        return Seg_Tracker, origin_frame, click_stack
    Seg_Tracker.curr_idx = refine_idx
    
    # Get the refined mask for the specific object
    predicted_mask, _ = Seg_Tracker.seg_acc_click(
                                                    origin_frame=origin_frame, 
                                                    coords=np.array(prompt["points_coord"]),
                                                    modes=np.array(prompt["points_mode"]),
                                                    multimask=prompt["multimask"],
                                                )
    # Ensure we don't try to clear the mask if the index is invalid (e.g. background)
    if refine_idx is not None and refine_idx != 0:
        curr_mask[curr_mask == refine_idx]  = 0

    curr_mask[predicted_mask != 0]  = refine_idx
    merged_mask = curr_mask

    # Update the tracker state with the new merged mask
    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, merged_mask)
    
    # Redraw the frame visualization based on the final merged mask
    masked_frame = draw_mask(origin_frame.copy(), merged_mask)

    return Seg_Tracker, masked_frame, click_stack

# FIX 1: Update sam_stroke to handle ImageEditor input and prevent KeyError.
def sam_stroke(Seg_Tracker, origin_frame, drawing_board, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, sam_ckpt, gd_ckpt):

    Seg_Tracker = _ensure_tracker(
        Seg_Tracker, aot_model, long_term_mem, max_len_long_term,
        sam_gap, max_obj_num, points_per_side, sam_ckpt, gd_ckpt, origin_frame
    )
    if Seg_Tracker is None:
        # If initialization failed, return current state.
        return None, origin_frame, drawing_board

    print("Stroke")
    
    # Handle different input formats from Gradio components (ImageEditor vs older Sketch)
    mask = None
    if isinstance(drawing_board, dict):
        # Try extracting mask from modern gr.ImageEditor (layers)
        if "layers" in drawing_board and drawing_board["layers"]:
            # Initialize mask based on origin frame dimensions
            h, w = origin_frame.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for layer in drawing_board["layers"]:
                # layer might be a PIL Image or numpy array depending on Gradio version/config
                if isinstance(layer, Image.Image):
                    layer = np.array(layer)
                
                if not isinstance(layer, np.ndarray):
                    continue

                # Ensure layer dimensions match mask dimensions (Gradio might resize input)
                if layer.shape[0] != h or layer.shape[1] != w:
                     # Resize layer to match frame dimensions if necessary
                     try:
                        layer = cv2.resize(layer, (w, h), interpolation=cv2.INTER_NEAREST)
                     except cv2.error:
                        continue

                if layer.ndim == 3 and layer.shape[2] == 4: # RGBA
                    # Use alpha channel for the stroke mask
                    stroke_mask = (layer[:, :, 3] > 0).astype(np.uint8)
                    mask = np.maximum(mask, stroke_mask)

        # Fallback for older behavior if "mask" key is present (e.g., gr.Image(tool="sketch"))
        elif "mask" in drawing_board and drawing_board["mask"] is not None:
            print("Using fallback 'mask' key.")
            mask_input = drawing_board["mask"]
            if isinstance(mask_input, Image.Image):
                mask_input = np.array(mask_input)

            if isinstance(mask_input, np.ndarray):
                # Ensure mask is 2D binary
                if mask_input.ndim == 3:
                    # Take the first channel if it's 3D (assuming grayscale mask)
                    mask = (mask_input[:, :, 0] > 0).astype(np.uint8)
                else:
                    mask = (mask_input > 0).astype(np.uint8)

    if mask is None or np.sum(mask) == 0:
        print("No stroke detected or unable to extract mask from input.")
        # Return current state without changes if no mask is found
        return Seg_Tracker, origin_frame, drawing_board

    bbox = mask2bbox(mask)  # bbox: [[x0, y0], [x1, y1]]
    
    # Check if bbox is valid (mask2bbox returns [[0,0],[0,0]] if mask is empty)
    if np.array_equal(bbox, np.array([[0, 0], [0, 0]], dtype=np.int64)):
         print("Could not generate a valid bbox from stroke.")
         return Seg_Tracker, origin_frame, drawing_board

    predicted_mask, masked_frame = Seg_Tracker.seg_acc_bbox(origin_frame, bbox)

    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)

    # Return the segmented visualization and reset the drawing board to the original frame
    return Seg_Tracker, masked_frame, origin_frame

def gd_detect(Seg_Tracker, origin_frame, grounding_caption, box_threshold, text_threshold, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, sam_ckpt, gd_ckpt):
    Seg_Tracker = _ensure_tracker(
        Seg_Tracker, aot_model, long_term_mem, max_len_long_term,
        sam_gap, max_obj_num, points_per_side, sam_ckpt, gd_ckpt, origin_frame
    )
    if Seg_Tracker is None:
        # Adjusted return to match expected outputs in the listener
        return None, origin_frame

    print("Detect")
    predicted_mask, annotated_frame= Seg_Tracker.detect_and_seg(origin_frame, grounding_caption, box_threshold, text_threshold)

    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)


    masked_frame = draw_mask(annotated_frame, predicted_mask)

    # Adjusted return to match expected outputs in the listener
    return Seg_Tracker, masked_frame

def segment_everything(Seg_Tracker, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, sam_ckpt, gd_ckpt, origin_frame):
    Seg_Tracker = _ensure_tracker(
        Seg_Tracker, aot_model, long_term_mem, max_len_long_term,
        sam_gap, max_obj_num, points_per_side, sam_ckpt, gd_ckpt, origin_frame
    )
    if Seg_Tracker is None:
        return None, origin_frame

    print("Everything")

    frame_idx = 0

    # Use appropriate device type for autocast
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.amp.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
        pred_mask = Seg_Tracker.seg(origin_frame)
        
        if device_type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        if pred_mask is not None:
            Seg_Tracker.add_reference(origin_frame, pred_mask, frame_idx)
            Seg_Tracker.first_frame_mask = pred_mask
            masked_frame = draw_mask(origin_frame.copy(), pred_mask)
        else:
            # Handle case where segmentation fails or returns nothing
            masked_frame = origin_frame


    return Seg_Tracker, masked_frame

def add_new_object(Seg_Tracker):
    if Seg_Tracker is None:
        print("Warning: SegTracker not initialized.")
        return None, [[], []]

    if Seg_Tracker.first_frame_mask is not None:
        prev_mask = Seg_Tracker.first_frame_mask
        Seg_Tracker.update_origin_merged_mask(prev_mask)    
    
    Seg_Tracker.curr_idx += 1

    print("Ready to add new object!")

    return Seg_Tracker, [[], []]

def tracking_objects(Seg_Tracker, input_video, input_img_seq, fps, frame_num=None): 
    if Seg_Tracker is None:
        print("Error: SegTracker not initialized before tracking.")
        return None, None
           
    print("Start tracking !")
    start_frame = int(frame_num) if frame_num is not None else 0
    # Start actual tracking pipeline
    return tracking_objects_in_video(Seg_Tracker, input_video, input_img_seq, fps, start_frame)

# Robust helper function for rollback/refinement
def res_by_num(input_video, input_img_seq, frame_num):
    # Ensure frame_num is a valid integer
    try:
        frame_num = int(frame_num)
    except (TypeError, ValueError):
        return None, None, None

    # 1. Load the original frame from the source (Video or Image Sequence)
    video_name = None
    ori_frame = None

    if input_video is not None:
        video_name = os.path.basename(input_video).split('.')[0]

        cap = cv2.VideoCapture(input_video)
        try:
            # Optimization: Seek directly to the frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ok, ori_frame = cap.read()
        finally:
            cap.release()
            
        if not ok or ori_frame is None:
            return None, None, None
        ori_frame = cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB)
        
    elif input_img_seq is not None:
        try:
            file_name = os.path.splitext(os.path.basename(input_img_seq.name))[0]
        except AttributeError:
             return None, None, None

        file_path = os.path.join(ASSETS_DIR, file_name)
        video_name = file_name
        
        if not os.path.isdir(file_path):
             return None, None, None

        imgs_path = sorted(
            os.path.join(file_path, n)
            for n in os.listdir(file_path)
            if n.lower().endswith(IMAGE_EXTS)
        )
        if frame_num >= len(imgs_path) or frame_num < 0:
            return None, None, None
        ori_frame = cv2.imread(imgs_path[frame_num])
        if ori_frame is None:
             return None, None, None
        ori_frame = cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB)
    
    # 2. Load the corresponding results (Masked Frame and Mask)
    if video_name is None:
        return None, None, None

    tracking_result_dir = os.path.join(TRACKING_RESULTS_DIR, video_name)
    output_masked_frame_dir = f'{tracking_result_dir}/{video_name}_masked_frames'
    output_mask_dir = f'{tracking_result_dir}/{video_name}_masks'

    if not os.path.isdir(output_masked_frame_dir) or not os.path.isdir(output_mask_dir):
        return None, None, None

    output_masked_frame_path = sorted(
        os.path.join(output_masked_frame_dir, n)
        for n in os.listdir(output_masked_frame_dir)
        if n.lower().endswith(IMAGE_EXTS)
    )
    output_mask_path = sorted(
        os.path.join(output_mask_dir, n)
        for n in os.listdir(output_mask_dir)
        if n.lower().endswith(IMAGE_EXTS)
    )

    if len(output_masked_frame_path) == 0 or len(output_mask_path) == 0:
        return None, None, None
    else:
        if frame_num >= len(output_masked_frame_path) or frame_num >= len(output_mask_path):
            print("num out of frames range")
            return None, None, None
            
        print("choose", frame_num, "to refine")
        chosen_frame_show = cv2.imread(output_masked_frame_path[frame_num])
        if chosen_frame_show is None:
             return None, None, None
             
        chosen_frame_show = cv2.cvtColor(chosen_frame_show, cv2.COLOR_BGR2RGB)
        # Mask saved as palette image; preserve ids by using PIL 'P' mode.
        try:
            chosen_mask = np.array(Image.open(output_mask_path[frame_num]).convert('P'))
        except IOError:
             return None, None, None

        return chosen_frame_show, chosen_mask, ori_frame

def show_res_by_slider(input_video, input_img_seq, frame_per):
    video_name = None
    if input_video is not None:
        video_name = os.path.basename(input_video).split('.')[0]
    elif input_img_seq is not None:
        try:
            # Handle potential path variations in Gradio file input name
            file_name = os.path.splitext(os.path.basename(input_img_seq.name))[0]
            video_name = file_name
        except AttributeError:
             return None, None
    
    if video_name is None:
        return None, None

    tracking_result_dir = os.path.join(TRACKING_RESULTS_DIR, video_name)
    output_masked_frame_dir = f'{tracking_result_dir}/{video_name}_masked_frames'
    
    if not os.path.isdir(output_masked_frame_dir):
        return None, None

    # List files robustly
    output_masked_frame_path = sorted([
        os.path.join(output_masked_frame_dir, img_name) 
        for img_name in os.listdir(output_masked_frame_dir)
        if img_name.lower().endswith(IMAGE_EXTS)
        ])
        
    total_frames_num = len(output_masked_frame_path)
    if total_frames_num == 0:
        return None, None
    else:
        frame_num = math.floor(total_frames_num * frame_per / 100)
        # Clamp the frame number to the last index if it reaches the end
        if frame_num >= total_frames_num and total_frames_num > 0:
            frame_num = total_frames_num - 1
        if frame_num < 0:
            frame_num = 0
            
        chosen_frame_show, _, _ = res_by_num(input_video, input_img_seq, frame_num)
        return chosen_frame_show, frame_num

# FIX 2: Removed default value for evt.
def choose_obj_to_refine(input_video, input_img_seq, Seg_Tracker, frame_num, evt:gr.SelectData):
    # FIX 2: Safety check for evt is None
    if evt is None or evt.index is None:
        print("Warning: Click event data (evt) or index is missing during object selection for refinement.")
        chosen_frame_show, _, _ = res_by_num(input_video, input_img_seq, frame_num)
        return chosen_frame_show, None

    chosen_frame_show, curr_mask, _ = res_by_num(input_video, input_img_seq, frame_num)
    
    if curr_mask is not None and chosen_frame_show is not None:
        # Ensure click coordinates are within bounds
        y, x = evt.index[1], evt.index[0]
        if 0 <= y < curr_mask.shape[0] and 0 <= x < curr_mask.shape[1]:
            idx = curr_mask[y, x]
            
            # Highlight the selected area (even if background)
            curr_idx_mask = (curr_mask == idx).astype(np.uint8)
            chosen_frame_show = draw_points(points=np.array([[x, y]]), modes=np.array([[1]]), frame=chosen_frame_show)
            chosen_frame_show = draw_outline(mask=curr_idx_mask, frame=chosen_frame_show)
            print(f"Selected Object ID: {idx}")
            return chosen_frame_show, idx
        else:
            print("Click out of bounds.")
            return chosen_frame_show, None
    
    return chosen_frame_show, None

def show_chosen_idx_to_refine(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, sam_ckpt, gd_ckpt, input_video, input_img_seq, Seg_Tracker, frame_num, idx):
    # Check if a valid object was selected (idx is not None)
    if idx is None:
        # Return existing state without changes, ensure caption is clear.
        _, _, ori_frame = res_by_num(input_video, input_img_seq, frame_num)
        return ori_frame, Seg_Tracker, ori_frame, [[], []], ""

    chosen_frame_show, curr_mask, ori_frame = res_by_num(input_video, input_img_seq, frame_num)
    
    if ori_frame is None:
        return None, Seg_Tracker, None, [[], []], ""

    # Ensure tracker is initialized before refinement.
    Seg_Tracker = _ensure_tracker(Seg_Tracker, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, sam_ckpt, gd_ckpt, ori_frame)

    # Reset the tracker state completely to prepare for refinement from this frame
    if Seg_Tracker:
        Seg_Tracker.reset_state()
        # Initialize tracker with the current state of the frame for accurate refinement context
        if curr_mask is not None:
            Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, ori_frame, curr_mask)
            Seg_Tracker.curr_idx = idx # Set the index we are refining
        
    # Return state for refine_res visualization, Seg_Tracker, origin_frame state, click_stack, grounding_caption
    return ori_frame, Seg_Tracker, ori_frame, [[], []], ""

def seg_track_app():
    ##########################################################
    ######################  Front-end ########################
    ##########################################################
    app = gr.Blocks()

    with app:
        gr.Markdown(
            '''
            <div style="text-align:center;">
                <span style="font-size:3em; font-weight:bold;">Segment and Track Anything(SAM-Track)</span>
            </div>
            '''
        )

        # State variables
        click_stack = gr.State([[],[]]) # Storage clicks status
        origin_frame = gr.State(None)
        Seg_Tracker = gr.State(None)

        refine_idx = gr.State(None)
        frame_num = gr.State(None)

        # Removed redundant gr.State definitions for configuration parameters

        with gr.Row():
            # video input
            with gr.Column(scale=2):

                tab_video_input = gr.Tab(label="Video type input")
                with tab_video_input:
                    input_video = gr.Video(label='Input video', height=550)
                
                tab_img_seq_input = gr.Tab(label="Image-Seq type input")
                with tab_img_seq_input:
                    with gr.Row():
                        input_img_seq = gr.File(label='Input Image-Seq (ZIP)', file_types=['.zip'], height=550)
                        with gr.Column(scale=4):
                            extract_button = gr.Button(value="extract")
                            fps = gr.Slider(label='fps', minimum=5, maximum=50, value=8, step=1)

                # Set interactive=True and type="numpy" for visualization and click capture
                input_first_frame = gr.Image(label='Segment result of first frame',interactive=True, height=550, type="numpy")

                tab_everything = gr.Tab(label="Everything")
                with tab_everything:
                    with gr.Row():
                        seg_every_first_frame = gr.Button(value="Segment everything for first frame", interactive=True)
                        # Note: point_mode is defined multiple times. The last definition (in tab_click) is the one referenced by the click listener.
                        # We keep this structure for compatibility but acknowledge the ambiguity.
                        point_mode_every = gr.Radio(
                            choices=["Positive"],
                            value="Positive",
                            label="Point Prompt (Refinement)",
                            interactive=True,
                            visible=False) # Hide this as the main one is in the Click tab

                        every_undo_but = gr.Button(
                                    value="Undo",
                                    interactive=True
                                    )

                tab_click = gr.Tab(label="Click")
                with tab_click:
                    with gr.Row():
                        # This is the primary point_mode control used by the listener.
                        point_mode = gr.Radio(
                                    choices=["Positive",  "Negative"],
                                    value="Positive",
                                    label="Point Prompt",
                                    interactive=True)

                        # args for modify and tracking 
                        click_undo_but = gr.Button(
                                    value="Undo",
                                    interactive=True
                                    )

                tab_stroke = gr.Tab(label="Stroke")
                with tab_stroke:
                    # Ensure type="numpy" for compatibility with backend processing
                    drawing_board = gr.ImageEditor(label='Drawing Board', type="numpy", brush=gr.Brush(default_size=10), interactive=True)
                    with gr.Row():
                        seg_acc_stroke = gr.Button(value="Segment", interactive=True)
                
                tab_text = gr.Tab(label="Text")
                with tab_text:
                    grounding_caption = gr.Textbox(label="Detection Prompt", placeholder="e.g., car. person.")
                    detect_button = gr.Button(value="Detect")
                    with gr.Accordion("Advanced options", open=False):
                        with gr.Row():
                            with gr.Column(scale=2):
                                box_threshold = gr.Slider(
                                    label="Box Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                                )
                            with gr.Column(scale=2):
                                text_threshold = gr.Slider(
                                    label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                                )
                
                with gr.Row():
                    with gr.Column(scale=2): 
                        with gr.Tab(label="SegTracker Args"):
                            # args for tracking in video do segment-everthing
                            points_per_side = gr.Slider(
                                label = "points_per_side",
                                minimum= 1,
                                step = 1,
                                maximum=100,
                                value=16,
                                interactive=True
                            )

                            sam_gap = gr.Slider(
                                label='sam_gap',
                                minimum = 1,
                                step=1,
                                maximum = 9999,
                                value=100,
                                interactive=True,
                            )

                            max_obj_num = gr.Slider(
                                label='max_obj_num',
                                minimum = 50,
                                step=1,
                                maximum = 300,
                                value=255,
                                interactive=True
                            )
                            with gr.Accordion("aot advanced options", open=False):
                                aot_model = gr.Dropdown(
                                    label="aot_model",
                                    choices = [
                                        "deaotb",
                                        "deaotl",
                                        "r50_deaotl"
                                    ],
                                    value = "r50_deaotl",
                                    interactive=True,
                                )
                                long_term_mem = gr.Slider(label="long term memory gap", minimum=1, maximum=9999, value=9999, step=1)
                                max_len_long_term = gr.Slider(label="max len of long term memory", minimum=1, maximum=9999, value=9999, step=1)
                            with gr.Accordion("model checkpoints", open=False):
                                sam_ckpt = gr.Dropdown(
                                    label="SAM checkpoint",
                                    choices=[
                                        "ckpt/sam_vit_h_4b8939.pth",
                                        "ckpt/sam_vit_l_0b3195.pth",
                                        "ckpt/sam_vit_b_01ec64.pth",
                                    ],
                                    value=default_sam_args["sam_checkpoint"],
                                    interactive=True,
                                )
                                gd_ckpt = gr.Dropdown(
                                    label="GroundingDINO checkpoint",
                                    choices=[
                                        "ckpt/groundingdino_swint_ogc.pth",
                                        "ckpt/groundingdino_swinb_cogcoor.pth",
                                    ],
                                    value=default_gd_args["ckpt_path"],
                                    interactive=True,
                                )
                    
                    with gr.Column():
                        new_object_button = gr.Button(
                            value="Add new object", 
                            interactive=True
                        )
                        reset_button = gr.Button(
                            value="Reset",
                            interactive=True,
                        )
                        track_for_video = gr.Button(
                            value="Start Tracking",
                                interactive=True,
                                )

            with gr.Column(scale=2):
                # output_video = gr.Video(label='Output video', height=550)
                output_video = gr.File(label="Predicted video")
                output_mask = gr.File(label="Predicted masks")
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Accordion("roll back options", open=False):
                            output_res = gr.Image(label='Segment result of all frames', height=550, type="numpy")
                            frame_per = gr.Slider(
                                label = "Percentage of Frames Viewed",
                                minimum= 0.0,
                                maximum= 100.0,
                                step=0.01,
                                value=0.0,
                            )
                            frame_per.release(show_res_by_slider, inputs=[input_video, input_img_seq, frame_per], outputs=[output_res, frame_num])
                            roll_back_button = gr.Button(value="Choose this mask to refine")
                            refine_res = gr.Image(label='Refine masks', height=550, type="numpy", interactive=True)

                            tab_roll_back_click = gr.Tab(label="Click")
                            with tab_roll_back_click:
                                with gr.Row():
                                    roll_back_point_mode = gr.Radio(
                                                choices=["Positive",  "Negative"],
                                                value="Positive",
                                                label="Point Prompt",
                                                interactive=True)

                                    # args for modify and tracking 
                                    roll_back_click_undo_but = gr.Button(
                                                value="Undo",
                                                interactive=True
                                                )
                                    roll_back_track_for_video = gr.Button(
                                    value="Start tracking to refine",
                                        interactive=True,
                                        )

    ##########################################################
    ######################  back-end #########################
    ##########################################################
        
        # Define configuration inputs list for easier reuse in listeners
        config_inputs = [
            aot_model, long_term_mem, max_len_long_term, 
            sam_gap, max_obj_num, points_per_side, sam_ckpt, gd_ckpt
        ]

        # listen to the input_video to get the first frame of video
        input_video.change(
            fn=get_meta_from_video,
            inputs=[
                input_video
            ],
            outputs=[
                input_first_frame, origin_frame, drawing_board, grounding_caption
            ]
        )

        # listen to the input_img_seq to get the first frame of video
        # Triggered by upload or the extract button
        img_seq_trigger = lambda seq: get_meta_from_img_seq(seq)
        
        # Use .upload for immediate feedback, and the button for confirmation/retry
        input_img_seq.upload(
            fn=img_seq_trigger,
            inputs=[input_img_seq],
            outputs=[input_first_frame, origin_frame, drawing_board, grounding_caption]
        )
        
        extract_button.click(
            fn=img_seq_trigger,
            inputs=[input_img_seq],
            outputs=[input_first_frame, origin_frame, drawing_board, grounding_caption]
        )
        
        #-------------- Input component -------------
        # FIX 3: Updated clean outputs to include grounding_caption
        clean_outputs = [
                input_video, input_img_seq, Seg_Tracker, input_first_frame,
                origin_frame, drawing_board, click_stack, grounding_caption
            ]

        tab_video_input.select(
            fn = clean,
            inputs=[],
            outputs=clean_outputs
        )

        tab_img_seq_input.select(
            fn = clean,
            inputs=[],
            outputs=clean_outputs
        )
        

        # ------------------- Interactive component -----------------
        # FIX 3: Use specific init functions to ensure correct return types for outputs
        
        # listen to the tab to init SegTracker
        tab_everything.select(
            fn=init_SegTracker, # Standard init is fine here (only core state needed)
            inputs=config_inputs + [origin_frame],
            outputs=[
                Seg_Tracker, input_first_frame, click_stack
            ],
            queue=False,
        )
        
        tab_click.select(
            fn=init_SegTracker, # Standard init is fine here
            inputs=config_inputs + [origin_frame],
            outputs=[
                Seg_Tracker, input_first_frame, click_stack
            ],
            queue=False,
        )

        tab_stroke.select(
            fn=init_for_stroke_tab,
            inputs=config_inputs + [origin_frame],
            outputs=[
                Seg_Tracker, input_first_frame, click_stack, drawing_board
            ],
            queue=False,
        )

        tab_text.select(
            fn=init_for_text_tab,
            inputs=config_inputs + [origin_frame],
            outputs=[
                Seg_Tracker, input_first_frame, click_stack, grounding_caption
            ],
            queue=False,
        )

        # Use SAM to segment everything for the first frame of video
        seg_every_first_frame.click(
            fn=segment_everything,
            inputs=[Seg_Tracker] + config_inputs + [origin_frame],
            outputs=[
                Seg_Tracker,
                input_first_frame,
            ],
            )
        
        # Interactively modify the mask acc click
        # Note: This uses the 'point_mode' defined in the 'Click' tab due to variable scoping.
        input_first_frame.select(
            fn=sam_click,
            inputs=[
                Seg_Tracker, origin_frame, point_mode, click_stack] + config_inputs,
            outputs=[
                Seg_Tracker, input_first_frame, click_stack
            ]
        )

        # Interactively segment acc stroke
        seg_acc_stroke.click(
            fn=sam_stroke,
            inputs=[Seg_Tracker, origin_frame, drawing_board] + config_inputs,
            outputs=[
                Seg_Tracker, input_first_frame, drawing_board
            ]
        )

        # Use grounding-dino to detect object
        detect_button.click(
            fn=gd_detect, 
            inputs=[
                Seg_Tracker, origin_frame, grounding_caption, box_threshold, text_threshold] + config_inputs,
            outputs=[
                Seg_Tracker, input_first_frame
                ]
                )

        # Add new object
        new_object_button.click(
            fn=add_new_object,
            inputs=
            [
                Seg_Tracker
            ],
            outputs=
            [
                Seg_Tracker, click_stack
            ]
        )

        # Track object in video
        track_for_video.click(
            fn=tracking_objects,
            inputs=[
                Seg_Tracker,
                input_video,
                input_img_seq,
                fps,
            ],
            outputs=[
                output_video, output_mask
            ]
        )

        # ----------------- Refine Mask ---------------------------
        output_res.select(
            fn = choose_obj_to_refine,
            inputs=[
                input_video, input_img_seq, Seg_Tracker, frame_num
            ],
            outputs=[output_res, refine_idx]
        )
        
        roll_back_button.click(
            fn=show_chosen_idx_to_refine,
            inputs=config_inputs + [input_video, input_img_seq, Seg_Tracker, frame_num, refine_idx],
            outputs=[
                refine_res, Seg_Tracker, origin_frame, click_stack, grounding_caption
            ],
            queue=False,
        )

        roll_back_click_undo_but.click(
            fn = roll_back_undo_click_stack_and_refine_seg,
            inputs=[
                Seg_Tracker, origin_frame, click_stack, input_video, input_img_seq, frame_num, refine_idx
            ],
            outputs=[
                Seg_Tracker, refine_res, click_stack
            ]
        ) 

        refine_res.select(
            fn=roll_back_sam_click,
            inputs=[
                Seg_Tracker, origin_frame, roll_back_point_mode, click_stack] +
                config_inputs + [
                input_video, input_img_seq, frame_num, refine_idx],
            outputs=[
                Seg_Tracker, refine_res, click_stack
            ]
        )

        # Track object in video
        roll_back_track_for_video.click(
            fn=tracking_objects,
            inputs=[
                Seg_Tracker,
                input_video,
                input_img_seq,
                fps, frame_num
            ],
            outputs=[
                output_video, output_mask
            ]
        )

        # ----------------- Reset and Undo ---------------------------
        # Rest 
        # FIX 3: Use reset_app_state to clear both caption and drawing board
        reset_button.click(
            fn=reset_app_state,
            inputs=config_inputs + [origin_frame],
            outputs=[
                Seg_Tracker, input_first_frame, click_stack, grounding_caption, drawing_board
            ],
            queue=False,
        ) 

        # Undo click
        click_undo_but.click(
            fn = undo_click_stack_and_refine_seg,
            inputs=[
                Seg_Tracker, origin_frame, click_stack
            ],
            outputs=[
                Seg_Tracker, input_first_frame, click_stack
            ]
        )

        every_undo_but.click(
            fn = undo_click_stack_and_refine_seg,
            inputs=[
                Seg_Tracker, origin_frame, click_stack
            ],
            outputs=[
                Seg_Tracker, input_first_frame, click_stack
            ]
        )
    
    # Set queue concurrency
    # app.queue(concurrency_count=1)
    # Launch the app with debug=True to see server-side errors
    app.launch(server_name="localhost", server_port=12345, debug=True, share=True)

if __name__ == "__main__":
    # Ensure necessary directories exist
    os.makedirs(ASSETS_DIR, exist_ok=True)
    os.makedirs(TRACKING_RESULTS_DIR, exist_ok=True)
    seg_track_app()
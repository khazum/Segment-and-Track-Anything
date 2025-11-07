import os
import shutil
import zipfile
import gc

import cv2
import imageio
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation

from aot_tracker import _palette

def save_prediction(pred_mask, output_dir, file_name):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask.save(os.path.join(output_dir, file_name))

def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)

def draw_mask(img, mask, alpha=0.5, id_countour=False):
    img_mask = img.copy()
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids!=0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id*3:id*3+3]
            else:
                color = [0,0,0]
            foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask!=0)
        countours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
        foreground = img * (1 - alpha) + colorize_mask(mask) * alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours, :] = 0
        
    return img_mask.astype(img.dtype)

def create_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


TRACKING_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "tracking_results")
ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')
   
aot_model2ckpt = {
    "deaotb": "./ckpt/DeAOTB_PRE_YTB_DAV.pth",
    "deaotl": "./ckpt/DeAOTL_PRE_YTB_DAV.pth",
    "r50_deaotl": "./ckpt/R50_DeAOTL_PRE_YTB_DAV.pth",
}

def _zip_directory(base_dir: str, rel_dir: str, out_zip_path: str) -> None:
    """Zip the contents of ``rel_dir`` (relative to ``base_dir``) into ``out_zip_path``."""
    with zipfile.ZipFile(out_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        root_dir = os.path.join(base_dir, rel_dir)
        for root, _, files in os.walk(root_dir):
            for f in sorted(files):
                full = os.path.join(root, f)
                arc = os.path.relpath(full, base_dir)
                zf.write(full, arc)

def _iterate_frames(input_video=None, input_img_seq=None):
    """
    Generator to yield frames (RGB), frame names (str), and indices (int) 
    from either a video file or an image sequence directory.
    """
    
    # Define supported image extensions
    supported_exts = (".png", ".jpg", ".jpeg", ".bmp")

    if input_video:
        # Video Source
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {input_video}")
        
        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Use zero-padded index as filename base for consistency
                frame_name = str(frame_idx).zfill(5) 
                yield frame_rgb, frame_name, frame_idx
                frame_idx += 1
        finally:
            cap.release()

    elif input_img_seq:
        # Image Sequence Source
        # Determine path to extracted images (assuming extraction happened in app.py)
        file_name = os.path.basename(input_img_seq.name).split('.')[0]
        file_path = os.path.join(ASSETS_DIR, file_name)
        
        if not os.path.isdir(file_path):
            raise FileNotFoundError(f"Image sequence directory not found: {file_path}. Ensure extraction occurred.")

        # List and sort image paths
        imgs_path = sorted([
            os.path.join(file_path, n) for n in os.listdir(file_path) 
            if n.lower().endswith(supported_exts)
        ])
        
        if not imgs_path:
            raise ValueError("No images found in the sequence directory.")

        for frame_idx, img_path in enumerate(imgs_path):
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Warning: Skipping unreadable image {img_path}")
                continue
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Use the original filename (without extension) as the frame name
            frame_name = os.path.basename(img_path).split('.')[0]
            yield frame_rgb, frame_name, frame_idx
    else:
        raise ValueError("No input source provided.")

def _get_source_metadata(input_video, input_img_seq, input_fps):
    """Extracts metadata (fps, width, height) from the input source."""
    try:
        if input_video:
            cap = cv2.VideoCapture(input_video)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {input_video}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return fps, width, height
        elif input_img_seq:
            # Rely on the generator to find the files and handle errors
            frame_generator = _iterate_frames(input_video, input_img_seq)
            # Peek the first frame to get dimensions
            try:
                first_frame, _, _ = next(frame_generator)
                height, width = first_frame.shape[:2]
                fps = input_fps # Use the provided FPS for image sequences
                return fps, width, height
            except StopIteration:
                raise ValueError("Image sequence is empty or contains no readable images.")
    except Exception as e:
        print(f"Error determining source metadata: {e}")
        return None, None, None
    return None, None, None


def tracking_objects_in_video(SegTracker, input_video, input_img_seq, input_fps, start_frame_num=0):
    """
    Unified function to track objects in a video or image sequence.
    """
    
    # 1. Determine video name and setup output directories
    if input_video is not None:
        video_name = os.path.basename(input_video).split('.')[0]
    elif input_img_seq is not None:
        video_name = os.path.basename(input_img_seq.name).split('.')[0]
    else:
        return None, None

    tracking_result_dir = os.path.join(TRACKING_RESULTS_DIR, video_name)
    output_mask_dir = os.path.join(tracking_result_dir, f'{video_name}_masks')
    output_masked_frame_dir = os.path.join(tracking_result_dir, f'{video_name}_masked_frames')
    output_video_path = os.path.join(tracking_result_dir, f'{video_name}_seg.mp4')
    output_gif_path = os.path.join(tracking_result_dir, f'{video_name}_seg.gif')

    # Clean up previous results if starting from the beginning
    if start_frame_num == 0:
        shutil.rmtree(tracking_result_dir, ignore_errors=True)

    create_dir(tracking_result_dir)
    create_dir(output_mask_dir)
    create_dir(output_masked_frame_dir)

    # 2. Get metadata (fps, width, height)
    fps, width, height = _get_source_metadata(input_video, input_img_seq, input_fps)
    if fps is None:
        return None, None

    # 3. Initialize Video and GIF writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))    
    gif_fps = int(max(1, round(fps)))

    # 4. Core Tracking Loop (Unified and Single-Pass)
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam_gap = SegTracker.sam_gap
    
    # Optimization: Reduce frequency of explicit GC/cache clearing
    CLEANUP_FREQUENCY = 50 # frames

    try:
        # Initialize the frame iterator
        frame_generator = _iterate_frames(input_video, input_img_seq)
        
        with imageio.get_writer(output_gif_path, mode='I', fps=gif_fps) as gif_writer:
            
            # Handle Resume/Refinement (start_frame_num > 0)
            if start_frame_num > 0:
                print(f"Resuming tracking from frame {start_frame_num}...")
                # A. Prefill writers with existing results (if they exist)
                existing_masked = sorted(os.listdir(output_masked_frame_dir))
                prefill_count = min(start_frame_num, len(existing_masked))
                
                for i in range(prefill_count):
                    img_path = os.path.join(output_masked_frame_dir, existing_masked[i])
                    bgr = cv2.imread(img_path)
                    if bgr is None:
                        continue
                    # Ensure dimensions match if source changed size (less common but possible)
                    if bgr.shape[1] != width or bgr.shape[0] != height:
                        bgr = cv2.resize(bgr, (width, height))
                    video_writer.write(bgr)
                    gif_writer.append_data(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

                # B. Skip frames in the generator *before* start_frame_num
                for _ in range(start_frame_num):
                    next(frame_generator, None)

            # Start processing from the current position of the generator
            torch.cuda.empty_cache()
            gc.collect()

            use_autocast = device_type == 'cuda'
            with torch.amp.autocast(device_type=device_type, enabled=use_autocast):
                
                # We use enumerate starting from start_frame_num for correct indexing
                for loop_idx, (frame_rgb, frame_name, source_idx) in enumerate(frame_generator, start=start_frame_num):
                    current_frame_idx = loop_idx
                    # --- Tracking Logic ---
                    if current_frame_idx == start_frame_num:
                        # Use the mask initialized in SegTracker (either first frame or refined frame)
                        pred_mask = SegTracker.first_frame_mask
                        torch.cuda.empty_cache(); gc.collect()
                    elif (current_frame_idx % sam_gap) == 0:
                        # Periodic re-segmentation (SAM) and fusion
                        seg_mask = SegTracker.seg(frame_rgb)
                        track_mask = SegTracker.track(frame_rgb)
                        # find new objects, and update tracker with new objects
                        new_obj_mask = SegTracker.find_new_objs(track_mask, seg_mask)
                        # Optional: save new object masks for debugging
                        # save_prediction(new_obj_mask, output_mask_dir, f'{frame_name}_new.png')

                        pred_mask = track_mask + new_obj_mask
                        SegTracker.add_reference(frame_rgb, pred_mask)
                    else:
                        pred_mask = SegTracker.track(frame_rgb, update_memory=True)

                    # --- Output Generation (Single Pass) ---
                    
                    # Save raw mask (palette)
                    save_prediction(pred_mask, output_mask_dir, f'{frame_name}.png')

                    # Generate overlay
                    masked_frame_rgb = draw_mask(frame_rgb, pred_mask)
                    # Persist per-frame overlay PNG for rollback UI
                    cv2.imwrite(
                        os.path.join(output_masked_frame_dir, f'{frame_name}.png'),
                        masked_frame_rgb[:, :, ::-1] # RGB to BGR
                    )
                    
                    # Write to video and GIF
                    video_writer.write(cv2.cvtColor(masked_frame_rgb, cv2.COLOR_RGB2BGR))
                    gif_writer.append_data(masked_frame_rgb)
                    print(f"Processed frame {current_frame_idx}, obj_num {SegTracker.get_obj_num()}", end='\r')
                    
                    # Periodic cleanup
                    if (current_frame_idx + 1) % CLEANUP_FREQUENCY == 0:
                        torch.cuda.empty_cache(); gc.collect()

        print(f"\nTracking finished.")
        print(f"{output_video_path} saved.")
        print(f"{output_gif_path} saved.")

    except Exception as e:
        print(f"\nAn error occurred during tracking: {e}")
        # Clean up resources in case of error
        torch.cuda.empty_cache(); gc.collect()
        return None, None
    finally:
        video_writer.release()

    # 5. Zip predicted masks
    zip_path = os.path.join(tracking_result_dir, f"{video_name}_pred_mask.zip")
    _zip_directory(tracking_result_dir, f"{video_name}_masks", zip_path)
                
    # 6. Final memory release
    del SegTracker
    torch.cuda.empty_cache()
    gc.collect()
    return output_video_path, zip_path
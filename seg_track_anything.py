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

def tracking_objects_in_video(SegTracker, input_video, input_img_seq, fps, frame_num=0):
    if input_video is not None:
        video_name = os.path.basename(input_video).split('.')[0]
    elif input_img_seq is not None:
        file_name = input_img_seq.name.split('/')[-1].split('.')[0]
        file_path = f'./assets/{file_name}'
        imgs_path = sorted([os.path.join(file_path, img_name) for img_name in os.listdir(file_path)])
        video_name = file_name
    else:
        return None, None

    # create dir to save result 
    tracking_result_dir = f'{os.path.join(os.path.dirname(__file__), "tracking_results", f"{video_name}")}'
    create_dir(tracking_result_dir)
    
    io_args = {
        'tracking_result_dir': tracking_result_dir,
        'output_mask_dir': f'{tracking_result_dir}/{video_name}_masks',
        'output_masked_frame_dir': f'{tracking_result_dir}/{video_name}_masked_frames',
        'output_video': f'{tracking_result_dir}/{video_name}_seg.mp4', # keep same format as input video
        'output_gif': f'{tracking_result_dir}/{video_name}_seg.gif',
    }

    if input_video is not None:
        return video_type_input_tracking(SegTracker, input_video, io_args, video_name, frame_num)
    elif input_img_seq is not None:
        return img_seq_type_input_tracking(SegTracker, io_args, video_name, imgs_path, fps, frame_num)

def video_type_input_tracking(SegTracker, input_video, io_args, video_name, frame_num=0):
    pred_list = []
    masked_pred_list = []

    # source video to segment
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # create dir to save predicted mask and masked frame
    if frame_num == 0:
        shutil.rmtree(io_args['output_mask_dir'], ignore_errors=True)
        shutil.rmtree(io_args['output_masked_frame_dir'], ignore_errors=True)
    output_mask_dir = io_args['output_mask_dir']
    create_dir(io_args['output_mask_dir'])
    create_dir(io_args['output_masked_frame_dir'])

    # Prepare video writer and streaming GIF writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))
    gif_fps = int(max(1, round(fps)))

    try:
        with imageio.get_writer(io_args['output_gif'], mode='I', fps=gif_fps) as gif_writer:
            # If resuming, pre-fill MP4/GIF with existing masked frames, then skip those frames in cap
            if frame_num > 0:
                existing_masked = sorted([img_name for img_name in os.listdir(io_args['output_masked_frame_dir'])])
                prefill = min(frame_num, len(existing_masked))
                for i in range(prefill):
                    bgr = cv2.imread(os.path.join(io_args['output_masked_frame_dir'], existing_masked[i]))
                    if bgr is None:
                        continue
                    out.write(bgr)
                    gif_writer.append_data(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                # Fast-forward the capture
                for _ in range(prefill):
                    cap.read()

            torch.cuda.empty_cache()
            gc.collect()
            sam_gap = SegTracker.sam_gap
            frame_idx = 0
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            with torch.amp.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    if frame_idx == 0:
                        pred_mask = SegTracker.first_frame_mask
                        torch.cuda.empty_cache(); gc.collect()
                    elif (frame_idx % sam_gap) == 0:
                        seg_mask = SegTracker.seg(frame_rgb)
                        torch.cuda.empty_cache(); gc.collect()
                        track_mask = SegTracker.track(frame_rgb)
                        # find new objects, and update tracker with new objects
                        new_obj_mask = SegTracker.find_new_objs(track_mask, seg_mask)
                        save_prediction(new_obj_mask, output_mask_dir, str(frame_idx + frame_num).zfill(5) + '_new.png')
                        pred_mask = track_mask + new_obj_mask
                        SegTracker.add_reference(frame_rgb, pred_mask)
                    else:
                        pred_mask = SegTracker.track(frame_rgb, update_memory=True)
                    torch.cuda.empty_cache(); gc.collect()

                    # Save raw mask (palette) for downstream usage/refinement
                    save_prediction(pred_mask, output_mask_dir, str(frame_idx + frame_num).zfill(5) + '.png')

                    # Overlay and stream-write MP4 & GIF
                    masked_frame_rgb = draw_mask(frame_rgb, pred_mask)
                    # Persist per-frame overlay PNG for rollback UI
                    cv2.imwrite(
                        f"{io_args['output_masked_frame_dir']}/{str(frame_idx + frame_num).zfill(5)}.png",
                        masked_frame_rgb[:, :, ::-1]
                    )
                    out.write(cv2.cvtColor(masked_frame_rgb, cv2.COLOR_RGB2BGR))
                    gif_writer.append_data(masked_frame_rgb)

                    print("processed frame {}, obj_num {}".format(frame_idx + frame_num, SegTracker.get_obj_num()), end='\r')
                    frame_idx += 1
        cap.release()
        print("\n{} saved".format(io_args['output_video']))
        print("{} saved".format(io_args['output_gif']))
        print('\nfinished')
    finally:
        out.release()

    # zip predicted mask
    zip_path = f"{io_args['tracking_result_dir']}/{video_name}_pred_mask.zip"
    _zip_directory(io_args['tracking_result_dir'], f"{video_name}_masks", zip_path)
                
    # manually release memory (after cuda out of memory)
    del SegTracker
    torch.cuda.empty_cache()
    gc.collect()

    return io_args['output_video'], f"{io_args['tracking_result_dir']}/{video_name}_pred_mask.zip"

def img_seq_type_input_tracking(SegTracker, io_args, video_name, imgs_path, fps, frame_num=0):
    # create dir to save predicted mask and masked frame
    if frame_num == 0:
        # Clean previous outputs without noisy shell errors
        shutil.rmtree(io_args['output_mask_dir'], ignore_errors=True)
        shutil.rmtree(io_args['output_masked_frame_dir'], ignore_errors=True)

    output_mask_dir = io_args['output_mask_dir']
    create_dir(io_args['output_mask_dir'])
    create_dir(io_args['output_masked_frame_dir'])
    
    i_frame_num = frame_num 
    torch.cuda.empty_cache(); gc.collect()
    sam_gap = SegTracker.sam_gap
    frame_idx = 0

    # Prepare video writer and streaming GIF writer from the first image size
    first = cv2.imread(imgs_path[0])
    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))
    gif_fps = int(max(1, round(fps)))

    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.amp.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
        for img_path in imgs_path:
            if i_frame_num > 0:
                i_frame_num = i_frame_num - 1
                continue

            frame_name = os.path.basename(img_path).split('.')[0]
            frame = cv2.imread(img_path)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            
            if frame_idx == 0:
                pred_mask = SegTracker.first_frame_mask
                torch.cuda.empty_cache()
                gc.collect()
            elif (frame_idx % sam_gap) == 0:
                seg_mask = SegTracker.seg(frame)
                torch.cuda.empty_cache()
                gc.collect()
                track_mask = SegTracker.track(frame)
                # find new objects, and update tracker with new objects
                new_obj_mask = SegTracker.find_new_objs(track_mask,seg_mask)
                save_prediction(new_obj_mask, output_mask_dir, f'{frame_name}_new.png')
                pred_mask = track_mask + new_obj_mask
                SegTracker.add_reference(frame, pred_mask)
            else:
                pred_mask = SegTracker.track(frame,update_memory=True)
            torch.cuda.empty_cache()
            gc.collect()
            
            save_prediction(pred_mask, output_mask_dir, f'{frame_name}.png')
            pred_list.append(pred_mask)

            print("processed frame {}, obj_num {}".format(frame_idx+frame_num, SegTracker.get_obj_num()),end='\r')
            frame_idx += 1
        print('\nfinished')
    
    ##################
    # Visualization
    ##################

    # draw pred mask on frame and save as a video
    height, width = pred_list[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    i_frame_num =frame_num 

    out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))

    frame_idx = 0
    for img_path in imgs_path:
        frame_name = os.path.basename(img_path).split('.')[0]
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        pred_mask = pred_list[frame_idx]
        masked_frame = draw_mask(frame, pred_mask)
        masked_pred_list.append(masked_frame)
        cv2.imwrite(f"{io_args['output_masked_frame_dir']}/{frame_name}.png", masked_frame[:, :, ::-1])

        masked_frame = cv2.cvtColor(masked_frame,cv2.COLOR_RGB2BGR)
        out.write(masked_frame)
        print('frame {} writed'.format(frame_name),end='\r')
        frame_idx += 1
    out.release()
    print("\n{} saved".format(io_args['output_video']))
    print('\nfinished')

    # save colorized masks as a gif
    imageio.mimsave(io_args['output_gif'], masked_pred_list, fps=fps)
    print("{} saved".format(io_args['output_gif']))

    # zip predicted mask (stdlib, no shell)
    zip_path = f"{io_args['tracking_result_dir']}/{video_name}_pred_mask.zip"
    _zip_directory(io_args['tracking_result_dir'], f"{video_name}_masks", zip_path)

    # manually release memory (after cuda out of memory)
    del SegTracker
    torch.cuda.empty_cache()
    gc.collect()

    return io_args['output_video'], f"{io_args['tracking_result_dir']}/{video_name}_pred_mask.zip"
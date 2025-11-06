import torch
import torch.nn.functional as F
import sys
from typing import List, Tuple, Union, Optional, Dict, Any
sys.path.append("./aot")  # keep for local package resolution
from aot.networks.engines.aot_engine import AOTEngine,AOTInferEngine
from aot.networks.engines.deaot_engine import DeAOTEngine,DeAOTInferEngine
import importlib
import numpy as np
import aot.dataloaders.video_transforms as tr
from aot.utils.checkpoint import load_network
from aot.networks.models import build_vos_model
from aot.networks.engines import build_engine
from torchvision import transforms

np.random.seed(200)
_palette = ((np.random.random((3 * 255)) * 0.7 + 0.3) * 255).astype(np.uint8).tolist()
_palette = [0,0,0]+_palette


class AOTTracker:
    """Thin wrapper around AOT/DeAOT engines with consistent transforms and APIs."""

    def __init__(self, cfg, gpu_id: int = 0):
        self.gpu_id: int = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        # Build and load model
        self.model = build_vos_model(cfg.MODEL_VOS, cfg).to(self.device)
        self.model, _ = load_network(self.model, cfg.TEST_CKPT_PATH, gpu_id)
        # Keep a canonical dtype to avoid float/half mismatches with DeAOT
        # (many checkpoints run in fp16). We'll cast inputs/labels to this dtype.
        self.param_dtype = next(self.model.parameters()).dtype
        self.engine = build_engine(cfg.MODEL_ENGINE,
                                   phase='eval',
                                   aot_model=self.model,
                                   gpu_id=gpu_id,
                                   short_term_mem_skip=1,
                                   long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP,
                                   max_len_long_term=cfg.MAX_LEN_LONG_TERM)
        self.transform = transforms.Compose([
            tr.MultiRestrictSize(cfg.TEST_MAX_SHORT_EDGE,
                                 cfg.TEST_MAX_LONG_EDGE, cfg.TEST_FLIP, 
                                 cfg.TEST_MULTISCALE, cfg.MODEL_ALIGN_CORNERS),
            tr.MultiToTensor()
        ])

        self.model.eval()

    @torch.no_grad()
    def add_reference_frame(self,
                            frame: np.ndarray,
                            mask: np.ndarray,
                            obj_nums: Union[int, List[int]],
                            frame_step: int,
                            incremental: bool = False) -> None:        
        # mask = cv2.resize(mask, frame.shape[:2][::-1], interpolation = cv2.INTER_NEAREST)
        sample = {
            'current_img': frame,
            'current_label': mask,
        }
    
        sample = self.transform(sample)
        frame_t = sample[0]['current_img'].unsqueeze(0).to(self.device, dtype=self.param_dtype)
        mask_t = sample[0]['current_label'].unsqueeze(0).to(self.device, dtype=self.param_dtype)
        _mask = F.interpolate(mask_t, size=frame_t.shape[-2:], mode='nearest')

        if incremental:
            self.engine.add_reference_frame_incremental(frame_t, _mask, obj_nums=obj_nums, frame_step=frame_step)
        else:
            self.engine.add_reference_frame(frame_t, _mask, obj_nums=obj_nums, frame_step=frame_step)



    @torch.no_grad()
    def track(self, image: np.ndarray) -> torch.Tensor:
        """Track objects for a single frame and return the label map tensor on CPU."""
        output_height, output_width = image.shape[0], image.shape[1]
        output_height, output_width = image.shape[0], image.shape[1]
        sample = {'current_img': image}
        sample = self.transform(sample)
        # Use model dtype (fp16 on many checkpoints) to keep ops consistent
        image_t = sample[0]['current_img'].unsqueeze(0).to(self.device, dtype=self.param_dtype)        
        self.engine.match_propogate_one_frame(image_t)
        pred_logit = self.engine.decode_current_logits((output_height, output_width))
        # Cast argmax to model dtype; keep on device for immediate memory update.
        pred_label = torch.argmax(pred_logit, dim=1, keepdim=True).to(self.param_dtype)
        return pred_label
    
    @torch.no_grad()
    def update_memory(self, pred_label: torch.Tensor) -> None:
        # Engine expects same device/dtype as model params
        pred_label = pred_label.to(device=self.device, dtype=self.param_dtype)
        self.engine.update_memory(pred_label)    
        
    @torch.no_grad()
    def restart(self) -> None:
        self.engine.restart_engine()
    
    @torch.no_grad()
    def build_tracker_engine(self, name: str, **kwargs):
        if name == 'aotengine':
            return AOTTrackerInferEngine(**kwargs)
        elif name == 'deaotengine':
            return DeAOTTrackerInferEngine(**kwargs)
        else:
            raise NotImplementedError


class AOTTrackerInferEngine(AOTInferEngine):
    def __init__(self, aot_model, gpu_id=0, long_term_mem_gap=9999, short_term_mem_skip=1, max_aot_obj_num=None):
        super().__init__(aot_model, gpu_id, long_term_mem_gap, short_term_mem_skip, max_aot_obj_num)
    def add_reference_frame_incremental(self, img, mask, obj_nums, frame_step: int = -1):
        if isinstance(obj_nums, list):
            obj_nums = obj_nums[0]
        self.obj_nums = obj_nums
        aot_num = max(np.ceil(obj_nums / self.max_aot_obj_num), 1)
        while (aot_num > len(self.aot_engines)):
            new_engine = AOTEngine(self.AOT, self.gpu_id,
                                   self.long_term_mem_gap,
                                   self.short_term_mem_skip)
            new_engine.eval()
            self.aot_engines.append(new_engine)

        separated_masks, separated_obj_nums = self.separate_mask(
            mask, obj_nums)
        img_embs = None
        for aot_engine, separated_mask, separated_obj_num in zip(
                self.aot_engines, separated_masks, separated_obj_nums):
            if aot_engine.obj_nums is None or aot_engine.obj_nums[0] < separated_obj_num:
                aot_engine.add_reference_frame(img,
                                            separated_mask,
                                            obj_nums=[separated_obj_num],
                                            frame_step=frame_step,
                                            img_embs=img_embs)
            else:
                aot_engine.update_short_term_memory(separated_mask)
                
            if img_embs is None:  # reuse image embeddings
                img_embs = aot_engine.curr_enc_embs

        self.update_size()



class DeAOTTrackerInferEngine(DeAOTInferEngine):
    def __init__(self, aot_model, gpu_id=0, long_term_mem_gap=9999, short_term_mem_skip=1, max_aot_obj_num=None):
        super().__init__(aot_model, gpu_id, long_term_mem_gap, short_term_mem_skip, max_aot_obj_num)
    def add_reference_frame_incremental(self, img, mask, obj_nums, frame_step=-1):
        if isinstance(obj_nums, list):
            obj_nums = obj_nums[0]
        self.obj_nums = obj_nums
        aot_num = max(np.ceil(obj_nums / self.max_aot_obj_num), 1)
        while (aot_num > len(self.aot_engines)):
            new_engine = DeAOTEngine(self.AOT, self.gpu_id,
                                   self.long_term_mem_gap,
                                   self.short_term_mem_skip)
            new_engine.eval()
            self.aot_engines.append(new_engine)

        separated_masks, separated_obj_nums = self.separate_mask(
            mask, obj_nums)
        img_embs = None
        for aot_engine, separated_mask, separated_obj_num in zip(
                self.aot_engines, separated_masks, separated_obj_nums):
            if aot_engine.obj_nums is None or aot_engine.obj_nums[0] < separated_obj_num:
                aot_engine.add_reference_frame(img,
                                            separated_mask,
                                            obj_nums=[separated_obj_num],
                                            frame_step=frame_step,
                                            img_embs=img_embs)
            else:
                aot_engine.update_short_term_memory(separated_mask)
                
            if img_embs is None:  # reuse image embeddings
                img_embs = aot_engine.curr_enc_embs

        self.update_size()


def get_aot(args):
    """Factory for AOTTracker from simple args dict.

    Expected keys: phase, model, model_path, long_term_mem_gap, max_len_long_term, gpu_id
    """
    # build vos engine
    engine_config = importlib.import_module('configs.' + 'pre_ytb_dav')
    cfg = engine_config.EngineConfig(args['phase'], args['model'])
    cfg.TEST_CKPT_PATH = args['model_path']
    cfg.TEST_LONG_TERM_MEM_GAP = args['long_term_mem_gap']
    cfg.MAX_LEN_LONG_TERM = args['max_len_long_term']
    # init AOTTracker
    tracker = AOTTracker(cfg, args['gpu_id'])
    return tracker

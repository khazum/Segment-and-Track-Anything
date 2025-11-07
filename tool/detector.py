import os
import torch
import numpy as np
import cv2
import PIL

from groundingdino.models import build_model as build_grounding_dino
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, predict
import groundingdino.datasets.transforms as T
import groundingdino

from torchvision.ops import box_convert
def _resolve_config_path(config_file: str) -> str:
    """Return a usable path for a GroundingDINO config, trying local repo first, then the pip package."""
    if os.path.exists(config_file):
        return config_file
    pkg_base = os.path.join(os.path.dirname(groundingdino.__file__), "config")
    candidate = os.path.join(pkg_base, os.path.basename(config_file))
    if os.path.exists(candidate):
        return candidate
    # Fallback to SwinT OGC config in this repo
    return "config/GroundingDINO_SwinT_OGC.py"

DEFAULT_CKPT_PATH = "./ckpt/groundingdino_swint_ogc.pth"
DEFAULT_CONFIG_FILE = "config/GroundingDINO_SwinT_OGC.py"


class Detector:
    def __init__(self, device, ckpt_path: str = DEFAULT_CKPT_PATH,
                 config_file: str = DEFAULT_CONFIG_FILE):
        config_file = _resolve_config_path(config_file)
        args = SLConfig.fromfile(config_file)
        args.device = device
        self.device = device
        self.gd = build_grounding_dino(args)

        grounding_dino_ckpt = ckpt_path
        checkpoint = torch.load(grounding_dino_ckpt, map_location='cpu')
        log = self.gd.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(grounding_dino_ckpt, log))
        self.gd.eval()
    
    def image_transform_grounding(self, init_image):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image, _ = transform(init_image, None) # 3, h, w
        return init_image, image

    def image_transform_grounding_for_vis(self, init_image):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
        ])
        image, _ = transform(init_image, None) # 3, h, w
        return image

    def transfer_boxes_format(self, boxes, height, width):
        boxes = boxes * torch.Tensor([width, height, width, height])
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")

        transfered_boxes = []
        for i in range(len(boxes)):
            box = boxes[i]
            transfered_box = [[int(box[0]), int(box[1])], [int(box[2]), int(box[3])]]
            transfered_boxes.append(transfered_box)
        
        transfered_boxes = np.array(transfered_boxes)
        return transfered_boxes
        
    @torch.no_grad()
    def run_grounding(self, origin_frame, grounding_caption, box_threshold, text_threshold):
        '''
            return:
                annotated_frame:nd.array
                transfered_boxes: nd.array [N, 4]: [[x0, y0], [x1, y1]]
        '''
        height, width, _ = origin_frame.shape
        img_pil = PIL.Image.fromarray(origin_frame)
        re_width, re_height = img_pil.size
        _, image_tensor = self.image_transform_grounding(img_pil)
        # img_pil = self.image_transform_grounding_for_vis(img_pil)

        # run grounding
        boxes, logits, phrases = predict(self.gd, image_tensor, grounding_caption, box_threshold, text_threshold, device=self.device)        
        annotated_frame = annotate(image_source=np.asarray(img_pil), boxes=boxes, logits=logits, phrases=phrases)[:, :, ::-1]
        annotated_frame = cv2.resize(annotated_frame, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # transfer boxes to sam-format 
        transfered_boxes = self.transfer_boxes_format(boxes, re_height, re_width)
        return annotated_frame, transfered_boxes

if __name__ == "__main__":
    detector = Detector("cuda")
    origin_frame = cv2.imread('./debug/point.png')
    origin_frame = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2RGB)
    grounding_caption = "swan.water"
    box_threshold = 0.25
    text_threshold = 0.25

    annotated_frame, boxes = detector.run_grounding(origin_frame, grounding_caption, box_threshold, text_threshold)
    cv2.imwrite('./debug/x.png', annotated_frame)

    for i in range(len(boxes)):
        bbox = boxes[i]
        origin_frame = cv2.rectangle(origin_frame, bbox[0], bbox[1], (0, 0, 255))
    cv2.imwrite('./debug/bbox_frame.png', origin_frame)
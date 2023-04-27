import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from os import path



import torch
import numpy as np
from PIL import Image

from IPython.display import display

from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.inference_core import InferenceCore
from progressbar import progressbar
from inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis

image = cv2.imread('images/first_frame.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
device = "cuda"
torch.set_grad_enabled(False)

#apply autoSAM
def autoSAM(image):
  sam_checkpoint = "sam_vit_h_4b8939.pth"
  model_type = "vit_h"
  sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
  sam.to(device=device)

  mask_generator = SamAutomaticMaskGenerator(sam)
  masks = mask_generator.generate(image)

  print(len(masks))
  print(masks[0].keys())

  height, width, channels = image.shape
  cv2.imwrite('first_mask.png',masks)


#XMem

def apply_xmem(video_name = 'output.mp4', mask_name = 'red_mask.png', frames_to_propagate= 7, visualize_every = 1):

  # default configuration
  config = {
      'top_k': 30,
      'mem_every': 5,
      'deep_update_every': -1,
      'enable_long_term': True,
      'enable_long_term_count_usage': True,
      'num_prototypes': 128,
      'min_mid_term_frames': 5,
      'max_mid_term_frames': 7,
      'max_long_term_elements': 10000,
  }

  network = XMem(config, './saves/XMem.pth').eval().to(device)

  # Convert the mask to a numpy array
  # Note that the object IDs should be consecutive and start from 1 (0 represents the background). If they are not, see `inference.data.mask_mapper` and `eval.py` on how to use it.

  mask = np.array(Image.open(mask_name))
  print(np.unique(mask))
  num_objects = len(np.unique(mask)) - 1
  print(num_objects)

  # Propagte frame-by-frame
  processor = InferenceCore(network, config=config)
  processor.set_all_labels(range(1, num_objects+1)) # consecutive labels
  cap = cv2.VideoCapture(video_name)

  current_frame_index = 0

  with torch.cuda.amp.autocast(enabled=True):
    while (cap.isOpened()):
      # load frame-by-frame
      _, frame = cap.read()
      if frame is None or current_frame_index > frames_to_propagate:
        break

      # convert numpy array to pytorch tensor format
      frame_torch, _ = image_to_torch(frame, device=device)
      if current_frame_index == 0:
        # initialize with the mask
        mask_torch = index_numpy_to_one_hot_torch(mask, num_objects+1).to(device)
        # the background mask is not fed into the model
        prediction = processor.step(frame_torch, mask_torch[1:])
      else:
        # propagate only
        prediction = processor.step(frame_torch)

      # argmax, convert to numpy
      prediction = torch_prob_to_numpy_mask(prediction)

      if current_frame_index % visualize_every == 0:
        visualization = overlay_davis(frame, prediction)
        display(Image.fromarray(visualization))

      current_frame_index += 1
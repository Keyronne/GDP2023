import torch
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.torch_utils import select_device
from models.experimental import attempt_load
import matplotlib.pyplot as plt
import cv2
import numpy as np

def load_model(poseweights= 'yolov7-w6-pose.pt'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(poseweights, map_location=device)['model']
    # Put in inference mode
    model.float().eval()
    if torch.cuda.is_available():
      # half() turns predictions into float16 tensors
      # which significantly lowers inference time
      model.half().to(device)
    return model

def video_output(frame,model):
    image = frame
    # Resize and pad image
    image = letterbox(image, 640, stride=64, auto=True)[0] 
    # Apply transforms
    image = transforms.ToTensor()(image)
    # Turn image into batch
    image = image.unsqueeze(0) 
    output, _ = model(image)
    output = non_max_suppression_kpt(output, 
                                     0.25, # Confidence Threshold
                                     0.65, # IoU Threshold
                                     nc=model.yaml['nc'], # Number of Classes
                                     nkpt=model.yaml['nkpt'], # Number of Keypoints
                                     kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
    return nimg,output

def run_inference(img,model):
    image = cv2.imread(img) 
    # Resize and pad image
    image = letterbox(image, 640, stride=64, auto=True)[0] 
    # Apply transforms
    image = transforms.ToTensor()(image) 
    image = image.cuda()
    # Turn image into batch
    image = image.unsqueeze(0) 
    output, _ = model(image) 
    return output, image

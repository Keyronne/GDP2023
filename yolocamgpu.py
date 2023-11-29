import cv2
import time
import torch
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.torch_utils import select_device, prune
from models.experimental import attempt_load
import matplotlib.pyplot as plt
import numpy as np
from yollloogpu import video_output, run_inference

device = torch.device("cuda:0")
model = torch.load('yolov7-w6-pose.pt', map_location=device)['model']
prune(model,0.3)
model.float().eval()
if torch.cuda.is_available():
    model.half().to(device)
frame_count, total_fps = 0, 0
cap = cv2.VideoCapture(0)
while cap.isOpened:
    check, frame = cap.read()
    frame = letterbox(frame, 640, stride=64, auto=True)[0]
    start_time = time.time()
    nimg,output=video_output(frame,model)
    key = cv2.waitKey(1)
    if key == ord('c'):
        break
    cv2.imshow("Detection", nimg)
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    print(f"Frame {frame_count} Processing,FPS: {fps}")
    total_fps += fps
    frame_count += 1
cap.release()
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
cv2.destroyAllWindows()
print(output.shape)
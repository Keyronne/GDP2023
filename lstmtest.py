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
from lstm import MyModel
device = torch.device("cuda:0")
model = torch.load('yolov7-w6-pose.pt', map_location=device)['model']
model.float().eval()
frame_count, total_fps = 0, 0
cap = cv2.VideoCapture(0)
lstm = MyModel(56,100,2).cuda()
lstm.load_state_dict(torch.load('lstmmodel_weights.pth'))
label_map = {"fit": 0, "unfit": 1}
buffer_window = np.empty((0,56))
while cap.isOpened:
    check, frame = cap.read()
    frame = letterbox(frame, 640, stride=64, auto=True)[0]
    start_time = time.time()
    nimg,output=video_output(frame,model)
    output=output[0,2:]
    if len(buffer_window) < 30:
        buffer_window=np.vstack((buffer_window,output))
    else:
        test=torch.tensor(buffer_window, device='cuda', dtype=torch.float).unsqueeze(0)
        lstm.eval()
        with torch.no_grad():
            # Forward pass
            Y_pred = lstm(test)
        # Get the index of the maximum value in Y_pred tensor
        pred_index = torch.argmax(Y_pred, dim=1)

        # Convert the index tensor to a numpy array
        pred_index = pred_index.cpu().numpy()

        # Map the index values to the corresponding action name using the label map dictionary
        action_name = [k for k, v in label_map.items() if v == pred_index[0]][0]
        buffer_window=buffer_window[1:]
        buffer_window=np.vstack((buffer_window,output))
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        cv2.putText(nimg, "{}".format(action_name), (int(output[0]), int(output[1])), cv2.FONT_HERSHEY_COMPLEX, 0.9, (102, 255, 255), 2)
        cv2.imshow("detection", nimg)
    key = cv2.waitKey(1)
    if key == ord('c'): # press C to stop camera
        break
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    print(f"Frame {frame_count} Processing,FPS: {fps}")
    total_fps += fps
    frame_count += 1
cap.release()
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
cv2.destroyAllWindows()

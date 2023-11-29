Before running yolocam.py you need to get yolov7 repository 

git clone https://github.com/WongKinYiu/yolov7.git

make sure you are in the clloned file

cd yolov7 
or cd your/path/yolov7

Dowwnload the pre trained weight

curl -L https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt -o yolov7-w6-pose.pt

install all the requirnment from yolo7

pip install -r requirements.txt

then just run the yolocam.py

python yolocam.py

only works on cpu. will work on cuda next

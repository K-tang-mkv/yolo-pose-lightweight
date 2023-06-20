### 数据处理
项目地址：[http://192.168.100.40:8888/lab/tree/data1/edgeai-yolov5](http://192.168.100.40:8888/lab/tree/data1/edgeai-yolov5)
1. 先用检测人的模型检测出人体框，得到yolo格式的txt结果

`
python detect.py --weights persondet_offical.pt --source /data1/tot_yd_data/add-2/squat/images/ --img-size 960 --device 0 --save-txt --name squat --kpt-label
`

在目录runs/detect/official/labels，得到每张图片的检测结果，如下：

`
0 0.54 0.34 0.65 0.64       #class, center_x(normalize), center_y, w, h
`

2. 原始骨骼点json转yolo格式，并添加可见性标签，有坐标的标为2， 坐标为0，0的就标为0

step：
* 读取json
* 归一化坐标
* 有坐标的标为2，没有的标0
* 输出txt
* 最后的标签格式如：`0.590982 0.215625 2.000000` (x y visibility)
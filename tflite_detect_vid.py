import tensorflow as tf
import cv2
import torch
import numpy as np
import os

from utils.general import non_max_suppression
from utils.datasets import letterbox
from utils.tflite3output_postprocess import Detect, detect
# from utils.tflite3output_postprocess import Detect
from onnx_inference.yolo_pose_onnx_inference import post_process
from utils.general import scale_coords, xyxy2xywh
from utils.plots import plot_one_box, colors

img_file = './test.jpg'
dst_path = './output/'
vid_file = './data/videos/ticao.mp4'
os.makedirs(dst_path, exist_ok=True)

if __name__ == "__main__":
    # Load model
    interpreter = tf.lite.Interpreter("./ds_8w_content/ds16_8w.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("all good")

    # load video
    cap  = cv2.VideoCapture(vid_file)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter("./output/output.mp4", fourcc, 60.0, (width, height))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"fps: {fps} /n total_frames: {total_frames}")

    for i in range(total_frames):
        ret, im0 = cap.read()
    # Load image
    # img = cv2.imread(img_file)[:, :, ::-1]
        if ret:
            # im0 = cv2.imread(img_file)
            img = letterbox(im0, (192,192),auto=False)[0]
            # img = cv2.resize(img, (192, 192), interpolation=cv2.INTER_LINEAR)
            img = img / 255.0
            img = np.asarray(img, dtype=np.float32)
            img = np.expand_dims(img, 0)
            img = img.transpose(0, 3, 1, 2)

            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            # pred = torch.from_numpy(interpreter.get_tensor(output_details[0]['index']))
            output = [torch.from_numpy(interpreter.get_tensor(index)) for index in sorted(i['index'] for i in output_details)]
            # last_Detect = Detect()
            # pred = last_Detect(output)
            # det = post_process(det)
            pred = detect(output)
            pred = non_max_suppression(pred, kpt_label=True)
            # Rescale boxes from img_size to im0 size
            prediction = []
            for i, det in enumerate(pred):
                scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=True, step=3)
                prediction.append(det)
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                for det_index, det_person in enumerate(reversed(det)):
                    *xyxy, conf, cls = det_person[:6]
                    keypoint_coords = det_person[6:]
                    if True:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = [cls, *xywh, conf] if False else [float(cls), *xywh]  # label format
                        # 关键点的信息
                        keypoint_coords = keypoint_coords.view(17, 3)
                        for i, coord in enumerate(keypoint_coords):
                            x, y, vis_conf = coord
                            if vis_conf >= 0.5:
                                x /= gn[0]
                                y /= gn[1]
                                visible = float(2.0)
                                line.extend([float(x), float(y), visible])
                            else:
                                # 归一化关键点坐标
                                line.extend([float(0.0), float(0.0), float(0)])  # 添加关键点的坐标和可见性

                        with open(dst_path+"test" + '.txt', 'a') as f:
                            f.write(('%.6f ' * len(line)).rstrip() % tuple(line) + '\n')

                    if True:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = f'person {conf:.2f}'
                        kpts = det[det_index, 6:]
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=3,
                                     kpt_label=True, kpts=kpts, steps=3, orig_shape=im0.shape[:2])
                        des_file = os.path.join(dst_path, os.path.basename(img_file))
                        cv2.imshow("Frame", im0)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        out.write(im0)
                        # cv2.imwrite(des_file, im0)

    # post_process(img_file, des_file, prediction[0])
    print("all good")
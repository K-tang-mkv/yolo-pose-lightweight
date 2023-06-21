import tensorflow as tf
import cv2
import torch
import numpy as np

from utils.general import non_max_suppression
from utils.datasets import letterbox
from utils.tflite3output_postprocess import Detect
# from utils.tflite3output_postprocess import Detect


def post_process(output):
    z = []
    for det_scale in output:
        x = det_scale.reshape(1, -1, det_scale.shape[4])
        z.append(x)

    det = tf.concat(z, 1)
    return det


if __name__ == "__main__":
    # Load model
    interpreter = tf.lite.Interpreter("./posedet_offical_640-fp32.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("all good")

    # Load image
    img = cv2.imread('./test.jpg')[:, :, ::-1]
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    img = img / 255.
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img, 0)
    # img = img.transpose(0, 3, 1, 2)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    pred = torch.from_numpy(interpreter.get_tensor(output_details[0]['index']))
    # det = [torch.from_numpy(interpreter.get_tensor(output_details[i]['index'])) for i in range(len(output_details))]
    last_Detect = Detect()
    # pred = last_Detect(det)
    # det = post_process(det)
    pred = non_max_suppression(pred, kpt_label=True)[0].numpy()

    print("all good")
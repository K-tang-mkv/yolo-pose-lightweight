import tensorflow as tf
import cv2
import numpy as np

from utils.general import non_max_suppression


def post_process(output):
    z = []
    for det_scale in output:
        x = det_scale.reshape(1, -1, det_scale.shape[4])
        z.append(x)

    det = tf.concat(z, 1)
    return det


if __name__ == "__main__":
    # Load model
    interpreter = tf.lite.Interpreter("./cocoyd-fp32.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("all good")

    # Load image
    img = cv2.imread('./test.jpg')
    im_size = input_details[0]['shape'][1]
    input = cv2.resize(img, (im_size, im_size)).reshape((1, im_size, im_size, 3)).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input)
    interpreter.invoke()
    det = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    det = post_process(det)
    pred = non_max_suppression(det, kpt_label=True)

    print("all good")
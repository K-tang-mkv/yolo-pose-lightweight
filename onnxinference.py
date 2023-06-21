import onnxruntime as ort
import numpy as np
import cv2
from utils.datasets import letterbox


ort_session = ort.InferenceSession('ds_8w.onnx')

if __name__ == "__main__":
    # Load image
    img = cv2.imread('./test.jpg')[:, :, ::-1]
    img = cv2.resize(img, (192, 192), interpolation=cv2.INTER_LINEAR)
    img = (img - 127.5) * 1/127.5
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img, 0)
    img = img.transpose(0, 3, 1, 2)
    outputs = ort_session.run(
        None,
        {'images': img}
    )
    print("all good")
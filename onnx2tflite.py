import onnx
from onnx_tf.backend import prepare
import tensorflow as tf


if __name__ == "__main__":
    # onnx_model = onnx.load("./ds_8w.onnx")
    # tf_rep = prepare(onnx_model)
    # tf_rep.export_graph("ds_8w_tf")
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model("ds_8w_tf")
    tflite_model = converter.convert()

    # Save the model
    with open("ds_8w.tflite", 'wb') as f:
        f.write(tflite_model)
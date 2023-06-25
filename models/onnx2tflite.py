import tensorflow as tf


saved_model = "../ds_8w_Detect/"
saved_file = saved_model+"ds_8w_detect.tflite"

if __name__ == "__main__":
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()

    with open(saved_file, 'wb') as f:
        f.write(tflite_model)
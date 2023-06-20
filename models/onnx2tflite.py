import sys
import os

# Import the model optimizer tool from the openvino_dev package
from openvino.tools.mo import main as mo_main
import onnx
from onnx_tf.backend import prepare
from mltk.utils.shell_cmd import run_shell_cmd


if __name__ == "__main__":

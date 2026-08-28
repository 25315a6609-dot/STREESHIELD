import sys
import numpy
import pandas
import cv2
import sklearn
import tensorflow as tf
import streamlit
from PIL import Image
print("streesheild test environment is working fine")
print("----------------------------------------------")
print("Python version:", sys.version)
print("NumPy version:", numpy.__version__)
print("Pandas version:", pandas.__version__)    
print("OpenCV version:", cv2.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("TensorFlow version:", tf.__version__)
print("Streamlit version:", streamlit.__version__)
print("-----------------------------------------------")
print("tensor flow devices")
devices = tf.config.list_physical_devices()
for device in devices:
    print(device)   
print("-----------------------------------------------")
print("environment is ready to use")
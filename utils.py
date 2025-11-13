import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)

def preprocess(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype('float32')
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, 0)
    return arr

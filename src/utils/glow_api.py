import utils.start_tf as glow
from PIL import Image
import numpy as np


def load_encode(path, file_name):
    img = Image.open(path+file_name)
    new_size = (256, 256)
    img = img.resize(new_size)
    img = np.reshape(np.array(img), [1, 256, 256, 3])
    eps = glow.encode(img)
    return eps


def save_decoce(eps, path, file_name):
    dec = glow.decode(eps)
    img = Image.fromarray(dec[0])
    img.save(path+file_name)

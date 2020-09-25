import os
import numpy as np
import glob
from PIL import Image

file_path = 'D:\\It_worked_yesterday\\img_224x224\\'
pokemon = os.listdir(file_path)

for name in pokemon:
    print(name)
    directory = file_path + name + '\\'
    files = glob.glob(directory + "*.jpg")

    for file in files:
        img = Image.open(file)
        data = np.asarray(img) / 255

        if np.shape(data[..., np.newaxis])[2] == 1:
            os.remove(file)

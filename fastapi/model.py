from typing import Dict
import numpy as np
from PIL import Image
import io
# import matplotlib as mpl

class FMIModel:

    def __init__(self):
        pass

    def preprocess(self, image_array):
        image_array[image_array == -9999] = np.nan
        return image_array

    def predict(self, arr):
        arr = self.preprocess(arr)
        # cm_hot = mpl.cm.get_cmap('YlOrBr')
        arr = arr[5000:5500,:]
        print(arr, arr.tobytes())
        # arr = cm_hot(arr)
        return io.BytesIO(arr.tobytes())      ## output 1


# # test ##
# if __name__ == "__main__":
#     im = FMIModel()
#     print(im.predict(img))
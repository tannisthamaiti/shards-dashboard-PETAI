from typing import Dict
import numpy as np
import cv2
import matplotlib.pyplot as plt

class FMIModel:

    def __init__(self):
        pass

    def preprocess_output1(self, image_array):
        image_array[image_array == -9999] = np.nan
        plt.imshow(image_array[5000:5500,:], cmap='YlOrBr')
        plt.savefig("output1.png")

    def preprocess_output2(self, image_array):
        image_array[image_array == -9999] = np.nan
        img = image_array[3800:4300]
        img =  img[:,~np.all(np.isnan(img), axis=0)]
        _, thresh1 = cv2.threshold(img, 245, 255, cv2.THRESH_BINARY)
        img_bw = 255 - thresh1
        img = img_bw.astype(np.uint8)
        img = cv2.medianBlur(img, 5)
        edges = cv2.Canny(img, 200, 255, apertureSize = 7)
        plt.imshow(edges, cmap="gray")
        plt.savefig("output2.png")

    def predict(self, arr):
        self.preprocess_output1(arr)
        self.preprocess_output2(arr)
        return {"output1": "Saved at output1.png",
                "output2": "Saved at output2.png"}     ## output 1


# # test ##
# if __name__ == "__main__":
#     im = FMIModel()
#     print(im.predict(img))
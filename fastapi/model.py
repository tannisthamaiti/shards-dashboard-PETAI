from typing import Dict
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from mynet import MyNet

class FMIModel:

    def __init__(self, verbose=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.width = 0
        self.height = 0
        self.nChannel = 100
        self.lr = 0.1
        self.stepsize_sim = 1
        self.stepsize_con = 1
        self.stepsize_scr = 0.5
        # TODO: look into how to get this using another npz file -> line number: 132 in train.py
        self.label_colours = np.random.randint(255, size=(self.nChannel, 3))
        self.args_label_colours = False ## as we are using randomint above set it True if we upload another npz file
        self.maxIter = 200
        self.use_scribble = False
        self.minLabels = 3

    def preprocess_output1(self, image_array):
        image_array[image_array == -9999] = np.nan
        plt.imshow(image_array[5000:5500,:], cmap='YlOrBr')
        plt.savefig("outputs/output1.png")

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
        plt.savefig("outputs/output2.png")

    def load_image(self, img_path):
        im = cv2.imread(img_path)
        im_arr = im.transpose((2, 0, 1)).astype('float32')/255.
        data = torch.from_numpy(np.array(im_arr)).to(self.device)
        return im, Variable(data)
    
    def load_scribble(self, img_path):
        mask = cv2.imread(img_path)[self.width:, self.height:, -1]
        mask = mask.reshape(-1)
        mask_inds = np.unique(mask)
        mask_inds = np.delete(mask_inds, np.argwhere(mask_inds == 255))
        inds_sim = torch.from_numpy(np.where(mask == 255)[0]).to(self.device)
        inds_scr = torch.from_numpy(np.where(mask != 255)[0]).to(self.device)
        target_scr = torch.from_numpy(mask.astype(np.int)).to(self.device)
        return inds_sim, inds_scr, Variable(target_scr)
    
    def prepare_data(self, arr):
        self.preprocess_output1(arr)
        self.preprocess_output2(arr)
        self.im, self.data = self.load_image("outputs/output1.png")
        self.inds_sim, self.inds_scr, self.target_scr = self.load_scribble("outputs/output2.png")

    def train(self):
        # get the network
        self.model = MyNet(self.data.size(0)).to(self.device)   ## currently passing c from [c,h,w] in train.py h is passed
        self.model.train()
        # similarity loss definition
        loss_fn = torch.nn.CrossEntropyLoss()
        # scribble loss definition
        loss_fn_scr = torch.nn.CrossEntropyLoss()
        # continuity loss definition
        loss_hpy = torch.nn.L1Loss(size_average = True)
        loss_hpz = torch.nn.L1Loss(size_average = True)
        HPy_target = torch.zeros(self.im.shape[0]-1, self.im.shape[1], self.nChannel).to(self.device)
        HPz_target = torch.zeros(self.im.shape[0], self.im.shape[1]-1, self.nChannel).to(self.device)
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        loss = None

        for batch_idx in range(self.maxIter):
            # forwarding
            optimizer.zero_grad()
            output = self.model(self.data[None])[0]
            output = output.permute(1, 2, 0).contiguous().view(-1, self.nChannel)
            outputHP = output.reshape((self.im.shape[0], self.im.shape[1], self.nChannel))
            HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
            HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
            lhpy = loss_hpy(HPy, HPy_target)
            lhpz = loss_hpz(HPz, HPz_target)
            _, target = torch.max(output, 1)
            im_target = target.data.cpu().numpy()
            self.nLabels = len(np.unique(im_target))

            if self.use_scribble:
                loss = self.stepsize_sim * loss_fn(output[self.inds_sim], target[self.inds_sim]) + \
                        self.stepsize_scr * loss_fn_scr(output[self.inds_scr], self.target_scr[self.inds_scr]) + \
                        self.stepsize_con * (lhpy + lhpz)
            else:
                loss = self.stepsize_sim * loss_fn(output, target) + self.stepsize_con * (lhpy + lhpz)
        
            loss.backward()
            optimizer.step()

            if self.verbose:
                print(output.shape)
                print(target.shape)
                print(self.inds_sim)
                print(len(self.inds_sim))
                print(self.target_scr.shape)
                print(self.inds_scr)
                print(len(self.inds_scr))
                print(batch_idx, '/', self.maxIter, '|', ' label num :', self.nLabels, ' | loss :', loss.item())
            if self.nLabels <= self.minLabels:
                if self.verbose:
                    print ("nLabels", self.nLabels, "reached minLabels", self.minLabels, ".")
                break

    def predict(self, arr):
        
        # prepare data
        self.prepare_data(arr)
        
        # train model
        self.train()
        
        # predict
        self.model.eval()
        output = self.model(self.data[None])[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, self.nChannel)
        _, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()
        im_target_rgb = np.array([self.label_colours[c % self.nChannel] for c in im_target])
        im_target_rgb = im_target_rgb.reshape(self.im.shape).astype(np.uint8)
        
        if self.use_scribble:
            cv2.imwrite("outputs/scribble_output_" + str(self.stepsize_con) + "_" + str(self.nLabels) + ".png", im_target_rgb)
            if not self.args_label_colours:
                np.save("outputs/scribble_color_coding_" + str(self.stepsize_con) + "_" + str(self.nLabels) + ".npy", self.label_colours)
            m_name = "scribble_checkpoint_" + str(self.stepsize_con) + "_" + str(self.nLabels) + ".pth"
            torch.save(self.model, f"outputs/{m_name}")
        else:
            cv2.imwrite("outputs/output_" + str(self.stepsize_con) + "_" + str(self.nLabels) + ".png", im_target_rgb)
            if not self.args_label_colours:
                np.save("outputs/color_coding_" + str(self.stepsize_con) + "_" + str(self.nLabels) + ".npy", self.label_colours)
            m_name = "checkpoint_" + str(self.stepsize_con) + "_" + str(self.nLabels) + ".pth"
            torch.save(self.model, f"outputs/{m_name}")

        return {"output1": "Saved at outputs/output1.png",
                "output2": "Saved at outputs/output2.png"
                }

# # test ##
# if __name__ == "__main__":
#     im = FMIModel()
#     print(im.predict(img))

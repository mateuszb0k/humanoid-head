import numpy as np
import cv2
from cv2 import dnn
import os

# importing and load models
proto_file = r'Model\colorization_deploy_v2.prototxt'
model_file = r'Model\colorization_release_v2.caffemodel'
hull_pts   = r'Model\pts_in_hull.npy'
#import dataset
data = "./dataset_to_rgb/FINAL_bez_dzieci"
#dictionary with emotions in dataset
emotions = ["happy", "sad", "fear", "neutral", "angry", "disgust", "surprise"]
#create dir for rgb photos
os.makedirs("./dataset_rgb", exist_ok=True)

#reading the models parameters
net    = dnn.readNetFromCaffe(proto_file, model_file)
kernel = np.load(hull_pts)

# adding cluster centres as 1x1 convolutions to the model
pts = kernel.transpose().reshape(2, 313, 1, 1)
class8 = net.getLayerId("class8_ab")
conv8  = net.getLayerId("conv8_313_rh")
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs  = [np.full([1, 313], 2.606, dtype="float32")]


for emotion in emotions:
    # get the path for each emotion
    img_dir = os.path.join(data, emotion)
    os.makedirs(f"./dataset_rgb/{emotion}", exist_ok=True)

    # for loop on each img in each folder
    for idx, filename in enumerate(os.listdir(img_dir)):
        full_path = os.path.join(img_dir, filename)
        # load img
        img = cv2.imread(full_path)
        # resie img from 96x96 to 224x224 using INTER_LANCZOS4 interpolation
        img = cv2.resize(img, (224, 224),interpolation=cv2.INTER_LANCZOS4)
        #scaling our gray img
        scaled  = img.astype("float32") / 255.0
        # convert bgr to lab img
        lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        #split the L channel
        L = cv2.split(lab_img)[0]
        # mean subtraction
        L -= 50

        #predicting the ab channels from the input L channel
        net.setInput(cv2.dnn.blobFromImage(L))
        ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
        # resize the predic 'ab' volume to the same dim as our input img
        ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))

        # taking L channel from the img
        L = cv2.split(lab_img)[0]
        # Join the L channel with predicted ab channel
        colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)
        # Then convert the image from Lab to BGR
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)
        # change the image to 0-255 range and convert it from float32 to int
        colorized = (255 * colorized).astype("uint8")

        # save the new img in created folder
        cv2.imwrite(f"./dataset_rgb/{emotion}/{emotion}_{idx}.jpg", colorized)

print("END")
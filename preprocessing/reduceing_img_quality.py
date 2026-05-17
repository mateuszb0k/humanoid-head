import cv2
import numpy as np
import os

path =""
mean = 0
sigma = 10**0.5

for root,dirs,files in os.walk(path):
    for file_name in files:
        if file_name.endswith(".png"):
            full_path = os.path.join(root,file_name)
            image = cv2.imread(full_path)
            base_img = image
            h,w = image.shape[:2]
            kernel_x = cv2.getGaussianKernel(w,w/2)
            kernel_y = cv2.getGaussianKernel(h, h/ 2)
            kernel = kernel_y * kernel_x.T
            mask = kernel/kernel.max()
            mask = mask[:,:,np.newaxis]
            img_float = image.astype(np.float32)
            img_with_mask = img_float * mask
            gausian_noise = np.random.normal(mean, sigma, image.shape)
            noisy_image = img_with_mask +  gausian_noise
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
            image_resize = cv2.resize(noisy_image, (112, 112), interpolation=cv2.INTER_LINEAR)
            image_resize = cv2.resize(image_resize, (h,w), interpolation=cv2.INTER_NEAREST)

            cv2.imshow("normalImage",base_img)
            cv2.imshow("Noisy Image", noisy_image)
            cv2.imshow("Resized Image", image_resize)

            img_with_mask_toDisplay = np.clip(img_with_mask, 0, 255).astype(np.uint8)
            cv2.imshow("Image with mask",img_with_mask_toDisplay)

            cv2.waitKey(0)
            continue
import cv2
import os
import scipy
from tqdm import tqdm
import numpy as np


video_folder = "output_frames_2"

images = [f'{video_folder}/{file_name}' for file_name in os.listdir(video_folder) if file_name != '.DS_Store']

print(len(images))
images = sorted(images)

print(images)

init_img = cv2.imread(images[0], 0)

canvas  = cv2.selectROI(init_img)
#canvas = (347, 118, 176, 220)
print(canvas)
template_box = cv2.selectROI(init_img)
#template_box = (379, 180, 120, 120)
print(template_box)

template = init_img[template_box[1]: template_box[1] + template_box[3],template_box[0] : template_box[0] + template_box[2]]

template_mean = np.mean(template)

box_filter = np.ones((template_box[2], template_box[3]))/ (template_box[3]*template_box[2])

template_shift = template - template_mean
template_var = np.var(template_shift)


shifts = []

focus_image = np.zeros((init_img.shape[0], init_img.shape[1], 3))

t = np.arange(init_img.shape[1])
s = np.arange(init_img.shape[0])

for id, img_file in tqdm(enumerate(images)):

    if img_file == '.DS_Store':
        continue
    #print(id, img_file)
    img = cv2.imread(img_file, 0)
    img_search = img[canvas[1]:canvas[1]+canvas[3],\
                        canvas[0]:canvas[0]+canvas[2]]
    
    #print(img_search.shape, box_filter.shape)
    img_boxed = scipy.signal.correlate2d(img_search, box_filter, mode='same')

    img_boxed_shift = img_search-img_boxed
    img_boxed_var = np.var(img_boxed_shift)

    #print(img_boxed_shift.shape, template_shift.shape)
    h = scipy.signal.correlate2d(img_boxed_shift, template_shift, mode='same')

    h = h/(np.sqrt(img_boxed_var*template_var))

    h_w = (((h - np.min(h))/np.max(h))*255).astype(np.uint8)

    max_idx = np.argmax(h)

    y = max_idx//canvas[2]
    x = max_idx%canvas[2]

    curr_img = cv2.imread(img_file)

    for channel in range(3):
        
        func = scipy.interpolate.interp2d(t,s,curr_img[:,:,channel])

        if id == 0:

            x_prime = x
            y_prime = y

            currT = t
            currS = s
        
        else:

            shiftX = x_prime - x
            shiftY = y_prime - y

            currT = t - shiftX
            currS = s - shiftY

        focus_image[:,:,channel] += func(currT, currS) 
    

focus_image = focus_image/len(images)

print(focus_image.shape)
cv2.imwrite('output.jpg', focus_image)


     








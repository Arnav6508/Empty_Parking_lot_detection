from skimage.transform import resize
import cv2
import pickle
import numpy as np

model = pickle.load(open('./model.p', 'rb'))

empty_path = './data_img/empty/00000000_00000161.jpg'
not_path = './data_img/not_empty/00000000_00000003.jpg'

def isEmpty(spot_bgr):
    flat_data = []
    img = resize(spot_bgr, (15, 15, 3))
    flat_data.append(img.flatten())
    flat_data = np.array(flat_data)

    y_output = model.predict(flat_data)

    if y_output[0] == 0: return 1
    else: return 0

def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components
    
    coef = 1
    slots = []

    for i in range(1,totalLabels):

        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)
    
        slots.append([x1, y1, w, h])
    
    return slots

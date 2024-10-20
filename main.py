import cv2
from util import get_parking_spots_bboxes, isEmpty
import numpy as np

video_path = './data_video/parking_1920_1080_loop.mp4'
mask_path = './mask_1920_1080.png'

mask = cv2.imread(mask_path,0)
cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

prev_frame = None
ret = True
frame_nmr = 0
steps = 30
spots_status = [0 for j in spots]
total_spots = 396
empty_spots = 0

while ret:
    ret, frame = cap.read()

    if frame_nmr % steps == 0:
        empty_spots = 0
        for spot_idx, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1+h, x1:x1+w, :]

            spot_status = isEmpty(spot_crop)
            spots_status[spot_idx] = spot_status
            empty_spots += spot_status

        prev_frame = frame.copy()

    for spot_idx, spot in enumerate(spots):
        x1, y1, w, h = spot
        spot_status = spots_status[spot_idx]

        if spot_status : 
            frame = cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
        else: 
            frame = cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 0, 255), 2)

    frame_nmr += 1

    cv2.rectangle(frame, (15, 25), (440, 70), (0, 0, 0), -1)
    cv2.putText(frame, f'Available spots: {empty_spots}/{total_spots}', (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
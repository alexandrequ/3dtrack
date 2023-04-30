import cv2
import numpy as np
from segment_anything-gui import get_mask, new_tow

def track_3d(input, first_mask):

    # Load the mask image
    mask = cv2.imread(first_mask, cv2.IMREAD_GRAYSCALE)

    for frame in input:

        # Threshold the mask to create a binary image
        thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]

        # Find the contours in the binary image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cx = []
        cy = []
        # Loop over each contour and calculate its centroid
        for cnt in contours:
            # Calculate the moments of the contour
            M = cv2.moments(cnt)
            # Check if the contour has a non-zero area
            if M['m00'] != 0:
                # Calculate the centroid of the contour
                center_x = int(M['m10'] / M['m00'])
                center_y = int(M['m01'] / M['m00'])
                cx += center_x
                cy += center_y
                # Draw a circle at the centroid coordinates
                cv2.circle(mask, (cx, cy), 5, (255, 0, 0), -1)
            else:
                # Handle the case where the contour has zero area
                print('Contour has zero area')

        
        for x_add, y_add in zip(cx, cy):
            x_rem = cx.remove(x_add)
            y_rem = cx.remove(y_add)
            get_mask()
            new_tow()

       mask = save_annotation(self, labels_file_outpath)









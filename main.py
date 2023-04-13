"""
    Programmer: Emmanuel Samuel
    Date Created: 12th April 2023
    Revised: 13th April 2023
    Purpose: A Human, Animal, Object Detection and Tracking surveillance monitoring system using computer vision
"""
# import modules
import numpy as np
import cv2
import sys

# color of text defined
text_color = (0, 255, 0)
# tracker color defined
tracker_color = (255, 0, 0)
# font type defined
font_type = cv2.FONT_HERSHEY_SIMPLEX
# source of video defined
input_video = "{input your video source here}"

# define a list to hold background subtractor algorithms
bgs_types = ["GMG", "MOG", "MOG2", "KNN", "CNT"]
# select an algorithm
bgs_type = bgs_types[2]


# function to choose the type kernel for image preprocessing
def get_kernel(kernel_type):
    if kernel_type == "dilation":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if kernel_type == "opening":
        kernel = np.ones((3, 3), np.uint8)
    if kernel_type == "closing":
        kernel = np.ones((3, 3), np.uint8)

    return kernel


# function to apply morphological operations
def get_filter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, get_kernel("closing"), iterations=2)

    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, get_kernel("opening"), iterations=2)

    if filter == 'dilation':
        return cv2.dilate(img, get_kernel("dilation"), iterations=2)

    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, get_kernel("closing"), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, get_kernel("opening"), iterations=2)
        dilation = cv2.dilate(opening, get_kernel("dilation"), iterations=2)

        return dilation


# function to implement the algorithm
def get_bgsubtractor(bgs_type):
    if bgs_type == "GMG":
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if bgs_type == "MOG":
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if bgs_type == "MOG2":
        return cv2.createBackgroundSubtractorMOG2()
    if bgs_type == "KNN":
        return cv2.createBackgroundSubtractorKNN()
    if bgs_type == "CNT":
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    print("Invalid detector")
    sys.exit(1)


# capture the video
cap = cv2.VideoCapture(input_video)
# apply the background subtractor function
bg_subtractor = get_bgsubtractor(bgs_type)
# Minimum Area defined
minArea = 300


# main function definition
def main():
    while cap.isOpened:
        # check for frames
        success, frame = cap.read()
        # if no frames
        if not success:
            print("Video processing finished")
            break

        # resize frame of video for processing to be faster
        frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)

        # apply the algorithm to frame
        bg_mask = bg_subtractor.apply(frame)
        # apply the morphological process
        bg_mask = get_filter(bg_mask, 'combine')
        # apply the median blur preprocessing specifying the GaussianBlur as 5
        bg_mask = cv2.medianBlur(bg_mask, 5)

        # to detect contour in the video
        (contours, hierarchy) = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # iterate over contours and apply rectangle
        for cnt in contours:
            area = cv2.contourArea(cnt)
            """ 
            more functionalities can be added below
            such as alarm system, send image to owner/police
            """
            if area >= minArea:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (10, 30), (250, 55), (255, 0, 0), -1)
                cv2.putText(frame, 'Motion detected!', (10, 50), font_type, 0.8, text_color, 2, cv2.LINE_AA)

                cv2.drawContours(frame, cnt, -1, tracker_color, 3)
                cv2.drawContours(frame, cnt, -1, (255, 255, 255), 1)
                cv2.rectangle(frame, (x, y), (x+w, y+h), tracker_color, 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

                # # https://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
                # for alpha in np.arange(0.8, 1.1, 0.9)[::-1]:
                #     frame_copy = frame.copy()
                #     output = frame.copy()
                #     cv2.drawContours(frame_copy, [cnt], -1, tracker_color, -1)
                #     frame = cv2.addWeighted(frame_copy, alpha, output, 1 - alpha, 0, output)

        # improve the img processing result using a bitwise function
        result = cv2.bitwise_and(frame, frame, mask=bg_mask)
        cv2.imshow('Frame', frame)
        cv2.imshow('Mask', result)

        # press the q button to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


main()

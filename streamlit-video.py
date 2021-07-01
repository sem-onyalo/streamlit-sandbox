import cv2 as cv
from numpy import string_
import streamlit as st
import argparse
import time

videoFile = 'video.mp4'
videoOutFile = 'video-out.avi'

class TransformParams:
    transformType: str
    applyGaussianBlur: bool
    applyColorToGray: bool
    applyDilation: bool
    applyErosion: bool
    dilationSize: int
    erosionSize: int

def applyTransforms(img, params: TransformParams):
    if params.applyGaussianBlur:
        img = cv.GaussianBlur(img, (3,3), 0)

    if params.transformType == "Canny":
        if params.applyColorToGray:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        img = cv.Canny(img, 100, 200)
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    elif params.transformType == "Laplacian":
        if params.applyColorToGray:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ddepth = cv.CV_32F
        img = cv.Laplacian(img, ddepth)
        img = cv.convertScaleAbs(img)

        if params.applyColorToGray:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    elif params.transformType == "Sobel":
        if params.applyColorToGray:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ddepth = cv.CV_16S
        scale = 1
        delta = 2
        grad_x = cv.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
        grad_y = cv.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)
        img = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        
        if params.applyColorToGray:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    if params.applyDilation:
        element = cv.getStructuringElement(cv.MORPH_ELLIPSE, 
            (2 * params.dilationSize + 1, 2 * params.dilationSize + 1),
            (params.dilationSize, params.dilationSize))
        img = cv.dilate(img, element)

    if params.applyErosion:
        element = cv.getStructuringElement(cv.MORPH_ELLIPSE, 
            (2 * params.erosionSize + 1, 2 * params.erosionSize + 1),
            (params.erosionSize, params.erosionSize))
        img = cv.erode(img, element)
    
    return img

def main():

    videoWriter = None
    transformParams = TransformParams()

    timeFrame = st.empty()
    videoFrame = st.empty()
    writeToFile = st.button("Write to file")
    startTransformTime = st.text_input("Start transform in output video at (s)", 0)
    transformParams.applyTransforms = st.checkbox("Apply transforms")

    if applyTransforms:
        transformParams.applyColorToGray = st.checkbox("Apply color to gray")

        transformParams.applyGaussianBlur = st.checkbox("Apply Gaussian blur")

        transformParams.transformType = st.radio("Transform type", ('Canny', 'Laplacian', 'Sobel'))

        transformParams.applyDilation = st.checkbox("Apply dilation")
        if transformParams.applyDilation:
            transformParams.dilationSize = st.slider("Dilation size", 0, 10, 0)

        transformParams.applyErosion = st.checkbox("Apply erosion")
        if transformParams.applyErosion:
            transformParams.erosionSize = st.slider("Erosion size", 0, 10, 0)

    startTime = time.time()
    cap = cv.VideoCapture(videoFile)
    while True:
        elapsedTime = time.time() - startTime
        res, img = cap.read()

        if res:
            if transformParams.applyTransforms and (not writeToFile or (writeToFile and elapsedTime >= float(startTransformTime))):
                img = applyTransforms(img, transformParams)
            
            timeFrame.text(f'Video time: {elapsedTime:.2f}')
            videoFrame.image(img, channels="BGR")
            
            if writeToFile:
                if videoWriter == None:
                    st.text("writing to file " + videoOutFile + "...")
                    fourcc = cv.VideoWriter_fourcc('M','J','P','G')
                    videoWriter = cv.VideoWriter(videoOutFile, fourcc, 30, (img.shape[1], img.shape[0]), True)
                videoWriter.write(img)
        elif writeToFile:
            videoWriter.release()
            st.text("finished writing file!")
            break
        else:
            startTime = time.time()
            cap = cv.VideoCapture(videoFile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="video.mp4")
    parser.add_argument("-o", "--output", type=str, default="video-out.avi")
    args = parser.parse_args()

    videoFile = args.input
    videoOutFile = args.output

    main()
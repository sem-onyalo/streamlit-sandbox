import cv2 as cv
import streamlit as st
import argparse
import numpy as np

modelPath = 'models/mobilenet_ssd_v1_coco/frozen_inference_graph.pb'
configPath = 'models/mobilenet_ssd_v1_coco/ssd_mobilenet_v1_coco_2017_11_17.pbtxt'
classNames = { 
    0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
    13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
    18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
    24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
    32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
    37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
    67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
    75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
    86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush' 
}

class TransformParams:
    totalFrames: int
    currentFrame: int
    transformType: str
    channelIsolation: str
    applyGaussianBlur: bool
    cannyThreshold1: str
    cannyThreshold2: str
    dilationSize: int
    erosionSize: int
    contrast: float
    brightness: int
    xCropFlip: bool
    xCropStart: int
    xCropEnd: int

def applyTransform(img, params: TransformParams):
    orig_img = img

    if params.applyGaussianBlur:
        img = cv.GaussianBlur(img, (3,3), 0)

    if params.transformType == "Canny":
        img = cv.Canny(img, params.cannyThreshold1, params.cannyThreshold2)
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    elif params.transformType == "Laplacian":
        ddepth = cv.CV_32F
        img = cv.Laplacian(img, ddepth)
        img = cv.convertScaleAbs(img)
    elif params.transformType == "Sobel":
        ddepth = cv.CV_16S
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        grad_x = cv.Sobel(img, ddepth, 1, 0, ksize=3, scale=1, delta=2, borderType=cv.BORDER_DEFAULT)
        grad_y = cv.Sobel(img, ddepth, 0, 1, ksize=3, scale=1, delta=2, borderType=cv.BORDER_DEFAULT)
        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)
        img = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    element = cv.getStructuringElement(cv.MORPH_ELLIPSE,
        (2 * params.dilationSize + 1, 2 * params.dilationSize + 1),
        (params.dilationSize, params.dilationSize))
    img = cv.dilate(img, element)

    element = cv.getStructuringElement(cv.MORPH_ELLIPSE,
        (2 * params.erosionSize + 1, 2 * params.erosionSize + 1),
        (params.erosionSize, params.erosionSize))
    img = cv.erode(img, element)

    if params.channelIsolation != "None":
        (B, G, R) = cv.split(img)
        zeros = np.zeros(img.shape[:2], dtype="uint8")
        if params.channelIsolation == "Blue":
            img = cv.merge([B, zeros, zeros])
        elif params.channelIsolation == "Green":
            img = cv.merge([zeros, G, zeros])
        elif params.channelIsolation == "Red":
            img = cv.merge([zeros, zeros, R])

    # (B, G, R) = cv.split(img)
    # zeros = np.zeros(img.shape[:2], dtype="uint8")
    # div = int(params.totalFrames / 3)
    # if params.currentFrame <= div:
    #     img = cv.merge([B, zeros, zeros])
    # elif params.currentFrame <= div * 2:
    #     img = cv.merge([zeros, G, zeros])
    # else:
    #     img = cv.merge([zeros, zeros, R])

    img = cv.convertScaleAbs(img, alpha=params.contrast, beta=params.brightness)

    if params.xCropFlip:
        img1 = img
        img2 = orig_img
    else:
        img1 = orig_img
        img2 = img

    if params.xCropStart  > 0 and params.xCropEnd == img.shape[1]:
        left_img = img1[0:img.shape[0], 0:params.xCropStart]
        right_img = img2[0:img.shape[0], params.xCropStart + 1:params.xCropEnd]
        img = cv.hconcat([left_img, right_img])
    elif params.xCropStart == 0 and params.xCropEnd < img.shape[1]:
        left_img = img2[0:img.shape[0], 0:params.xCropEnd]
        right_img = img1[0:img.shape[0], params.xCropEnd + 1:img.shape[1]]
        img = cv.hconcat([left_img, right_img])
    elif params.xCropStart  > 0 and params.xCropEnd < img.shape[1]:
        left_img = img1[0:img.shape[0], 0:params.xCropStart]
        middle_img = img2[0:img.shape[0], params.xCropStart + 1:params.xCropEnd]
        right_img = img1[0:img.shape[0], params.xCropEnd + 1:img.shape[1]]
        img = cv.hconcat([left_img, middle_img, right_img])

    return img

def getTransformParams(img):
    transformParams = TransformParams()

    x_crop_range = [i for i in range(img.shape[1])]
    x_crop_range.append(img.shape[1])
    transformParams.xCropStart, transformParams.xCropEnd = st.select_slider(
        'x-axis transform cropping',
        options=x_crop_range,
        value=(0, x_crop_range[len(x_crop_range) - 1]))

    transformParams.xCropFlip = st.sidebar.checkbox("Flip x-axis transform crop")
    transformParams.applyGaussianBlur = st.sidebar.checkbox("Apply Gaussian blur")

    transformParams.contrast = st.sidebar.slider("Contrast", 1.0, 5.0, 1.0, 0.5)
    transformParams.brightness = st.sidebar.slider("Brightness", 0, 100, 0)
    
    transformParams.channelIsolation = st.sidebar.radio("Channel Isolation", ("None", "Blue", "Green", "Red"))
    transformParams.transformType = st.sidebar.radio("Transform type", ("Canny", "Laplacian", "Sobel"))

    transformParams.cannyThreshold1 = st.sidebar.slider("Canny threshold 1", 0, 500, 100)
    transformParams.cannyThreshold2 = st.sidebar.slider("Canny threshold 2", 0, 500, 200)

    transformParams.dilationSize = st.sidebar.slider("Dilation size", 0, 10, 0)
    transformParams.erosionSize = st.sidebar.slider("Erosion size", 0, 10, 0)

    return transformParams

def getTotalFrames(input):
    cap = cv.VideoCapture(input)
    res, _ = cap.read()
    totalFrames = 0
    while res:
        totalFrames += 1
        res, _ = cap.read()
    return totalFrames

def labelClass(img, detection, labelThickness):
    boxColor = (255, 0, 255)

    rows = img.shape[0]
    cols = img.shape[1]
    xLeft = int(detection[3] * cols)
    yTop = int(detection[4] * rows)
    xRight = int(detection[5] * cols)
    yBottom = int(detection[6] * rows)

    cv.rectangle(img, (xLeft, yTop), (xRight, yBottom), boxColor, labelThickness)

def main(args):
    cap = cv.VideoCapture(args.input)
    _, img = cap.read()
    
    videoWriter = None
    videoFrame = st.empty()
    writeToFile = st.button("Write to file")
    scoreThreshold = st.slider("Detection accuracy", 0., 1., args.score_threshold)
    labelThickness = st.slider("Label thickness", 1, 50, 5)

    applyTransforms = st.checkbox("Apply transforms")
    if applyTransforms:
        transformParams = getTransformParams(img)
        transformParams.totalFrames = getTotalFrames(args.input)

    cap = cv.VideoCapture(args.input)
    net = cv.dnn.readNetFromTensorflow(modelPath, configPath)

    currentFrame = 0
    while True:
        res, img = cap.read()
        currentFrame += 1

        if res:
            net.setInput(cv.dnn.blobFromImage(img, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
            detections = net.forward()

            if applyTransforms:
                transformParams.currentFrame = currentFrame
                img = applyTransform(img, transformParams)

            for detection in detections[0,0,:,:]:
                score = float(detection[2])
                classId = int(detection[1])
                if args.class_name == classNames[classId] and score > scoreThreshold:
                    labelClass(img, detection, labelThickness)

            videoFrame.image(img, channels="BGR")

            if writeToFile:
                if videoWriter == None:
                    st.info("writing transforms to file...")
                    fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
                    videoWriter = cv.VideoWriter(args.output, fourcc, 30, (img.shape[1], img.shape[0]), True)
                videoWriter.write(img)
        else:
            if writeToFile:
                videoWriter.release()
                videoWriter = None
                st.success("file written successfully!")
                st.stop()
            cap = cv.VideoCapture(args.input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="in.mp4")
    parser.add_argument("-o", "--output", type=str, default="out.avi")
    parser.add_argument("-c", "--class_name", type=str, default="car")
    parser.add_argument("-s", "--score_threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
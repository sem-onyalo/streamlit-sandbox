import cv2 as cv
import streamlit as st
import argparse

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

def labelClass(img, detection):
    boxThickness = 4
    boxColor = (255, 0, 255)

    rows = img.shape[0]
    cols = img.shape[1]

    xLeft = int(detection[3] * cols)
    yTop = int(detection[4] * rows)
    xRight = int(detection[5] * cols)
    yBottom = int(detection[6] * rows)

    cv.rectangle(img, (xLeft, yTop), (xRight, yBottom), boxColor, boxThickness)

def displayCount(img, className, detectionCount):
    cols = img.shape[1]

    detectionCountText = str(detectionCount) + ' ' + className + ('' if detectionCount == 1 else 's')
    detectionCountTextSize = cv.getTextSize(detectionCountText, cv.FONT_HERSHEY_COMPLEX, 3, 4)
    detectionCountTextX = int(round((cols - detectionCountTextSize[0][0]) / 2))
    detectionCountTextY = int(50 + detectionCountTextSize[0][1])
    cv.putText(img, detectionCountText, (detectionCountTextX, detectionCountTextY), cv.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 4, cv.LINE_AA)

def main(args):
    totalFrames = 0
    lastDetectionCount = 0

    cap = cv.VideoCapture(args.input)

    net = cv.dnn.readNetFromTensorflow(modelPath, configPath)

    videoFrame = st.empty()
    scoreThreshold = st.slider("Detection accuracy", 0., 1., args.score_threshold)

    while True:
        res, img = cap.read()

        if res:
            net.setInput(cv.dnn.blobFromImage(img, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
            detections = net.forward()

            detectionCount = 0
            for detection in detections[0,0,:,:]:
                score = float(detection[2])
                classId = int(detection[1])
                if args.class_name == classNames[classId] and score > scoreThreshold:
                    labelClass(img, detection)
                    detectionCount += 1

            if totalFrames % args.skip_frames == 0:
                lastDetectionCount = detectionCount
            
            displayCount(img, args.class_name, lastDetectionCount)

            totalFrames += 1

            videoFrame.image(img, channels="BGR")
        else:
            cap = cv.VideoCapture(args.input)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="in.mp4")
    parser.add_argument("-c", "--class_name", type=str, default="car")
    parser.add_argument("-f", "--skip_frames", type=int, default=5)
    parser.add_argument("-s", "--score_threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args)

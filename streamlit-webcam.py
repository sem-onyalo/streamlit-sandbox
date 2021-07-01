import av
import cv2 as cv
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

class VideoTransformer(VideoTransformerBase):
    def __init__(self) -> None:
        self.cannyThreshold1 = 100
        self.cannyThreshold2 = 200
        self.applyTransforms = False
        self.transformType = 'Canny'
        self.applyGaussianBlur = False
        self.applyDilation = False
        self.applyErosion = False
        self.dilationSize = 0
        self.erosionSize = 0

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")

        if self.applyTransforms:
            if self.applyGaussianBlur:
                img = cv.GaussianBlur(img, (3,3), 0)

            if self.transformType == 'Canny':
                img = cv.Canny(img, self.cannyThreshold1, self.cannyThreshold2)
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            elif self.transformType == 'Laplacian':
                ddepth = cv.CV_32F
                img = cv.Laplacian(img, ddepth)
                img = cv.convertScaleAbs(img)
            elif self.transformType == 'Sobel':
                ddepth = cv.CV_16S
                scale = 1
                delta = 2
                grad_x = cv.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
                grad_y = cv.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
                abs_grad_x = cv.convertScaleAbs(grad_x)
                abs_grad_y = cv.convertScaleAbs(grad_y)
                img = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

            if self.applyDilation:
                element = cv.getStructuringElement(cv.MORPH_ELLIPSE, 
                    (2 * self.dilationSize + 1, 2 * self.dilationSize + 1),
                    (self.dilationSize, self.dilationSize))
                img = cv.dilate(img, element)

            if self.applyErosion:
                element = cv.getStructuringElement(cv.MORPH_ELLIPSE, 
                    (2 * self.erosionSize + 1, 2 * self.erosionSize + 1),
                    (self.erosionSize, self.erosionSize))
                img = cv.erode(img, element)

        return img

ctx = webrtc_streamer(key="Edge Detection Sandbox", video_transformer_factory=VideoTransformer)

if ctx.video_transformer:
    ctx.video_transformer.applyTransforms = st.checkbox("Apply transforms")
    if ctx.video_transformer.applyTransforms:
        ctx.video_transformer.transformType = st.radio("Transform type", ('Canny', 'Laplacian', 'Sobel'))
        ctx.video_transformer.applyGaussianBlur = st.checkbox("Apply Gaussian blur")
        ctx.video_transformer.applyDilation = st.checkbox("Apply dilation")
        ctx.video_transformer.applyErosion = st.checkbox("Apply erosion")

        if ctx.video_transformer.transformType == 'Canny':
            ctx.video_transformer.cannyThreshold1 = st.slider("Canny threshold 1", 0, 500, 100)
            ctx.video_transformer.cannyThreshold2 = st.slider("Canny threshold 2", 0, 500, 200)
            
        if ctx.video_transformer.applyDilation:
            ctx.video_transformer.dilationSize = st.slider("Dilation size", 0, 10, 0)

        if ctx.video_transformer.applyErosion:
            ctx.video_transformer.erosionSize = st.slider("Erosion size", 0, 10, 0)

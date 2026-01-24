import cv2
import torch
import fastcv


img = cv2.imread("../artifacts/dog.png", cv2.IMREAD_GRAYSCALE)
#img = cv2.imread("../artifacts/grayscale.jpg", cv2.IMREAD_GRAYSCALE)

img_tensor = torch.from_numpy(img).cuda()

img_tresh_opencv = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
cv2.imwrite("output_adaptive_thresh_opencv.jpg", img_tresh_opencv)

print("saved adaptive thresholded image.")

img_thresh_our = fastcv.adaptive_thresh(img_tensor, 5, 2, 255)
img_thresh_our_np = img_thresh_our.squeeze(-1).cpu().numpy()
cv2.imwrite("output_adaptive_thresh_our.jpg", img_thresh_our_np)

print("saved adaptive thresholded image.")

img_diff = cv2.absdiff(img_tresh_opencv, img_thresh_our_np)
cv2.imwrite("output_adaptive_thresh_diff.jpg", img_diff)

print("saved adaptive thresholded difference image.")
import cv2
import torch
import fastcv

img = cv2.imread("../artifacts/grayscale.jpg", cv2.IMREAD_GRAYSCALE)
img_tensor = torch.from_numpy(img).cuda()
gray_tensor = fastcv.adaptive_thresh_meanC_thrustSYNC(img_tensor, 3, 5, 255)
gray_np = gray_tensor.squeeze(-1).cpu().numpy()
cv2.imwrite("output_adaptive_thresh_meanC_thrustSYNC.jpg", gray_np)

print("saved adaptive thresholded image.")
import time

import cv2
import torch
import fastcv
import numpy as np

def benchamrk_adaptive_thresh(sizes=[1024, 2048, 4096], runs=50):
    results = []

    for size in sizes:
        print(f"\n=== Benchmarking {size}x{size} image ===")

        # image generation
        img_np = np.random.randint(0, 2, (size, size), dtype=np.uint8) * 255 # to do
        img_torch = torch.from_numpy(img_np).cuda()

        # cv2 adaptive meanC threshold
        start = time.perf_counter()
        for _ in range(runs):
            _ = cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)

        end = time.perf_counter()
        cv_time = (end - start) / runs * 1000  # ms per run

        # fastcv adaptive meanC thresh - thrust ASYNC
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = fastcv.adaptive_thresh_meanC_thrustASYNC(img_torch, 5, 2, 255)
        torch.cuda.synchronize()
        end = time.perf_counter()
        fc_thrustASYNC_time = (end - start) / runs * 1000  # ms per run

        # fastcv adaptive meanC thresh - thrust SYNC
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = fastcv.adaptive_thresh_meanC_thrustSYNC(img_torch, 5, 2, 255)
        torch.cuda.synchronize()
        end = time.perf_counter()
        fc_thrustSYNC_time = (end - start) / runs * 1000  # ms per run

        # Displaying results
        results.append((size, cv_time, fc_thrustASYNC_time, fc_thrustSYNC_time))
        print(f"OpenCV (CPU): {cv_time:.4f} ms | fastcv - thrust ASYNC (CUDA): {fc_thrustASYNC_time:.4f} ms | fastcv - thrust SYNC (CUDA): {fc_thrustSYNC_time:.4f} ms")

    return results

if __name__ == "__main__":
    results = benchamrk_adaptive_thresh()
    print("\n=== Final Results ===")
    print("Size\t\tOpenCV (CPU)\tfastcv - thrust ASYNC (CUDA)\tfastcv - thrust SYNC (CUDA)")
    for size, cv_time, fc_thrustASYNC_time, fc_thrustSYNC_time in results:
        print(f"{size}x{size}\t{cv_time:.4f} ms\t    {fc_thrustASYNC_time:.4f} ms\t                    {fc_thrustSYNC_time:.4f} ms")

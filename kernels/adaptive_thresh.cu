#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"


__global__ void adaptive_threshKernel(unsigned char *Pixel_in,
                                        unsigned char *Pixel_out,
                                        int width,
                                        int height,
                                        int maxValue,
                                        int blockSize,
                                        int constance) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int num_of_pixels = 0;

    if (col < width && row < height){

        float sum = 0;
        for (int t_row = -blockSize; t_row <= blockSize; t_row++) {
            for (int t_col = -blockSize; t_col <= blockSize; t_col++) {

                int current_row = row + t_row;
                int current_col = col + t_col;
                if (current_row >= 0 && current_col >= 0 && current_row < height && current_col < width){
                    unsigned char neighbour_pixel = Pixel_in[current_row * width + current_col];
                    sum += neighbour_pixel;
                    num_of_pixels++;
                }
            }
        }

        float mean = sum / num_of_pixels;
        int threshold = int(mean) - constance;

        if (Pixel_in[row * width + col] > threshold) {
            Pixel_out[row * width + col] = maxValue;
        }
        else {
            Pixel_out[row * width + col] = 0;
        }
    }

}

torch::Tensor adaptive_thresh(torch::Tensor img, int filterSize, int constance, int maxValue) {
    assert(img.device().type() == torch::kCUDA);
    assert(img.dtype() == torch::kByte);

    const auto height = img.size(0);
    const auto width = img.size(1);

    dim3 dimBlock = getOptimalBlockDim(width, height);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));

    auto result = torch::empty({height, width},
                              torch::TensorOptions().dtype(torch::kByte).device(img.device()));

    adaptive_threshKernel<<<dimGrid, dimBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        img.data_ptr<unsigned char>(),
        result.data_ptr<unsigned char>(),
        width, height, maxValue, filterSize, constance);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}
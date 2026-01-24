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
                                        int filterSize,
                                        int constance) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int half_filterSize = filterSize / 2;


    extern __shared__ unsigned char tile[];

    int nTileThreads = blockDim.x * blockDim.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int tileWidth  = blockDim.x + (half_filterSize * 2);
    int tileHeight = blockDim.y + (half_filterSize * 2);
    int nTile = tileWidth * tileHeight;

    int numberOfLooping = 1 + nTile/nTileThreads;

    int base_col = blockIdx.x * blockDim.x - half_filterSize;
    int base_row = blockIdx.y * blockDim.y - half_filterSize;

    for (int i = 0; i < numberOfLooping; i++)
    {
        int tilePixelIndex = i * nTileThreads + tid;

        if (tilePixelIndex < nTile){
            int tileGlobalIndex = (base_row * width + base_col) + (tilePixelIndex / tileWidth ) * width + (tilePixelIndex % tileWidth );

            int currentRow = base_row + (tilePixelIndex / tileWidth);
            int currentCol = base_col + (tilePixelIndex % tileWidth);

            if (currentRow >= 0 && currentRow < height && currentCol >= 0 && currentCol < width) {
                tile[tilePixelIndex] = Pixel_in[tileGlobalIndex];
            } else {
                tile[tilePixelIndex] = 0;
            }
        }
    }

    __syncthreads();




    if (col < width && row < height){

        float sum = 0;
        int num_of_pixels = 0;

        for (int t_row = -half_filterSize; t_row <= half_filterSize; t_row++) {
            for (int t_col = -half_filterSize; t_col <= half_filterSize; t_col++) {

                int sharedMemCol = threadIdx.x + half_filterSize + t_col;
                int sharedMemRow = threadIdx.y + half_filterSize + t_row;
                int tileIndex = sharedMemRow * tileWidth + sharedMemCol;

                int currentRow = row + t_row;
                int currentCol = col + t_col;

                if (currentRow >= 0 && currentCol >= 0 && currentRow < height && currentCol < width){
                    sum += tile[tileIndex];
                    num_of_pixels++;
                }
            }
        }

        float mean = sum / num_of_pixels;
        int threshold = int(mean+0.5) - constance;

        int center_idx = (threadIdx.y + half_filterSize) * tileWidth + (threadIdx.x + half_filterSize);

        if (tile[center_idx] > threshold) {
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
    size_t shared_memorySize = (dimBlock.x + int(filterSize / 2) * 2) * (dimBlock.y + int(filterSize / 2) * 2);

    auto result = torch::empty({height, width},
                              torch::TensorOptions().dtype(torch::kByte).device(img.device()));

    adaptive_threshKernel<<<dimGrid, dimBlock, shared_memorySize, at::cuda::getCurrentCUDAStream()>>>(
        img.data_ptr<unsigned char>(),
        result.data_ptr<unsigned char>(),
        width, height, maxValue, filterSize, constance);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}
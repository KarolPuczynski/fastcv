#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/tabulate.h>
#include "nvtx3.hpp"
#include <thrust/iterator/counting_iterator.h>

#include "utils.cuh"

__global__ void adaptive_thresh_thrustASYNC_Kernel(int *row_summed_pixels, unsigned char *Pixel_in, unsigned char *Pixel_out,
                                       int width, int height, unsigned char maxValue, int filterSize, int constance) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int half_filterSize = filterSize / 2;
    int pad = half_filterSize + 1;

    extern __shared__ int tile[];

    int nTileThreads = blockDim.x * blockDim.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int tileWidth  = blockDim.x + (pad * 2);
    int tileHeight = blockDim.y + (pad * 2);
    int nTile = tileWidth * tileHeight;

    int numberOfLooping = 1 + nTile/nTileThreads;

    int base_col = blockIdx.x * blockDim.x - pad;
    int base_row = blockIdx.y * blockDim.y - pad;

    // allocating shared memory
    for (int i = 0; i < numberOfLooping; i++)
    {
        int tilePixelIndex = i * nTileThreads + tid;

        if (tilePixelIndex < nTile){
            int tileGlobalIndex = (base_row * width + base_col) + (tilePixelIndex / tileWidth ) * width + (tilePixelIndex % tileWidth );

            int currentRow = base_row + (tilePixelIndex / tileWidth);
            int currentCol = base_col + (tilePixelIndex % tileWidth);

            if (currentRow >= 0 && currentRow < height && currentCol >= 0 && currentCol < width) {
                tile[tilePixelIndex] = row_summed_pixels[tileGlobalIndex];
            } else {
                tile[tilePixelIndex] = 0;
            }
        }
    }

    __syncthreads(); //ensuring that all the memory is allocated

    if (col < width && row < height){ //check if we are not out of bounds with threads for image
        float sum = 0;
        int num_of_pixels = 0;

        for (int t_row = -half_filterSize; t_row <= half_filterSize; t_row++) {

            int currentTileRow = (threadIdx.y + pad) + t_row;

            int globalRow = row + t_row;

            if (globalRow >= 0 && globalRow < height) {

                int edge_left_col = col - half_filterSize - 1;
                int edge_right_col = col + half_filterSize;

                if (edge_right_col >= width) {
                    edge_right_col = width - 1;
                }

                int tile_idx_right = edge_right_col - base_col;
                int tile_idx_left = edge_left_col - base_col;

                if (edge_left_col >= 0) {
                    sum += (tile[currentTileRow * tileWidth + tile_idx_right] - tile[currentTileRow * tileWidth + tile_idx_left]);
                    num_of_pixels += (edge_right_col - edge_left_col);
                }
                else {
                    sum += tile[currentTileRow * tileWidth + tile_idx_right];
                    num_of_pixels += edge_right_col;
                }
            }
        }
        float mean = sum / num_of_pixels;
        int threshold = int(mean+0.5) - constance;

        int global_idx = row * width + col;

        if (Pixel_in[global_idx] > threshold) {
            Pixel_out[global_idx] = maxValue;
        }
        else {
            Pixel_out[global_idx] = 0;
        }
    }
}

struct UcharToInt {
    __host__ __device__
    int operator()(unsigned char x) const {
        return int(x);
    }
};

struct FindRowIndex {
    int width;
    FindRowIndex(int w) {
        width = w;
    }

    __host__ __device__
    int operator()(int i) const {
        return i / width;
    }
};

torch::Tensor adaptive_thresh_meanC_thrustASYNC(torch::Tensor img, int filterSize, int constance, int maxValue) {
    nvtxRangePushA("adaptive thresh meanC start");
    assert(img.device().type() == torch::kCUDA);
    assert(img.dtype() == torch::kByte);
    assert(filterSize % 2 == 1);

    const int height = img.size(0);
    const int width = img.size(1);

    dim3 dimBlock = getOptimalBlockDim(width, height);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));

    int pad = int(filterSize / 2) + 1;
    size_t shared_memorySize = (dimBlock.x + pad * 2) * (dimBlock.y + pad * 2) * sizeof(int);

    nvtxRangePushA("Memory allocation");
    auto output_img = torch::empty({height, width}, torch::TensorOptions().dtype(torch::kByte).device(img.device()));
    auto row_summed_pixels = torch::empty({width * height}, torch::TensorOptions().dtype(torch::kInt).device(img.device()));
    nvtxRangePop();

    int* row_summed_pixels_ptr = row_summed_pixels.data_ptr<int>();
    unsigned char* input_img_ptr = img.data_ptr<unsigned char>();
    unsigned char* output_img_ptr = output_img.data_ptr<unsigned char>();

    // wrapping raw pointers with a device_ptr to use it in thrust algorithms (tabulate, inclusive_scan_by_key)
    thrust::device_ptr<int> dev_row_summed_pixels_ptr(row_summed_pixels_ptr);
    thrust::device_ptr<unsigned char> dev_input_img_ptr(input_img_ptr);
    // transform dev_input_img_ptr from uchar to int
    auto img_int_iter = thrust::make_transform_iterator(dev_input_img_ptr, UcharToInt());

    thrust::counting_iterator<int> row_iterator(0);
    auto row_index_iterator = thrust::make_transform_iterator(row_iterator, FindRowIndex(width));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nvtxRangePushA("thrust functions executions");
    thrust::inclusive_scan_by_key(thrust::cuda::par.on(stream),
                                  row_index_iterator, row_index_iterator + (width * height),
                                  img_int_iter,
                                  dev_row_summed_pixels_ptr);
    nvtxRangePop();

    nvtxRangePushA("Kernel execution");
    adaptive_thresh_thrustASYNC_Kernel<<<dimGrid, dimBlock, shared_memorySize, stream>>>(
        row_summed_pixels_ptr,
        input_img_ptr,
        output_img_ptr,
        width, height, maxValue, filterSize, constance);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    nvtxRangePop();

    nvtxRangePop();

    return output_img;
}
#include <stdio.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        return -1; \
    }


#define KERNEL_SIZE 3


__constant__ float kernal[KERNEL_SIZE*KERNEL_SIZE] = {
    1.0f/9 , 1.0f/9 , 1.0f/9,
    1.0f/9 , 1.0f/9 , 1.0f/9,

    1.0f/9 , 1.0f/9 , 1.0f/9
};




__global__ void change_image2d(float*input , float*output , int width , int height){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < height && col < width) {
        float sum = 0.0f;
        int kHalf = KERNEL_SIZE / 2;

         for(int i = -kHalf; i <= kHalf; i++) {
            for(int j = -kHalf; j <= kHalf; j++) {
                int r = min(max(row + i, 0), height - 1); 
                int c = min(max(col + j, 0), width - 1);
                sum += input[r * width + c] * kernal[(i+kHalf) * KERNEL_SIZE + (j+kHalf)];
            }
        }

        output[row * width + col] = sum;
    }
}




int main(){
    cv::Mat img = cv::imread("E:/cuda_programs/test.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        printf("Error: Image not found!\n");
        return -1;
    }

    int width = img.cols;
    int height = img.rows;
    printf("Image loaded: %dx%d\n", width, height);

    cv::Mat imgFloat;
    img.convertTo(imgFloat, CV_32F);

    size_t size = width * height * sizeof(float);

    float* h_input = imgFloat.ptr<float>(0);
    float* h_output = (float*)malloc(size);

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // COPY KERNEL TO CONSTANT MEMORY
    float h_kernel[9] = {
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9
    };
    CUDA_CHECK(cudaMemcpyToSymbol(kernal, h_kernel, sizeof(h_kernel)));

    dim3 threads(16,16);
    dim3 blocks(
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y
    );

    change_image2d<<<blocks, threads>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    cv::Mat outImg(height, width, CV_32F, h_output);
    outImg.convertTo(outImg, CV_8U);

    cv::imshow("CUDA Output", outImg);
    cv::waitKey(0);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output);

    printf("Done\n");
}
// opencv_core4.dll
// opencv_imgproc4.dll
// opencv_imgcodecs4.dll
// opencv_highgui4.dll
// opencv_videoio4.dll
#include <stdio.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>


#define WIDTH 1024
#define HEIGHT 1024
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
    cv::Mat img = cv::imread("your_image.jpg", cv::IMREAD_GRAYSCALE);
    if(img.empty()) {
        printf("Error: Image not found!\n");
        return -1;
    }

    size_t size = WIDTH*HEIGHT * sizeof(float);

    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    for (int i = 0; i < WIDTH*HEIGHT; i++)
    {
       h_input[i] = (float)(i%256); 
    }

    float *d_input ,  *d_output;
    cudaMalloc(&d_input , size);
    cudaMalloc(&d_output , size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);


    dim3 threads(16,16);
    dim3 blocks((WIDTH + threads.x - 1)/threads.x, (HEIGHT + threads.y - 1)/threads.y);

    change_image2d<<<blocks, threads>>>(d_input, d_output, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    printf("Output[0][0] = %f\n", h_output[0]);
    printf("Output[512][512] = %f\n", h_output[512*WIDTH + 512]);

 
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    


}
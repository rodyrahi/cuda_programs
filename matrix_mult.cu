#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

float *h_m_A;
float *h_m_B;

float *h_m_C;

__global__ void matrixmult(float *A , float *B , float *C ){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

     if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
        {
            sum +=A[row*N+k] * B[k*N + col];
        }

        C[row*N + col] = sum;
        
    }

}

__global__ void matrixadd(float *A, float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        C[row * N + col] = A[row * N + col] + B[row * N + col];
    }
}



void cpu_matmul(float *A, float *B, float *C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}




int main(){
    size_t size = N*N * sizeof(float);

    h_m_A = (float*)malloc(size);
    h_m_B = (float*)malloc(size);
    h_m_C = (float*)malloc(size);


    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_m_A[i*N+j] = 5.0f;
            h_m_B[i*N+j] = 2.0f; 
        }
        
    }
    

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_m_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_m_B, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);  
    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (N + threads.y - 1) / threads.y
    );

    matrixmult<<<blocks, threads>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_m_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("MatMul C[0][0] = %f\n", h_m_C[0]);
    printf("MatMul C[1][1] = %f\n", h_m_C[1*N + 1]);


    matrixadd<<<blocks, threads>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_m_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("MatAdd C[0][0] = %f\n", h_m_C[0]);
    printf("MatAdd C[1][1] = %f\n", h_m_C[1*N + 1]);


    //// testing with the cpu 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // matrixmult<<<blocks, threads>>>(d_A, d_B, d_C);
    matrixadd<<<blocks, threads>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("GPU MatMul time: %f ms\n", ms);



    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_m_A);
    free(h_m_B);
    free(h_m_C);

}

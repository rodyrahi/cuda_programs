#include <stdio.h>
#include <cuda_runtime.h>

#define N 1000000  
int *d_arr;


__global__ void initarray(int *arr ){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N){
        arr[idx] = idx+1;
    }
}



__global__ void testcuda(int *arr ){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    

    if(idx <N){
        for(int i =0 ; i<1 ; i++){
            arr[idx] *=2; 
        }
        // printf("Thread idx=%d, value=%d\n", idx, arr[idx]);
        // printf("Thread idx = %d\n",idx);
    }



}

int main(){
    
    
    int threadsPerBlock = 1024;

    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc((void**)&d_arr , N*sizeof(int));


    initarray<<<blocks , threadsPerBlock>>>(d_arr );
    cudaDeviceSynchronize();


    testcuda<<<blocks,threadsPerBlock>>>(d_arr);
    cudaDeviceSynchronize();


    int *h_arr = (int*)malloc(N * sizeof(int));
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // printf("Final array: ");
    // for(int i = 0; i < N; i++) {
    //     printf("%d ", h_arr[i]);
    // }
    printf("done");
    
    free(h_arr); 
    cudaFree(d_arr);
    return 0;



}
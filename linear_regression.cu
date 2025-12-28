#include <stdio.h>
#include <cuda_runtime.h>


__global__ void compute_gradient(float *x , float *y , float *dw , float*db , float *w , float*b , int n){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i<n)
    {
        // formula:
        // y=wx+b
        // dw=∑(ypred​−y)⋅x
        // db=∑(ypred​−y)
        // ---------------

        // y=wx+b
        float y_pred = (*w)*x[i] + (*b) ;
        // dw=∑(<<<<< | ypred​−y | >>>>>)⋅x
        float loss =  y_pred - y[i] ;
        
        // dw=∑(ypred​−y)⋅x
        atomicAdd(dw , loss*x[i]);
        
        // db=∑(ypred​−y)
        atomicAdd(db , loss);

    }
    

}


__global__ void update_params(float *w , float *b , float *dw , float*db , float lr , int n){
    if (threadIdx.x == 0)
    {
        *w -= lr * (*dw / n);
        *b -= lr * (*db / n);
        *dw = 0.0f;
        *db = 0.0f;
    }
    

}

int main(){
    const int N = 5;
    float data_x[N] = {1, 2, 3, 4, 5};
    float data_y[N] = {2, 4, 6, 8, 10};


    float *d_x, *d_y;
    float *d_w, *d_b, *d_dw, *d_db;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_w, sizeof(float));
    cudaMalloc(&d_b, sizeof(float));
    cudaMalloc(&d_dw, sizeof(float));
    cudaMalloc(&d_db, sizeof(float));




        cudaMemcpy(d_x, data_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, data_y, N * sizeof(float), cudaMemcpyHostToDevice);

    float w = 0.0f, b = 0.0f;
    cudaMemcpy(d_w, &w, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_dw, 0, sizeof(float));
    cudaMemset(d_db, 0, sizeof(float));
    
    int epochs = 1000;
    float lr = 0.01f;

    for (int i = 0; i < epochs; i++)
    {
        compute_gradient<<< 1 , 256  >>> (d_x, d_y, d_dw, d_db, d_w, d_b, N);
        update_params<<<1, 1>>>(d_w, d_b, d_dw, d_db, lr, N);

        cudaMemcpy(&w, d_w, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&b, d_b, sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    printf("Trained Model: y = %.3fx + %.3f\n", w, b);


    cudaFree(d_x); cudaFree(d_y);
    cudaFree(d_w); cudaFree(d_b);
    cudaFree(d_dw); cudaFree(d_db);


}
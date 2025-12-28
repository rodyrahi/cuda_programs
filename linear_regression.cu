#include <stdio.h>
#include <cuda_runtime.h>

__global__ void predict(
    float *x,       // (M × F) input matrix
    float *w,       // (F) trained weights
    float *b,       // scalar bias
    float *y_out,   // (M) output predictions
    int m,          // number of samples
    int f           // number of features
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m) {
        float y = *b;
        for (int j = 0; j < f; j++) {
            y += w[j] * x[i * f + j];
        }
        y_out[i] = y;
    }
}



__global__ void compute_gradient(float *x , float *y , float *dw , float*db , float *w , float*b , int n , int f){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i<n)
    {
        // formula:
        // y=wx+b
        // dw=∑(ypred​−y)⋅x
        // db=∑(ypred​−y)
        // ---------------

        float y_pred = *b;

        for (int j = 0; j < f; j++)
        {
            
            y_pred += w[j]*x[i * f +j ] ;
        }
        

        // y=wx+b
        // dw=∑(<<<<< | ypred​−y | >>>>>)⋅x
        float loss =  y_pred - y[i] ;
        
        for (int j = 0; j < f; j++) {
            atomicAdd(&dw[j], loss * x[i * f + j]);
        }
        
        // db=∑(ypred​−y)
        atomicAdd(db , loss);   

    }
    

}


__global__ void update_params(float *w , float *b , float *dw , float*db , float lr , int n , int f){
    if (threadIdx.x == 0)
    {   
        for (int i = 0; i < f; i++)
        {
            /* code */
            w[i] -= lr * (dw[i] / n);
            dw[i] = 0.0f;
        }
        
        *b -= lr * (*db / n);
        *db = 0.0f;
    }
    

}


int main(){
    const int N = 5;
    const int F = 4;


    float data_x[N*F] = {
        1, 2, 3, 4,
        2, 3, 4, 5,   
        3, 4, 5, 6,   
        4, 5, 6, 7, 
        5, 6, 7, 8  
        
    };
    float data_y[N] = {2, 4, 6, 8, 10};


    float *d_x, *d_y, *d_w, *d_b, *d_dw, *d_db;
    cudaMalloc(&d_x, N * F * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_w, F * sizeof(float));
    cudaMalloc(&d_b, sizeof(float));
    cudaMalloc(&d_dw, F * sizeof(float));
    cudaMalloc(&d_db, sizeof(float));




    float w[F] = {0, 0, 0, 0};
    float b = 0;

    cudaMemcpy(d_x, data_x, N * F * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, data_y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, F * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_dw, 0, F * sizeof(float));
    cudaMemset(d_db, 0, sizeof(float));



    
    int epochs = 1000;
    float lr = 0.01f;

    for (int i = 0; i < epochs; i++)
    {
        compute_gradient<<< 1 , 256  >>> (d_x, d_y, d_dw, d_db, d_w, d_b, N, F);
        update_params<<<1, 1>>>(d_w, d_b, d_dw, d_db, lr, N, F);

        cudaMemcpy(w, d_w, F * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&b, d_b, sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    printf("Trained Model: y = %.3f + %.3fx1 + %.3fx2 + %.3fx3 + %.3fx4\n", b, w[0], w[1], w[2], w[3]);


    const int M = 3;  

    float new_x[M * F] = {
        6, 7, 8, 9,
        7, 8, 9, 10,
        8, 9, 10, 11
    };

    float *d_x_new, *d_y_pred;
    float y_pred[M];

    cudaMalloc(&d_x_new, M * F * sizeof(float));
    cudaMalloc(&d_y_pred, M * sizeof(float));

    cudaMemcpy(d_x_new, new_x, M * F * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (M + threads - 1) / threads;

    predict<<<blocks, threads>>>(d_x_new, d_w, d_b, d_y_pred, M, F);
    cudaDeviceSynchronize();


    cudaMemcpy(y_pred, d_y_pred, M * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nPredictions on new data:\n");
    for (int i = 0; i < M; i++) {
        printf("Sample %d -> y_pred = %.3f\n", i, y_pred[i]);
    }


    cudaFree(d_x_new);
    cudaFree(d_y_pred);



    cudaFree(d_x); cudaFree(d_y);
    cudaFree(d_w); cudaFree(d_b);
    cudaFree(d_dw); cudaFree(d_db);


}
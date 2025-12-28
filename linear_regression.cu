#include <stdio.h>
#include <cuda_runtime.h>


// __global__ void compute_gradient(float *x , float *y , float *dw , float*db , float *w , float*b , int n){

//     int i = blockIdx.x * blockDim.x + threadIdx.x;

//     if (i<n)
//     {
//         // formula:
//         // y=wx+b
//         // dw=∑(ypred​−y)⋅x
//         // db=∑(ypred​−y)
//         // ---------------

//         // y=wx+b
//         float y_pred = (*w)*x[i] + (*b) ;
//         // dw=∑(<<<<< | ypred​−y | >>>>>)⋅x
//         float loss =  y_pred - y[i] ;
        
//         // dw=∑(ypred​−y)⋅x
//         atomicAdd(dw , loss*x[i]);
        
//         // db=∑(ypred​−y)
//         atomicAdd(db , loss);

//     }
    

// }





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


// __global__ void update_params(float *w , float *b , float *dw , float*db , float lr , int n){
//     if (threadIdx.x == 0)
//     {
//         *w -= lr * (*dw / n);
//         *b -= lr * (*db / n);
//         *dw = 0.0f;
//         *db = 0.0f;
//     }
    

// }

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



    cudaFree(d_x); cudaFree(d_y);
    cudaFree(d_w); cudaFree(d_b);
    cudaFree(d_dw); cudaFree(d_db);


}
#include <stdio.h>
#include <cuda_runtime.h>

// CUDA Kernel function to add the elements of two arrays on the GPU
__global__ void softmax(float *x, int N)
{
    // Index for each thread
    int index = threadIdx.x;
    float max_val = 0.0f;
    float sum_exp = 0.0f;

    // Find maximum value from input array
    for (int i = 0; i < N; i++)
    {
        if (x[i] > max_val)
            max_val = x[i];
    }

    // Calculate the sum of exponentials
    for (int i = 0; i < N; i++)
    {
        sum_exp += expf(x[i] - max_val);
    }

    // Finally, apply the softmax function for the particular thread
    x[index] = expf(x[index] - max_val) / sum_exp;
}

// Main function
int main()
{
    int N = 5;
    float h_x[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float *d_x;

    // Allocate the memory on the GPU
    cudaMalloc(&d_x, N*sizeof(float));

    // Copy the array 'h_x' to the GPU
    cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice);

    // Call the Kernel function
    softmax<<<1, N>>>(d_x, N);

    // Copy back the result array to the CPU
    cudaMemcpy(h_x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Free the memory
    cudaFree(d_x);
    printf("Softmax output:\n");
    for (int i = 0; i < N; i++)
    {
        printf("%.6f ", h_x[i]);
    }
    printf("\n");

    return 0;
}

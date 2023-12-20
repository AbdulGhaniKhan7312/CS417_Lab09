#include <stdio.h>

#define TILE_SIZE 16 // Define the tile size for shared memory

// Function to perform matrix multiplication on CPU for verification
void matrixMulCPU(int *A, int *B, int *C, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int temp = 0;
            for (int k = 0; k < size; ++k) {
                temp += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = temp;
        }
    }
}

__global__ void matrixMulShared(int *A, int *B, int *C, int size) {
    __shared__ int sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ int sharedB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    int temp = 0;

    for (int i = 0; i < size / TILE_SIZE; ++i) {
        // Load tiles of A and B into shared memory
        sharedA[ty][tx] = A[row * size + i * TILE_SIZE + tx];
        sharedB[ty][tx] = B[(i * TILE_SIZE + ty) * size + col];

        __syncthreads();

        // Compute partial result
        for (int k = 0; k < TILE_SIZE; ++k) {
            temp += sharedA[ty][k] * sharedB[k][tx];
        }

        __syncthreads();
    }

    C[row * size + col] = temp;
}

int main() {
    const int size = 3; // Change this value to test with different matrix sizes
    const int matrixSize = size * size;
    const int bytes = matrixSize * sizeof(int);

    int A[size][size] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int B[size][size] = {{9, 1, 5}, {1, 2, 5}, {2, -5, 2}};
    int C[size][size] = {0};

    int *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);

    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(size / TILE_SIZE, size / TILE_SIZE);

    matrixMulShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);

    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verification using CPU
    int verifyC[size][size] = {0};
    matrixMulCPU(*A, *B, *verifyC, size);

    // Compare results
    bool success = true;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (C[i][j] != verifyC[i][j]) {
                success = false;
                break;
            }
        }
    }

    if (success) {
        printf("GPU and CPU results match!\n");
    } else {
        printf("GPU and CPU results do not match!\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

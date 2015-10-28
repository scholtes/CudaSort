#include <stdlib.h>
#include <stdio.h>

#define TEST_SIZE 35
#define BLOCK_WIDTH 4
#define CEILING_DIVIDE(X, Y) (1 + (((X) - 1) / (Y)))

// Computes a blockwise exclusive sum scan
__global__ void partialScan(unsigned int *d_in,
                            unsigned int *d_out,
                            unsigned int *d_total,
                            size_t n)
{
    __shared__ unsigned int temp[BLOCK_WIDTH];
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int index = BLOCK_WIDTH * bx + tx;
    size_t temp_size = n % BLOCK_WIDTH < BLOCK_WIDTH ? n % BLOCK_WIDTH : BLOCK_WIDTH;

    if(index < n) {
        temp[tx] = d_in[index];
    }
    __syncthreads();

    // Perform the actual scan
    for(int offset = 1; offset < temp_size; offset <<= 1) {
        if(tx + offset < temp_size) {
            temp[tx + offset] += temp[tx];
        }
        __syncthreads();
    }

    // Shift when copying the result so as to make it an exclusive scan
    if(index + 1 < n) {
        d_out[index + 1] = temp[tx];
    }
    d_out[0] = 0;

    // Store the total sum of each block
    d_total[bx] = temp[temp_size - 1];
}

// Compute a map on a partial scan to create a total scan from
__global__ void mapScan(unsigned int *d_array, unsigned int *d_total, size_t n) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int index = BLOCK_WIDTH * bx + tx;

    if(index < n) {
        d_array[index] += d_total[bx];
    }
}


void totalScan(unsigned int *d_in, unsigned int *d_out, size_t n) {
    size_t numBlocks = CEILING_DIVIDE(n, BLOCK_WIDTH);
    unsigned int *d_total;
    cudaMalloc(&d_total, sizeof(unsigned int) * numBlocks);

    partialScan<<<numBlocks, BLOCK_WIDTH>>>(d_in, d_out, d_total, n);

    if(numBlocks > 1) {
        totalScan(d_total, d_total, numBlocks);
        mapScan<<<numBlocks, BLOCK_WIDTH>>>(d_out, d_total, n);
    }

    cudaFree(d_total);
}


int main(int argc, char **argv) {

    return 0;
}

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

    if(index < n) {
        temp[tx] = d_in[index];
    } else { temp[tx] = 0; }
    __syncthreads();

    // Perform the actual scan
    for(int offset = 1; offset < BLOCK_WIDTH; offset <<= 1) {
        if(tx + offset < BLOCK_WIDTH) {
            temp[tx + offset] += temp[tx];
        }
        __syncthreads();
    }

    // Shift when copying the result so as to make it an exclusive scan
    if(tx +1 < BLOCK_WIDTH && index + 1 < n) {
        d_out[index + 1] = temp[tx];
    }
    d_out[0] = 0;

    // Store the total sum of each block
    d_total[bx] = temp[BLOCK_WIDTH - 1];
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


// Compute exclusive sum scan for arbitrary sized array (device pointers as input)
void totalScan(unsigned int *d_in, unsigned int *d_out, size_t n) {
    size_t numBlocks = CEILING_DIVIDE(n, BLOCK_WIDTH);
    unsigned int *d_total;
    cudaMalloc(&d_total, sizeof(unsigned int) * numBlocks);
    cudaMemset(d_total, 0, sizeof(unsigned int) * numBlocks);

    partialScan<<<numBlocks, BLOCK_WIDTH>>>(d_in, d_out, d_total, n);

    if(numBlocks > 1) {
        unsigned int *d_total_scanned;
        cudaMalloc(&d_total_scanned, sizeof(unsigned int) * numBlocks);
        cudaMemset(d_total_scanned, 0, sizeof(unsigned int) * numBlocks);

        totalScan(d_total, d_total_scanned, numBlocks);

        mapScan<<<numBlocks, BLOCK_WIDTH>>>(d_out, d_total_scanned, n);

        cudaFree(d_total_scanned);
    }

    cudaFree(d_total);
}


////////////////////////////////////////////////////////////////////////////////


// Wrapper for totalScan (host pointers as input)
void totalScanHost(unsigned int *h_in, unsigned int *h_out, size_t n) {
    unsigned int *d_in;
    unsigned int *d_out;
    size_t memsize = sizeof(unsigned int) * n;

    cudaMalloc(&d_in, memsize);
    cudaMalloc(&d_out, memsize);

    cudaMemcpy(d_in, h_in, memsize, cudaMemcpyHostToDevice);

    totalScan(d_in, d_out, n);

    cudaMemcpy(h_out, d_out, memsize, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}


int main(int argc, char **argv) {
    unsigned int *h_in;
    unsigned int *h_out;
    size_t memsize = sizeof(unsigned int) * TEST_SIZE;

    h_in = (unsigned int*)malloc(memsize);
    h_out = (unsigned int*)malloc(memsize);

    // Test values 1 .. TEST_SIZE
    for(int i=0; i<TEST_SIZE; i++){ h_in[i] = i+1; }

    // Compute
    totalScanHost(h_in, h_out, TEST_SIZE);

    // Print input
    printf("h_in = [ ");
    for(int i=0; i<TEST_SIZE; i++){ printf("%d ", h_in[i]); }
    printf("];\n");

    // Print output
    printf("h_out = [ ");
    for(int i=0; i<TEST_SIZE; i++){ printf("%d ", h_out[i]); }
    printf("];\n");

    free(h_in);
    free(h_out);
    return 0;
}

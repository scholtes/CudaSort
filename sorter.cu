#include <stdlib.h>
#include <stdio.h>

#define TEST_SIZE 35
#define RAND_RANGE 10
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
    }
    __syncthreads();

    // Perform the actual scan
    for(int offset = 1; offset < BLOCK_WIDTH; offset <<= 1) {
        if(tx + offset < BLOCK_WIDTH) {
            temp[tx + offset] += temp[tx];
        }
        __syncthreads();
    }

    // Shift when copying the result so as to make it an exclusive scan
    if(tx + 1 < BLOCK_WIDTH) {
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

    partialScan<<<numBlocks, BLOCK_WIDTH>>>(d_in, d_out, d_total, n);

    if(numBlocks > 1) {
        unsigned int *d_total_scanned;
        cudaMalloc(&d_total_scanned, sizeof(unsigned int) * numBlocks);

        totalScan(d_total, d_total_scanned, numBlocks);
        mapScan<<<numBlocks, BLOCK_WIDTH>>>(d_out, d_total_scanned, n);

        cudaFree(d_total_scanned);
    }

    cudaFree(d_total);
}


// Do radix sort on d_inputVals and store to d_outputVals. The assosciated
// positions are also moved accordingly
void radix(unsigned int* const d_inputVals,
           unsigned int* const d_inputPos,
           unsigned int* const d_outputVals,
           unsigned int* const d_outputPos,
           const size_t numElems)
{
    unsigned int *inVals;
    unsigned int *inPos;
    unsigned int *zerosPredicate;
    unsigned int *onesPredicate;
    unsigned int *zerosScan;
    unsigned int *onesScan;

    for(int bit = 1; bit <= 1; bit++) {

    }

    cudaFree(inVals);
    cudaFree(inPos);
    cudaFree(zerosPredicate);
    cudaFree(onesPredicate);
    cudaFree(zerosScan);
    cudaFree(onesScan);
}


////////////////////////////////////////////////////////////////////////////////


// Wrapper for totalScan (host pointers as input)
void radixHost(unsigned int* const h_inputVals,
               unsigned int* const h_inputPos,
               unsigned int* const h_outputVals,
               unsigned int* const h_outputPos,
               const size_t numElems)
{
    unsigned int *d_inputVals;
    unsigned int *d_inputPos;
    unsigned int *d_outputVals;
    unsigned int *d_outputPos;
    size_t memsize = sizeof(unsigned int) * numElems;

    cudaMalloc(&d_inputVals, memsize);
    cudaMalloc(&d_inputPos, memsize);
    cudaMalloc(&d_outputVals, memsize);
    cudaMalloc(&d_outputPos, memsize);

    cudaMemcpy(d_inputVals, h_inputVals, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputPos, h_inputPos, memsize, cudaMemcpyHostToDevice);

    radix(d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems);

    cudaMemcpy(h_outputVals, d_outputVals, memsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_outputPos, d_outputPos, memsize, cudaMemcpyDeviceToHost);

    cudaFree(d_inputVals);
    cudaFree(d_inputPos);
    cudaFree(d_outputVals);
    cudaFree(d_outputPos);
}


int main(int argc, char **argv) {
    unsigned int *h_inVals;
    unsigned int *h_inPos;
    unsigned int *h_outVals;
    unsigned int *h_outPos;

    srand(0);

    size_t memsize = sizeof(unsigned int) * TEST_SIZE;

    h_inVals = (unsigned int*)malloc(memsize);
    h_inPos = (unsigned int*)malloc(memsize);
    h_outVals = (unsigned int*)malloc(memsize);
    h_outPos = (unsigned int*)malloc(memsize);

    // Random test values (seeded)
    for(int i=0; i<TEST_SIZE; i++){ h_inVals[i] = rand() % RAND_RANGE; }
    // Test positions 0 ... TEST_SIZE
    for(int i=0; i<TEST_SIZE; i++){ h_inPos[i] = i; }

    // Compute
    radixHost(h_inVals, h_inPos, h_outVals, h_outPos, TEST_SIZE);

    // Print input
    printf("h_inVals = [ ");
    for(int i=0; i<TEST_SIZE; i++){ printf("%d ", h_inVals[i]); }
    printf("];\nh_inPos = [ ");
    for(int i=0; i<TEST_SIZE; i++){ printf("%d ", h_inPos[i]); }
    printf("];\n");

    // Print output
    printf("h_outVals = [ ");
    for(int i=0; i<TEST_SIZE; i++){ printf("%d ", h_outVals[i]); }
    printf("];\nh_outPos = [ ");
    for(int i=0; i<TEST_SIZE; i++){ printf("%d ", h_outPos[i]); }
    printf("];\n");

    free(h_inVals);
    free(h_inPos);
    free(h_outVals);
    free(h_outPos);
    return 0;
}

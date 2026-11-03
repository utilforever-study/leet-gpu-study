#include <cuda_runtime.h>

// CPU에서 호출되어 GPU에서 스레드에 의해 병렬로 실행되는 커널 함수
// 반환 타입은 항상 void이며, __global__ 키워드로 선언되어야 함.
__global__ void vector_add(const float* A, const float* B, float* C, int N)
{
    // blockIdx.x 현재 스레드가 속한 그리드 내에서의 블록 인덱스
    // blockDim.x 한 블록이 가지고 있는 스레드의 총 개수
    // threadIdx.x 현재 블록 내에서 이 스레드의 고유 인덱스
    // ex) 1번 블록의 5번 스레드 = (1 * 256) + 5 = 261번째 인덱스를 처리
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 전체 스레드 개수가 데이터 개수(N)보다 많으면 안되므로
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    // 1개 블록에 배치할 스레드의 개수(보통 32의 배수, 최대 1024)
    int threadsPerBlock = 256;

    // 전체 데이터 개수(N)를 처리하기 위해 필요한 총 블록의 개수(Grid 크기)를 계산
    // N이 threadsPerBlock의 배수가 아닐 경우, 나머지 데이터를 처리하기 위해 +1 블록을 추가
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // <<<...>>> GPU 커널 함수를 호출하는 CUDA 문법
    // 지정한 blocksPerGrid 크기의 그리드와 threadsPerBlock 크기의 블록 구조
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);

    // CPU와 GPU 동기화
    // GPU 커널 호출은 비동기 동작
    // 이 함수를 호출함으로써 GPU가 vector_add 연산을 모두 마칠 때까지 CPU 대기
    cudaDeviceSynchronize();
}

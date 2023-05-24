/**
 * this file ported from FFMPEG
*/
#include "yadif.h"

template<typename T>
__inline__ __device__ T spatial_predictor(T a, T b, T c, T d, T e, T f, T g,
                                          T h, T i, T j, T k, T l, T m, T n)
{
    int spatial_pred = (d + k)/2;
    int spatial_score = abs(c - j) + abs(d - k) + abs(e - l);

    int score = abs(b - k) + abs(c - l) + abs(d - m);
    if (score < spatial_score) {
        spatial_pred = (c + l)/2;
        spatial_score = score;
        score = abs(a - l) + abs(b - m) + abs(c - n);
        if (score < spatial_score) {
          spatial_pred = (b + m)/2;
          spatial_score = score;
        }
    }
    score = abs(d - i) + abs(e - j) + abs(f - k);
    if (score < spatial_score) {
        spatial_pred = (e + j)/2;
        spatial_score = score;
        score = abs(e - h) + abs(f - i) + abs(g - j);
        if (score < spatial_score) {
          spatial_pred = (f + i)/2;
          spatial_score = score;
        }
    }
    return spatial_pred;
}

__inline__ __device__ int max3(int a, int b, int c)
{
    int x = max(a, b);
    return max(x, c);
}

__inline__ __device__ int min3(int a, int b, int c)
{
    int x = min(a, b);
    return min(x, c);
}

template<typename T>
__inline__ __device__ T temporal_predictor(T A, T B, T C, T D, T E, T F,
                                           T G, T H, T I, T J, T K, T L,
                                           T spatial_pred, bool skip_check)
{
    int p0 = (C + H) / 2;
    int p1 = F;
    int p2 = (D + I) / 2;
    int p3 = G;
    int p4 = (E + J) / 2;

    int tdiff0 = abs(D - I);
    int tdiff1 = (abs(A - F) + abs(B - G)) / 2;
    int tdiff2 = (abs(K - F) + abs(G - L)) / 2;

    int diff = max3(tdiff0, tdiff1, tdiff2);

    if (!skip_check) {
      int maxi = max3(p2 - p3, p2 - p1, min(p0 - p1, p4 - p3));
      int mini = min3(p2 - p3, p2 - p1, max(p0 - p1, p4 - p3));
      diff = max3(diff, mini, -maxi);
    }

    if (spatial_pred > p2 + diff) {
      spatial_pred = p2 + diff;
    }
    if (spatial_pred < p2 - diff) {
      spatial_pred = p2 - diff;
    }

    return spatial_pred;
}




__inline__ __device__  int getIndex(int x, int y, int pitch)
{
    return y*pitch+x;
}

template<typename T>
__inline__ __device__ void yadif_single(T *dst,
                                        T *prev,
                                        T *cur,
                                        T *next,
                                        int dst_width, int dst_height, int dst_pitch,
                                        int src_width, int src_height,
                                        int parity, int tff, bool skip_spatial_check)
{
    // Identify location
    int xo = blockIdx.x * blockDim.x + threadIdx.x;
    int yo = blockIdx.y * blockDim.y + threadIdx.y;

    if (xo >= dst_width || yo >= dst_height) { // TODO: add condition for x-3 and so on 
        return;
    }

    // Don't modify the primary field
    if (yo % 2 == parity) {
      dst[yo*dst_pitch+xo] = cur[getIndex(xo , yo,dst_pitch)]; //tex2D<T>(cur, xo, yo);
      return;
    }

    T a = cur[getIndex(xo - 3, yo - 1,dst_pitch)];
    T b = cur[getIndex(xo - 2, yo - 1,dst_pitch)];
    T c = cur[getIndex(xo - 1, yo - 1,dst_pitch)];
    T d = cur[getIndex(xo - 0, yo - 1,dst_pitch)];
    T e = cur[getIndex(xo + 1, yo - 1,dst_pitch)];
    T f = cur[getIndex(xo + 2, yo - 1,dst_pitch)];
    T g = cur[getIndex(xo + 3, yo - 1,dst_pitch)];

    T h = cur[getIndex(xo - 3, yo + 1,dst_pitch)];
    T i = cur[getIndex(xo - 2, yo + 1,dst_pitch)];
    T j = cur[getIndex(xo - 1, yo + 1,dst_pitch)];
    T k = cur[getIndex(xo - 0, yo + 1,dst_pitch)];
    T l = cur[getIndex(xo + 1, yo + 1,dst_pitch)];
    T m = cur[getIndex(xo + 2, yo + 1,dst_pitch)];
    T n = cur[getIndex(xo + 3, yo + 1,dst_pitch)];


    T spatial_pred =
        spatial_predictor(a, b, c, d, e, f, g, h, i, j, k, l, m, n);

    // Calculate temporal prediction
    int is_second_field = !(parity ^ tff);

    T* prev2 = prev;
    T* prev1 = is_second_field ? cur : prev;
    T* next1 = is_second_field ? next : cur;
    T* next2 = next;

    T A = prev2[getIndex(xo , yo - 1,dst_pitch)];
    T B = prev2[getIndex(xo , yo + 1,dst_pitch)];

    T C = prev1[getIndex(xo , yo - 2,dst_pitch)];
    T D = prev1[getIndex(xo , yo + 0,dst_pitch)];
    T E = prev1[getIndex(xo , yo + 2,dst_pitch)];

    T F = cur[getIndex(xo , yo - 1,dst_pitch)];
    T G = cur[getIndex(xo , yo + 1,dst_pitch)];

    T H = next1[getIndex(xo , yo - 2,dst_pitch)];
    T I = next1[getIndex(xo , yo + 0,dst_pitch)];
    T J = next1[getIndex(xo , yo + 2,dst_pitch)];

    T K = next2[getIndex(xo , yo - 1,dst_pitch)];
    T L = next2[getIndex(xo , yo + 1,dst_pitch)];

    spatial_pred = temporal_predictor(A, B, C, D, E, F, G, H, I, J, K, L,
                                      spatial_pred, skip_spatial_check);

    dst[yo*dst_pitch+xo] = spatial_pred;
}

__global__ void yadif_uchar(unsigned char *dst,
                            unsigned char *prev,
                            unsigned char *cur,
                            unsigned char *next,
                            int dst_width, int dst_height, int dst_pitch,
                            int src_width, int src_height,
                            int parity, int tff, bool skip_spatial_check=false)
{
    yadif_single(dst, prev, cur, next,
                 dst_width, dst_height, dst_pitch,
                 src_width, src_height,
                 parity, tff, skip_spatial_check);
}


#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )
#define ALIGN_UP(a, b) (((a) + (b) - 1) & ~((b) - 1))
#define BLOCKX 32
#define BLOCKY 16

// #define CUDA(x)				cudaCheckError((x), #x, __FILE__, __LINE__)


cudaError_t Yadif::yadif_cuda(  unsigned char *dst,
                                unsigned char *prev,
                                unsigned char *cur,
                                unsigned char *next,    
                                int dst_width, int dst_height, int dst_pitch,
                                int src_width, int src_height,
                                int parity, int tff, bool skip_spatial_check)

{
    const dim3 blockDim(BLOCKX, BLOCKY);
	const dim3 gridDim(DIV_UP(dst_width, blockDim.x), DIV_UP(dst_height, blockDim.y));

    yadif_uchar<<<gridDim,blockDim>>>(  dst, prev, cur, next,
                                        dst_width, dst_height, dst_pitch,
                                        src_width, src_height,
                                        parity, tff, skip_spatial_check);

    return cudaGetLastError();
}


/**
 * this file ported from FFMPEG
*/
#include "yadif.h"
#include "stdio.h"

template<typename T>
inline __device__ T spatial_predictor(T a, T b, T c, T d, T e, T f, T g,
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

inline __device__ int max3(int a, int b, int c)
{
    int x = max(a, b);
    return max(x, c);
}

inline __device__ int min3(int a, int b, int c)
{
    int x = min(a, b);
    return min(x, c);
}

template<typename T>
inline __device__ T temporal_predictor(T A, T B, T C, T D, T E, T F,
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




inline __device__  int getIndex(int x, int y, int pitch)
{
    return y*pitch+x;
}

template<typename T>
inline __device__ void yadif_single(T *dst,
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

    if (xo - 3 < 0 || yo - 1 < 0 || xo + 3 >= dst_width || yo + 1 >= dst_height) {
        dst[yo*dst_pitch+xo] = cur[getIndex(xo , yo,dst_pitch)];
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

    if (xo - 3 < 0 || yo - 2 < 0 || xo + 3 >= dst_width || yo + 2 >= dst_height) {
        dst[yo*dst_pitch+xo] = cur[getIndex(xo , yo,dst_pitch)];
        return;
    }


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


__global__ void cuda_get_y_channel(unsigned char *uyvy,unsigned char *y_channel,int width, int height)
{
    // Identify location
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    y_channel[x + width*y] = uyvy[1+2*x+width*2*y];
}



cudaError_t Yadif::getYChannel(unsigned char *uyvy,unsigned char *y)
{
    const dim3 blockDim(BLOCKX, BLOCKY);
	const dim3 gridDim(DIV_UP(m_im_width, blockDim.x), DIV_UP(m_im_height, blockDim.y));

    cuda_get_y_channel<<<gridDim,blockDim>>>(uyvy, y,m_im_width,m_im_height);
    return cudaGetLastError();
}


__global__ void cuda_get_uv_channel(unsigned char *uyvy,unsigned char *u_channel,unsigned char *v_channel,int width, int height)
{
    // Identify location
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    u_channel[x+width*y/2] = uyvy[4*x+width*2*y];
    v_channel[x+width*y/2] = uyvy[2+4*x+width*2*y];
}

cudaError_t Yadif::getUVChannel(unsigned char *uyvy,unsigned char *u,unsigned char *v)
{
    const dim3 blockDim(BLOCKX, BLOCKY);
	const dim3 gridDim(DIV_UP(m_im_width/2, blockDim.x), DIV_UP(m_im_height, blockDim.y));

    cuda_get_uv_channel<<<gridDim,blockDim>>>(uyvy, u,v,m_im_width,m_im_height);
    return cudaGetLastError();
}



__global__ void cuda_merge_uyvy(unsigned char *uyvy,unsigned char *y_channel, unsigned char *u_channel,unsigned char *v_channel,int width, int height)
{
        // Identify location
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    uyvy[ 0 + 4*x + width*2*y ] = u_channel[ x + width/2*y ];
    uyvy[ 1 + 4*x + width*2*y ] = y_channel[ 0 + 2*x + width*y ];
    uyvy[ 2 + 4*x + width*2*y ] = v_channel[ x + width/2*y ];
    uyvy[ 3 + 4*x + width*2*y ] = y_channel[ 1 + 2*x + width*y ];
}

cudaError_t Yadif::mergeUYVY(unsigned char* uyvy,unsigned char*y,unsigned char*u,unsigned char*v)
{
    const dim3 blockDim(BLOCKX, BLOCKY);
	const dim3 gridDim(DIV_UP(m_im_width/2, blockDim.x), DIV_UP(m_im_height, blockDim.y));
    cuda_merge_uyvy<<<gridDim,blockDim>>>(uyvy, y, u, v, m_im_width, m_im_height);
    return cudaGetLastError();
}

__global__ void cuda_spilit_uyvy(unsigned char *uyvy,unsigned char *y_channel, unsigned char *u_channel,unsigned char *v_channel,int width, int height)
{
        // Identify location
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    u_channel[ x + width/2*y ]      = uyvy[ 0 + 4*x + width*2*y ];
    y_channel[ 0 + 2*x + width*y]   = uyvy[ 1 + 4*x + width*2*y ];
    v_channel[ x + width/2*y ]      = uyvy[ 2 + 4*x + width*2*y ];
    y_channel[ 1 + 2*x + width*y]   = uyvy[ 3 + 4*x + width*2*y ];
}


cudaError_t Yadif::splitUYVY(unsigned char* uyvy,unsigned char*y,unsigned char*u,unsigned char*v)
{
    const dim3 blockDim(BLOCKX, BLOCKY);
	const dim3 gridDim(DIV_UP(m_im_width/2, blockDim.x), DIV_UP(m_im_height, blockDim.y));
    cuda_spilit_uyvy<<<gridDim,blockDim>>>(uyvy, y, u, v, m_im_width, m_im_height);
    return cudaGetLastError();
}


cudaError_t Yadif::yadifCuda(   unsigned char *dst,
                                unsigned char *prev,
                                unsigned char *cur,
                                unsigned char *next,    
                                int dst_width, int dst_height, int dst_pitch,
                                int src_width, int src_height)

{
    const dim3 blockDim(BLOCKX, BLOCKY);
	const dim3 gridDim(DIV_UP(dst_width, blockDim.x), DIV_UP(dst_height, blockDim.y));

    yadif_uchar<<<gridDim,blockDim>>>(  dst, prev, cur, next,
                                        dst_width, dst_height, dst_pitch,
                                        src_width, src_height,
                                        (int)m_parity, m_tff,false);

    return cudaGetLastError();
}


cudaError_t Yadif::filterCuda()
{
    const dim3 blockDim(BLOCKX, BLOCKY);
	const dim3 gridDim_y(DIV_UP(m_im_width, blockDim.x), DIV_UP(m_im_height, blockDim.y));
    const dim3 gridDim_uv(DIV_UP(m_im_width/2, blockDim.x), DIV_UP(m_im_height, blockDim.y));

    yadif_uchar<<<gridDim_y,blockDim>>>(  m_dst_y, m_prev_y, m_cur_y, m_next_y,
                                            m_im_width, m_im_height, m_im_width,
                                            m_im_width, m_im_height,
                                            (int)m_parity, m_tff, false);

    yadif_uchar<<<gridDim_uv,blockDim>>>(  m_dst_u, m_prev_u, m_cur_u, m_next_u,
                                            m_im_width/2, m_im_height, m_im_width/2,
                                            m_im_width/2, m_im_height,
                                            (int)m_parity, m_tff, false);

    yadif_uchar<<<gridDim_uv,blockDim>>>(  m_dst_v, m_prev_v, m_cur_v, m_next_v,
                                            m_im_width/2, m_im_height, m_im_width/2,
                                            m_im_width/2, m_im_height,
                                            (int)m_parity, m_tff, false);
    
    
    return cudaGetLastError();

}



Yadif::Yadif(unsigned int im_height,unsigned  int im_width,unsigned int row_bytes)
    :m_im_height{im_height},m_im_width{im_width}, m_row_bytes{row_bytes}
{
    m_parity = PARITY_TFF;
    cudaMalloc((void **)&m_frame_d, m_im_height*m_row_bytes);
    cudaMalloc((void **)&m_dst, m_im_height*m_row_bytes);

    cudaMalloc((void **)&m_frame_y, m_im_height*m_im_width);
    cudaMalloc((void **)&m_frame_u, m_im_height*m_im_width/2);
    cudaMalloc((void **)&m_frame_v, m_im_height*m_im_width/2);

    cudaMalloc((void **)&m_next_y, m_im_height*m_im_width);
    cudaMalloc((void **)&m_prev_y, m_im_height*m_im_width);
    cudaMalloc((void **)&m_cur_y, m_im_height*m_im_width);
    cudaMalloc((void **)&m_dst_y, m_im_height*m_im_width);

    cudaMalloc((void **)&m_next_u, m_im_height*m_im_width/2);
    cudaMalloc((void **)&m_prev_u, m_im_height*m_im_width/2);
    cudaMalloc((void **)&m_cur_u, m_im_height*m_im_width/2);
    cudaMalloc((void **)&m_dst_u, m_im_height*m_im_width/2);

    cudaMalloc((void **)&m_next_v, m_im_height*m_im_width/2);
    cudaMalloc((void **)&m_prev_v, m_im_height*m_im_width/2);
    cudaMalloc((void **)&m_cur_v, m_im_height*m_im_width/2);
    cudaMalloc((void **)&m_dst_v, m_im_height*m_im_width/2);
}

Yadif::~Yadif()
{
    cudaFree(m_frame_y);
    cudaFree(m_frame_u);
    cudaFree(m_frame_v);

    cudaFree(m_frame_d);
    cudaFree(m_dst);
    cudaFree(m_next_y);
    cudaFree(m_prev_y);
    cudaFree(m_cur_y);
    cudaFree(m_dst_y);

    cudaFree(m_next_u);
    cudaFree(m_prev_u);
    cudaFree(m_cur_u);
    cudaFree(m_dst_u);

    cudaFree(m_next_v);
    cudaFree(m_prev_v);
    cudaFree(m_cur_v);
    cudaFree(m_dst_v);
}


/**
 * @brief in this function the input fomat is in bmdFormat8BitYUV : ‘UYVY’ 4:2:2 Representation
 * otherwise it WONT Work! as expected!
 * 
 * @param frame 
 * @param out 
 */
void Yadif::filter(unsigned char* frame,unsigned char* out)
{
    cudaMemcpyAsync(m_frame_d, frame, m_im_height*m_row_bytes, cudaMemcpyHostToDevice);
    cudaError_t ret;

    ret = splitUYVY(m_frame_d,m_frame_y,m_frame_u,m_frame_v);
    if (ret != cudaSuccess)
        printf("error in getUVChannel: %d\n",ret);

    cudaMemcpyAsync(m_prev_y, m_cur_y, m_im_height*m_im_width, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(m_cur_y, m_next_y, m_im_height*m_im_width, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(m_next_y, m_frame_y, m_im_height*m_im_width, cudaMemcpyDeviceToDevice);

    cudaMemcpyAsync(m_prev_u, m_cur_u, m_im_height*m_im_width/2, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(m_cur_u, m_next_u, m_im_height*m_im_width/2, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(m_next_u, m_frame_u, m_im_height*m_im_width/2, cudaMemcpyDeviceToDevice);

    cudaMemcpyAsync(m_prev_v, m_cur_v, m_im_height*m_im_width/2, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(m_cur_v, m_next_v, m_im_height*m_im_width/2, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(m_next_v, m_frame_v, m_im_height*m_im_width/2, cudaMemcpyDeviceToDevice);
    // cudaStreamSynchronize()
    ret = filterCuda();

    if (ret != cudaSuccess)
        printf("error in filterCuda: %d\n",ret);

    ret = mergeUYVY(m_dst,m_dst_y,m_dst_u,m_dst_v);
    // ret = mergeUYVY(m_dst,m_frame_y,m_frame_u, m_frame_v);
     if (ret != cudaSuccess)
        printf("error in mergeUYVY: %d\n",ret);

    cudaMemcpy(out, m_dst, m_im_height*m_im_width*2, cudaMemcpyDeviceToHost);
}

void Yadif::filter(unsigned char* frame)
{
    filter(frame,frame);
}


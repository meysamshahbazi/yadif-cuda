/**
 * this file ported from FFMPEG
*/

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

template<typename T>
__inline__ __device__ void yadif_single(T *dst,
                                        cudaTextureObject_t prev,
                                        cudaTextureObject_t cur,
                                        cudaTextureObject_t next,
                                        int dst_width, int dst_height, int dst_pitch,
                                        int src_width, int src_height,
                                        int parity, int tff, bool skip_spatial_check)
{
    // Identify location
    int xo = blockIdx.x * blockDim.x + threadIdx.x;
    int yo = blockIdx.y * blockDim.y + threadIdx.y;

    if (xo >= dst_width || yo >= dst_height) {
        return;
    }

    // Don't modify the primary field
    if (yo % 2 == parity) {
      dst[yo*dst_pitch+xo] = tex2D<T>(cur, xo, yo);
      return;
    }

    // Calculate spatial prediction
    T a = tex2D<T>(cur, xo - 3, yo - 1);
    T b = tex2D<T>(cur, xo - 2, yo - 1);
    T c = tex2D<T>(cur, xo - 1, yo - 1);
    T d = tex2D<T>(cur, xo - 0, yo - 1);
    T e = tex2D<T>(cur, xo + 1, yo - 1);
    T f = tex2D<T>(cur, xo + 2, yo - 1);
    T g = tex2D<T>(cur, xo + 3, yo - 1);

    T h = tex2D<T>(cur, xo - 3, yo + 1);
    T i = tex2D<T>(cur, xo - 2, yo + 1);
    T j = tex2D<T>(cur, xo - 1, yo + 1);
    T k = tex2D<T>(cur, xo - 0, yo + 1);
    T l = tex2D<T>(cur, xo + 1, yo + 1);
    T m = tex2D<T>(cur, xo + 2, yo + 1);
    T n = tex2D<T>(cur, xo + 3, yo + 1);

    T spatial_pred =
        spatial_predictor(a, b, c, d, e, f, g, h, i, j, k, l, m, n);

    // Calculate temporal prediction
    int is_second_field = !(parity ^ tff);

    cudaTextureObject_t prev2 = prev;
    cudaTextureObject_t prev1 = is_second_field ? cur : prev;
    cudaTextureObject_t next1 = is_second_field ? next : cur;
    cudaTextureObject_t next2 = next;

    T A = tex2D<T>(prev2, xo,  yo - 1);
    T B = tex2D<T>(prev2, xo,  yo + 1);
    T C = tex2D<T>(prev1, xo,  yo - 2);
    T D = tex2D<T>(prev1, xo,  yo + 0);
    T E = tex2D<T>(prev1, xo,  yo + 2);
    T F = tex2D<T>(cur,   xo,  yo - 1);
    T G = tex2D<T>(cur,   xo,  yo + 1);
    T H = tex2D<T>(next1, xo,  yo - 2);
    T I = tex2D<T>(next1, xo,  yo + 0);
    T J = tex2D<T>(next1, xo,  yo + 2);
    T K = tex2D<T>(next2, xo,  yo - 1);
    T L = tex2D<T>(next2, xo,  yo + 1);

    spatial_pred = temporal_predictor(A, B, C, D, E, F, G, H, I, J, K, L,
                                      spatial_pred, skip_spatial_check);

    dst[yo*dst_pitch+xo] = spatial_pred;
}


__global__ void yadif_uchar(unsigned char *dst,
                            cudaTextureObject_t prev,
                            cudaTextureObject_t cur,
                            cudaTextureObject_t next,
                            int dst_width, int dst_height, int dst_pitch,
                            int src_width, int src_height,
                            int parity, int tff, bool skip_spatial_check)
{
    yadif_single(dst, prev, cur, next,
                 dst_width, dst_height, dst_pitch,
                 src_width, src_height,
                 parity, tff, skip_spatial_check);
}


#include "yadif.h"
#include <stdio.h>
#include <stdlib.h>


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
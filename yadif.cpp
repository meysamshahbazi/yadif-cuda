#include "yadif.h"
#include <stdio.h>
#include <stdlib.h>


Yadif::Yadif(unsigned int im_height,unsigned  int im_width,unsigned int row_bytes)
    :m_im_height{im_height},m_im_width{im_width}, m_row_bytes{row_bytes}
{
    m_parity = PARITY_TFF;
    cudaMalloc((void **)&m_next, m_im_height*m_row_bytes);
    cudaMalloc((void **)&m_prev, m_im_height*m_row_bytes);
    cudaMalloc((void **)&m_cur, m_im_height*m_row_bytes);
    cudaMalloc((void **)&m_dst, m_im_height*m_row_bytes);
}

Yadif::~Yadif()
{
    cudaFree(m_next);
    cudaFree(m_prev);
    cudaFree(m_cur);
    cudaFree(m_dst);
}

void Yadif::filter(unsigned char* frame,unsigned char* out)
{
    cudaMemcpy(m_prev, m_cur, m_im_height*m_row_bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(m_cur, m_next, m_im_height*m_row_bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(m_next, frame, m_im_height*m_row_bytes, cudaMemcpyHostToDevice);

    cudaError_t ret = yadifCuda( m_dst, m_prev, m_cur, m_next,
                m_im_width,m_im_height,m_im_width, //we assume the pitch is width!!!
                m_im_width, m_im_height,
                (int) m_parity,m_tff,false);
    
    if (ret != cudaSuccess)
        printf("error in yadif_cuda: %d\n",ret);

    cudaMemcpy(out, m_dst, m_im_height*m_row_bytes, cudaMemcpyDeviceToHost);
}
#include "yadif.h"

Yadif::Yadif(unsigned int im_height,unsigned  int im_width,unsigned int row_bytes)
    :im_height{im_height}, im_width{im_width}, row_bytes{row_bytes}
{
    parity = PARITY_TFF;
    cudaMalloc((void **)&next, im_height*row_bytes);
    cudaMalloc((void **)&prev, im_height*row_bytes);
    cudaMalloc((void **)&cur, im_height*row_bytes);

    cudaMalloc((void **)&dst, im_height*row_bytes);
}

Yadif::~Yadif()
{

}

void Yadif::filter(unsigned char* frame,unsigned char* out)
{
    cudaMemcpy(prev, cur, im_height*row_bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(cur, next, im_height*row_bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(next, frame, im_height*row_bytes, cudaMemcpyHostToDevice);

    yadif_cuda( dst, prev, cur, next,
                im_width,im_height,im_width,// we assume the pitch is width!!!
                im_width, im_height,
                (int) parity,tff,false);
            
    cudaMemcpy(out, dst, im_height*row_bytes, cudaMemcpyDeviceToHost);

}
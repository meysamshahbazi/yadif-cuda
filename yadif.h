#ifndef __YADIF_H__
#define __YADIF_H__

#include  <cstdint>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

cudaError_t yadif_cuda(     unsigned char *dst,
                            unsigned char *prev,
                            unsigned char *cur,
                            unsigned char *next,    
                            int dst_width, int dst_height, int dst_pitch,
                            int src_width, int src_height,
                            int parity, int tff, bool skip_spatial_check=false);


enum Parity_t {
        PARITY_TFF  =  0, ///< top field first
        PARITY_BFF  =  1, ///< bottom field first
        PARITY_AUTO = -1, ///< auto detection
};                    

class Yadif{
public:
    Yadif(unsigned int im_height, unsigned int im_width, unsigned int row_bytes);
    ~Yadif();
    void filter(unsigned char* frame,unsigned char* out);

    cudaError_t yadif_cuda(     unsigned char *dst,
                            unsigned char *prev,
                            unsigned char *cur,
                            unsigned char *next,    
                            int dst_width, int dst_height, int dst_pitch,
                            int src_width, int src_height,
                            int parity, int tff, bool skip_spatial_check=false);
private:
    
    /// @brief: 1 for is TOP filed came first and 0 otherwise
    int tff{1}; // TODO: check this with APIs bmdUpperFieldFirst 
    Parity_t parity;   
    const unsigned int im_height;
    const unsigned int im_width;
    const unsigned int row_bytes;

    // unsigned char* d_frame;
    unsigned char* prev;
    unsigned char* cur;
    unsigned char* next;
    unsigned char* dst;
    
};

#endif

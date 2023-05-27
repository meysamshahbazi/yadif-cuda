#ifndef __YADIF_H__
#define __YADIF_H__

#include  <cstdint>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"


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

    cudaError_t yadifCuda(  unsigned char *dst,
                            unsigned char *prev,
                            unsigned char *cur,
                            unsigned char *next,    
                            int dst_width, int dst_height, int dst_pitch,
                            int src_width, int src_height,
                            int parity, int tff, bool skip_spatial_check=false);
private:
    /// @brief: 1 for is TOP filed came first and 0 otherwise
    int m_tff{1}; // TODO: check this with APIs bmdUpperFieldFirst 
    Parity_t m_parity;   
    const unsigned int m_im_height;
    const unsigned int m_im_width;
    const unsigned int m_row_bytes;

    // unsigned char* d_frame;
    unsigned char* m_prev;
    unsigned char* m_cur;
    unsigned char* m_next;
    unsigned char* m_dst;
    
};

#endif

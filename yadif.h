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

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )
#define ALIGN_UP(a, b) (((a) + (b) - 1) & ~((b) - 1))
#define BLOCKX 32
#define BLOCKY 16


/**
 * @brief this class supposed to 
 * 
 */
class Yadif{
public:
    Yadif(unsigned int im_height, unsigned int im_width, unsigned int row_bytes);
    ~Yadif();
    void filter(unsigned char* frame,unsigned char* out);
    void filter(unsigned char* frame);


private:
    cudaError_t yadifCuda(  unsigned char *dst,
                            unsigned char *prev,
                            unsigned char *cur,
                            unsigned char *next,    
                            int dst_width, int dst_height, int dst_pitch,
                            int src_width, int src_height);
    cudaError_t getYChannel(unsigned char *uyvy,unsigned char *y);
    cudaError_t getUVChannel(unsigned char *uyvy,unsigned char *u,unsigned char *v);
    cudaError_t mergeUYVY(unsigned char* uyvy,unsigned char*y,unsigned char*u,unsigned char*v);
    cudaError_t filterCuda(); 
    /// @brief: 1 for is TOP filed came first and 0 otherwise
    int m_tff{1}; // TODO: check this with APIs bmdUpperFieldFirst 
    Parity_t m_parity;   
    const unsigned int m_im_height;
    const unsigned int m_im_width;
    const unsigned int m_row_bytes;


    unsigned char* m_frame_d;

    unsigned char* m_frame_y;
    unsigned char* m_frame_u;
    unsigned char* m_frame_v;

    unsigned char* m_prev_y;
    unsigned char* m_cur_y;
    unsigned char* m_next_y;
    unsigned char* m_dst_y;
    
    unsigned char* m_prev_u;
    unsigned char* m_cur_u;
    unsigned char* m_next_u;
    unsigned char* m_dst_u;

    unsigned char* m_prev_v;
    unsigned char* m_cur_v;
    unsigned char* m_next_v;
    unsigned char* m_dst_v;

    unsigned char* m_dst;
};

#endif

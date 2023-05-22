#ifndef __YADIF_H__
#define __YADIF_H__

class Yadif{
public:
    Yadif();
    ~Yadif();
    void filter(unsigned char* frame,unsigned char* out);
private:
    enum class Parity_t {
        YADIF_PARITY_TFF  =  0, ///< top field first
        YADIF_PARITY_BFF  =  1, ///< bottom field first
        YADIF_PARITY_AUTO = -1, ///< auto detection
    };
    /// @brief: 1 for is TOP filed came first and 0 otherwise
    int tff{1}; // TODO: check this with APIs bmdUpperFieldFirst 
    Parity_t parity;        
    
};

#endif

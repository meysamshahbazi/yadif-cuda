
sudo apt install cmake 

sudo apt install nvidia-cuda-toolkit

export PATH=/usr/local/cuda-11.3/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


unix {

  # auto-detect CUDA path

  CUDA_DIR = $$system(which nvcc | sed 's,/bin/nvcc$,,')

  INCLUDEPATH += $$CUDA_DIR/include

  QMAKE_LIBDIR += $$CUDA_DIR/lib

  LIBS += -lcudart

 cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.obj

  cuda.commands = nvcc -c -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

  cuda.depends = nvcc -M -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} | sed "s,^.*: ,," | sed "s,^ *,," | tr -d '\\n'

}

cuda.input = CUDA_SOURCES

QMAKE_EXTRA_UNIX_COMPILERS += cuda


###############################
cuda.commands = $$CUDA_DIR/bin/nvcc -g -G -arch=$$CUDA_ARCH -c $$NVCCFLAGS $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

cuda.dependcy_type = TYPE_C

cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -G -M $$CUDA_INC $$
 ${QMAKE_FILE_NAME} |sed “s,^.*:,”|sed “s,^ *,” | tr -d ‘\\n’

cuda.input = CUDA_SOURCES

QMAKE_EXTRA_UNIX_COMPILERS += cuda
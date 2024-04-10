
                #m_upper = min(m_ + block_size, m) - так не получилось
                #m_upper = (m_ + block_size) if (m_ + block_size) < m else m - так тоже

cython: linetrace=True, language_level=3 ###какая жесть вколько времени потратил пока нашел ААААААААА
distutils: extra_compile_args=-fopenmp
distutils: extra_link_args=-fopenmp



#distutils: extra_compile_args=-fopenmp
#distutils: extra_link_args=-fopenmp


import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

#---------так не работает вызывает ошибку---------------#
#def matmul_tiled_cython(a, b, c, int block_size):
#    m = a.shape[0]
#    n = b.shape[1]
#    k = a.shape[1]
#    matmul_tiled_cython_inner(a, b, c, m, n, k, block_size)
#    return c
#---------так не работает вызывает ошибку---------------#

@cython.boundscheck(False)
@cython.wraparound(False)

def matmul_tiled_cython(np.ndarray[np.float32_t, ndim=2] a,
                        np.ndarray[np.float32_t, ndim=2] b,
                        np.ndarray[np.float32_t, ndim=2] c,
                        int block_size):
    # Приведение массивов NumPy к memoryview
    cdef float[:, :] a_view = a
    cdef float[:, :] b_view = b
    cdef float[:, :] c_view = c



#---------так не работает вызывает ошибку--------------- так как не стоит void ----------#
#cdef matmul_tiled_cython_inner(np.ndarray[np.float32_t, ndim=2] a,
#                               np.ndarray[np.float32_t, ndim=2] b,
#                               np.ndarray[np.float32_t, ndim=2] c,
#                               int m, int n, int k,
#                               int block_size) nogil:
#    
#    cdef int m_, n_, k_, nc_, kc_, i, j, l, m_upper, n_upper, k_upper
#    cdef float temp
#---------так не работает вызывает ошибку---------------#GIL Global Interpreter Lock и еще Buffer may not be acquired without the GIL. Consider using memoryview slices instead.
#Coercion from Python not allowed without the GIL

@cython.boundscheck(False)
@cython.wraparound(False)

cdef int matmul_tiled_cython_inner(float[:, :] a, 
                                   float[:, :] b, 
                                   float[:, :] c, 
                                   int m, int n, int k, int block_size) noexcept nogil:

    cdef int m_, n_, k_, i, j, l
    cdef int m_upper, n_upper, k_upper
    cdef float temp
  
  #При использовании prange с параметром nogil=True, важно весь код внутри цикла может быть 
  #выполнен без Global Interpreter Lock (GIL). 
  #внутри такого цикла не должно быть операций, требующих взаимодействия с Python-объектами.

  #Получили еще пачку ошибок
    #performance hint: sjk012/tiled_gemm/tiled_gemm.pyx:47:5: Exception check on 'matmul_tiled_cython_inner' will always require the GIL to be acquired.
    #Possible solutions:
	#1. Declare 'matmul_tiled_cython_inner' as 'noexcept' if you control the definition and you're sure you don't want the function to raise exceptions.
	#2. Use an 'int' return type on 'matmul_tiled_cython_inner' to allow an error code to be returned.
 
 #Почитали поехали пробовать nogil noexcept: - не подошло
 # или 
 # return 0  # Возвращаем 0 для индикации успеха, другие значения могут представлять различные ошибки


    ###########################################################################
    # TODO: Implement the tiled matmul operation using Cython code.           #
    # Parallelize the loop via prange.                                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for m_ in prange(0, m, block_size, nogil=True):
        for n_ in range(0, n, block_size):
            for k_ in range(0, k, block_size):
                m_upper = min(m_ + block_size, m)
                n_upper = min(n_ + block_size, n)
                k_upper = min(k_ + block_size, k)

                for i in range(m_, m_upper):
                    for j in range(n_, n_upper):
                        temp = 0.0
                        for l in range(k_, k_upper):
                            temp += a[i, l] * b[l, j]
                        c[i, j] += temp

    return 0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
                
                        
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

def relu_bwd_cython(dy, mask):
    shape = dy.shape
    cdef np.ndarray dx = np.zeros((np.prod(shape)), dtype=dy.dtype)

    relu_backward_cython_inner(dx.reshape(-1), dy.reshape(-1), 
                               mask.astype(np.int8).reshape(-1))
    return dx.reshape(shape)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef relu_backward_cython_inner(np.ndarray[np.float32_t, ndim=1] dx,
                                np.ndarray[np.float32_t, ndim=1] dy,
                                np.ndarray[np.int8_t, ndim=1] mask):
    cdef int i
    ###########################################################################
    # TODO: Implement the relu_backward operation using Cython code.          #
    # Parallelize the loop using prange.                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    cdef int n = dy.shape[0]
    #prange для параллельной итерации по элементам массивов. 
    for i in prange(n, nogil=True): #nogil=True позволяет освободить глобальную блокировку интерпретатора 
        #Для каждого элемента в dy --градиенты, полученные от верхнего слоя-- 
        #и соответствующего элемента в mask 
        # mask - бинарная маска, указывающая активные нейроны во время прямого прохода
        # - вычисляем градиент dx по входу. 
        
        #Если элемент в mask = True => соответствующий нейрон был активен
        if mask[i]:
            #dx[i] устанавливается равным dy[i]
            dx[i] = dy[i]
        else:
            #В противном случае, если нейрон был неактивен
            dx[i] = 0
     #поэлементная операция соответствует логике обратного прохода для ReLU: 
     #градиенты распространяются назад только через те нейроны, 
     #которые были активны во время прямого прохода.

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################        

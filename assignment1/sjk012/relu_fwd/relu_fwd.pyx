import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

def relu_fwd_cython(x):
    shape = x.shape
    cdef np.ndarray max = np.zeros((np.prod(shape)), dtype=x.dtype)
    cdef np.ndarray mask = np.zeros((np.prod(shape)), dtype=np.int8)

    relu_cython_inner(x.reshape(-1), max, mask)
    
    return max.reshape(shape), mask.reshape(shape)

#для отключения проверки границ индексов массива во время выполнения
@cython.boundscheck(False) 
#отключает поддержку отрицательных индексов
@cython.wraparound(False)
cdef relu_cython_inner(np.ndarray[np.float32_t, ndim=1] x,
                       np.ndarray[np.float32_t, ndim=1] max,
                       np.ndarray[np.int8_t, ndim=1] mask):
    cdef int i

    ###########################################################################
    # TODO: Implement the ReLU activation using Cython code.                  #
    # Parallelize the loop using prange.                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #prange для параллелизации обработки. 
    #Эта функция будет 
    # 1 - поэлементно применять операцию ReLU к входному массиву x, 
    # 2 - сохраняя результаты в массив max 
    #  3 - соответствующие маски активации в mask.
    
    
    cdef int n = x.shape[0]
    #prange из модуля cython.parallel для распараллеливания цикла.
    
    for i in prange(n, nogil=True):
    #nogil=True означает, что во время выполнения этого цикла 
    #глобальная интерпретаторская блокировка Python будет снята, 
    #что позволяет выполнение кода на нескольких ядрах процессора параллельно.
    
        #x[i] больше нуля, то
        if x[i] > 0:
            #его значение копируется в max[i]
            max[i] = x[i]
            #а соответствующий элемент mask[i] устанавливается в 1
            mask[i] = 1
        # Если элемент меньше или равен нулю
        else:
            #устанавливается в 0
            max[i] = 0
            #устанавливается в 0
            mask[i] = 0

    #Это соответствует логике функции активации ReLU.        

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################        

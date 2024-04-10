#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange


def matmul_tiled_cython(a, b, c, int block_size):
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]
    matmul_tiled_cython_inner(a, b, c, m, n, k, block_size)
    return c

#? Why can not do this way
#def matmul_tiled_cython(np.ndarray[np.float32_t, ndim=2] a, 
#                        np.ndarray[np.float32_t, ndim=2] b,
#                        int block_size):


@cython.boundscheck(False)
@cython.wraparound(False)

#для выполнения блочного умножения матриц.
cdef matmul_tiled_cython_inner(np.ndarray[np.float32_t, ndim=2] a,
                               np.ndarray[np.float32_t, ndim=2] b,
                               np.ndarray[np.float32_t, ndim=2] c,
                               int m, int n, int k,
                               int block_size):

#Она принимает семь аргументов:
#a, b, c: двумерные массивы типа float (представляют матрицы)
#M, N, K: целые числа, размеры матриц 
#blockSize: целое число, размер блока для блочного умножения
#M - строк в A, 
#N - столбцов в B, 
#K - столбцов в A и строк в B
#a[M, K]
#b[K, N]
#b[B, N]

    cdef int m_, n_, k_, nc_, kc_, i, j, l, m_upper, n_upper, k_upper
    cdef float temp
    #Объявляет 6 - int переменных m, n, k, i, j, l
    # 3 - int mUpper, nUpper, kUpper использоваться для определения верхних границ циклов
    #  Они будут использоваться как счетчики внутри функции.
    #  cdef указывает, что эти переменные объявлены на уровне Cython
    #

    ###########################################################################
    # TODO: Implement the tiled matmul operation using Cython code.           #
    # Parallelize the loop via prange.                                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    #---NightMare fight with Errors---------------#
    #Iterating over Python object not allowed without gil
    #Constructing Python tuple not allowed without gil
    #Converting to Python object not allowed without gil
    #Coercion from Python not allowed without the GIL
    #Iterating over Python object not allowed without gil
    #Calling gil-requiring function not allowed without gi

 #-----iT shoul work like this but NOT-----------------
 #Этот цикл перебирает строки матрицы A по блокам.
 #m: переменная цикла, указывает на начало текущего блока.
 #   for m_ in prange(0, m, block_size, nogil=True):
 #конструкция Cython для параллельных циклов (OpenMP).
 #  0: начало диапазона (первая строка).
 #  M: конец диапазона (последняя строка в A).
 #  blockSize: шаг - размер блока.
 #  nogil=True: отключает проверку исключений (GIL) внутри цикла
 #       for n_ in range(0, n, block_size):
        # Вложенный цикл перебирает столбцы матрицы B по блокам.
        # n: переменная цикла, указывает на начало текущего блока.
        # range(0, N, blockSize): обычный цикл Python для последовательной iteraции.  
            # Еще один вложенный цикл перебирает столбцы матрицы A (и строки матрицы B)
            # по блокам.k: переменная цикла, указывает на начало текущего блока.  
 #           for k_ in range(0, k, block_size):
                #Вычисляет верхнюю границу текущего блока для строк матрицы A. 
                #min(m + blockSize, M) берет меньшее значение из m + blockSize и M.
                #Это гарантирует, что цикл не выходит за пределы матрицы.
 #               m_upper = min(m_ + block_size, m)
                #Вычисляет верхнюю границу текущего блока для столбцов матрицы B.
 #               n_upper = min(n_ + block_size, n)
                #Вычисляет верхнюю границу текущего блока для столбцов матрицы A (и строк матрицы B).
 #               k_upper = min(k_ + block_size, k)
 #
 #               for i in range(m_, m_upper):
                # Цикл по строкам текущего блока матрицы A
                #i: переменная цикла, указывает на индекс строки внутри блока
 #                   for j in range(n_, n_upper):
                     #Вложенный цикл по столбцам текущего блока матрицы B 
                     #j: переменная цикла, указывает на индекс столбца внутри блока.  
 #                       temp = 0.0
                         #Инициализирует временную переменную temp для хранения суммы по элементам.
 #                       for l in range(k_, k_upper):
                         #Вложеннейший цикл по элементам, участвующим в вычислении одного элемента 
                         #результирующей матрицы C. 
                         #l: переменная цикла, перебирает элементы по 
                         #столбцу текущего блока матрицы A (и строки текущего блока матрицы B).       
 #                           temp += a[i, l] * b[l, j]
                             #Вычисляет произведение элементов из соответствующих позиций матриц A и B 
                             #и добавляет его к temp.   
 #                       c[i, j] += temp
                         #После того, как все элементы для вычисления одного элемента 
                         #результата перебраны, полученная сумма (temp) добавляется к 
                         #соответствующему элементу матрицы C.
    
    ##-----Another way but NOT again -----------------
 #   with nogil, parallel():  # Release GIL for parallel execution
 #       for m_ in prange(0, m, block_size, schedule='static'):  # Parallelized outer loop
 #           for n_ in range(0, n, block_size):
 #               for k_ in range(0, k, block_size):

 #                   # Calculate upper bounds for blocks (correct indentation)
 #                   m_upper = min(m_ + block_size, m)
 #                   n_upper = min(n_ + block_size, n)
 #                   k_upper = min(k_ + block_size, k)

                    # Access elements using C-style indexing (avoid Python slicing)
 #                   for i in range(m_, m_upper):
 #                       for j in range(n_, n_upper):
 #                           temp = 0.0
 #                           for l in range(k_, k_upper):
 #                               temp += a_in[i, l] * b_in[l, j]  # Direct element access
 #                           c_out[i, j] += temp  # Direct element access
                            
 
    #? for n from 0 to N step blockSize: - Why it is in manual ?
     #? it is for n_ in range(0, n, block_size): Error ?

##???-----Another way - Compile But it is -   structure wrong?   -----------------
 
    # Используем prange для параллелизации внешнего цикла без GIL  (No GIL)
    for m_ in prange(0, m, block_size, nogil=True):
        for n_ in range(n):  # Изменено для совместимости без GIL (No GIL)
            for k_ in range(k):  # Изменено для совместимости без GIL (No GIL)
                m_upper = min(m_ + block_size, m)
                n_upper = min(n_ + block_size, n)
                k_upper = min(k_ + block_size, k)
                for i in range(m_, m_upper):
                    for j in range(n_, n_upper):
                        temp = 0.0
                        for l in range(k_, k_upper):
                            temp += a[i, l] * b[l, j]
                        c[i, j] += temp
 
    #Add this sudo apt-get install python3.11-dev
    # And Thid --define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    # Magik it compils. 
#----------------------------------------------------------------------------------------

##-------Test all nogil=True-------------------
##---------What is better - -
##???-----Another way - Compile But it is - structure wrong?   -----------------    
#    
#    for m_ in prange(0, m, block_size, nogil=True):
#        for n_ in prange(0, n, block_size, nogil=True):
#            for k_ in prange(0, k, block_size, nogil=True):
#                m_upper = min(m_ + block_size, m)
#                n_upper = min(n_ + block_size, n)
#                k_upper = min(k_ + block_size, k)
#
#                for i in range(m_, m_upper):
#                    for j in range(n_, n_upper):
#                        temp = 0.0
#                        for l in range(k_, k_upper):
#                            temp += a[i, l] * b[l, j]
#                        c[i, j] += temp

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
                
                        
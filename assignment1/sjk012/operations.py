import ctypes
import inspect
import math
import os
import platform
import numpy as np
from ctypes.util import find_library #используется для поиска системных библиотек..
from importlib import import_module  #позволяет динамически загружать модули Python.

#Глобальная переменная
MATMUL_METHOD = "CBLAS" #Это строковая переменная, определяющая метод матричного умножения, 
                        #который будет использоваться. В данном случае установлено значение "CBLAS", 
                        #что означает использование библиотеки CBLAS для операции умножения.

# Matmul operation
def matmul(a, b, c=None):           #Эта функция — точка входа для операции матричного умножения. 
                                    #Она выбирает метод умножения на основе значения MATMUL_METHOD
    
    if MATMUL_METHOD == "NAIVE":    #Вызывает функцию matmul_naive, реализующую наивный алгоритм умножения.
        return matmul_naive(a, b, c)
    
    elif MATMUL_METHOD == "NUMPY":  #Предполагается использование функции matmul_numpy для выполнения умножения
                                    # с использованием возможностей NumPy.
        
        return matmul_numpy(a, b, c)  
     

    elif MATMUL_METHOD == "CBLAS":    #Вызывает функцию matmul_cblas для выполнения умножения с помощью библиотеки CBLAS.
        return matmul_cblas(libopenblas(), a, b, c)
        
        
def matmul_naive(a, b, c=None): #Это место для реализации наивного алгоритма матричного умножения на чистом P
                                #Python без использования внешних библиотек. Вам предстоит заполнить эту часть 
                                #кода, реализовав умножение матриц a и b, а затем, если предоставлена матрица c,
                                #добавить результат умножения к c.
    ###########################################################################
    # TODO: Implement the matmul operation using only Python code.            #
    # If c is not None you should accumulate the result onto it.              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    M, N = a.shape
    N, P = b.shape
    if c is None:
        # Если c не предоставлена, инициализируем её нулевой матрицей подходящего размера
        c = np.zeros((M, P), dtype=a.dtype)
    
    # Выполнение матричного умножения и аккумуляция результата на c
    for i in range(M):
        for j in range(P):
            for k in range(N):
                c[i, j] += a[i, k] * b[k, j]
    
    #учитывает возможность наличия начального значения в c, что делает её более универсальной 
    #для различных вычислительных задач.

    # создает новую матрицу c (если она не предоставлена), инициализирует ее нулями и затем аккумулирует в 
    #нее результат умножения матриц a и b.
                
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return c


def matmul_numpy(a, b, c=None):
    
    ###########################################################################
    # TODO: Implement the matmul operation using only Numpy.                  #
    # If c is not None you should accumulate the result onto it.              #
    # Check the documentation of the numpy library at https://numpy.org/doc/  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 #библиотеки NumPy и возможностью аккумулирования результата на предоставленную матрицу c, 
 #можно использовать функцию np.dot или оператор @, которые оба поддерживают матричное умножение.

 #Используем np.dot(a, b) для выполнения матричного умножения массивов a и b. 
    #Это основной способ выполнения таких операций в NumPy.
 #Проверяется, передан ли массив c. Если c не None, то к результату умножения добавляется матрица c
    #, что реализует требуемую аккумуляцию.
   #Возвращается итоговый результат.
    # Выполняем матричное умножение a и b
    res = np.dot(a, b)
    
    # Если c не None, аккумулируем результат на c, иначе возвращаем только результат умножения
    if c is not None:
        res += c
    # ...
        
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return res #+ c if c is not None else res


def matmul_cblas(lib, a, b, c=None): #реализаций BLAS для C
    # библиотеку CBLAS (lib), две матрицы для умножения (a и b) и опционально матрицу c, 
    #к которой будет добавлен результат умножения. 
   
   #101 означает, что элементы будут храниться в строко-мажорном порядке.
   #102 означает, что элементы будут храниться в столбцо-мажорном порядке.
    order = 101  # 101 for row-major, 102 for column major data structures

    #Значение 101 соответствует порядку хранения данных по строкам (row-major order), 
    #который является стандартным для C и NumPy. Это важно для правильного интерпретирования данных при 
    #передаче в CBLAS.

    m = a.shape[0] # устанавливает значение переменной m равным количеству строк в массиве a
    n = b.shape[1] # устанавливает значение переменной n равным количеству столбцов в массиве b.
    k = a.shape[1] # устанавливает значение переменной k равным количеству столбцов в массиве a.
    
    alpha = 1.0  #alpha масштабирует произведение a*b
    # alpha часто используется как весовой коэффициент при выполнении операций с массивами. 
    # В данном случае, значение 1.0 означает, что элемент-по-элементное умножение a и b 
    # будет применено без изменений.

    #Эта строка проверяет, инициализирована ли переменная c.
    if c is None: 
        c = np.zeros((m, n), a.dtype, order="C") 
        #np.zeros - функция из библиотеки NumPy, которая создает массив, заполненный нулями.
        #(m, n) - это размерность создаваемого массива. Он будет иметь m строк и n столбцов (такие же размеры, как у a и потенциально b).
        # a.dtype - это тип данных элементов массива a. Новый массив c будет иметь такой же тип данных.
        # order="C" - указывает на строко-мажорный порядок хранения элементов в массиве c.
        beta = 0.0  #beta масштабирует матрицу c перед тем, как к ней будет добавлен результат умножения. 
        # весовой коэффициент для массива, который добавляется к результату операции. 
        #В данном случае, значение 0.0 означает, что к результату никакого массива добавляться не буде    
    else:
        beta = 1.0
    """
     A = [[1, 2, 3],
         [4, 5, 6]]

     A_transpose = [[1, 4],
                    [2, 5],
                    [3, 6]]

        С помощью цикла for.
        С помощью функции transpose() из библиотеки NumPy.
        A_transpose = np.transpose(A)
        С помощью оператора @.
    """    
    
    
    
    #BLAS (Basic Linear Algebra Subprograms)
    # trans_{a,b} = 111 for no transpose, 112 for transpose, and 113 for conjugate transpose
    
    #111: No transpose (matrix is stored as is)
    #112: Transpose (rows become columns and vice versa)
    #113: Conjugate transpose (transpose and complex conjugate - relevant for complex number matrices)
    
    """This checks if the elements of matrix a are stored in row-major order (C-contiguous).
        If yes, trans_a is set to 111 (no transpose) 
        and lda is set to k (leading dimension based on number of columns)."""
    if a.flags["C_CONTIGUOUS"]: #проверяется, хранится ли она в памяти по строкам (C_CONTIGUOUS) 
        trans_a = 111 # значени  транспонирования матриц
        lda = k  # lda (leading dimension of a) 
    
    """This checks if the elements of matrix a are stored in column-major order (F-contiguous).
        If yes, trans_a is set to 112 (transpose) 
        as BLAS expects row-major for matrix multiplication 
        and lda is set to m (leading dimension based on number of rows)."""
    elif a.flags["F_CONTIGUOUS"]: #или по столбцам (F_CONTIGUOUS),
        trans_a = 112 ## значени  транспонирования матриц
        lda = m  # lda (leading dimension of a) 
    
    """If the data layout of a is not supported (neither C nor F-contiguous), a ValueError is raised."""
    else:
        raise ValueError(f"Matrix a data layout not supported.")
    if b.flags["C_CONTIGUOUS"]:
        trans_b = 111
        ldb = n #leading dimension of b
    elif b.flags["F_CONTIGUOUS"]:
        trans_b = 112
        ldb = k #leading dimension of b
    else:
        raise ValueError(f"Matrix a data layout not supported.")
    ldc = n #leading dimension результирующей матрицы c, которая здесь равна количеству столбцов в матрице b

   ###Прототип функции в C 
    #void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
    #             const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
    #             const int K, const float alpha, const float *A,
    #             const int lda, const float *B, const int ldb,
    #             const float beta, float *C, const int ldc);



    ###########################################################################
    # TODO: Call to lib.cblas_sgemm function using the ctypes library         #
    # See its interface here:                                                 #
    # https://netlib.org/lapack/explore-html/de/da0/cblas_8h_a1446cddceb275e7cd299157a5d61d5e4.html 
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    
    #Определение аргументов и возвращаемого типа функции
    #cblas_sgemm является функцией без возвращаемого значения (она возвращает void в C), 
    #поэтому не нужно устанавливать restype. Однако необходимо указать типы аргументов с помощью argtypes. 
    
    # Определение размеров
    m, k = a.shape
    _, n = b.shape
 
    # Проверка или инициализация матрицы C
    if c is None:
        c = np.zeros((m, n), dtype=np.float32)

    
    
    # Определение типов аргументов
    lib.cblas_sgemm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, 
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]


       # Вызов функции
    lib.cblas_sgemm(
        101, 111, 111, m, n, k, 1.0,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), k,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n,
        1.0, c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n
    ) 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return c


def matmul_tiled(lib, a, b, c=None, block_size=32): 
                                             #(разбиение на блоки) для оптимизации матричного умножения
                                             # для уменьшения промахов кэша 
                                             #обработки данных малыми блоками, которые помещаются в кэш процессора.
    from tiled_gemm import tiled_gemm_cython

    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]
    if c is None:
        c = np.zeros((m, n), a.dtype, order="C")
    
    ###########################################################################
    # TODO: Call to lib.tiled_gemm function using the ctypes library          #
    #  
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Предполагаем, что tiled_gemm_cython принимает аргументы: матрицы a, b, c и размер блока
    tiled_gemm_cython(a, b, c, block_size)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
        
    return c
    

def load_library(name):
    """
    Loads an external library using ctypes.CDLL.

    It searches the library using ctypes.util.find_library(). If the library is
    not found, it traverses the LD_LIBRARY_PATH until it finds it. If it is not
    in any of the LD_LIBRARY_PATH paths, an ImportError exception is raised.

    Parameters
    ----------
    name : str
        The library name without any prefix like lib, suffix like .so, .dylib or
        version number (this is the form used for the posix linker option -l).

    Returns
    -------
    The loaded library.
    """
    path = None
    full_name = f"lib{name}.%s" % {"Linux": "so", "Darwin": "dylib"}[platform.system()] #Построение полного имени файла
                                                                                        #добавляя префикс lib и соответствующий суффикс (.so для Linux или .dylib для macOS).

#Поиск библиотеки в переменной окружения LD_LIBRARY_PATH: Функция итерирует по путям, указанным в 
    #переменной окружения LD_LIBRARY_PATH, в поисках файла библиотеки. 
    #Если файл находится, сохраняется полный путь к нему.    
    
    for current_path in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
        if os.path.exists(os.path.join(current_path, full_name)):
            path = os.path.join(current_path, full_name)
            break
            
    if path is None:
        # Didn't find the library
        raise ImportError(f"Library '{name}' could not be found. Please add its path to LD_LIBRARY_PATH.")
        #Если библиотека не была найдена ни в одном из путей LD_LIBRARY_PATH, функция генерирует исключение ImportError с сообщением о том, что библиотека не найдена.
        
    return ctypes.CDLL(path)  #Загрузка библиотеки: Если библиотека была найдена, она загружается с использованием ctypes.CDLL, передавая полный путь к файлу библиотеки.

def libopenblas():
    if not hasattr(libopenblas, "lib"):
        libopenblas.lib = load_library("openblas")
    return libopenblas.lib

def libblis():
    if not hasattr(libblis, "lib"):
        libblis.lib = load_library("blis")
    return libblis.lib
#Важным требованием является наличие переменной окружения LD_LIBRARY_PATH, которая содержит пути к директориям
# с библиотеками. Это стандартный способ указания дополнительных путей поиска для динамических библиотек
# в Unix-подобных системах.
#Использование ctypes позволяет напрямую вызывать функции из C/C++ библиотек в коде Python, 
#что открывает широкие возможности для интеграции Python с существующими библиотеками на этих языках.
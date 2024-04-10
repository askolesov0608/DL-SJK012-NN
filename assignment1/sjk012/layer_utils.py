from .layers import *


def fc_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    #Полносвязный слой: Входные данные x преобразуются с использованием полносвязного слоя 
    #fc_forward, который принимает входные данные x, веса w и смещения b. 
    #Эта функция возвращает выходные данные a и кэш fc_cache, содержащий информацию, 
    #необходимую для обратного прохода через полносвязный слой.
    a, fc_cache = fc_forward(x, w, b)
    
    #Слой ReLU: Затем выходные данные a из полносвязного слоя подаются на вход слоя ReLU 
    #(relu_forward_cython), который применяет нелинейную функцию активации к каждому 
    #элементу входного массива. Функция возвращает выходные данные out и кэш relu_cache 
    #для использования в обратном проходе.
    out, relu_cache = relu_forward_cython(a)
    
    #Кэш для обратного прохода: Кэш fc_cache от полносвязного слоя и кэш relu_cache 
    #от слоя ReLU упаковываются вместе и возвращаются как cache для последующего 
    #использования в обратном проходе.
    cache = (fc_cache, relu_cache)
    return out, cache


def fc_relu_backward(dy, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    #Распаковка кэша: Кэш cache, полученный от fc_relu_forward, 
    #распаковывается на кэши fc_cache и relu_cache.
    fc_cache, relu_cache = cache
    
    #Обратный проход ReLU: Сначала градиенты, полученные сверху (dy), 
    #пропускаются через обратный проход ReLU relu_backward_cython, 
    #который использует relu_cache для определения, через какие нейроны должен пройти градиент. 
    #Функция возвращает градиент da по входным данным слоя ReLU.
    da = relu_backward_cython(dy, relu_cache)
    
    #Обратный проход полносвязного слоя: Затем градиент da пропускается через обратный проход 
    #полносвязного слоя (fc_backward), используя fc_cache. 
    #Это дает градиенты по отношению к входным данным dx, весам dw и смещениям db полносвязного слоя.
    dx, dw, db = fc_backward(da, fc_cache)
    return dx, dw, db


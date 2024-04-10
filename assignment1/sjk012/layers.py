from builtins import range
from .operations import matmul
import numpy as np

def fc_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass using the matmul function       #
    # declared in operations.py.  Store the result in out.                    #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = x.shape[0]  # Получение размера батча
    x_flat = x.reshape(N, -1)  # Преобразование входных данных в двумерный массив
    #в двумерный массив, где первое измерение N соответствует количеству примеров в мини-батче, 
    #а второе измерение автоматически рассчитывается так, чтобы включить все оставшиеся размерности каждого примера. 
    #Это преобразование позволяет рассматривать каждый пример как одномерный вектор.

    out = x_flat.dot(w) + b  # Выполнение матричного умножения и добавление биаса
    #выполняет матричное умножение векторизованных входных данных на матрицу весов и добавляет к результату смещение.
    #выходной массив out, где каждая строка соответствует выходным данным одного примера из мини-батча.

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def fc_backward(dy, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dy: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the FC backward pass.                                   #
    # For the backward pass, we need to compute gradients dx, dw, and db      #
    # with respect to inputs x, weights w, and biases b, respectively.        #
    # Here are the steps for the backward pass:                               #
    # 1. Reshape the input x to 2D, keeping the batch size unchanged.         #
    #    This allows us to perform matrix multiplication efficiently.         #
    # 2. Compute the gradient of the input with respect to the loss (dx).     #
    #    This is done by multiplying the upstream gradient (dy) by the        #
    #    transpose of the weight matrix (w^T). The resulting dx has the same  #
    #    shape as the original input x.                                       #
    # 3. Compute the gradient of the weights with respect to the loss (dw).   #
    #    This is done by multiplying the transpose of the reshaped input      #
    #    (x_reshaped^T) by the upstream gradient (dy). The resulting dw       #
    #    has the same shape as the weight matrix w.                           #
    # 4. Compute the gradient of the biases with respect to the loss (db).    #
    #    This is simply the sum of the upstream gradient (dy) along each      #
    #    dimension, representing the contribution of each sample in the batch #
    #    to the bias gradient.                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #cache содержит исходные входные данные x, веса w и смещения b, сохраненные во время прямого прохода.
    x, w, b = cache

    # Шаг 1: Преобразуем входные данные x в 2D, сохраняя размер батча неизменным.
    x_reshaped = x.reshape(x.shape[0], -1)  # N x D
    
    #dy это градиент функции потерь по отношению к выходу полносвязного слоя, полученный из "верхнего" слоя.
    
    # Шаг 2: Вычисляем градиент по входным данным dx.
        #dx представляет градиент функции потерь по отношению к входным данным этого слоя. 
        #Он вычисляется путем матричного умножения dy на транспонированную матрицу весов w.T, 
        #чтобы "отправить" градиент обратно к входным данным.
    dx = np.dot(dy, w.T)  # dy: N x M, w.T: M x D => dx: N x D
    dx = dx.reshape(*x.shape)  # Преобразуем dx обратно в исходную форму входных данных x

    # Шаг 3: Вычисляем градиент по весам dw.
        #dw представляет градиент функции потерь по отношению к весам этого слоя. 
        #Он вычисляется путем матричного умножения 
        #транспонированных векторизованных входных данных x_reshaped.T на dy.
    dw = np.dot(x_reshaped.T, dy)  # x_reshaped.T: D x N, dy: N x M => dw: D x M

    # Шаг 4: Вычисляем градиент по смещениям db.
        #db представляет градиент функции потерь по отношению к смещениям. 
        #Он вычисляется как сумма dy по всем примерам в батче, 
        #так как каждый элемент батча вносит вклад в общий градиент смещения.
    db = np.sum(dy, axis=0)  # Суммируем градиенты по размеру батча

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward_numpy(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass using Numpy functions             #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #- y: Output, of the same shape as x
            #- mask: A boolean array that indicates whether each element of x is greater than zero
    y = np.maximum(0, x)  # Применяем ReLU к входу x поэлементно

    mask = x > 0  # Создаем маску для значений x, которые больше нуля
    #mask продолжает быть булевым массивом, который индицирует, 
    #был ли каждый элемент входного массива x больше нуля. 
    #Эта маска будет использоваться во время обратного прохода для эффективного распространения 
    #градиентов через слой ReLU.

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return y, mask


def relu_forward_cython(x):
    from sjk012.relu_fwd.relu_fwd import relu_fwd_cython
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None #? What for here
    ###########################################################################
    # TODO: Implement the ReLU forward pass calling the relu_fwd_cython function
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Вызываем Cython реализацию ReLU для получения выходных данных и маски
    y, mask = relu_fwd_cython(x)
    # Маска не используется здесь напрямую, но может быть полезна для обратного прохода,
    # так что ее можно сохранить в кэше, если это необходимо
    cache = x
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return y, cache


def relu_backward_numpy(dy, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, mask = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Для реализации обратного прохода функции активации ReLU в NumPy, 
     # 1 -  нужно использовать маску активации, полученную во время прямого прохода, 
        #чтобы вычислить градиент по отношению к входу. 
         # Маска активации указывает, какие нейроны были активны
         # то есть имели положительные значения во время прямого прохода. 
         # В контексте обратного прохода, 
            #Маска активации используется для "пропускания" градиентов 
            #только через активные нейроны, 
            #тогда градиенты неактивных нейронов 
             #тех, которые были занулены во время прямого прохода = устанавливаются в ноль.
    
    #cache содержит входные данные x для данного слоя, 
    #которые были сохранены во время прямого прохода.
    x = cache
    # Создаем маску, которая равна True для элементов x > 0
        #определяет, какие нейроны были активны во время прямого прохода 
        #те, у которых значение больше нуля.
    mask = x > 0
    # Инициализируем dx, используя градиент из верхнего слоя dout и маску
        #dy представляет собой градиент потерь относительно выхода данного слоя, 
        #полученный из верхнего последующего слоя.
            #Градиент по входу -dx-  
              #вычисляется как произведение градиента потерь относительно выхода 
              #данного слоя -dy- и маски -mask-. 
              #Градиент передается обратно через активные нейроны без изменений
              #--для неактивных нейронов градиент равен нулю.

    dx = dy * mask  # Применяем маску к градиенту из верхнего слоя

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx
    

def relu_backward_cython(dy, cache):
    from sjk012.relu_bwd.relu_bwd import relu_bwd_cython
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, mask = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #(cache) извлекаются входные данные слоя x 
    #(которые были сохранены во время прямого прохода).
    
    x = cache  # Извлекаем x из кэша

    # Передаем dy и маску в функцию Cython для вычисления dx
      #relu_bwd_cython, в которую передаются градиенты, полученные от верхнего слоя (dy), и маска активации. Маска активации генерируется на лету, используя условие x > 0, которое возвращает True для элементов x, больших нуля (активные нейроны во время прямого прохода), и False в противном случае.
        #Возвращается градиент dx, который представляет собой градиент функции потерь 
        #по отношению к входным данным слоя.
    
    dx = relu_bwd_cython(dy, x > 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    
    ###########################################################################
    # TODO: Implement the Softmax Loss function using Numpy                   #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Стабилизация входных данных x путем вычитания максимального значения
        # !!Сначала происходит стабилизация входных данных x путем вычитания максимального значения 
        # в каждой строке. Это предотвращает возможные числовые нестабильности 
        # при вычислении экспоненты больших чисел
    shifted_logits = x - np.max(x, axis=1, keepdims=True)

    # Вычисление softmax
        # Далее, вычисляется softmax, преобразовывая логиты - 
        #(нелинейные преобразования входных данных - в вероятности). 
        #log_probs — это логарифм вероятностей каждого класса, 
        #probs — соответствующие вероятности.
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)

    # Вычисление потерь: среднее отрицательного логарифма вероятности правильного класса
        #Потери (loss) вычисляются как 
        #среднее значение отрицательного логарифма вероятности правильных классов. 
        #Это - кросс-энтропийная потеря.
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N 

    # Вычисление градиента потерь по x
        # Градиент потерь по входным данным x (dx) вычисляется 
          #с учетом разницы между распределением вероятностей и истинным распределением классов. 
          # Для истинного класса y[i] от вероятности вычитается единица, 
          # это соответствует производной отрицательного логарифма вероятности
            #по соответствующему логиту.
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return loss, dx

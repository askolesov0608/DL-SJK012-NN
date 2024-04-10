from __future__ import print_function, division

from builtins import range
from builtins import object
import os
import pickle as pickle

import numpy as np

from sjk012 import optim


class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training classification
    models. The Solver performs stochastic gradient descent using different
    update rules defined in optim.py.

    Solver выполняет стохастический градиентный спуск, 
    используя различные правила обновления, определенные в optim.py.

    The solver accepts both training and validation data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.

    Решатель принимает как обучающие, так и валидационные данные и метки, 
     что позволяет ему периодически проверять точность классификации 
     как на обучающих, так и на валидационных данных, чтобы контролировать переобучение.

    To train a model, you will first construct a Solver instance, passing the
    model, dataset, and various options (learning rate, batch size, etc) to the
    constructor. You will then call the train() method to run the optimization
    procedure and train the model.

    Чтобы обучить модель, сначала вы создаете экземпляр Solver, 
    передавая в конструктор модель, набор данных и различные параметры 
    (скорость обучения, размер батча и т.д.). 
    Затем вы вызываете метод train() для запуска процедуры оптимизации и обучения модели.

    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_acc_history and solver.val_acc_history will be lists of the
    accuracies of the model on the training and validation set at each epoch.

    После возвращения метода train(), 
    model.params будет содержать параметры, которые показали лучший результат на валидационном наборе
      в течение обучения. 
    переменная экземпляра solver.loss_history будет содержать список всех потерь, 
      встреченных во время обучения, 
    переменные экземпляра solver.train_acc_history и solver.val_acc_history б
      будут списками точностей модели на обучающем и валидационном наборах на каждой эпохе.

    Example usage might look something like this:

    data = {
      'X_train': # training data
      'y_train': # training labels
      'X_val': # validation data
      'y_val': # validation labels
    }
    model = MyAwesomeModel(hidden_size=100, reg=10)
    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    lr_decay=0.95,
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()


    A Solver works on a model object that must conform to the following API:

    - model.params must be a dictionary mapping string parameter names to numpy
      arrays containing parameter values.
    - model.params должен быть словарем, сопоставляющим строковые имена параметров 
      с массивами numpy, содержащими значения параметров.

    - model.loss(X, y) must be a function that computes training-time loss and
      gradients, and test-time classification scores, with the following inputs
      and outputs:
    - model.loss(X, y) должна быть функцией, которая вычисляет потери во время обучения и градиенты,
      а также баллы классификации во время тестирования, 
       с следующими входными и выходными данными:

      Inputs:
      - X: Array giving a minibatch of input data of shape (N, d_1, ..., d_k)
      - X: Массив, предоставляющий мини-пакет входных данных формы (N, d_1, ..., d_k)
      - y: Array of labels, of shape (N,) giving labels for X where y[i] is the
      - y: Массив меток формы (N,), дающий метки для X, где y[i] является меткой для X[i].
        label for X[i].

      Returns:
      If y is None, run a test-time forward pass and return:
      - scores: Array of shape (N, C) giving classification scores for X where
        scores[i, c] gives the score of class c for X[i].
      Если y равно None, выполнить тестовый прямой проход и вернуть:
      - scores: Массив формы (N, C), дающий баллы классификации для X, 
        -scores[i, c] дает балл класса c для X[i].

      If y is not None, run a training time forward and backward pass and
      return a tuple of:
      - loss: Scalar giving the loss
      - grads: Dictionary with the same keys as self.params mapping parameter
        names to gradients of the loss with respect to those parameters.

      Если y не равно None, выполнить прямой и обратный проход во время обучения и вернуть кортеж:
      - loss: Скаляр, дающий значение потерь
      - grads: Словарь с теми же ключами, что и self.params, 
        сопоставляющий имена параметров с градиентами потерь по этим параметрам.
 
  --model.loss инкапсулирует прямой и обратный проходы   
    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A model object conforming to the API described above
        - model: Объект модели, соответствующий описанному API. 
                  Это центральный элемент, который будет обучаться с помощью Solver
        - data: A dictionary of training and validation data containing:
          'X_train': Array, shape (N_train, d_1, ..., d_k) of training images
          'X_val': Array, shape (N_val, d_1, ..., d_k) of validation images
          'y_train': Array, shape (N_train,) of labels for training images
          'y_val': Array, shape (N_val,) of labels for validation images
        - data: Словарь с обучающими и валидационными данными. 
                Включает массивы 
                X_train и X_val с изображениями для обучения и валидации соответственно, 
                массивы y_train и y_val с метками классов для этих изображений.  

        Optional arguments:
        - update_rule: A string giving the name of an update rule in optim.py.
          Default is 'sgd'.
        - update_rule: Строка, указывающая имя правила обновления в optim.py. 
          По умолчанию используется 'sgd' (стохастический градиентный спуск).  
        
        - optim_config: A dictionary containing hyperparameters that will be
          passed to the chosen update rule. Each update rule requires different
          hyperparameters (see optim.py) but all update rules require a
          'learning_rate' parameter so that should always be present.
        - optim_config: Словарь с гиперпараметрами, которые будут переданы выбранному правилу
          обновления. Все правила обновления требуют параметра learning_rate,
          поэтому он должен быть всегда указан.  
        
        - lr_decay: A scalar for learning rate decay; after each epoch the
          learning rate is multiplied by this value.
        - lr_decay: Скаляр для уменьшения скорости обучения. 
          После каждой эпохи скорость обучения умножается на это значение. 
        
        - batch_size: Size of minibatches used to compute loss and gradient
          during training.
        - batch_size: Размер мини-пакетов, используемых для 
          вычисления потерь и градиента во время обучения.
        
        - num_epochs: The number of epochs to run for during training.
        - num_epochs: Количество эпох обучения.

        - print_every: Integer; training losses will be printed every
          print_every iterations.
        - print_every: Число итераций, через которое будут печататься потери во время обучения.  

        - verbose: Boolean; if set to false then no output will be printed
          during training.
        - verbose: Булевый флаг. Если установлен в false, то во время обучения 
          вывод в консоль не будет осуществляться.

        - num_train_samples: Number of training samples used to check training
          accuracy; default is 1000; set to None to use entire training set.
        - num_train_samples: Количество обучающих образцов, используемых для проверки точности 
          на обучающем наборе. По умолчанию 1000. Если установлено None, используется весь обучающий набор.  

        - num_val_samples: Number of validation samples to use to check val
          accuracy; default is None, which uses the entire validation set.
        - num_val_samples: Количество валидационных образцов для проверки точности на валидационном наборе. 
          По умолчанию используется весь валидационный набор.  

        - checkpoint_name: If not None, then save model checkpoints here every
          epoch.
        - heckpoint_name: Если не None, то в этом файле будут сохраняться контрольные точки модели после каждой эпохи.  
        """
        self.model = model
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]

        # Unpack keyword arguments
        #идет распаковка опциональных аргументов с использованием метода pop для словаря kwargs, 
         #что позволяет извлечь значение по ключу и установить значение по умолчанию, 
         #если ключ не найден в словаре.
        self.update_rule = kwargs.pop("update_rule", "sgd")
        self.optim_config = kwargs.pop("optim_config", {})
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)
        self.num_train_samples = kwargs.pop("num_train_samples", 1000)
        self.num_val_samples = kwargs.pop("num_val_samples", None)

        self.checkpoint_name = kwargs.pop("checkpoint_name", None)
        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)

        # Throw an error if there are extra keyword arguments
        # происходит проверка на наличие лишних ключей в kwargs, 
        # что может указывать на ошибку в именах параметров или их избыточность. 
        # Если обнаружены лишние ключи, генерируется исключение ValueError.
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        #Проверяется, существует ли выбранное правило обновления (update_rule) в модуле optim. 
        #Если такого правила нет, генерируется исключение ValueError. 
        #В случае успеха строковое название правила обновления заменяется на соответствующую функцию
        #из модуля optim.
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)
        
        #Вызывается внутренний метод _reset, который предназначен для инициализации или сброса 
        #внутренних переменных, используемых в процессе обучения. 
        #Это может включать инициализацию оптимизационного состояния, счетчиков итераций, 
        #истории потерь и точности и т.д.
        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        #Инициализируются переменные для ведения учета процесса обучения: 
        self.epoch = 0 #для отслеживания текущей эпохи обучения
        self.best_val_acc = 0 #для хранения лучшей точности на валидационном наборе
        self.best_params = {} #для сохранения параметров модели, которые дали лучшую точность на валидационном наборе.
        self.loss_history = [] #Создаются пустые списки
        self.train_acc_history = [] #Создаются пустые списки
        self.val_acc_history = [] #Создаются пустые списки

        # Make a deep copy of the optim_config for each parameter
        #Для каждого параметра модели создается глубокая копия конфигурации оптимизатор
        self.optim_configs = {}
        # содержит гиперпараметры для правила обновления параметров 
          #скорость обучения - для каждого параметра была своя индивидуальная конфигурация оптимизатора, 
          #позволяет использовать разные скорости обучения для различных параметров модели
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        выполнение одного шага градиентного обновления в процессе обучения. 
        Этот метод вызывается внутри метода train()
        """
        # Make a minibatch of training data Формирование мини-пакета обучающих данных
        num_train = self.X_train.shape[0] #Сначала определяется количество обучающих примеров
        
        #случайным образом выбираются индексы для формирования мини-пакета batch_mask
        #размер которого определен в self.batch_size.
        batch_mask = np.random.choice(num_train, self.batch_size) 

        #На основе batch_mask формируются мини-пакеты 
        #входных данных (X_batch) и соответствующих меток (y_batch).
        X_batch = self.X_train[batch_mask] #
        y_batch = self.y_train[batch_mask]

        # Compute loss and gradient
        #Вызывается метод loss модели с мини-пакетом в качестве аргумента для вычисления
        #потерь и градиентов по параметрам модели.
        #Полученное значение потерь добавляется в историю потерь (self.loss_history).
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # Perform a parameter update Обновление параметров модели
        for p, w in self.model.params.items(): #Для каждого параметра модели (p) происходит его обновление. 
            dw = grads[p]  # Используется градиент этого параметра (dw)
            config = self.optim_configs[p] #текущая конфигурация оптимизатора для данного параметра  (config) 
            
            #и правило обновления (self.update_rule)
            next_w, next_config = self.update_rule(w, dw, config) 
            #которое принимает текущее значение параметра (w)
            #его градиент (dw) 
            #kонфигурацию (config)
            #возвращая обновленное значение параметра (next_w)
            # обновленную конфигурацию (next_config).
            self.model.params[p] = next_w #Обновленное значение параметра (next_w) сохраняется в модели
            self.optim_configs[p] = next_config #обновленная конфигурация оптимизатора (next_config) сохраняется для использования в следующем шаге обновления.

    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return
        checkpoint = {
            "model": self.model,
            "update_rule": self.update_rule,
            "lr_decay": self.lr_decay,
            "optim_config": self.optim_config,
            "batch_size": self.batch_size,
            "num_train_samples": self.num_train_samples,
            "num_val_samples": self.num_val_samples,
            "epoch": self.epoch,
            "loss_history": self.loss_history,
            "train_acc_history": self.train_acc_history,
            "val_acc_history": self.val_acc_history,
        }
        filename = "%s_epoch_%d.pkl" % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, "wb") as f:
            pickle.dump(checkpoint, f)

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        Check accuracy of the model on the provided data.

        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using
          too much memory.

        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """

        # Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc

    def train(self):
        """
        Run optimization to train the model.
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print(
                    "(Iteration %d / %d) loss: %f"
                    % (t + 1, num_iterations, self.loss_history[-1])
                )

            # At the end of every epoch, increment the epoch counter and decay
            # the learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]["learning_rate"] *= self.lr_decay

            # Check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            first_it = t == 0
            last_it = t == num_iterations - 1
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(
                    self.X_train, self.y_train, num_samples=self.num_train_samples
                )
                val_acc = self.check_accuracy(
                    self.X_val, self.y_val, num_samples=self.num_val_samples
                )
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                self._save_checkpoint()

                if self.verbose:
                    print(
                        "(Epoch %d / %d) train acc: %f; val_acc: %f"
                        % (self.epoch, self.num_epochs, train_acc, val_acc)
                    )

                # Keep track of the best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # At the end of training swap the best params into the model
        self.model.params = self.best_params

import numpy as np

def softmax(x):
    """
    Вычисляет softmax функцию для входного массива x.
    
    Softmax функция преобразует входные значения в вероятности, распределяя
    их таким образом, что их сумма равна 1. Это полезно в задачах классификации,
    где требуется получить вероятности принадлежности к каждому классу.
    
    Параметры:
    ----------
    x : numpy.ndarray
        Входной массив значений размером (n_samples, n_classes), для которых необходимо вычислить softmax.
    
    Возвращает:
    ----------
    numpy.ndarray
        Массив значений softmax, где каждый элемент является вероятностью, и сумма всех элементов равна 1.
    """
    
    x_shifted = x - np.max(x, axis=1, keepdims=True)

    exp_x = np.exp(x_shifted)

    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    
    return exp_x / sum_exp_x

def one_hot_encode(y, n_classes=None):
    """
    Выполняет one-hot кодирование для заданного списка меток.

    Параметры:
    ----------
    y : numpy.ndarray или list
        Вектор или список меток классов, которые необходимо закодировать.
        Значения меток должны быть целыми числами от 0 до n_classes-1.

    n_classes : int или None, по умолчанию None
        Количество классов (размерность выходного пространства).
        Если None, то количество классов определяется автоматически как максимум значения в y плюс один.

    Возвращает:
    ----------
    numpy.ndarray
        Массив размером (n_samples, n_classes), где n_samples — количество образцов, а n_classes — количество классов.
        Каждая строка представляет собой one-hot закодированное представление соответствующей метки из y.
    """
    
    y = np.array(y)

    if n_classes is None:
        n_classes = np.max(y) + 1

    one_hot = np.zeros((len(y), n_classes))

    one_hot[np.arange(len(y)), y] = 1

    return one_hot


class StandardScaler:
    """
    Класс для стандартизации данных путем удаления среднего и масштабирования к единичной дисперсии.

    Стандартизация данных улучшает производительность большинства алгоритмов машинного обучения,
    приводя их к единому масштабу. Класс `StandardScaler` вычисляет среднее и дисперсию по
    обучающим данным и использует эти параметры для стандартизации новых данных.

    Атрибуты:
    ---------
    mean_ : numpy.ndarray размера (n_features,) или None
        Среднее значение каждого признака в обучающем наборе данных. Инициализируется как None до вызова метода `fit`.

    var_ : numpy.ndarray размера (n_features,) или None
        Дисперсия каждого признака в обучающем наборе данных. Инициализируется как None до вызова метода `fit`.
    """

    def __init__(self):
        """
        Инициализирует объект класса StandardScaler.

        Инициализирует атрибуты mean_ и var_ как None, которые будут заполнены
        после выполнения метода `fit`.
        """
        self.mean_ = None
        self.var_ = None
        self.eps = 1e-45

    def fit(self, X):
        """
        Вычисляет среднее и дисперсию для каждого признака в обучающем наборе данных.

        Метод `fit` обучает модель на данных, вычисляя среднее и дисперсию, которые будут
        использоваться для стандартизации данных.

        Параметры:
        ----------
        X : numpy.ndarray
            Входной массив данных размером (n_samples, n_features), где n_samples — количество образцов,
            а n_features — количество признаков.

        Возвращает:
        ----------
        self : StandardScaler
            Возвращает экземпляр объекта `StandardScaler` с вычисленными атрибутами mean_ и var_.
        """
        self.mean_ = np.mean(X, axis=0)
        self.var_ = np.var(X, axis=0)

        return self


    def transform(self, X):
        """
        Преобразует данные, применяя стандартизацию на основе среднего и дисперсии, вычисленных в методе `fit`.

        Метод `transform` стандартизирует новые данные на основе параметров, вычисленных в `fit`.

        Параметры:
        ----------
        X : numpy.ndarray
            Входной массив данных размером (n_samples, n_features), который необходимо стандартизировать.

        Возвращает:
        ----------
        X_scaled : numpy.ndarray
            Стандартизированные данные того же размера, что и входной массив X.
        """

        return (X - self.mean_) / np.sqrt(self.var_ + self.eps)

    def fit_transform(self, X):
        """
        Комбинированный метод для выполнения обучения и трансформации данных.

        Этот метод сначала вычисляет среднее и дисперсию по обучающим данным,
        а затем сразу же стандартизирует их.

        Параметры:
        ----------
        X : numpy.ndarray
            Входной массив данных размером (n_samples, n_features), который необходимо обучить и стандартизировать.

        Возвращает:
        ----------
        X_scaled : numpy.ndarray
            Стандартизированные данные того же размера, что и входной массив X.
        """
        self.fit(X)

        return self.transform(X)


class SoftmaxRegression:
    """
    Класс для выполнения многоклассовой логистической регрессии с использованием softmax-функции
    и поддержкой L1 и L2 регуляризации.

    Параметры:
    ----------
    lr : float, default=0.01
        Скорость обучения (learning rate) для обновления коэффициентов модели.

    weight_decay : float, default=0
        Коэффициент регуляризации, который предотвращает переобучение путем добавления штрафа
        за большие значения весов. В зависимости от параметра `penalty`, может использоваться
        для L1 или L2 регуляризации.

    n_epochs : int, default=100
        Количество эпох для обучения модели.

    penalty : str, default='l2'
        Тип регуляризации. Поддерживаются значения 'l1' для L1-регуляризации и 'l2' для L2-регуляризации.

    batch_size : int, default=16
        Размер батча для mini-batch градиентного спуска (mini-batch GD).
    
    Атрибуты:
    ---------
    coef_ : numpy.ndarray или None
        Коэффициенты (веса) модели размером (n_features, n_classes), которые обучаются на данных. Инициализируются как None до вызова метода `fit`.

    intercept_ : numpy.ndarray или None
        Свободный член (сдвиг) модели размером (n_classes). Инициализируется как None до вызова метода `fit`.

    self.n_classes_ : int
        Количество классов, определяемое на основе уникальных меток в обучающем наборе данных.
        Этот параметр устанавливается после вызова метода `fit` и используется для определения 
        размерности выходного пространства модели. Он равен максимальному значению метки в данных плюс один.
    """

    def __init__(self, lr=0.01, weight_decay=0, n_epochs=100, penalty='l2', batch_size=16, fit_intercept=True):
        """
        Инициализация объекта класса SoftmaxRegression с заданными гиперпараметрами.

        Параметры:
        ----------
        lr : float, default=0.01
            Скорость обучения (learning rate) для обновления коэффициентов модели.

        weight_decay : float, default=0
            Коэффициент регуляризации.

        n_epochs : int, default=100
            Количество эпох для обучения модели.

        penalty : str, default='l2'
            Тип регуляризации. 'l1' для L1-регуляризации, 'l2' для L2-регуляризации.

        batch_size : int, default=16
            Размер батча для mini-batch градиентного спуска (mini-batch GD).

        fit_intercept : bool, по умолчанию True
            Включать ли свободный член (сдвиг) в модель.
        """
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.penalty = penalty
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.n_classes_ = None

    def predict_proba(self, X):
        """
        Предсказывает вероятности классов для входных данных на основе обученной модели softmax-регрессии.

        Параметры:
        ----------
        X : numpy.ndarray
            Входной массив признаков размером (n_samples, n_features).

        Возвращает:
        ----------
        numpy.ndarray
            Массив предсказанных вероятностей для каждого класса, где каждая строка соответствует одному объекту.
        """
        
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet.")

        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
            weights = np.vstack([self.intercept_, self.coef_])
        else:
            weights = self.coef_

        logits = np.dot(X, weights)

        return softmax(logits)

    def predict(self, X):
        """
        Предсказывает метки классов для входных данных на основе обученной модели softmax-регрессии.

        Параметры:
        ----------
        X : numpy.ndarray
            Входной массив признаков размером (n_samples, n_features).

        Возвращает:
        ----------
        numpy.ndarray
            Вектор предсказанных меток классов (значения от 0 до n_classes-1).
        """

        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


    def loss(self, y_true, probs):
        """
        Вычисляет функцию потерь для многоклассовой классификации на основе кросс-энтропии
        с учетом L1/L2 регуляризации

        Параметры:
        ----------
        y_true : numpy.ndarray
            Вектор истинных меток классов (значения от 0 до n_classes-1).

        probs : numpy.ndarray
            Массив предсказанных вероятностей для каждого класса.

        Возвращает:
        ----------
        float
            Значение функции потерь на основе кросс-энтропии.
        """

        n_samples = y_true.shape[0]

        y_one_hot = one_hot_encode(y_true, self.n_classes_)

        log_probs = -np.log(probs + 1e-45)
        cross_entropy_loss = np.sum(y_one_hot * log_probs) / n_samples

        reg_term = 0
        if self.penalty == 'l2':
            reg_term = self.weight_decay * np.sum(self.coef_ ** 2)
        elif self.penalty == 'l1':
            reg_term = self.weight_decay * np.sum(np.abs(self.coef_))

        return cross_entropy_loss + reg_term


      
    def loss_grad(self, X, y_true):
        """
        Вычисляет градиент функции потерь по отношению к весам модели для softmax-регрессии.

        В случае использования регуляризации, градиент включает соответствующие компоненты для
        штрафа за большие значения весов.

        Параметры:
        ----------
        X : numpy.ndarray
            Входной массив признаков размером (n_samples, n_features), где n_samples — количество образцов,
            а n_features — количество признаков.

        y_true : numpy.ndarray
            Вектор истинных меток классов (значения от 0 до n_classes-1).

        Возвращает:
        ----------
        grad : numpy.ndarray
            Градиент функции потерь по отношению к весам модели.

        grad_intercept : numpy.ndarray
            Градиент функции потерь по отношению к свободному члену.
        """
        
        n_samples = X.shape[0]

        probs = self.predict_proba(X)

        y_one_hot = one_hot_encode(y_true, self.n_classes_)

        error = probs - y_one_hot
        
        if self.fit_intercept:
            grad_intercept = np.sum(error, axis=0) / n_samples
        else:
            grad_intercept = None

        grad_coef = np.dot(X.T, error) / n_samples

        if self.penalty == 'l2':
            grad_coef += 2 * self.weight_decay * self.coef_
        elif self.penalty == 'l1':
            grad_coef += self.weight_decay * np.sign(self.coef_)

        return grad_coef, grad_intercept


    def step(self, grad, grad_intercept):
        """
        Выполняет один шаг обновления весов модели с использованием вычисленного градиента.

        Параметры:
        ----------
        grad : numpy.ndarray
            Градиент функции потерь по отношению к весам модели (размером как coef_).
        
        grad_intercept : numpy.ndarray или None
            Градиент функции потерь по отношению к свободному члену (размером как intercept_).
            Если fit_intercept=False, этот параметр будет равен None.

        Возвращает:
        ----------
        None
        """
        
        self.coef_ -= self.lr * grad
        if self.fit_intercept and grad_intercept is not None:
            self.intercept_ -= self.lr * grad_intercept

    def fit(self, X, y):
        """
        Обучает модель softmax-регрессии с использованием mini-batch градиентного спуска (mini-batch GD).

        Параметры:
        ----------
        X : numpy.ndarray
            Входной массив признаков размером (n_samples, n_features), где n_samples — количество образцов,
            а n_features — количество признаков.

        y : numpy.ndarray
            Вектор истинных меток классов (значения от 0 до n_classes-1).

        Возвращает:
        ----------
        self : SoftmaxRegression
            Обученная модель softmax-регрессии.
        """
        
        n_samples, n_features = X.shape
        self.n_classes_ = np.max(y) + 1

        self.coef_ = np.zeros((n_features, self.n_classes_))
        self.intercept_ = np.zeros(self.n_classes_) if self.fit_intercept else None

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                grad_coef, grad_intercept = self.loss_grad(X_batch, y_batch)

                self.step(grad_coef, grad_intercept)

        return self



class Trainer:
    """
    Класс для управления процессом обучения и предсказания с использованием модели машинного обучения.

    Этот класс предоставляет методы для обучения модели на данных и выполнения предсказаний на новых данных.
    Он инкапсулирует логику обучения и предсказания, обеспечивая удобный интерфейс для выполнения этих задач.
    """

    def __init__(self, lr=0.028, weight_decay=0.0001, n_epochs=400, penalty='l2', batch_size=15, fit_intercept=True):
        """
        Инициализирует объект класса Trainer.

        Этот конструктор может использоваться для инициализации необходимых атрибутов, таких как модель,
        средства для предобработки данных, или другие параметры, необходимые для обучения и предсказания.
        """

        self.scaler = StandardScaler()
        self.model = SoftmaxRegression(lr=lr, weight_decay=weight_decay, n_epochs=n_epochs, 
        penalty=penalty, batch_size=batch_size, fit_intercept=fit_intercept)

        
    def train_model(self, X, y):
        """
        Обучает модель машинного обучения на предоставленных данных.

        Параметры:
        ----------
        X : numpy.ndarray
            Входной массив признаков размером (n_samples, n_features), где n_samples — количество образцов,
            а n_features — количество признаков.

        y : numpy.ndarray или pandas.Series
            Вектор меток классов, соответствующих каждому объекту из X.

        Возвращает:
        ----------
        self : Trainer
            Возвращает экземпляр текущего объекта класса Trainer после обучения модели.
        """

        X_train_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_train_scaled, y)

        return self

    
    def predict_model(self, X):
        """
        Делает предсказания на новых данных с использованием обученной модели.

        Параметры:
        ----------
        X : numpy.ndarray
            Входной массив признаков размером (n_samples, n_features) для новых данных.

        Возвращает:
        ----------
        predictions : numpy.ndarray
            Вектор предсказанных меток классов для входных данных.
        """

        X_test_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_test_scaled)

        return predictions
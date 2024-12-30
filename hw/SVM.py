import numpy as np
from scipy import optimize


class BinaryEstimatorSVM:
    """
    Класс для построения модели бинарной классификации методом опорных 
    векторов путем решения прямой задачи оптимизации.

    Параметры:
    ----------
    lr : float, default=0.01
        Скорость обучения (learning rate) для обновления коэффициентов модели.

    C : float, default=1.0
        Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.

    n_epochs : int, default=100
        Количество эпох для обучения модели.

    batch_size : int, default=16
        Размер батча для mini-batch градиентного спуска (mini-batch GD).
    
    Атрибуты:
    ---------
    lr : float, default=0.01
        Скорость обучения (learning rate) для обновления коэффициентов модели.

    C : float, default=1.0
        Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.

    n_epochs : int, default=100
        Количество эпох для обучения модели.

    batch_size : int, default=16
        Размер батча для mini-batch градиентного спуска (mini-batch GD).

    fit_intercept : bool, по умолчанию True
        Включать ли свободный член (сдвиг) в модель.

    drop_last : bool, по умолчанию True
        Удалять ли последний неполный батч из обучения.

    coef_ : numpy.ndarray или None
        Коэффициенты (веса) модели размером (n_features, 1), которые обучаются на данных. Инициализируются как None до вызова метода `fit`.

    intercept_ : numpy.ndarray или None
        Свободный член (сдвиг) модели размером (1). Инициализируется как None до вызова метода `fit`.

    n_classes_ : int
        Количество классов.

    """

    def __init__(self, lr=0.01, C=1.0, n_epochs=100, batch_size=16, fit_intercept=True, drop_last=True):
        """
        Инициализация объекта класса LinearPrimalSVM с заданными гиперпараметрами.

        Параметры:
        ----------
        lr : float, default=0.01
            Скорость обучения (learning rate) для обновления коэффициентов модели.

        C : float, default=1.0
            Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.

        n_epochs : int, default=100
            Количество эпох для обучения модели.

        batch_size : int, default=16
            Размер батча для mini-batch градиентного спуска (mini-batch GD).

        fit_intercept : bool, по умолчанию True
            Включать ли свободный член (сдвиг) в модель.

        drop_last : bool, по умолчанию True
            Удалять ли последний неполный батч из обучения.
        """

        self.lr = lr
        self.C = C
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.drop_last = drop_last
        self.coef_ = None
        self.intercept_ = None
        self.n_classes_ = None


    def predict(self, X):
        """
        Предсказывает расстояние до разделяющей классы гиперплоскости для входных данных на основе обученной модели.

        Параметры:
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Входные данные для предсказания меток классов.

        Возвращает:
        ----------
        numpy.ndarray
            Вектор предсказанных расстояний, ориентированных по нормали к разделяющей гиперплоскости.
        """

        if self.coef_ is None:
            raise ValueError("invalid coef_")

        decision = np.dot(X, self.coef_)

        if self.fit_intercept and self.intercept_ is not None:
            decision += self.intercept_

        return decision


    def loss(self, X, y_true):
        """
        Вычисляет функцию потерь для бинарной классификации на основе HingeLoss
        с учетом L2 регуляризации

        Параметры:
        ----------
        X : numpy.ndarray
            Входной массив признаков размером (n_samples, n_features), где n_samples — количество образцов,
            а n_features — количество признаков.

        y_true : numpy.ndarray
            Вектор истинных меток классов.

        Возвращает:
        ----------
        float
            Значение функции потерь.
        """
        if self.coef_ is None:
            raise ValueError("invalid coef_")

        y_true = y_true.reshape(-1)

        predictions = self.predict(X)

        margins = y_true * predictions
        hinge_loss = np.maximum(0, 1 - margins).mean()

        l2_reg = 0.5 * np.dot(self.coef_.T, self.coef_).item()

        return hinge_loss + self.C * l2_reg


    def loss_grad(self, X, y_true):
        """
        Вычисляет градиент функции потерь по отношению к весам модели.

        В случае использования регуляризации, градиент включает соответствующие компоненты для
        штрафа за большие значения весов.

        Параметры:
        ----------

        X : numpy.ndarray
            Входной массив признаков размером (n_samples, n_features), где n_samples — количество образцов,
            а n_features — количество признаков.

        y_true : numpy.ndarray
            Вектор истинных меток классов.

        Возвращает:
        ----------
        grad : numpy.ndarray
            Градиент функции потерь по отношению к весам модели.

        grad_intercept : numpy.ndarray
            Градиент функции потерь по отношению к свободному члену.
        """
        if self.coef_ is None:
            raise ValueError("invalid coef_")

        y_true = y_true.reshape(-1, 1)

        margins = y_true * self.predict(X)

        indicator = (margins < 1).astype(float)

        grad = -np.dot(X.T, y_true * indicator) + self.C * self.coef_

        grad_intercept = 0
        if self.fit_intercept:
            grad_intercept = -np.sum(y_true * indicator)

        return grad, grad_intercept

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
        if self.fit_intercept:
            self.intercept_ -= self.lr * grad_intercept

    def fit(self, X, y):
        """
        Обучает модель SVM с использованием mini-batch градиентного спуска (mini-batch GD).

        Параметры:
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Тренировочные данные.

        y : numpy.ndarray, shape (n_samples,)
            Целевые метки классов.

        Возвращает:
        ----------
        self : LinearPrimalSVM
            Обученная модель.
        """

        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        if self.coef_ is None:
            self.coef_ = np.zeros((X.shape[1], 1))
        if self.intercept_ is None and self.fit_intercept:
            self.intercept_ = 0.0

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                if self.drop_last and end > n_samples:
                    break
                batch_indices = indices[start:end]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                grad, grad_intercept = self.loss_grad(X_batch, y_batch)

                self.step(grad, grad_intercept)
        return self

def one_vs_rest(y, n_classes=None):
    """
    Преобразует целевые метки в матрицу, где метки целевого класса
    принимают значение 1, а остальные метки — значение -1.

    Параметры:
    ----------
    y : numpy.ndarray или list
        Вектор или список меток классов, которые необходимо закодировать.
        Значения меток должны быть целыми числами от 0 до n_classes-1.

    n_classes : int или None, по умолчанию None
        Количество классов (размерность выходного пространства).
        Если None, то количество классов определяется автоматически как максимум значения в y плюс один.

    Возвращает:
    -----------
    numpy.ndarray
        Двумерная матрица размером (n_samples, n_classes), где для каждого образца целевой
        класс представлен значением 1, а все остальные классы имеют значение -1.

    """
    y = np.asarray(y)
    if n_classes is None:
        n_classes = np.max(y) + 1

    one_vs_rest_matrix = -1 * np.ones((y.shape[0], n_classes))

    one_vs_rest_matrix[np.arange(y.shape[0]), y] = 1

    return one_vs_rest_matrix


class LinearPrimalSVM:
    """
    Класс для построения модели многоклассовой классификации методом опорных 
    векторов путем решения прямой задачи оптимизации.

    Параметры:
    ----------
    lr : float, default=0.01
        Скорость обучения (learning rate) для обновления коэффициентов модели.

    C : float, default=1.0
        Параметр регуляризации, контролирующий баланс между максимизацией зазора 
        и минимизацией ошибок классификации.

    n_epochs : int, default=100
        Количество эпох для обучения модели.

    batch_size : int, default=16
        Размер батча для mini-batch градиентного спуска (mini-batch GD).
    
    Атрибуты:
    ---------
    lr : float, default=0.01
        Скорость обучения (learning rate) для обновления коэффициентов модели.

    C : float, default=1.0
        Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.

    n_epochs : int, default=100
        Количество эпох для обучения модели.

    batch_size : int, default=16
        Размер батча для mini-batch градиентного спуска (mini-batch GD).

    fit_intercept : bool, по умолчанию True
        Включать ли свободный член (сдвиг) в модель.

    drop_last : bool, по умолчанию True
        Удалять ли последний неполный батч из обучения.

    self.n_classes_ : int
        Количество классов, определяемое на основе уникальных меток в обучающем наборе данных.
        Этот параметр устанавливается после вызова метода `fit` и используется для определения 
        размерности выходного пространства модели. Он равен максимальному значению метки в данных плюс один.

    list_of_models : list
        Список, содержащий бинарные модели.
    """

    def __init__(self, lr=0.01, C=1.0, n_epochs=100, batch_size=16, fit_intercept=True, drop_last=True):
        """
        Инициализация объекта класса LinearPrimalSVM с заданными гиперпараметрами.

        Параметры:
        ----------
        lr : float, default=0.01
            Скорость обучения (learning rate) для обновления коэффициентов модели.

        C : float, default=1.0
            Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.

        n_epochs : int, default=100
            Количество эпох для обучения модели.

        batch_size : int, default=16
            Размер батча для mini-batch градиентного спуска (mini-batch GD).

        fit_intercept : bool, по умолчанию True
            Включать ли свободный член (сдвиг) в модель.

        drop_last : bool, по умолчанию True
            Удалять ли последний неполный батч из обучения.
        """

        self.lr = lr
        self.C = C
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.drop_last = drop_last
        self.n_classes_ = None
        self.list_of_models = []


    def predict(self, X):
        """
        Предсказывает метки классов для входных данных на основе обученной модели.

        Параметры:
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Входные данные для предсказания меток классов.

        Возвращает:
        ----------
        numpy.ndarray
            Вектор предсказанных меток классов (значения от 0 до n_classes-1).
        """
        if not self.list_of_models:
            raise ValueError("Модель не была обучена. Сначала используйте метод `fit`.")

        n_samples = X.shape[0]
        scores = np.zeros((n_samples, self.n_classes_))

        for i, model in enumerate(self.list_of_models):
            scores[:, i] = model.predict(X).flatten()

        return np.argmax(scores, axis=1)


    def fit(self, X, y):
        """
        Обучает модель SVM с использованием mini-batch градиентного спуска (mini-batch GD).

        Параметры:
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Тренировочные данные.

        y : numpy.ndarray, shape (n_samples,)
            Целевые метки классов.

        Возвращает:
        ----------
        self : LinearPrimalSVM
            Обученная модель.
        """
        y_encoded = one_vs_rest(y, n_classes=self.n_classes_)
        self.n_classes_ = y_encoded.shape[1]
        self.list_of_models = []

        for i in range(self.n_classes_):
            model = BinaryEstimatorSVM(lr=self.lr, C=self.C, n_epochs=self.n_epochs, batch_size=self.batch_size)

            model.fit(X, y_encoded[:, i])

            self.list_of_models.append(model)

        return self
    
    
def kernel_linear(x1, x2):
    """
    Линейное ядро для SVM.

    Вычисляет скалярное произведение двух векторов, что соответствует линейной 
    границе разделения в пространстве признаков.

    Параметры:
    ----------
    x1 : np.array, shape (n_features,)
        Первый вектор признаков.
    
    x2 : np.array, shape (n_features,)
        Второй вектор признаков.

    Возвращает:
    ----------
    float
        Скалярное произведение векторов x1 и x2.
    """
    return np.dot(x1, x2)


def kernel_poly(x1, x2, d=2):
    """
    Полиномиальное ядро для SVM.

    Вычисляет полиномиальное скалярное произведение двух векторов, 
    что позволяет моделировать нелинейные границы разделения.

    Параметры:
    ----------
    x1 : np.array, shape (n_features,)
        Первый вектор признаков.
    
    x2 : np.array, shape (n_features,)
        Второй вектор признаков.
    
    d : int, default=2
        Степень полинома.

    Возвращает:
    ----------
    float
        Полиномиальное скалярное произведение векторов x1 и x2.
    """
    return (np.dot(x1, x2) + 1) ** d

def kernel_rbf(x1, x2, l=1.0):
    """
    Радиально-базисное (гауссовское) ядро для SVM.

    Вычисляет расстояние между двумя векторами с использованием радиально-базисной функции (RBF),
    которая позволяет моделировать сложные нелинейные зависимости.

    Параметры:
    ----------
    x1 : np.array, shape (n_features,)
        Первый вектор признаков.
    
    x2 : np.array, shape (n_features,)
        Второй вектор признаков.
    
    l : float, default=1.0
        Параметр ширины гауссовской функции (коэффициент сглаживания).

    Возвращает:
    ----------
    float
        Значение RBF-ядра между векторами x1 и x2.
    """
    distance = np.linalg.norm(x1 - x2)
    return np.exp(- (distance ** 2) / (2 * (l ** 2)))

def lagrange(gramm_matrix, alpha):
    """
    Двойственная функция Лагранжа для SVM.

    Вычисляет двойственную функцию для оптимизации SVM с использованием
    заранее рассчитанной матрицы Грамма.

    Параметры:
    ----------
    gramm_matrix : np.array, shape (n_samples, n_samples)
        Матрица Грамма (значения ядер между всеми парами обучающих объектов).
    
    alpha : np.array, shape (n_samples,)
        Двойственные переменные (лямбда), используемые для оптимизации.

    Возвращает:
    ----------
    float
        Значение двойственной функции Лагранжа.
    """
    return np.sum(alpha) - 0.5 * np.dot(alpha, np.dot(gramm_matrix, alpha))

def lagrange_derive(gramm_matrix, alpha):
    """
    Производная двойственной функции Лагранжа по alpha.

    Вычисляет градиент (производную) двойственной функции Лагранжа,
    что необходимо для решения задачи оптимизации.

    Параметры:
    ----------
    gramm_matrix : np.array, shape (n_samples, n_samples)
        Матрица Грама (значения ядер между всеми парами обучающих объектов).
    
    alpha : np.array, shape (n_samples,)
        Двойственные переменные (лямбда), используемые для оптимизации.

    Возвращает:
    ----------
    np.array, shape (n_samples,)
        Градиент двойственной функции по alpha.
    """
    gradient = np.ones_like(alpha) - np.dot(gramm_matrix, alpha)
    return gradient

def one_vs_one(X, y, n_classes=None):
    """
    Преобразует целевые метки в матрицу, где метки первого класса
    принимают значение 1, а метки второго — значение -1.

    Параметры:
    ----------
    y : numpy.ndarray
        Вектор или список меток классов, которые необходимо закодировать.
        Значения меток должны быть целыми числами от 0 до n_classes-1.

    n_classes : int или None, по умолчанию None
        Количество классов (размерность выходного пространства).
        Если None, то количество классов определяется автоматически как максимум значения в y плюс один.

    Возвращает:
    -----------
    list of tuples
        (X_cut, y_cut (Бинарный таргет 1 или -1), соответствующий '1' класс, соответствующий '-1' класс)
        
    """
    if n_classes is None:
        n_classes = np.max(y) + 1

    results = []

    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            mask = (y == i) | (y == j)
            X_cut = X[mask]
            y_cut = y[mask]

            y_cut = np.where(y_cut == i, 1, -1)

            results.append((X_cut, y_cut, i, j))

    return results


class SoftMarginSVM:
    """
    Реализация SVM с мягким зазором (Soft Margin SVM) с возможностью использовать произвольные ядра.
    
    Атрибуты:
    ----------
    C : float, default=1.0
        Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.
    
    alpha : np.array, shape (n_samples,)
        Двойственные переменные (лямбда) для решения задачи оптимизации.

    supportVectors : np.array, shape (n_support_vectors, n_features)
        Опорные вектора — обучающие объекты, которые оказывают влияние на разделяющую гиперплоскость.

    supportLabels : np.array, shape (n_support_vectors,)
        Метки классов для опорных векторов.

    supportalpha : np.array, shape (n_support_vectors,)
        Значения альфа (лямбда) для опорных векторов.

    kernel : function
        Ядро для вычисления скалярных произведений в пространстве признаков.

    classes_names : list or array-like, shape (2,)
        Имена классов. Используются для преобразования предсказанных значений {-1, 1} в имена классов.
    
    b: float 
        Смещение.

    Методы:
    -------
    fit(X, y):
        Обучает модель.

    predict(X):
        Предсказывает метки классов для входных данных.
    """
    
    def __init__(self, C, kernel_func, classes_names=None):
        """
        Инициализирует модель Soft Margin SVM.
        
        Параметры:
        ----------
        C : float, default=1.0
            Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.
        
        kernel_func : function
            Функция ядра, определяющая метод вычисления скалярных произведений в новом пространстве признаков.
        
        classes_names : list
            Список имен классов. Ожидается, что в обучающих данных метки классов {-1, 1}.
        """
        self.C = C                                 
        self.alpha = None
        self.supportVectors = None
        self.supportLabels = None
        self.supportalpha = None
        self.kernel = kernel_func
        self.classes_names = classes_names
        self.b = None

    
    def fit(self, X, y):
        """
        Обучает модель с использованием оптимизации двойственной задачи для SVM.
        
        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Обучающие данные (матрица признаков).
        
        y : np.array, shape (n_samples,)
            Вектор меток классов, должен содержать значения {-1, 1}.
        
        Возвращает:
        ----------
        self : SoftMarginSVM
            Обученная модель.
        """
        N = len(y)
        Xy = X * y.reshape(-1, 1)
        GramXy = np.matmul(Xy, Xy.T)

        first_constraint = {'type': 'eq', 'fun': lambda a: np.dot(a, y), 'jac': lambda a: y}
        A = np.vstack((-np.eye(N), np.eye(N)))
        b = np.hstack((np.zeros(N), self.C * np.ones(N)))

        second_constraint = {'type': 'ineq', 'fun': lambda a: b - np.dot(A, a), 'jac': lambda a: -A}
        constraints = (first_constraint, second_constraint)

        opt_res = optimize.minimize(fun=lambda a: -lagrange(GramXy, a),
                                    x0=np.ones(N),
                                    method='SLSQP',
                                    jac=lambda a: -lagrange_derive(GramXy, a),
                                    constraints=constraints)

        self.alpha = opt_res.x
        self.w = np.sum((self.alpha.reshape(-1, 1) * Xy), axis=0)

        valid_indices = (self.alpha >  1e-6) & (self.alpha <= self.C)
        self.supportVectors = X[valid_indices]
        supportLabels = y[valid_indices]

        if len(self.supportVectors) > 0:
            b_values = []
            for x_s, y_s in zip(self.supportVectors, supportLabels):
                b_values.append(y_s - np.dot(self.w, x_s))
            self.b = np.mean(b_values)
        else:
            print("No valid support vectors to compute bias.")
            self.b = 0
        return self

    def predict(self, X):
        """
        Предсказывает метки классов для входных данных.
        
        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Массив объектов для предсказания.
        
        Возвращает:
        ----------
        np.array, shape (n_samples,)
            Вектор предсказанных меток классов, где метки соответствуют значениям из `classes_names`.
        """
        assert (self.w is not None)
        assert (self.w.shape[0] == X.shape[1])
        # по формуле 2(w^Tx + b) - 1 получаем значение объекта +1 или -1
        return 2 * (np.matmul(X, self.w) + self.b > 0) - 1

class NonLinearDualSVM:
    """
    NonLinearDualSVM реализует SVM one-vs-one с использованием двойственной задачи. 
    Поддерживает использование различных ядерных функций для задач классификации.

    Атрибуты:
    ---------
    estimators : list или None
        Список бинарный SVM моделе one-vs-one

    C : float, default=1.0
        Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.

    kernel : str, default='rbf'
        Ядерная функция, используемая в модели (возможные значения: 'poly', 'rbf', 'linear').

    """

    def __init__(self, C=1.0, kernel='rbf', kernel_parameter=1.0):
        """
        Инициализирует модель NonLinearDualSVM с указанной ядерной функцией.
        
        Параметры:
        ----------
        C : float, default=1.0
            Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.

        kernel : str, default='rbf'
            Ядерная функция, используемая в модели (возможные значения: 'poly', 'rbf', 'linear').

        kernel_parameter : float, default=1.0
            Гиперпарметр ядра
      
        """
        self.C = C

        if kernel == 'poly':
          self.kernel = lambda x, y: kernel_poly(x, y, d=kernel_parameter)
        elif kernel == 'rbf':
          self.kernel = lambda x, y: kernel_rbf(x, y, l=kernel_parameter)
        else:
          self.kernel = kernel_linear

        self.kernel.__name__='kernel'

        self.estimators = []

    def predict(self, X):
        """
        Предсказывает метки классов для входных данных X.
        
        Параметры:
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Входные данные для предсказания меток классов.

        Возвращает:
        -------
        numpy.ndarray, shape (n_samples,)
            Предсказанные метки классов для каждого образца.
        """

        arr = X.shape[0]
        array = []

        i = 0
        while i < arr:
            accuracy = {}
            for estimator in self.estimators:
                predict = tuple(estimator.predict(X[i].reshape(1, -1))[0])
                if predict in accuracy:
                    accuracy[predict] = accuracy[predict] + 1
                else:
                    accuracy[predict] = 1

            array.append(max(accuracy.items(), key=lambda x: x[1])[0])
            i += 1

        result = np.array(array)
        return result.flatten()

    def fit(self, X, y):
        """
        Обучает модель SVM на тренировочных данных (X, y) с использованием двойственной задачи.
        
        Параметры:
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Тренировочные данные.

        y : numpy.ndarray, shape (n_samples,)
            Целевые метки классов.

        Возвращает:
        -------
        self : NonLinearDualSVM
            Обученная модель.
        """
        
        classes = np.max(y) + 1
        one = one_vs_one(X, y, classes)

        for X_subset, y_subset, class1, class2 in one:
            svm_model = SoftMarginSVM(self.C, self.kernel, [class1, class2])
            svm_model.fit(X_subset, y_subset)

            self.estimators.append(svm_model)

        return self
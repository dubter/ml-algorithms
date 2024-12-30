import numpy as np
from sklearn import clone
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.ensemble import GradientBoostingClassifier


class RandomForestClassifier:
    """
    Модель случайного леса (RandomForest).

    Атрибуты:
    ----------
    n_estimators : int
        Количество деревьев в лесу.
    
    bootstrap : bool
        Используется ли бутстрап при построении деревьев. 
        Если False, то для каждого дерева используется весь набор данных.
    
    estimators : list
        Список деревьев с заданными параметрами (**kwargs).
    
    kwargs : dict
        Параметры для каждого дерева, такие как min_samples_split, max_depth и др.
        Передаются в DecisionTreeClassifier.
    """
    def __init__(self, n_estimators=100, bootstrap=True, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_feature='sqrt', **kwargs):
        self.bootstrap = bootstrap
        self.n_estimators = n_estimators
        self.estimators = []
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_feature = max_feature
        self.kwargs = kwargs
    
    def fit(self, X, y):
        """
        Обучает случайный лес на тренировочной выборке (X, y). 
        Если bootstrap=False, используется весь набор данных для каждого дерева,
        если bootstrap=True, используются бутстрап-выборки.

        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Тренировочные данные (признаки).
        
        y : np.array, shape (n_samples,)
            Целевые метки классов.

        Возвращает:
        ----------
        self : RandomForestClassifier
            Обученная модель.
        """
        for _ in range(self.n_estimators):
            if self.bootstrap:
                X_sample, y_sample = resample(X, y)
            else:
                X_sample, y_sample = X, y

            tree = DecisionTreeClassifier(**self.kwargs)
            tree.fit(X_sample, y_sample)
            self.estimators.append(tree)

        return self

    def predict_proba(self, X):
        """
        Предсказывает вероятности классов для каждого объекта на основе 
        предсказаний всех деревьев в лесу. Возвращает средние вероятности по всем деревьям.

        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Входные данные для предсказания вероятностей классов.

        Возвращает:
        ----------
        np.array, shape (n_samples, n_classes)
            Вероятности для каждого класса и каждого объекта.
        """
        proba_predictions = np.array([estimator.predict_proba(X) for estimator in self.estimators])
        return proba_predictions.mean(axis=0)

    def predict(self, X):
        """
        Предсказывает метки классов для каждого объекта входной выборки 
        на основе голосования всех деревьев.

        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Входные данные для предсказания меток классов.

        Возвращает:
        ----------
        np.array, shape (n_samples,)
            Вектор предсказанных меток классов.
        """
        return np.argmax(self.predict_proba(X), axis=1)


class BlendingClassifier:
    """
    Модель ансамбля методом блендинга (Blending).

    Атрибуты:
    ----------
    estimators : list
        Список инициализированных базовых моделей.

    final_estimator : объект модели
        Метамодель, обучаемая на предсказаниях базовых моделей.

    test_size : float
        Доля данных, используемая для обучения метамодели (блендинга).

    """

    def __init__(self, estimators=None, final_estimator=None, test_size=0.2):
        if estimators is None:
            self.estimators = [
                LogisticRegression(random_state=42, max_iter=200, solver="newton-cg"),
                KNeighborsClassifier(),
                DecisionTreeClassifier(random_state=42)
            ]
        else:
            self.estimators = estimators

        if final_estimator is None:
            self.final_estimator = LogisticRegression(random_state=42, max_iter=200, solver='lbfgs')
        else:
            self.final_estimator = final_estimator
        self.test_size = test_size

    def fit(self, X, y):
        """
        Разделяет входную выборку на тренировочную и валидационную части.
        Базовые модели обучаются на тренировочной части, а метамодель — на предсказаниях
        базовых моделей на валидационной части.

        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Тренировочные данные (признаки).

        y : np.array, shape (n_samples,)
            Целевые метки классов.

        Возвращает:
        ----------
        self : Blending
            Обученная модель.
        """
        self.kClasses = len(np.unique(y))
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.test_size, random_state=42)
        for estimator in self.estimators:
            estimator.fit(X_train, y_train)

        X_meta_train = np.hstack([
            estimator.predict_proba(X_val)[:, :self.kClasses - 1] for estimator in self.estimators
        ])
        self.final_estimator.fit(X_meta_train, y_val)

        return self

    def predict_proba(self, X):
        """
        Предсказывает вероятности классов с использованием базовых моделей,
        передает их метамодели для получения финального предсказания.

        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Входные данные для предсказания вероятностей классов.

        Возвращает:
        ----------
        np.array, shape (n_samples, n_classes)
            Предсказанные вероятности для каждого класса.
        """
        X_meta = np.column_stack([
            estimator.predict_proba(X)[:, :self.kClasses - 1] for estimator in self.estimators
        ])
        return self.final_estimator.predict_proba(X_meta)

    def predict(self, X):
        """
        Предсказывает метки классов на основе предсказаний базовых моделей
        и метамодели.

        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Входные данные для предсказания меток классов.

        Возвращает:
        ----------
        np.array, shape (n_samples,)
            Вектор предсказанных меток классов.
        """
        X_meta = np.column_stack([
            estimator.predict_proba(X)[:, :self.kClasses - 1] for estimator in self.estimators
        ])
        return self.final_estimator.predict(X_meta)


class StackingClassifier:
    """
    Модель ансамбля методом стекинга (Stacking).

    Атрибуты:
    ----------
    estimators : list
        Список инициализированных базовых моделей.

    final_estimator : объект модели
        Метамодель, обучаемая на мета-признаках (предсказаниях базовых моделей).

    folds : int
        Количество фолдов для кросс-валидации при обучении базовых моделей.

    """

    def __init__(self, estimators=None, final_estimator=None, folds=5):
        if estimators is None:
            self.estimators = [
                DecisionTreeClassifier(random_state=42),
                GradientBoostingClassifier(n_estimators=100, random_state=42),
                # ('svm', SVC(probability=True, random_state=42)),
                GaussianNB()
            ]
        else:
            self.estimators = estimators

        if final_estimator is None:
            self.final_estimator = LogisticRegression(random_state=42)
        else:
            self.final_estimator = final_estimator

        self.folds = folds
        self.base_models = []
    def fit(self, X, y):
        """
        Обучает базовые модели на тренировочных фолдах и использует
        их предсказания на валидационных фолдах для обучения метамодели.
        Применяется кросс-валидация с заданным количеством фолдов.

        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Тренировочные данные (признаки).

        y : np.array, shape (n_samples,)
            Целевые метки классов.

        Возвращает:
        ----------
        self : Stacking
            Обученная модель.
        """
        kf = KFold(n_splits=self.folds, shuffle=True, random_state=42)
        n_classes = len(np.unique(y))
        self.n_classes = n_classes
        meta_features = np.zeros((X.shape[0], len(self.estimators) * (n_classes - 1)))

        for i, model in enumerate(self.estimators):
            model_meta_features = np.zeros((X.shape[0], n_classes - 1))
            for train_index, val_index in kf.split(X):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                model.fit(X_train, y_train)
                model_meta_features[val_index] = model.predict_proba(X_val)[:, :n_classes - 1]
            print(model_meta_features.shape)
            meta_features[:, i * (n_classes - 1):(i + 1) * (n_classes - 1)] = model_meta_features

            # Обучение модели на всем датасете для инференса
            model.fit(X, y)
            self.base_models.append(model)
            # print(name)
        # print(meta_features)
        self.final_estimator.fit(meta_features, y)
        return self

    def predict_proba(self, X):
        """
        Предсказывает вероятности классов с помощью базовых моделей,
        передает их метамодели для получения финального предсказания.

        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Входные данные для предсказания вероятностей классов.

        Возвращает:
        ----------
        np.array, shape (n_samples, n_classes)
            Предсказанные вероятности для каждого класса.
        """
        meta_features = np.column_stack([model.predict_proba(X)[:, :self.n_classes - 1] for model in self.base_models])
        return self.final_estimator.predict_proba(meta_features)

    def predict(self, X):
        """
        Предсказывает метки классов на основе предсказаний базовых моделей
        и метамодели.

        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Входные данные для предсказания меток классов.

        Возвращает:
        ----------
        np.array, shape (n_samples,)
            Вектор предсказанных меток классов.
        """
        return self.final_estimator.predict(np.column_stack([model.predict_proba(X)[:, :self.n_classes - 1] for model in self.base_models]))

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
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

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
    y = np.asarray(y)
    if n_classes is None:
        n_classes = np.max(y) + 1
    one_hot = np.zeros(len(y), n_classes)
    one_hot[np.arrange(len(y))] = 1
    return one_hot


class BoostingClassifier:
    """
    Модель Бустинга (BoostingClassifier).

    Атрибуты:
    ----------
    n_estimators : int
        Количество деревьев в ансамбле.
    
    bootstrap : bool
        Используется ли бутстрап при построении деревьев. 
        Если False, то для каждого дерева используется весь набор данных.
    
    estimators : list
        Список деревьев с заданными параметрами (**kwargs).
    
    kwargs : dict
        Параметры для каждого дерева, такие как min_samples_split, max_depth и др.
        Передаются в DecisionTreeRegressor.
    """
    def __init__(self, n_estimators=100, bootstrap=True, lr=0.1, **kwargs):
        self.bootstrap = bootstrap
        self.n_estimators = n_estimators
        self.lr = lr
        self.kwargs = kwargs
        self.estimators = []
    
    def fit(self, X, y):
        """
        Обучает ансамбль на тренировочной выборке (X, y). 
        Если bootstrap=False, используется весь набор данных для каждого дерева,
        если bootstrap=True, используются бутстрап-выборки.

        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Тренировочные данные (признаки).
        
        y : np.array, shape (n_samples,)
            Целевые метки классов.

        Возвращает:
        ----------
        self : BoostingClassifier
            Обученная модель.
        """
        n_samples, n_classes = X.shape[0], len(np.unique(y))

        self.F = np.zeros((n_samples, n_classes))

        for _ in range(self.n_estimators):
            # Compute gradients
            probs = softmax(self.F)
            gradients = probs - np.eye(n_classes)[y]

            trees = []
            for k in range(n_classes):
                if self.bootstrap:
                    X_boot, grad_boot = resample(X, gradients[:, k])
                else:
                    X_boot, grad_boot = X, gradients[:, k]

                tree = DecisionTreeRegressor(**self.kwargs)
                tree.fit(X_boot, grad_boot)
                trees.append(tree)

            self.estimators.append(trees)

            for k in range(n_classes):
                self.F[:, k] -= self.lr * self.estimators[-1][k].predict(X)

        return self
        

    def predict_proba(self, X):
        """
        Предсказывает вероятности классов для каждого объекта входной выборки.

        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Входные данные для предсказания вероятностей классов.

        Возвращает:
        ----------
        np.array, shape (n_samples, n_classes)
            Вероятности для каждого класса и каждого объекта.
        """
        n_samples = X.shape[0]
        n_classes = len(self.estimators[0])

        F = np.zeros((n_samples, n_classes))

        for trees in self.estimators:
            for k, tree in enumerate(trees):
                F[:, k] -= self.lr * tree.predict(X)

        return softmax(F)
        
    def predict(self, X):
        """
        Предсказывает метки классов для каждого объекта входной выборки.

        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Входные данные для предсказания меток классов.

        Возвращает:
        ----------
        np.array, shape (n_samples,)
            Вектор предсказанных меток классов.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

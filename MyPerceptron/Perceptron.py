import numpy as np


class MyPerceptron:
    def __init__(self, number_neuron, activation="logistic"):
        '''
        :param number_neuron: Содержит количество связей между нейронов (синапсов)
        '''
        # Веса синапсов
        self.synaptic_weights = np.random.sample((number_neuron, 1))
        self.activation = activation

    def __logistic__(x):
        '''
        :param x: Содержит входной набор данных для обучения или одно число для прогноза
        :return: Возвращает результат выполнения логистической функции для переменной x
        '''
        return 1 / (1 + np.exp(-x))

    def __relu__(x):  # Плохо работает с relu
        '''
        :param x: Содержит входной набор данных для обучения или одно число для прогноза
        :return: Возвращает результат выполнения функции MAX(0, x) для переменной x
        '''
        return np.clip(x, 0, np.finfo(x.dtype).max, out=x)

    ACTIVATIONS = {'logistic': __logistic__,
                   'relu': __relu__}

    def fit(self, X, y, eras=2000):
        '''
        :param X: Содержит входной набор данных для обучения
        :param y: Содержит выходной набор данных для обучения
        :param eras: Количество эпох(итераций) обучения
        :return:
        '''
        # Метод обратного распространения ошибки
        for i in range(eras):
            input_layer = X
            output = self.ACTIVATIONS[self.activation](np.dot(input_layer, self.synaptic_weights))

            error = y - output
            adjustment = np.dot(input_layer.T, error * (output * (1 - output))) # Проблема плохой работы с активациоными функциями находится тут

            self.synaptic_weights += adjustment
        print(output)

    def prediction(self, X):
        '''
        :param X: Содержит данные по которым нужно сделать прогноз
        :return: Возвращает число от 0 до 1
        '''
        return self.ACTIVATIONS[self.activation](np.dot(X, self.synaptic_weights))

    def get_weights(self):
        '''
        :return: Возвращает значение весов всех синапсов
        '''
        return self.synaptic_weights

    def set_weights(self, synaptic_weights):
        '''
        :param synaptic_weights: Содержит веса синапсов
        '''
        self.synaptic_weights = synaptic_weights


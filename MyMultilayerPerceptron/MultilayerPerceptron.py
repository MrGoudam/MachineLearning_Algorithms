# -*- coding: utf-8 -*-
import numpy as np
from Functions import ACTIVATIONS, GRADIENTS


class Layer:
    """
    Класс Layer предназначен для хранения слоев нейронной сети

    Параметры:
        __input_layer: Внутренний параметр класса. Нужен лишь для того, чтобы обоозначать, является ли слой входным.
        number_neurons: Хранит количество нейронов в слое
        w: Хранит веса синапсов между этим слоем и предедущим
        activation: Хранит название ативиационной функции данного слоя

    Методы:
        get_value: Возвращает параметры класса в виде словоря
    """
    def __init__(self, number_neurons, input_layer, number_neurons_previous_layer=None, activation=None):
        self.__input_layer = input_layer

        self.number_neurons = number_neurons

        if not input_layer:
            self.w = np.random.uniform(-0.5, 0.5, (number_neurons, number_neurons_previous_layer))
            self.activation = activation

    def get_value(self):
        """
        :return: Возвращает параметры класса в виде словоря
        """
        if not self.__input_layer:
            return {'w': self.w,
                    'number_neurons': self.number_neurons,
                    'activation': self.activation}
        else:
            return {'w': None,
                    'number_neurons': self.number_neurons,
                    'activation': None}


class MultilayerPerceptron:
    """
    Класс MultilayerPerceptron предназначен для послойной реализации многослойного песептрона

    Параметры:
        architecture: Хранит архитектуру многослойного персептрона в виде списка слоев
        number_of_layers: Хранит количество слоев в архитектуре

    Методы:
        add_layer: Добавляет слой в архитектуру
        __signal_accumulation: Сумма произведения весов и сигналов для каждого слоя
        __kernels: Прямое распространения сигнала
        __gradient: Вычисление градиент для слоев
        __errors: Вычисление ошибки для слоев
        __verification_of_compliance: Вспомогательный метод для проверки входных и выходных сигналов
        fit: Обучение нейронной сети
        predict: Прогноз для новых значений
        show_architecture: Послойный вывод архитектуры нейронной сети
        save: Сохранение архитектуры в файл
        load: Загрузка архитектуры из файла
    """
    def __init__(self):
        self.architecture = []
        self.number_of_layers = 0

    def add_layer(self, number_neurons, input_layer=False, activation='logistic'):
        """
        Добавляет слой нейронной сети в архитектуру

        :param number_neurons: Хранит количество нейронов в слое
        :param input_layer: Содержит булевое значение, которое являетя ответом на вопрос: "Является ли слой входным?"
        :param activation:  Хранит название ативиационной функции данного слоя
        """
        if not input_layer:
            layer = Layer(number_neurons, input_layer, self.architecture[-1]['number_neurons'],  activation).get_value()
        else:
            layer = Layer(number_neurons, input_layer).get_value()

        self.architecture.append(layer)
        self.number_of_layers += 1

    def __signal_accumulation(self, x):
        """
        Вычисление суммы произведения весов и сигналов для каждого слоя

        :param x: Хранит сигнал, который поступает на входной слой
        :return: Возвращает сумму произведения весов и сигналов (w * x) для каждого слоя в виде списка
        """
        res = []
        for i in range(1, self.number_of_layers):
            x = np.dot(self.architecture[i]['w'], x)
            res.append(x)
        return res

    def __kernels(self, x):
        """
        Прямое распространения сигнала

        :param x: Хранит сигнал, который поступает на входной слой
        :return: Возвращает сигнал (результат метода __signal_accumulation) прошедший через активиационную функуию
        """
        signals = self.__signal_accumulation(x)
        for i in range(1, self.number_of_layers):
            signals[i - 1] = ACTIVATIONS[self.architecture[i]['activation']](signals[i-1])
        return signals

    def __gradient(self, x):
        """
        Вычисление градиент для слоев

        :param x: Хранит сигнал, который поступает на входной слой
        :return: Возвращает сигнал (результат метода __signal_accumulation) прошедший через градиент активиационной функции
        """
        signals = self.__signal_accumulation(x)
        for i in range(1, self.number_of_layers):
            signals[i - 1] = GRADIENTS[self.architecture[i]['activation']](signals[i - 1])
        return signals

    def __errors(self, y, output_y):
        """
        Вычисление ошибки для слоев

        :param y: Теоритический выходной сигнал
        :param output_y: Фактический выходной сигнал
        :return: Список ошибок для каждого нейрона
        """
        errors = []

        error = y - output_y
        errors.insert(0, error)

        for i in reversed(range(2, self.number_of_layers)):
            error = np.dot(self.architecture[i]['w'].T, error)
            errors.insert(0, error)

        return errors

    def __verification_of_compliance(self, x, y=[]):
        """
        Вспомогательный метод для проверки:
            - количество входных сигналов сотвестует количеству нейронов во входном слое
            - количество ответов соответствует количеству нейронов в выходном слое
        :param x: Массив входных сигналов
        :param y: Массив выходных сигналов
        """
        for i in x:
            if len(i) != self.architecture[0]['number_neurons']:
                raise ValueError('The number of input signals does not match the number of neurons in the input layer')
        if y != []:
            for i in y:
                if len(i) != self.architecture[self.number_of_layers-1]['number_neurons']:
                    raise ValueError('The number of responses does not match the number of neurons in the output layer')

    def fit(self, X, Y, learning_rate=0.5, epoch=1000):
        """
        Обучение нейронной сети

        :param X: Массив наборов сигналов для обучения
        :param Y: Массив наборов ответов для обучения
        :param learning_rate: Скорость обучения нейронной сети
        :param epoch: Количество эпох обучения
        """
        self.__verification_of_compliance(X, Y)

        for e in range(epoch):
            for (x, y) in zip(X, Y):
                kernel = self.__kernels(x)
                error = self.__errors(y, kernel[-1])
                gradient = self.__gradient(x)

                for i in range(len(error)):
                    error[i] *= gradient[i]
                    error[i].shape = (len(error[i]), 1)

                self.architecture[1]['w'] += learning_rate * error[0] * x

                for i in range(2, self.number_of_layers):
                    self.architecture[i]['w'] += learning_rate * error[i-1] * kernel[i-2]

    def predict(self, x):
        """
        Прогноз для новых значений

        :param x: Сигнал для которого нужно сделать прогноз
        :return: Возвращает сигнал с последнего слоя нейронной сети
        """
        self.__verification_of_compliance([x])
        return self.__kernels(x)[-1]

    def show_architecture(self):
        """
        Послойный вывод архитектуры нейронной сети
        """
        for (i, layer) in enumerate(self.architecture):
            if i == 0:
                print('Input layer:')
                for el in layer:
                    print('\t', el, ':', layer[el])
            else:
                if i == self.number_of_layers-1:
                    print('Output layer:')
                else:
                    print('Hidden layer', i, ':')

                for el in layer:
                    if el == 'w':
                        print('\t', el, ':')
                        for j in layer[el]:
                            print('\t  ', j)
                    else:
                        print('\t', el, ':', layer[el])
            print()

    def save(self, file='architecture', path=''):
        """
        Сохранение архитектуры в файл в виде массива. Файл имеет расширение .npy

        :param file: Название файла
        :param path: Путь по которому лежит файл
        """
        np.save(path+file, self.architecture)

    def load(self, file='architecture.npy', path=''):
        """
        Загрузка архитектуры из файла

        :param file: Название файла
        :param path: Путь по которому лежит файл
        """
        self.architecture = np.load(path+file, allow_pickle=True).tolist()
        self.number_of_layers = len(self.architecture)

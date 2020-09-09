import numpy as np
import Perceptron

if __name__ == '__main__':

    # Тренировочные данные
    training_input = np.array([[0, 0, 1],
                               [1, 0, 1],
                               [0, 1, 1],
                               [1, 0, 0]])

    training_output = np.array([[0, 1, 0, 1]]).T

    # Работа с персептроном
    perceptron = Perceptron.MyPerceptron(len(training_input[0]), activation='logistic')

    print("Сгенерированные веса:")
    print(perceptron.get_weights())

    perceptron.fit(training_input, training_output, 2000)

    print("Веса после обучения:")
    print(perceptron.get_weights())

    test_input = np.array([1, 1, 0])
    print('Прогноз для новой ситуации:')
    print(perceptron.prediction(test_input))


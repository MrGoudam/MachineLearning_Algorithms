# -*- coding: utf-8 -*-
import numpy as np
import MultilayerPerceptron

if __name__ == '__main__':
    x = np.array([[0, 0, 1],
                  [1, 0, 1],
                  [0, 1, 1],
                  [1, 0, 0]])
    y = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])

    mp = MultilayerPerceptron.MultilayerPerceptron()

    mp.add_layer(3, input_layer=True)
    mp.add_layer(2)
    mp.add_layer(3)
    mp.add_layer(2, activation='softmax')

    mp.fit(x, y, epoch=1000)

    mp.save('model.npy')
    mp.load('model.npy')

    print(mp.predict([0, 1, 0]))




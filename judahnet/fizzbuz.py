"""
FizzBuzz is the following problem:

For each of the numbers 1 to 100:
*if the number is divisible by 3, print "fizz"
*if the number is divisible by 5, print "buzz"
*if the number if divisible by 15, print "fizzbuzz"
*otherwise, just print the number
"""
from typing import List
import numpy as np

from judahnet.train import train
from judahnet.nn import NeuralNet
from judahnet.layers import Linear, Tanh
from judahnet.optim import SGD

def fizz_buzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


def binary_encode(x: int) -> List[int]:

    return [x >> i & 1 for i in range(10)]

input = np.array([
    binary_encode(x)
    for x in range(101, 1024)

])

target = np.array([
    fizz_buzz_encode(x)
    for x in range(101, 1024)
])

net = NeuralNet([
    Linear(input_size=10, output_size=50),
    Tanh(),
    Lneat(input_size=50, output_size=4)
])

train(net,
      inputs,
      targets,
      num_epochs=50,
      optimizer=SGD(lr=0.001)
      )
    for x in range(1, 101):
        predicted = net.forward(binary_encode(x))
        predicted_idx = np.argmax(predicted)
        actual_idx = np.argmax(fizz_buzz_encode(x))
        labels = [str(x), "fizz", "bizz", "fizzbizz"]
        print(x, labels[predicted_idx], labels[actual_idx])
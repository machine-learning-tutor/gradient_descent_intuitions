from __future__ import print_function
from pprint import pprint

n_dim = 14
FILE_NAME = "boston_data.csv"
mini_batch_size = 5

def read_boston_data(file_name):
    with open(file_name, "r") as f:
        f.readline()
        lines = f.read().split("\n")


    X, Y = [], []
    for l in lines:
        if len(l) < 1: continue
        numbers = [float(i) for i in l.split(",")]
        x = [1, ]
        y = numbers[-1]
        x.extend(numbers[: -1])
        X.append(x)
        Y.append(y)
        
    return X, Y    
        

def cost(weights, X, Y):
    c = 0
    for x, y in zip(X, Y):
        c += (sum([x_ * w_ for x_, w_ in zip(x, weights)]) - y)**2
    return c / len(X)
    

X, Y = read_boston_data(FILE_NAME)
n_dim = len(X[0]) # В нашем конкретном случае это 14

weights = [0] * 14
print("Cost:", cost(weights, X, Y))
N = 10**4
lr = 0.000003

mini_batch_size = 50
for i in range(N):
    for mini_batch_i in range(len(X) // mini_batch_size):
        gr = [0] * n_dim
        for dim in range(0, n_dim):
            current_X = X[mini_batch_i * mini_batch_size: (mini_batch_i + 1) * mini_batch_size]
            current_Y = Y[mini_batch_i * mini_batch_size: (mini_batch_i + 1) * mini_batch_size]
            for x, y in zip(current_X, current_Y):
                hyp = sum([x_ * w for x_, w in zip(x, weights)])
                gr[dim] += 2 * (hyp - y) * x[dim] / mini_batch_size


        weights = [w - lr*gr for w, gr in zip(weights, gr)]
        
        
    if i % 10 == 0:
        error = cost(weights, X, Y)
        print("Cost:", error, "{}%".format(round(100 * i / N)))


# Оцениваем то, что вышло
hyp = [sum([x_ * w for x_, w in zip(x, weights)]) for x in X]
mae = 1.0 / len(X) * sum([abs(h - y) for h, y in zip(hyp, Y)])


print("Истинная и предсказанная цена")
pprint(list(zip(Y, hyp)))

print("В среднем предсказанная цена отличается от истинной на", mae)

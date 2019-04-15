import numpy as np

with open("./data/matrixA.txt", "r") as lines:
    matrixA = lines.read().split(',')
    # print(matrixA)
    matrixA = list(map(int, matrixA))

with open("./data/matrixB.txt", "r") as lines:
    matrixB = lines.readlines()
    matrixB = [line.strip() for line in matrixB]
    matrixB = [list(map(int, line.split(','))) for line in matrixB]
    # print(matrixB)

A = np.array(matrixA)
B = np.array(matrixB)

C = A.dot(B)
C = np.sort(C)

print(C)
np.savetxt("ans_one.txt", C, fmt="%d")
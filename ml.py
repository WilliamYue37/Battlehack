import tensorflow as tf
from tensorflow import keras
import random
import numpy as np

board_size = 16
class PawnAction:
    PASS = 0
    FORWARD = 1
    CAPTURE_LEFT = 2
    CAPTURE_RIGHT = 3

class PawnType:
    EMPTY = 0
    WHITE = 1
    BLACK = 2

def inversePawnType(pawnType):
    return 0 if pawnType == PawnType.EMPTY else (PawnType.BLACK if pawnType == PawnType.WHITE else PawnType.WHITE)

# you are at (2, 2)
# r and c are already transformed from the original location of the pawn
def pawnMove(r, c, mat):
    if mat[3][3] == PawnType.BLACK:
        return PawnAction.CAPTURE_LEFT
    elif mat[3][1] == PawnType.BLACK:
        return PawnAction.CAPTURE_RIGHT
    elif mat[3][2] == PawnType.EMPTY and mat[4][1] != PawnType.BLACK and mat[4][3] != PawnType.BLACK and r + 1 < board_size:
        return PawnAction.FORWARD
    else:
        return PawnAction.PASS


model = keras.Sequential([
    keras.layers.Dense(64, input_dim=74),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(4)
])

model.compile(optimizer='adam', loss='mean_squared_error')

def gen_rand_input():
    r = random.randrange(board_size)
    c = random.randrange(board_size)
    mat = []
    for i in range(5):
        mat.append([])
        for j in range(5):
            if 0 <= r+i-2 < board_size and 0 <= c+j-2 < board_size:
                mat[i].append(random.choice([0, 0, 1, 2]))
            else:
                mat[i].append(0)
    
    return (r, c, mat)

def matToInVec(mat):
    invec = [r / board_size, c / board_size]
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            if i != 2 or j != 2:
                for k in range(3):
                    invec.append(1 if mat[i][j] == k else 0)
    return invec

for batch in range(10000):
    in1 = []
    out1 = []
    for _ in range(20):
        r, c, mat = gen_rand_input()
        
        invec = matToInVec(mat)
        
        action = pawnMove(r, c, mat)
        out_vec = [1 if x == action else 0 for x in range(4)]

        # print(mat)
        # print(invec)
        # print(out_vec)
        in1.append(invec)
        out1.append(out_vec)
    model.train_on_batch(np.array(in1), np.array(out1))

for _ in range(10):
    r, c, mat = gen_rand_input()
    print(r, c)
    for i in range(5):
        for j in range(5):
            print(mat[i][j], end='')
        print()

    outvec = model.predict(np.array([matToInVec(mat)]))
    print(outvec)
    action = pawnMove(r, c, mat)
    print([1 if x == action else 0 for x in range(4)])
    

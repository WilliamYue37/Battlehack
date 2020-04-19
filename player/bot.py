import random
# from enum import Enum

# from mybattle.stubs import *

# This is an example bot written by the developers!
# Use this to help write your own code, or run it against your bot to see how well you can do!

def dlog(str):
    DEBUG = 1
    if DEBUG > 0:
        log(str)

board_size = get_board_size()

team = get_team()
opp_team = Team.WHITE if team == Team.BLACK else team.BLACK
dlog('Team: ' + str(team))

robottype = get_type()
dlog('Type: ' + str(robottype))

def check_space_wrapper(r, c, board_size):
    # check space, except doesn't hit you with game errors
    if r < 0 or c < 0 or c >= board_size or r >= board_size:
        return False
    try:
        return check_space(r, c)
    except:
        return None

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

# mat is white
# return the column to spawn on relative to mat
# -1 if no spawn 
def lordMove(mat):
    ret = -1
    retv = -1
    for c in range(board_size):
        if mat[0][c] != PawnType.EMPTY:
            continue
        if c - 1 >= 0 and mat[1][c - 1] == PawnType.BLACK or c + 1 < board_size and mat[1][c + 1] == PawnType.BLACK:
            continue
        v = board_size
        for r in range(board_size):
            if mat[r][c] == PawnType.WHITE:
                v = r
                break
            elif mat[r][c] == PawnType.BLACK:
                v = 100 + (board_size - r)
                break
        if v > retv:
            ret = c
            retv = v
    return ret
            

def blackToWhiteMat(mat):
    return [[inversePawnType(x) for x in mat[len(mat) - 1 - r]] for r in range(len(mat))]

def blackToWhiteR(r):
    return board_size - 1 - r

def blackToWhiteC(c):
    return c

# r and c are raw 
def sensedToMat(r, c, sensed):
    ret = [[PawnType.EMPTY] * 5 for r in range(5)]
    for r2, c2, team2 in sensed:
        ret[r2 - r + 2][c2 - c + 2] = PawnType.WHITE if team2 == Team.WHITE else PawnType.BLACK
    return ret

def boardToMat(board):
    return [[PawnType.EMPTY if x is None else (PawnType.WHITE if x == Team.WHITE else PawnType.BLACK) for x in board[r]] for r in range(board_size)]

def turn():
    """
    MUST be defined for robot to run
    This function will be called at the beginning of every turn and should contain the bulk of your robot commands
    """
    dlog('Starting Turn!')
    assert board_size == get_board_size(), "board size assertion"
    assert team == get_team(), "team assertion"
    assert robottype==get_type(), "type assertion"

    if robottype == RobotType.PAWN:
        row, col = get_location()
        dlog('My location is: ' + str(row) + ' ' + str(col))

        sensed = sense()
        mat = sensedToMat(row, col, sensed)
        r = row
        c = col

        if team == Team.BLACK:
            mat = blackToWhiteMat(mat)
            r = blackToWhiteR(r)
            c = blackToWhiteC(c)
        
        move = pawnMove(r, c, mat)
        if move == PawnAction.FORWARD:
            move_forward()
        elif move == PawnAction.CAPTURE_LEFT:
            if team == Team.WHITE:
                capture(row + 1, col + 1)
            else:
                capture(row - 1, col + 1)
        elif move == PawnAction.CAPTURE_RIGHT:
            if team == Team.WHITE:
                capture(row + 1, col - 1)
            else:
                capture(row - 1, col - 1)

    else:
        mat = boardToMat(get_board())
        if team == Team.BLACK:
            mat = blackToWhiteMat(mat)
        move = lordMove(mat)
        if move >= 0:
            spawn(0 if team == Team.WHITE else board_size - 1, move)


    bytecode = get_bytecode()
    dlog('Done! Bytecode left: ' + str(bytecode))


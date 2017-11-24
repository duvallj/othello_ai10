import Othello_Core as oc
import numpy as np

stdict = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12)

def stage(p):
    return stdict[p]

BLACK = 1
WHITE= -1
EMPTY = 0

legal = {11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48, 51, 52, 53, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 66, 67, 68, 71, 72, 73, 74, 75, 76, 77, 78, 81, 82, 83, 84, 85, 86, 87, 88}
def ib(spot):
    return spot in legal

def board2ndarray(board):
    s=oc.BLACK+oc.BLACK+oc.EMPTY+oc.OUTER
    return np.array([s.count(board[i])-1 for i in range(100)])

def ndarray2board(ndarray):
    r=(oc.WHITE,oc.EMPTY,oc.BLACK)
    return [r[ndarray[i]+1] if ib(i) else oc.OUTER for i in range(100)]

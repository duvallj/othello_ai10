import numpy as np
from math import inf
#import ai5
import Othello_Core as oc
import my_core as mc
import pickle
import sys
sys.stdout.write('ai8 imported\n')

stdict = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12)

def stage(p):
    return stdict[p]

BLACK = 1
WHITE= -1
EMPTY = 0

d3a=(13, 22, 31)
d3b=(16, 27, 38)
d3c=(83, 72, 61)
d3d=(86, 77, 68)

d4a=(14, 23, 32, 41)
d4b=(15, 26, 37, 48)
d4c=(84, 73, 62, 51)
d4d=(85, 76, 67, 58)

d5a=(15, 24, 33, 42, 51)
d5b=(14, 25, 36, 47, 58)
d5c=(85, 74, 63, 52, 41)
d5d=(84, 75, 66, 57, 48)

d6a=(16, 25, 34, 43, 52, 61)
d6b=(13, 24, 35, 46, 57, 68)
d6c=(86, 75, 64, 53, 42, 31)
d6d=(83, 74, 65, 56, 47, 38)

d7a=(17, 26, 35, 44, 53, 62, 71)
d7b=(12, 23, 34, 45, 56, 67, 78)
d7c=(87, 76, 65, 54, 43, 32, 21)
d7d=(82, 73, 64, 55, 46, 37, 28)

d8a=(18, 27, 36, 45, 54, 63, 72, 81)
d8b=(11, 22, 33, 44, 55, 66, 77, 88)

h1a=(11, 21, 31, 41, 51, 61, 71, 81)
h1b=(18, 28, 38, 48, 58, 68, 78, 88)
h1c=(11, 12, 13, 14, 15, 16, 17, 18)
h1d=(81, 82, 83, 84, 85, 86, 87, 88)

h2a=(12, 22, 32, 42, 52, 62, 72, 82)
h2b=(17, 27, 37, 47, 57, 67, 77, 87)
h2c=(28, 27, 26, 25, 24, 23, 22, 21)
h2d=(78, 77, 76, 75, 74, 73, 72, 71)

h3a=(13, 23, 33, 43, 53, 63, 73, 83)
h3b=(16, 26, 36, 46, 56, 66, 76, 86)
h3c=(38, 37, 36, 35, 34, 33, 32, 31)
h3d=(68, 67, 66, 65, 64, 63, 62, 61)

h4a=(14, 24, 34, 44, 54, 64, 74, 84)
h4b=(15, 25, 35, 45, 55, 65, 75, 85)
h4c=(48, 47, 46, 45, 44, 43, 42, 41)
h4d=(58, 57, 56, 55, 54, 53, 52, 51)

eda=(11, 21, 31, 41, 51, 61, 71, 81, 22, 72)
edb=(18, 28, 38, 48, 58, 68, 78, 88, 27, 77)
edc=(11, 12, 13, 14, 15, 16, 17, 18, 22, 27)
edd=(81, 82, 83, 84, 85, 86, 87, 88, 72, 77)

tfa=(11, 12, 13, 14, 15, 21, 22, 23, 24, 25)
tfb=(11, 21, 31, 41, 51, 12, 22, 32, 42, 52)
tfc=(18, 17, 16, 15, 14, 28, 27, 26, 25, 24)
tfd=(81, 71, 61, 51, 41, 82, 72, 62, 52, 42)
tfe=(81, 82, 83, 84, 85, 71, 72, 73, 74, 75)
tff=(18, 28, 38, 48, 58, 17, 27, 37, 47, 57)
tfg=(88, 87, 86, 85, 84, 78, 77, 76, 75, 74)
tfh=(88, 78, 68, 58, 48, 87, 77, 67, 57, 47)

tta=(11, 12, 13, 21, 22, 23, 31, 32, 33)
ttb=(18, 17, 16, 28, 27, 26, 38, 37, 36)
ttc=(81, 82, 83, 71, 72, 73, 61, 62, 63)
ttd=(88, 87, 86, 78, 77, 76, 68, 67, 66)

# Wow that was a lot of patterns

all_patterns = (d3a,d3b,d3c,d3d,
                h1a,h1b,h1c,h1d,
                # SEP 1
                d4a,d4b,d4c,d4d,
                d5a,d5b,d5c,d5d,
                d6a,d6b,d6c,d6d,
                d7a,d7b,d7c,d7d,
                d8a,d8b,
                h2a,h2b,h2c,h2d,
                h3a,h3b,h3c,h3d,
                h4a,h4b,h4c,h4d,
                # SEP 2
                eda,edb,edc,edd,
                tta,ttb,ttc,ttd,
                tfa,tfb,tfc,tfd,tfe,tff,tfg,tfh,
)

pat2wght = (-1,-1,-1,-1,
            -1,-1,-1,-1,
            # SEP 1
            0, 0, 0, 0,
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,
            4, 4,
            5, 5, 5, 5,
            6, 6, 6, 6,
            7, 7, 7, 7,
            # SEP 2
            8, 8, 8, 8,
            9, 9, 9, 9,
            10,10,10,10,10,10,10,10,
)

#all_pattern_strs = ('tta','ttb','ttc','ttd',
#                    'd4a','d4b','d4c','d4d',
#                    'd5a','d5b','d5c','d5d',
#                    'd6a','d6b','d6c','d6d',
#                    'd7a','d7b','d7c','d7d',
#                    'd8a','d8b',
#                    'h2a','h2b','h2c','h2d',
#                    'h3a','h3b','h3c','h3d',
#                    'h4a','h4b','h4c','h4d',
#                    'eda','edb','edc','edd',
#                    #'tfa','tfb','tfc','tfd','tfe','tff','tfg','tfh',
#)

legal = {11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48, 51, 52, 53, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 66, 67, 68, 71, 72, 73, 74, 75, 76, 77, 78, 81, 82, 83, 84, 85, 86, 87, 88}
def ib(spot):
    return spot in legal

r_flip = range(0,38)
r_wght = range(8,54)
r_allp = range(0,54)

def board2ndarray(board):
    s=oc.BLACK+oc.BLACK+oc.EMPTY+oc.OUTER
    return np.array([s.count(board[i])-1 for i in range(100)])

def ndarray2board(ndarray):
    r=(oc.WHITE,oc.EMPTY,oc.BLACK)
    return [r[ndarray[i]+1] for i in range(100)]

# Made programatically, hardcoded for SPEED
# index: location, key: pattern index
spot2patterns = {11: (4, 6, 25), 12: (6, 21, 26), 13: (0, 6, 17, 30), 14: (6, 8, 13, 34), 15: (6, 9, 12, 35), 16: (1, 6, 16, 31), 17: (6, 20, 27), 18: (5, 6, 24), 21: (4, 22, 28), 22: (0, 25, 26, 28), 23: (8, 21, 28, 30), 24: (12, 17, 28, 34), 25: (13, 16, 28, 35), 26: (9, 20, 28, 31), 27: (1, 24, 27, 28), 28: (5, 23, 28), 31: (0, 4, 18, 32), 32: (8, 22, 26, 32), 33: (12, 25, 30, 32), 34: (16, 21, 32, 34), 35: (17, 20, 32, 35), 36: (13, 24, 31, 32), 37: (9, 23, 27, 32), 38: (1, 5, 19, 32), 41: (4, 8, 14, 36), 42: (12, 18, 26, 36), 43: (16, 22, 30, 36), 44: (20, 25, 34, 36), 45: (21, 24, 35, 36), 46: (17, 23, 31, 36), 47: (13, 19, 27, 36), 48: (5, 9, 15, 36), 51: (4, 10, 12, 37), 52: (14, 16, 26, 37), 53: (18, 20, 30, 37), 54: (22, 24, 34, 37), 55: (23, 25, 35, 37), 56: (19, 21, 31, 37), 57: (15, 17, 27, 37), 58: (5, 11, 13, 37), 61: (2, 4, 16, 33), 62: (10, 20, 26, 33), 63: (14, 24, 30, 33), 64: (18, 23, 33, 34), 65: (19, 22, 33, 35), 66: (15, 25, 31, 33), 67: (11, 21, 27, 33), 68: (3, 5, 17, 33), 71: (4, 20, 29), 72: (2, 24, 26, 29), 73: (10, 23, 29, 30), 74: (14, 19, 29, 34), 75: (15, 18, 29, 35), 76: (11, 22, 29, 31), 77: (3, 25, 27, 29), 78: (5, 21, 29), 81: (4, 7, 24), 82: (7, 23, 26), 83: (2, 7, 19, 30), 84: (7, 10, 15, 34), 85: (7, 11, 14, 35), 86: (3, 7, 18, 31), 87: (7, 22, 27), 88: (5, 7, 25)}
'''
spot2patterns = dict()
for spot in legal:
    l = []
    for i in r_flip:
        if spot in all_patterns[i]:
            l.append(i)
    spot2patterns[spot] = tuple(l)
'''
def things2sum(pieces):
    s = 0
    for p in pieces:
        s *= 3
        s += p
    return s

ref = (0,1,-1)
def sum2things(s,slen):
    l = [0]*slen
    for i in range(1,slen+1):
        x = ref[s%3]
        l[slen-i] = x
        s -= x
        s //= 3
    return tuple(l)

import pickle

# Make the lookup tables of move in pattern to next pattern
# Format:
#  * index: pattern index, key: dict
#    * index: pattern, key: dict
#      * index: black or white, key: dict
#        * index: spot, key: next set of indicies
file = open('fliplookup.pkl','rb')
fliplookup = pickle.load(file) #Friggin huge
file.close()

# Goodness value table
# Dict Format:
#  * index: pattern index, value: dict
#    * index: pattern, value: tuple
#      * index: stage, value: goodness (for BLACK)
file = open('gvals2.pkl','rb')
gvals = pickle.load(file) # Friggin huge-er
file.close()

sys.stdout.write('lookups read\n')

#fliplookup = dict()

def fidig(board, color, start, direction):
    flipped = []
    cur = start + direction
    opponent = -color
    board[start] = color
    inbetween = False

    while ib(cur) and board[cur]==opponent:
        flipped.append(cur)
        board[cur] *= -1
        cur += direction
        inbetween = True

    if board[cur] != color or not inbetween:
        board[start] = 0
        for s in flipped:
            board[s] *= -1
        flipped = []
        
    return flipped
'''
from itertools import product
for i in r_flip:
    pat = all_patterns[i]
    eboard = np.zeros((100,))#board2ndarray(oc.OthelloCore().initial_board())
    thislookup = dict()
    for plist in product(ref, repeat=len(pat)):
        d = things2sum(plist)
        thislookup[d] = dict()
        thislookup[d][1] = dict()
        thislookup[d][-1] = dict()
        for spot,piece in zip(pat,plist):
            eboard[spot] = piece
        for spot in pat:
            if eboard[spot]==0:
                for c in (-1,1):
                    fl = []
                    for di in (-11,-10,-9,-1,1,9,10,11):
                        fl.extend(fidig(eboard, c, spot, di))
                    if fl:
                        eboard[spot] = c
                        thislookup[d][c][spot] = tuple(eboard[s] for s in pat)
                        eboard[spot] = 0
                        for s in fl:
                            eboard[s] *= -1
    fliplookup[i] = thislookup
    print(i)
'''

class Node(object):
    def __init__(self, board, tomove, prevmove, pcount):
        self.board = board
        self.pcount=pcount
        self.stage = stage(pcount)
        #self.plists = 
        self.tomove = tomove
        self.moves = []
        self.gmrun = True
        self.srun = True
        self.gcrun = True
        self.prevmove = prevmove

    def get_moves(self):
        if self.gmrun:
            self.plists = tuple(tuple(self.board[x] for x in all_patterns[i]) for i in r_allp)
            self.blackmoves = set()
            self.whitemoves = set()
            
            for i in r_flip:
                self.blackmoves.update(fliplookup[i][self.plists[i]][BLACK].keys())
                self.whitemoves.update(fliplookup[i][self.plists[i]][WHITE].keys())

            if self.tomove == BLACK:
                if len(self.blackmoves):
                    self.moves = self.blackmoves
                else:
                    self.tomove = WHITE
                    self.moves = self.whitemoves
            elif self.tomove == WHITE:
                if len(self.whitemoves):
                    self.moves = self.whitemoves
                else:
                    self.tomove = BLACK
                    self.moves = self.blackmoves
                    
            self.gmrun = False
                
        return self.moves
        # Other methods responsible for catching no moves

    def score_old(self):
        '''
            Returns: ranking for BLACK based on mobility and
            sum of wieghted piece places
        '''
        if self.srun:
            if self.gmrun:
                self.get_moves()
            s = 0
            for i in r_wght:
                for x in range(len(self.plists[i])):
                    s += self.plists[i][x] * gvals[i][x]

            if s * self.tomove > 0:
                self.s=s * (-1 / (len(self.moves)+1) + 2)
            else:
                self.s=s * (1 / (len(self.moves)+1)  + 1)
            self.srun = False
        return self.s

    def score(self):
        if self.srun:
            if self.gmrun:
                self.get_moves()
            s = gvals[11].get(self.stage, 0)*(len(self.blackmoves)-len(self.whitemoves)) + gvals[12].get(self.stage, 0)*(self.pcount&1==0) + gvals[13].get(self.stage, 0)
            for i in r_wght:
                s += gvals[pat2wght[i]].get(self.plists[i],dict()).get(stage,0)
            self.s = s # is this right?
            self.srun = False
        return self.s
            
    def gen_children(self):
        '''
            Returns: All the possible next moves from this node
            based on the 'tomove' variable
        '''
        # Assumed get_moves has been called
        if self.gcrun:
            if self.gmrun:
                self.get_moves()
            children = []
            for spot in self.moves:
                next_board = np.copy(self.board)
                for pi in spot2patterns[spot]:
                    new_plist = fliplookup[pi][self.plists[pi]][self.tomove]
                    if spot in new_plist:
                        for x in range(len(new_plist[spot])):
                            next_board[all_patterns[pi][x]] = new_plist[spot][x]
                children.append(Node(next_board, self.tomove*-1, spot, self.pcount+1))
            self.children = children
        return self.children


def abprune(node, depth, maxdepth, alpha, beta):
    if depth > maxdepth:
        return node.score(), node.prevmove

    if node.tomove == BLACK:
        v = -inf
        best = None
        moves = node.get_moves()
        if len(moves)==0:
            return node.score(), node.prevmove
        for child in node.gen_children():
            newv, a = abprune(child, depth+1, maxdepth, alpha, beta)
            #print('    '*depth, v, newv, child.prevmove)
            if newv > v:
                v = newv
                best = child
            if v >= beta:
                return v, best.prevmove
            if v > alpha:
                alpha = v
        return v, best.prevmove
    
    else:
        v = inf
        best = None
        moves = node.get_moves()
        if len(moves)==0:
            return node.score(), -1
        for child in node.gen_children():
            newv, a = abprune(child, depth+1, maxdepth, alpha, beta)
            #print('    '*depth, v, newv, child.prevmove)
            if newv < v:
                v = newv
                best = child
            if v <= alpha:
                return v, best.prevmove
            if v < beta:
                beta = v
        return v, best.prevmove
        
class Value:
    def __init__(self):
        self.value = 0

class Strategy(mc.MyCore):
    def __init__(self):
        self.players = [oc.BLACK, oc.WHITE]
        sys.stdout.write('object created\n')
        super(Strategy, self).__init__()
        
    def best_strategy(self, board, player, move, flag):
        if player==oc.BLACK:
            tomove = BLACK
        else:
            tomove = WHITE
        rboard = board2ndarray(board)
        root = Node(rboard, tomove, -1,0)

        for d in (2, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30):
            move.value = abprune(root, 0, d, -inf, inf)[1]
            print(d, move.value)
        return move.value
if __name__=="__main__":
    while True:
        rboard = board2ndarray(input("Enter board: "))
        root = Node(rboard, 1, -1, sum([bool(x) for x in rboard]))
        print(root.score())
        move = abprune(root, 0, 4, -inf, inf)
        print(move)

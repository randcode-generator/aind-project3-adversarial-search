from collections import defaultdict, Counter
import random
from sample_players import DataPlayer
import pickle

class CustomPlayer(DataPlayer):
    def writeFile(self, filename, history, data=None):
        with open(filename, "wb") as f:
            t = {}
            for key in history:
                g = history[key]
                t[g["board"]] = g["move"]
            if(data != None):
                t.update(data)
            pickle.dump(t, f)

    def __del__(self):
        if(self.newRecords == False):
            return
        self.filename = "data.pickle"

        history = self.context["history"]
        if(history == None):
            return
        from os import path
        exists = path.exists("data.pickle")
        if(exists):
            with open("data.pickle", "rb") as f:
                data = pickle.load(f)
                self.writeFile(self.filename, history, data)
        else:
            self.writeFile(self.filename, history)

    def __init__(self, player_id):
        super().__init__(player_id)
        self.newRecords = False

        self.q1 = []
        start = 52
        end = 58
        for _ in range(5):
            self.q1 += list(range(start, end))
            start += 13
            end += 13
        self.q1 = set(self.q1)

        self.q2 = []
        start = 0
        end = 6
        for _ in range(5):
            self.q2 += list(range(start, end))
            start += 13
            end += 13
        self.q2 = set(self.q2)

        self.q3 = []
        start = 5
        end = 11
        for _ in range(5):
            self.q3 += list(range(start, end))
            start += 13
            end += 13
        self.q3 = set(self.q3)

        self.q4 = []
        start = 57
        end = 63
        for _ in range(5):
            self.q4 += list(range(start, end))
            start += 13
            end += 13
        self.q4 = set(self.q4)

        self.context = {"book": self.data, "history": {}}
    
    def insertHistory(self, state, bestMove):
        if bestMove == None:
            self.context["history"] = None
        else:
            history = self.context["history"]
            if(history != None and len(history) < 3):
                self.context["history"][state.ply_count] = {"board": state.board, "move": bestMove}

    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        book = self.context["book"]
        if(state.ply_count <= 4):
            if(state.board in book):
                act = book[state.board]
                self.queue.put(act)
                self.insertHistory(state, act)
            else:
                self.newRecords = True

        if state.ply_count < 2:
            acts = state.actions()
            # Choose the middle action. If this is the first move, choose the center
            if 57 in acts:
                index = acts.index(57)
                act = acts[index]
                self.queue.put(act)
                self.insertHistory(state, act)
            else:
                index = int(len(acts)/2)-1
                act = state.actions()[index]
                self.queue.put(act)
                self.insertHistory(state, act)
        else:
            for i in range(1, 5):
                bestMove = self.alpha_beta_search(state, depth=i)
                self.insertHistory(state, bestMove)
                if bestMove == None:
                    bestMove = state.actions()[0]
                self.queue.put(bestMove)

    def alpha_beta_search(self, state, depth):
        """ Return the move along a branch of the game tree that
        has the best possible value.  A move is a pair of coordinates
        in (column, row) order corresponding to a legal move for
        the searching player.
        
        You can ignore the special case of calling this function
        from a terminal state.
        """
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        for a in state.actions():
            v = self.min_value(state.result(a), alpha, beta, depth)
            alpha = max(alpha, v)
            if v > best_score:
                best_score = v
                best_move = a
        return best_move
    
    def min_value(self, state, alpha, beta, depth):
        if state.terminal_test():
            return state.utility(self.player_id)
        if depth <= 0:
            return self.score(state, depth)
        v = float("inf")
        for a in state.actions():
            v = min(v, self.max_value(state.result(a), alpha, beta, depth-1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def max_value(self, state, alpha, beta, depth):
        if state.terminal_test(): return state.utility(self.player_id)
        if depth <= 0: return self.score(state, depth)
        v = float("-inf")
        for a in state.actions():
            v = max(v, self.min_value(state.result(a), alpha, beta, depth-1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def numOfFreeInQuadrant(self, state, quad):
        num = 0
        for loc in quad:
            if state.board & (1 << loc):
                num += 1
        return num

    def score(self, state, depth):
        own_loc = state.locs[self.player_id]
        own_liberties = state.liberties(own_loc)
        f = 0
        if own_loc in self.q1:
            f = self.numOfFreeInQuadrant(state, self.q1)  
        elif own_loc in self.q2:
            f = self.numOfFreeInQuadrant(state, self.q2)
        elif own_loc in self.q3:
            f = self.numOfFreeInQuadrant(state, self.q3)
        elif own_loc in self.q4:
            f = self.numOfFreeInQuadrant(state, self.q4)
        else:
            raise Exception("not found in any quad")

        return len(own_liberties) + f

from collections import defaultdict, Counter
import random
from sample_players import DataPlayer
import math

class CustomPlayer(DataPlayer):
    def __init__(self, player_id):
        super().__init__(player_id)
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
        if state.ply_count < 2:
            acts = state.actions()
            # Choose the middle action. If this is the first move, choose the center
            if 57 in acts:
                index = acts.index(57)
                self.queue.put(acts[index])
            else:
                index = int(len(acts)/2)-1
                act = state.actions()[index]
                self.queue.put(act)
        else:
            best = self.alpha_beta_search(state, depth=4)
            if best == None:
                best = state.actions()[0]
            self.queue.put(best)

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
            return self.score1(state, depth)
        v = float("inf")
        for a in state.actions():
            v = min(v, self.max_value(state.result(a), alpha, beta, depth-1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def max_value(self, state, alpha, beta, depth):
        if state.terminal_test(): return state.utility(self.player_id)
        if depth <= 0: raise Exception("shouldn't be here")
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

    def score1(self, state, depth):
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

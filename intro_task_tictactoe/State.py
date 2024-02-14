import numpy as np


class State:

    def __init__(self):
        self.board = np.zeros((3, 3))
        self.winner = None
        self.hash_val = None
        self.end = None

    def hash(self):
        if self.hash_val is None:
            self.hash_val = 0
            for i in np.nditer(self.board):
                self.hash_val = self.hash_val * 3 + i + 1
        return self.hash_val

    def is_game_over(self):
        if self.end is not None:
            return self.end

        results = []

        # check row
        for i in range(3):
            results.append(np.sum(self.board[i, :]))
        # check columns
        for i in range(3):
            results.append(np.sum(self.board[:, i]))

        # check diagonals
        trace = 0
        reverse_trace = 0
        for i in range(3):
            trace += self.board[i, i]
            reverse_trace += self.board[i, 3 - 1 - i]
        results.append(trace)
        results.append(reverse_trace)

        for result in results:
            if result == 3:
                self.winner = 1
                self.end = True
                return self.end
            if result == -3:
                self.winner = -1
                self.end = True
                return self.end

        # whether it's a tie
        sum_values = np.sum(np.abs(self.board))
        if sum_values == 9:
            self.winner = 0
            self.end = True
            return self.end

        # game is still going on
        self.end = False
        return self.end

    def next_state(self, i, j, symbol):
        new_state = State()
        new_state.board = np.copy(self.board)
        new_state.board[i, j] = symbol
        return new_state

    def render(self):
        # print the board
        for i in range(3):
            print('-------------')
            out = '| '
            for j in range(3):
                if self.board[i, j] == 1:
                    token = '*'
                elif self.board[i, j] == -1:
                    token = 'x'
                else:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-------------')


class StateUtils:

    @staticmethod
    def get_all_states_impl(current_state, current_symbol, all_states):
        for i in range(3):
            for j in range(3):
                if current_state.board[i][j] == 0:
                    new_state = current_state.next_state(i, j, current_symbol)
                    new_hash = new_state.hash()
                    if new_hash not in all_states:
                        is_end = new_state.is_game_over()
                        all_states[new_hash] = (new_state, is_end)
                        if not is_end:
                            StateUtils.get_all_states_impl(new_state, -current_symbol, all_states)

    @staticmethod
    def get_all_states():
        current_symbol = 1
        current_state = State()
        all_states = dict()
        all_states[current_state.hash()] = (current_state, current_state.is_game_over())
        StateUtils.get_all_states_impl(current_state, current_symbol, all_states)
        return all_states


all_states = StateUtils.get_all_states()

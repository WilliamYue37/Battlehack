import random
from .robot import Robot
from .team import Team
from .robottype import RobotType
from .constants import GameConstants

import tensorflow as tf
from tensorflow import keras
import numpy as np

batch_size = 20

class PawnAction:
    PASS = 0
    FORWARD = 1
    CAPTURE_LEFT = 2
    CAPTURE_RIGHT = 3

class PawnType:
    EMPTY = 0
    WHITE = 1
    BLACK = 2

class Game:

    def __init__(self, code, board_size=GameConstants.BOARD_SIZE, max_rounds=GameConstants.MAX_ROUNDS, 
                 seed=GameConstants.DEFAULT_SEED, sensor_radius=2, debug=False, colored_logs=True, pawn_model=None, lord_model=None,
                 do_train=[]):
        random.seed(seed)

        self.code = code

        self.debug = debug
        self.colored_logs = colored_logs
        self.running = True
        self.winner = None

        self.robot_count = 0
        self.queue = {}
        self.leaders = []

        self.sensor_radius = sensor_radius
        self.board_size = board_size
        self.board = [[None] * self.board_size for _ in range(self.board_size)]
        self.round = 0
        # self.max_rounds = max_rounds
        self.max_rounds = 50

        self.lords = []
        self.new_robot(None, None, Team.WHITE, RobotType.OVERLORD)
        self.new_robot(None, None, Team.BLACK, RobotType.OVERLORD)

        self.board_states = []
        
        self.special_turn = 30
        self.pawn_turns = random.randint(0, self.special_turn - 1)
        self.lord_turns = random.randint(0, self.special_turn - 1)

        self.do_train = do_train

        self.pawn_ins, self.pawn_outs, self.lord_ins, self.lord_outs = [], [], [], []
        self.pawn_teams, self.lord_teams = [], []

        self.pawn_model = pawn_model
        self.lord_model = lord_model

        self.white_heu = 0
        self.black_heu = 0
        self.good_heu = 500

    def delete_robot(self, i):
        robot = self.queue[i]
        self.board[robot.row][robot.col] = None
        robot.kill()
        del self.queue[i]

    def serialize(self):
        def serialize_robot(robot):
            if robot is None:
                return None

            return {'id': robot.id, 'team': robot.team, 'health': robot.health, 'logs': robot.logs[:]}

        return [[serialize_robot(c) for c in r] for r in self.board]

    def log_info(self, msg):
        if self.colored_logs:
            print(f'\u001b[32m[Game info] {msg}\u001b[0m')
        else:
            print(f'[Game info] {msg}')

    def check_over(self):
        white, black = 0, 0
        for col in range(self.board_size):
            if self.board[0][col] and self.board[0][col].team == Team.BLACK: black += 1
            if self.board[self.board_size - 1][col] and self.board[self.board_size - 1][col].team == Team.WHITE: white += 1

        if self.round > self.max_rounds:
            self.running = False
            if white == black:
                self.winner = None
            else:
                self.winner = Team.WHITE if white > black else Team.BLACK

        if white >= (self.board_size + 1) // 2:
            self.running = False
            self.winner = Team.WHITE

        if black >= (self.board_size + 1) // 2:
            self.running = False
            self.winner = Team.BLACK

        if not self.running:
            white_cols, black_cols = set(), set()
            for r in range(self.board_size):
                for c in range(self.board_size):
                    if self.board[r][c] is not None:
                        if self.board[r][c].team == Team.WHITE:
                            self.white_heu += r + 1
                            white_cols.add(c)
                        else:
                            self.black_heu += self.board_size - r
                            black_cols.add(c)

            self.white_heu += len(white_cols) * 8
            self.black_heu += len(black_cols) * 8

            if self.white_heu > self.black_heu:
                self.winner = Team.WHITE
            elif self.white_heu < self.black_heu:
                self.winner = Team.BLACK
            else:
                self.winner = None

            self.board_states.append([row[:] for row in self.board])
            self.process_over()

    def process_over(self):
        """
        Helper method to process once a game is finished (e.g. deleting robots)
        """
        for i in range(self.robot_count):
            if i in self.queue:
                self.delete_robot(i)


    def new_robot(self, row, col, team, robot_type):
        if robot_type == RobotType.OVERLORD:
            id = f'{team.name} HQ'
        else:
            id = self.robot_count
        robot = Robot(row, col, team, id, robot_type)

        shared_methods = {
            'GameError': GameError,
            'RobotType': RobotType,
            'RobotError': RobotError,
            'Team': Team,
            'get_board_size': lambda : self.get_board_size(),
            'get_bytecode' : lambda : robot.runner.bytecode,
            'get_team': lambda : self.get_team(robot),
            'get_type': lambda: self.get_type(robot),
        }

        if robot_type == RobotType.OVERLORD:
            methods = {
                'check_space': lambda row, col: self.hq_check_space(row, col),
                'get_board': lambda : self.get_board(),
                'spawn': lambda row, col: self.spawn(robot, row, col)
            }
        else:
            methods = {
                'capture': lambda row, col: self.capture(robot, row, col),
                'check_space': lambda row, col: self.pawn_check_space(robot, row, col),
                'get_location': lambda : self.get_location(robot),
                'move_forward': lambda: self.move_forward(robot),
                'sense': lambda : self.sense(robot)
            }

        methods.update(shared_methods)

        robot.animate(self.code[team.value], methods, debug=self.debug)

        if robot_type == RobotType.PAWN:
            self.queue[self.robot_count] = robot
            self.board[row][col] = robot
        else:
            self.lords.append(robot)

        self.robot_count += 1

    def is_on_board(self, row, col):
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            return True
        return False


    #### SHARED METHODS ####

    def get_board_size(self):
        """
        @HQ_METHOD, @PAWN_METHOD

        Return the size of the board (int)
        """
        return self.board_size

    def get_team(self, robot):
        """
        @HQ_METHOD, @PAWN_METHOD

        Return the current robot's team (Team.WHITE or Team.BLACK)
        """
        return robot.team

    def get_type(self, robot):
        """
        @HQ_METHOD, @PAWN_METHOD

        Return the type of the unit - either RobotType.PAWN or RobotType.OVERLORD
        """
        return robot.type


    #### HQ METHODS ####

    def get_board(self):
        """
        @HQ_METHOD

        Return the current state of the board as an array of Team.WHITE, Team.BLACK, and None, representing white-occupied,
        black-occupied, and empty squares, respectively.
        """
        board = [[None] * self.board_size for _ in range(self.board_size)]

        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j]:
                    board[i][j] = self.board[i][j].team

        return board

    def hq_check_space(self, row, col):
        """
        @HQ_METHOD

        Checks whether a specific board space is occupied and if yes returns the team of the robot occupying the space;
        otherwise returns False. Pawns have a similar method but can only see within their sensory radius
        """
        if not self.board[row][col]:
            return False
        return self.board[row][col].team

    def spawn(self, robot, row, col):
        """
        @HQ_METHOD

        Spawns a pawn at the given location. Pawns can only be spawned at the edge of the board on your side of the board.
        Only the HQ can spawn units, and it can only spawn one unit per turn.
        :loc should be given as a tuple (row, col), the space to spawn on
        """
        if robot.has_moved:
            raise RobotError('you have already spawned a unit this turn')

        if (robot.team == Team.WHITE and row != 0) or (robot.team == Team.BLACK and row != self.board_size - 1):
            raise RobotError('you can only spawn in the end row of your side of the board')

        if not self.is_on_board(row, col):
            raise RobotError('you cannot spawn a unit on a space that is not on the board')

        if self.board[row][col]:
            raise RobotError('you cannot spawn a unit on a space that is already occupied')

        self.new_robot(row, col, robot.team, RobotType.PAWN)
        robot.has_moved = True


    #### PAWN METHODS ####

    def capture(self, robot, new_row, new_col):
        """
        @PAWN_METHOD

        Diagonally capture an enemy piece.
        :new_row, new_col the position of the enemy to capture.
        Units can only capture enemy pieces that are located diagonally left or right in front of them on the board.
        """
        if robot.has_moved:
            raise RobotError('this unit has already moved this turn; robots can only move once per turn')

        row, col = robot.row, robot.col

        if self.board[row][col] != robot:
            raise RobotError

        if not self.is_on_board(new_row, new_col):
            raise RobotError('you cannot capture a space that is not on the board')

        if not self.board[new_row][new_col]:
            raise RobotError('you cannot capture an empty space')

        if self.board[new_row][new_col].team == robot.team:
            raise RobotError('you cannot capture your own piece')

        if abs(col - new_col) != 1:
            raise RobotError('you must capture diagonally')

        if (robot.team == Team.WHITE and row - new_row != -1) or (robot.team == Team.BLACK and row - new_row != 1):
            raise RobotError('you must capture diagonally forwards')

        captured_robot = self.board[new_row][new_col]

        self.delete_robot(captured_robot.id)
        self.board[row][col] = None

        robot.row = new_row
        robot.col = new_col

        self.board[new_row][new_col] = robot
        robot.has_moved = True

    def get_location(self, robot):
        """
        @PAWN_METHOD

        Return the current location of the robot
        """
        row, col = robot.row, robot.col
        if self.board[row][col] != robot:
            raise RobotError('something went wrong; please contact the devs')
        return row, col

    def move_forward(self, robot):
        """
        @PAWN_METHOD

        Move the current unit forward. A unit can only be moved if there is no unit already occupying the space.
        """
        if robot.has_moved:
            raise RobotError('this unit has already moved this turn; robots can only move once per turn')

        row, col = robot.row, robot.col
        if self.board[row][col] != robot:
            raise RobotError('something went wrong; please contact the devs')

        if robot.team == Team.WHITE:
            new_row, new_col = row + 1, col
        else:
            new_row, new_col = row - 1, col

        if not self.is_on_board(new_row, new_col):
            raise RobotError('you cannot move to a space that is not on the board')

        if self.board[new_row][new_col]:
            raise RobotError('you cannot move to a space that is already occupied')

        self.board[row][col] = None

        robot.row = new_row
        robot.col = new_col
        self.board[new_row][new_col] = robot
        robot.has_moved = True

    def pawn_check_space(self, robot, row, col):
        """
        @PAWN_METHOD

        Checks whether a specific board space is occupied and if yes returns the team of the robot occupying the space;
        otherwise returns False.

        Raises a RobotError if the space is not within the sensory radius

        HQs have a similar method but can see the full board
        """
        if self.board[robot.row][robot.col] != robot:
            raise RobotError('something went wrong; please contact the devs')

        drow, dcol = abs(robot.row - row), abs(robot.col - col)
        if max(drow, dcol) > 2:
            raise RobotError('that space is not within sensory radius of this robot')

        if not self.board[row][col]:
            return False
        return self.board[row][col].team

    def sense(self, robot):
        """
        @PAWN_METHOD

        Sense nearby units; returns a list of tuples of the form (row, col, robot.team) within sensor radius of this robot (excluding yourself)
        You can sense another unit other if it is within sensory radius of you; e.g. max(|robot.x - other.x|, |robot.y - other.y|) <= sensory_radius
        """
        row, col = robot.row, robot.col

        robots = []

        for i in range(-self.sensor_radius, self.sensor_radius + 1):
            for j in range(-self.sensor_radius, self.sensor_radius + 1):
                if i == 0 and j == 0:
                    continue

                new_row, new_col = row + i, col + j
                if not self.is_on_board(new_row, new_col):
                    continue

                if self.board[new_row][new_col]:
                    robots.append((new_row, new_col, self.board[new_row][new_col].team))

        return robots


    #### DEBUG METHODS: NOT AVAILABLE DURING CONTEST ####

    def view_board(self, colors=True):
        """
        @DEBUG_METHOD
        THIS METHOD IS NOT AVAILABLE DURING ACTUAL GAMES.

        Helper method that displays the full board as a human-readable string.
        """
        board = ''
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j]:
                    board += '['
                    if colors:
                        if self.board[i][j].team == Team.WHITE:
                            board += '\033[1m\u001b[37m'
                        else:
                            board += '\033[1m\u001b[36m'
                    board += str(self.board[i][j])
                    if colors:
                        board += '\033[0m\u001b[0m] '
                else:
                    board += '[    ] '
            board += '\n'
        return board

    def check_space_wrapper(r, c, board_size):
        # check space, except doesn't hit you with game errors
        if r < 0 or c < 0 or c >= board_size or r >= board_size:
            return False
        try:
            return check_space(r, c)
        except:
            return None


    def inversePawnType(self, pawnType):
        return 0 if pawnType == PawnType.EMPTY else (PawnType.BLACK if pawnType == PawnType.WHITE else PawnType.WHITE)      

    def blackToWhiteMat(self, mat):
        return [[self.inversePawnType(x) for x in mat[len(mat) - 1 - r]] for r in range(len(mat))]

    def blackToWhiteR(self, r):
        return self.board_size - 1 - r

    def blackToWhiteC(self, c):
        return c
    
    def rightToLeftMat(self, mat):
        return [[mat[r][c] for c in range(len(mat[r]) - 1, -1, -1)] for r in range(len(mat))]
    
    def rightToLeftR(self, r):
        return r
    
    def rightToLeftC(self, c):
        return self.board_size - 1 - c

    # r and c are raw 
    def sensedToMat(self, r, c, sensed):
        ret = [[PawnType.EMPTY] * 5 for r in range(5)]
        for r2, c2, team2 in sensed:
            ret[r2 - r + 2][c2 - c + 2] = PawnType.WHITE if team2 == Team.WHITE else PawnType.BLACK
        return ret

    def boardToMat(self, board):
        return [[PawnType.EMPTY if x is None else (PawnType.WHITE if x == Team.WHITE else PawnType.BLACK) for x in board[r]] for r in range(self.board_size)]

    def pawnToInVec(self, r, c, mat, action):
        invec = [r / self.board_size, c / (self.board_size / 2)]
        for k in range(4):
            invec.append(1 if k == action else 0)
        for r in range(len(mat)):
            for c in range(len(mat[r])):
                if r == 2 and c == 2: continue
                for k in range(3):
                    invec.append(1 if mat[r][c] == k else 0)
        return invec
    
    def lordToInVec(self, mat):
        invec = []
        for r in range(len(mat)):
            for c in range(len(mat[r])):
                for k in range(3):
                    invec.append(1 if mat[r][c] == k else 0)
        return invec

    # you are at (2, 2)
    # r, c, and mat are already transformed from the original location of the pawn
    def pawnMove(self, r, c, mat, team):
        if self.pawn_model is None or not team in self.do_train:
            if mat[3][3] == PawnType.BLACK:
                return PawnAction.CAPTURE_LEFT
            elif mat[3][1] == PawnType.BLACK:
                return PawnAction.CAPTURE_RIGHT
            elif mat[3][2] == PawnType.EMPTY and mat[4][1] != PawnType.BLACK and mat[4][3] != PawnType.BLACK and r + 1 < self.board_size:
                return PawnAction.FORWARD
            else:
                return PawnAction.PASS
        else:
            pot_moves = [PawnAction.PASS]
            if mat[3][3] == PawnType.BLACK:
                pot_moves.append(PawnAction.CAPTURE_LEFT)
            elif mat[3][1] == PawnType.BLACK:
                pot_moves.append(PawnAction.CAPTURE_RIGHT)
            elif mat[3][2] == PawnType.EMPTY and r + 1 < self.board_size:
                pot_moves.append(PawnAction.FORWARD)
            
            if len(pot_moves) == 1:
                return pot_moves[0]
            
            best_move, best_value = None, -float("inf")
            for move in pot_moves:
                value = self.pawn_model.predict(np.array([self.pawnToInVec(r, c, mat, move)]))[0][0]
                if value > best_value:
                    best_move = move
                    best_value = value
                        
            return best_move
                

    # mat is white
    # return the column to spawn on relative to mat
    # -1 if no spawn
    def lordMove(self, mat, team):
        if self.lord_model is None or not team in self.do_train:
            ret = -1
            retv = -1
            for c in range(self.board_size):
                if mat[0][c] != PawnType.EMPTY:
                    continue
                if c - 1 >= 0 and mat[1][c - 1] == PawnType.BLACK or c + 1 < self.board_size and mat[1][c + 1] == PawnType.BLACK:
                    continue

                v = self.board_size
                for r in range(self.board_size):
                    if mat[r][c] == PawnType.WHITE:
                        v = r
                        break

                    elif mat[r][c] == PawnType.BLACK:
                        v = 100 + (self.board_size - r)
                        break

                if v > retv:
                    ret = c
                    retv = v
            return ret
        else:
            pot_moves = [-1]
            for c in range(self.board_size):
                if mat[0][c] == PawnType.EMPTY:
                    pot_moves.append(c)

            if len(pot_moves) == 1:
                return pot_moves[0]

            best_move, best_value = None, -float("inf")
            for move in pot_moves:
                if move >= 0:
                    mat[0][move] = PawnType.WHITE
                new_value = self.lord_model.predict(np.array([self.lordToInVec(mat)]))[0][0]
                if move >= 0:
                    mat[0][move] = PawnType.EMPTY
                if new_value > best_value:
                    best_move = move
                    best_value = new_value
            return best_move
                

    def bot_turn(self, robot):
        team = self.get_team(robot)
        robottype = self.get_type(robot)

        if robottype == RobotType.PAWN:
            r, c = self.get_location(robot)

            sensed = self.sense(robot)
            mat = self.sensedToMat(r, c, sensed)

            nr, nc = r, c
            if team == Team.BLACK:
                mat = self.blackToWhiteMat(mat)
                nr = self.blackToWhiteR(r)
                nc = self.blackToWhiteC(c)
            
            move = self.pawnMove(nr, nc, mat, team)
            if move == PawnAction.FORWARD:
                self.move_forward(robot)
            elif move == PawnAction.CAPTURE_LEFT:
                if team == Team.WHITE:
                    self.capture(robot, r + 1, c + 1)
                else:
                    self.capture(robot, r - 1, c + 1)
            elif move == PawnAction.CAPTURE_RIGHT:
                if team == Team.WHITE:
                    self.capture(robot, r + 1, c - 1)
                else:
                    self.capture(robot, r - 1, c - 1)
                
            return nr, nc, mat, move

        else:
            mat = self.boardToMat(self.get_board())
            if team == Team.BLACK:
                mat = self.blackToWhiteMat(mat)
            move = self.lordMove(mat, team)
            if move >= 0:
                self.spawn(robot, 0 if team == Team.WHITE else self.board_size - 1, move)
            
            return self.boardToMat(self.get_board())
    
    def pawn_rand(self, robot):
        team = self.get_team(robot)
        r, c = self.get_location(robot)
        sensed = self.sense(robot)
        mat = self.sensedToMat(r, c, sensed)
        nr, nc = r, c
        if team == Team.BLACK:
            mat = self.blackToWhiteMat(mat)
            nr = self.blackToWhiteR(nr)
            nc = self.blackToWhiteC(nc)
        if c >= self.board_size / 2:
            mat = self.rightToLeftMat(mat)
            nr = self.rightToLeftR(nr)
            nc = self.rightToLeftC(nc)

        pot_moves = [PawnAction.PASS]
        if nr < self.board_size - 1 and mat[3][2] == PawnType.EMPTY:
            pot_moves.append(PawnAction.FORWARD)
        if nr < self.board_size - 1 and nc < self.board_size - 1 and mat[3][3] == PawnType.BLACK:
            pot_moves.append(PawnAction.CAPTURE_LEFT)
        if nr < self.board_size - 1 and nc > 0 and mat[3][1] == PawnType.BLACK:
            pot_moves.append(PawnAction.CAPTURE_RIGHT)

        move = random.choice(pot_moves)

        if move == PawnAction.FORWARD:
            self.move_forward(robot)
        elif move == PawnAction.CAPTURE_LEFT:
            if team == Team.WHITE:
                if c < self.board_size / 2:
                    self.capture(robot, r + 1, c + 1)
                else:
                    self.capture(robot, r + 1, c - 1)
            else:
                if c < self.board_size / 2:
                    self.capture(robot, r - 1, c + 1)
                else:
                    self.capture(robot, r - 1, c - 1)
        elif move == PawnAction.CAPTURE_RIGHT:
            if team == Team.WHITE:
                if c < self.board_size / 2:
                    self.capture(robot, r + 1, c - 1)
                else:
                    self.capture(robot, r + 1, c + 1)
            else:
                if c < self.board_size / 2:
                    self.capture(robot, r - 1, c - 1)
                else:
                    self.capture(robot, r - 1, c + 1)
        
        return nr, nc, mat, move
    
    def lord_rand(self, robot):
        team = self.get_team(robot)
        mat = self.boardToMat(self.get_board())
        if team == Team.BLACK:
            mat = self.blackToWhiteMat(mat)

        pot_spawn = [-1]
        for c in range(self.board_size):
            if mat[0][c] == PawnType.EMPTY:
                pot_spawn.append(c)
        
        spawn = random.choice(pot_spawn)

        if spawn > -1:
            self.spawn(robot, 0 if team == Team.WHITE else self.board_size - 1, spawn)
        
        return self.boardToMat(self.get_board())
    
    def pawn_move(self, robot):
        team = self.get_team(robot)

        robot.logs.clear()
        robot.has_moved = False

        if team in self.do_train:
            move = None
            if self.pawn_turns % self.special_turn == 0:
                nr, nc, mat, move = self.pawn_rand(robot)
            else:
                nr, nc, mat, move = self.bot_turn(robot)

            self.pawn_ins.append(self.pawnToInVec(nr, nc, mat, move))
            self.pawn_teams.append(team)
            self.pawn_turns += 1
        else:
            self.bot_turn(robot)
    
    def lord_move(self, robot):
        team = self.get_team(robot)

        robot.logs.clear()
        robot.has_moved = False

        if team in self.do_train:
            mat = None
            if self.lord_turns % self.special_turn == 0:
                mat = self.lord_rand(robot)
            else:
                mat = self.bot_turn(robot)

            self.lord_ins.append(self.lordToInVec(mat))
            self.lord_teams.append(team)
            self.lord_turns += 1
        else:
            self.bot_turn(robot)

    def turn(self):
        if self.running:
            self.round += 1

        if self.round > self.max_rounds:
            self.check_over()

        for i in range(self.robot_count):
            if i in self.queue:
                robot = self.queue[i]
                self.pawn_move(robot)
                self.check_over()

        if self.running:
            for robot in self.lords:
                self.lord_move(robot)

            self.lords.reverse()  # the HQ's will alternate spawn order
            self.board_states.append([row[:] for row in self.board])
        else:
            # for i in self.pawn_teams:
            #     self.pawn_outs.append([1 if i == self.winner else 0])
            # for i in self.lord_teams:
            #     self.lord_outs.append([1 if i == self.winner else 0])

            self.log_info('white heu ' + str(self.white_heu))
            self.log_info('black heu ' + str(self.black_heu))

            for i in self.pawn_teams:
                self.pawn_outs.append([(self.white_heu if i == Team.WHITE else self.black_heu) / self.good_heu])
            for i in self.lord_teams:
                self.lord_outs.append([(self.white_heu if i == Team.WHITE else self.black_heu) / self.good_heu])

            # raise GameError('game is over')
    
    def getTrainingData(self):
        return self.pawn_ins, self.pawn_outs, self.lord_ins, self.lord_outs
    
    def getHeuristics(self):
        return self.white_heu, self.black_heu

class RobotError(Exception):
    """Raised for illegal robot inputs"""
    pass


class GameError(Exception):
    """Raised for errors that arise within the Game"""
    pass

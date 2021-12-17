# ！/usr/bin/env python
# -*- coding:utf-8 -*-
# author:Dongze Xie & Guochen Yu time: 2021/11/30

from tkinter import *
import tkinter.messagebox
from robot import Robot
import random

class GameGo:

    def __init__(self):
        """
        initialization content：
            board: the checkerboard（two-dimensional array），-1：empty；0：white；1：black/ Or it can set up as 1 to 225，odds are black ，evens are white
            someone_win: the flag that one side wins，0：white win；1：black win   undecided/true:false
            mode: Game mode selection（0/1），0：human vs human；1：human vs AI
            select_color: who goes first（0/1），0：pick white side；1：pick black side
            window: the window for game
            can: canvas，for drawing the checkerboard
            white_done: save the positions done by the white side
            black_done: save the positions done by the black side
            chess_piece_done：save occupied positions，index stands for odd:black, even:white
        """
        self.mode = 0  # default pvp
        self.is_start = False
        self.someone_win = False
        self.select_color = 1  # default color black player
        self.player_turn = 1  # default black move first
        self.board = self.init_board()
        self.white_done = []
        self.black_done = []
        self.white_done_done = []
        self.black_done_done = []
        self.window = Tk()
        self.window.title("CIS667 final project GOMOKU")
        self.window.geometry("600x470+100+100")
        self.window.resizable(0, 0)
        self.can = Canvas(self.window, bg="skyblue", width=470, height=470)
        self.draw_board()
        self.robot = Robot(self.board)
        self.can.grid(row=0, column=100)

    @staticmethod
    def init_board():
        """initialize the abstract board ,two dimension array"""
        list1 = [[-1] * 15 for i in range(15)]
        return list1

    def draw_board(self):
        """draw the board"""
        for i in range(0,15):
            if i == 0 or i == 14:
                self.can.create_line((25, 25 + i * 30), (445, 25 + i * 30), width=3)
                self.can.create_line((25 + i * 30, 25), (25 + i * 30, 445), width=3)
            else:
                self.can.create_line((25, 25 + i * 30), (445, 25 + i * 30), width=1)
                self.can.create_line((25 + i * 30, 25), (25 + i * 30, 445), width=1)
        self.can.create_oval(112, 112, 118, 118, fill="black")
        self.can.create_oval(352, 112, 358, 118, fill="black")
        self.can.create_oval(112, 352, 118, 358, fill="black")
        self.can.create_oval(232, 232, 238, 238, fill="black")
        self.can.create_oval(352, 352, 358, 358, fill="black")

    def select_mode(self, mode_flag):
        """modeselection"""
        if self.is_start is False:
            if mode_flag == "pvp":
                """human vs human"""
                print("zhi xing le pvp")
                self.mode = 0
            elif mode_flag == "cvp_b":
                """AI vs human ,AI use black"""
                print("zhi xing le cvp_b")
                self.mode = 1
                self.select_color = 0
            elif mode_flag == "cvp_w":
                """AI vs human ,AI use white"""
                print("zhi xing le cvp_w")
                self.mode = 1
                self.select_color = 1
        else:
            return

    def select_mode1(self):
        """pvp"""
        self.select_mode("pvp")

    def select_mode2(self):
        """cvp_b"""
        self.select_mode("cvp_b")

    def select_mode3(self):
        """cvp_w"""
        self.select_mode("cvp_w")

    def pos_in_game_board(self, position):
        """find position in the board UI"""
        global r
        r = random.randint(0, 9)
        if r != 9:
            return position[0] * 30 + 25, position[1] * 30 + 25
        else:#10%chance teleport to its left position
            return position[0] * 30 - 5, position[1] * 30 + 25

    def pos_to_draw(self, x, y):
        """return two coordinate of the oval's circumscribed rectangle """
        return x - 10, y - 10, x + 10, y + 10

    def draw_chess_pieces(self, position, player=None):
        """draw ovals already on the board"""
        global r
        print(position)  # position stands for the coordinate of the board's two dimension array
        _x, _y = self.pos_in_game_board(position)
        oval = self.pos_to_draw(_x, _y)
        if player == 0:
            if r == 9:
                self.can.create_oval(oval, fill="white")
                self.white_done.append([position[0] - 1, position[1], 0])
                self.board[position[0]][position[1]] = 0
            else:  # teleport to the left
                self.can.create_oval(oval, fill="white")
                self.white_done.append([position[0], position[1], 0])
                self.board[position[0]][position[1]] = 0
        elif player == 1:
            if r == 9: #teleport to the left
                self.can.create_oval(oval, fill="black")
                self.black_done.append([position[0] - 1, position[1], 1])
                self.board[position[0]][position[1]] = 1
            else:
                self.can.create_oval(oval, fill="black")
                self.black_done.append([position[0], position[1], 1])
                self.board[position[0]][position[1]] = 1
        print("white_done: ", self.white_done)
        print("black_done: ", self.black_done)

    def not_done(self, position):
        """check if the point in two dimension array is already occupied"""
        return self.board[position[0]][position[1]] == -1

    def not_done1(self, x, y, chess):
        """check if the corresponding (x,y)already occupied,AKA already in the white done or black done"""
        if len(chess) == 0:
            return True
        flag = 0
        # point = x, y
        for p in chess:
            if p[0] == x and p[1] == y:
                flag = 1
        if flag == 1:
            return False
        else:
            return True

    @staticmethod
    def get_pos_in_board(x, y):
        """Get the position in the two-dimensional array chessboard"""
        return (x + 25) // 30 - 1, (y + 25) // 30 - 1

    def man_play(self, event):

        if self.someone_win is True or self.is_start is False:
            """If someone wins or hasn't started yet, you can't play"""
            return

        ex = event.x
        ey = event.y
        # print(ex, ey)
        if not (10 < ex < 460 and 10 < ey < 460):
            """If the mouse clicks outside the chessboard, you cannot play chess"""
            return

        pos_in_board = self.get_pos_in_board(ex, ey)
        print(pos_in_board)
        print("mode:", self.mode)
        """
            If the point has not been played, then according to the black and white moves of the pieces you hold.
            If the opponent chooses the game between everyone or the game based on the value of the model
        """
        
        if self.someone_win is False and self.is_start is True:
            if self.not_done(pos_in_board):
                if self.mode == 0:  # man to man
                    print("player_turn0:", self.player_turn)
                    self.draw_chess_pieces(pos_in_board, self.player_turn)
                    print("player_turn1:", self.player_turn)
                    self.someone_win = self.check_someone_win()
                    self.player_turn = 1 - self.player_turn
                    print("player_turn2:", self.player_turn)
                else:  # man-machine game
                    if self.select_color == 1:  # man take black first
                        if self.player_turn == 1:
                            self.draw_chess_pieces(pos_in_board, 1)
                            self.someone_win = self.check_someone_win()

                            self.ai_play()
                            self.someone_win = self.check_someone_win()
                    else:
                        if self.player_turn == 0:
                            self.draw_chess_pieces(pos_in_board, 0)
                            self.someone_win = self.check_someone_win()

                            self.ai_play()
                            self.someone_win = self.check_someone_win()


    def ai_play(self):
        """AI to play"""
        print("play_turn:", self.player_turn)
        if self.player_turn == 1:
            # man take black
            _x, _y, _z = self.robot.MaxValue_po(1, 0)
            position_in_matrix = _x, _y
            self.draw_chess_pieces(position_in_matrix, 0)
        else:
            _x, _y, _z = self.robot.MaxValue_po(0, 1)
            position_in_matrix = _x, _y
            print("position_in_matrix:", position_in_matrix)
            self.draw_chess_pieces(position_in_matrix, 1)


    def check_someone_win(self):
        """check if anyone has won"""
        if self.five_in_a_row(self.black_done):
            self.show_win_info("Black Win!!!")
            return True
        elif self.five_in_a_row(self.white_done):
            self.show_win_info("White Win!!!")
            return True
        else:
            return False

    def five_in_a_row(self, someone_done):
        """five stones"""
        if len(someone_done) == 0:
            return False
        else:
            for row in range(15):
                for col in range(15):
                    if ( not self.not_done1(row, col, someone_done) and
                         not self.not_done1(row + 1, col, someone_done) and
                         not self.not_done1(row + 2, col, someone_done) and
                         not self.not_done1(row + 3, col, someone_done) and
                         not self.not_done1(row + 4, col, someone_done)):
                        return True
                    elif (not self.not_done1(row, col, someone_done) and
                          not self.not_done1(row, col + 1, someone_done) and
                          not self.not_done1(row, col + 2, someone_done) and
                          not self.not_done1(row, col + 3, someone_done) and
                          not self.not_done1(row, col + 4, someone_done)):
                        return True
                    elif (not self.not_done1(row, col, someone_done) and
                          not self.not_done1(row+ +1, col + 1, someone_done) and
                          not self.not_done1(row + 2, col + 2, someone_done) and
                          not self.not_done1(row + 3, col + 3, someone_done) and
                          not self.not_done1(row + 4, col + 4, someone_done)):
                        return True
                    elif (not self.not_done1(row, col, someone_done) and
                          not self.not_done1(row + 1, col - 1, someone_done) and
                          not self.not_done1(row + 2, col - 2, someone_done) and
                          not self.not_done1(row + 3, col - 3, someone_done) and
                          not self.not_done1(row + 4, col - 4, someone_done)):
                        return True
                    else:
                        pass
        return False

    def show_win_info(self, winner):
        """Prompt for winning information"""
        print(winner)
        tkinter.messagebox.showinfo("Game Over", winner)

    def draw_chess_pieces_done(self, chess):
        """Draw the dots in black_done and white_done during single-step reset"""
        for p in chess:
            _x, _y = self.pos_in_game_board(p)
            oval = self.pos_to_draw(_x, _y)
            if p[2] == 0:
                self.can.create_oval(oval, fill="white")
            else:
                self.can.create_oval(oval, fill="black")


    def start_button(self):
        """Start Game"""
        if self.is_start is False:
            self.is_start = True
            if self.mode == 0:  # man to man
                print("ren ren dui yi")
            elif self.mode == 1:  # man to machine
                print("ren ji dui yi")
                if self.select_color == 1:  # man take black first
                    return

                elif self.select_color == 0:  # machine takes black first
                    self.player_turn = 0
                    if len(self.black_done) == 0 and len(self.white_done) == 0:
                        print("dian nao zhi hei xian shou")
                        position_in_matrix = 7, 7
                        self.draw_chess_pieces(position_in_matrix, 1)
                    return
        else:
            return

    def undo_all(self):
        """reset board"""
        self.board = self.init_board()
        print("undo_all:")
        # for i in range(15):
        #     for j in range(15):
        #         print(self.board[i][j], end="  ")
        #     print("\n")
        self.someone_win = False
        self.is_start = False
        self.mode = 0
        self.select_color = 1
        self.player_turn = 1
        self.white_done.clear()
        self.black_done.clear()
        self.can.delete("all")
        self.draw_board()
        self.robot = Robot(self.board)
        self.can.grid(row=0, column=0)

    def start(self):
        """Function button"""
        b1 = Button(self.window, text="Start", command=self.start_button, width=10)  
        b1.place(relx=0, rely=0, x=495, y=150)

        b2 = Button(self.window, text="Restart", command=self.undo_all, width=10)
        b2.place(relx=0, rely=0, x=495, y=250)

        """menu bar"""
        menu = Menu(self.window)
        submenu = Menu(menu, tearoff=0)
        submenu.add_command(label="New")
        submenu.add_command(label="Rule")
        submenu.add_command(label="Quit")
        menu.add_cascade(label="Game", menu=submenu)

        submenu = Menu(menu, tearoff=0)
        submenu.add_command(label="Player VS Player", command=self.select_mode1)
        submenu.add_command(label="Computer plays black", command=self.select_mode2)
        submenu.add_command(label="Computer plays white", command=self.select_mode3)
        menu.add_cascade(label="ModeSelect", menu=submenu)
        self.window.config(menu=menu)
        # Detect the click action of the left mouse button and return the coordinate value in the canvas
        self.can.bind("<Button-1>", lambda x: self.man_play(x))
        self.window.mainloop()


if __name__ == "__main__":
    game = GameGo()
    game.start()
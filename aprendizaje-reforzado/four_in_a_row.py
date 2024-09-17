import numpy as np
import random
import math
import os

ROW_COUNT = 4
COLUMN_COUNT = 5
PLAYER = 0
AI = 1
EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2
WINDOW_LENGTH = 4

if not os.path.exists('análisis'):
    os.makedirs('análisis')

def create_board():
    return np.zeros((ROW_COUNT, COLUMN_COUNT))

def drop_piece(board, row, col, piece): #piece can be from AI / Human
	board[row][col] = piece

def is_valid_location(board, col): 
	return board[ROW_COUNT-1][col] == 0

def get_next_open_row(board, col):
	for r in range(ROW_COUNT):
		if board[r][col] == 0:
			return r

def print_board(board):
	print(np.flip(board, 0))

def winning_move(board, piece):
	# Check horizontal locations for win
	for c in range(COLUMN_COUNT-3):
		for r in range(ROW_COUNT):
			if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
				return True

	# Check vertical locations for win
	for c in range(COLUMN_COUNT):
		for r in range(ROW_COUNT-3):
			if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
				return True

	# Check positively sloped diaganols
	for c in range(COLUMN_COUNT-3):
		for r in range(ROW_COUNT-3):
			if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
				return True

	# Check negatively sloped diaganols
	for c in range(COLUMN_COUNT-3):
		for r in range(3, ROW_COUNT):
			if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
				return True

# FUNCIÓN DE EVALUACIÓN POR HILERAS -> A MAYOR NUMERO DE PIEZAS, MAYOR SCORE
def evaluate_window(window, piece):
	score = 0
	opp_piece = PLAYER_PIECE
	if piece == PLAYER_PIECE:
		opp_piece = AI_PIECE

	if window.count(piece) == 4:
		score += 100
	elif window.count(piece) == 3 and window.count(EMPTY) == 1:
		score += 5
	elif window.count(piece) == 2 and window.count(EMPTY) == 2:
		score += 2

	if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
		score -= 4

	return score

def score_position(board, piece):
	score = 0

	## Score center column
	center_array = [int(i) for i in list(board[:, COLUMN_COUNT//2])]
	center_count = center_array.count(piece)
	score += center_count * 3

	## Score Horizontal -> EVALUA CADA FILA
	for r in range(ROW_COUNT):
		row_array = [int(i) for i in list(board[r,:])]
		for c in range(COLUMN_COUNT-3):
			window = row_array[c:c+WINDOW_LENGTH]
			score += evaluate_window(window, piece)

	## Score Vertical -> EVALUA CADA COLUMNA
	for c in range(COLUMN_COUNT):
		col_array = [int(i) for i in list(board[:,c])]
		for r in range(ROW_COUNT-3):
			window = col_array[r:r+WINDOW_LENGTH]
			score += evaluate_window(window, piece)

	## Score posiive sloped diagonal -> EVALUA CADA DIAGONAL HACIA LA DERECHA
	for r in range(ROW_COUNT-3):
		for c in range(COLUMN_COUNT-3):
			window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
			score += evaluate_window(window, piece)

	## -> EVALUA CADA DIAGONAL HACIA LA IZQUIERDA
	for r in range(ROW_COUNT-3):
		for c in range(COLUMN_COUNT-3):
			window = [board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
			score += evaluate_window(window, piece)

	# RETORNA LA SUMA DE TODAS LAS EVALUACIONES
	return score

def is_terminal_node(board):
	return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0

# INICIALMENTE SE LANZA CON PROFUNDIDAD 5 Y DESDE UN NODO DE MAXIMIZACIÓN
# @maximizingPlayer INDICA SI ES UN NODO MAX
def minimax(board, depth, alpha, beta, maximizingPlayer):
	# COLUMNAS NO LLENAS
    valid_locations = get_valid_locations(board)

	# VALIDA SI EL JUEGO TERMINA EN ESTE TABLERO: AI GANO/PLAYER GANO/EMPATE
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:

			# RETORNA BUENA VALORACIÓN PARA JUGADAS GANADORAS DE IA
            if winning_move(board, AI_PIECE):
                return (None, 100000000000000)

			# RETORNA MALA VALORACIÓN PARA JUGADAS GANADORAS DE PLAYER
            elif winning_move(board, PLAYER_PIECE):
                return (None, -10000000000000)

			# RETORNA 0 PARA JUGADAS DE EMPATE
            else: # Game is over, no more valid moves
                return (None, 0)
        else: # Depth is zero
			# RETORNA LA EVALUACIÓN DE LA POSICIÓN DEL TABLERO
            return (None, score_position(board, AI_PIECE))

	# SI EL JUEGO NO TERMINÓ -> Y EL ALGORITMO TIENE UN NODO MAX
    if maximizingPlayer:
        value = -math.inf

		# CÁLCULA MEJOR MOVIMIENTO INMEDIATO
        column = pick_best_move(board, AI_PIECE)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)

			# ESTIMA LA EVALUACIÓN DE LAS JUGADAS SIGUIENTES...
            new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]

			# SI, TRAS ESAS JUGADAS LA EVALUACIÓN ES MEJOR
            # SE ELIGE COMO JUGADA SIGUIENTE
            if new_score > value:
                value = new_score
                column = col

			# PODA DEL ÁRBOL
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        log_turn(depth, "MAX", column, value)
        return column, value

    else:
        value = math.inf
        column = pick_best_move(board, AI_PIECE)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        log_turn(depth, "MIN", column, value)
        return column, value

def get_valid_locations(board):
	valid_locations = []
	for col in range(COLUMN_COUNT):
		if is_valid_location(board, col):
			valid_locations.append(col)
	return valid_locations

# CALCULADORA DE LA MEJOR POSICIÓN A JUGAR
def pick_best_move(board, piece):

	valid_locations = get_valid_locations(board)
	best_score = -10000
	best_col = random.choice(valid_locations)
	for col in valid_locations:
		row = get_next_open_row(board, col)
		temp_board = board.copy()
		drop_piece(temp_board, row, col, piece)
		score = score_position(temp_board, piece)
		if score > best_score:
			best_score = score
			best_col = col

	return best_col

def log_turn(depth, player_type, column, value):
    indent = "    " * (5-depth)
    game_status = f"{indent}{player_type} [{column}]: {value}"
    with open(f'análisis/turn_{cont_turn}.txt', 'w') as f:
        f.write(game_status + '\n')

board = create_board()
Human = input("Input your Name : ")
print_board(board)
game_over = False
cont_turn = 0
turn = random.randint(PLAYER, AI)

while not game_over:
#PLAYER INPUT
    if turn == PLAYER:
        print("{0}, Make your Move (0-6) :".format(Human))
        #col = int(input("{0}, Make your Move (0-6) :"))
        col = int(input())
                
        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, PLAYER_PIECE)
            print_board(board)
            
            if winning_move(board, PLAYER_PIECE):
                game_over = True
                print("{0} has Won".format(Human))

        turn += 1
        turn = turn % 2



#AI INPUT
    if turn == AI and not game_over:
        print("AI's Move :")
        col, minimax_score = minimax(board, 5, -math.inf, math.inf, True)

        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, AI_PIECE)
            print_board(board)            

            if winning_move(board, AI_PIECE):
                game_over = True
                print("AI has Won")

            print_board(board)

        turn += 1
        turn = turn % 2

        cont_turn += 1

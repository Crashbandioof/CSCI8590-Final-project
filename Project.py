import torch
import torch.nn as nn
import torch.optim as optim
import chess
import random
import numpy as np
import math
def print_board(board):
    print("\n  a b c d e f g h")
    print(" +----------------+")

    rows = str(board).split("\n")
    for i, row in enumerate(rows):
        print(f"{8-i}|{' '.join(row.split())}|")

    print(" +----------------+")
    print()


def human_vs_ai(model):
    board = chess.Board()
    mcts = MCTS(model, simulations=100)

    print("Chess vs AI")
    print("Enter moves like: e2e4")
    print("Type 'quit' to exit\n")

    while not board.is_game_over():
        print_board(board)

        if board.turn == chess.WHITE:
            # Human move
            while True:
                move_str = input("Your move: ").strip()

                if move_str.lower() == "quit":
                    return

                try:
                    move = chess.Move.from_uci(move_str)

                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Illegal move.")
                except:
                    print("Invalid format. Example: e2e4")

        else:
            # AI move
            print("AI thinking...")
            move = mcts.run(board)
            print(f"AI plays: {move}")
            board.push(move)

    print_board(board)
    print("Game Over:", board.result())
def backpropagate(self, path, value):
    for node in reversed(path):
        node.visits += 1
        node.value_sum += value
        value = -value  # switch perspective
def expand(self, node):
    board = node.board

    if board.is_game_over():
        result = board.result()
        if result == "1-0":
            return 1
        elif result == "0-1":
            return -1
        return 0

    state = board_to_tensor(board).unsqueeze(0)

    with torch.no_grad():
        policy_logits, value = self.model(state)

    policy = torch.softmax(policy_logits, dim=1)[0]

    for move in board.legal_moves:
        idx = move_to_index(move)

        new_board = board.copy()
        new_board.push(move)

        child = MCTSNode(new_board, parent=node)
        child.prior = policy[idx].item()

        node.children[move] = child

    return value.item()
def select(self, node):
    best_score = -float("inf")
    best_move = None
    best_child = None

    for move, child in node.children.items():
        ucb = child.value() + self.c_puct * child.prior * math.sqrt(node.visits) / (1 + child.visits)

        if ucb > best_score:
            best_score = ucb
            best_move = move
            best_child = child

    return best_move, best_child
class MCTS:
    def __init__(self, model, simulations=50, c_puct=1.0):
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct

    def run(self, board):
        root = MCTSNode(board)

        # Expand root
        self.expand(root)

        for _ in range(self.simulations):
            node = root
            path = [node]

            # --- Selection ---
            while node.is_expanded():
                move, node = self.select(node)
                path.append(node)

            # --- Expansion ---
            value = self.expand(node)

            # --- Backprop ---
            self.backpropagate(path, value)

        return self.select_action(root)
class MCTSNode:
    def __init__(self, board, parent=None):
        self.board = board
        self.parent = parent
        self.children = {}

        self.visits = 0
        self.value_sum = 0
        self.prior = 0

    def value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0

    def is_expanded(self):
        return len(self.children) > 0
# --- Board Encoding ---
def board_to_tensor(board):
    piece_map = board.piece_map()
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    for square, piece in piece_map.items():
        row = square // 8
        col = square % 8
        idx = piece.piece_type - 1 + (0 if piece.color else 6)
        tensor[idx][row][col] = 1

    return torch.tensor(tensor)

# --- Neural Network ---
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(12, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
        )

        # Policy head (predict moves)
        self.policy = nn.Linear(256, 4672)  # max legal moves

        # Value head (win/loss prediction)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return self.policy(x), torch.tanh(self.value(x))


# --- Move Encoding ---
def move_to_index(move):
    return move.from_square * 64 + move.to_square

def index_to_move(index):
    return chess.Move(index // 64, index % 64)


# --- Self-Play Game ---
def play_game(model):
    board = chess.Board()
    states, policies, values = [], [], []

    while not board.is_game_over():
        state = board_to_tensor(board).unsqueeze(0)

        with torch.no_grad():
            policy_logits, _ = model(state)

        legal_moves = list(board.legal_moves)
        move_probs = []

        for move in legal_moves:
            idx = move_to_index(move)
            move_probs.append(policy_logits[0][idx].item())

        probs = torch.softmax(torch.tensor(move_probs), dim=0).numpy()
        move = random.choices(legal_moves, weights=probs)[0]

        states.append(state)
        policies.append(move_to_index(move))

        board.push(move)

    result = board.result()

    if result == "1-0":
        value = 1
    elif result == "0-1":
        value = -1
    else:
        value = 0

    values = [value] * len(states)

    return states, policies, values


# --- Training Loop ---
def train(model, epochs=10, games_per_epoch=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        all_states, all_policies, all_values = [], [], []

        for _ in range(games_per_epoch):
            s, p, v = play_game(model)
            all_states.extend(s)
            all_policies.extend(p)
            all_values.extend(v)

        states = torch.cat(all_states)
        policies = torch.tensor(all_policies)
        values = torch.tensor(all_values, dtype=torch.float32)

        pred_policy, pred_value = model(states)

        loss_policy = nn.CrossEntropyLoss()(pred_policy, policies)
        loss_value = nn.MSELoss()(pred_value.squeeze(), values)

        loss = loss_policy + loss_value

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    model = ChessNet()

    print("Training...")
    train(model, epochs=20, games_per_epoch=20)

    human_vs_ai(model)
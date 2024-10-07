import random
import numpy as np
from scipy.optimize import linprog
import math

# Given the initial distance D and c,k parameters of the accuracy functions of the players, calculates the payoff matrix of the game
def calculate_A(D, c1, k1, c2, k2):
    A = np.zeros((D, D))
    for d1 in range(1, D + 1):
        for d2 in range(1, D + 1):
            for x1 in range(0, D + 1):
                for x2 in range(D, -1, -1):
                    d = abs(x1 - x2)
                    if d != 0:  # Check if d is not zero to avoid division by zero
                        if d % 2 == 0 and d <= d1:
                            A[d1 - 1, d2 - 1] = 2 * c1 / (d ** k1) - 1
                        elif d % 2 == 1 and d <= d2:
                            A[d1 - 1, d2 - 1] = 1 - 2 * c2 / (d ** k2)
    return A


# Solves the game given its' payoff matrix. Returns the value of the game and the optimal strategy for every player
def minmax(A):
    r, c = A.shape

    # Solve for Player 1's strategy
    AA1 = np.hstack([-A.T, np.ones((c, 1))])
    Aeq1 = np.append(np.ones(r), 0).reshape(1, -1)
    b1 = np.zeros(c)
    beq1 = 1
    lb1 = [(0, None)] * r + [(-np.inf, None)]
    f1 = np.append(np.zeros(r), -1)

    result1 = linprog(f1, A_ub=AA1, b_ub=b1, A_eq=Aeq1, b_eq=beq1, bounds=lb1, method='highs')

    if not result1.success:
        raise ValueError("Linear programming did not converge for Player 1")

    p1 = result1.x[:r]
    v1 = result1.x[r]

    # Solve for Player 2's strategy
    AA2 = np.hstack([-A, np.ones((r, 1))])
    Aeq2 = np.append(np.ones(c), 0).reshape(1, -1)
    b2 = np.zeros(r)
    beq2 = 1
    lb2 = [(0, None)] * c + [(-np.inf, None)]
    f2 = np.append(np.zeros(c), -1)

    result2 = linprog(f2, A_ub=AA2, b_ub=b2, A_eq=Aeq2, b_eq=beq2, bounds=lb2, method='highs')

    if not result2.success:
        raise ValueError("Linear programming did not converge for Player 2")

    p2 = result2.x[:c]
    v2 = result2.x[c]

    return v1, p1, p2

# # Example usage with a payoff matrix
# A = np.array([
#     [0.0024, 0.0024, 0.0024, 0.0024, -0.0024],
#     [0.4648, 0.0718, 0.0718, 0.0718, 0.0718],
#     [0.4648, 0.3777, 0.1684, 0.1684, 0.1684],
#     [0.4648, 0.3777, 0.2385, 0.3195, 0.3195],
#     [0.4648, 0.3777, 0.2385, -0.0346, 0.6245]
# ])

# v, p1, p2 = minmax(A)
# print("Value of the game:", v)
# print("Optimal strategy for Player 1:", p1)
# print("Optimal strategy for Player 2:", p2)


def accuracy_function(distance,c_n, k_n):
    if distance <= 1:
        return 1.0
    else:
        return max(0.1, c_n / distance**k_n)

class Player:
    def __init__(self, name, accuracy_function):
        self.name = name
        self.accuracy_function = accuracy_function
        self.position = 0
        self.bullet = 1
        self.distribution=[]
    def move(self):
        if self.name == "Player 1":
            self.position += 1
        elif self.name == "Player 2":
            self.position -= 1
        print(f"{self.name} moves to position {self.position}")

    def shoot(self, target_distance,c_n,k_n):
        if self.bullet > 0:
            accuracy = self.accuracy_function(target_distance,c_n,k_n)
            self.bullet -= 1
            hit = random.random() < accuracy
            print(f"{self.name} shoots with accuracy {accuracy:.2f} and {'hits' if hit else 'misses'}!")
            return hit
        else:
            print(f"{self.name} has no bullets left!")
            return False

# The simulation of the game. Inputs:
    # The players
    # The initial distance between the players
    # The number of max rounds (a round is completed when one player makes a move)
    # The real k parameters of the players
    # The initial estimations of the k parameters
def duel(player1, player2, initial_distance, rounds,c_1,c_2,k_1=0.6,k_2=0.7):
    player1.position = 0
    player2.position = initial_distance
    round_number = 1
    player1.bullet = 1
    player2.bullet = 1
    end = 0



    estimated_strategy_1 = np.random.choice(initial_distance,p=player1.distribution)
    estimated_strategy_2 = np.random.choice(initial_distance,p=player2.distribution)


    print(estimated_strategy_1,estimated_strategy_2)
    wins1=0
    wins2=0

    while end != 1:

        if round_number <= rounds:
            #print(f"\n--- Round {round_number} ---")
            if round_number % 2 == 1:
                if player1.bullet > 0:
                    if player2.bullet == 0 and player2.position - player1.position > 1:
                        player1.move()
                    elif player2.position - player1.position == 1:
                        if player1.shoot(player2.position - player1.position,c_1,k_1):
                            print(f"{player1.name} wins!")
                            print(estimated_strategy_1)
                            player1.distribution[estimated_strategy_1]*=1.1
                            total = sum(player1.distribution)
                            player1.distribution= [p/total for p in  player1.distribution]
                            print(player1.distribution)
                            player2.distribution[estimated_strategy_2]-=0.2*player2.distribution[estimated_strategy_2]
                            total = sum(player2.distribution)
                            player2.distribution= [p/total for p in  player2.distribution]
                            end = 1
                    else:
                        if player2.position - player1.position  < (estimated_strategy_1 +1):
                            player1.move()
                        else:
                            d=player2.position - player1.position
                            if player1.shoot(player2.position - player1.position,c_1,k_1):
                                print(f"{player1.name} wins!")
                                print(estimated_strategy_1)
                                player1.distribution[estimated_strategy_1]*=1.1
                                total = sum(player1.distribution)
                                player1.distribution= [p/total for p in  player1.distribution]
                                print(player1.distribution)
                                player2.distribution[estimated_strategy_2]-=0.2*player2.distribution[estimated_strategy_2]
                                total = sum(player2.distribution)
                                player2.distribution= [p/total for p in  player2.distribution]
                                wins1+=1
                                end = 1


            if round_number % 2 == 0:
                if player2.bullet > 0:
                    if player1.bullet == 0 and player2.position - player1.position > 1:
                        player2.move()
                    elif player2.position - player1.position == 1:
                        if player2.shoot(player2.position - player1.position,c_2,k_2):
                            print(f"{player2.name} wins!")
                            print(f"{player2.name} wins!")
                            print(estimated_strategy_2)
                            player2.distribution[estimated_strategy_2]*=1.1
                            total = sum(player2.distribution)
                            player2.distribution= [p/total for p in  player2.distribution]
                            print(player2.distribution)
                            player1.distribution[estimated_strategy_1]-=0.2*player1.distribution[estimated_strategy_1]
                            total = sum(player1.distribution)
                            player1.distribution= [p/total for p in  player1.distribution]
                            end = 1
                    else:
                        if player2.position - player1.position  < (estimated_strategy_2+1):
                            player2.move()
                        else:
                            d=player2.position - player1.position
                            if player2.shoot(player2.position - player1.position,c_2,k_2):
                                print(f"{player2.name} wins!")
                                print(estimated_strategy_2)
                                player2.distribution[estimated_strategy_2]*=1.1
                                total = sum(player2.distribution)
                                player2.distribution= [p/total for p in  player2.distribution]
                                print(player2.distribution)
                                player1.distribution[estimated_strategy_1]-=0.2*player1.distribution[estimated_strategy_1]
                                total = sum(player1.distribution)
                                player1.distribution= [p/total for p in  player1.distribution]
                                wins2+=1
                                end = 1
                elif player2.bullet == 0 and player2.position - player1.position > 1:
                    player2.move()

            round_number += 1

    return wins1,wins2

# Simulation parameters
initial_distance = 10
close_range_distance = 1
rounds = 100
samples=100
# Create players
player1 = Player("Player 1", accuracy_function)
player2 = Player("Player 2", accuracy_function)
player1.distribution=initial_distance*[1/initial_distance]
player2.distribution=initial_distance*[1/initial_distance]
# Start the duel
k_1=1
k_2=0.6
stop_1=0
stop_2=0
stops1=0
stops2=0
wins1=0
wins2=0
d=0


for i in range(3000):
    print(f"===============Game {i + 1} ==============")

    wins1=0
    wins2=0
    for j in range(1):
      w1, w2= duel(player1, player2, initial_distance, rounds,c_1=1,c_2=1,k_1=k_1,k_2=k_2)
      if w1:
        wins1+=1

      if w2:
        wins2+=1
    print("=============End of sampling=============")
    print("Distance:"+ str(d))
    print("Wins1:"+ str(wins1))
    print("Wins2:"+ str(wins2))
    if wins1:
      p1=wins1/samples

      print("Player1:" + str(p1))

      if d!=1:
          print("Hello")
    if wins2:
      p2=wins2/samples

      print("Player2:" + str(p2))

      if d!=1:
          print("Hello")
max1=max(player1.distribution)
max2=max(player2.distribution)
print(player1.distribution)
print(player2.distribution)
print(max1)
print(player1.distribution.index(max1)+1)
print(player2.distribution.index(max2)+1)
A2 = calculate_A(10, 1, k_1, 1, k_2)

v4, strategy_1, strategy_2 = minmax(A2)
strategy_1 = np.argmax(strategy_1)+1
strategy_2 = np.argmax(strategy_2)+1
print(strategy_1)
print(strategy_2)

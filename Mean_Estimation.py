
import random
from multiprocessing import Pool
import numpy as np
from scipy.optimize import linprog
from scipy.special import rel_entr
class Player:
    def __init__(self, name, distance):
        self.name = name
        self.threshold = random.randint(1, distance)  # Random threshold
        self.bullets = 1
        self.alive = True
        self.c_n = 1
        self.k_n = 0.5
        self.accuracy = min((self.c_n)/(distance**self.k_n),1)

    def should_shoot(self, distance):
        """
        Determines whether the player should shoot based on the distance and threshold.
        """
        return distance <= self.threshold


def calculate_A(D, estimation1,estimation2):
    A = np.zeros((D, D))
    for d1 in range(1, D + 1):
        for d2 in range(1, D + 1):
            for x1 in range(0, D + 1):
                for x2 in range(D, -1, -1):
                    d = abs(x1 - x2)
                    if d != 0:  # Check if d is not zero to avoid division by zero
                        if d % 2 == 0 and d <= d1:
                            A[d1 - 1, d2 - 1] = 2*estimation1[d] - 1
                        elif d % 2 == 1 and d <= d2:
                            A[d1 - 1, d2 - 1] = 1 - 2*estimation2[d]
    return A


def calculate_A_real(D, c1, k1, c2, k2):
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

def simulate_round_1(rounds, initial_distance, target):
    success = [0] * ((2 * initial_distance) + 1)
    fails = [0] * ((2 * initial_distance) + 1)
    trials= [0] * ((2 * initial_distance) + 1)
    for i in range((2 * initial_distance) + 1):
        if i!=0:
          target[i] = min((1) / ((i) ** 0.5), 1)

    while rounds > 0:
        distance = 2 * initial_distance
        player1 = Player(name="Player 1", distance=distance)
        player2 = Player(name="Player 2", distance=distance)

        while distance > 1:
            if player1.should_shoot(distance):
                accuracy = min((player1.c_n) / (distance ** player1.k_n), 1)
                if random.random() < accuracy:
                    player2.alive = False
                    success[distance] += 1
                    break
                else:
                    player2.alive = True
                    fails[distance] += 1
                    break

            if player2.should_shoot(distance):
                accuracy = min((player2.c_n) / (distance ** player2.k_n), 1)
                if random.random() < accuracy:
                    player1.alive = False
                    break
                else:
                    player1.alive = True
                    break

            if player1.alive and player2.alive:
                distance -= 1

        rounds -= 1
    for i in range((2 * initial_distance) + 1):
        trials[i] = success[i]+fails[i]
    estimation = [success[u] / (success[u] + fails[u]) if success[u] > 0 else 0 for u in range((2 * initial_distance) + 1)]
    return estimation, target,trials

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

def simulate_round_2(rounds, initial_distance, target):
    success = [0] * ((2 * initial_distance) + 1)
    fails = [0] * ((2 * initial_distance) + 1)
    trials= [0] * ((2 * initial_distance) + 1)
    for i in range((2 * initial_distance) + 1):
        if i!=0:
          target[i] = min((1) / ((i) ** 0.5), 1)

    while rounds > 0:
        distance = 2 * initial_distance
        player1 = Player(name="Player 1", distance=distance)
        player2 = Player(name="Player 2", distance=distance)

        while distance > 1:
            if player1.should_shoot(distance):
                accuracy = min((player1.c_n) / (distance ** player1.k_n), 1)
                if random.random() < accuracy:
                    player2.alive = False

                    break
                else:
                    player2.alive = True

                    break

            if player2.should_shoot(distance):
                accuracy = min((player2.c_n) / (distance ** player2.k_n), 1)
                if random.random() < accuracy:
                    player1.alive = False
                    success[distance] += 1
                    break
                else:
                    player1.alive = True
                    fails[distance] += 1
                    break

            if player1.alive and player2.alive:
                distance -= 1

        rounds -= 1
    for i in range((2 * initial_distance) + 1):
        trials[i] = success[i]+fails[i]
    estimation = [success[u] / (success[u] + fails[u]) if success[u] > 0 else 0 for u in range((2 * initial_distance) + 1)]
    return estimation, target,trials

def simulation_parallel(num_rounds, initial_distance, num_processes):
    pool = Pool(processes=num_processes)
    target = [0] * ((2 * initial_distance) + 1)
    results1 = pool.starmap(simulate_round_1, [(num_rounds // num_processes, initial_distance, target)] * num_processes)
    results2 = pool.starmap(simulate_round_2, [(num_rounds // num_processes, initial_distance, target)] * num_processes)
    pool.close()
    pool.join()



    print("Estimations1:", results1[0][0])
    print("Target1:", results1[0][1])
    print("Trials1:", results1[0][2])


    print("Estimations2:", results2[0][0])
    print("Target2:", results2[0][1])
    print("Trials2:", results2[0][2])

    A=calculate_A(10,results1[0][0],results2[0][0])

    print(A)
    print(minmax(A))
    A_real=calculate_A_real(10,1,0.5,1,0.5)
    print(minmax(A_real))
    print("Results1:",results1[0][0])
    print("Results2:",results2[0][0])
    print("Target:", results1[0][1])
    print("Target:", results2[0][1])
    print("Relative Entropy1:",rel_entr(results1[0][0],results1[0][1]))
    print("Relative Entropy2:",rel_entr(results2[0][0],results2[0][1]))
    print("Relative Entropy1:",np.linalg.norm(rel_entr(results1[0][0],results1[0][1])))
    print("Relative Entropy2:",np.linalg.norm(rel_entr(results2[0][0],results2[0][1])))
    print(num_rounds)
if __name__ == "__main__":
    simulation_parallel(10000, 10, 4)  # 4 processes for parallel execution

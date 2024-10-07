import random
import numpy as np
from scipy.optimize import linprog
import math
import time
import matplotlib.pyplot as plt
from scipy.special import rel_entr
def calculate_A(D, c1, k1, c2, k2):
    """
    calculate_A: Given the initial parameteres of the game computes the payoff matrix of the game

    param D: The initial distance between the 2 players
    param c1: The parameter c of the accuracy function of the player 1
    param k1: The parameter k of the accuracy function of the player 1
    param c2: The parameter c of the accuracy function of the player 2
    param k2: The parameter k of the accuracy function of the player 2


    return: The payoff matrix of the game
    """

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

def minmax(A):
    """
    minmax: Given the payoff matrix of the game computes the Nash equilibrium of the game

    param A: The payoff matrix of the game



    return: The strategies of the 2 players that leads to a Nash equilibrium
    """
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

# Example usage with a payoff matrix
A = np.array([
    [0.0024, 0.0024, 0.0024, 0.0024, -0.0024],
    [0.4648, 0.0718, 0.0718, 0.0718, 0.0718],
    [0.4648, 0.3777, 0.1684, 0.1684, 0.1684],
    [0.4648, 0.3777, 0.2385, 0.3195, 0.3195],
    [0.4648, 0.3777, 0.2385, -0.0346, 0.6245]
])

v, p1, p2 = minmax(A)
print("Value of the game:", v)
print("Optimal strategy for Player 1:", p1)
print("Optimal strategy for Player 2:", p2)

# ----



def accuracy_function(distance,c_n=1, k_n=1):
    if distance <= 1:
        return 1.0
    else:
        return min(1.0, c_n / distance**k_n)


# ------



class Player:
    def __init__(self, name, accuracy_function):
        self.name = name
        self.accuracy_function = accuracy_function
        self.position = 0
        self.bullet = 1

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

# -----



def duel(player1, player2, initial_distance, rounds,c_1=1.0,k_1=0.6,c_2=1.0,k_2=0.7,estimatedc_1=1.0, estimatedk_1=0.4,estimatedc_2=1.0, estimatedk_2=0.8,stop_1=0,stop_2=0):

    player1.position = 0
    player2.position = initial_distance
    round_number = 1
    player1.bullet = 1
    player2.bullet = 1
    end = 0





    A1 = calculate_A(initial_distance, c_1, k_1, estimatedc_2, estimatedk_2)

    A2 = calculate_A(initial_distance, estimatedc_1, estimatedk_1, c_2, k_2)

    v3, estimated_strategy_1, strategy_2 = minmax(A1)
    v3, strategy_1, estimated_strategy_2 = minmax(A2)


    print("Estimated k1:"+ str(estimatedk_1))
    print("Estimated k2:"+ str(estimatedk_2))
    print("Estimated c1:"+ str(estimatedc_1))
    print("Estimated c2:"+ str(estimatedc_2))
    estimated_strategy_1 = np.argmax(estimated_strategy_1)+1
    estimated_strategy_2 = np.argmax(estimated_strategy_2)+1
    print("Strategy 1:"+str(estimated_strategy_1))
    print("Strategy 2:"+str(estimated_strategy_2))

    if estimated_strategy_2<=3 and estimated_strategy_1<=2:
       if np.random.random()<0.5:
           estimated_strategy_2=4
       else:
           estimated_strategy_1=4


    if stop_2:
      estimated_strategy_1=estimated_strategy_2-1
      print("Estimate:"+ str(estimatedk_1))
      print("STOP 2")
    elif stop_1:
      estimated_strategy_2=estimated_strategy_1-1
      print("Estimate:"+ str(estimatedk_2))
      print("STOP 1")

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
                        d=1
                        if player1.shoot(player2.position - player1.position,c_1,k_1):
                            print(f"{player1.name} wins!")
                            end = 1
                    else:
                        if  (player2.position - player1.position) >= estimated_strategy_1:
                            player1.move()
                        else:
                            d=player2.position - player1.position
                            if player1.shoot(player2.position - player1.position,c_1,k_1):
                                print(f"{player1.name} wins!")

                                wins1+=1
                                end = 1


            if round_number % 2 == 0:
                if player2.bullet > 0:
                    if player1.bullet == 0 and player2.position - player1.position > 1:
                        player2.move()
                    elif player2.position - player1.position == 1:
                        d=1
                        if player2.shoot(player2.position - player1.position,c_2,k_2):
                            print(f"{player2.name} wins!")
                            end = 1
                    else:
                        if  (player2.position - player1.position) >= estimated_strategy_2:
                            player2.move()
                        else:
                            d=player2.position - player1.position
                            if player2.shoot(player2.position - player1.position,c_2,k_2):
                                print(f"{player2.name} wins!")

                                wins2+=1
                                end = 1
                elif player2.bullet == 0 and player2.position - player1.position > 1:
                    player2.move()

            round_number += 1

    return wins1,wins2,d


# ---



# Simulation parameters
initial_distance = 10
close_range_distance = 1
rounds = 100
samples=100
# Create players
player1 = Player("Player 1", accuracy_function)
player2 = Player("Player 2", accuracy_function)

# Start the duel
k_1=0.6
k_2=0.9
stop_1=0
stop_2=0
stops1=0
stops2=0
wins1=0
wins2=0
d=0
c_1=1.2
c_2=1.9
c1=0
c2=0
estimated_k1=[0.0,1.5]
estimated_c1=[0.0,3.0]
estimated_c2=[0.0,3.0]
estimated_k2=[0.0,1.5]
graph_k1=20*[0]
graph_k2=20*[0]
graph_c1=20*[0]
graph_c2=20*[0]
previous_estimate_2= estimated_k2[1]-estimated_k2[0]
previous_estimate_1= estimated_k1[1]-estimated_k1[0]
graph_k1[0]=estimated_k1[1]- estimated_k1[0]
graph_k2[0]=estimated_k2[1]- estimated_k2[0]
graph_c1[0]=estimated_c1[1]- estimated_c1[0]
graph_c2[0]=estimated_c2[1]- estimated_c2[0]
for i in range(15): # i is the current game number

    print(f"===============Game {i + 1} ==============")
    print(estimated_k1, estimated_k2)

    if i>1 and wins1:
      if abs(abs(previous_estimate_1-(estimated_k1[0]+estimated_k1[1])/2)<0.001):

       print("Stop2:"+ '\n')
       print("Previous: "+ str(previous_estimate_1) + " " + "New: "+ str((estimated_k1[0]+estimated_k1[1])/2))
       stops1+=1
       if stops1>2:
        stop_2=1

    if i>1 and wins2:
      if abs(abs(previous_estimate_2-(estimated_k2[0]+estimated_k2[1])/2)<0.001):
        print("Stop1:"+ '\n')
        print("Previous: "+ str(previous_estimate_2) + " " + "New: "+ str((estimated_k2[0]+estimated_k2[1])/2))
        stops2+=1
        if stops2>2:
          stop_1=1
    if stop_1==1 and stop_2==1:
      break
    wins1=0
    wins2=0


    for j in range(samples):
      w1, w2 ,d_sample= duel(player1, player2, initial_distance, rounds,c_1=c_1,k_1=k_1,c_2=c_2,k_2=k_2,estimatedc_1=(estimated_c1[0]+estimated_c1[1])/2, estimatedk_1=(estimated_k1[0]+estimated_k1[1])/2,estimatedc_2=(estimated_c2[0]+estimated_c2[1])/2, estimatedk_2=(estimated_k2[0]+estimated_k2[1])/2,stop_1=stop_1,stop_2=stop_2)
      if w1 and d_sample!=1:
        wins1+=1
      if w2 and d_sample!=1:
        wins2+=1
      if d_sample!=1:
        d=d_sample
    print("=============End of sampling=============")
    print("Distance:"+ str(d))
    print("Wins1:"+ str(wins1))
    print("Wins2:"+ str(wins2))
    if wins1:
      p1=wins1/samples

      print("Player1:" + str(p1))


      previous_estimate_1=(estimated_k1[1]+estimated_k1[0])/2
      if c1==0:
        c1=p1*(d**previous_estimate_1)
      else:
       c1=(estimated_c1[1]+estimated_c1[0])/2
      if p1<=0.1:
          if estimated_k1[1]>math.log(c1/(p1-p1/2))/math.log(d) and estimated_k1[0]<math.log(c1/(p1-p1/2))/math.log(d):
            estimated_k1[1]=math.log(c1/(p1-p1/2))/math.log(d)
          if estimated_k1[0]<math.log(c1/(p1+0.1))/math.log(d) and estimated_k1[1]>math.log(c1/(p1+0.1))/math.log(d):
            estimated_k1[0]=math.log(c1/(p1+0.1))/math.log(d)
      else:
          print(math.log(c1/(p1-0.1))/math.log(d))
          print(math.log(c1/(p1+0.1))/math.log(d))
          print(estimated_k1[0])
          print(estimated_k1[1])
          if estimated_k1[1]>math.log(c1/(p1-0.1))/math.log(d) and estimated_k1[0]<math.log(c1/(p1-0.1))/math.log(d):
            estimated_k1[1]=math.log(c1/(p1-0.1))/math.log(d)
          if estimated_k1[0]<math.log(c1/(p1+0.1))/math.log(d) and estimated_k1[1]>math.log(c1/(p1+0.1))/math.log(d):
            estimated_k1[0]=math.log(c1/(p1+0.1))/math.log(d)
      if estimated_c1[0]<p1*d**estimated_k1[0] and estimated_c1[1]>p1*d**estimated_k1[0]:
       estimated_c1[0]=p1*d**estimated_k1[0]
      if estimated_c1[1]>p1*d**estimated_k1[1] and estimated_c1[0]<p1*d**estimated_k1[1]:
        estimated_c1[1]=p1*d**estimated_k1[1]
    if wins2:
      p2=wins2/samples
      print("Player2:" + str(p2))
      previous_estimate_2=(estimated_k2[1]+estimated_k2[0])/2
      if c2==0:
        c2=p2*d**previous_estimate_2
      else:
        c2=(estimated_c2[1]+estimated_c2[0])/2
      if p2<=0.1:
          if estimated_k2[1]>math.log(c2/(p2-p2/2))/math.log(d) and estimated_k2[0]<math.log(c2/(p2-p2/2))/math.log(d):
            estimated_k2[1]=math.log(c2/(p2-p2/2))/math.log(d)
          if estimated_k2[0]<math.log(c2/(p2+0.1))/math.log(d) and estimated_k2[1]>math.log(c2/(p2+0.1))/math.log(d):
            estimated_k2[0]=math.log(c2/(p2+0.1))/math.log(d)
      else:
          print(str(d))
          print(str(p2))
          print(math.log(c2/(p2-0.1))/math.log(d))
          print(math.log(c2/(p2+0.1))/math.log(d))
          if estimated_k2[1]>math.log(c2/(p2-0.1))/math.log(d) and estimated_k2[0]<math.log(c2/(p2-0.1))/math.log(d):
            estimated_k2[1]=math.log(c2/(p2-0.1))/math.log(d)
          if estimated_k2[0]<math.log(c2/(p2+0.1))/math.log(d) and estimated_k2[1]>math.log(c2/(p2+0.1))/math.log(d):
            estimated_k2[0]=math.log(c2/(p2+0.1))/math.log(d)
      if estimated_c2[0]<p2*d**estimated_k2[0] and estimated_c2[1]>p2*d**estimated_k2[0]:
       estimated_c2[0]=p2*d**estimated_k2[0]
      if estimated_c2[1]>p2*d**estimated_k2[1] and estimated_c2[0]<p2*d**estimated_k2[1]:
        estimated_c2[1]=p2*d**estimated_k2[1]
    if wins1:
        print("C1",c1)
    if wins2:
        print("C2",c2)
    print("Estimated C1: ",estimated_c1," Estimated K1: ",estimated_k1," Estimated C2: ",estimated_c2," Estimated K2: " ,estimated_k2)
    if estimated_c1[0]>estimated_c1[1]:
        print("In game ", i, " we have error in c1")
        print()
        break
    if estimated_c2[0]>estimated_c2[1]:
        print(estimated_c2,p2*d**estimated_k2[1],p2*d**estimated_k2[0],estimated_k2)
        print("In game ", i, " we have error in c2")
        break
    if estimated_k1[0]>estimated_k1[1]:
        print("In game ", i, " we have error in k1")
        break
    if estimated_k2[0]>estimated_k2[1]:
        print("In game ", i, " we have error in k2")
        break
    graph_k1[i+1]=estimated_k1[1]-estimated_k1[0]
    graph_k2[i+1]=estimated_k2[1]-estimated_k2[0] 
    graph_c1[i+1]=estimated_c1[1]-estimated_c1[0]
    graph_c2[i+1]=estimated_c2[1]-estimated_c2[0]   
print((estimated_k1[1]+estimated_k1[0])/2, (estimated_k2[1]+estimated_k2[0])/2)

estimation_k1=(estimated_k1[1]+estimated_k1[0])/2
estimation_k2=(estimated_k2[1]+estimated_k2[0])/2
estimation_c1=(estimated_c1[1]+estimated_c1[0])/2
estimation_c2=(estimated_c2[1]+estimated_c2[0])/2
A1 = calculate_A(initial_distance, estimation_c1, estimation_k1, estimation_c2, estimation_k2)
v3, estimated_strategy_1, estimated_strategy_2 = minmax(A1)
estimated_strategy_1 = np.argmax(estimated_strategy_1)+1
estimated_strategy_2 = np.argmax(estimated_strategy_2)+1

A2 = calculate_A(10, 1, 0.7, 1, 0.4)

v4, strategy_1, strategy_2 = minmax(A2)
strategy_1 = np.argmax(strategy_1)+1
strategy_2 = np.argmax(strategy_2)+1
print("k1", estimated_k1)
print("k2", estimated_k2)
print("c1", estimated_c1)
print("c2", estimated_c2)
print("Estimated 1 -> ", estimated_strategy_1  , " "+"Estimated 2 ->", estimated_strategy_2 )
print("Real 1 -> " , strategy_1 , " "+"Real 2 ->", strategy_2 )

graph_k2 = np.array(graph_k2)
graph_c2 = np.array(graph_c2)
graph_k1 = np.array(graph_k1)
graph_c1 = np.array(graph_c1)
x=np.nonzero(graph_k1)

x = np.squeeze(x)

y1= graph_k1[np.nonzero(graph_k1)]
y2= graph_k2[np.nonzero(graph_k2)]
y3= graph_c1[np.nonzero(graph_c1)]
y4= graph_c2[np.nonzero(graph_c2)]
print(x)
print(np.nonzero(graph_k1))
print(np.nonzero(graph_k2))
# plotting

# Plotting the first line
plt.plot(x, y1, label='k1', color='blue', marker='o')

# Plotting the second line
plt.plot(x, y2, label='k2', color='green', marker='x')

# Plotting the first line
plt.plot(x, y3, label='c1', color='red', marker='o')

# Plotting the second line
plt.plot(x, y4, label='c2', color='purple', marker='x')
# Adding labels and title
plt.xlabel('Παιχνίδια')
plt.ylabel('Μέγεθος διαστήματος γνώσης')
plt.title('Μέγεθος διαστημάτων γνώσης με την πάροδο των παιχνιδιών')

for i in range(initial_distance):
    target1=accuracy_function(i,1.4,0.7)
    estimate1=accuracy_function(i,(estimated_c1[1]+estimated_c1[0])/2,(estimated_k1[1]+estimated_k1[0])/2)
    target2=accuracy_function(i,1.7,0.5)
    estimate2=accuracy_function(i,(estimated_c2[1]+estimated_c2[0])/2,(estimated_k2[1]+estimated_k2[0])/2)
    
print("Rel Entr 1:", rel_entr(target1,estimate1))
print("Rel Entr 2:", rel_entr(target2,estimate2))
# Adding a legend
plt.legend()

# Displaying the plot
plt.show()


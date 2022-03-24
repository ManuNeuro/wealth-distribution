"""
Created on Wed July 07 11:43:15 2021
@author: Emmanuel Calvet
"""
import numpy as np
import matplotlib.pyplot as plt
from random import sample, seed
from scipy.optimize import curve_fit
import time
import math as mt

# Function for the fitting
def expFit(x, a):
    return np.exp(-x/a)

def multiply(iterable):
    prod = 1
    for x in iterable:
        prod *= x
    return prod

# Fix seeds for reproducibility
seed(747)
np.random.seed(747)
time_start = time.perf_counter()

# Parameters of simulation
N = 1000 # Number of agents
nbPerIteration = 500 # Number of transaction per iteration (/!\ Must be low compared to N)
nbiteration = 50000
wealth = 100*np.ones(N) # Dirac distribution of wealth
agentsIndex = np.arange(0, N) # For the shuffling of indexes

# Transaction options
giveOption = 'fix' # Two possibilities : 'fix' or 'random'
fixAmount = 1 # Used if giveOption is set to 'fix'
propToMax = 0.1 # Used if givenOption is set to 'random'; proportion of the maximal amount of money for a transaction
threshold = 0 # 0 means no debt, a negative value means debt are allowed, you can also set positive threshold

# Plot options
plotIterations = True
nbPlot = 3
modulo = int(nbiteration/nbPlot)
if plotIterations:
    plt.figure()
    N_fac = mt.factorial(N) # Number of possible combination of N agents
    W = [] # Total Number of possible configuration of agents money
    iteration = []

# SIMULATION
for t in range(0, nbiteration):
    # Select random agent1 (giver)
    agent1 = np.array(sample(list(agentsIndex), nbPerIteration)) # Giver can only be selected once per iteration
    # NB0 : With this code, if the number of transaction nbPerIteration is too close to the number of agent,
    #       the resulting distribution will be affected, so you need to keep it low compared to N.
    # NB1 : You could chose to make a giver selected twice or more per iteration
    #       It wouldn't change the result.

    # Amount of money to give
    if giveOption == 'random':
        # Select agent1 (giver)
        giversMoney = wealth[agent1]
        indexeGivers = np.where(giversMoney > threshold); indexeGivers=np.array(indexeGivers[0], dtype=int) # This is a bit slow to compute, it could be better optimized
        givingAgents = agent1[indexeGivers]
        # Randomly generate money amount
        money = [np.random.randint(0, propToMax*(giversMoney[idx]-threshold)) for idx in indexeGivers]
    elif giveOption == 'fix':
        givingAgents = agent1
        # Generate fixed money amount
        money = fixAmount*np.heaviside(wealth[agent1]-fixAmount-threshold, 1)
    
    # Select random agent2 (receiver)
    agent2 = sample(list(agentsIndex), len(givingAgents))
    # NB : I allow agent1 to give to himself, which basically means 
    #      there's no transaction from agent1 ... Good news is, the bigger N, the less probable.
    
    # Make transaction, balanced equation
    money = np.array(money, dtype=int)
    wealth[agent2] = wealth[agent2] + money
    wealth[givingAgents] = wealth[givingAgents] - money
    
    # Plot distribution over iterations
    if plotIterations and t % modulo == 0:
        print(f'Iteration: {t}')
        # Histogram of wealth
        nbBins, value = np.histogram(wealth, 100, density=False)
        
        # possible Configuration over iteration
        values, counts = np.unique(wealth, return_counts=True)
        W.append(mt.log( N_fac // multiply([ mt.factorial(counts[i]) for i in range(len(values)) ]) ))

        # PMF       
        P = nbBins/max(nbBins)
        plt.plot(value[:-1], P, label=f'Iteration :{t}')
        iteration.append(t)

    
if plotIterations:
    plt.legend()
    plt.title('Wealth distribution through iterations')
    plt.ylabel('P(m)')
    plt.xlabel('m [$]')


# Plot final distribution
plt.figure()
bins, value = np.histogram(wealth, 100)
P = bins/sum(bins)
value = value[:-1]
plt.plot(value, P, label='Wealth distribution')
plt.title('Final wealth distribution')
plt.ylabel('P(m)')
plt.xlabel('m [$]')
plt.savefig("./Final wealth distribution.png", bbox_inches='tight', format='png', dpi=1000)

# Fitting curve
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(value, P, label='Wealth distribution')
plt.title('Final wealth distribution')
plt.ylabel('P(m)')
plt.xlabel('m [$]')
popt, _ = curve_fit(expFit, value, P/max(P))
plt.plot(value, max(P)*expFit(value, popt[0]), '--', color='red', label='Exponential fitting')
plt.legend()
# Fitting curve LOG
plt.subplot(1, 2, 2)
plt.semilogy(value, P, label='Wealth distribution')
plt.title('Semi log scale')
plt.ylabel('P(m)')
plt.xlabel('m [$]')
popt, _ = curve_fit(expFit, value, P/max(P))
plt.plot(value, max(P)*expFit(value, popt[0]), '--', color='red', label='Exponential fitting')
print('Estimate of the average money', popt[0])
plt.savefig("./Fitting expoentnial.png", bbox_inches='tight', format='png', dpi=1000)
plt.show()

# Cumulative distribution function
avg = popt[0]
P = np.array(P)
plt.figure()
cdf_data = np.array([sum(P[:i]) for i in range(len(P))])
cdf_exp = 1-np.exp(-(1/avg)*value)  # For more info, see : https://statproofbook.github.io/P/exp-cdf.html
plt.plot(value, cdf_data, label='data')
plt.plot(value, cdf_exp, '--',  color='red', label='exponential')
plt.title('Cumulative distribution function')
plt.ylabel(r'P(M$\leq$m)')
plt.xlabel('m [$]')
plt.legend()
plt.savefig("./Cumulative distribution function.png", bbox_inches='tight', format='png', dpi=1000)
plt.show()

plt.plot(cdf_exp[::-1])

# Number of possible arrangement over iterations
if plotIterations:
    plt.figure()
    plt.plot(iteration, W)
    plt.xlabel('Iteration')
    plt.ylabel('$log_{10}(W)$')
    plt.title('The number of possible arrangements of agents over iterations')
    plt.show()

# 80 / 20 division
idx80 = np.argmin(abs(cdf_data-0.8))
value80 = value[idx80]
eighty = sum(wealth[wealth <=value80])
twenty = sum(wealth[wealth > value80])
Total = eighty+twenty
P_division = cdf_data[idx80]
text = f'There is {np.round(100*P_division, 1)}% of people detaining {np.round(100*eighty/Total, 1)}% of the total wealth, \n'
text += f'and {np.round(100*(1-P_division), 1)}% detaining the {np.round(100*twenty/Total, 1)}% remaining.'
print(text)

time_elapsed = (time.perf_counter() - time_start)
print('Time elapsed', time_elapsed)

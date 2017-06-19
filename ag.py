import numpy as np;
from numpy.random import random
import math

# given grid with initial positions and grid with permutations, guarantee that
# permutation will have the given number in the right positions
def normalizeGrid(grid1, grid2):
    new_grid = grid2.copy()
    for i, num in enumerate(grid1):
        if num != 0:
            pos = np.argwhere(grid2==num)[0,0]
            new_grid[i], new_grid[pos] = new_grid[pos], new_grid[i]
    return new_grid

#assume only perfect square boards
def generateInitialPopulation(initial_grid, bsize, popsize):
    pop = []
    for i in range(popsize):
        grid = []
        for j in range(bsize):
            index = j*bsize
            i_sub = initial_grid[index:index+bsize]
            subgrid = np.random.permutation(np.arange(1,bsize+1))
            grid.append(normalizeGrid(i_sub, subgrid))
        pop.append(np.concatenate(grid, axis=0))
    return np.asarray(pop)

#count how many duplicated values there are on each row and column
def fitnessFunction(sol_string):
    f_val = 0
    # printBoard(sol_string)
    reshaped_sol = sol_string.reshape(3,3,3,3)
    #rows
    for i in range(3):
        subgrids = reshaped_sol[i,:]
        for j in range(3):
            u_row, c_row = np.unique(subgrids[:,j,:], return_counts=True)
            for k,v in dict(zip(u_row, c_row)).items():
                if v > 1:
                    f_val += v-1
    #cols
    for i in range(3):
        subgrids = reshaped_sol[:,i]
        for j in range(3):
            u_col, c_col = np.unique(subgrids[:,:,j], return_counts=True)
            for k,v in dict(zip(u_col, c_col)).items():
                if v > 1:
                    f_val += v-1
    return f_val

def printBoard(sol_string):
    a = sol_string.reshape(9,9)
    for i in range(0,7,3):
        print(a[i,:3], a[i+1,:3], a[i+2,:3])
        print(a[i,3:6], a[i+1,3:6], a[i+2,3:6])
        print(a[i,6:], a[i+1,6:], a[i+2,6:])

#select a portion of initial population based on fitness function
def matingPool(population, ranks):
    quantity = len(population)
    weights = np.asarray([np.sum(ranks)] * ranks.size) - ranks
    weights = weights.astype('float32') / np.sum(weights)
    indexes = np.random.choice(np.arange(len(population)), size=quantity, p=weights)
    return population[indexes]

#check if crossover happens, then calculate crossover point
def crossover(sol_string1, sol_string2, crossover_prob):
    if np.random.random() < crossover_prob:
        cross_point = np.random.randint(9) * 9
        new_sol_string1 = np.concatenate((sol_string1[:cross_point], sol_string2[cross_point:]))
        new_sol_string2 = np.concatenate((sol_string2[:cross_point], sol_string1[cross_point:]))
        return new_sol_string1, new_sol_string2
    return sol_string1, sol_string2

def mutation(initial_grid, sol_string, mutation_prob):
    m_sol_string = sol_string.copy()
    i = np.random.randint(9)
    if np.random.random() < mutation_prob:
        s1,s2 = tuple(np.random.randint(9, size=2))
        while initial_grid[s1*i] != 0 or initial_grid[s2*i] != 0:
            s1,s2 = tuple(np.random.randint(9, size=2))
        m_sol_string[s1*i], m_sol_string[s2*i] = m_sol_string[s2*i], m_sol_string[s1*i]
    return m_sol_string

def rankPopulation(population):
    fitness_table = []
    for sol in population:
        fitness_table.append(fitnessFunction(sol))
    return np.asarray(fitness_table)


def main():
    NGENS = 100
    POPSIZE = 1000
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.85

    initial_grid = np.array([0,0,0,1,0,0,0,0,0,
                             7,0,0,0,0,0,4,3,0,
                             0,0,0,0,0,0,2,0,0,
                             0,0,0,0,0,0,0,0,0,
                             0,0,0,5,0,9,0,0,0,
                             0,0,6,0,0,0,4,1,8,
                             0,0,0,0,0,2,0,4,0,
                             0,8,1,0,0,0,0,0,0,
                             0,0,0,0,5,0,3,0,0])

    population = generateInitialPopulation(initial_grid, 9, POPSIZE)
    best_population = None
    local_minima = 0
    print('Initial configuration:')
    printBoard(initial_grid)
    previous_score = np.inf
    for gen in range(NGENS):
        #rank population by fitness function
        fitness_table = rankPopulation(population)
        best_population = population.copy()
        best_result, worst_result = np.min(fitness_table), np.max(fitness_table)
        if (previous_score - best_result) == 0:
            local_minima += 1
        if best_result == 0 or local_minima >= 10:
            print("Generation: %d: best score: %d, worst score: %d" % (gen, best_result, worst_result))
            print('\n\nBest Solution:')
            printBoard(population[p.argmin(fitness_table)])
            break
        print("Generation: %d: best score: %d, worst score: %d" % (gen, best_result, worst_result))

        #apply mutation and crossover to generate next gen population
        pairs = np.random.permutation(np.arange(POPSIZE)).reshape(np.int(POPSIZE/2), 2)
        new_pop = population.copy()
        for pair in pairs:
            p1,p2 = tuple(pair)
            new_pop[p1], new_pop[p2] = crossover(new_pop[p1], new_pop[p2], MUTATION_RATE)
        for i, sol in enumerate(new_pop):
            new_pop[i] = mutation(initial_grid, sol, MUTATION_RATE)
        population = matingPool(new_pop.copy(), np.array([fitnessFunction(sol) for sol in new_pop]))

    print('\n\nBest Solution:')
    printBoard(best_population[np.argmin(fitness_table)])


if __name__ == '__main__':
    main()

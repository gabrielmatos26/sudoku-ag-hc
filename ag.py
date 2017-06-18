import numpy as np; np.random.seed(1234)
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

def main():
    initial_grid = np.array([0,0,0,1,0,0,0,0,0,
                             7,0,0,0,0,0,4,3,0,
                             0,0,0,0,0,0,2,0,0,
                             0,0,0,0,0,0,0,0,0,
                             0,0,0,5,0,9,0,0,0,
                             0,0,6,0,0,0,4,1,8,
                             0,0,0,0,0,2,0,4,0,
                             0,8,1,0,0,0,0,0,0,
                             0,0,0,0,5,0,3,0,0])

    initial_pop = generateInitialPopulation(initial_grid, 9, 30)
    fitness_table = []
    for sol in initial_pop:
        fitness_table.append(fitnessFunction(sol))
    print(fitness_table)

if __name__ == '__main__':
    main()

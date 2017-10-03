from geneticalgs import BinaryGA

# define data whose best combination we are searching for
input_data = [1, 2, 3, 7, -1, -20]


# define a simple fitness function
def fitness_function(chromosome, data):
    # this function searches for the greatest sum of numbers in data
    # chromosome contains positions (from left 0 to right *len(data)-1) of bits 1
    sum = 0
    for bit in chromosome:
        sum += data[bit]

    return sum


# initialize standard binary GA
gen_alg = BinaryGA(input_data, fitness_function)
# initialize random population of size 6
gen_alg.init_random_population(6)
gen_alg.run(10)
print(gen_alg.best_solution)


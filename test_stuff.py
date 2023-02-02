import numpy as np
from brian2 import *
from tools import *
import itertools as iter

nb_inputs = 20
data = np.loadtxt('spiketrains_0_input_pattern')
indices = data[:, 1].astype(int)
times = data[:, 0] * second

nb_students = 10
""" Connecting all correctors to their students with massive overlap. 
The number of correctors is based on the number of students and the optimal number of combinations."""
students_indexes = list(range(nb_students))
all_list_of_combinations = []
nb_combinations = []

for size in range(11):
    combi = list(iter.combinations(students_indexes, size))
    all_list_of_combinations.append(combi)
    nb_combinations.append(len(combi))

combinations = all_list_of_combinations[np.argmax(nb_combinations)]
nb_correctors = len(combinations)
all_list_of_combinations = []
print(nb_correctors)

""" Connecting all students to their correctors with no overlap (as with climbing fibers in cerebellar cortex). 
A student will connect with a small subset of its correctors"""
connect_mat_correct_stud = np.zeros((nb_correctors, nb_students))

for cb in range(len(combinations)):
    for neuron in combinations[cb]:
        connect_mat_correct_stud[cb, neuron] = 1

connect_mat_stud_correct = np.zeros((nb_students, nb_correctors))
remaining_correctors_indexes = set(range(nb_correctors))
possibilities = set()

for std in range(nb_students-1):
    correct_connected = np.where(np.array(connect_mat_correct_stud[:, std]) == 1)[0]
    possibilities = remaining_correctors_indexes.intersection(correct_connected)
    chosen = np.random.choice(np.array(list(possibilities)), int(nb_correctors//nb_students), replace=False)
    connect_mat_stud_correct[std, chosen] = 1
    remaining_correctors_indexes = remaining_correctors_indexes-set(chosen)

# adding all remaining non connected correctors to the last student
connect_mat_stud_correct[nb_students-1, np.array(list(remaining_correctors_indexes))] = 1





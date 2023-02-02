import numpy as np
from brian2 import *
from tools import *
import itertools as iter

nb_inputs = 20
data = np.loadtxt('spiketrains_0_input_competitive')
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
print("nb_correctors = "+str(nb_correctors))

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


""" Defining parameters and equations for neurons groups"""
tau = 10 * ms
tau_window = 100*ms
beta = 1/300
eqs = '''
dv/dt = -v/tau : 1
learning_gate : 1
'''
eqs_to_inhibit = '''
dv/dt = -v/tau : 1
'''

start_scope()

""" Defining neuron groups"""
input = SpikeGeneratorGroup(nb_inputs, indices, times)
correctors = NeuronGroup(nb_correctors, eqs, threshold='v>1', reset='v = 0', method='exact',
                         events={'open_gate': 'learning_gate > 0.5'})
correctors.run_on_event('open_gate', 'learning_gate=0')
correctors.learning_gate = 0
students = NeuronGroup(10, eqs_to_inhibit, threshold='v>1')

""" Defining parameters and equations for synapses"""
model_syn = '''
w :1
dwindow/dt = -window/tau_window :1 (event-driven)
'''

on_pre_syn = '''
v_post+=w
window+=1 
w-= beta
'''

when_learning = """
w+= beta*int(window>0.05)+window*beta
"""

syn_input_correctors = Synapses(input, correctors, model=model_syn, on_pre={"pre": on_pre_syn},
                            on_post={"induce_learning": when_learning},
                            on_event={"pre": 'spike', "induce_learning": "open_gate"})
syn_input_correctors.connect()
syn_input_correctors.w = 0.3

syn_input_students = Synapses(input, students, on_pre="v+=0.8")
syn_input_students.connect(p=1)  # each student receive a small mix of inputs

syn_correctors_students = Synapses(correctors, students, on_pre="v-=0.03")
j_connect = np.array([])
i_connect = np.array([])
for i in range(len(connect_mat_correct_stud)):
    connected = np.where(connect_mat_correct_stud[i] == 1)[0]
    j_connect = np.append(j_connect, connected)
    i_connect = np.append(i_connect, np.full(len(connected), i))

print(list(i_connect))
print(list(j_connect))
#syn_correctors_students.connect()

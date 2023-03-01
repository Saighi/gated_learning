import itertools as iter
from brian2 import *
from tools import *

nb_inputs = 20
data = np.loadtxt('spiketrains_0_input_competitive_pattern')
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
print("nb_correctors = " + str(nb_correctors))

""" Connecting all students to their correctors with no overlap (as with climbing fibers in cerebellar cortex). 
A student will connect with a small subset of its correctors"""
connect_mat_correct_stud = np.zeros((nb_correctors, nb_students))

for cb in range(len(combinations)):
    for neuron in combinations[cb]:
        connect_mat_correct_stud[cb, neuron] = 1

connect_mat_stud_correct = np.zeros((nb_students, nb_correctors))
remaining_correctors_indexes = set(range(nb_correctors))
possibilities = set()

for std in range(nb_students - 1):
    correct_connected = np.where(np.array(connect_mat_correct_stud[:, std]) == 1)[0]
    possibilities = remaining_correctors_indexes.intersection(correct_connected)
    chosen = np.random.choice(np.array(list(possibilities)), int(nb_correctors // nb_students), replace=False)
    connect_mat_stud_correct[std, chosen] = 1
    remaining_correctors_indexes = remaining_correctors_indexes - set(chosen)

# adding all remaining non connected correctors to the last student
connect_mat_stud_correct[nb_students - 1, np.array(list(remaining_correctors_indexes))] = 1

""" Defining parameters and equations for neurons groups"""
tau = 10 * ms
tau_window = 100 * ms
beta = 1 / 300
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
students = NeuronGroup(nb_students, eqs_to_inhibit, threshold='v>1', reset='v = 0', method='exact')

""" Defining parameters and equations for synapses"""
model_syn = '''
w :1
dwindow/dt = -window/tau_window :1 (event-driven)
'''
# WINDOWS =1 !!!!

on_pre_syn = '''
v_post+=w
window+=1
w = clip(w-beta,0,1000)
'''

when_learning = """
w+= beta*int(window>0.05)+(window*beta)
"""

syn_input_correctors = Synapses(input, correctors, model=model_syn, on_pre={"pre": on_pre_syn},
                                on_post={"induce_learning": when_learning},
                                on_event={"pre": 'spike', "induce_learning": "open_gate"})

syn_input_correctors.connect()
syn_input_correctors.w = 0.4
#syn_input_correctors.w = 0

syn_input_students = Synapses(input, students, on_pre="v+=1.1")
j_connect = np.array([], dtype=int)
i_connect = np.array([], dtype=int)

nb_of_connexions = 8
for s in range(len(students)):
    i_connect = np.append(i_connect, np.random.choice(list(range(nb_inputs)), nb_of_connexions, replace=False))
    j_connect = np.append(j_connect, np.full(nb_of_connexions, s))

syn_input_students.connect(i=i_connect, j=j_connect)  # each student receive a small mix of inputs

syn_correctors_students = Synapses(correctors, students, on_pre="v-=0.03")
j_connect = np.array([], dtype=int)
i_connect = np.array([], dtype=int)

for i in range(len(connect_mat_correct_stud)):
    connected = np.where(connect_mat_correct_stud[i] == 1)[0]
    j_connect = np.append(j_connect, connected)
    i_connect = np.append(i_connect, np.full(len(connected), i))

syn_correctors_students.connect(i=i_connect, j=j_connect)

syn_students_correctors = Synapses(students, correctors, on_pre="learning_gate=1")
j_connect = np.array([], dtype=int)
i_connect = np.array([], dtype=int)

for i in range(len(connect_mat_stud_correct)):
    connected = np.where(connect_mat_stud_correct[i] == 1)[0]
    j_connect = np.append(j_connect, connected)
    i_connect = np.append(i_connect, np.full(len(connected), i))

syn_students_correctors.connect(i=i_connect, j=j_connect)

correctors_mon = SpikeMonitor(correctors)
students_mon = SpikeMonitor(students)
all_synapses = list(range(0, nb_inputs*nb_correctors))
to_record = np.random.choice(all_synapses, 100)
learning_synapes_mon = StateMonitor(syn_input_correctors, 'w', record=to_record, dt=0.1*second)

run(200000 * ms)
"""
plot(correctors_mon.t / ms, correctors_mon.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index')

plt.show()

plot(students_mon.t / ms, students_mon.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index')

plt.show()

plot(learning_synapes_mon.t / ms, (learning_synapes_mon.w).T)
xlabel('Time (ms)')
ylabel('w')

plt.show()
"""

spikes_correctors = correctors_mon.get_states(['t', 'i'], units=False, format='pandas')
spikes_correctors.to_csv("competitive_data/correctors_spikes.csv", index=False)

spikes_students = students_mon.get_states(['t', 'i'], units=False, format='pandas')
spikes_students.to_csv("competitive_data/students_spikes.csv", index=False)
np.savetxt("competitive_data/syn_w_time.csv", learning_synapes_mon.t)
for syn in range(len(learning_synapes_mon.w)):
    formated_string = "competitive_data/syn_w_"+str(syn)+".csv"
    np.savetxt(formated_string, learning_synapes_mon.w[syn])

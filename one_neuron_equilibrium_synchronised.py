import random as rnd
import numpy as np
from brian2 import *
from tools import *

nb_stud_connect = 60
nb_students = 50
nbtot_connections = nb_students * nb_stud_connect
inhib_post_val = 0.1/nb_students #Prev 0.3
data = np.loadtxt('spiketrains_0_input_pattern')
indices = data[:, 1].astype(int)
times = data[:, 0] * second

signal_to_learn = 0

where_inputs = np.where(indices != signal_to_learn)[0]
indices_inputs = indices[where_inputs]
times_inputs = times[where_inputs]
for i in range(len(indices_inputs)):
    if indices_inputs[i] > signal_to_learn:
        indices_inputs[i] -= 1

where_to_learn = np.where(indices == signal_to_learn)[0]
indices_to_learn = indices[where_to_learn]
times_to_learn = times[where_to_learn]

tau = 10 * ms
tau_teacher = 10 * ms
tau_inhib = 100 * ms
tau_window = 25 * ms
tau_detector = 10*ms
vrest_t = 1.1
eqs = '''
dv/dt = -v/tau : 1
learning_gate : 1
'''
eqs_teacher = '''
dv/dt = ((vrest_t-v)/tau_teacher) + (current/tau_teacher) - (inhib/tau_inhib): 1
dcurrent/dt = -current/tau_teacher : 1
dinhib/dt = -inhib/tau_inhib : 1
'''
eqs_detector = '''
dv/dt = -v/tau_detector : 1
'''

start_scope()

"""NEURONS"""

network_input = SpikeGeneratorGroup(nbtot_connections, indices_inputs, times_inputs)
teacher = NeuronGroup(1, eqs_teacher, threshold='v>1', reset='v=0', method='exact')
student = NeuronGroup(nb_students, eqs, threshold='v>1', reset='v = 0', method='exact',
                      events={'open_gate': 'learning_gate > 0.5'})
student.run_on_event('open_gate', 'learning_gate=0')
student.learning_gate = 0
to_learn_stim = SpikeGeneratorGroup(1, indices_to_learn, times_to_learn)

sync_detector = NeuronGroup(1, eqs_detector, threshold='v>1000', reset='v=0', method='exact')
#sync_inhibitor = NeuronGroup(1, eqs_detector, threshold='v>0.0', reset=0, method='exact')

"""SYNAPSES"""

syn_students_sync = Synapses(student, sync_detector, on_pre='v+=0.01')
syn_students_sync.connect()
syn_stim_teacher = Synapses(to_learn_stim, teacher, on_pre="current+=0.5") #PREV was 0.5
syn_stim_teacher.connect()
syn_teacher_student = Synapses(teacher, student, on_pre="learning_gate=1")
syn_teacher_student.connect()
syn_student_teacher = Synapses(student, teacher, on_pre="inhib_post+=inhib_post_val")
syn_student_teacher.connect()

model_syn = '''
w :1
dwindow/dt = -window/tau_window :1 (event-driven)
dwindow_dep/dt = -window_dep/tau_window : 1 (event-driven)
beta : 1
'''

on_pre_syn = '''
v_post+=w
window+=1 
w-= beta
'''

when_learning = """
w+= window*(beta)
"""

syn_input_student = Synapses(network_input, student, model=model_syn, on_pre={"pre": on_pre_syn},
                             on_post={"induce_learning": when_learning},
                             on_event={"pre": 'spike', "induce_learning": "open_gate"})
# Each student is connected to X inputs, then the number of connections is nb_student*X
i_connects = np.random.choice(np.arange(0, nbtot_connections), nbtot_connections, replace=False)
j_connects = np.repeat(np.arange(0, nb_students), nb_stud_connect)
syn_input_student.connect(i=i_connects, j=j_connects)
syn_input_student.w = 0.3

#The learning rate spread over a wide distribution
learning_rates = np.zeros(nbtot_connections)
for i in range(len(learning_rates)):
    learning_rates[i] = 1 / (rnd.randint(5, 100))
syn_input_student.beta = learning_rates
# syn_input_student.beta = 1/1000

"""RECORDING"""
to_record = list(range(0, nbtot_connections))
syn_mon = StateMonitor(syn_input_student, "w", to_record, dt=1*second)
teacher_mon = SpikeMonitor(teacher)
to_learn_mon = SpikeMonitor(to_learn_stim)
student_mon = SpikeMonitor(student)
sync_mon = StateMonitor(sync_detector, "v", record=[0])
sync_spike_mon = SpikeMonitor(sync_detector)

run(200000 * ms, report='text', report_period=5*second)

"""Saving data :"""
for i in range(nbtot_connections):
    np.savetxt("data/syn_w_input_student_"+str(i)+".data", syn_mon.w[i])
np.savetxt("data/syn_w_input_student_times.data", syn_mon.t)

np.savetxt("data/spikes_teachers_index.data", teacher_mon.i)
np.savetxt("data/spikes_teachers_times.data", teacher_mon.t)
np.savetxt("data/spikes_student_index.data", student_mon.i)
np.savetxt("data/spikes_student_times.data", student_mon.t)
np.savetxt('data/spikes_to_learn_index.data', to_learn_mon.i)
np.savetxt('data/spikes_to_learn_times.data', to_learn_mon.t)
np.savetxt('data/sync_detector_voltage.data', sync_mon.v[0])
np.savetxt('data/sync_detector_spikes.data', sync_spike_mon.t)
"""
plot(M.t / ms, M.v[0])
xlabel('Time (ms)')
ylabel('v')

plt.show()

plot(spikemon.t / ms, spikemon.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index')

plt.show()

plot(syn_mon.t / ms, syn_mon.w.T)
xlabel('Time (ms)')
ylabel('w')

plt.show()

signal_times = np.loadtxt("pattern_times").T[0]
times_from_pat = SpikesDistFromPat(np.array(student_mon.t), 0.25, signal_times, window=0.5, offset=0.125)

plt.scatter(times_from_pat[0], times_from_pat[1])
plt.show()

# data = [teacher_mon.t, to_learn_mon.t, student_mon.t, signal_times]
data = [teacher_mon.t, to_learn_mon.t, student_mon.t]
plt.eventplot(data)
plt.show()

all_spike_pat = []
for i in range(len(times_from_pat[0])):
    if times_from_pat[0][i] > 70:
        all_spike_pat.append(times_from_pat[1][i])

all_spike_pat.sort()

plt.hist(all_spike_pat, bins=30)
plt.show()
"""

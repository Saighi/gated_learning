from brian2 import *
from tools import *
import random

nb_inputs = 21
data = np.loadtxt('spiketrains_0_input_pattern_solo')
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
tau = 10*ms
tau_teacher = 10*ms
tau_inhib = 100*ms
tau_window = 25*ms
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

start_scope()

network_input = SpikeGeneratorGroup(nb_inputs, indices_inputs, times_inputs)
teacher = NeuronGroup(1, eqs_teacher, threshold='v>1', reset='v=0', method='exact')
student = NeuronGroup(1, eqs, threshold='v>1', reset='v = 0', method='exact',
                     events={'open_gate': 'learning_gate > 0.5'})
student.run_on_event('open_gate', 'learning_gate=0')
student.learning_gate = 0
to_learn_stim = SpikeGeneratorGroup(1, indices_to_learn, times_to_learn)
syn_stim_teacher = Synapses(to_learn_stim, teacher, on_pre="current+=0.5")
syn_stim_teacher.connect()
syn_teacher_student = Synapses(teacher, student, on_pre="learning_gate=1")
syn_teacher_student.connect()
syn_student_teacher = Synapses(student, teacher, on_pre="inhib_post += 0.3")
syn_student_teacher.connect()
model_syn = '''
w :1
dwindow/dt = -window/tau_window :1 (event-driven)
dwindow_dep/dt = -window_dep/tau_window : 1 (event-driven)
beta : 1
'''
#on_pre_syn = '''
#v_post+=w
#window+=1
#w-= window_dep*beta
#'''

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
syn_input_student.connect()
syn_input_student.w = 0.3
learning_rates = np.zeros(nb_inputs)
for i in range(len(learning_rates)):
    learning_rates[i] = 1/(random.randint(50, 1000))
syn_input_student.beta = learning_rates
#syn_input_student.beta = 1/1000

to_record = list(range(0, nb_inputs))
syn_mon = StateMonitor(syn_input_student, "w", to_record)
teacher_mon = SpikeMonitor(teacher)
to_learn_mon = SpikeMonitor(to_learn_stim)
student_mon = SpikeMonitor(student)

run(100000*ms)
"""
plot(M.t / ms, M.v[0])
xlabel('Time (ms)')
ylabel('v')

plt.show()

plot(spikemon.t / ms, spikemon.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index')

plt.show()
"""
plot(syn_mon.t / ms, syn_mon.w.T)
xlabel('Time (ms)')
ylabel('w')

plt.show()

signal_times = np.loadtxt("pattern_times").T[0]
times_from_pat = SpikesDistFromPat(np.array(student_mon.t), 0.25, signal_times, window=0.5, offset=0.125)

plt.scatter(times_from_pat[0], times_from_pat[1])
plt.show()

#data = [teacher_mon.t, to_learn_mon.t, student_mon.t, signal_times]
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
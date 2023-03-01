from brian2 import *
from tools import *


nb_inputs = 80
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
tau_window = 100*ms
beta = 1/300
eqs = '''
dv/dt = -v/tau : 1
learning_gate : 1
'''

start_scope()

input = SpikeGeneratorGroup(nb_inputs, indices_inputs, times_inputs)
output = NeuronGroup(1, eqs, threshold='v>1', reset='v = 0', method='exact',
                     events={'open_gate': 'learning_gate > 0.5'})
output.run_on_event('open_gate', 'learning_gate=0')
output.learning_gate = 0
to_learn_group = SpikeGeneratorGroup(1, indices_to_learn, times_to_learn)
syn_to_learn_outputs = Synapses(to_learn_group, output, on_pre="learning_gate=1")
syn_to_learn_outputs.connect()

model_syn = '''
w :1
dwindow/dt = -window/tau_window :1 (event-driven)
dwindow_dep/dt = -window_dep/tau_window : 1 (event-driven)
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
w+= (window*beta)
window_dep+=1
"""

#syn_input_output = Synapses(input, output, model=model_syn, on_pre={"pre": on_pre_syn},
#                            on_post={"induce_learning": when_learning, "post": "w = clip(w-(beta/4),0,1000)"},
#                            on_event={"pre": 'spike', "induce_learning": "open_gate"})
syn_input_output = Synapses(input, output, model=model_syn, on_pre={"pre": on_pre_syn},
                            on_post={"induce_learning": when_learning},
                            on_event={"pre": 'spike', "induce_learning": "open_gate"})
syn_input_output.connect()
syn_input_output.w = 0.3

#M = StateMonitor(output, 'v', record=0)
to_record = list(range(0, nb_inputs))
#S = StateMonitor(syn_input_output, 'w', record=to_record)
spikemon = SpikeMonitor(output)

run(250000 * ms)
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
#plot(S.t / ms, (S.w).T)
#xlabel('Time (ms)')
#ylabel('w')

plt.show()

signal_times = np.loadtxt("pattern_times").T[0]
times_from_pat = SpikesDistFromPat(np.array(spikemon.t), 0.15, signal_times, window=0.5, offset=0.075)

plt.scatter(times_from_pat[0], times_from_pat[1])
plt.show()

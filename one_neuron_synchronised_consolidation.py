from brian2 import *

from tools import *

nb_stud_connect = 50
beta_student = 1 / 50
nb_students = 50
nbtot_connections = nb_students * nb_stud_connect
inhib_post_val = 0.2  # Prev 0.3
inhib_post_val_conso = 0.1 / nb_students
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
tau_current = 20 * ms
tau_conso = 10 * ms
tau_inhib = 100 * ms
tau_inhib_conso = 10 * ms
tau_window = 25 * ms
tau_window_conso = 10 * ms
tau_detector = 10 * ms
vrest_t = 1.5
vrest_student = -0.8
vrest_conso = 1.1
excitatory_current = 1
eqs = '''
dv/dt = (vrest_student-v)/tau : 1
learning_gate : 1
'''
eqs_teacher = '''
dv/dt = ((vrest_t-v + current - inhib)/tau_teacher) : 1
dcurrent/dt = -current/tau_teacher : 1
dinhib/dt = -inhib/tau_inhib : 1
'''
eqs_detector = '''
dv/dt = (2-v)/tau_detector : 1
'''
eqs_conso = '''
dv/dt = ((vrest_conso-v-inhib_conso)/tau_conso) : 1
dinhib_conso/dt = -inhib_conso/tau_inhib_conso : 1
learning_gate : 1
learning_gate_2 : 1
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

consolidation_unit = NeuronGroup(1, eqs_conso, threshold='v>1', reset='v=0', method='euler',
                                 events={'open_gate': 'learning_gate > 0.5', 'open_gate_2': 'learning_gate_2>0.5'})
consolidation_unit.run_on_event('open_gate', 'learning_gate=0')
consolidation_unit.run_on_event('open_gate_2', 'learning_gate_2=0')
sync_detector = NeuronGroup(1, eqs_detector, threshold='v>1.5', reset='v=0', method='exact')
sync_detector_2 = NeuronGroup(1, eqs_detector, threshold='v<1.1', reset='v=2', method='exact')
# sync_inhibitor = NeuronGroup(1, eqs_detector, threshold='v>0.0', reset=0, method='exact')

"""SYNAPSES"""

syn_students_sync = Synapses(student, sync_detector, on_pre='v-=0.01')
syn_students_sync.connect()
syn_students_sync_2 = Synapses(student, sync_detector_2, on_pre='v-=0.01')
syn_students_sync_2.connect()
syn_stim_teacher = Synapses(to_learn_stim, teacher, on_pre="current+=excitatory_current")  # PREV was 0.5
syn_stim_teacher.connect()
syn_teacher_student = Synapses(teacher, student, on_pre="learning_gate=1")
syn_teacher_student.connect()

model_syn = '''
w :1
dwindow/dt = -window/tau_window :1 (event-driven)
dwindow_dep/dt = -window_dep/tau_window : 1 (event-driven)
beta : 1
scale : 1
'''

on_pre_syn = '''
v_post+=(w*scale)
window+=1 
w += beta
'''

when_learning = """
w = clip(w-window*(beta),0,100)
"""

syn_input_student = Synapses(network_input, student, model=model_syn, on_pre={"pre": on_pre_syn},
                             on_post={"induce_learning": when_learning},
                             on_event={"pre": 'spike', "induce_learning": "open_gate"})
# Each student is connected to X inputs, then the number of connections is nb_student*X
i_connects = np.random.choice(np.arange(0, nbtot_connections), nbtot_connections, replace=False)
j_connects = np.repeat(np.arange(0, nb_students), nb_stud_connect)
syn_input_student.connect(i=i_connects, j=j_connects)
syn_input_student.w = 0.2
# scale_synapse = expon.rvs(scale=1, size=nbtot_connections)
# syn_input_student.scale = scale_synapse
syn_input_student.scale = 1

# The learning rate spread over a wide distribution
# learning_rates = np.zeros(nbtot_connections)
# for i in range(len(learning_rates)):
#     learning_rates[i] = 1 / (rnd.randint(40, 400))
# syn_input_student.beta = learning_rates
syn_input_student.beta = beta_student

model_syn_conso = '''
w :1
beta :1
dwindow/dt = -window/tau_window_conso :1 (event-driven)
plasticity_on : 1
'''

on_pre_conso = '''
window+=1
v+=w
'''

syn_sync_conso = Synapses(sync_detector, consolidation_unit, on_pre="learning_gate+=1")
syn_sync_conso.connect()
syn_sync_2_conso = Synapses(sync_detector_2, consolidation_unit, on_pre="learning_gate_2+=1")
syn_sync_2_conso.connect()

when_learning_conso = '''
w = w+window*(beta)*plasticity_on
'''
when_learning_conso_2 = '''
w = clip(w-window*(beta)*plasticity_on,0,100)
'''

nb_connection_input_conso = 2000
syn_input_conso = Synapses(network_input, consolidation_unit, model=model_syn_conso, on_pre={"pre": on_pre_conso},
                           on_post={"induce_learning": when_learning_conso, "induce_learning_2": when_learning_conso_2},
                           on_event={"pre": 'spike', "induce_learning": "open_gate",
                                     "induce_learning_2": "open_gate_2"})
i_connects_conso = np.random.choice(np.arange(0, nbtot_connections), nb_connection_input_conso, replace=False)
j_connects_conso = np.full(nb_connection_input_conso, 0)
j_connects_conso = np.repeat(np.arange(0, ), nb_connection_input_conso)
syn_input_conso.connect(i=i_connects_conso, j=j_connects_conso)
syn_input_conso.w = (0.025 / 6.5) / 10
syn_input_conso.beta = 1 / 5000
syn_input_conso.plasticity_on = 0

nb_connection_input_teacher = 30
syn_input_teacher = Synapses(network_input, teacher, model=model_syn_conso, on_pre="v+= w")
i_connects = np.random.choice(np.arange(0, nbtot_connections), nb_connection_input_teacher, replace=False)
j_connects = np.full(nb_connection_input_teacher, 0)
syn_input_teacher.connect(i=i_connects, j=j_connects)
syn_input_teacher.w = 0.025

syn_students_conso = Synapses(student, consolidation_unit, on_pre="inhib_conso_post+=inhib_post_val_conso")
syn_students_conso.connect()

syn_conso_teacher = Synapses(consolidation_unit, teacher, on_pre="inhib_post+=inhib_post_val")
syn_conso_teacher.connect()

"""RECORDING"""
to_record = list(range(0, nbtot_connections))
syn_mon = StateMonitor(syn_input_student, "w", to_record, dt=1 * second)
syn_mon_conso = StateMonitor(syn_input_conso, "w", list(range(nb_connection_input_conso)), dt=1 * second)
teacher_mon = SpikeMonitor(teacher)
to_learn_mon = SpikeMonitor(to_learn_stim)
student_mon = SpikeMonitor(student)
sync_mon = StateMonitor(sync_detector, "v", record=[0])
sync_spike_mon = SpikeMonitor(sync_detector)
sync_mon_2 = StateMonitor(sync_detector_2, "v", record=[0])
sync_spike_mon_2 = SpikeMonitor(sync_detector_2)
conso_spike_mon = SpikeMonitor(consolidation_unit)

run(50000 * ms, report='text', report_period=10 * second)
store('initialized')
restore('initialized')
syn_input_conso.plasticity_on = 1
run(150000 * ms, report='text', report_period=10 * second)

"""Saving data :"""
for i in range(100):
    np.savetxt("data_conso/syn_w_input_student_" + str(i) + ".data", syn_mon.w[i])
np.savetxt("data_conso/syn_w_input_student_times.data", syn_mon.t)
for i in range(nb_connection_input_conso):
    np.savetxt("data_conso/syn_w_input_conso_" + str(i) + ".data", syn_mon_conso.w[i])
np.savetxt("data_conso/syn_w_input_conso_times.data", syn_mon_conso.t)
np.savetxt("data_conso/syn_source_input_conso.data", i_connects_conso)

np.savetxt("data_conso/spikes_teachers_index.data", teacher_mon.i)
np.savetxt("data_conso/spikes_teachers_times.data", teacher_mon.t)
np.savetxt("data_conso/spikes_student_index.data", student_mon.i)
np.savetxt("data_conso/spikes_student_times.data", student_mon.t)
np.savetxt('data_conso/spikes_to_learn_index.data', to_learn_mon.i)
np.savetxt('data_conso/spikes_to_learn_times.data', to_learn_mon.t)
np.savetxt('data_conso/sync_detector_voltage.data', sync_mon.v[0])
np.savetxt('data_conso/sync_detector_spikes.data', sync_spike_mon.t)
np.savetxt('data_conso/sync_detector_voltage_2.data', sync_mon_2.v[0])
np.savetxt('data_conso/sync_detector_spikes_2.data', sync_spike_mon_2.t)
np.savetxt('data_conso/conso_spikes.data', conso_spike_mon.t)
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

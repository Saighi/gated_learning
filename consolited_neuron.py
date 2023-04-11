from brian2 import *
from tools import *
import numpy as np

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

weight_in_time_conso = []
for i in range(2000):
    weight_in_time_conso.append(np.loadtxt("data_conso/syn_w_input_conso_"+str(i)+".data"))
weight_in_time_conso= np.array(weight_in_time_conso)

source_input_conso = np.loadtxt("data_conso/syn_source_input_conso.data")
final_weights = weight_in_time_conso[:,-1]
tau_conso = 10*ms

eqs_conso = '''
dv/dt = -v/tau_conso : 1
'''

nb_stud_connect = 200
nb_students = 50
nbtot_connections = nb_students * nb_stud_connect
network_input = SpikeGeneratorGroup(nbtot_connections, indices_inputs, times_inputs)
consolidation_unit = NeuronGroup(1, eqs_conso, method='euler')

eqs_syn = '''
w : 1
'''
syn = Synapses(network_input,consolidation_unit,eqs_syn,on_pre="v_post+=w")
syn.connect(i=source_input_conso.astype(int), j=np.full(len(source_input_conso),0))
syn.w = final_weights
monitor = StateMonitor(consolidation_unit, "v", record=[0])

run(200000*ms,report='text',report_period=10*second)
print(monitor.v)
plt.plot(monitor.v[0])
plt.show()

from multiprocessing import Pool

import numba
import numpy as np
from neo.core import SpikeTrain
from quantities import s


def inPatternOld(my_ras, dure_simu, duree_pattern, delay, multWindow=1,
                 time_step=0.0001):
    batch = duree_pattern + delay
    tile = np.append(np.ones(int(duree_pattern // time_step)),
                     np.zeros(int(delay // time_step)))
    signal = np.tile(tile, int(dure_simu // batch) + 1)
    rs_ntime = []

    for t in my_ras:
        rs_ntime.append(int(t // time_step))

    sizeSimb = int(dure_simu // batch) // multWindow
    batchSim = np.zeros(sizeSimb + 1)
    batchCpt = np.zeros(sizeSimb + 1)
    nbatch = int(batch // time_step) * multWindow
    for nt in rs_ntime:
        batchCpt[int(nt // nbatch)] += 1
        if signal[nt] == 1:
            batchSim[int(nt // nbatch)] += 1

    return batchSim / np.where(batchCpt == 0, 1, batchCpt)


def inOrOutPattern(my_ras, dure_simu, duree_pattern, delay, multWindow=1,
                   time_step=0.0001):
    batch = duree_pattern + delay
    tile = np.append(np.ones(int(duree_pattern // time_step)),
                     np.zeros(int(delay // time_step)))
    signal = np.tile(tile, int(dure_simu // (batch)) + 1)
    rs_ntime = []

    for t in my_ras:
        rs_ntime.append(int(t // time_step))

    sizeSimb = int(dure_simu // batch) // multWindow
    batchSim = np.zeros(sizeSimb + 1)
    batchCpt = np.zeros(sizeSimb + 1)
    nbatch = int(batch // time_step) * multWindow
    for nt in rs_ntime:
        batchCpt[int(nt // nbatch)] += 1
        if signal[nt] == 1:
            batchSim[int(nt // nbatch)] += 1

    return np.max([batchSim / batchCpt, 1 - batchSim / batchCpt], axis=0)


def extractTrainsFiles(namefiles, nb_neurons, neo=False):
    if type(namefiles) == list:
        trainsStim = np.concatenate([np.loadtxt(name) for name in namefiles])
    else:
        trainsStim = np.loadtxt(namefiles)

    ListTrains = [[] for _ in range(nb_neurons)]
    for spike in range(len(trainsStim)):
        ListTrains[int(trainsStim[:, 1][spike])].append(
            trainsStim[:, 0][spike])
    if neo:
        ListNeoTrains = []
        for train in ListTrains:
            ListNeoTrains.append(SpikeTrain(train * s, t_stop=max(train)))
        return ListNeoTrains
    else:
        return ListTrains


def extractTrainFiles(spiketrain, nb_neurons):
    ListTrains = [[] for _ in range(nb_neurons)]
    for spike in range(len(spiketrain)):
        ListTrains[int(spiketrain[:, 1][spike])].append(
            spiketrain[:, 0][spike])

    return ListTrains


def TakeSampleTrain(namefiles, time):
    spikes = np.concatenate([np.loadtxt(name) for name in namefiles])
    spikes = spikes[spikes[:, 0].argsort()]
    return spikes[:np.argmax(spikes[:, 0] > time)]


def SpikesDistFromPatOld(spikeTrain, duree_pattern, delay):
    times = []
    dists = []
    for time in spikeTrain:
        times.append(time)
        dists.append(time % (duree_pattern + delay))
    return times, dists


""" spikeTrains and signal_times need to be ordered """


def SpikesDistFromPat(spikeTrain, duree_pattern, signal_times, window=0.5, offset=0):
    times = []
    dists = []
    r = 0

    for event in signal_times:
        for spike_i in range(r, len(spikeTrain)):
            if spikeTrain[spike_i] > (event + window / 2) + offset:
                break
            if spikeTrain[spike_i] < (event - window / 2) + offset:
                r += 1
            else:
                dists.append(spikeTrain[spike_i] - event)
                times.append(spikeTrain[spike_i])

    return times, dists


@numba.jit(nopython=True)
def SpikesDistNeurons(spikeTrainN, listsizesN, signal_times, window=0.5, offset=0, nb_neuron=4096):
    spikeTrain = spikeTrainN[:, 0]
    neurones = spikeTrainN[:, 1]

    # listdistN = [[] for _ in range(nb_neuron)]
    listdistN = np.empty((nb_neuron, 10000, 2))

    r = 0

    for event in signal_times:
        for spike_i in range(r, len(spikeTrain)):
            if spikeTrain[spike_i] > (event + window / 2) + offset:
                break
            if spikeTrain[spike_i] < (event - window / 2) + offset:
                r += 1
            else:
                neuron = int(neurones[spike_i])
                mtuple = spikeTrain[spike_i] - event, spikeTrain[spike_i]
                listdistN[neuron, listsizesN[neuron]] = mtuple
                listsizesN[neuron] += 1

    return listdistN, listsizesN


def in_pattern(spikeTrain, duree_pattern, signal_times):
    in_pat = np.zeros(len(spikeTrain))

    r = 0
    for i in range(len(spikeTrain)):
        time = spikeTrain[i][0]
        distance_event2 = signal_times[1 + r] - time

        if distance_event2 < 0 and r < len(signal_times) - 2:
            while signal_times[1 + r] - time < 0:
                r += 1
                if r > len(signal_times) - 2:
                    break

        distance_event1 = time - signal_times[0 + r]

        if distance_event1 < duree_pattern and distance_event1 > 0:
            in_pat[i] = 1

    return in_pat


@numba.jit(nopython=True)
def in_pattern_proportion(spikeTrain, duree_pattern, signal_times):
    in_pat_number = 0
    r = 0
    for i in range(len(spikeTrain)):

        time = spikeTrain[i][0]
        distance_event2 = signal_times[1 + r] - time

        if distance_event2 < 0 and r < len(signal_times) - 2:
            while signal_times[1 + r] - time < 0:
                r += 1
                if r > len(signal_times) - 2:
                    break

        distance_event1 = time - signal_times[0 + r]

        if distance_event1 < duree_pattern and distance_event1 > 0:
            in_pat_number += 1

    return in_pat_number / len(spikeTrain)


@numba.jit(nopython=True)
def in_pattern_proportion_neurons(spikeTrain, duree_pattern, signal_times, nb_neuron):
    # spikeTrain = spikeTrainN[:,0]
    # neurones = spikeTrainN[:,1]

    # listdistN = [[] for _ in range(nb_neuron)]
    in_pat_neurons = np.zeros(nb_neuron)
    total_neurons = np.zeros(nb_neuron)

    r = 0
    for i in range(len(spikeTrain)):

        time = spikeTrain[i, 0]
        neuron = int(spikeTrain[i, 1])
        total_neurons[neuron] += 1
        distance_event2 = signal_times[1 + r] - time

        if distance_event2 < 0 and r < len(signal_times) - 2:
            while signal_times[1 + r] - time < 0:
                r += 1
                if r > len(signal_times) - 2:
                    break

        distance_event1 = time - signal_times[0 + r]

        if distance_event1 < duree_pattern and distance_event1 > 0:
            in_pat_neurons[neuron] += 1

    return in_pat_neurons / total_neurons


@numba.jit(nopython=True)
def in_events_nbspike_neurons(spikeTrain, start_event_times, end_event_times, nb_neuron):
    in_pat_neurons = np.zeros((len(start_event_times), nb_neuron))

    r = 0
    for i in range(len(spikeTrain)):

        time = spikeTrain[i, 0]
        neuron = int(spikeTrain[i, 1])
        distance_event2 = start_event_times[1 + r] - time

        if distance_event2 < 0 and r < len(start_event_times) - 2:
            while start_event_times[1 + r] - time < 0:
                r += 1
                if r > len(start_event_times) - 2:
                    break

        distance_event1 = time - start_event_times[r]

        if distance_event1 < end_event_times[r] and distance_event1 > 0:
            in_pat_neurons[r][neuron] += 1

    return in_pat_neurons


def psth_data(spikes, wch_pat, nb_neurons, time_range, signals_times, size_window, duree_pattern):
    spikes_p = spikes
    dis_p, tailles_p = SpikesDistNeurons(spikes_p, np.zeros((nb_neurons,), dtype=np.int),
                                         np.array(signals_times[wch_pat]), window=size_window, offset=duree_pattern / 2,
                                         nb_neuron=nb_neurons)

    all_dis_p = []
    for d in range(len(dis_p)):
        all_dis_p.append(dis_p[d, :tailles_p[d]])

    data_p = all_dis_p
    dist_p = []
    tms_p = []
    for i_p in data_p:
        for j_p in i_p:
            dist_p.append(j_p[0])
            tms_p.append(j_p[1])

    return data_p, spikes_p, dist_p, tms_p


def parallelize(procces_number, number_iter, time_range, nb_signal, size_window, sfo, dure_simu, signals_times,
                nb_neurons, duree_pattern, starting_time=0):
    spikes_in_time = dict()
    dist_in_time = dict()
    times_in_time = dict()
    data_in_time = dict()
    dure_simu = dure_simu - starting_time

    for wch_pat in range(nb_signal):
        spikes_in_time[wch_pat] = dict()
        dist_in_time[wch_pat] = dict()
        times_in_time[wch_pat] = dict()
        data_in_time[wch_pat] = dict()
        times = []
        tms = [T for T in range(starting_time + int(dure_simu / number_iter) - time_range, starting_time + dure_simu,
                                int(dure_simu / number_iter))]
        print("extracting_spikes...")
        all_spikes = []
        for T in tms:
            all_spikes.append(np.array(sfo.get_spikes(t_start=T, t_stop=T + time_range)))

        print("computing...")
        with Pool(processes=procces_number) as pool:
            multiple_thread = [pool.apply_async(psth_data, (
            spikes, wch_pat, nb_neurons, time_range, signals_times, size_window, duree_pattern)) for spikes in
                               all_spikes]
            for letem in range(len(tms)):
                data_in_time[wch_pat][tms[letem]], spikes_in_time[wch_pat][tms[letem]], dist_in_time[wch_pat][
                    tms[letem]], times_in_time[wch_pat][tms[letem]] = multiple_thread[letem].get()

        times = list(data_in_time[wch_pat].keys())
    return spikes_in_time, dist_in_time, times_in_time, data_in_time, times

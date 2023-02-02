
python generating_spikes.py -refperiod 0.01 -rate 10 -timesim 10 -nbneurons 21 -nbsegment 1 -outdir "." -namefile "input"
python generating_spikes.py -refperiod 0.01 -rate 10 -timesim 100 -nbneurons 41 -nbsegment 1 -outdir "." -namefile "input_pattern" -pattern -nbpattern 1 -patternsize 0.25 -patternfrequency 0.5 -sparsitypattern 1 -refpattern 0.1

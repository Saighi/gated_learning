
python generating_spikes.py -refperiod 0.01 -rate 10 -timesim 100 -nbneurons 41 -nbsegment 1 -outdir "." -namefile "input"
python generating_spikes.py -refperiod 0.01 -rate 10 -timesim 100 -nbneurons 41 -nbsegment 1 -outdir "." -namefile "input_pattern" -pattern -nbpattern 1 -patternsize 0.25 -patternfrequency 0.5 -sparsitypattern 1 -refpattern 0.1
python generating_spikes.py -refperiod 0.01 -rate 10 -timesim 200 -nbneurons 20 -nbsegment 1 -outdir "." -namefile "input_competitive"
python generating_spikes.py -refperiod 0.01 -rate 10 -timesim 200 -nbneurons 20 -nbsegment 1 -outdir "." -namefile "input_competitive_pattern" -pattern -nbpattern 1 -patternsize 0.25 -patternfrequency 0.5 -sparsitypattern 1 -refpattern 0.1

python generating_spikes.py -refperiod 0.01 -rate 10 -timesim 200 -nbneurons 10000 -nbsegment 1 -outdir "." -namefile "input_pattern" -pattern -nbpattern 1 -patternsize 0.25 -patternfrequency 0.25 -sparsitypattern 1 -refpattern 1
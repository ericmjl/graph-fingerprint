from itertools import product

sge_header = "#!/bin/sh\n"\
             "#$ -S /bin/sh\n"\
             "#$ -cwd\n"\
             "#$ -V\n"\
             "#$ -m e\n"\
             "#$ -M ericmjl@mit.edu\n"\
             "#$ -pe whole_nodes 1\n"\
             "#############################################\n\n"\

archs = ['fp_linear', 'one_conv', 'two_conv']
scorefuncs = ['cf.score', 'cf.score_sum', 'cf.score_sine']
num_feats = [10, 50]

with open('nnet_master.sh', 'w') as master:
    master.write(sge_header)

    for i, (arch, scorefunc, nfeats) in enumerate(
            product(archs, scorefuncs, num_feats)):
        with open('nnet_synth{0}.sh'.format(i), 'w') as f:
            f.write(sge_header)
            f.write('python nnet_arch.py {0} {1} {2}'.format(arch,
                                                             scorefunc,
                                                             nfeats))

        master.write('qsub nnet_synth{0}.sh\n'.format(i))
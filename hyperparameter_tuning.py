import test
import os
import shutil

## Configure Hyperparameter Grid
num_blocks = [1, 2, 3]
num_heads = [1, 2, 3]
hidden_units = [100, 150, 200]
maxlen = [100, 101, 101, 101] ## 100 for T1 and 101 for T2, T3 and T4
datasets = ['T1', 'T2', 'T3', 'T4']
model = 'ADER'

dirpath = os.path.join('results')
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)

case = 1
for i in range(4):
    for b in num_blocks:
        for h in num_heads:
            for u in hidden_units:
                path = 'results/' + datasets[i] + '-' + model + '-CASE' + str(case)
                if not os.path.isdir(path):
                    os.makedirs(path)
                f = open(os.path.join(path, 'Case_Information.txt'), "w")
                f.write('CASE ' + str(case) + '\n')
                f.write('------------------------------\n')
                f.write('Dataset = ' + datasets[i] + '\n')
                f.write('Model = ' + model + '\n')
                f.write('MaxLen = ' + str(maxlen[i]) + '\n')
                f.write('Number of Blocks = ' + str(b) + '\n')
                f.write('Number of Heads = ' + str(h) + '\n')
                f.write('Hidden Units = ' + str(u) + '\n')
                f.close()
                for j in range(5):
                    test.main(datasets[i], maxlen[i], model + '-CASE' + str(case), b, h, u)
                case += 1

import test
import os
import shutil

## Configure Hyperparameter Grid
# learning_rate = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
learning_rate = [0.001, 0.002]
# batch_size = [64, 128, 256, 512, 1024]
batch_size = [256, 512]
# dropout_rate = [0.2, 0.3, 0.5]
dropout_rate = [0.3, 0.5]
maxlen = [100, 101, 101, 101] ## 100 for T1 and 101 for T2, T3 and T4
datasets = ['T1', 'T2', 'T3', 'T4']
model = 'ADER'

dirpath = os.path.join('results')
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)

case = 1
for i in range(4):
    for lr in learning_rate:
        for bs in batch_size:
            for dr in dropout_rate:
                path = 'results/' + datasets[i] + '-' + model + '-CASE' + str(case)
                if not os.path.isdir(path):
                    os.makedirs(path)
                f = open(os.path.join(path, 'Case_Information.txt'), "w")
                f.write('CASE ' + str(case) + '\n')
                f.write('------------------------------\n')
                f.write('Dataset = ' + datasets[i] + '\n')
                f.write('Model = ' + model + '\n')
                f.write('MaxLen = ' + str(maxlen[i]) + '\n')
                f.write('Learning_Rate = ' + str(lr) + '\n')
                f.write('Batch_Size = ' + str(bs) + '\n')
                f.write('Dropout_rate = ' + str(dr) + '\n')
                f.close()
                for j in range(5):
                    test.main(datasets[i], maxlen[i], model + '-CASE' + str(case), lr, bs, dr)
                case += 1

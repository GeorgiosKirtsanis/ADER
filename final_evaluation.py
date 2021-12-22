import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def main():
    metric_list = ['ACCURACY', 'MRR5', 'HIT5', 'NDCG5']
    task_list = ['T1', 'T2', 'T3', 'T4']
    mode_list = ['STEP', 'EPOCH']

    for metric in metric_list:
        for task in task_list:
            for mode in mode_list:
                # Creating file path
                script_dir = os.path.dirname(__file__)

                # Running for 5 iterations
                my_metric = np.array([])
                for iteration in range(1, 6):
                    metric_path = 'results/' + task + '-ADER/test_' + str(iteration) + '/metrics/'
                    results_dir = os.path.join(script_dir, metric_path)
                    if os.path.isdir(results_dir):
                        sample_file_name = task + '_' + mode + '_' + metric + '.csv'
                        full_file_name = os.path.join(results_dir, sample_file_name)
                        df = pd.read_csv(full_file_name, header=None)
                        array = pd.DataFrame.to_numpy(df)
                        if iteration == 1:
                            my_metric = array
                        else:
                            my_metric = np.append(my_metric, array, axis=0)
                    else:
                        break

                # Calculating average and standard deviation
                std = np.std(my_metric, axis=0)
                avg = np.average(my_metric, axis=0)
                iterations = np.arange(1, len(std) + 1)

                # Creating Figure
                fig = plt.figure(figsize=(10, 5))
                plt.plot(iterations, avg, label="Average")
                plt.plot(iterations, std, label="Standard Deviation")
                plt.xlabel(mode)
                plt.ylabel(metric)
                title = task + ' | ' + 'Average and Standard Deviation of ' + metric + ' per ' + mode
                plt.title(title)
                plt.legend()

                # Saving Figure
                final_path = 'results/' + task + '-ADER/test_final/metrics/'
                results_dir = os.path.join(script_dir, final_path)
                if not os.path.isdir(results_dir):
                    os.makedirs(results_dir)
                sample_file_name = task + '_' + mode + '_' + metric
                full_file_name = os.path.join(results_dir, sample_file_name)
                plt.savefig(full_file_name)

                # Saving to .csv
                metric_values = np.array([avg.T, std.T])
                df = pd.DataFrame(metric_values)
                df.to_csv(full_file_name + '.csv', index=False, header=False)


if __name__ == '__main__':
    main()

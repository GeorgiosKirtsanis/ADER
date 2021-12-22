import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


class Metric_Evaluation:
    def __init__(self, options):

        # Getting options values
        metric = options['metric']
        metric_values = options['metric_values']
        task = options['task']
        mode = options['mode']
        set = options['set']
        iterations = np.arange(1, len(metric_values)+1)

        # Creating Figure
        fig = plt.figure(figsize=(10, 5))
        plt.plot(iterations, metric_values)
        plt.xlabel(mode)
        plt.ylabel(metric)
        title = task + ' | ' + metric + " per " + mode + " for " + set + " set."
        plt.title(title)

        # Creating path
        metrics_dir = 'metrics/'
        sample_file_name = task + '_' + mode + '_' + metric
        full_file_name = os.path.join(metrics_dir, sample_file_name)

        # Saving Figure
        plt.savefig(full_file_name)
        # Saving Array to csv
        df = pd.DataFrame(metric_values).T
        df.to_csv(full_file_name + '.csv', index=False, header=False)

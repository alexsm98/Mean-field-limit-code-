import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler


def loss_plot(activ_results, log_scale_x=False, log_scale_y=False, save_path=None):
    title_map = {'train_error': 'Train Error', 'test_error': 'Test Error', 'l2_reg_loss': 'L2 regularized loss'}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    plt.rc('axes', prop_cycle=(cycler('color', ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'])))
    for layers, metrics_dict in activ_results.items():
        label = 'Ridge' if layers == 0 else f'L = {layers}'
        for i, (metric, title) in enumerate(title_map.items()):
            x, y = zip(*metrics_dict[metric].items())
            axs[i].plot(x, y, label=label)
            axs[i].set_xlabel(r'$N$')
            axs[i].set_title(title)
        if log_scale_x:
            plt.xscale('log')
        if log_scale_y:
            plt.yscale('log')
    plt.legend(loc=(1.04, 0.6))

    if save_path is not None:
        plt.savefig(save_path)


def alpha1_plot(results, log_scale_x=False, log_scale_y=False, save_path=None):
    fig = plt.figure(figsize=(15, 4))
    plt.rc('axes', prop_cycle=(cycler('color', ['tab:orange', 'tab:green', 'tab:red', 'tab:purple'])))
    for i, (activation, metrics_dict) in enumerate(results.items()):
        plt.subplot(1, 3, i + 1)
        plt.title(activation)
        for layers in list(metrics_dict.keys())[1:]:
            x, y = zip(*metrics_dict[layers]['alpha1_energy'].items())
            plt.plot(x, y, label=f'L = {layers}')
        plt.xlabel(r'$N$')
        if log_scale_x:
            plt.xscale('log')
        if log_scale_y:
            plt.yscale('log')
        plt.legend()
        
        if save_path is not None:
            plt.savefig(save_path)




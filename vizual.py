import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler


def loss_plot(activ_results, sharey=True, log_scale_x=False, log_scale_y=False, save_path=None):
    """Plot L2 training loss, training and test errors for different activations and number of layers
    Parameters:
        - results: dict, dictionary of results for experiments with fixed activation 
        - sharey: bool, whether to share y-axis 
        - log_scale_x: bool, whether to show x-axis on log-scale
        - log_scale_y: bool, whether to show y-axis on log-scale,
        - save_path: str, save plot using specified path
    """
    title_map = {'train_error': 'Train Error', 'test_error': 'Test Error', 'l2_reg_loss': 'L2 regularized loss'}
    fig, axs = plt.subplots(1, 3, sharey=sharey, figsize=(15, 4))
    plt.rc('axes', prop_cycle=(cycler('color', ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'])))
    for layers, metrics_dict in activ_results.items():
        label = 'Ridge' if layers == 0 else f'L = {layers}'
        for i, (metric, title) in enumerate(title_map.items()):
            x, y = zip(*metrics_dict[metric].items())
            axs[i].plot(x, y, label=label,linewidth=2)
            axs[i].set_xlabel(r'$N$')
            axs[i].set_title(title)
            if log_scale_x:
                axs[i].set_xscale('log')
            if log_scale_y:
                axs[i].set_yscale('log')
    plt.legend(loc=(1.04, 0.6))
    
    if save_path is not None:
            plt.savefig(save_path)
            

def alpha1_plot(results, log_scale_x=False, sharey=True, log_scale_y=False, save_path=None):
    """Plot energy alpha1 for different activations and number of layers
    Parameters:
        - results: dict, dictionary of results for experiments with multiple activations
        - sharey: bool, whether to share y-axis 
        - log_scale_x: bool, whether to show x-axis on log-scale
        - log_scale_y: bool, whether to show y-axis on log-scale,
        - save_path: str, save plot using specified path
    """
    N_plots = len(results)
    fig, axs = plt.subplots(1, N_plots, sharey=sharey, figsize=(15, 4))
    plt.rc('axes', prop_cycle=(cycler('color', ['tab:orange', 'tab:green', 'tab:red', 'tab:purple'])))
    for i, (activation, metrics_dict) in enumerate(results.items()):
        axs[i].set_title(activation)
        for layers in list(metrics_dict.keys())[1:]:
            x, y = zip(*metrics_dict[layers]['alpha1_energy'].items())
            axs[i].plot(x, y, label=f'L = {layers}', linewidth=2)
        axs[i].set_xlabel(r'$N$')
        if log_scale_x:
            axs[i].set_xscale('log')
        if log_scale_y:
            axs[i].set_yscale('log')
    plt.legend(loc=(1.04, 0.6))
        
    if save_path is not None:
            plt.savefig(save_path)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_dataset(dataset, key,  names):
    for data, name in zip(dataset, names):
        plot_function(data, key, name)

def plot_function(data, key, activation):
    plt.plot(data['layer_size'], data[key], 'x--', label=activation)


def decorate_plot(ax, ylabel, ylim):
    ax.set_xlabel("layer_size")
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim[0], ylim[1])
    ax.legend(loc=6)
    ax.grid()

def plot_save_and_close(fig, name):
    plt.subplots_adjust(wspace=0.35)

    plt.savefig(name)
    plt.close(fig)
    
    
df = pd.read_csv(sys.argv[1], delimiter='\t')
layers = np.unique(df["layer_cnt"])
df = pd.read_csv('scores_new.tsv', delimiter='\t')
df["neurons"] = df['layer_cnt']*df['layer_size']
df['top_1_accuracy'] = df['top_1_accuracy'].astype('float64')
df['top_5_accuracy'] = df['top_5_accuracy'].astype('float64')
df['neurons'] = df['neurons'].astype('int32')
layers = np.unique(df["layer_cnt"])

for layer in layers:
    db = [df.loc[df['activation'] == fun].loc[df.loc[df['activation'] == fun].layer_cnt == layer] for fun in np.unique(df["activation"])]
    fun_names = [df.activation.iloc[0] for df in db]
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    plt.suptitle('{layer} layers'.format(layer=layer), y=0.95, size=20)
    top1_subplot = fig.add_subplot(1, 2, 1)
    plot_dataset(db, 'top_1_accuracy', fun_names)
    decorate_plot(top1_subplot, 'top_1_accuracy', [0.0, 1.1])
    top5_subplot = fig.add_subplot(1, 2, 2)
    plot_dataset(db, 'top_5_accuracy', fun_names)
    decorate_plot(top5_subplot, 'top_5_accuracy', [0.0, 1.1])
    plot_save_and_close(fig, '{layer}_layers.jpg'.format(layer=layer))
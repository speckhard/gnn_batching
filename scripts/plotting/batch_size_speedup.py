"""Let's make a bar chart plot of the speedup compared to slowest method."""
import pandas as pd
import os
import sys
from absl import app
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
from matplotlib import rc, font_manager


FONTSIZE = 12
FONT = 'serif'

fontProperties = {'family':'sans-serif','sans-serif':['Times'],
    'weight' : 'normal', 'size' : FONTSIZE}
ticks_font = font_manager.FontProperties(family='Times', style='normal',
    size=FONTSIZE, weight='normal', stretch='normal')
# rc('text', usetex=True)
rc('text')
rc('font',**fontProperties)


BASE_DIR = '/home/dts/Documents/hu/jraph_MPEU/batch_data'
## The 2 million profiling steps data:
# Used for MPEU/SchNet
# COMBINED_CSV = 'parsed_profiling_batching_2_000_000_steps_combined_19_01_2025.csv'
# Used for PaiNN
COMBINED_CSV = 'parsed_profiling_painn_batching_2_000_000_steps_15_05_2025.csv'


def plot_batch_speedup(df):
    """Create bar plot of # of recompilations.
    
    X-axis is the batch size.
    Y-axis is the number of recompilations.
    """
    
    # profile_column = 'recompilation'
    profile_column = 'step_2_000_000_update_time_mean'
    computing_type = 'gpu_a100'
    model = 'painn'
    dataset = 'aflow'
    color_list = ['#1f77b4', '#ff7f0e', '#9467bd']

    # Create a new batching method, batch-64 based on rounding.
    df.loc[
        (df.batching_type == 'static') & (df.batching_round_to_64 == True),'batching_type'] ='static-64'
    # Get data only for gpu and AFLOW and MPEU
    # Static 64 reuslts
    print(df[
        (df['dataset'] == dataset) & (df['model'] == model) &
        (df['batching_type'] == 'static') &
        (df['computing_type'] == computing_type) &
        (df['batch_size'] == 128) &
        (df['batching_round_to_64'] == True)][profile_column])
    # static 2N results
    print(df[
        (df['dataset'] == dataset) & (df['model'] == model) &
        (df['batching_type'] == 'static') &
        (df['computing_type'] == computing_type) &
        (df['batch_size'] == 128) &
        (df['batching_round_to_64'] == False)][profile_column])
    df = df[df['model'] == model]
    df = df[df['computing_type'] == computing_type]
    df = df[df['dataset'] == dataset]

    # Now take the mean over the different iterations.

    df = df[['batch_size', 'batching_type', profile_column]]
    print(df[df['batch_size'] == 128])
    df = df.groupby(['batch_size', 'batching_type']).mean()
    ungrouped_df = df.reset_index()
    slowest_time = {'16': 0, '32': 0, '64': 0, '128': 0}
    print(df.columns)
    print(df)
    print('ungrouped df')
    print(ungrouped_df)
    print(ungrouped_df.columns)
    # Speedup is defined as slowest time - time / slowest time.
    for batch_size in [16, 32, 64, 128]:

        for batching_type in ['static', 'static-64', 'dynamic']:
            time = ungrouped_df[(ungrouped_df['batch_size'] == batch_size) & (ungrouped_df['batching_type'] == batching_type)]['step_2_000_000_update_time_mean'].values[0]
            print(f' the slowest time for the batch size: {batch_size} is {slowest_time[str(batch_size)]}')
            print(f'time is {time}')
            if time > slowest_time[str(batch_size)]:
                slowest_time[str(batch_size)] = time
        for batching_type in ['static', 'static-64', 'dynamic']:
            time = ungrouped_df[(ungrouped_df['batch_size'] == batch_size) & (ungrouped_df['batching_type'] == batching_type)]['step_2_000_000_update_time_mean'].values[0]
            ungrouped_df.loc[(ungrouped_df['batch_size'] == batch_size) & (ungrouped_df['batching_type'] == batching_type), 'step_2_000_000_update_time_mean'] = (slowest_time[str(batch_size)]- time)/slowest_time[str(batch_size)]*100
    
    # There is no stdev since the data is alwasy the same shuffle.
    # df_std = df.groupby(['batch_size', 'batching_type']).std()
    # print(df_std)
    print('ungrouped df after modifying values')
    print(ungrouped_df)

    print(ungrouped_df.groupby(['batch_size', 'batching_type']))
    # df = ungrouped_df.groupby(['batch_size', 'batching_type']).mean()
    # ax = df.unstack().plot.bar(figsize=(5.1, 4), color=color_list)
    # ax = ungrouped_df.transpose().plot.bar(figsize=(5.1, 4), color=color_list)
    # ax = ungrouped_df.plot.bar(figsize=(5.1, 4), color=color_list)


    bar_width = 0.3
    x = np.arange(len([16, 32, 64, 128]))
    print(ungrouped_df[ungrouped_df['batching_type'] == 'dynamic']['step_2_000_000_update_time_mean'])
    __, ax = plt.subplots(figsize=(5.1, 4))
    # Grouped Bar Plot
    plt.bar(x - 0.3, ungrouped_df[ungrouped_df['batching_type'] == 'dynamic']['step_2_000_000_update_time_mean'], bar_width, label='dynamic', color='skyblue')
    plt.bar(x + 0.3, ungrouped_df[ungrouped_df['batching_type'] == 'static']['step_2_000_000_update_time_mean'], bar_width, label='static-$2^N$', color='green')
    plt.bar(x, ungrouped_df[ungrouped_df['batching_type'] == 'static-64']['step_2_000_000_update_time_mean'], bar_width, label='static-$64$', color='mediumvioletred')


    import matplotlib.font_manager as font_manager
    font = font_manager.FontProperties(family=FONT,
                                    # weight='bold',
                                    style='normal', size=FONTSIZE)


    ax.set_xlabel('Batch size', fontsize=FONTSIZE, font=FONT)
    ax.set_ylabel('Relative combined time speedup %', fontsize=FONTSIZE, font=FONT)
    ax.set_xticks([0, 1, 2, 3], font=FONT, fontsize=FONTSIZE, rotation=45)

    ax.set_xticklabels([16, 32, 64, 128], font=FONT, fontsize=FONTSIZE, rotation=45)

    if dataset == 'aflow':
        ax.set_yticks([0, 10, 20, 30, 40, 50], font=FONT, fontsize=FONTSIZE)

        ax.set_yticklabels([0, 10, 20, 30, 40, 50], font=FONT, fontsize=FONTSIZE)

    else:
        ax.set_yticklabels([0, 50, 100, 150, 200, 250], font=FONT, fontsize=FONTSIZE)
    plt.legend(
        ["dynamic", "static-$2^N$", "static-$64$"], fontsize=FONTSIZE,
        prop=font, edgecolor="black", fancybox=False, loc='upper left')
    offset=0
    ax.text(2.5, 45+offset, 'PaiNN', font=FONT, fontsize=FONTSIZE)
    ax.text(2.5, 42+offset, 'AFLOW', font=FONT, fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(
        f'/home/dts/Documents/theory/batching_paper/figs/batch_speed_up_combined_time_{dataset}_model_{model}.png',
        dpi=600)
    plt.show()



def main(argv):
    df = pd.read_csv(os.path.join(BASE_DIR, COMBINED_CSV))

    plot_batch_speedup(df)

if __name__ == '__main__':
    app.run(main)


"""Get t-test metrics for learning curves.

Reviewer asked for t-test metric.

I think it makes most sense to get the t-test between mean test RMSE curves
for different batching methods. We did this already in the other script by
by returning the mean dataframe.

It also would make sense to look at the t-test for the distribution of the
test RMSE after 2 million steps from different batching methods. This
is what the method here covers.



"""
import argparse
import os
import pickle
from absl import flags
from absl import app
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
import seaborn as sns
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'csv_filename',
    'None',
    'Where to store data as csv that has been parsed.')

# BASE_DIR = '/home/dts/Documents/hu/jraph_MPEU/batch_data'
# # COMBINED_CSV = 'parsed_profiling_batching_2_000_000_steps_aflow_qm9_20_12_2024.csv'
# COMBINED_CSV = 'parsed_profiling_batching_2_000_000_steps_combined_19_01_2025.csv'

BASE_DIR = '/home/dts/Documents/hu/'
COMBINED_CSV = 'parsed_profiling_batching_2_000_000_steps_combined_6_04_2025.csv'
# TRAINING_STEP = [
#     '100', '200', '300', '400', '500', '600', '700', '800', '900', '1_000',
#     '1_100', '1_200', '1_300', '1_400', '1_500', '1_600', '1_700', '1_800',
#     '1_900', '2_000']
TRAINING_STEP = ['2_000']

def get_column_list(data_split):
    """Get a list of columns to keep in the dataframe."""
    col_list = []
    for step in TRAINING_STEP:
        col_list.append('step_X_000_split_rmse'.replace('X', step).replace(
            'split', data_split))
    return col_list


def get_t_test_data(
        df, model, data_split, dataset, batch_size,
        primary_batching_method, secondary_batching_method, compute_type):
    

    # columns_to_keep = get_column_list(data_split)
    # df = df[columns_to_keep]
    column = 'step_2_000_000_test_rmse'
    if primary_batching_method == 'static-64':
        batching_round_to_64 = True
        primary_batching_method = 'static'

    else:
        batching_round_to_64 = False

    primary_df = df[
        (df['model'] == model) & (df['batching_type'] == primary_batching_method)
        & (df['computing_type'] == compute_type) & (df['batch_size'] == batch_size)
        & (df['dataset'] == dataset) & (df['batching_round_to_64'] == batching_round_to_64)]

    if secondary_batching_method == 'static-64':
        batching_round_to_64 = True
        secondary_batching_method = 'static'
    else:
        batching_round_to_64 = False
    secondary_df = df[
        (df['model'] == model) & (df['batching_type'] == secondary_batching_method)
        & (df['computing_type'] == compute_type) & (df['batch_size'] == batch_size)
        & (df['dataset'] == dataset) & (df['batching_round_to_64'] == batching_round_to_64)]
    print(f'the primary df is {primary_df[column]}')
    print(f'the secondary df is {secondary_df[column]}')
    return primary_df[column].to_numpy(), secondary_df[column].to_numpy()
# computing_typecomputing_type

def plot_t_test(t_test_matrix, batch_size, model, dataset):
    sns.set_theme(font="serif")
    f, ax = plt.subplots(figsize=(5.1, 4.3))
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    x_axis_labels = ['dynamic', 'static-$2^N$', 'static-64']
    y_axis_labels = ['dynamic', 'static-$2^N$', 'static-64']
    # sns.set_theme(font_scale=1.4)
    g = sns.heatmap(t_test_matrix, cmap=cmap, 
            square=True,
            linewidth=.8, cbar_kws={"shrink": .99}, ax=ax,
            annot_kws={'size': 15},
            xticklabels=x_axis_labels, yticklabels=y_axis_labels,
            annot=True, vmin=-3, vmax=3)
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 12)
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 12)

    plt.xlabel('Batching method', fontsize=12)
    plt.ylabel('Batching method', fontsize=12)
    # plt.title(f'T test statistics: Data: {dataset}, model: {model}, batch_size: {batch_size}', fontsize=16)
    plt.tight_layout()
    base_dir = '/home/dts/Documents/theory/batching_paper/figs/t_test_data/'
    plt.savefig(f'{base_dir}data_{dataset}_model_{model}_batch_size_{batch_size}_t_test_statistics')
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.yaxis.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    plt.show()

def plot_t_test_p_values(p_value_matrix, batch_size, model, dataset):
    sns.set_theme(font="serif")
    f, ax = plt.subplots(figsize=(5.1, 4.3))
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    x_axis_labels = ['dynamic', 'static-$2^N$', 'static-64']
    y_axis_labels = ['dynamic', 'static-$2^N$', 'static-64']
    # sns.set_theme(font_scale=1.4)
    g = sns.heatmap(p_value_matrix, cmap=cmap, 
            square=True,
            linewidth=.8, cbar_kws={"shrink": .99, 'ticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0]}, ax=ax,
            annot_kws={'size': 15},
            xticklabels=x_axis_labels, yticklabels=y_axis_labels,
            annot=True, vmin=0, vmax=1.0)
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 12)
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 12)

    plt.xlabel('Batching method', fontsize=12)
    plt.ylabel('Batching method', fontsize=12)
    # plt.title(f'T test p-values: Data: {dataset}, model: {model}, batch_size: {batch_size}', fontsize=16)
    plt.tight_layout()
    base_dir = '/home/dts/Documents/theory/batching_paper/figs/t_test_data/'
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.yaxis.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.savefig(f'{base_dir}data_{dataset}_model_{model}_batch_size_{batch_size}_t_test_p_values')
    plt.show()

def main(args):

    df = pd.read_csv(os.path.join(BASE_DIR, COMBINED_CSV))
    print(df.columns)
    # print(ttest_ind(schnet_df['static'], schnet_df['dynamic']))
    model = 'schnet'
    data_split = 'test'
    dataset = 'qm9'
    primary_batching_method = 'static'
    secondary_batching_method = 'dynamic'
    compute_type = 'gpu_a100'
    # BATCH_SIZE_LIST = [16, 32, 64, 128]
    # BATCHING_METHODS = ['dynamic', 'static', 'static-64']
    # MODEL_TYPE_LIST = ['schnet', 'MPEU']
    # DATASET_LIST = ['aflow', 'qm9']

    BATCH_SIZE_LIST = [64]
    BATCHING_METHODS = ['dynamic', 'static', 'static-64']
    MODEL_TYPE_LIST = ['schnet']
    DATASET_LIST = ['qm9']

    t_test_matrix = np.zeros((3,3))
    p_value_matrix = np.zeros((3, 3))
    for batch_size in BATCH_SIZE_LIST:
        for dataset in DATASET_LIST:
            for model in MODEL_TYPE_LIST:
                for i, primary_batching_method in enumerate(BATCHING_METHODS):
                    for j, secondary_batching_method in enumerate(BATCHING_METHODS):
                        
                        primary, secondary = get_t_test_data(
                            df, model, data_split, dataset, batch_size, primary_batching_method,
                            secondary_batching_method, compute_type)
                        print(f'i,j are: {i}, {j}')
                        t_test_matrix[i, j] = ttest_ind(primary, secondary).statistic
                        p_value_matrix[i, j] = ttest_ind(primary, secondary).pvalue
                
                print(f'primary to numpy {t_test_matrix}')
                print(f'secondary to numpy {p_value_matrix}')

                print(ttest_ind(primary, secondary))
                print(t_test_matrix)
                print(p_value_matrix)
                plot_t_test(
                    t_test_matrix=t_test_matrix, batch_size=batch_size,
                    model=model, dataset=dataset)
                plot_t_test_p_values(
                    p_value_matrix=p_value_matrix, batch_size=batch_size,
                    model=model, dataset=dataset)


if __name__ == '__main__':
    app.run(main)

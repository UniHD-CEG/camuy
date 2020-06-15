import os
import pandas as pd
import re
import numpy as np
from numpy import unravel_index
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker

font_size=9

def parse_file_name(filename):
    log_parameters = [int(substring) for substring in re.findall(r'\d+', filename)]
    return log_parameters[1], log_parameters[2]

def colorbar_tick_string_formatter(x, pos):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def heatmap(data, row_labels, col_labels, ax=None,
                font_size=12, cbar_kw={}, cbarlabel="",
                is_data_movement_cost_plot=False, **kwargs):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    
    if is_data_movement_cost_plot == False:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, **cbar_kw)
    else:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                                    format=ticker.FuncFormatter(colorbar_tick_string_formatter), **cbar_kw)
    
    cbar.ax.tick_params(axis='both', which='major', labelsize=font_size)
    cbar.ax.set_ylabel(cbarlabel, fontsize=font_size, rotation=-90, va="bottom")

    tick_range=np.arange(start=2, stop=32, step=4)

    ##We want to show all ticks...
    
    ax.set_xticks(tick_range)
    ax.set_yticks(tick_range)
    #... and label them with the respective list entries
    
    ax.set_xticklabels(col_labels[tick_range], fontsize=font_size)
    ax.set_yticklabels(row_labels[tick_range], fontsize=font_size)

    ##Let the horizontal axes labeling appear on top
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

if __name__ == '__main__':
    labels=np.arange(start=16, stop=257, step=8)
    data_movement_cost_df = pd.DataFrame(data=None, index=labels, columns=labels, dtype='int64')
    utilization_df = pd.DataFrame(data=None, index=labels, columns=labels, dtype='float64')
    for log_file_name in os.listdir("mpu_log/width_height_sweep/"):
        log_file_name_and_path = "mpu_log/width_height_sweep/{}".format(log_file_name)
        df = pd.read_csv(log_file_name_and_path, sep='\t')
        intra_pe_data_movements_total = sum(df.iloc[:, 18])
        inter_pe_data_movements_total = sum(df.iloc[:, 19])
        systolic_data_setup_unit_load_count_total = sum(df.iloc[:, 20])
        weight_fetcher_load_count_total = sum(df.iloc[:, 21])
        accumulator_array_load_count_total = sum(df.iloc[:, 24])
        cycle_count_total = sum(df.iloc[:, 27].astype('float64'))
        zero_weight_multiplications_total = sum(df.iloc[:, 29].astype('float64'))
        row, col = parse_file_name(log_file_name)
        accumulator_array_store_count_total = col*(cycle_count_total - row)
        data_movement_cost_df.loc[row, col] = intra_pe_data_movements_total + \
                                                2*(inter_pe_data_movements_total + \
                                                    accumulator_array_store_count_total) + \
                                                6*(systolic_data_setup_unit_load_count_total + \
                                                    weight_fetcher_load_count_total + \
                                                    accumulator_array_load_count_total)
        utilization_df.loc[row, col] =  100.0*(1.0 - (zero_weight_multiplications_total / \
                                                                (row*col*cycle_count_total)))

        
    data_movement_cost_sorted_np = data_movement_cost_df.sort_index(0).sort_index(1).fillna(1).astype('int64').to_numpy()
    utilization_sorted_np = utilization_df.sort_index(0).sort_index(1).fillna(1).astype('float64').to_numpy()
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 3.5))
    
    im, cbar = heatmap(data_movement_cost_sorted_np,
                            labels, labels,
                            font_size=font_size,
                            ax=ax[0], cmap="RdYlGn_r",
                            cbarlabel="Data Movement Cost",
                            is_data_movement_cost_plot=True)
    
    ax[0].set_xlabel('Systolic Array Width', fontsize=font_size)
    ax[0].xaxis.set_label_position('top') 
    ax[0].set_ylabel('Systolic Array Height', fontsize=font_size)
    
    im, cbar = heatmap(utilization_sorted_np,
                            labels, labels,
                            font_size=font_size,
                            ax=ax[1], cmap="RdYlGn",
                            cbarlabel="Utilization [%]")
    
    ax[1].set_xlabel('Systolic Array Width', fontsize=font_size)
    ax[1].xaxis.set_label_position('top') 
    ax[1].set_ylabel('Systolic Array Height', fontsize=font_size)


    fig.tight_layout()
    plt.savefig('heatmaps_combined_mobilenet_v3_small.png', dpi=900)
    
    

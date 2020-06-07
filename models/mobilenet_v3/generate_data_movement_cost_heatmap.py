import os
import pandas as pd
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def parse_file_name(filename):
    log_parameters = [int(substring) for substring in re.findall(r'\d+', filename)]
    return log_parameters[1], log_parameters[2]

def heatmap(data, row_labels, col_labels, ax=None,
                font_size=12, cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, **cbar_kw)
    cbar.ax.tick_params(axis='both', which='major', labelsize=font_size)
    cbar.ax.set_ylabel(cbarlabel, fontsize=font_size, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontsize=font_size)
    ax.set_yticklabels(row_labels, fontsize=font_size)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-60,
                ha="right", rotation_mode="anchor")

    #Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

if __name__ == '__main__':
    labels=np.arange(start=16, stop=257, step=8)
    result_df = pd.DataFrame(data=None, index=labels, columns=labels, dtype='uint64')
    for log_file_name in os.listdir("mpu_log/width_height_sweep/"):
        df = pd.read_csv("mpu_log/width_height_sweep/{}".format(log_file_name), sep='\t')
        intra_pe_data_movements_total = sum(df.iloc[:, 18])
        inter_pe_data_movements_total = sum(df.iloc[:, 19])
        systolic_data_setup_unit_load_count_total = sum(df.iloc[:, 20])
        weight_fetcher_load_count_total = sum(df.iloc[:, 21])
        accumulator_array_load_count_total = sum(df.iloc[:, 24])
        row, col = parse_file_name(log_file_name)
        accumulator_array_store_count_total = col*(sum(df.iloc[:, 27]) - row)
        result_df.loc[row, col] = intra_pe_data_movements_total + \
                                    2*(inter_pe_data_movements_total + \
                                        accumulator_array_store_count_total) + \
                                    6*(systolic_data_setup_unit_load_count_total + \
                                        weight_fetcher_load_count_total + \
                                        accumulator_array_load_count_total)
        
    result_df_sorted = result_df.sort_index(0).sort_index(1).fillna(1).astype('uint64')
    
    fig, ax = plt.subplots(figsize=(11, 11))
    
    im, cbar = heatmap(result_df_sorted.to_numpy(),
                            labels, labels,
                            font_size=12.5,
                            ax=ax, cmap="RdYlGn_r",
                            cbarlabel="Data Movement Cost")
    
    ax.set_xlabel('Systolic Array Width', fontsize=12.5)
    ax.xaxis.set_label_position('top') 
    ax.set_ylabel('Systolic Array Height', fontsize=12.5)
    #ax.set_title("MPU: Data Movement Cost for\n"
                    #"Inference of VGG-16 Model for\n"
                    #"Different Systolic Array Size Combinations",
                    #pad=20, fontsize=18)
    fig.tight_layout()
    #plt.savefig('accumulator_array_store_cost_heatmap_vgg16.png', dpi=300)
    plt.savefig('data_movement_cost_heatmap_mobilenet_v3.png', dpi=300)
    
    

import os
import pandas as pd
import re
import numpy as np
import pygmo as pg
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from adjustText import adjust_text

def parse_file_name(filename):
    log_parameters = [int(substring) for substring in re.findall(r'\d+', filename)]
    return log_parameters[1], log_parameters[2]

if __name__ == '__main__':
    cycle_count_df = pd.DataFrame(data=None, columns=['Cycle Count'], dtype='uint64')
    data_movement_cost_df = pd.DataFrame(data=None, columns=['Data Movement Cost'], dtype='uint64')
    utilization_df = pd.DataFrame(data=None, columns=['Utilization'], dtype='float64')
    utilization_inv_df = pd.DataFrame(data=None, columns=['Inverse Utilization'], dtype='float64')
    for log_file_name in os.listdir("mpu_log/width_height_sweep/"):
        df = pd.read_csv("mpu_log/width_height_sweep/{}".format(log_file_name), sep='\t')
        intra_pe_data_movements_total = sum(df.iloc[:, 18])
        inter_pe_data_movements_total = sum(df.iloc[:, 19])
        systolic_data_setup_unit_load_count_total = sum(df.iloc[:, 20])
        weight_fetcher_load_count_total = sum(df.iloc[:, 21])
        accumulator_array_load_count_total = sum(df.iloc[:, 24])
        cycle_count_total = sum(df.iloc[:, 27])
        zero_weight_multiplications_total = sum(df.iloc[:, 29])
        row, col = parse_file_name(log_file_name)
        element_id = "({}, {})".format(row, col)
        cycle_count_df.loc[element_id] = cycle_count_total
        accumulator_array_store_count_total = col*(cycle_count_total - row)
        data_movement_cost_df.loc[element_id] = \
                                    intra_pe_data_movements_total + \
                                    2*(inter_pe_data_movements_total + \
                                        accumulator_array_store_count_total) + \
                                    6*(systolic_data_setup_unit_load_count_total + \
                                        weight_fetcher_load_count_total + \
                                        accumulator_array_load_count_total)
        utilization_df.loc[element_id] = \
                        100.0*(1.0 - zero_weight_multiplications_total/ \
                                                (row*col*cycle_count_total))
        utilization_inv_df.loc[element_id] = zero_weight_multiplications_total/ \
                                                (row*col*cycle_count_total)
                                
    cycle_count_np = cycle_count_df.astype('uint64').to_numpy()
    data_movement_cost_np = data_movement_cost_df.astype('uint64').to_numpy()
    utilization_np = utilization_df.astype('float64').to_numpy()
    utilization_inv_np = utilization_inv_df.astype('float64').to_numpy()
                                
    pareto_optimal_points_data_movement_cost = \
                            pg.non_dominated_front_2d(points=(np.concatenate((cycle_count_np,
                                                                data_movement_cost_np), axis=1)))
    
    cycle_count_pareto_optimal_data_movement_cost_np = cycle_count_np[pareto_optimal_points_data_movement_cost]
    data_movement_cost_pareto_optimal_np = data_movement_cost_np[pareto_optimal_points_data_movement_cost]
    
    cycle_count_annotations_pareto_optimal_data_movement_cost = []
    
    for cycle_count in cycle_count_pareto_optimal_data_movement_cost_np:
        cycle_count_annotations_pareto_optimal_data_movement_cost.append(next(iter(cycle_count_df[cycle_count_df['Cycle Count']==cycle_count[0]].index), 'No Match'))
        
    data_movement_annotations_pareto_optimal = []
    
    for data_movement_cost in data_movement_cost_pareto_optimal_np:
        data_movement_annotations_pareto_optimal.append(next(iter(data_movement_cost_df[data_movement_cost_df['Data Movement Cost']==data_movement_cost[0]].index), 'No Match'))
        
    assert cycle_count_annotations_pareto_optimal_data_movement_cost == data_movement_annotations_pareto_optimal
    
    index_euclidean_distance_min_data_movement_cost = np.argmin(((cycle_count_pareto_optimal_data_movement_cost_np/
                                                                    max(cycle_count_pareto_optimal_data_movement_cost_np))**2 + \
                                                                    (data_movement_cost_pareto_optimal_np/ \
                                                                    max(data_movement_cost_pareto_optimal_np))**2)**(1.0/2.0))
    
    pareto_optimal_points_utilization = \
                        pg.non_dominated_front_2d(points = (np.concatenate((cycle_count_np,
                                                                utilization_inv_np), axis=1)))
                    
    cycle_count_pareto_optimal_utilization_np = cycle_count_np[pareto_optimal_points_utilization]
    utilization_pareto_optimal_np = utilization_np[pareto_optimal_points_utilization]
    utilization_inv_pareto_optimal_np = utilization_inv_np[pareto_optimal_points_utilization]
                    
    cycle_count_annotations_pareto_optimal_utilization = []
    
    for cycle_count in cycle_count_pareto_optimal_utilization_np:
        cycle_count_annotations_pareto_optimal_utilization.append(next(iter(cycle_count_df[cycle_count_df['Cycle Count']==cycle_count[0]].index), 'No Match'))
        
    utilization_annotations_pareto_optimal = []
    
    for utilization in utilization_pareto_optimal_np:
        utilization_annotations_pareto_optimal.append(next(iter(utilization_df[utilization_df['Utilization']==utilization[0]].index), 'No Match'))
        
    assert cycle_count_annotations_pareto_optimal_utilization == utilization_annotations_pareto_optimal
    
    index_euclidean_distance_min_utilization = np.argmin(((cycle_count_pareto_optimal_utilization_np/ \
                                                            max(cycle_count_pareto_optimal_utilization_np))**2 + \
                                                            (utilization_inv_pareto_optimal_np/ \
                                                            max(utilization_inv_pareto_optimal_np))**2)**(1.0/2.0))
    
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.scatter(cycle_count_np, data_movement_cost_np, color='whitesmoke')
    plt.scatter(cycle_count_pareto_optimal_data_movement_cost_np,
                    data_movement_cost_pareto_optimal_np, color='tab:blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Cycle Count')
    plt.ylabel('Data Movement Cost')
    
    parato_front_size = len(cycle_count_pareto_optimal_data_movement_cost_np)
    
    texts = []
    
    texts.append(plt.text(cycle_count_pareto_optimal_data_movement_cost_np[0],
                            data_movement_cost_pareto_optimal_np[0],
                            cycle_count_annotations_pareto_optimal_data_movement_cost[0],
                            ha='center', va='center'));
    
    #texts.append(plt.text(cycle_count_pareto_optimal_data_movement_cost_np[index_euclidean_distance_min_data_movement_cost],
                            #data_movement_cost_pareto_optimal_np[index_euclidean_distance_min_data_movement_cost],
                            #cycle_count_annotations_pareto_optimal_data_movement_cost[index_euclidean_distance_min_data_movement_cost],
                            #ha='center', va='center'));
    
    texts.append(plt.text(cycle_count_pareto_optimal_data_movement_cost_np[parato_front_size - 1],
                            data_movement_cost_pareto_optimal_np[parato_front_size - 1],
                            cycle_count_annotations_pareto_optimal_data_movement_cost[parato_front_size - 1],
                            ha='center', va='center'));
    
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k'))
    
    plt.annotate(cycle_count_annotations_pareto_optimal_data_movement_cost[15],
                    xy=(cycle_count_pareto_optimal_data_movement_cost_np[15],
                        data_movement_cost_pareto_optimal_np[15]),
                    xytext=(cycle_count_pareto_optimal_data_movement_cost_np[17],
                            data_movement_cost_pareto_optimal_np[13]),
                    arrowprops=dict(arrowstyle="-", color='k'))
    
    plt.subplot(122)
    plt.scatter(cycle_count_np, utilization_np, color='whitesmoke')
    plt.scatter(cycle_count_pareto_optimal_utilization_np,
                    utilization_pareto_optimal_np, color='tab:blue')
    plt.xscale('log')
    plt.xlabel('Cycle Count')
    plt.ylabel('Utilization [%]')
    
    parato_front_size = len(cycle_count_pareto_optimal_utilization_np)
    
    texts = []
    
    texts.append(plt.text(cycle_count_pareto_optimal_utilization_np[0],
                            utilization_pareto_optimal_np[0],
                            cycle_count_annotations_pareto_optimal_utilization[0],
                            ha='center', va='center'));
    
    #texts.append(plt.text(cycle_count_pareto_optimal_utilization_np[index_euclidean_distance_min_utilization],
                            #utilization_pareto_optimal_np[index_euclidean_distance_min_utilization],
                            #cycle_count_annotations_pareto_optimal_utilization[index_euclidean_distance_min_utilization],
                            #ha='center', va='center'));
    
    texts.append(plt.text(cycle_count_pareto_optimal_utilization_np[parato_front_size - 1],
                            utilization_pareto_optimal_np[parato_front_size - 1],
                            cycle_count_annotations_pareto_optimal_utilization[parato_front_size - 1],
                            ha='center', va='center'));
    
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k'))
    
    plt.annotate(cycle_count_annotations_pareto_optimal_utilization[15],
                    xy=(cycle_count_pareto_optimal_utilization_np[15],
                        utilization_pareto_optimal_np[15]),
                    xytext=(cycle_count_pareto_optimal_utilization_np[17],
                            utilization_pareto_optimal_np[13]),
                    arrowprops=dict(arrowstyle="-", color='k'))
    
    plt.tight_layout()
    plt.savefig('scatterplots_combined_mobilenet_v3_small.png', dpi=900)
    
    

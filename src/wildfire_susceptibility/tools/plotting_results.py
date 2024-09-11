# -*- coding: utf-8 -*-
"""
Created on Mon May  1 07:06:21 2023

@author: Giorg
"""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import StrMethodFormatter
import pandas as pd
import seaborn as sns
import geopandas as gpd
import os
import logging

os.chdir('/share/home/gruppo4/Bolivia/DATA/BolMLmonthly/test_model/annual_wildfire_susceptibility/tools')

from utils import df_statistic


class Plots():
    
    def __init__(self):
        
        self.my_theme = {
            # 'axes.grid': True,
            #           'grid.linestyle': '--',
                      'legend.framealpha': 1,
                      'legend.facecolor': 'white',
                      'legend.shadow': True,
                      'legend.fontsize': 14,
                      'legend.title_fontsize': 16,
                      'xtick.labelsize': 16,
                      'ytick.labelsize': 16,
                      'axes.labelsize': 18,
                      'axes.titlesize': 22,
                      'figure.dpi': 200,
                      }
        matplotlib.rcParams.update(self.my_theme)

        
        
    def plot_main_importances(self, df, out_plot_file):
        
        print(df)
        # df = pd.read_csv(path_importances)
        # df.index = df[df.columns[0]]
        
        # rename all the features that has perc has name
        percs = [str(i) for i in df.index if i.startswith('perc')]
        new_perc_names = ['neigh_veg_' + i[-2:] for i in percs]
        # bug correction if perc is already aggregated
        try:
            if new_perc_names[0] == 'neigh_veg_rc':
                new_perc_names[0] = 'neigh_veg'
        except:
            pass
        renaming_dict = dict(zip(percs, new_perc_names))
        df = df.rename(index = renaming_dict)

        plt.figure(figsize=(14,14)) 
        plt.bar(df.index, list(df[df.columns[0]]), color = '#1a1694')
        plt.xlabel('Classes')
        plt.xticks(rotation=45, rotation_mode="anchor", ha="right")

        plt.ylabel('Importance')
        plt.title('Input variables importance')
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        plt.savefig(out_plot_file, bbox_inches="tight")
        
        
        
    def plot_roc_curve(self, ns_fpr, ns_tpr, fpr, tpr, outfile):
        
        # plot the roc curve for the model
        plt.figure(figsize=(13,13))
        plt.plot(ns_fpr, ns_tpr, label='No Skill')
        plt.plot(fpr, tpr, label='Classification model')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        # show the legend
        plt.legend()
        # save fig
        plt.savefig(outfile, bbox_inches="tight")
        
    def plot_confusion_matrix(self, cm, outfile):
        
        plt.figure(figsize=(13,13))
        names_ = ['no fire', 'fire']
        sns.heatmap(cm.T, annot = True, fmt = ".0%", cmap = "cividis", xticklabels = names_, yticklabels = names_,
                    annot_kws={"size":20})
        plt.xlabel("True label")
        plt.ylabel("Predicted label")    
        plt.title('confusion matrix')
        plt.savefig(outfile, bbox_inches="tight")      

        
    def plot_BA_over_susc(self, susc_arr, ba_dict, outdir, 
                          reference_file, year_of_susc: str, country_name: str, pixel_size: int,
                          single_year = None, _type = 'Susceptibility', colname = 'finaldate'):

        for name, ba_path in ba_dict.items():
            # read fires data and create a year column
            ff_gdf = gpd.read_file(ba_path) 
            print("The number of all fires are: ", len(ff_gdf))
            #create a year column
            ff_gdf[colname] = pd.to_datetime(ff_gdf[colname]).dt.year
            
            # create a folder for histograms
            new_out = os.path.join(outdir, 'validation_plots')
            if not os.path.exists(new_out):
                os.mkdir(new_out)
            
            
            if len(ff_gdf[ff_gdf[colname] == single_year]) == 0 and single_year is not None:
                print(f'No fires in {single_year}, skipping the year')
                
            elif len(ff_gdf[ff_gdf[colname] == single_year]) >= 0:
                
                # outpath
                out1 = os.path.join(new_out, f'{country_name}_{year_of_susc}_susc_5cl_{name}_over_total.png')

                # create a df with the statistics to plot
                df =  df_statistic(susc_arr, ff_gdf, 
                                        col_years = colname, reference_file = reference_file,
                                        single_year = single_year, pixel_size = pixel_size)
                
                

                if df is not None:
                    # plotting
                    x = df.index.tolist()
                    y1 = df['burned_area(ha)']
                    y2 = df['percentage_of_class_burned_area_to_total_burned_area']  #[0:14]
                    x = [str(i) for i in x ]
                    # create the first axis and plot the first set of data
                    fig, ax1 = plt.subplots()

                    ax1.bar(x, y1, width=.7, color = '#5e140e')
                    ax1.set_xlabel(f'{_type} class',size=12)
                    ax1.set_ylabel('Burned area(ha)', color='black',size=12)
                    ax1.tick_params('y', colors='black', labelsize=10)
                    ax1.set_xticklabels(x, rotation=0, rotation_mode="anchor",ha="right")
                    ax1.set_xticks(x)

                    # create the second axis and plot the second set of data
                    ax2 = ax1.twinx()
                    ax2.bar(x, y2, color= '#5e140e', width=.5)
                    ax2.set_ylabel('Percentage WRT total burned area', color='black',size=12)

                    ax2.tick_params('y', colors='black', labelsize=10)

                    # add a title and legend
                    plt.title(f'Distribution of {_type} in burned areas of {country_name} {year_of_susc}', size=12)
                    # show the plot
                    fig.savefig(out1, dpi = 200, bbox_inches ="tight")
                else:
                    logging.info(f'I did not have burned points for{country_name}')    
            else:
                logging.info('smth unexpected occurred')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
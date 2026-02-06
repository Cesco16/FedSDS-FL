import os
import sys
import pickle

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

#for dataset_name in ['mds1','mds2','mds3']:
for dataset_name in ['lgg1_','lgg2_']:
    for s in np.arange(0,20,1):
        seed = str(s)

        syn_df = pd.read_csv(path/to/medical_low_sample_generator/results/sa_data/avg/'+dataset_name+'/vae/seed_'+seed+'_gen_data.csv)
        real_df = pd.read_csv(path/to/medical_low_sample_generator/data/processed_data/sa_data/'+dataset_name+'/preprocessed_data_all.csv)
        
        min_n = min(syn_df.shape[0], real_df.shape[0])
        x_real = real_df.values[0:min_n]
        x_gen = syn_df.values[0: min_n]

        distances = {}

# 2. Take real data and compare each sample with the rest checking distances
# Obtain the minimum distance from each point to all the real data (one to all)
      
        real_dists = np.ones((x_real.shape[0], x_real.shape[0])) * np.inf
        for i in range(x_real.shape[0]):  # Note: this may not be the most efficient implementation...
            for j in range(x_real.shape[0]):
                if i != j:
                    real_dists[i, j] = np.linalg.norm(x_real[i] - x_real[j])
# Save distances
        distances['real'] = real_dists

# 3. Take synthetic data and compare each sample with the rest real samples checking distances
# Obtain the minimum distance from each point to all the real data (one to all)
      
        gen_dists = np.ones((x_gen.shape[0], x_real.shape[0])) * np.inf
        for i in range(x_gen.shape[0]):  # Note: this may not be the most efficient implementation...
            for j in range(x_real.shape[0]):
                if i != j:
                    gen_dists[i, j] = np.linalg.norm(x_gen[i] - x_real[j])
# Save distances
        distances['gen'] = gen_dists

# 4. Obtain minimum distances (one to one)
        min_real_dists = np.min(distances['real'], axis=0)
        min_gen_dists = np.min(distances['gen'], axis=0)
        # 4. Generate histograms with minimum distances for each one
        fig, ax = plt.subplots(figsize=(7, 6))
        plt.tight_layout()
        min_real_dists_df = pd.DataFrame(min_real_dists)
        min_real_dists_df.plot(ax=ax, kind='hist', density=True, alpha=0.8, color='skyblue', bins=300)
        min_real_dists_df.plot(ax=ax, kind='kde', color='blue', bw_method=0.3)
        min_gen_dists_df = pd.DataFrame(min_gen_dists)
        min_gen_dists_df.plot(ax=ax, kind='hist', density=True, alpha=0.3, color='red', bins=300)
        min_gen_dists_df.plot(ax=ax, kind='kde', color='purple', bw_method=0.3)
        ax.legend(labels=['Real', 'Real KDE', 'Synthetic', 'Synthetic KDE'], loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 5)
        ax.set_xlabel('Distances')
        ax.set_ylabel('Density')
        plt.grid(True)
        plt.yticks()
        plt.xticks()
        plt.tight_layout()
        plt.savefig(path/to/distances_pdf_brca_.svg, bbox_inches='tight')
        plt.show()
        plt.close()

        fig, ax = plt.subplots(figsize=(7,6))
        min_real_dists_df = pd.DataFrame(min_real_dists)
        min_gen_dists_df = pd.DataFrame(min_gen_dists)
        ax.hist(min_real_dists_df, bins=3000, density=True, cumulative=True, color='blue', label='Real CDF',
                        linewidth=2, histtype='step', fill=False)
        ax.hist(min_gen_dists_df, bins=3000, density=True, cumulative=True, color='red', label='Synthetic CDF',
                        linewidth=2, histtype='step', fill=False)
        ax.set_xlabel('Distances')
        ax.set_ylabel('Cumulative Probability')
        ax.legend( loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)
        ax.set_xlim(0, 5)
        plt.yticks()
        plt.xticks()
        plt.tight_layout()
        plt.savefig(path/to/distances_cdf_brca_.svg, bbox_inches='tight')
        plt.show()
        plt.close()

# Should never be zero because it would mean that the real sample is the same as the synthetic one
        print('Wilcoxon test between real and synthetic data: ',
        stats.wilcoxon(min_real_dists - min_gen_dists, alternative='less').pvalue)
        print('KS test between real and synthetic data: ',
        stats.kstest(min_real_dists, min_gen_dists, alternative='greater').pvalue)

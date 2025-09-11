from data_manager import DataManager
from gen_model.data_generation.generator import Generator
import pandas as pd
import numpy as np

def get_col_distribution(df):
    col_dist = []
    for i in range(df.shape[1]):
        values = df.iloc[:,i].unique()
        no_nan_values = values[~pd.isnull(values)]
        if no_nan_values.size <= 2:
            col_dist.append(('bernoulli',1))
        else:
            col_dist.append(('gaussian',2))
    col_dist[-2] = ('weibull',2)

    return col_dist


def denorm_data(data, params):
    transformed_df = data.copy()
    for i, x in zip(data.columns, range(0, data.shape[1])):
        loc, scale = params.loc[x]
        col = data.iloc[:, x]
        if scale != 0:
            transformed_df.iloc[:, x] = (col * scale + loc).astype(data.iloc[:, x].dtype)
        else:
            transformed_df.iloc[:, x] = (col + loc).astype(data.iloc[:, x].dtype)

    return transformed_df

# IMPORT THE BEST SEEDS

seed1 = str(pd.read_csv('./../FL/Node1/best_seed1.csv')['seed 1'][0])
seed2 = str(pd.read_csv('./../FL/Node2/best_seed2.csv')['seed 2'][0])
#seed3 = str(pd.read_csv('./../FL/Node3/best_seed3.csv')['seed 3'][0])
seed1 = str(4)
seed2 = str(10)

print('The selected seed are: ', str(seed1), ', ', str(seed2),'\n')


for k,s,b in zip(['lgg1_','lgg2_'],[seed1, seed2],['1','2']): #['0','4','1'] #OR PUT MANUALLY THE SEED YOU WANT
    
    #BIAS THE DATA FOR NODE b
    print(f'Biasing the synthetic datasets for Node {b}')
    #print(k, "\n")
    ##synth_data = pd.read_csv('./../results/sa_data/big_data/'+k+'/vae/seed_'+s+'_gen_data.csv')
    
    real_data = pd.read_csv('./../data/processed_data/sa_data/'+k+'/preprocessed_data_all.csv') #upload real data in node j
    parameters = pd.read_csv('./../results/sa_data/avg/'+k+'/vae/best_parameters.csv') #upload best VAE-bgm parameters for node j

    ##s = str(parameters['seed'].values[0])
    
    synth_data1 = pd.read_csv(f'./../results/sa_data/avg/lgg1_/vae/seed_{seed1}_gen_data.csv') #best synthetic data node 1 (normalized)
    synth_data2 = pd.read_csv(f'./../results/sa_data/avg/lgg2_/vae/seed_{seed2}_gen_data.csv') #best synthetic data node 2 (normalized)
    #synth_data3 = pd.read_csv(f'./../results/sa_data/avg/lgg3_/vae/seed_{seed3}_gen_data.csv') #best synthetic data node 3 (normalized)
    #print(synth_data1['Age'])
    #print(synth_data2['Age']) 
    #print(synth_data3['Age'])     
    #DENORMALIZE EACH SYNTHETIC DATASET USING THEIR NORMALIZATION PARAMETERS

    #params1 = pd.read_csv('./../data/processed_data/sa_data/mds1/norm_params.csv')
    #params2 = pd.read_csv('./../data/processed_data/sa_data/mds2/norm_params.csv')
    #params3 = pd.read_csv('./../data/processed_data/sa_data/mds3/norm_params.csv')

    #transform_data1 = denorm_data(synth_data1, params1)
    ##transform_data2 = denorm_data(synth_data2, params2)
    #transform_data3 = denorm_data(synth_data3, params3)

    #UPLOAD DE-NORMALIZED BEST SYNTHETIC DATASETS

    transform_data1 = pd.read_csv(f'./../data/denorm_data/sa_data/genlgg1_/seed_{seed1}_lgg1_.csv', index_col=0)  # best synthetic data node 1 (normalized)
    transform_data2 = pd.read_csv(f'./../data/denorm_data/sa_data/genlgg2_/seed_{seed2}_lgg2_.csv', index_col=0)  # best synthetic data node 2 (normalized)
    #transform_data3 = pd.read_csv(f'./../data/denorm_data/sa_data/genlgg3_/seed_{seed3}_lgg3_.csv', index_col=0)  # best synthetic data node 3 (normalized)
    
    transform_data1 = transform_data1[transform_data1['time']>0]
    transform_data2 = transform_data2[transform_data2['time']>0]
    #transform_data3 = transform_data3[transform_data3['time']>0]
    
        
    #SEND AND SAVE THE 3 UNBIASED DENORMALIZED SYNTHETIC DATASETS TO EACH NODE b
        
    transform_data1.to_csv(f'./../FL/Node{b}/synth_lgg1__unbiased_denorm.csv', index=False)
    transform_data2.to_csv(f'./../FL/Node{b}/synth_lgg2__unbiased_denorm.csv', index=False)
    #transform_data3.to_csv(f'./../FL/Node{b}/synth_lgg3__unbiased_denorm.csv', index=False)
        
    #NOW I NORMALIZE AGAIN THE THREE SYNTHETIC DATASETS IN EACH NODE b
    #DISTR = []
    #data_manager1 = DataManager('synth1', transform_data1, transform_data1.copy())
    
    #feat_distributions = []
    #for i in range(transform_data1.shape[1]):
    #    values = transform_data1.iloc[:,i].unique()
    #    no_nan_values = values[~pd.isnull(values)]
    #    #print(no_nan_values)
    #    if no_nan_values.size <= 2 and np.all(np.sort(no_nan_values).astype(int) ==
    #                                          np.array(range(no_nan_values.min().astype(int),
    #                                                         no_nan_values.min().astype(int) + len(no_nan_values)))):
    #        feat_distributions.append(('bernoulli',1))
    #    elif np.amin(np.equal(np.mod(no_nan_values, 1), 0)):
    #        # Check if values are floats but don't have decimals and transform to int. They are floats because of NaNs
    #        if no_nan_values.dtype == 'float64':
    #            no_nan_values = no_nan_values.astype(int)
    #        if np.unique(no_nan_values).size < 50:
    #          if np.amin(no_nan_values) == 0:
    #            feat_distributions.append(('categorical', (np.max(no_nan_values) + 1).astype(int)))
    #          elif np.amin(no_nan_values) > 0:
    #            feat_distributions.append(('categorical', (np.max(no_nan_values)).astype(int)))#else:
    #        else:
    #            feat_distributions.append(('gaussian', 2))
    #    else:
    #        feat_distributions.append(('gaussian', 2))
    #    #print(feat_distributions[-1])
    #data_manager1.set_feat_distributions(feat_distributions)
    #DISTR.append(feat_distributions)
    
    #data_manager1.norm_df, norm_params1 = data_manager1.transform_data(transform_data1)
    #data_manager1.imp_norm_df = data_manager1.impute_data(data_manager1.norm_df)
    #data_manager1.imp_norm_df.to_csv(f'./../FL/Node{b}/synth_lgg1__unbiased_norm.csv', index=False)  # save pre-processed data
    #pd.DataFrame(norm_params1).to_csv(f'./../FL/Node{b}/norm_params1.csv', index=False)  # save scale and local parameters for de-normalizing

    #data_manager2 = DataManager('synth2', transform_data2, transform_data2.copy())
    
    #feat_distributions = []
    #for i in range(transform_data2.shape[1]):
    #    values = transform_data2.iloc[:,i].unique()
    #    no_nan_values = values[~pd.isnull(values)]
    #    #print(no_nan_values)
    #    if no_nan_values.size <= 2 and np.all(np.sort(no_nan_values).astype(int) ==
    #                                          np.array(range(no_nan_values.min().astype(int),
    #                                                         no_nan_values.min().astype(int) + len(no_nan_values)))):
    #        feat_distributions.append(('bernoulli',1))
    #    elif np.amin(np.equal(np.mod(no_nan_values, 1), 0)):
    #        # Check if values are floats but don't have decimals and transform to int. They are floats because of NaNs
    #        if no_nan_values.dtype == 'float64':
    #            no_nan_values = no_nan_values.astype(int)
    #        if np.unique(no_nan_values).size < 50:
    #          if np.amin(no_nan_values) == 0:
    #            feat_distributions.append(('categorical', (np.max(no_nan_values) + 1).astype(int)))
    #          elif np.amin(no_nan_values) > 0:
    #            feat_distributions.append(('categorical', (np.max(no_nan_values)).astype(int)))#else:
    #        else:
    #            feat_distributions.append(('gaussian', 2))
    #    else:
    #        feat_distributions.append(('gaussian', 2))
    #    #print(feat_distributions[-1])
    #data_manager2.set_feat_distributions(feat_distributions)
    #DISTR.append(feat_distributions)
    
    #data_manager2.norm_df, norm_params2 = data_manager2.transform_data(transform_data2)
    #data_manager2.imp_norm_df = data_manager2.impute_data(data_manager2.norm_df)
    #data_manager2.imp_norm_df.to_csv(f'./../FL/Node{b}/synth_lgg2__unbiased_norm.csv',index=False)  # save pre-processed data
    #pd.DataFrame(norm_params2).to_csv(f'./../FL/Node{b}/norm_params2.csv',index=False)  # save scale and local parameters for de-normalizing

    #data_manager3 = DataManager('synth3', transform_data3, transform_data3.copy())
    
    #feat_distributions = []
    #for i in range(transform_data3.shape[1]):
    #    values = transform_data3.iloc[:,i].unique()
    #    no_nan_values = values[~pd.isnull(values)]
    #    #print(no_nan_values)
    #    if no_nan_values.size <= 2 and np.all(np.sort(no_nan_values).astype(int) ==
    #                                          np.array(range(no_nan_values.min().astype(int),
    #                                                         no_nan_values.min().astype(int) + len(no_nan_values)))):
    #        feat_distributions.append(('bernoulli',1))
    #    elif np.amin(np.equal(np.mod(no_nan_values, 1), 0)):
    #        # Check if values are floats but don't have decimals and transform to int. They are floats because of NaNs
    #        if no_nan_values.dtype == 'float64':
    #            no_nan_values = no_nan_values.astype(int)
    #        if np.unique(no_nan_values).size < 50:
    #          if np.amin(no_nan_values) == 0:
    #            feat_distributions.append(('categorical', (np.max(no_nan_values) + 1).astype(int)))
    #          elif np.amin(no_nan_values) > 0:
    #            feat_distributions.append(('categorical', (np.max(no_nan_values)).astype(int)))#else:
    #        else:
    #            feat_distributions.append(('gaussian', 2))
    #    else:
    #        feat_distributions.append(('gaussian', 2))
    #    #print(feat_distributions[-1])
    #data_manager3.set_feat_distributions(feat_distributions)
    #DISTR.append(feat_distributions)
    
    #data_manager3.norm_df, norm_params3 = data_manager3.transform_data(transform_data3)
    #data_manager3.imp_norm_df = data_manager3.impute_data(data_manager3.norm_df)
    #data_manager3.imp_norm_df.to_csv(f'./../FL/Node{b}/synth_lgg3__unbiased_norm.csv',index=False)  # save pre-processed data
    #pd.DataFrame(norm_params3).to_csv(f'./../FL/Node{b}/norm_params3.csv',index=False)  # save scale and local parameters for de-normalizing
        
        #synth_data = pd.concat((synth_data1, synth_data2), axis=0)
        #print(synth_data.values.shape)
        #synth_data = pd.concat((synth_data, synth_data3), axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
        
    #UPLOAD THE 3 UNBIASED NORMALIZED SYNTHETIC DATASETS
        
    synth_data1 = pd.read_csv(f'./../FL/Node{b}/synth_lgg1__unbiased_norm.csv', index_col=False)#('./../data/processed_data/sa_data/mm1/preprocessed_data_all.csv')#
    synth_data2 = pd.read_csv(f'./../FL/Node{b}/synth_lgg2__unbiased_norm.csv', index_col=False)
    #synth_data3 = pd.read_csv(f'./../FL/Node{b}/synth_lgg3__unbiased_norm.csv', index_col=False)
    #print(synth_data1['Age'])
    #print(synth_data2['Age'])
    #print(synth_data3['Age'])
    #print(synth_data1.shape)
    n_data = 700 #number of samples I want for each biased dataset
    #DISTR = feat_distributions
    #DISTR[0][-2] = ('weibull',2)
    #DISTR[1][-2] = ('weibull',2)
    #DISTR[2][-2] = ('weibull',2)
    
    import pickle

    with open(f"./../data/processed_data/sa_data/{k}/metadata.pkl", "rb") as f:
      metadata = pickle.load(f)

    #print(metadata['feat_distributions'])
    #DISTR = metadata['feat_distributions']
    #DISTR[-2] = ('weibull',2)
    
    params = {
            'latent_dim': int(parameters['params'].values[0].split('_')[0]),
            'hidden_size': int(parameters['params'].values[0].split('_')[1]),
            'input_dim': synth_data1.shape[1], #.values[:,0:-2].shape[1],
            'feat_distributions': metadata['feat_distributions'] #get_col_distribution(synth_data1),#[:,0:-2], DISTR[int(b)-1] #
            }
    print(metadata['feat_distributions']) 
    #print(params,'\n')
    #print(feat_distributions)
    print('seed: ', s)
    
    print('Real data shape: ', real_data.values.shape)#[:,0:-2].shape)
    print('Synthetic data shape: ', synth_data1.values.shape)#[:,0:-2].shape)
        
    # upload VAE weights in node k
    bias_gen = Generator(params=params)
    weights = ('./../results/sa_data/avg/'+k+'/vae/seed_'+s+'')
    bias_gen.load(path=weights)
        
    #print('Real data shape: ', real_data.values.shape)#[:,0:-2].shape)
    #print('Synthetic data shape: ', synth_data1.values.shape)#[:,0:-2].shape)
        
    real_latent = bias_gen.get_latent(input_data=real_data.values)[0]
    
    # GENERO GLI SPAZI LATENTI PER I 3 DATASET CHE MI SERVONO
    synth_latent1 = bias_gen.get_latent(input_data=synth_data1.values)[0]
    synth_latent2 = bias_gen.get_latent(input_data=synth_data2.values)[0]
    #synth_latent3 = bias_gen.get_latent(input_data=synth_data3.values)[0]


    gen_dists1 = np.ones((synth_latent1.shape[0], real_latent.shape[0]))*np.inf
    gen_dists2 = np.ones((synth_latent2.shape[0], real_latent.shape[0]))*np.inf
    #gen_dists3 = np.ones((synth_latent3.shape[0], real_latent.shape[0]))*np.inf
    
    for i in range(synth_latent1.shape[0]):
        for j in range(real_latent.shape[0]):
            gen_dists1[i,j] = np.linalg.norm(synth_latent1[i] - real_latent[j])
    for i in range(synth_latent2.shape[0]):
        for j in range(real_latent.shape[0]):
            gen_dists2[i,j] = np.linalg.norm(synth_latent2[i] - real_latent[j])
    #for i in range(synth_latent3.shape[0]):
    #    for j in range(real_latent.shape[0]):
    #        gen_dists3[i,j] = np.linalg.norm(synth_latent3[i] - real_latent[j])
    
    min_dists1 = np.min(gen_dists1, axis=1)
    min_dists2 = np.min(gen_dists2, axis=1)
    #min_dists3 = np.min(gen_dists3, axis=1)
    print(min_dists1)
    print(min_dists2)
    #print(min_dists3)
    sorted_indices1 = np.argsort(min_dists1)
    sorted_indices2 = np.argsort(min_dists2)
    #sorted_indices3 = np.argsort(min_dists3)

    bias_data1 = synth_data1.iloc[sorted_indices1[:n_data]].reset_index(drop=True)
    bias_data2 = synth_data2.iloc[sorted_indices2[:n_data]].reset_index(drop=True)
    #bias_data3 = synth_data3.iloc[sorted_indices3[:n_data]].reset_index(drop=True)
    print('Biased data shape for ', k, ': ', bias_data1.shape)
    print('Biased data shape for ', k, ': ', bias_data2.shape)
    #print('Biased data shape for ', k, ': ', bias_data3.shape)
        
    # SAVE BIASED NORMALIZED DATA IN NODE b
        
    bias_data1.to_csv(f'./../FL/Node{b}/lgg1__biased_norm_{k}.csv')
    bias_data2.to_csv(f'./../FL/Node{b}/lgg2__biased_norm_{k}.csv')
    #bias_data3.to_csv(f'./../FL/Node{b}/lgg3__biased_norm_{k}.csv')
        
    # DENORMALIZE BIASED DATASETS
    
    print('DENORMALIZING DATASET')
        
    #norm_params1 = pd.read_csv(f'./../FL/Node{b}/norm_params1.csv')
    #norm_params2 = pd.read_csv(f'./../FL/Node{b}/norm_params2.csv')
    #norm_params3 = pd.read_csv(f'./../FL/Node{b}/norm_params3.csv')
    
    norm_params = pd.read_csv('./../data/processed_data/sa_data/'+k+'/norm_params.csv')
    #norm_params2 = pd.read_csv('./../data/processed_data/sa_data/'+k+'/norm_params.csv')
    #norm_params3 = pd.read_csv('./../data/processed_data/sa_data/'+k+'/norm_params.csv')
    #print(bias_data1['Age'])
    #print(bias_data2['Age'])
    #print(bias_data3['Age'])
    print('-> range of Age: ', bias_data1.sort_values(by='age_at_diagnosis')['age_at_diagnosis'][0], ' - ', bias_data1.sort_values(by='age_at_diagnosis')['age_at_diagnosis'][n_data-1])
    print('-> range of Age: ', bias_data2.sort_values(by='age_at_diagnosis')['age_at_diagnosis'][0], ' - ', bias_data2.sort_values(by='age_at_diagnosis')['age_at_diagnosis'][n_data-1])
    #print('-> range of Age: ', bias_data3.sort_values(by='age_at_diagnosis')['age_at_diagnosis'][0], ' - ', bias_data3.sort_values(by='age_at_diagnosis')['age_at_diagnosis'][n_data-1])
    bias_data_denorm1 = denorm_data(bias_data1, norm_params)
    bias_data_denorm2 = denorm_data(bias_data2, norm_params)
    #bias_data_denorm3 = denorm_data(bias_data3, norm_params)
    #print(bias_data_denorm1['Age'])
    #print(bias_data_denorm2['Age'])
    #print(bias_data_denorm3['Age'])
    print('-> range of Age: ', bias_data_denorm1.sort_values(by='age_at_diagnosis')['age_at_diagnosis'][0], ' - ', bias_data_denorm1.sort_values(by='age_at_diagnosis')['age_at_diagnosis'][n_data-1])
    print('-> range of Age: ', bias_data_denorm2.sort_values(by='age_at_diagnosis')['age_at_diagnosis'][0], ' - ', bias_data_denorm2.sort_values(by='age_at_diagnosis')['age_at_diagnosis'][n_data-1])
    #print('-> range of Age: ', bias_data_denorm3.sort_values(by='age_at_diagnosis')['age_at_diagnosis'][0], ' - ', bias_data_denorm3.sort_values(by='age_at_diagnosis')['age_at_diagnosis'][n_data-1])
        
    # SAVE BIASED DE-NORMALIZED DATA IN NODE K
        
    bias_data_denorm1.to_csv(f'./../FL/Node{b}/lgg1__biased_denorm_{k}.csv')
    bias_data_denorm2.to_csv(f'./../FL/Node{b}/lgg2__biased_denorm_{k}.csv')
    #bias_data_denorm3.to_csv(f'./../FL/Node{b}/lgg3__biased_denorm_{k}.csv')

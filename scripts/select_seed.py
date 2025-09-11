from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np

def denorm_data(data, params, length):
    transformed_df = data.copy()
    for i, x in zip(data.columns, range(0, length)):
        loc, scale = params.loc[x]
        col = data.iloc[:, x]
        if scale != 0:
            transformed_df.iloc[:, x] = (col * scale + loc).astype(data.iloc[:, x].dtype)
        else:
            transformed_df.iloc[:, x] = (col + loc).astype(data.iloc[:, x].dtype)

    return transformed_df

# DE-NORMALIZE DATA
sett = 'avg'
for i in ['lgg1_','lgg2_']:#,'lgg2','lgg3']:
    print('DE-NORMALIZING SYNTHETIC DATASETS FROM: ', i)
    params = pd.read_csv('./../data/processed_data/sa_data/'+i+'/norm_params.csv')
    #print(params)
    for j in range(0,20,1):
        data = pd.read_csv('./../results/sa_data/'+sett+'/'+i+'/vae/seed_'+str(j)+'_gen_data.csv')
        ll = data.shape[1]
        transform_data = denorm_data(data, params, length=ll)
        transform_data.to_csv('./../data/denorm_data/sa_data/gen'+i+'/seed_'+str(j)+'_'+i+'.csv')


features= ['Gender', 'AOD', 'Neutrophils', 'Hemoglobin', 'Platelets', 'BMB',
       'ASXL1', 'BCOR', 'BCORL1', 'CBL', 'CEBPA', 'DNMT3A', 'ETV6', 'EZH2',
       'FLT3', 'GATA2', 'GNAS', 'IDH1', 'IDH2', 'JAK2', 'KIT', 'KRAS', 'MPL',
       'NPM1', 'NRAS', 'PTPN11', 'RUNX1', 'SF3B1', 'SRSF2', 'STAG2', 'TET2',
       'TP53', 'U2AF1', 'WT1', 'ZRSR2', 'SETBP1', 'OverallSurvival', 'DeadOrAliveAtTransplantation']

ACC1 = []
ACC2 = []
ACC3 = []
# iterate on the 10 available seed

for i in ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']:

    seed = i
    d1 = pd.read_csv('./../data/raw_data/sa_data/lgg1_/lgg1_.csv', index_col=0)#[features]
    synth_size1 = d1.shape[0]
    n1 = pd.read_csv('./../data/denorm_data/sa_data/genlgg1_/seed_'+seed+'_lgg1_.csv', index_col=0)#.sample(n=synth_size1, random_state=42)
    n1 = n1[n1['time']>0]
    if n1.shape[0] >= synth_size1:
        n1 = n1.sample(n=synth_size1, random_state=42)
    
    d2 = pd.read_csv('./../data/raw_data/sa_data/lgg2_/lgg2_.csv', index_col=0)#[features]
    synth_size2 = d2.shape[0]
    n2 = pd.read_csv('./../data/denorm_data/sa_data/genlgg2_/seed_'+seed+'_lgg2_.csv', index_col=0)
    n2 = n2[n2['time']>0]
    if n2.shape[0] >= synth_size2:
        n2=n2.sample(n=synth_size2, random_state=42)

    l = n1.shape[1]
    n1.insert(l, 'class', synth_size1*['synthetic'])
    d1.insert(l, 'class', d1.shape[0]*['real'])

    DATA1 = pd.concat((n1, d1), axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    X1 = DATA1.iloc[:,:l]
    Y1 = DATA1['class']

    n2.insert(l, 'class', synth_size2*['synthetic'])
    d2.insert(l, 'class', d2.shape[0]*['real'])
    DATA2 = pd.concat((n2, d2), axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    X2 = DATA2.iloc[:,:l]
    Y2 = DATA2['class']

    n3.insert(l, 'class', synth_size3*['synthetic'])
    d3.insert(l, 'class', d3.shape[0]*['real'])
    DATA3 = pd.concat((n3, d3), axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    X3 = DATA3.iloc[:,:l]
    Y3 = DATA3['class']

    xtrain1, xtest1, ytrain1, ytest1 = train_test_split(X1, Y1, test_size=0.35, shuffle=True, random_state = 42, stratify=Y1)
    xtrain2, xtest2, ytrain2, ytest2 = train_test_split(X2, Y2, test_size=0.35, shuffle=True, random_state = 42, stratify=Y2)
    xtrain3, xtest3, ytrain3, ytest3 = train_test_split(X3, Y3, test_size=0.35, shuffle=True, random_state = 42, stratify=Y3)

    clf1 = RandomForestClassifier(max_depth=100, random_state=0)
    clf1.fit(xtrain1, ytrain1)

    clf2 = RandomForestClassifier(max_depth=100, random_state=0)
    clf2.fit(xtrain2, ytrain2)

    clf3 = RandomForestClassifier(max_depth=100, random_state=0)
    clf3.fit(xtrain3, ytrain3)
    
    print('TRAINING THE RANDOM FOREST CLASSIFIERS')

    ypred1 = clf1.predict(xtest1)
    ypred2 = clf2.predict(xtest2)
    ypred3 = clf3.predict(xtest3)

    # Calculate accuracy
    #acc1 = accuracy_score(ytest1, ypred1)
    ##acc2 = accuracy_score(ytest2, ypred2)
    ##acc3 = accuracy_score(ytest3, ypred3)
    acc1 = f1_score(ytest1, ypred1, pos_label='real')
    acc2 = f1_score(ytest2, ypred2, pos_label='real')
    acc3 = f1_score(ytest3, ypred3, pos_label='real')
    print('SEED ',i, ' - ACCURACY MDS1: ',acc1)
    print('SEED ',i, ' - ACCURACY MDS2: ',acc2)
    print('SEED ',i, ' - ACCURACY MDS3: ',acc3, '\n')

    ACC1.append(acc1)
    ACC2.append(acc2)
    ACC3.append(acc3)

# I CHOSE THE SEED FOR WHICH THE ACCURACY IS MINIMIZED
min1 = min(ACC1)
min2 = min(ACC2)
min3 = min(ACC3)

seed1 = ACC1.index(min1)
seed2 = ACC2.index(min2)
seed3 = ACC3.index(min3)

print('BEST SEED NODE1: ', seed1, '\n')
print('BEST SEED NODE2: ', seed2, '\n')
print('BEST SEED NODE3: ', seed3, '\n')

print('SAVING BEST SEEDS')

pd.DataFrame({'seed 1': [seed1]}).to_csv('./../FL/Node1/best_seed1.csv', index=None)
pd.DataFrame({'seed 2': [seed2]}).to_csv('./../FL/Node2/best_seed2.csv', index=None)
pd.DataFrame({'seed 3': [seed3]}).to_csv('./../FL/Node3/best_seed3.csv', index=None)

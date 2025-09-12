## GENERATE HEATMAPS TO COMPARE THE 3 MODES
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
#parser.add_argument('train', type=str, choices=['iso','fed_unbias','fed_bias'], help='training mode')
parser.add_argument('--preproc', type=str, choices=['norm', 'denorm'], help='choice between normalized and denormalized data')

args = parser.parse_args()

#mode = args.train
preproc = args.preproc

#from lifelines import KaplanMeierFitter
import numpy as np
#from lifelines.statistics import logrank_test, multivariate_logrank_test
import pandas as pd

c11 = pd.read_csv('./../FL/Results/c11_'+preproc+'_iso.csv', header=0)['0']
c22 = pd.read_csv('./../FL/Results/c22_'+preproc+'_iso.csv', header=0)['0']
#c33 = pd.read_csv('./../FL/Results/c33_'+preproc+'_iso.csv', header=0)['0']

C11 = pd.read_csv('./../FL/Results/C11_'+preproc+'_fed_bias.csv', header=0)['0']
C22 = pd.read_csv('./../FL/Results/C22_'+preproc+'_fed_bias.csv', header=0)['0']
#C33 = pd.read_csv('./../FL/Results/C33_'+preproc+'_fed_bias.csv', header=0)['0']

G11 = pd.read_csv('./../FL/Results/C11_'+preproc+'_fed_unbias.csv', header=0)['0']
G22 = pd.read_csv('./../FL/Results/C22_'+preproc+'_fed_unbias.csv', header=0)['0']
#G33 = pd.read_csv('./../FL/Results/C33_'+preproc+'_fed_unbias.csv', header=0)['0']

#FEDAVG

C1 = pd.read_csv('./../FL/Results/C1_denorm_fed_avg.csv', header=0)['0']
C2 = pd.read_csv('./../FL/Results/C2_denorm_fed_avg.csv', header=0)['0']
#C3 = pd.read_csv('./../FL/Results/C3_denorm_fed_avg.csv', header=0)['0']



#BOXPLOT
box11 = pd.DataFrame({'C-index':c11, 
                   'mode': ['Isolated']*10,
                  'node_type':['Client 1']*10})
box22 = pd.DataFrame({'C-index':c22, 
                   'mode': ['Isolated']*10,
                  'node_type':['Client 2']*10})
#box33 = pd.DataFrame({'C-index':c33, 
#                   'mode': ['Isolated']*10,
#                  'node_type':['Client 3']*10})

boxf1 = pd.DataFrame({'C-index':C1, 
                   'mode': ['FedAvg']*10,
                  'node_type':['Client 1']*10})
boxf2 = pd.DataFrame({'C-index':C2, 
                   'mode': ['FedAvg']*10,
                  'node_type':['Client 2']*10})
##boxf3 = pd.DataFrame({'C-index':C3, 
##                   'mode': ['FedAvg']*10,
#                 'node_type':['Client 3']*10})

boxu1 = pd.DataFrame({'C-index':G11, 
                   'mode': ['FedSDS - unbiased']*10,
                  'node_type':['Client 1']*10})
boxu2 = pd.DataFrame({'C-index':G22, 
                   'mode': ['FedSDS - unbiased']*10,
                  'node_type':['Client 2']*10})
#boxu3 = pd.DataFrame({'C-index':G33, 
##                   'mode': ['FedSDS - unbiased']*10,
#                  'node_type':['Client 3']*10})

boxb1 = pd.DataFrame({'C-index':C11, 
                   'mode': ['FedSDS - biased']*10,
                  'node_type':['Client 1']*10})
boxb2 = pd.DataFrame({'C-index':C22, 
                   'mode': ['FedSDS - biased']*10,
                  'node_type':['Client 2']*10})
#boxb3 = pd.DataFrame({'C-index':C33, 
#                   'mode': ['FedSDS - biased']*10,
#                  'node_type':['Client 3']*10})


BOX = pd.concat((box11, box22), axis=0)
#BOX = pd.concat((BOX, box33), axis=0)
BOX = pd.concat((BOX, boxf1), axis=0)
BOX = pd.concat((BOX, boxf2), axis=0)
#BOX = pd.concat((BOX, boxf3), axis=0)
BOX = pd.concat((BOX, boxu1), axis=0)
BOX = pd.concat((BOX, boxu2), axis=0)
#BOX = pd.concat((BOX, boxu3), axis=0)
BOX = pd.concat((BOX, boxb1), axis=0)
BOX = pd.concat((BOX, boxb2), axis=0)
#BOX = pd.concat((BOX, boxb3), axis=0)

sns.boxplot(data = BOX, x = 'node_type', y = 'C-index', hue='mode')
plt.title('Isolated vs FedAvg vs FedSDS')
plt.legend(title='Models',loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Client where the model is evaluated')
plt.ylim()#(0.58,0.82)
plt.grid()
plt.savefig('./../FL/Results/boxplot.svg', bbox_inches='tight')

# FACCIO IL TEST SOLO TRA I CORRISPONDENTI
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests

p_values = []
labels = []

alternative = 'greater'

def corrected_test(group1, group2, label):
    #stat, p = mannwhitneyu(round(group1,3), round(group2,3), alternative=alternative
    try:
        stat, p = wilcoxon(round(group1,3), round(group2,3), alternative=alternative)
        #stat, p = wilcoxon(group1, group2, alternative=alternative)
    except:
        p = 1.0
        label = "identity"
    p_values.append(p)
    labels.append(label)
    #print(f"{label}: statistic={stat}, raw p-value={p}")

n1 = [c11, C1, G11, C11]
n2 = [c22, C2, G22, C22]
#n3 = [c33,C3,G33,C33]

medie1 = [np.mean(array) for array in n1]
medie2 = [np.mean(array) for array in n2]
#medie3 = [np.mean(array) for array in n3]


indice1= np.argmax(medie1)
indice2= np.argmax(medie2)
#indice3= np.argmax(medie3)


# Seleziona l'array con la media più alta
max1 = n1[indice1]
max2 = n2[indice2]
#max3 = n3[indice3]

# Run your tests (abbreviated for clarity; use all 30 like before)
corrected_test(max1, c11, "f11 vs i11")
corrected_test(max2, c22, "f22 vs i22")

corrected_test(max1, C1, "f11 vs avg11")
corrected_test(max2, C2, "f11 vs avg21")

corrected_test(max1, G11, "f11 vs g11")
corrected_test(max2, G22, "f11 vs g21")

corrected_test(max1, C11, "f11 vs f11")
corrected_test(max2, C22, "f11 vs f22")


# Apply Holm-Bonferroni correction (FWER control)
reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='holm')

print(f"\nHolm-Bonferroni correction applied (FWER control).")
for i in range(len(p_values)):
    print(f"{labels[i]}: corrected p-value = {pvals_corrected[i]}, significant: {reject[i]}")
    
    
medie = [
    np.mean(c11),  #modello1 su set 1, 2 e 3
     np.mean(c22), #modello2 su set 1, 2 e 3
 #modello3 su set 1, 2 e 3
    np.mean(C1),  np.mean(C2),
    np.mean(G11),
    np.mean(G22),

    np.mean(C11),
    np.mean(C22),

]

# Organizza le medie in una matrice 5x5
matrice_medie = np.array(medie).reshape(4, 2).T

# Crea le etichette per gli assi
etichette_y1 = [f'ISO_{i}' for i in range(1, 4)]
etichette_y2 = ['FEDAVG']
etichette_y3 = [f'BIAS_{i}' for i in range(1, 4)]
etichette_y4 = [f'UNB_{i}' for i in range(1, 4)]
etichette_x = etichette_y1 +  etichette_y2 + etichette_y4 + etichette_y3
etichette_y = [f'val{i}' for i in range(1, 4)]

pp = [
    pvals_corrected[0], 
    pvals_corrected[1], 
    pvals_corrected[2], 
    pvals_corrected[3], pvals_corrected[4], pvals_corrected[5],
    pvals_corrected[6], 
    pvals_corrected[7], 
    #pvals_corrected[8], 
    #pvals_corrected[9],  
    #pvals_corrected[10], 
    #pvals_corrected[11], 
]
# Organizza le medie in una matrice 5x5
matrice_pp = np.array(pp).reshape(4, 2).T

# Crea le etichette per gli assi
etichette_y1 = ['ISO']
etichette_y2 = ['FEDAVG']
etichette_y3 = ['BIAS']
etichette_y4 = ['UNBIAS']
etichette_x = etichette_y1 +  etichette_y2 + etichette_y4 + etichette_y3
etichette_y = [f'val{i}' for i in range(1, 4)]

# Crea la heatmap
plt.figure(figsize=(16, 6))
#sns.heatmap(matrice_medie, annot=True, fmt=".2f", cmap="viridis", xticklabels=etichette_x, yticklabels=etichette_y)
plt.subplot(1, 2, 1)
plt.imshow(matrice_medie, cmap="viridis", aspect='auto')

# Aggiungi annotazioni
for i in range(matrice_medie.shape[0]):
    for j in range(matrice_medie.shape[1]):
        text = plt.text(j, i, f"{matrice_medie[i, j]:.2f}",
                        ha="center", va="center", color="w")

# Imposta le etichette degli assi
plt.xticks(np.arange(len(etichette_x)), etichette_x)
plt.yticks(np.arange(len(etichette_y)), etichette_y)


# Aggiungi il titolo
plt.title('Isolated vs FedSDS training performance')

# Aggiungi la scritta "c-index" sopra la barra dei colori
cbar = plt.gcf().axes[-1]
cbar.set_title('Isolated vs FedSS')
#for x in [3, 4, 7]:  # Posizioni delle linee verticali
#    plt.axvline(x=x - 0.5, color='red', linestyle='--', linewidth=2)


plt.subplot(1, 2, 2)
plt.imshow(matrice_pp, cmap="viridis", aspect='auto')

# Aggiungi annotazioni
for i in range(matrice_pp.shape[0]):
    for j in range(matrice_pp.shape[1]):
        text = plt.text(j, i, f"{matrice_pp[i, j]:.4f}",
                        ha="center", va="center", color="w")

# Imposta le etichette degli assi
plt.xticks(np.arange(len(etichette_x)), etichette_x)
plt.yticks(np.arange(len(etichette_y)), etichette_y)

# Aggiungi il titolo
plt.title('Isolated vs FedSDS training performance')

# Aggiungi la scritta "c-index" sopra la barra dei colori
cbar = plt.gcf().axes[-1]
cbar.set_title('p-values')
#for x in [3, 4, 7]:  # Posizioni delle linee verticali
#    plt.axvline(x=x - 0.5, color='red', linestyle='--', linewidth=2)

# Mostra il plot
plt.savefig('/home/PERSONALE/francesco.casadei20/LGG_FL_ok/FL/Results/heatmap_corr_ok.svg',bbox_inches='tight')
plt.show()

import numpy as np
from scipy import stats

for dd in [c11, c12, c13, c21, c22, c23, c31, c32, c33, C1, C2, C3, G11, G12, G13, G21, G22, G23, G31, G32, G33, C11, C12, C13, C21, C22, C23, C31, C32, C33]:
    print('mean: ', np.mean(dd), ' confidence interval: ', stats.t.interval(0.95, len(dd)-1, loc=np.mean(dd), scale=stats.sem(dd)))
    
## PLOT THE SURVIVAL CURVES FOR EACH CLIENT

features = ['Gender', 'AOD', 'Neutrophils', 'Hemoglobin', 'Platelets', 'BMB',
            'ASXL1', 'BCOR', 'BCORL1', 'CBL', 'CEBPA', 'DNMT3A', 'ETV6', 'EZH2',
            'FLT3', 'GATA2', 'GNAS', 'IDH1', 'IDH2', 'JAK2', 'KIT', 'KRAS', 'MPL',
            'NPM1', 'NRAS', 'PTPN11', 'RUNX1', 'SF3B1', 'SRSF2', 'STAG2', 'TET2',
            'TP53', 'U2AF1', 'WT1', 'ZRSR2', 'SETBP1', 'time', 'event']

cols_standardize = ['AOD', 'Neutrophils', 'Hemoglobin', 'Platelets', 'BMB']

path = '/home/PERSONALE/francesco.casadei20/LGG/FL/'

g1 = pd.read_csv(path + '/Node1/synth_lgg1_unbiased_denorm.csv', index_col=0).sample(n=300,
                                                                                     random_state=42).reset_index(
    drop=True)
g2 = pd.read_csv(path + '/Node2/synth_lgg2_unbiased_denorm.csv', index_col=0).sample(n=300,
                                                                                     random_state=42).reset_index(
    drop=True)
g3 = pd.read_csv(path + '/Node3/synth_lgg3_unbiased_denorm.csv', index_col=0).sample(n=300,
                                                                                     random_state=42).reset_index(
    drop=True)

G = pd.concat((g1, g2), axis=0)
G = pd.concat((G, g3), axis=0)

# WITH DENORMALIZED DATA
s11 = pd.read_csv(path + '/Node1/lgg1_biased_denorm_lgg1.csv', index_col=0)
s12 = pd.read_csv(path + '/Node1/lgg2_biased_denorm_lgg1.csv', index_col=0)
s13 = pd.read_csv(path + '/Node1/lgg3_biased_denorm_lgg1.csv', index_col=0)
s21 = pd.read_csv(path + '/Node2/lgg1_biased_denorm_lgg2.csv', index_col=0)
s22 = pd.read_csv(path + '/Node2/lgg2_biased_denorm_lgg2.csv', index_col=0)
s23 = pd.read_csv(path + '/Node2/lgg3_biased_denorm_lgg2.csv', index_col=0)
s31 = pd.read_csv(path + '/Node3/lgg1_biased_denorm_lgg3.csv', index_col=0)
s32 = pd.read_csv(path + '/Node3/lgg2_biased_denorm_lgg3.csv', index_col=0)
s33 = pd.read_csv(path + '/Node3/lgg3_biased_denorm_lgg3.csv', index_col=0)

S1 = pd.concat((s11, s12), axis=0)
S1 = pd.concat((S1, s13), axis=0)

S2 = pd.concat((s21, s22), axis=0)
S2 = pd.concat((S2, s23), axis=0)

S3 = pd.concat((s31, s32), axis=0)
S3 = pd.concat((S3, s33), axis=0)

r1 = pd.read_csv(path + '/Node1/lgg1.csv')
r2 = pd.read_csv(path + '/Node2/lgg2.csv')
r3 = pd.read_csv(path + '/Node3/lgg3.csv')

t1 = pd.read_csv(path + '/Node1/lggtest1.csv')
t2 = pd.read_csv(path + '/Node2/lggtest2.csv')
t3 = pd.read_csv(path + '/Node3/lggtest3.csv')

r1.rename(columns={'OverallSurvival': 'time', 'DeadOrAliveAtTransplantation': 'event', }, inplace=True)  # [features]
r2.rename(columns={'OverallSurvival': 'time', 'DeadOrAliveAtTransplantation': 'event', }, inplace=True)  # [features]
r3.rename(columns={'OverallSurvival': 'time', 'DeadOrAliveAtTransplantation': 'event', }, inplace=True)  # [features]
t1.rename(columns={'OverallSurvival': 'time', 'DeadOrAliveAtTransplantation': 'event', }, inplace=True)  # [features]
t2.rename(columns={'OverallSurvival': 'time', 'DeadOrAliveAtTransplantation': 'event', }, inplace=True)  # [features]
t3.rename(columns={'OverallSurvival': 'time', 'DeadOrAliveAtTransplantation': 'event', }, inplace=True)  # [features]

r1 = r1[features]
r2 = r2[features]
r3 = r3[features]
t1 = t1[features]
t2 = t2[features]
t3 = t3[features]

d11 = pd.DataFrame({'time': r1['time'], 'event': r1['event'], 'group': np.full(len(r1), '1')})
d12 = pd.DataFrame({'time': G['time'], 'event': G['event'], 'group': np.full(len(G), '2')})
d13 = pd.DataFrame({'time': S1['time'], 'event': S1['event'], 'group': np.full(len(S1), '3')})

data = pd.concat((d11, d12))
data = pd.concat((data, d13))

# Passo 2: Adatta il modello Kaplan-Meier
kmf1 = KaplanMeierFitter()
kmf1.fit(durations=r1['time'], event_observed=r1['event']),
kmf2 = KaplanMeierFitter()
kmf2.fit(durations=G['time'], event_observed=G['event']),
kmf3 = KaplanMeierFitter()
kmf3.fit(durations=S1['time'], event_observed=S1['event']),

colors = sns.color_palette("Set1", n_colors=num_clients)

# Passo 3: Traccia la curva di Kaplan-Meier
kmf1.plot_survival_function(title='KM curves - 3 clients', label='real', color=colors[0])
kmf2.plot_survival_function(title='KM curves - 3 clients', label='unbiased', color=colors[1])
kmf3.plot_survival_function(title='KM curves - 3 clients', label='biased', color=colors[2])
plt.grid()
plt.xlabel('Time')
plt.ylabel('Survival probability')
plt.title('KM curves - node 1')
plt.savefig('./../FL/Node1/km_curves_node1.svg', bbox_inches='tight')
#plt.show()

d11 = pd.DataFrame({'time': r2['time'], 'event': r2['event'], 'group': np.full(len(r2), '1')})
d12 = pd.DataFrame({'time': G['time'], 'event': G['event'], 'group': np.full(len(G), '2')})
d13 = pd.DataFrame({'time': S2['time'], 'event': S2['event'], 'group': np.full(len(S2), '3')})

data = pd.concat((d11, d12))
data = pd.concat((data, d13))

# Passo 2: Adatta il modello Kaplan-Meier
kmf4 = KaplanMeierFitter()
kmf4.fit(durations=r2['time'], event_observed=r2['event']),
kmf5 = KaplanMeierFitter()
kmf5.fit(durations=G['time'], event_observed=G['event']),
kmf6 = KaplanMeierFitter()
kmf6.fit(durations=S2['time'], event_observed=S2['event']),

colors = sns.color_palette("Set1", n_colors=num_clients)

# Passo 3: Traccia la curva di Kaplan-Meier
kmf4.plot_survival_function(title='KM curves - 3 clients', label='real', color=colors[0])
kmf5.plot_survival_function(title='KM curves - 3 clients', label='unbiased', color=colors[1])
kmf6.plot_survival_function(title='KM curves - 3 clients', label='biased', color=colors[2])
plt.grid()
plt.xlabel('Time')
plt.ylabel('Survival probability')
plt.title('KM curves - node 2')
plt.savefig('./../FL/Node2/km_curves_node2.svg', bbox_inches='tight')
#plt.show()

d11 = pd.DataFrame({'time': r3['time'], 'event': r3['event'], 'group': np.full(len(r3), '1')})
d12 = pd.DataFrame({'time': G['time'], 'event': G['event'], 'group': np.full(len(G), '2')})
d13 = pd.DataFrame({'time': S3['time'], 'event': S3['event'], 'group': np.full(len(S3), '3')})

data = pd.concat((d11, d12))
data = pd.concat((data, d13))

# Passo 2: Adatta il modello Kaplan-Meier
kmf7 = KaplanMeierFitter()
kmf7.fit(durations=r3['time'], event_observed=r3['event']),
kmf8 = KaplanMeierFitter()
kmf8.fit(durations=G['time'], event_observed=G['event']),
kmf9 = KaplanMeierFitter()
kmf9.fit(durations=S3['time'], event_observed=S3['event']),

colors = sns.color_palette("Set1", n_colors=num_clients)

# Passo 3: Traccia la curva di Kaplan-Meier
kmf7.plot_survival_function(title='KM curves - 3 clients', label='real', color=colors[0])
kmf8.plot_survival_function(title='KM curves - 3 clients', label='unbiased', color=colors[1])
kmf9.plot_survival_function(title='KM curves - 3 clients', label='biased', color=colors[2])
plt.grid()
plt.xlabel('Time')
plt.ylabel('Survival probability')
plt.title('KM curves - node 3')
plt.savefig('./../FL/Node3/km_curves_node3.svg', bbox_inches='tight')
#plt.show()

# PLOT KM CURVES ALL

colors = sns.color_palette("Set1", n_colors=7)

# Passo 3: Traccia la curva di Kaplan-Meier
kmf1.plot_survival_function(title='KM curves - 3 clients', label='real-node1', color = colors[0])
kmf2.plot_survival_function(title='KM curves - 3 clients', label='unbiased', color = colors[1])
kmf3.plot_survival_function(title='KM curves - 3 clients', label='biased-node1', color = colors[2])
kmf4.plot_survival_function(title='KM curves - 3 clients', label='real-node2', color = colors[3])
kmf6.plot_survival_function(title='KM curves - 3 clients', label='biased-node2', color = colors[4])
kmf7.plot_survival_function(title='KM curves - 3 clients', label='real-node3', color = colors[5])
kmf9.plot_survival_function(title='KM curves - 3 clients', label='biased-node3', color = colors[6])
plt.grid()
plt.xlabel('Time')
plt.ylabel('Survival probability')
plt.title('KM curves')
plt.savefig('./../FL/Results/km_curves_all.svg', bbox_inches='tight')
#plt.show()

import copy
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import scipy as sc
import json
import yaml 

file_path = 'saida.csv'

og_df = pd.read_csv("experimento.csv")
#sm_df = pd.read_csv("saida.csv")

def avaliar(ind):
    if os.path.exists(file_path):
        os.remove(file_path)
    
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)    

    if os.path.exists('config.yml'):
        os.remove('config.yml')

    with open('config.yml',"x") as tst:
        config['simulation'] = ind
        yaml.dump(config, tst, default_flow_style=True)

    xyz = 0.0
    os.system("cargo run -- --config config.yml --headless")
    if os.path.exists(file_path):
        sm_df = pd.read_csv("saida.csv")

        concat_experimento = pd.concat([og_df['A1'], og_df['A2'], og_df['A3'],og_df['A4'], og_df['A5'],
                                        og_df['A6'], og_df['A7'], og_df['A8'],og_df['A9'], og_df['A10'],
                                        og_df['A11'], og_df['A12'], og_df['A13'],og_df['A14'], og_df['A15']], ignore_index=True)

        concat_simulacao = pd.concat([sm_df['A1'], sm_df['A2'], sm_df['A3'], sm_df['A4'], sm_df['A5'],
                                        sm_df['A6'], sm_df['A7'], sm_df['A8'], sm_df['A9'], sm_df['A10'],
                                        sm_df['A11'], sm_df['A12'], sm_df['A13'],sm_df['A14'], sm_df['A15']], ignore_index=True)
        a = sc.stats.pearsonr(concat_experimento,concat_simulacao).statistic
        return a
    else:
        return 0

def mutar(individuo, taxa_mutacao=0.3, intensidade_mutacao=0.0050):
    novo = copy.deepcopy(individuo)
    for chave, valor in novo.items():
        if random.random() < taxa_mutacao:
            if isinstance(valor, float):
                ruido = random.uniform(-intensidade_mutacao, intensidade_mutacao)
                novo[chave] = float(np.nan_to_num(max(valor + ruido, 0)))
            else:
                novo[chave] = valor
    return novo

def crossover(pai1, pai2):
    filho = {}
    for chave in pai1.keys():
        if isinstance(pai1[chave], float):
            filho[chave] = float((pai1[chave] + pai2[chave]) / 2)
        else:
            filho[chave] = pai1[chave]

    return filho

def selecionar(populacao, fitnesses, n):
    # Seleção por ranking dos melhores
    ordenados = sorted(zip(populacao, fitnesses), key=lambda x: x[1], reverse=True)
    selecionados = [ind for ind, fit in ordenados[:n]]
    return selecionados

def algoritmo_genetico(base_individuo, tamanho_pop=5, geracoes=10):
    populacao = [mutar(base_individuo, taxa_mutacao=1.0, intensidade_mutacao=0.5) for _ in range(tamanho_pop)]
    df_grafico = pd.DataFrame(data = {"geracao": [0], "avaliacao": [0.5]}, columns=["geracao","avaliacao"])

    melhor_ger = 0
    for geracao in range(geracoes):
        fitnesses = [avaliar(ind) for ind in populacao]
        melhores = selecionar(populacao, fitnesses, n=2)

        # Nova população: elite + filhos cruzados
        nova_populacao = copy.deepcopy(melhores)
        while len(nova_populacao) < tamanho_pop:
            pai1, pai2 = random.sample(melhores, 2)
            filho = crossover(pai1, pai2)
            filho = mutar(filho, taxa_mutacao=0.5, intensidade_mutacao=0.05)
            nova_populacao.append(filho)

        populacao = nova_populacao

        melhor_fitness = max(fitnesses)
        print(f'Geração {geracao + 1} - Melhor fitness: {melhor_fitness:.4f}')
        df_grafico_aux = pd.DataFrame(data = {"geracao": [geracao + 1], "avaliacao": [melhor_fitness]}, columns=["geracao","avaliacao"])
        df_grafico = pd.concat([df_grafico,df_grafico_aux])

    # Melhor solução final
    fitnesses = [avaliar(ind) for ind in populacao]
    melhor_individuo = populacao[fitnesses.index(max(fitnesses))]
    plt.plot(df_grafico['geracao'], df_grafico['avaliacao'], color='red', alpha=0.5, linewidth=0.9)
    plt.ylim(0.5, 1)
    plt.savefig('figure.png')
    return melhor_individuo

#with open('./assets/config.json',"r") as tst:
#    parametros = json.load(tst)
with open('config.yml', 'r') as f:
    parametros = yaml.safe_load(f)

#for variavel, valor in parametros.items():
#    print(f'{variavel} = {valor}')

melhor = algoritmo_genetico(parametros['simulation'])
if os.path.exists('configDef.yml'):
        os.remove('configDef.yml')

print("\nMelhor solução encontrada:")
for variavel, valor in melhor.items():
    print(f'{variavel} = {valor}')


with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)    
with open('configDef.yml', 'x') as f:
    config['simulation'] = melhor
    yaml.dump(config, f, default_flow_style=True)
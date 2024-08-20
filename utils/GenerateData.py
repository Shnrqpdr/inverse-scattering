import numpy as np
import scipy
import scipy.special as sc
import pandas as pd

def coefCircularBarrier(sigma, l, gamma, r, k):
        coef = (2*np.pi*r*gamma*sigma*(sc.jv(l, k*r)**2))/(1 - 2*np.pi*r*gamma*sigma*(sc.jv(l, k*r))*(sc.hankel1(l, k*r)))

        return coef

def circularBarrierDataset(
                    R_min, 
                    R_max, 
                    points_R, 
                    gamma_min, 
                    gamma_max, 
                    points_gamma, 
                    n_min, 
                    n_max,
                    k_min, 
                    k_max, 
                    points_k,
                    path,
                    m = 1.0,
                    hbar = 1.0):

    M = m
    HBAR = hbar
    i = 0 + 1j

    R = np.linspace(R_min, R_max, points_R)
    gamma = np.linspace(gamma_min, gamma_max, points_gamma)

    SIGMA = (-i/4.0)*(2*M/(HBAR**2))

    k = np.linspace(k_min, k_max, points_k)
    delta_k = (k_max - k_min)/points_k

    n_array = np.arange(n_min, n_max + 1, 1)

    list_gamma = list(gamma)
    inputs_array = np.zeros((1, points_k + 9))

    for g in list_gamma:
        # index = list_gamma.index(g)
        for r in R:
            l_array = np.array([])

            for k_value in k:
                soma = 0.0

                for n in n_array:
                    soma = soma + np.real(coefCircularBarrier(SIGMA, n, g, r, k_value))
                    
                it = (-4/k_value)*soma
                l_array = np.append(l_array, it)

            row_array = np.array([])
            row_array = np.append(row_array, [M, HBAR, k_min, k_max, delta_k, n_min, n_max, g, r])
            row_array = np.append(row_array, l_array)

            inputs_array = np.vstack([inputs_array, row_array])

    inputs_array = np.delete(inputs_array, (0), axis=0)
    inputs_array.shape

    # Nomes das primeiras 9 colunas
    initial_columns = ['M', 'HBAR', 'k_min', 'k_max', 'delta_k', 'n_min', 'n_max', 'gamma', 'R']

    # Nomes das outras k_points colunas
    l_k_columns = [f'l_k{i+1}' for i in range(points_k)]

    # Concatenando todos os nomes de colunas
    all_columns = initial_columns + l_k_columns

    # Convertendo a matriz para DataFrame
    df = pd.DataFrame(inputs_array, columns=all_columns)

    df.to_csv(path, index=False)

    print(f"DataFrame salvo")

def circularBarrierCrossSection(gamma, 
                                R, 
                                n_min,
                                n_max,
                                k_min,
                                k_max, 
                                points_k,
                                m = 1.0,
                                hbar = 1.0): 
    M = m
    HBAR = hbar
    i = 0 + 1j

    SIGMA = (-i/4.0)*(2*M/(HBAR**2))
    k = np.linspace(k_min, k_max, points_k)
    n_array = np.arange(n_min, n_max + 1, 1)
    l_array = np.array([])

    delta_k = (k_max - k_min)/points_k

    dataCrossSection = np.zeros((1, points_k + 9))
    for k_value in k:
        soma = 0.0

        for n in n_array:
            soma = soma + np.real(coefCircularBarrier(SIGMA, n, gamma, R, k_value))
            
        it = (-4/k_value)*soma
        l_array = np.append(l_array, it)

    row_array = np.array([])
    row_array = np.append(row_array, [M, HBAR, k_min, k_max, delta_k, n_min, n_max, gamma, R])
    row_array = np.append(row_array, l_array)

    dataCrossSection = np.vstack([dataCrossSection, row_array])

    dataCrossSection = np.delete(dataCrossSection, (0), axis=0)

    initial_columns = ['M', 'HBAR', 'k_min', 'k_max', 'delta_k', 'n_min', 'n_max', 'gamma', 'R']

    # Nomes das outras k_points colunas
    l_k_columns = [f'l_k{i+1}' for i in range(points_k)]

    # Concatenando todos os nomes de colunas
    all_columns = initial_columns + l_k_columns

    # Convertendo a matriz para DataFrame
    df = pd.DataFrame(dataCrossSection, columns=all_columns)

    print(df)
    return df

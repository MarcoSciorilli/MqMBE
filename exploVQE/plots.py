import math
import statistics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

def plotter_layer(nodes, layers, starting=0, ending=50, optimization=['COBYLA'],
                  initial_point='True', random='True', quantity='overlaps', save_fig=None, ylim=(0, 1.1), ansaz=False, entanglement=['linear'] ):
    #lines = ['-','--','-.',':','.',',','o','v','^','<','>','1','2','3','4','s','p','*','h','H','+','x','D','d','|','_' ]
    lines = ['-', '--', '-.', ':', 'None', 'solid', 'dashed', 'dashdot', 'dotted']
    sns.set(rc={'figure.figsize': (12, 9)})
    x = np.array(nodes)
    opt_num,entan_num = -1, -1
    for opt in optimization:
        opt_num += 1
        for entan in entanglement:
            entan_num +=1
            for i in range(len(nodes)):
                if quantity == 'energies':
                    total_energies = np.empty(len(layers))
                    total_errors_energies = np.empty(len(layers))
                    for j in range(len(layers)):
                        with open(
                                f'overlap_average_p_{layers[j]}_n_{nodes[i]}_s_{starting}_e_{ending}_opt_{opt}_init_{initial_point}_random_{random}_entang_{entan}.npy',
                                'rb') as f:
                            overlap = np.load(f)
                            energies = np.load(f)
                        total_energies[j] = statistics.mean(energies)
                        total_errors_energies[j] = stats.sem(energies, ddof=0)
                    plt.plot(layers, total_energies, label=f'{entan},{opt}:En(n={nodes[i]})',  linestyle=lines[opt_num+entan_num])
                    plt.fill_between(layers, total_energies - total_errors_energies, total_energies + total_errors_energies,
                                     alpha=0.2)
                if quantity == 'time':
                    total_time = np.empty(len(layers))
                    for j in range(len(layers)):
                        with open(
                                f'overlap_average_p_{layers[j]}_n_{nodes[i]}_s_{starting}_e_{ending}_opt_{opt}_init_{initial_point}_random_{random}_entang_{entan}.npy',
                                'rb') as f:
                            overlap = np.load(f)
                            energies = np.load(f)
                            time = np.load(f)
                        total_time[j] = time
                    plt.plot(layers, total_time, label=f'{entan},{opt}:time (n={nodes[i]})', linestyle=lines[opt_num+entan_num])
                if quantity == 'overlaps':
                    total_overlaps = np.empty(len(layers))
                    total_errors = np.empty(len(layers))
                    for j in range(len(layers)):
                        with open(
                                f'overlap_average_p_{layers[j]}_n_{nodes[i]}_s_{starting}_e_{ending}_opt_{opt}_init_{initial_point}_random_{random}_entang_{entan}.npy',
                                'rb') as f:
                            overlaps = np.load(f)
                        total_overlaps[j] = statistics.mean(overlaps)
                        total_errors[j] = stats.sem(overlaps, ddof=0)
                    plt.plot(layers, total_overlaps, label=f'{entan},{opt}:Ov(n={nodes[i]})', linestyle=lines[opt_num+entan_num])
                    plt.fill_between(layers, total_overlaps - total_errors, total_overlaps + total_errors, alpha=0.2)
                    if ansaz:
                        plt.plot(layers, [ansaz_function(l, nodes[i]) for l in layers], alpha=0.9, linestyle='--',
                                 label=f"Ansaz for nodes n={nodes[i]}")

    plt.legend(title='Number of qubits')
    plt.xticks(layers)
    plt.xlim(min(layers), max(layers))
    plt.ylim(ylim)
    plt.xlabel("Number of layers", fontsize=18)
    if quantity == 'energies':
        plt.ylabel(f"Average energy ratio over {ending -starting} graphs", fontsize=18)
    if quantity == 'time':
        plt.ylabel(f"Average time over {ending -starting} graphs", fontsize=18)
    else:
        plt.ylabel(f"Average overlap over {ending -starting} graphs", fontsize=18)
    if save_fig:
        plt.savefig(f"{save_fig}.svg")
    plt.show()



def ansaz_function(p, n):
    return 2 ** (-(0.18 / (p) + 0.52) * (n / np.log(2)) +1.24)


def prediction_function(z, x, kind='linear'):
    if kind == 'linear' or kind == 'quadratic':
        return np.polyval(z, x)
    if kind == 'sqrt':
        return np.polyval(z, np.sqrt(x))
    if kind == 'log':
        return np.polyval(z, np.log(x))
    if kind == 'over':
        return np.polyval(z, 1 / x)
    if kind == 'const':
        return np.full(x.shape, z)


def fitting_function(z, x, kind='linear'):
    if kind == 'linear':
        return np.polyfit(x, z, 1)
    if kind == 'quadratic':
        return np.polyfit(x, z, 2)
    if kind == 'sqrt':
        return np.polyfit(np.sqrt(x), z, 1)
    if kind == 'log':
        return np.polyfit(np.log(x), z, 1)
    if kind == 'over':
        return np.polyfit(1 / x, z, 1)
    if kind == 'const':
        return np.mean(z)


def plotter_quibits(min_node_number, max_node_number, layers, starting=0, ending=50, optimization='COBYLA',
                    initial_point='True',
                     random='True', quantity='overlaps', save_fig=None, ylim=(0, 1.1), alpha=None,
                    beta=None, coefficients=None, intercepts=None, entanglement='linear'):
    fig, axs = plt.subplots(figsize=(20, 10))
    x = np.empty(max_node_number - min_node_number + 1)
    for i in range(min_node_number, max_node_number + 1):
        x[i - min_node_number] = i
    l = 0
    if alpha == None and beta == None:
        coefficients = np.empty(len(layers))
        intercepts = np.empty(len(layers))
    for j in layers:
        if quantity == 'energies':
            total_energies = np.empty(max_node_number - min_node_number + 1)
            total_errors_energies = np.empty(max_node_number - min_node_number + 1)
            for i in range(min_node_number, max_node_number + 1):
                with open(
                        f'overlap_average_p_{j}_n_{i}_s_{starting}_e_{ending}_opt_{optimization}_init_{initial_point}_random_{random}_entang_{entanglement}.npy',
                        'rb') as f:
                    overlaps = np.load(f)
                    energies = np.load(f)
                total_energies[i - min_node_number] = statistics.mean(energies)
                total_errors_energies[i - min_node_number] = stats.sem(energies) / total_energies[i - min_node_number]
            if alpha == None and beta == None:
                total_energies = np.log(total_energies)
                z, cov = np.polyfit(x, total_energies, 1, cov=True)
                fit_ov = z[0] * x + z[1]
                prediction = np.polyval(z, x)
                mse = np.mean((prediction - total_energies) ** 2)
                plt.plot(x, fit_ov, alpha=0.9, linestyle='--',
                         label=f'En(p={j}): y=({round(z[0], 2)}$\pm${round(np.sqrt(np.diag(cov))[0], 2)})x +{round(z[1], 2)}$\pm$ {round(np.sqrt(np.diag(cov))[1], 2)}, RMSE:{round(math.sqrt(mse), 2)}')
                coefficients[l] = round(z[0], 2)
                intercepts[l] = round(z[1], 2)
                l += 1
                plt.plot(x, total_energies, label=f'En(p={j})')
                plt.fill_between(x, total_energies - total_errors_energies, total_energies + total_errors_energies,
                                 alpha=0.2)
            else:
                z_coefficients = fitting_function(coefficients, layers, kind=alpha)
                z_intercepts = fitting_function(intercepts, layers, kind=beta)
                total_energies = np.log(
                    total_energies / np.exp(prediction_function(z_intercepts, j, beta))) / prediction_function(
                    z_coefficients, j, alpha)
                z, cov = np.polyfit(x, total_energies, 1, cov=True)
                fit_ov = z[0] * x + z[1]
                prediction = np.polyval(z, x)
                mse = np.mean((prediction - total_energies) ** 2)
                plt.plot(x, fit_ov, alpha=0.9, linestyle='--',
                         label=f'En(p={j}): y=({round(z[0] / math.log(2), 2)}$\pm${round(np.sqrt(np.diag(cov))[0], 2)})x$\log 2$ +{round(z[1], 2)}$\pm$ {round(np.sqrt(np.diag(cov))[1], 2)}, RMSE:{round(math.sqrt(mse), 2)} ')
                plt.plot(x, total_energies, label=f'En(p={j})')
        #                 plt.fill_between(x, total_energies - total_errors_energies, total_energies + total_errors_energies,  alpha=0.2)

        else:
            total_overlaps = np.empty(max_node_number - min_node_number + 1)
            total_errors = np.empty(max_node_number - min_node_number + 1)
            for i in range(min_node_number, max_node_number + 1):
                with open(
                        f'overlap_average_p_{j}_n_{i}_s_{starting}_e_{ending}_opt_{optimization}_init_{initial_point}_random_{random}_entang_{entanglement}.npy',
                        'rb') as f:
                    overlaps = np.load(f)
                total_overlaps[i - min_node_number] = statistics.mean(overlaps)
                total_errors[i - min_node_number] = stats.sem(overlaps) / total_overlaps[i - min_node_number]
            if alpha == None and beta == None:
                total_overlaps = np.log(total_overlaps)
                z, cov = np.polyfit(x, total_overlaps, 1, cov=True)
                fit_ov = z[0] * x + z[1]
                prediction = np.polyval(z, x)
                mse = np.mean((prediction - total_overlaps) ** 2)
                plt.plot(x, fit_ov, alpha=0.9, linestyle='--',
                         label=f'Ov(p={j}): y=({round(z[0], 2)}$\pm${round(np.sqrt(np.diag(cov))[0], 2)})x +{round(z[1], 2)}$\pm$ {round(np.sqrt(np.diag(cov))[1], 2)}, RMSE:{round(math.sqrt(mse), 2)}')
                coefficients[l] = round(z[0], 2)
                intercepts[l] = round(z[1], 2)
                l += 1
                plt.plot(x, total_overlaps, label=f'Ov(p={j})')
                plt.fill_between(x, total_overlaps - total_errors, total_overlaps + total_errors, alpha=0.2)
            else:
                z_coefficients = fitting_function(coefficients, layers, kind=alpha)
                z_intercepts = fitting_function(intercepts, layers, kind=beta)
                total_overlaps = np.log(
                    total_overlaps / np.exp(prediction_function(z_intercepts, j, beta))) / prediction_function(
                    z_coefficients, j, alpha)
                z, cov = np.polyfit(x, total_overlaps, 1, cov=True)
                fit_ov = z[0] * x + z[1]
                prediction = np.polyval(z, x)
                mse = np.mean((prediction - total_overlaps) ** 2)
                plt.plot(x, fit_ov, alpha=0.9, linestyle='--',
                         label=f'Ov(p={j}): y=({round(z[0], 2)}$\pm${round(np.sqrt(np.diag(cov))[0], 2)})x +{round(z[1], 2)}$\pm$ {round(np.sqrt(np.diag(cov))[1], 2)}, RMSE:{round(math.sqrt(mse), 2)}')
                plt.plot(x, total_overlaps, label=f'Ov(p={j})')
        #                 plt.fill_between(x, total_overlaps - total_errors, total_overlaps + total_errors,  alpha=0.2)
        plt.legend(title='Number of layers')
        axs.set_xticks(range(min_node_number, max_node_number))
        axs.set_xlim(min_node_number, max_node_number)
        axs.set_xlabel("Number of qubits", fontsize=18)
        axs.set_ylim(ylim)
        if quantity == 'energies':
            axs.set_ylabel(f"Average energy ratio over {starting - ending} graphs", fontsize=18)
        else:
            axs.set_ylabel(f"Average overlap over {starting - ending} graphs", fontsize=18)
        if save_fig:
            plt.savefig(f"{save_fig}.svg")
        labels = [float('%.2g' % np.exp(item)) for item in axs.get_yticks()]
        axs.set_yticklabels(labels)
        plt.show()
        if alpha == None and beta == None:
            return coefficients, intercepts



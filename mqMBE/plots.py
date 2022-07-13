import math
import statistics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from mqMBE.datamanager import read_data
import mqMBE.multibase as mb
import pandas as pd
from matplotlib.pyplot import cm
import cmath
import json
from matplotlib.animation import FuncAnimation

class plotter(object):
    def __init__(self, x, y, flags, name_database):
        self.x = x
        self.y = y
        self.flags = flags
        self.name_database = name_database
        self.dimension = (12, 9)
        self.lines = ['-', '--', '-.', ':']
        self.instances = None
        self.trials = None
        self.expl_x = None
        self.expl_y = None
        self.x_ax_name = self.x
        self.y_ax_name = self.y
        self.y_lim = None
        self.x_lim = None

    def set_size(self, dimension):
        self.dimension = dimension

    def set_name_database(self, name_database):
        self.name_database = name_database

    def set_instances(self, instances):
        self.instances = instances

    def set_trials(self, trials):
        self.trials = trials

    def set_flags(self, flags):
        self.flags = flags

    def set_expl_x(self, x):
        self.expl_x = x

    def set_expl_y(self, y):
        self.expl_y = y

    def set_x_ax_name(self, name):
        self.x_ax_name = name

    def set_y_ax_name(self, name):
        self.y_ax_name = name

    def set_y_lim(self, lim):
        self.y_lim = lim

    def set_x_lim(self, lim):
        self.x_lim = lim
    def single_plot(self, method='mean'):
        sns.set(rc={'figure.figsize': self.dimension})
        flags = self.flags
        data = read_data(self.name_database, self.name_database, [self.x, self.y], flags)
        data = pd.DataFrame(data, columns=[self.x, self.y])
        data = data.mask(data.eq('None')).dropna()
        if self.expl_x is not None:
            data = data[data[self.x].isin(self.expl_x)]
        if self.expl_y is not None:
            pass
        if method == 'max':
            data = data.groupby([self.x], as_index=False).max()
        elif method == 'min':
            data = data.groupby([self.x], as_index=False).min()
        elif method == 'median':
            data = data.groupby([self.x], as_index=False).median()
        else:
            pass
        plot = sns.lineplot(data=data, x=self.x, y=self.y)
        plot.set_xlabel(f"{self.x_ax_name}", fontsize=20)
        plot.set_ylabel(f"{self.y_ax_name}", fontsize=20)
        plt.xticks(data[self.x])
        if method == 'mean':
            data_mean = data.groupby([self.x], as_index=False).mean()
            data_std = data.groupby([self.x], as_index=False).std()
            data = data_mean
            data['std'] = data_std[self.y]
        plot.set_ylim(self.y_lim)
        plot.set_xlim(self.x_lim)
        print(data)


    def compare_plot(self, label, method):
        sns.set(rc={'figure.figsize': self.dimension})
        flags = self.flags
        data = read_data(self.name_database, self.name_database, [self.x, self.y, label], flags)
        data = pd.DataFrame(data, columns=[self.x, self.y, label])
        data = data.mask(data.eq('None')).dropna()
        if self.expl_x is not None:
            data = data[data[self.x].isin(self.expl_x)]
        if self.expl_y is not None:
            pass
        if method == 'max':
            data = data.groupby([self.x, label], as_index=False).max()
        elif method == 'min':
            data = data.groupby([self.x, label], as_index=False).min()
        elif method == 'median':
            data = data.groupby([self.x, label], as_index=False).median()
        else:
            pass
        plot = sns.lineplot(data=data, x=self.x, y=self.y, hue=label)
        plot.set_xlabel(f"{self.x_ax_name}", fontsize=20)
        plot.set_ylabel(f"{self.y_ax_name}", fontsize=20)
        plt.xticks(data[self.x])
        if method == 'mean':
            data_mean = data.groupby([self.x, label], as_index=False).mean()
            data_std = data.groupby([self.x, label], as_index=False).std()
            data = data_mean
            data['std'] = data_std[self.y]
        plot.set_ylim(self.y_lim)
        plot.set_xlim(self.x_lim)
        print(data)

    def multiple_database_compare_plot(self, label, method, other_database):
        sns.set(rc={'figure.figsize': self.dimension})
        flags = self.flags
        data = read_data(self.name_database, self.name_database, [self.x, self.y, label], flags)
        data = pd.DataFrame(data, columns=[self.x, self.y, label])
        data = data.mask(data.eq('None')).dropna()
        data_other = read_data(other_database, other_database, [self.x, self.y, label], flags)
        data_other = pd.DataFrame(data_other, columns=[self.x, self.y, label])
        data_other = data_other.mask(data_other.eq('None')).dropna()
        data = pd.concat([data, data_other], keys=[self.name_database, other_database])
        data.index.names = ['database_name', 'index']
        data = data.reset_index(level=['database_name'])
        data = data.reset_index(drop=True)
        if self.expl_x is not None:
            data = data[data[self.x].isin(self.expl_x)]
        if self.expl_y is not None:
            pass
        if method == 'max':
            data = data.groupby([self.x, label, 'database_name'], as_index=False).max()
        elif method == 'min':
            data = data.groupby([self.x, label, 'database_name'], as_index=False).min()
        elif method == 'median':
            data = data.groupby([self.x, label, 'database_name'], as_index=False).median()
        else:
            pass
        plot = sns.lineplot(data=data, x=self.x, y=self.y, hue=label, style='database_name')
        plot.set_xlabel(f"{self.x_ax_name}", fontsize=20)
        plot.set_ylabel(f"{self.y_ax_name}", fontsize=20)
        plt.xticks(data[self.x])
        if method == 'mean':
            data_mean = data.groupby([self.x, label,'database_name'], as_index=False).mean()
            data_std = data.groupby([self.x, label,'database_name'], as_index=False).std()
            data = data_mean
            data['std'] = data_std[self.y]
        plot.set_ylim(self.y_lim)
        plot.set_xlim(self.x_lim)
        print(data)

    def multiple_database_plot(self, method, other_database):
        sns.set(rc={'figure.figsize': self.dimension})
        flags = self.flags
        data = read_data(self.name_database, self.name_database, [self.x, self.y], flags)
        data = pd.DataFrame(data, columns=[self.x, self.y])
        data = data.mask(data.eq('None')).dropna()
        data_other = read_data(other_database, other_database, [self.x, self.y], flags)
        data_other = pd.DataFrame(data_other, columns=[self.x, self.y])
        data_other = data_other.mask(data_other.eq('None')).dropna()
        data = pd.concat([data, data_other], keys=[self.name_database, other_database])
        data.index.names = ['database_name', 'index']
        data = data.reset_index(level=['database_name'])
        data = data.reset_index(drop=True)
        if self.expl_x is not None:
            data = data[data[self.x].isin(self.expl_x)]
        if self.expl_y is not None:
            pass
        if method == 'max':
            data = data.groupby([self.x, 'database_name'], as_index=False).max()
        elif method == 'min':
            data = data.groupby([self.x, 'database_name'], as_index=False).min()
        elif method == 'median':
            data = data.groupby([self.x, 'database_name'], as_index=False).median()
        else:
            pass
        plot = sns.lineplot(data=data, x=self.x, y=self.y, style='database_name')
        plot.set_xlabel(f"{self.x_ax_name}", fontsize=20)
        plot.set_ylabel(f"{self.y_ax_name}", fontsize=20)
        plt.xticks(data[self.x])
        if method == 'mean':
            data_mean = data.groupby([self.x,'database_name'], as_index=False).mean()
            data_std = data.groupby([self.x,'database_name'], as_index=False).std()
            data = data_mean
            data['std'] = data_std[self.y]
        plot.set_ylim(self.y_lim)
        plot.set_xlim(self.x_lim)
        print(data)

    def plot_histogram(self,flag, name_database, bins, quibits, save_fig=None):
        pauli_letter = {'X': 1, 'Y': 2, 'Z': 3}
        sns.set(rc={'figure.figsize': (12, 9)})
        solutions = read_data(name_database, 'MaxCutDatabase', ['unrounded_solution'], flag)
        solutions = [json.loads(solutions[j][0]) for j in range(len(solutions))]
        # pauli_string = [0] * 5
        # for pauli in axis:
        #     for index in axis[pauli]:
        #         pauli_string[index] = pauli_letter[pauli]
        pauli_string_list = mb.MultibaseVQA._pauli_string(quibits, 2)
        index_list_two = [i for i in range(len(pauli_string_list)) if pauli_string_list[i].count(0) == (quibits - 2)]
        index_list_single = [i for i in range(len(pauli_string_list)) if pauli_string_list[i].count(0) == (quibits - 1)]

        my_values_two = []
        # for index in range(len(index_list_two)):
        #     # for index in index_list_two:
        #     for l in range(len(solutions)):
        #         my_values_two.append(solutions[l][index])

        for l in range(len(solutions)):
            for i in solutions[l]:
                my_values_two.append(i)
        plt.hist(x=my_values_two, bins=bins, range=(-1, 1))
        plt.savefig(f"two_quibit_nolinear.svg")
        plt.show()
        # my_values_single = []
        # for index in index_list_single:
        #     for l in range(len(solutions)):
        #         my_values_single.append(solutions[l][index])
        # plt.hist(x=my_values_single,  bins=bins, range=(-1,1))
        # plt.savefig(f"single_quibit_nolinear.svg")
        # plt.show()
        # print(statistics.mean(list(map(abs, my_values_single))),statistics.mean(list(map(abs, my_values_two))))


    def animated_histograms(self, range_layer, flag, name_database, bins, ylimit, namefile=None):
        fig = plt.figure()
        sns.set(rc={'figure.figsize': (12, 9)})
        def plot_histogram_frame(i, range_layer,flag, name_database, bins):
            flag['layer_number'] = f'{range_layer[i]}'
            solutions = read_data(name_database, 'MaxCutDatabase', ['unrounded_solution'], flag)
            solutions = [json.loads(solutions[j][0]) for j in range(len(solutions))]

            my_values_two = []
            for l in range(len(solutions)):
                for i in solutions[l]:
                    my_values_two.append(i)
            plt.cla()
            plt.ylim(ylimit)
            results = plt.hist(x=my_values_two, bins=bins, range=(-1, 1))
            modes_index = np.argpartition(results[0], -2)[-2:]
            mode = (abs(results[1][modes_index[0]])+abs(results[1][modes_index[1]]))/2
            print(mode)


        ani = FuncAnimation(fig, plot_histogram_frame, frames=len(range_layer), interval=500,
                            fargs=(range_layer,flag, name_database, bins))
        if namefile is None:
            ani.save(f'Dummy.gif')
        else:
            ani.save(f'{namefile}.gif')
        plt.show()



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


def solve_quadratic(a, b, c):
    d = (b ** 2) - (4 * a * c)

    # find two solutions
    sol1 = (-b - cmath.sqrt(d)) / (2 * a)
    sol2 = (-b + cmath.sqrt(d)) / (2 * a)
    return int(max(sol1.real, sol2.real))



def plotter_compare(x, y, flags, fixed, compares, pick_method='average', instances=None, save_fig=None, ylim=(0, 1.1)):
    sns.set(rc={'figure.figsize': (12, 9)})
    x_array = np.array(x[1])
    color = iter(cm.rainbow(np.linspace(0, 1, 5)))
    for flag in flags[1]:
        average_y = np.empty(len(x[1]))
        error_y = np.empty(len(x[1]))
        for i in range(len(x_array)):
            fixed_labels = fixed
            fixed_labels[flags[0]] = flag
            total_y = []
            for term in compares[1]:
                if term == 'goemans_williamson':
                    name_database = 'MaxCutDatabase_gw'
                    fixed_labels = {'optimization': 'None', 'qubits': 'None', 'entanglement': 'None',
                                    'graph_kind': 'indexed', 'activation_function': 'None', flags[0]: flag}
                else:
                    name_database = 'MaxCutDatabase'
                    fixed_labels[x[0]] = x_array[i]
                    # fixed_labels['compression'] = int(1)
                fixed_labels[compares[0]] = term
                if instances:
                    data_y = []
                    for j in instances:
                        fixed_labels['instance'] = j
                        y_instance = read_data(name_database, 'MaxCutDatabase', [compares[0], y, 'instance', 'trial'],
                                               fixed_labels)
                        data_y.append(y_instance[0][0])
                else:
                    data_y = read_data(name_database, 'MaxCutDatabase', [compares[0], y, 'instance', 'trial'],
                                       fixed_labels)
                data_y = pd.DataFrame(data_y, columns=[compares[0], y, 'instance', 'trial'])
                total_y.append(data_y)
            total_y = pd.concat(total_y)
            if pick_method == 'average':
                total_y = total_y.groupby(['instance', compares[0]], as_index=False).mean()
            else:
                total_y = total_y.groupby(['instance', compares[0]], as_index=False).max()
            total_y_ratio = [float(total_y[(total_y['instance'] == f'{i}') & (total_y[compares[0]] == compares[1][0])][
                                       'max_energy']) / float(
                total_y[(total_y['instance'] == f'{i}') & (total_y[compares[0]] == compares[1][1])]['max_energy']) for i
                             in range(int(len(total_y) / 2))]
            average_y[i] = statistics.mean(total_y_ratio)
            error_y[i] = stats.sem(total_y_ratio, ddof=0)
        print('Averages y:', average_y, "Errors y:", error_y)
        c = next(color)
        plt.plot(x_array, average_y, label=f'{flag}', color=c)
        plt.fill_between(x_array, average_y - error_y, average_y + error_y, alpha=0.2, color=c)

    plt.legend(title=f'{flags[0]}')
    plt.xticks(x_array)
    plt.xlim(min(x_array), max(x_array))
    plt.ylim(ylim)
    plt.xlabel(f"{x[0]}", fontsize=18)
    plt.ylabel(f"Cut ratio", fontsize=18)
    plt.title(f"Average cut ratio (multibase/goeman) over {int(len(total_y) / 2)} graphs, 5 trials for instance",
              fontsize=18)
    if save_fig:
        plt.savefig(f"{save_fig}.svg")
    plt.show()

import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from abc import ABC, abstractmethod
from typing import Type

plt.rcParams["font.size"] = 18

class Base(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def calc(self, data):
        pass

class Normal(Base):
    def __init__(self):
        self.stored_data = np.empty([0])

    def __str__(self):
        return "Normal method"

    def calc(self, data):
        # https://qiita.com/byeron/items/f84b54201aced12fec80
        stored_data_converted_to_list = self.stored_data.tolist()
        stored_data_converted_to_list.append(data)
        self.stored_data = np.asarray(stored_data_converted_to_list)
        mean = np.mean(self.stored_data)
        if len(self.stored_data) ==1:
            st_dev = 0
        else:
            st_dev = np.sqrt(np.var(self.stored_data, ddof=1))
        return mean, st_dev

class Sequential(Base):
    def __init__(self):
        self.num_data = 0
        self.sum = 0
        self.sum_of_squares = 0
    
    def __str__(self):
        return "Sequential method"

    def calc(self, data):
        self.num_data += 1
        self.sum += data
        self.sum_of_squares += data**2
        mean = self.sum / self.num_data
        if self.num_data == 1:
            st_dev = 0
        else:
            st_dev = np.sqrt((1/(self.num_data-1))*(self.sum_of_squares - self.sum**2/self.num_data))
        return mean, st_dev

class Welford(Base):
    def __init__(self):
        self.num_data = 0
        self.mean = 0
        self.Mn = 0
    
    def __str__(self):
        return "Welford method"

    def calc(self, data):
        self.num_data += 1
        # update mean
        mean_before = self.mean
        self.mean = self.mean + (data - mean_before)/self.num_data
        # update st_dev
        right_term = (data - self.mean)*(data - mean_before)
        self.Mn = (self.Mn + right_term)
        if self.num_data == 1:
            st_dev = 0
        else:
            st_dev = np.sqrt(self.Mn/(self.num_data-1))
        return self.mean, st_dev

# https://en.wikipedia.org/wiki/Kahan_summation_algorithm
class Kahan():
    def __init__(self):
        self.correct_sum_value = 0
        self.c = 0

    def add(self, data):
        y = data - self.c
        t = self.correct_sum_value + y
        self.c = (t - self.correct_sum_value) - y
        self.correct_sum_value = t
        return self

class KahanWelford(Base):
    def __init__(self):
        self.num_data = 0
        self.mean = Kahan()
        self.Mn = Kahan()

    def __str__(self):
        return "Welford method with Kahan summation"

    def calc(self, data):
        self.num_data += 1
        # update mean
        mean_before = self.mean.correct_sum_value
        self.mean = self.mean.add((data - mean_before)/self.num_data)
        # update st_dev
        right_term = (data - self.mean.correct_sum_value)*(data - mean_before)
        self.Mn = self.Mn.add(right_term)
        if self.num_data == 1:
            st_dev = 0
        else:
            st_dev = np.sqrt(self.Mn.correct_sum_value/(self.num_data-1))
        return self.mean.correct_sum_value, st_dev

def calc_elapsed_time(start):
    elapsed_time = time.time() - start
    return elapsed_time

def define_fig_element(ax, one_method, perform_obj_dic, ylabel, title, marker):
        mem_dic = perform_obj_dic[f"{one_method.__name__}"]
        x = mem_dic.keys()
        y = mem_dic.values()
        ax.get_yaxis().get_major_formatter().set_useOffset(False) # remove exponential representation
        ax.plot(x, y, label=f"{one_method.__name__}", marker=marker)
        ax.set_xlabel("Number of calculation iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        return ax

if __name__ == '__main__':
    
    # input data definition
    N_input_data = 50_000
    pseudo_stream_data = 100 + 10*np.random.randn(N_input_data) # {N_input_data} values generated from normal distribution with mean=100, st_dev=10
    # plt.hist(pseudo_stream_data, bins=10)
    # plt.show()

    # variables for testing performance
    elapsed_time = {}
    mean_val = {}
    st_dev_val = {}

    # select one method for evaluation
    class_list = [Normal, Sequential, Welford, KahanWelford]
    for one_method in class_list:
        calculator: Type[Base] = one_method()
        print(f"--------- {calculator} ---------")
        
        _elapsed_time = {}
        _mean = {}
        _st_dev = {}

        start = time.time()
        for i, num in enumerate(pseudo_stream_data):
            mean, st_dev = calculator.calc(num)
            if (i+1)%5_000 == 0 and i != 0:
                _elapsed_time[f"{i+1}"] = calc_elapsed_time(start)
                _mean[f"{i+1}"] = mean
                _st_dev[f"{i+1}"] = st_dev
                print(f"{i+1}-loops : {mean=}, {st_dev=}")
        print(f"------------------\n")

        # memorize elapsed time for each method
        elapsed_time[f"{one_method.__name__}"] = _elapsed_time
        mean_val[f"{one_method.__name__}"] = _mean
        st_dev_val[f"{one_method.__name__}"] = _st_dev

    # visualize performance
    fig = plt.figure()
    ax_calc_time = fig.add_subplot(131)
    ax_mean = fig.add_subplot(132)
    ax_st_dev = fig.add_subplot(133)
    fig_elem_dic = {
        ax_calc_time: [elapsed_time, "Elapsed time [s]", "Calculation time performace"],
        ax_mean: [mean_val, "Mean", "Calculation accuracy of mean"],
        ax_st_dev: [st_dev_val, "Standard deviation", "Calculation accuracy of standard deviation"],
    }
    marker_list = ["o", "^", "s", "*"]
    for i, one_method in enumerate(class_list):
        marker = marker_list[i]
        for one_ax, fig_elem in fig_elem_dic.items():
            one_ax = define_fig_element(
                one_ax, 
                one_method, 
                fig_elem[0], 
                fig_elem[1], 
                fig_elem[2], 
                marker
                )
            
    # visualize detail accuracy
    fig_detail = plt.figure(figsize=(10,10))
    ax_mean_diff = fig_detail.add_subplot(121)
    ax_st_dev_diff = fig_detail.add_subplot(122)

    # define (x, y) values
    x = mean_val["Normal"].keys()
    y_mean_diff_1 = np.fromiter(mean_val["Sequential"].values(), dtype=float) - np.fromiter(mean_val["Normal"].values(), dtype=float)
    y_mean_diff_2 = np.fromiter(mean_val["Welford"].values(), dtype=float) - np.fromiter(mean_val["Normal"].values(), dtype=float)
    y_mean_diff_3 = np.fromiter(mean_val["KahanWelford"].values(), dtype=float) - np.fromiter(mean_val["Normal"].values(), dtype=float)
    y_st_dev_diff_1 = np.fromiter(st_dev_val["Sequential"].values(), dtype=float) - np.fromiter(st_dev_val["Normal"].values(), dtype=float)
    y_st_dev_diff_2 = np.fromiter(st_dev_val["Welford"].values(), dtype=float) - np.fromiter(st_dev_val["Normal"].values(), dtype=float)
    y_st_dev_diff_3 = np.fromiter(st_dev_val["KahanWelford"].values(), dtype=float) - np.fromiter(st_dev_val["Normal"].values(), dtype=float)

    # define chart settings
    ax_mean_diff.plot(x, y_mean_diff_1, label="Sequential-Normal")
    ax_mean_diff.plot(x, y_mean_diff_2, label="Welford-Normal")
    ax_mean_diff.plot(x, y_mean_diff_3, label="KahanWelford-Normal")
    ax_mean_diff.set_xlabel("Number of calculation iteration")
    ax_mean_diff.set_ylabel("Difference of mean from Normal method")
    ax_mean_diff.legend()
    
    # define chart settings
    ax_st_dev_diff.plot(x, y_st_dev_diff_1, label="Sequential-Normal")
    ax_st_dev_diff.plot(x, y_st_dev_diff_2, label="Welford-Normal")
    ax_st_dev_diff.plot(x, y_st_dev_diff_3, label="KahanWelford-Normal")
    ax_st_dev_diff.set_xlabel("Number of calculation iteration")
    ax_st_dev_diff.set_ylabel("Difference of standard deviation from Normal method")
    ax_st_dev_diff.legend()
    plt.tight_layout()
    plt.show()
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

# Code for generating figures 1 and 2

def size_freq_analysis(size_vector):

    max_size = max(size_vector)
    min_size = min(size_vector)
    print("Minimum Molecule Size: {}".format(min_size))
    print("Maximum Molecule Size: {}".format(max_size))
    num_bins = max_size - min_size
    # Get distribution of sizes
    dist_dict = {}
    total_size = len(size_vector)

    for size in size_vector:
        if size in dist_dict.keys():
            dist_dict[size] += 1
        else:
            dist_dict[size] = 1

    distribution_less_than = 0
    for key in sorted(dist_dict.keys()):
        val = dist_dict[key]
        prob = round((val/total_size) * 100, 2)
        distribution_less_than += prob
        print("Size: {}, Frequency in dataset: {}. Percentage of dataset: {}. Percentage less than or equal to: {}"
              .format(key, val, prob, round(distribution_less_than, 2)))

    plt.hist(size_vector, bins=num_bins, density=False)
    plt.gca().set(title='Frequency Histogram', ylabel='Frequency', xlabel='Molecule Size')
    plt.show()


def analysis(data):

    for column in data:

        avg = np.mean(data[column])
        std = np.std(data[column])
        med = np.median(data[column])
        max = np.max(data[column])
        min = np.min(data[column])
        print("\n\n")
        print("PROPERTY: {}".format(column))
        print("Average: {}".format(round(avg, 2)))
        print("Standard Deviation: {}".format(round(std, 2)))
        print("Median: {}".format(round(med, 2)))
        print("Minimum: {}".format(round(min, 2)))
        print("Maximum: {}".format(round(max, 2)))



df = pd.read_csv("J:\PythonProjects\MolGen\Data\ZINC\ZINC_Augmented.csv")
print("Data Loaded")

size_distro = df['NumAtoms']
size_freq_analysis(size_distro)

# this drops all molecules larger than the designated size
df = df[df['NumAtoms'] <= 16].reset_index(drop=True)
# There are some broken SMILES, this removes them
df = df[df['MR'] >= 0.1].reset_index(drop=True)

# df = df.drop(["Unnamed: 0", "ZINC_ID", "Desolv_apolar", "Desolv_polar", "SMILES"], axis=1)
# print(df.shape)

# corrMatrix = df.corr()
# print(corrMatrix)
#
# analysis(df)
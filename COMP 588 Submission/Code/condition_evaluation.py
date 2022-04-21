import warnings
import numpy as np
import pandas as pd
import pickle as pkl
from rdkit import RDLogger
from scipy.stats.stats import pearsonr
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings(action='ignore')

# Since we don't care to disambiguate VUN, this rapidly removes non-unique and non-novel entries
# Non-valid entries are already eliminated at the generation step
def ensure_unique_novel(df, training_data):

    print(df.shape)
    df = df.drop_duplicates(subset='SMILES', keep="last")
    df = df[~df.SMILES.isin(training_data)]
    print(df.shape)

    return df

# A function to comprehensively evaluate various aspects of molecular properties with respect to their target values
def constraint_evaluation(theoretical_total, experiment_df, conditions, output_name):

    print("Base VUN compliance: {}".format(experiment_df.shape[0]/theoretical_total))
    tallies = {}
    total_corr = []
    for cond in conditions:
        property_vector = experiment_df[cond]
        property_targets = experiment_df[cond + " Target"]
        target_set = list(set(experiment_df[cond + " Target"]))
        tally = []
        for property, prop_target in zip(property_vector, property_targets):
            nearest_target = target_set[min(range(len(target_set)), key=lambda i: abs(target_set[i] - property))]
            if nearest_target == prop_target:
                tally.append(1)
            else:
                tally.append(0)
        tallies[cond + " Tally"] = tally
        correlation = pearsonr(property_targets, property_vector)
        total_corr.append(correlation[0])
        print("The correlation between {} and its target is: {}".format(cond, correlation))
    print("The average correlation between properties and their target is: {}".format(np.mean(total_corr)))


    sum_tally = []
    for key in tallies.keys():
        experiment_df[key] = tallies[key]
        sum_tally.append(tallies[key])
        print("Total compliance of {}: {} out of {} VUN compliant molecules {}%".format(key.split(" ")[0], sum(tallies[key]),
                                                                                    experiment_df.shape[0], round(100 * sum(tallies[key])/experiment_df.shape[0], 2)))

    sum_tally = [sum(x) for x in zip(*sum_tally)]
    experiment_df["Sum Tally"] = sum_tally

    compliance = list(set(sum_tally))
    for v in compliance:
        print("Number of molecules that complied with {} properties: {}".format(v, sum_tally.count(v)))
        print("This represents {}% of VUN compliant molecules, and {}% of all generated molecules"
              .format(round(100 *sum_tally.count(v)/experiment_df.shape[0], 2), round(100 * sum_tally.count(v)/theoretical_total, 2)))

    print("Average property compliance per molecule: {}".format(np.mean(sum_tally)/len(tallies.keys())))

    experiment_df.to_csv(output_name)


folder = "J:\PythonProjects\MolGen\Results\Summary\\"

# Again, I'd have made a better system if I had time, but this works for what it needs to do and show

# file = "Norm_3_4_logp_mr_lr_5-04_unique_93-1.csv"
# ideal = 1000 * 50
# conditions = ["LogP", "MR"]

# file = "Norm_3_4_logp_mr_tpsa_lr_5-05_unique_90-41.csv"
# ideal = 1000 * 344
# conditions = ["LogP", "MR", "tPSA"]

# file = "Unnorm_logp_mr_lr_5-04_unique_82-79.csv"
# ideal = 10000 * 20
# conditions = ["LogP", "MR"]

# file = "Unnorm_logp_mr_lr_5-05_unique_84-0.csv"
# ideal = 10000 * 20
# conditions = ["LogP", "MR"]

# file = "Unnorm_logp_mr_tpsa_lr_5-05_unique_73-95.csv"
# ideal = 10000 * 80
# conditions = ["LogP", "MR", "tPSA"]

# file = "Unnorm_z_256_logp_mr_tpsa_lr_adaptive_unique_76-43.csv"
# ideal = 10000 * 80
# conditions = ["LogP", "MR", "tPSA"]

# file = "Unnorm_mwt_hbd_nrb_lr_5-04_unique_85-74.csv"
# ideal = 1000 * 123
# conditions = ["MWT", "HBD", "Rotatable Bonds"]

file = "Unnorm_mwt_hbd_tpsa_nrb_lr_5-04_unique_73-75.csv"
ideal = 1000 * 500
conditions = ["MWT", "HBD", "tPSA", "Rotatable Bonds"]


# file = "Unnorm_logp_mr_tpsa_nrb_lr_5-04_unique_80-64.csv"
# ideal = 1000 * 400
# conditions = ["LogP", "MR", "tPSA", "Rotatable Bonds"]


df = pd.read_csv(folder + file)
training_data = pkl.load(open("J:\PythonProjects\MolGen\Results\Sample_Mols_Full_16.pkl", 'rb'))

clean_df = ensure_unique_novel(df, training_data)
output_file = folder + file.split(".")[0] + "_property_stats.csv"
constraint_evaluation(ideal, clean_df, conditions, output_file)
import pandas as pd
import numpy as np
import pickle as pkl


def analysis(data, cond, target):
    avg = np.mean(data)
    std = np.std(data)
    med = np.median(data)
    max = np.max(data)
    min = np.min(data)
    print("The average {} was {} (std: {}) with the target being {}".format(cond, round(avg, 2), round(std, 2), target))
    print("The median {} was {} (max: {}, min: {})".format(cond, round(med, 2), round(max, 2), round(min, 2)))

    return [avg, std, med, max, min]

# A very messy way to summarize and consolidate generated outputs
# I would have loved to have made it less...ugly, but there is only so much time.
conditions = pd.read_csv('J:\PythonProjects\MolGen\Data\conditions_mwt_hbd_tpsa_nrb.csv')
mwt_tar = [int(i) for i in conditions['MWT'] if str(i) != "nan"]
hbd_tar = [int(i) for i in conditions['HBD'] if str(i) != "nan"]
#tpsa_tar = [int(i) for i in conditions['tPSA'] if str(i) != "nan"]
nrb_tar = [int(i) for i in conditions['NRB'] if str(i) != "nan"]

num_cond = 3
combinations = np.array(np.meshgrid(mwt_tar, hbd_tar, nrb_tar)).T.reshape(-1, num_cond)
num_generated = 1000
folder = "J:\PythonProjects\MolGen\\Results\Three_Low_Correlation_Constraints\\"
sample = pkl.load(open("J:\PythonProjects\MolGen\Results\Sample_Mols_Full_16.pkl", 'rb'))
total_df = pd.DataFrame([])
validity = []
novelty = []

# for unnorm
# unnorm_mr_std = 10
unnorm_tpsa_std = 10
unnorm_mwt_std = 25

summary_stats = []

for combo in combinations:
    file = "Full_5-04_1024_16_Conditions_{}_{}_{}.csv".format(combo[0], combo[1], combo[2])
    present_df = pd.read_csv(folder + file)
    val_score = present_df.shape[0] / num_generated
    validity.append(val_score)

    not_novel = 0
    for molecule in present_df['SMILES']:
        if molecule in sample:
            not_novel += 1

    novelty_score = (present_df.shape[0] - not_novel) / present_df.shape[0]
    novelty.append(novelty_score)
    print("Validity of {}: {}".format(file, val_score))
    print("Novelty of {}: {}".format(file, novelty_score))

    target_value_mwt = combo[0] * unnorm_mwt_std
    mwt_avg, mwt_std,mwt_med, mwt_max, mwt_min = analysis(present_df["MWT"], "MWT", target_value_mwt)

    target_value_hbd = combo[1]
    hbd_avg, hbd_std, hbd_med, hbd_max, hbd_min = analysis(present_df["HBD"], "HBD", target_value_hbd)

    #target_value_logp = combo[0] * unnorm_logP_std + unnorm_logP_mean
    #target_value_logp = combo[0]
    #logp_avg, logp_std, logp_med, logp_max, logp_min = analysis(present_df["LogP"], "LogP", target_value_logp)

    #target_value_mr = combo[1] * unnorm_mr_std + unnorm_mr_mean
    #target_value_mr = combo[1] * unnorm_mr_std
    #mr_avg, mr_std, mr_med, mr_max, mr_min = analysis(present_df["MR"], "MR", target_value_mr)


    file_name = file.split(".")[0]

    if num_cond == 2:
        summary_stats.append([file_name, num_generated, val_score, novelty_score,
                              target_value_mwt, mwt_avg, mwt_std,mwt_med, mwt_max, mwt_min,
                              target_value_hbd, hbd_avg, hbd_std, hbd_med, hbd_max, hbd_min])

        present_df["MWT Target"] = target_value_mwt
        present_df["HBD Target"] = target_value_hbd

    elif num_cond == 3:

        target_value_nrb = combo[2]
        nrb_avg, nrb_std, nrb_med, nrb_max, nrb_min = analysis(present_df["Rotatable Bonds"], "NRB", target_value_nrb)

        summary_stats.append([file_name, num_generated, val_score, novelty_score,
                              target_value_mwt, mwt_avg, mwt_std,mwt_med, mwt_max, mwt_min,
                              target_value_hbd, hbd_avg, hbd_std, hbd_med, hbd_max, hbd_min,
                              target_value_nrb, nrb_avg, nrb_std, nrb_med, nrb_max, nrb_min])

        present_df["MWT Target"] = target_value_mwt
        present_df["HBD Target"] = target_value_hbd
        present_df["Rotatable Bonds Target"] = target_value_nrb


    elif num_cond == 4:

        target_value_tpsa = combo[2] * unnorm_tpsa_std
        tpsa_avg, tpsa_std, tpsa_med, tpsa_max, tpsa_min = analysis(present_df["tPSA"], "tPSA", target_value_tpsa)

        target_value_nrb = combo[3]
        nrb_avg, nrb_std, nrb_med, nrb_max, nrb_min = analysis(present_df["Rotatable Bonds"], "NRB", target_value_nrb)

        summary_stats.append([file_name, num_generated, val_score, novelty_score,
                              target_value_mwt, mwt_avg, mwt_std, mwt_med, mwt_max, mwt_min,
                              target_value_hbd, hbd_avg, hbd_std, hbd_med, hbd_max, hbd_min,
                              target_value_tpsa, tpsa_avg, tpsa_std, tpsa_med, tpsa_max, tpsa_min,
                              target_value_nrb, nrb_avg, nrb_std, nrb_med, nrb_max, nrb_min])

        present_df["MWT Target"] = target_value_mwt
        present_df["HBD Target"] = target_value_hbd
        present_df["tPSA Target"] = target_value_tpsa
        present_df["Rotatable Bonds Target"] = target_value_nrb


    if not total_df.empty:
        total_df = pd.concat((total_df, present_df))
    else:
        total_df = present_df

if num_cond == 2:
    header = ["File Name", "# generated molecules", "File Validity", "File Novelty",
              "LogP Target", "LogP Average", "LogP std", "LogP Median", "LogP Max", "LogP Min",
              "MR Target", "MR Average", "MR std", "MR Median", "MR Max", "MR Min"]

elif num_cond == 3:
    header = ["File Name", "# generated molecules", "File Validity", "File Novelty",
              "MWT Target", "MWT Average", "MWT std", "MWT Median", "MWT Max", "MWT Min",
              "HBD Target", "HBD Average", "HBD std", "HBD Median", "HBD Max", "HBD Min",
              "Rotatable Bonds Target", "Rotatable Bonds Average", "Rotatable Bonds std", "Rotatable Bonds Median",
              "Rotatable Bonds Max", "Rotatable Bonds Min",
              ]

elif num_cond == 4:
    header = ["File Name", "# generated molecules", "File Validity", "File Novelty",
              "MWT Target", "MWT Average", "MWT std", "MWT Median", "MWT Max", "MWT Min",
              "HBD Target", "HBD Average", "HBD std", "HBD Median", "HBD Max", "HBD Min",
              "tPSA Target", "tPSA Average", "tPSA std", "tPSA Median", "tPSA Max", "tPSA Min",
              "Rotatable Bonds Target", "Rotatable Bonds Average", "Rotatable Bonds std", "Rotatable Bonds Median",
              "Rotatable Bonds Max", "Rotatable Bonds Min"]


print("The validity of the generated molecules is: {}%".format(round(np.mean(validity) * 100, 2)))
print("The novelty of the generated molecules is: {}%".format(round(np.mean(novelty) * 100, 2)))

molecules = total_df["SMILES"].tolist()
total_gen = len(molecules)
novel = total_gen
unique = total_gen

i = 0
while len(molecules) > 1:
    if i % 1000 == 0:
        print("{} of {} complete to assess uniqueness.".format(i, total_gen))
    mol = molecules.pop()
    if mol in molecules:
        unique -= 1
    i += 1

uniqueness = round((unique / total_gen) * 100, 2)
print("The uniqueness of the generated molecules is: {}%".format(uniqueness))
save_file = "Unnorm_mwt_hbd_nrb_lr_5-04_unique_{}".format(str(uniqueness).replace(".", "-"))
save_directory = "J:\PythonProjects\MolGen\Results\Three_Low_Correlation_Constraints\\" + save_file
total_df.to_csv(save_directory + ".csv")
summary_stats = pd.DataFrame(summary_stats, columns=header)
summary_stats.to_csv(save_directory + "_summary_stats.csv")

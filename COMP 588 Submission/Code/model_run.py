import torch
import warnings
import pickle as pkl
import numpy as np
import pandas as pd
import mgcvae
import argparse
from rdkit import Chem, RDLogger
from graph_converter import graph_features, feature_size, graph_adjacency, graph2mol, results
from torch import optim
from datetime import datetime
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings(action='ignore')


def normalize(input_vector, num_std=4, mean=0, std=3):

    # Should z-score normalize, but somehow mutilates the values such that it ruins the results
    native_mean = np.mean(input_vector)
    native_std = np.std(input_vector)
    normalized_vector = []
    bound = mean + (num_std * std)
    neg_bound = mean - (num_std * std)

    for value in input_vector:
        norm_val = ((value - native_mean)/native_std) + mean
        norm_val = norm_val*std
        if norm_val < neg_bound:
            norm_val = neg_bound
        if norm_val > bound:
            norm_val = bound

        normalized_vector.append(norm_val)

    return normalized_vector, native_mean, native_std


def unnormalize(input_vector, native_mean, native_std, mean=0, std=3):

    unnormalized_vector = []
    for value in input_vector:
        unnorm_val = value/std
        unnorm_val = unnorm_val - mean
        unnorm_val = unnorm_val * native_std
        unnorm_val = unnorm_val + native_mean

        unnormalized_vector.append(unnorm_val)

    return unnormalized_vector


def train(model, epoch, output_dim, zero, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        condition_list = []
        graph = data.pop(0)
        graph = graph.cuda()

        # This is taking the property value and one hot encoding it
        for cond in data:
            condition_list.append(mgcvae.one_hot(cond, cond_dim, zero_value=zero).cuda())

        optimizer.zero_grad()
        recon_batch, mu, log_var = model(graph, condition_list, output_dim)
        loss = mgcvae.loss_function(recon_batch, graph, mu, log_var, output_dim)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    return train_loss / len(train_loader.dataset)


def test(model, out_dim, zero):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            condition_list = []
            graph = data.pop(0)
            graph = graph.cuda()
            for cond in data:
                condition_list.append(mgcvae.one_hot(cond, cond_dim, zero_value=zero).cuda())
            recon, mu, log_var = model(graph, condition_list, out_dim)
            test_loss += mgcvae.loss_function(recon, graph, mu, log_var, out_dim).item()
    test_loss /= len(test_loader.dataset)
    print('> Test set loss: {:.4f}'.format(test_loss))

    return test_loss


print(f"Pytorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print("Pytorch using GPU: {}".format(torch.cuda.is_available()))
print(f"Number of GPUs: {torch.cuda.device_count()}")

# all paths are specific to this machine, but should be easy enough to reconstruct if desired
abs_path = "J:\PythonProjects\MolGen\\"
parser = argparse.ArgumentParser(
    description='Small Molecular Graph Conditional Variational Autoencoder for Multi-objective Optimization (logP & Molar Refractivity)')
parser.add_argument('--size', type=int, default=16, help='molecule size (default=10)')
parser.add_argument('--dataset', type=str, default=abs_path + 'Data\ZINC\ZINC_Augmented.csv', help="dataset path (default='../QM9_Complete.csv')")
parser.add_argument('--conditions', type=str, default=abs_path + 'Data\conditions_mwt_hbd_tpsa_nrb.csv', help="conditions path (default='../conditions.csv')")
parser.add_argument('--test', type=float, default=0.1, help='test set ratio (default=0.1)')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate (default=0.00005)')
parser.add_argument('--data', type=int, default=953131, help='Sampling (default=80000)')
parser.add_argument('--batch', type=int, default=1024, help='batch size (default=100)')
parser.add_argument('--epochs', type=int, default=200, help='epoch (default=1000)')
parser.add_argument('--gen', type=int, default=1000, help='number of molecules to be generated (default=10000)')
# Read as: sample used, learning rate, batch size, max molecules, and then all conditions
parser.add_argument('--output', type=str, default=abs_path + '\Results\Three_Low_Correlation_Constraints\\Full_5-04_1024_16_Conditions',
                    help="output files path (default='../Results/Generated')")
# Test parameter zone

args, unknown = parser.parse_known_args()

print(f'- Sampling: {args.data}')
print(f'- Molecule size: {args.size}')
print(f'- Dataset: {args.dataset}')
print(f'- Conditions: {args.conditions}')
print(f'- Batch size: {args.batch}')
print(f'- Epoch: {args.epochs}')
print(f'- Test set ratio: {args.test}')
print(f'- Learning rate: {args.lr}')
print(f'- Generated molecules: {args.gen}')
print(f'- Output path: {args.output}')
current_time = datetime.now().strftime("%H:%M:%S")
print("Current Time is: ", current_time)

df = pd.read_csv(args.dataset)

# this drops all molecules larger than the designated size
df = df[df['NumAtoms'] <= args.size].reset_index(drop=True)
# There are some broken SMILES, this removes them
df = df[df['MR'] >= 0.1].reset_index(drop=True)


print('- Total allowable data:', df.shape[0])
try:
    df = df.sample(n=args.data).reset_index(drop=True)
    print('- Sampled data:', df.shape[0])
except:
    print(f'Sampling error: Set the value of --data lower than {df.shape[0]}.')
    quit()

smiles = df['SMILES'].tolist()
data = [Chem.MolFromSmiles(line) for line in smiles]

# Because the Chem library outputs a canon SMILES representation, we can use that for comparisons
canon_smiles = [Chem.MolToSmiles(line) for line in data]

# Save for faster novelty assessment
# output = open(abs_path + '\Results\Baseline_Normalized\Sample_Mols.pkl', 'wb')
# pkl.dump(canon_smiles, output)
# output.close()

# used with normalization, excluded from present experiments
# property_mean = 10
# property_std = 3
# property_bound = 3

logp = df['LogP'].tolist()
# logp, logp_native_mean, logp_native_std = normalize(logp, mean=property_mean, std=property_std, num_std=property_bound)

mr = [v/10 for v in df['MR'].tolist()]
# mr, mr_native_mean, mr_native_std = normalize(mr, mean=property_mean, std=property_std, num_std=property_bound)

tpsa = [v/10 for v in df['tPSA'].tolist()]
# tpsa, tpsa_native_mean, tpsa_native_std = normalize(tpsa, mean=property_mean, std=property_std, num_std=property_bound)

mwt = [v/25 for v in df['MWT'].tolist()]
# mwt, mwt_native_mean, mwt_native_std = normalize(mwt, mean=property_mean, std=property_std, num_std=property_bound)

nrb = df['NRB'].tolist()
hbd = df['HBD'].tolist()


atom_labels = sorted(set([atom.GetAtomicNum() for mol in data for atom in mol.GetAtoms()] + [0]))
atom_encoder_m = {l: i for i, l in enumerate(atom_labels)}
atom_decoder_m = {i: l for i, l in enumerate(atom_labels)}

bond_labels = [Chem.rdchem.BondType.ZERO] + list(
    sorted(set(bond.GetBondType() for mol in data for bond in mol.GetBonds())))
bond_encoder_m = {l: i for i, l in enumerate(bond_labels)}
bond_decoder_m = {i: l for i, l in enumerate(bond_labels)}

print('Converting to graphs...')
data_list = []
atom_number = args.size

for i in range(len(data)):
    adj_matrix = graph_adjacency(data[i], atom_number, bond_encoder_m)
    feature_matrix = feature_size(data[i], atom_labels, atom_number)

    length = [[0] for i in range(args.size)]
    length[int(df['NumAtoms'][i]) - 1] = [1]
    length = torch.tensor(length)

    data_list.append(torch.cat([length, feature_matrix,
                                adj_matrix], 1).float())

train_list = []
for i in range(len(data_list)):
    train_list.append([np.array([np.array(data_list[i])]), np.array(mwt[i]), np.array(hbd[i]),
                       np.array(nrb[i])])

bs = args.batch
tr = 1 - args.test
train_loader = torch.utils.data.DataLoader(dataset=train_list[:int(len(train_list) * tr)], batch_size=bs, shuffle=True,
                                           drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=train_list[int(len(train_list) * tr):], batch_size=bs, shuffle=True,
                                          drop_last=True)

print('- Train set:', len(train_list[:int(len(train_list) * tr)]))
print('- Test set:', len(train_list[int(len(train_list) * tr):]))

row_dim = train_list[0][0][0].shape[0]
col_dim = train_list[0][0][0].shape[1]


# essentially arbitrary, just must be large enough to accomodate adjusted values. More meaningful with normalization
cond_dim = 35
#cond_dim = (property_bound * property_std * 2) + 1

# This theoretically allows for proper one-hot indexing when normalizing
# zero_value = property_bound * property_std
zero_value = 0
out_dim = row_dim * col_dim
z_dim = 128

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
num_conditions = 3

cvae = mgcvae.CVAE(x_dim=out_dim, h_dim1=512, h_dim2=256, z_dim=z_dim, c_dim=cond_dim, num_conditions=num_conditions)
cvae = cvae.to(device)

with torch.cuda.device(1):
    optimizer = optim.Adam(cvae.parameters(), lr=args.lr)
    print('Training the model...')
    train_loss_list = []
    test_loss_list = []
    for epoch in range(1, args.epochs + 1):
        # naive adaptive learning rate, not actually used in most experiments
        # if epoch % 100 == 0 and epoch != 0:
        #     args.lr *= 2
        #     optimizer = optim.Adam(cvae.parameters(), lr=args.lr)
        #     print("New learning rate: {}".format(args.lr))
        train_loss = train(cvae, epoch, out_dim, zero_value, optimizer)
        train_loss_list.append(train_loss)
        test_loss = test(cvae, out_dim, zero_value)
        test_loss_list.append(test_loss)

    model_dir = "J:\PythonProjects\MolGen\\Models\\three_cond_low_corr_04_lr_200ep.pt"
    torch.save(cvae, model_dir)
    print('Model saved at {}'.format(model_dir))

# USE TO RESUME SAVED MODELS FOR RAPID INFERENCE WITHOUT TRAINING
#     cvae = torch.load("J:\PythonProjects\MolGen\Models\\three_cond_expanded_adaptive_lr_200ep.pt")

    print('Generating molecules...')
    conditions = pd.read_csv(args.conditions)

    # this is a somewhat messy way of ensuring the output is rich and easy to alter across different experiments
    cond_1 = [int(x) for x in list(set(conditions['MWT'].tolist())) if str(x) != 'nan']
    cond_2 = [int(x) for x in list(set(conditions['HBD'].tolist())) if str(x) != 'nan']
    cond_3 = [int(x) for x in list(set(conditions['NRB'].tolist())) if str(x) != 'nan']
    #cond_4 = [int(x) for x in list(set(conditions['NRB'].tolist())) if str(x) != 'nan']

    if num_conditions == 2:
        combinations = np.array(np.meshgrid(cond_1, cond_2)).T.reshape(-1, num_conditions)
        for combo in combinations:
            cvae_df = results(cvae, combo, args.gen, z_dim, cond_dim,
                              atom_number, atom_labels, row_dim, col_dim, atom_decoder_m, bond_decoder_m)
            cvae_df.to_csv(f'{args.output}_{combo[0]}_{combo[1]}.csv', index=False)
            print(f'Saving {args.output}_{combo[0]}_{combo[1]}.csv ({cvae_df.shape[0]})...')
    elif num_conditions == 3:
        combinations = np.array(np.meshgrid(cond_1, cond_2, cond_3)).T.reshape(-1, num_conditions)
        print(combinations)
        for combo in combinations:
            cvae_df = results(cvae, combo, args.gen, z_dim, cond_dim,
                              atom_number, atom_labels, row_dim, col_dim, atom_decoder_m, bond_decoder_m)
            cvae_df.to_csv(f'{args.output}_{combo[0]}_{combo[1]}_{combo[2]}.csv', index=False)
            print(f'Saving {args.output}_{combo[0]}_{combo[1]}_{combo[2]}.csv ({cvae_df.shape[0]})...')
    elif num_conditions == 4:
        combinations = np.array(np.meshgrid(cond_1, cond_2, cond_3, cond_4)).T.reshape(-1, num_conditions)
        for combo in combinations:
            cvae_df = results(cvae, combo, args.gen, z_dim, cond_dim,
                              atom_number, atom_labels, row_dim, col_dim, atom_decoder_m, bond_decoder_m)
            cvae_df.to_csv(f'{args.output}_{combo[0]}_{combo[1]}_{combo[2]}_{combo[3]}.csv', index=False)
            print(f'Saving {args.output}_{combo[0]}_{combo[1]}_{combo[2]}_{combo[3]}.csv ({cvae_df.shape[0]})...')


current_time = datetime.now().strftime("%H:%M:%S")
print("Current Time is: ", current_time)
print('Done!')

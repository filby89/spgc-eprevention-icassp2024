import torch
import argparse
from model import TransformerClassifier
from dataset import PatientDataset
import pickle
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import sklearn
from sklearn.svm import OneClassSVM

def parse():
    '''Returns args passed to the train.py script.'''
    parser = argparse.ArgumentParser()

    # transformer parameters
    parser.add_argument('--window_size', type=int, default=48)
    parser.add_argument('--input_features', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--nlayers', type=int, default=2)

    # num_patients - 10 for track 1, 9 for track 2
    parser.add_argument('--num_patients', type=int, default=10)

    # input paths
    parser.add_argument('--features_path', type=str, help='features to use')
    parser.add_argument('--dataset_path', type=str, help='dataset path') # to get relapse labels
    parser.add_argument('--submission_path', type=str, help='where to save the submission files') # to get relapse labels

    # checkpoint
    parser.add_argument('--load_path', type=str, help='path to saved model', default='checkpoints/best.pth.tar')
    parser.add_argument('--scaler_path', type=str, help='path to saved scaler', default='checkpoints/scaler.pkl')

    parser.add_argument('--device', type=str, help='device to use (cpu, cuda, cuda[number])', default='cuda')

    parser.add_argument('--mode', type=str, help='val, test', default='test')

    args = parser.parse_args()
    args.seq_len = args.window_size
            
    return args

def main():

    # parse arguments
    args = parse()
    window_size = args.window_size

            
    columns_to_scale = ['acc_norm', 'heartRate_mean', 'rRInterval_mean', 'rRInterval_rmssd', 'rRInterval_sdnn', 'rRInterval_lombscargle_power_high']
    data_columns = columns_to_scale + ['sin_t', 'cos_t']

    device = args.device
    print('Using device', args.device)

    # Model
    model = TransformerClassifier(vars(args))
    model.to(device)

    # load checkpoint
    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint)
    model.eval()
    torch.set_grad_enabled(False)

    # load scaler
    with open(args.scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # first calculate the one class svms
    print('Getting features from train distribution...')

    # add a collate fn which ignores None
    def collate_fn(batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    train_dataset = PatientDataset(features_path=args.features_path, 
                                   dataset_path=args.dataset_path,
                                   mode='train', window_size=args.window_size)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True, collate_fn=collate_fn)

    all_features_train = []
    all_labels_train = []

    for batch in tqdm(train_loader):
        if batch is None:
            continue

        x = batch['data'].to(device)
        user_ids = batch['user_id'].to(device)
        
        # Forward
        _, features = model(x)

        # append features
        all_features_train.append(features.detach().cpu())
        all_labels_train.append(user_ids.detach().cpu())

    all_features_train = torch.vstack(all_features_train).numpy()
    all_labels_train = torch.hstack(all_labels_train).numpy()

    # ----- train the outlier detection of this epoch ----- #
    print('Training Robust Covariance...')
    clfs = []

    from sklearn.covariance import EllipticEnvelope
    # -------- train elliptic envelopes to predict scores from classification features -------- # 
    for subject in range(args.num_patients):
        subject_features_train = all_features_train[all_labels_train==subject]

        clf = EllipticEnvelope(support_fraction=1).fit(subject_features_train)
        # clf = OneClassSVM().fit(subject_features_train)
        clfs.append(clf)

    all_auroc = []
    all_auprc = []

    random_auroc = []
    random_auprc = []

    for patient in os.listdir(args.features_path):
        patient_dir = os.path.join(args.features_path, patient)

        user_relapse_labels = []
        user_anomaly_scores = []
        relapse_shapes = 0

        for subfolder in os.listdir(patient_dir):
            if (args.mode == 'val' and 'val' in subfolder) or (args.mode == 'test' and 'test' in subfolder):
                subfolder_dir = os.path.join(patient_dir, subfolder)
                file = 'features.csv'
                file_path = os.path.join(subfolder_dir, file)
                df = pd.read_csv(file_path, index_col=0)
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.dropna()
            
                relapse_df = pd.read_csv(os.path.join(args.dataset_path, patient, subfolder, 'relapses.csv'))
                # IMPORTANT - drop last row as it was falsely added by the organizers
                relapse_df = relapse_df.iloc[:-1]

                # count 0 and 1 to calculate random chance
                if args.mode != 'test':
                    relapse_shapes += relapse_df['relapse'].sum()

                day_indices = relapse_df['day_index'].unique() # get all day indices for this patient

                for day_index in day_indices:
                    day_data = df[df['day_index'] == day_index].copy()

                    relapse_label = relapse_df[relapse_df['day_index'] == day_index]['relapse'].to_numpy()[0]

                    patient_id = int(patient[1:]) - 1

                    if len(day_data) < args.window_size:
                        # predict zero anomaly score for days without enough data - right in the middle of the inlier/outlier of robust covariance
                        relapse_df.loc[relapse_df['day_index'] == day_index, 'anomaly_score'] = 0
                        user_anomaly_scores.append(0)
                        user_relapse_labels.append(relapse_label)
                        continue
                    
                    sequences = []
                    if len(day_data) == window_size:
                        start_idx = 0
                        sequence = day_data.iloc[start_idx:start_idx + window_size]
                        sequence = sequence[data_columns].copy().to_numpy()
                        sequence[:, :-2] = scaler.transform(sequence[:, :-2])
                        sequences.append(sequence)
                    else:
                        for start_idx in range(0, len(day_data) - window_size, window_size//3): # 1/3 overlap
                            sequence = day_data.iloc[start_idx:start_idx + window_size]
                            sequence = sequence[data_columns].copy().to_numpy()
                            sequence[:, :-2] = scaler.transform(sequence[:, :-2])
                            sequences.append(sequence)
                    sequence = np.stack(sequences)
                    sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
                    sequence_tensor = sequence_tensor.permute(0, 2, 1)


                    # Forward
                    logits, features = model(sequence_tensor.to(device))


                    current_user_clf = clfs[patient_id]
                    features = features.detach().cpu().numpy()


                    anomaly_score = -current_user_clf.decision_function(features)

                    anomaly_score = anomaly_score.mean()


                    # add this to the relapse_df
                    relapse_df.loc[relapse_df['day_index'] == day_index, 'anomaly_score'] = anomaly_score
                    user_anomaly_scores.append(anomaly_score)
                    user_relapse_labels.append(relapse_label)

                # save subfolder in submission_path
                os.makedirs(os.path.join(args.submission_path, patient, subfolder), exist_ok=True)
                relapse_df.to_csv(os.path.join(args.submission_path, patient, subfolder, 'submission.csv'))

        user_anomaly_scores = np.array(user_anomaly_scores)
        user_relapse_labels = np.array(user_relapse_labels)

        # if mode is not test, calculate scores
        if args.mode != 'test':

            # create df per user if you want to see per user anomaly scores/relapses
            # user_df = pd.DataFrame()
            # user_df['anomaly_score'] = user_anomaly_scores
            # user_df['relapse'] = user_relapse_labels
            # # sort by anomaly score
            # user_df = user_df.sort_values(by=['anomaly_score'], ascending=False)
            # user_df.to_csv("{}.csv".format(patient), index=False)

            # Compute ROC Curve
            precision, recall, _ = sklearn.metrics.precision_recall_curve(user_relapse_labels, user_anomaly_scores)

            fpr, tpr, _ = sklearn.metrics.roc_curve(user_relapse_labels, user_anomaly_scores)

            # # Compute AUROC
            auroc = sklearn.metrics.auc(fpr, tpr)

            # # Compute AUPRC
            auprc = sklearn.metrics.auc(recall, precision)

            random_auroc.append(0.5)

            random_auprc.append(user_relapse_labels.mean())
            all_auroc.append(auroc)
            all_auprc.append(auprc)
            # auprc = pr_auc_score(user_relapse_labels, user_anomaly_scores)
            print(f'USER: {patient}, AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, Random AUPRC: {user_relapse_labels.mean():.4f}')

    if args.mode != 'test':

        total_auroc = sum(all_auroc)/len(all_auroc)
        total_auprc = sum(all_auprc)/len(all_auprc)
        random_auroc = sum(random_auroc)/len(random_auroc)
        random_auprc = sum(random_auprc)/len(random_auprc)
        total_avg = (total_auroc + total_auprc) / 2
        print(f'Total AUROC: {total_auroc:.4f}, Total AUPRC: {total_auprc:.4f}, Total AVG: {total_avg:.4f}, Random AUROC: {random_auroc:.4f}, Random AUPRC: {random_auprc:.4f}, Ideal AVG: {(random_auroc + random_auprc)/2:.4f}')





if __name__ == '__main__':
    main()

from collections import defaultdict

import numpy as np
import pandas as pd


def get_zero_shot_train_data():
    """
    Load the positive PanPep base dataset and filter to only keep peptides with 5 or more binding TCRs.

    Returns: The positive PanPep zeroshot train dataset.
    """
    base_dataset = pd.read_csv("Data/base_dataset.csv")
    pep_counts = base_dataset['peptide'].value_counts()
    train_peps = pep_counts[pep_counts >= 5]
    train_data = base_dataset[base_dataset['peptide'].isin(train_peps.index)].rename(
        columns={'peptide': 'Peptide', 'binding_TCR': 'CDR3'})
    return train_data


def check_overlap(train_df, test_df):
    """
    Check the overlap between a positive test dataset and train dataset. Calculates how many samples of the test
    dataset were also in the train dataset, similar for epitopes, cdr3s and unique epitopes and unique cdr3s.
    Args:
        train_df:   a dataframe with the train dataset
        test_df:    a dataframe with the test dataset

    Returns: All necessary statistics
    """

    full_in = pd.merge(train_df, test_df, how='inner', on=['Peptide', 'CDR3'])
    pep_in = pd.Series(list(set(train_df['Peptide']) & set(test_df['Peptide'])))
    pep_samples_in = test_df[test_df['Peptide'].isin(pep_in)]
    cdr3_in = pd.Series(list(set(train_df['CDR3']) & set(test_df['CDR3'])))
    cdr3_samples_in = test_df[test_df['CDR3'].isin(cdr3_in)]
    return {
        'full_samples_in': len(full_in),
        'full_num': len(test_df),
        'pep_in': len(pep_in),
        'pep_num': len(test_df['Peptide'].unique()),
        'pep_samples_in': len(pep_samples_in),
        'cdr3_in': len(cdr3_in),
        'cdr3_num': len(test_df['CDR3'].unique()),
        'cdr3_samples_in': len(cdr3_samples_in)
    }


def check_data_overlap():
    """
    Print an overview of the overlap between the positive 5-fold cross-validation test dataset and the PanPep train
    data, an average over the 5 cross-validation folds is given.
    """

    train_df = get_zero_shot_train_data()
    cv_overlap = defaultdict(list)
    for cv in range(5):
        test_df = pd.read_csv(f"Data/cross-validation_shuffled-negatives_splits/test_fold_{cv}.csv")
        test_df = test_df[test_df['y'] == 1]
        overlap = check_overlap(train_df, test_df)
        for k, v in overlap.items():
            cv_overlap[k].append(v)

    print(f"{np.mean(cv_overlap['full_samples_in']):.1f}/{np.mean(cv_overlap['full_num']):.1f} "
          f"({np.mean(cv_overlap['full_samples_in']) / np.mean(cv_overlap['full_num']) * 100:.2f}%) "
          f"postive test samples were already in the train data")
    print(f"{np.mean(cv_overlap['pep_samples_in']):.1f}/{np.mean(cv_overlap['full_num']):.1f} "
          f"({np.mean(cv_overlap['pep_samples_in']) / np.mean(cv_overlap['full_num']) * 100:.2f}%) "
          f"positive test samples had a epitope already in the train data")
    print(f"{np.mean(cv_overlap['cdr3_samples_in']):.1f}/{np.mean(cv_overlap['full_num']):.1f} "
          f"({np.mean(cv_overlap['cdr3_samples_in']) / np.mean(cv_overlap['full_num']) * 100:.2f}%) "
          f"positive test samples had a cdr3 already in the train data")

    print(f"{np.mean(cv_overlap['pep_in']):.1f}/{np.mean(cv_overlap['pep_num']):.1f} "
          f"({np.mean(cv_overlap['pep_in']) / np.mean(cv_overlap['pep_num']) * 100:.2f}%) "
          f"unique epitopes were already in the train data")
    print(f"{np.mean(cv_overlap['cdr3_in']):.1f}/{np.mean(cv_overlap['cdr3_num']):.1f} "
          f"({np.mean(cv_overlap['cdr3_in']) / np.mean(cv_overlap['cdr3_num']) * 100:.2f}%) "
          f"unique cdr3 sequences were already in the train data")


if __name__ == "__main__":
    check_data_overlap()

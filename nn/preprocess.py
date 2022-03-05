# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
from collections import Counter

# Defining preprocessing functions
def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one hot encoding of a list of nucleic acid sequences
    for use as input into a fully connected neural net.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence
            length due to the one hot encoding scheme for nucleic acids.

            For example, if we encode 
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    """
    code = {'A':[1, 0, 0, 0], 'T':[0, 1, 0, 0], 'C':[0, 0, 1, 0], 'G':[0, 0, 0, 1]}
    encodings = []
    for seq in seq_arr:
        encoding = np.array([code[c] for c in seq])
        encodings.append(encoding.flatten())
    return np.array(encodings)

def sample_seqs(
        seqs: List[str],
        labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample your sequences to account for class imbalance. 
    Consider this as a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    counter = Counter(labels)
    labels = np.array(labels)
    max_size = np.max(list(counter.values()))
    resampled_indices = np.array([], dtype = int)
    for label, count in counter.items():
        indices = np.where(labels == label)[0]
        if count != max_size:
            indices = np.random.choice(indices, max_size)
        resampled_indices = np.concatenate((resampled_indices, indices))    
    seqs = np.array(seqs)
    sampled_seqs = seqs[resampled_indices]
    sampled_labels = labels[resampled_indices]
    return sampled_seqs.tolist(), sampled_labels.tolist()     
        


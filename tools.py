import os
import random
import torch
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
import csv


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics(y, pred):
    f1_macro = f1_score(y, pred, average='macro')

    f1_weighted = f1_score(y, pred, average='weighted')
    return f1_macro, f1_weighted


def get_sample_pairs(x, y, num_classes, top_pos_percent=30, bottom_neg_percent=20):
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    sim_matrix = cosine_similarity(x, x)

    class_samples = {i: np.where(y == i)[0] for i in range(num_classes)}

    positive_pairs = []
    negative_pairs = []

    for class_id, samples in class_samples.items():
        if len(samples) < 2:
            continue

        sub_sim_matrix = sim_matrix[np.ix_(samples, samples)]

        triu_indices = np.triu_indices_from(sub_sim_matrix, k=1)
        triu_values = sub_sim_matrix[triu_indices]

        num_top_samples = max(1, int(len(triu_values) * top_pos_percent / 100))
        top_indices = np.argsort(-triu_values)[:num_top_samples]
        random_indices = np.random.choice(len(triu_values),
                                          size=min(len(top_indices), len(triu_values) // 4),
                                          replace=False)
        final_indices = np.concatenate([top_indices[:len(top_indices) // 2], random_indices])
        pos_pairs = [(samples[triu_indices[0][i]], samples[triu_indices[1][i]]) for i in final_indices]
        positive_pairs.extend(pos_pairs)

    for i in range(num_classes):
        class_i_samples = class_samples[i]
        for j in range(i + 1, num_classes):
            class_j_samples = class_samples[j]
            cross_sim_matrix = sim_matrix[np.ix_(class_i_samples, class_j_samples)]

            flat_sim = cross_sim_matrix.flatten()

            num_bottom_samples = max(1, int(len(flat_sim) * bottom_neg_percent / 100))
            bottom_indices = np.argsort(-flat_sim)[:num_bottom_samples]
            neg_pairs = [(class_i_samples[divmod(idx, cross_sim_matrix.shape[1])[0]],
                          class_j_samples[divmod(idx, cross_sim_matrix.shape[1])[1]])
                         for idx in bottom_indices]
            negative_pairs.extend(neg_pairs)

    return np.array(positive_pairs), np.array(negative_pairs)


def model_parameters(sample, parser):
    if sample == 'BRCA':
        parser.add_argument('--k_s', type=int, default=5)
        parser.add_argument('--lr', type=float, default=0.0003)
        parser.add_argument('--epoch', type=int, default=200)
        parser.add_argument('--mrna_meth_input_size', type=int, default=1000)
        parser.add_argument('--mrna_meth_hidden_size', type=int, default=512)
        parser.add_argument('--mirna_input_size', type=int, default=503)
        parser.add_argument('--mirna_hidden_size', type=int, default=256)
        parser.add_argument('--sample_output_size', type=int, default=128)
        parser.add_argument('--mlp_hidden_size', type=int, default=16)
        parser.add_argument('--num_classes', type=int, default=5)
    elif sample == 'ACC':
        parser.add_argument('--k_s', type=int, default=8)
        parser.add_argument('--lr', type=float, default=0.0003)
        parser.add_argument('--epoch', type=int, default=200)
        parser.add_argument('--mrna_input_size', type=int, default=1000)
        parser.add_argument('--mrna_hidden_size', type=int, default=512)
        parser.add_argument('--sample_output_size', type=int, default=128)
        parser.add_argument('--mlp_hidden_size', type=int, default=16)
        parser.add_argument('--num_classes', type=int, default=4)
    elif sample == 'MESO':
        parser.add_argument('--k_s', type=int, default=4)
        parser.add_argument('--lr', type=float, default=0.0003)
        parser.add_argument('--epoch', type=int, default=200)
        parser.add_argument('--mrna_input_size', type=int, default=1000)
        parser.add_argument('--mrna_hidden_size', type=int, default=512)
        parser.add_argument('--sample_output_size', type=int, default=128)
        parser.add_argument('--mlp_hidden_size', type=int, default=16)
        parser.add_argument('--num_classes', type=int, default=4)
    elif sample == 'UCS':
        parser.add_argument('--k_s', type=int, default=5)
        parser.add_argument('--lr', type=float, default=0.0003)
        parser.add_argument('--epoch', type=int, default=200)
        parser.add_argument('--mrna_input_size', type=int, default=1000)
        parser.add_argument('--mrna_hidden_size', type=int, default=512)
        parser.add_argument('--sample_output_size', type=int, default=128)
        parser.add_argument('--mlp_hidden_size', type=int, default=16)
        parser.add_argument('--num_classes', type=int, default=4)
    args = parser.parse_args()
    return args


def save_result_once(result_file, acc, f1_macro, f1_weighted):
    file_exists = os.path.exists(result_file)

    with open(result_file, "a", newline="") as csvfile:
        fieldnames = ["Acc", "F1_macro", "F1_weighted"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "Acc": acc,
            "F1_macro": f1_macro,
            "F1_weighted": f1_weighted
        })


def load_and_compute_stats(result_file):
    accs, f1_macros, f1_weighteds = [], [], []

    with open(result_file, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            accs.append(float(row["Acc"]))
            f1_macros.append(float(row["F1_macro"]))
            f1_weighteds.append(float(row["F1_weighted"]))

    mean_acc = np.mean(accs) * 100
    mean_f1_macro = np.mean(f1_macros) * 100
    mean_f1_weighted = np.mean(f1_weighteds) * 100

    var_acc = np.var(accs, ddof=1) * 100
    var_f1_macro = np.var(f1_macros, ddof=1) * 100
    var_f1_weighted = np.var(f1_weighteds, ddof=1) * 100

    print(f"Acc: {mean_acc:.2f}±{var_acc:.2f}")
    print(f"F1_macro: {mean_f1_macro:.2f}±{var_f1_macro:.2f}")
    print(f"F1_weighted: {mean_f1_weighted:.2f}±{var_f1_weighted:.2f}")

    return mean_acc, mean_f1_macro, mean_f1_weighted

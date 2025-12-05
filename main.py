import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import sourceModels, Target, final_loss
from data_processing import get_sample_graph
from tools import model_parameters, set_seed, get_sample_pairs, metrics, save_result_once, load_and_compute_stats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_source(model_source, mrna_sample_data,
                 meth_sample_data, mirna_sample_data):
    epochs = source_args.epoch

    optimizer = optim.Adam(model_source.parameters(), lr=source_args.lr, weight_decay=5e-4)
    source_criterion = nn.CrossEntropyLoss()
    train_loss_list = []
    test_loss_list = []
    acc_source_list = []

    best_acc = 0.0

    for epoch in range(epochs):
        model_source.train()
        optimizer.zero_grad()
        class_output, mrna_sample_out, meth_sample_out, mirna_sample_out, sample_out = model_source(mrna_sample_data,
                                                                                                    meth_sample_data,
                                                                                                    mirna_sample_data)

        mrna_loss, mrna_reg, mrna_contrast = final_loss(mrna_sample_out, mrna_sample_out[mrna_sample_data.train_mask],
                                                        mrna_sample_data.edge_index, mrna_sample_out.size(0),
                                                        mrna_train_pairs[0], mrna_train_pairs[1])
        meth_loss, meth_reg, meth_contrast = final_loss(meth_sample_out, meth_sample_out[meth_sample_data.train_mask],
                                                        meth_sample_data.edge_index, meth_sample_out.size(0),
                                                        meth_train_pairs[0], meth_train_pairs[1])
        mirna_loss, mirna_reg, mirna_contrast = final_loss(mirna_sample_out,
                                                           mirna_sample_out[mirna_sample_data.train_mask]
                                                           , mirna_sample_data.edge_index, mirna_sample_out.size(0),
                                                           mirna_train_pairs[0],
                                                           mirna_train_pairs[1])
        cross_loss = source_criterion(class_output[mrna_sample_data.train_mask],
                                      mrna_sample_data.y[mrna_sample_data.train_mask])
        if epoch <= 40:
            L_total = cross_loss + mrna_loss + meth_loss + mirna_loss
        else:
            L_total = cross_loss + mrna_contrast + meth_contrast + mirna_contrast

        L_total.backward()
        optimizer.step()

        test_loss, acc_source = eval_source(model_source, mrna_sample_data, meth_sample_data, mirna_sample_data)
        if acc_source > best_acc:
            best_acc = acc_source
            best_model = model_source.state_dict()

            torch.save(best_model, f"source_{source_sample}.pth")

        print(f"Epoch {epoch + 1}, Loss: {L_total.item():.4f}, Test Loss: {test_loss:.4f}, "
              f"Acc source: {acc_source:.4f}, Best Acc: {best_acc:.4f}")
        train_loss_list.append(L_total.item())
        test_loss_list.append(test_loss)
        acc_source_list.append(acc_source)


def eval_source(model, mrna_sample_data, meth_sample_data, mirna_sample_data):
    model.eval()
    with torch.no_grad():
        class_output, mrna_sample_out, meth_sample_out, mirna_sample_out, sample_out = model(mrna_sample_data,
                                                                                             meth_sample_data,
                                                                                             mirna_sample_data)

        test_loss = nn.CrossEntropyLoss()(sample_out[mrna_sample_data.test_mask],
                                          mrna_sample_data.y[mrna_sample_data.test_mask])
        pred = class_output[mrna_sample_data.test_mask].argmax(dim=1)
        correct = pred == mrna_sample_data.y[mrna_sample_data.test_mask]
        acc = int(correct.sum()) / int(mrna_sample_data.test_mask.sum())

        return test_loss.item(), acc


def train_target(model_target, model_source, pt_sample_data):
    epochs = target_args.epoch

    model_target.conv1.load_state_dict(model_source.mrna_sample_gat_model.conv1.state_dict())
    model_target.conv2.load_state_dict(model_source.mrna_sample_gat_model.conv2.state_dict())
    model_target.conv3.load_state_dict(model_source.mrna_sample_gat_model.conv3.state_dict())
    model_target.residual_transform1.load_state_dict(
        model_source.mrna_sample_gat_model.residual_transform1.state_dict())
    model_target.residual_transform2.load_state_dict(
        model_source.mrna_sample_gat_model.residual_transform2.state_dict())
    for param in model_target.parameters():
        param.requires_grad = False
    for param in model_target.mlp.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model_target.parameters(), lr=target_args.lr, weight_decay=5e-4)
    target_criterion = nn.CrossEntropyLoss()
    train_loss_list = []
    test_loss_list = []
    acc_source_list = []
    acc_target_list = []
    best_acc = 0.0
    best_model = None

    best_f1_m = 0.0
    best_f1_w = 0.0

    for epoch in range(epochs):
        model_target.train()
        model_source.eval()
        optimizer.zero_grad()

        class_output, _ = model_target(target_sample_data.x, target_sample_data.edge_index)
        cross_loss = target_criterion(class_output[pt_sample_data.train_mask],
                                      pt_sample_data.y[pt_sample_data.train_mask])

        L_total = cross_loss
        L_total.backward()
        optimizer.step()

        test_loss, acc_target, f1_macro, f1_weighted = eval_target(model_target, model_source, pt_sample_data)
        print(f"Epoch {epoch + 1}, Loss: {L_total.item():.4f}, Test Loss: {test_loss:.4f}, "
              f"Acc target: {acc_target:.4f}")
        train_loss_list.append(L_total.item())
        test_loss_list.append(test_loss)
        acc_target_list.append(acc_target)
        if acc_target >= best_acc:
            best_acc = acc_target
            best_model = model_target.state_dict()

            best_f1_m = f1_macro
            best_f1_w = f1_weighted
            # torch.save(best_model, f"dann_target_{target_sample}.pth")
        print(
            f"Best model saved with accuracy: {best_acc:.4f}, F1 Macro: {best_f1_m:.4f}, F1 Weighted: {best_f1_w:.4f}")
    save_result_once(f'./results_{target_sample}.txt', best_acc, best_f1_m, best_f1_w)
    load_and_compute_stats(f'./results_{target_sample}.txt')


def eval_target(model_target, model_source, target_sample_data):
    model_target.eval()
    model_source.eval()
    with torch.no_grad():
        class_output, _ = model_target(target_sample_data.x, target_sample_data.edge_index)

        cross_loss = nn.CrossEntropyLoss()(class_output[target_sample_data.test_mask],
                                           target_sample_data.y[target_sample_data.test_mask])

        L_total = cross_loss

        pred = class_output[target_sample_data.test_mask].argmax(dim=1)
        correct = pred == target_sample_data.y[target_sample_data.test_mask]
        acc = int(correct.sum()) / int(target_sample_data.test_mask.sum())
        f1_macro, f1_weighted = metrics(target_sample_data.y[target_sample_data.test_mask].cpu().numpy(),
                                        pred.cpu().numpy())
        return L_total.item(), acc, f1_macro, f1_weighted


if __name__ == "__main__":

    source_sample = "BRCA"
    target_sample = 'ACC'   # ACC / MESO / UCS

    source_parser = argparse.ArgumentParser()
    target_parser = argparse.ArgumentParser()

    source_args = model_parameters(source_sample, source_parser)
    target_args = model_parameters(target_sample, target_parser)
    for seed in [42, 43, 44, 45, 46]:
        set_seed(seed)

        source_sample_k = source_args.k_s
        target_sample_k = target_args.k_s

        mrna_sample_data = get_sample_graph(source_sample, 1, source_sample_k).to(
            device)
        meth_sample_data = get_sample_graph(source_sample, 2, source_sample_k).to(
            device)
        mirna_sample_data = get_sample_graph(source_sample, 3, source_sample_k).to(
            device)

        target_sample_data = get_sample_graph(target_sample, 1, target_sample_k).to(device)

        source_number_classes = source_args.num_classes
        target_number_classes = target_args.num_classes
        mrna_train_pairs = get_sample_pairs(mrna_sample_data.x[mrna_sample_data.train_mask],
                                            mrna_sample_data.y[mrna_sample_data.train_mask], source_number_classes)
        meth_train_pairs = get_sample_pairs(meth_sample_data.x[meth_sample_data.train_mask],
                                            meth_sample_data.y[meth_sample_data.train_mask], source_number_classes)
        mirna_train_pairs = get_sample_pairs(mirna_sample_data.x[mirna_sample_data.train_mask],
                                             mirna_sample_data.y[mirna_sample_data.train_mask],
                                             source_number_classes)
        mrna_test_pairs = get_sample_pairs(mrna_sample_data.x[mrna_sample_data.test_mask],
                                           mrna_sample_data.y[mrna_sample_data.test_mask], source_number_classes)
        meth_test_pairs = get_sample_pairs(meth_sample_data.x[meth_sample_data.test_mask],
                                           meth_sample_data.y[meth_sample_data.test_mask], source_number_classes)
        mirna_test_pairs = get_sample_pairs(mirna_sample_data.x[mirna_sample_data.test_mask],
                                            mirna_sample_data.y[mirna_sample_data.test_mask],
                                            source_number_classes)
        source = sourceModels(source_args).to(device)

        target = Target(target_args.mrna_input_size, target_args.mrna_hidden_size,
                        target_args.sample_output_size, target_number_classes).to(device)
        train_source(source, mrna_sample_data, meth_sample_data, mirna_sample_data)
        source.load_state_dict(torch.load(f"source_{source_sample}.pth"))

        train_target(target, source, target_sample_data)

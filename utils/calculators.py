import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from utils.common import get_cell_index


def calculate_weights_labels(path, dataset, dataloader, nb_classes):

    z = np.zeros((nb_classes,))
    print('='*30, '=>Calculating classes weights')
    for (images, labels) in tqdm(dataloader):
        labels = labels.detach().cpu().numpy()
        mask = (labels >= 0) & (labels < nb_classes)
        labels = labels[mask].astype(np.uint8)
        count_1 = np.bincount(labels, minlength=nb_classes)
        z += count_1
    total_frequency = np.sum(z)
    class_weights = []
    for freq in z:
        class_weight = 1 / (np.log(1.02 + (freq / total_frequency)))
        class_weights.append(class_weight)

    # TODO save class_weigths
    ret = np.array(class_weights)
    classes_weights_path = os.path.join(path, dataset+'_classes_weights.npy')
    np.save(classes_weights_path, ret)

    return ret

def calculate_network_level_search_space_with_constraint():
    # 1189 * 36 = 42804

    counter = torch.zeros((13, 4), dtype=torch.int)
    counter[0][0]=1
    for i in range(1, 13):
        for j in range(4):
            if i == 1:
                if j == 0: counter[i][j] = counter[i-1][j]
                elif j == 1: counter[i][j] = counter[i-1][j-1]
            elif i == 2:
                if j == 0:
                    counter[i][j] = counter[i-1][j]
                elif j == 1:
                    counter[i][j] = counter[i-1][j-1] + counter[i-1][j]
                elif j == 2:
                    counter[i][j] = counter[i-1][j-1]
            elif i == 3:
                if j == 0:
                    counter[i][j] = counter[i-1][j]
                elif j == 1:
                    counter[i][j] = counter[i-1][j-1] + counter[i-1][j]
                elif j == 2:
                    counter[i][j] = counter[i-1][j-1] + counter[i-1][j]
                elif j == 3:
                    counter[i][j] = counter[i-1][j-1]
            else:
                if i <= 8:
                    if j == 0:
                        counter[i][j] = counter[i-1][j]
                    elif j == 1:
                        counter[i][j] = counter[i-1][j-1] + counter[i-1][j]
                    elif j == 2:
                        counter[i][j] = counter[i-1][j-1] + counter[i-1][j]
                    elif j == 3:
                        counter[i][j] = counter[i-1][j-1] + counter[i-1][j]
                else:
                    if j == 0:
                        counter[i][j] = counter[i-1][j] + counter[i-1][j+1]
                    elif j == 1:
                        counter[i][j] = counter[i-1][j+1] + counter[i-1][j]
                    elif j == 2:
                        counter[i][j] = counter[i-1][j+1] + counter[i-1][j]
                    elif j == 3:
                        counter[i][j] =  counter[i-1][j]
    print(counter)
    print(sum(counter[-1, :]))

def calculate_network_level_search_space():
    # 75025 * 36 = 2700900

    counter = torch.zeros((13,4), dtype=torch.int)
    counter[0][0] = 1
    for i in range(1, 13):
        for j in range(4):
            if i == 1:
                if j == 0: counter[i][j] = counter[i-1][j]
                elif j == 1: counter[i][j] = counter[i-1][j-1]
            elif i == 2:
                if j == 0:
                    counter[i][j] = counter[i-1][j] + counter[i-1][j+1]
                elif j == 1:
                    counter[i][j] = counter[i-1][j-1] + counter[i-1][j]
                elif j == 2:
                    counter[i][j] = counter[i-1][j-1]
            elif i == 3:
                if j == 0:
                    counter[i][j] = counter[i-1][j] + counter[i-1][j+1]
                elif j == 1:
                    counter[i][j] = counter[i-1][j-1] + counter[i-1][j] + counter[i-1][j+1]
                elif j == 2:
                    counter[i][j] = counter[i-1][j-1] + counter[i-1][j]
                elif j == 3:
                    counter[i][j] = counter[i-1][j-1]
            else:
                if j == 0:
                    counter[i][j] = counter[i-1][j] + counter[i-1][j+1]
                elif j == 1:
                    counter[i][j] = counter[i-1][j-1] + counter[i-1][j] + counter[i-1][j+1]
                elif j == 2:
                    counter[i][j] = counter[i-1][j-1] + counter[i-1][j] + counter[i-1][j+1]
                elif j == 3:
                    counter[i][j] = counter[i-1][j-1] + counter[i-1][j]
    print(counter)
    print(sum(counter[-1, :]))


# TODO: need test
def calculate_derived_model_entropy(search_checkpoint, search_arch_checkpoint, nb_layers=12, eps=1e-8):
    # need checkpoint in search phase.
    # return :: network_arch_entropy, cell_arch_entropy
    checkpoint = torch.load(search_checkpoint)
    arch_checkpoint = torch.load(search_arch_checkpoint)
    actual_path = arch_checkpoint['actual_path']

    network_arch_parameters = checkpoint['state_dict']['network_arch_parameters']
    network_arch_entropy = 0.
    cell_arch_entropy = 0.
    current_scale = 0
    for layer in range(nb_layers):
        next_scale = int(actual_path[layer])
        cell_index = get_cell_index(layer, current_scale, next_scale)

        # cell entropy
        cell_probs = F.softmax(checkpoint['state_dict']['cells.{:}.cell_arch_parameters'.format(cell_index)].cell_arch_parameters, -1)
        cell_log_probs = torch.log(cell_probs + eps)
        cell_entropy = - torch.sum(torch.mul(cell_probs,
                                             cell_log_probs))  # / torch.log(torch.tensor(len(self.conv_candidates), dtype=torch.float))
        # network node entropy
        if next_scale == 0:
            if layer == 0:
                network_probs = F.softmax(network_arch_parameters[layer][next_scale][1], -1) * (1 / 3)
                network_log_probs = torch.log(network_probs + eps)
                network_entropy = - torch.sum(
                    torch.mul(network_probs, network_log_probs))  # / torch.log(torch.tensor(1, dtype=torch.float))
            else:
                network_probs = F.softmax(network_arch_parameters[layer][next_scale][1:], -1) * (2 / 3)
                network_log_probs = torch.log(network_probs + eps)
                network_entropy = - torch.sum(
                    torch.mul(network_probs, network_log_probs))  # / torch.log(torch.tensor(2, dtype=torch.float))
        elif next_scale == 1:
            if layer == 0:
                network_probs = F.softmax(network_arch_parameters[layer][next_scale][0], -1) * (1 / 3)
                network_log_probs = torch.log(network_probs + eps)
                network_entropy = - torch.sum(
                    torch.mul(network_probs, network_log_probs))  # / torch.log(torch.tensor(1, dtype=torch.float))
            elif layer == 1:
                network_probs = F.softmax(network_arch_parameters[layer][next_scale][:2], -1) * (2 / 3)
                network_log_probs = torch.log(network_probs + eps)
                network_entropy = - torch.sum(
                    torch.mul(network_probs, network_log_probs))  # / torch.log(torch.tensor(2, dtype=torch.float))
            else:
                network_probs = F.softmax(network_arch_parameters[layer][next_scale], -1)
                network_log_probs = torch.log(network_probs + eps)
                network_entropy = - torch.sum(
                    torch.mul(network_probs, network_log_probs))  # / torch.log(torch.tensor(3, dtype=torch.float))
        elif next_scale == 2:
            if layer == 1:
                network_probs = F.softmax(network_arch_parameters[layer][next_scale][0], -1) * (1 / 3)
                network_log_probs = torch.log(network_probs + eps)
                network_entropy = - torch.sum(
                    torch.mul(network_probs, network_log_probs))  # / torch.log(torch.tensor(1, dtype=torch.float))
            elif layer == 2:
                network_probs = F.softmax(network_arch_parameters[layer][next_scale][:2], -1) * (2 / 3)
                network_log_probs = torch.log(network_probs + eps)
                network_entropy = 1 - torch.sum(
                    torch.mul(network_probs, network_log_probs))  # / torch.log(torch.tensor(2, dtype=torch.float))
            else:
                network_probs = F.softmax(network_arch_parameters[layer][next_scale], -1)
                network_log_probs = torch.log(network_probs + eps)
                network_entropy = - torch.sum(
                    torch.mul(network_probs, network_log_probs))  # / torch.log(torch.tensor(3, dtype=torch.float))
        elif next_scale == 3:
            if layer == 2:
                network_probs = F.softmax(network_arch_parameters[layer][next_scale][0], -1) * (1 / 3)
                network_log_probs = torch.log(network_probs + eps)
                network_entropy = - torch.sum(
                    torch.mul(network_probs, network_log_probs))  # / torch.log(torch.tensor(1, dtype=torch.float))
            else:
                network_probs = F.softmax(network_arch_parameters[layer][next_scale][:2], -1) * (2 / 3)
                network_log_probs = torch.log(network_probs + eps)
                network_entropy = - torch.sum(
                    torch.mul(network_probs, network_log_probs))  # / torch.log(torch.tensor(2, dtype=torch.float))
        else:
            raise ValueError('invalid scale value {:}'.format(next_scale))

        network_arch_entropy += network_entropy
        cell_arch_entropy += cell_entropy

    return network_arch_entropy, cell_arch_entropy



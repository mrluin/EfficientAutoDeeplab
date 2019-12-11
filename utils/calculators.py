import os
import numpy as np
from tqdm import tqdm
import torch


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
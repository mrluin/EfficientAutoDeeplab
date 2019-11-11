import os
import numpy as np
from tqdm import tqdm



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


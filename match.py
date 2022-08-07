import torch
import numpy as np

import time
import os

from utils.config import cfg
from utils.utils import get_logger
from data.data_loader import GMDataset, get_dataloader
from models.model import Net


def matching_on_batch(similarity_matrices: list, threshold=0.8):
    pred_mean_scores = []
    for similarity_matrix in similarity_matrices:
        pred_matching_mat_1 = torch.zeros_like(similarity_matrix)
        pred_matching_mat_2 = torch.zeros_like(similarity_matrix)

        max_indices_for_rows = torch.argmax(similarity_matrix, dim=1)
        max_indices_for_cols = torch.argmax(similarity_matrix, dim=0)
        for i, j in zip(list(range(similarity_matrix.shape[0])), max_indices_for_rows):
            if similarity_matrix[i, j] >= threshold:
                pred_matching_mat_1[i, j] = 1

        for i, j in zip(max_indices_for_cols, list(range(similarity_matrix.shape[1]))):
            if similarity_matrix[i, j] >= threshold:
                pred_matching_mat_2[i, j] = 1

        mask = pred_matching_mat_1 * pred_matching_mat_2

        pred_mean_score = torch.sum(mask) / min(mask.shape[0], mask.shape[1])

        pred_mean_scores.append(pred_mean_score.item())

    return pred_mean_scores


def matching(model, test_loader, sample_list, threshold, device, logger):
    count = 0
    pred_scores = []
    model.eval()

    start_time = time.time()

    for i, data in enumerate(test_loader):
        images = [_.to(device) for _ in data["image_list"]]
        points = [_.to(device) for _ in data["points_list"]]
        n_points = [_.to(device) for _ in data["n_points_list"]]
        graphs = [_.to(device) for _ in data["graph_list"]]
        perm_mats = data["gt_perm_mat"].to(device)

        try:
            with torch.no_grad():
                similarity_matrices = model(images, points, graphs, n_points, perm_mats)

            pred_mean_scores = matching_on_batch(similarity_matrices, threshold)
            for pred_mean_score in pred_mean_scores:
                pred_scores.append(pred_mean_score)
                print_info = "pair {} and {}, matching score = {:4f}"
                print(print_info.format(sample_list[count][0], sample_list[count][1], pred_mean_score))
                count += 1
        except:
            count += len(images[0])
            continue

    epoch_time = time.time() - start_time

    epoch_info = " time for epoch: {:.4f}"
    print('\n')
    logger.info(epoch_info.format(epoch_time))

    return np.asarray(pred_scores)


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result_dir = "./results/"
    net = Net().to(device)
    params_path = result_dir + "xxxx.pt"
    net.load_state_dict(torch.load(params_path))

    scores_dir = result_dir + "xx/"
    if not os.path.exists(scores_dir):
        os.mkdir(scores_dir)

    test_matching_set = GMDataset(cfg.DATASET_NAME, sets='test_matching', obj_resize=cfg.OBJ_RESIZE)
    test_matching_loader = get_dataloader(test_matching_set)
    test_non_matching_set = GMDataset(cfg.DATASET_NAME, sets='test_non_matching', obj_resize=cfg.OBJ_RESIZE)
    test_non_matching_loader = get_dataloader(test_non_matching_set)

    logger = get_logger(result_dir + 'matching.log')

    matching_sample_list = []

    non_matching_sample_list = []

    threshold = 0.5
    matching_scores = matching(net, test_matching_loader, matching_sample_list, threshold, device, logger)
    np.save(scores_dir + "matching.npy", matching_scores)
    non_matching_scores = matching(net, test_non_matching_loader, non_matching_sample_list, threshold, device, logger)
    np.save(scores_dir + "non_matching.npy", non_matching_scores)




import torch
import torch.nn.functional as f


def heatmap_similarity(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Characterises similarity between the heatmaps of the prediction and target
    :param prediction: batch with predictions
    :param target:  batch with targets
    :return: PSNR
    """

    predicted_probabilities = 1 - f.softmax(prediction, dim=3)[:, :, :, 0]
    pos_target = torch.argmax(target, dim=3)
    pos_target[pos_target >= 1] = 1
    mse = f.mse_loss(pos_target, predicted_probabilities, reduction="none")
    mse = torch.mean(torch.mean(mse, dim=-1), dim=-1)
    psnr = 10 * torch.log10(1 / mse)

    return psnr


def wrong_bonds(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Computes wrong bonds fractions (position only and exact)
    :param prediction: batch with predictions
    :param target:  batch with targets
    :return: {pos_wrong: (batch_size), wrong: (batch_size)}
    """

    pos_target = torch.argmax(target, dim=3)
    num_bonds = torch.sum(torch.count_nonzero(pos_target, dim=1), dim=1)
    pos_pred = torch.argmax(prediction, dim=3)

    diff = pos_target - pos_pred
    wrong_bonds_frac = torch.sum(torch.count_nonzero(diff, dim=1), dim=1) / num_bonds

    pos_target[pos_target >= 1] = 1
    pos_pred[pos_pred >= 1] = 1

    pos_diff = pos_target - pos_pred
    pos_wrong_bonds = torch.sum(torch.count_nonzero(pos_diff, dim=1), dim=1)
    pos_wrong_bonds_frac = pos_wrong_bonds / num_bonds

    return {"pos_wrong": pos_wrong_bonds_frac, "wrong": wrong_bonds_frac}


def excess_bonds(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes excess bonds fraction
    :param prediction: batch with predictions
    :param target:  batch with targets
    :return: excess bonds (batch_size)
    """

    pos_pred = torch.argmax(prediction, dim=3)
    num_bonds = torch.sum(torch.count_nonzero(pos_pred, dim=1), dim=1)
    num_bonds[num_bonds == 0] = 1
    pos_target = torch.argmax(target, dim=3)
    pos_target[pos_target >= 1] = 1
    pos_pred[pos_pred >= 1] = 1

    diff = pos_target - pos_pred

    diff[diff > 0] = 0

    excess_bonds_value = torch.sum(torch.count_nonzero(diff, dim=1), dim=1) / num_bonds

    return excess_bonds_value

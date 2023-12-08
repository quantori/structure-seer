import torch
from statistics import mean
from utils.metrics import heatmap_similarity, wrong_bonds, excess_bonds
from tqdm import tqdm


def evaluate_model(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    evaluation_loader: torch.utils.data.DataLoader,
) -> None:
    """
    Evaluates the model
    :param encoder: an Encoder object
    :param decoder: a Decoder object
    :param evaluation_loader: a PyTorch dataloader with the dataset for evaluation
    :return: prints metrics of the model's performance
    """
    pos_wrong_bonds_dict = {
        "> 1": 0,
        "1 - 0.9": 0,
        "0.9 - 0.8": 0,
        "0.8 - 0.7": 0,
        "0.7 - 0.6": 0,
        "0.6 - 0.5": 0,
        "0.5 - 0.4": 0,
        "0.4 - 0.3": 0,
        "0.3 - 0.2": 0,
        "0.2 - 0.1": 0,
        "< 0.1": 0,
    }

    wrong_bonds_dict = {
        "> 1": 0,
        "1 - 0.9": 0,
        "0.9 - 0.8": 0,
        "0.8 - 0.7": 0,
        "0.7 - 0.6": 0,
        "0.6 - 0.5": 0,
        "0.5 - 0.4": 0,
        "0.4 - 0.3": 0,
        "0.3 - 0.2": 0,
        "0.2 - 0.1": 0,
        "< 0.1": 0,
    }

    psnr_dict = {
        "< 20": 0,
        "20 - 25": 0,
        "25 - 30": 0,
        "30 - 35": 0,
        "35 - 40": 0,
        "40 - 45": 0,
        "45 - 50": 0,
        "> 50": 0,
    }

    pos_total = []
    exact_total = []
    psnr = []
    av_fragment_accuracy = []

    encoder = encoder.eval()
    decoder = decoder.eval()

    with torch.no_grad():
        print("Evaluating...")
        for item in tqdm(evaluation_loader):
            atoms_matrix = [
                item["atoms_matrix"][0],
                item["atoms_matrix"][1],
            ]

            adj_matrix = item["adjacency_matrix"]

            src, embedding = encoder(atoms_matrix)
            prediction = decoder(src=src, embedding=embedding)

            metrics = wrong_bonds(prediction=prediction, target=adj_matrix)

            pos_total += metrics["pos_wrong"].tolist()
            exact_total += metrics["wrong"].tolist()
            psnr += heatmap_similarity(
                prediction=prediction, target=adj_matrix
            ).tolist()
            av_fragment_accuracy += excess_bonds(
                prediction=prediction, target=adj_matrix
            ).tolist()

    for frac in pos_total:
        if frac >= 1:
            pos_wrong_bonds_dict["> 1"] += 1
        elif 1 > frac >= 0.9:
            pos_wrong_bonds_dict["1 - 0.9"] += 1
        elif 0.9 > frac >= 0.8:
            pos_wrong_bonds_dict["0.9 - 0.8"] += 1
        elif 0.8 > frac >= 0.7:
            pos_wrong_bonds_dict["0.8 - 0.7"] += 1
        elif 0.7 > frac >= 0.6:
            pos_wrong_bonds_dict["0.7 - 0.6"] += 1
        elif 0.6 > frac >= 0.5:
            pos_wrong_bonds_dict["0.6 - 0.5"] += 1
        elif 0.5 > frac >= 0.4:
            pos_wrong_bonds_dict["0.5 - 0.4"] += 1
        elif 0.4 > frac >= 0.3:
            pos_wrong_bonds_dict["0.4 - 0.3"] += 1
        elif 0.3 > frac >= 0.2:
            pos_wrong_bonds_dict["0.3 - 0.2"] += 1
        elif 0.2 > frac >= 0.1:
            pos_wrong_bonds_dict["0.2 - 0.1"] += 1
        elif frac < 0.1:
            pos_wrong_bonds_dict["< 0.1"] += 1

    for frac in exact_total:
        if frac >= 1:
            wrong_bonds_dict["> 1"] += 1
        elif 1 > frac >= 0.9:
            wrong_bonds_dict["1 - 0.9"] += 1
        elif 0.9 > frac >= 0.8:
            wrong_bonds_dict["0.9 - 0.8"] += 1
        elif 0.8 > frac >= 0.7:
            wrong_bonds_dict["0.8 - 0.7"] += 1
        elif 0.7 > frac >= 0.6:
            wrong_bonds_dict["0.7 - 0.6"] += 1
        elif 0.6 > frac >= 0.5:
            wrong_bonds_dict["0.6 - 0.5"] += 1
        elif 0.5 > frac >= 0.4:
            wrong_bonds_dict["0.5 - 0.4"] += 1
        elif 0.4 > frac >= 0.3:
            wrong_bonds_dict["0.4 - 0.3"] += 1
        elif 0.3 > frac >= 0.2:
            wrong_bonds_dict["0.3 - 0.2"] += 1
        elif 0.2 > frac >= 0.1:
            wrong_bonds_dict["0.2 - 0.1"] += 1
        elif frac < 0.1:
            wrong_bonds_dict["< 0.1"] += 1

    for frac in psnr:
        if frac >= 50:
            psnr_dict["> 50"] += 1
        elif 50 > frac >= 45:
            psnr_dict["45 - 50"] += 1
        elif 45 > frac >= 40:
            psnr_dict["40 - 45"] += 1
        elif 40 > frac >= 35:
            psnr_dict["35 - 40"] += 1
        elif 35 > frac >= 30:
            psnr_dict["30 - 35"] += 1
        elif 30 > frac >= 25:
            psnr_dict["25 - 30"] += 1
        elif 25 > frac >= 20:
            psnr_dict["20 - 25"] += 1
        elif 20 > frac:
            psnr_dict["< 20"] += 1

    print("Wrong Bonds (positions only):")
    print(f"\tAverage: {round(mean(pos_total) * 100, 2)} %")
    print("\tDistribution:")
    print(f"\t{pos_wrong_bonds_dict}\n")
    print("Wrong Bonds:")
    print(f"\tAverage: {round(mean(exact_total) * 100, 2)} %")
    print("\tDistribution:")
    print(f"\t{wrong_bonds_dict}\n")
    print("Heatmap Similarity (PSNR)")
    print(f"\tAverage: {round(mean(psnr), 3)} dB")
    print("\tDistribution:")
    print(f"\t{psnr_dict}\n")
    print("Excess Bonds")
    print(f"\tAverage: {round(mean(av_fragment_accuracy) * 100, 2)} %")

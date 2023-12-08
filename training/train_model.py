import os
import time
import torch
import torch.nn.functional as F
from statistics import mean
from utils.metrics import wrong_bonds
from tqdm import tqdm


def matrix_criteria(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes cross-entropy loss for each bond in adjacency matrices batch
    (batch_size, DIMENSION, DIMENSION, NUM_BOND_TYPEs)
    :param output: batch of predictions
    :param target: batch of targets
    :return: loss
    """
    # Reshapes the batch of adjacency matrices to a batch of single bonds of size
    # (batch_size * DIMENSION * DIMENSION, NUM_BOND_TYPES)
    t1 = torch.reshape(
        output,
        (
            output.size(dim=0) * output.size(dim=1) * output.size(dim=2),
            output.size(dim=3),
        ),
    )
    t2 = torch.reshape(
        target,
        (
            target.size(dim=0) * target.size(dim=1) * target.size(dim=2),
            target.size(dim=3),
        ),
    )

    return F.cross_entropy(t1, t2)


def process_batch(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    input_batch: dict,
    device: torch.device = "cpu",
) -> tuple:
    sorted_atoms_matrix = [
        input_batch["atoms_matrix"][0].to(device),
        input_batch["atoms_matrix"][1].to(device),
    ]

    sorted_adjacency_matrix = input_batch["adjacency_matrix"].to(device)

    src, embedding = encoder(atoms_matrix=sorted_atoms_matrix)
    output = decoder(src=src, embedding=embedding)
    loss = matrix_criteria(output=output, target=sorted_adjacency_matrix)

    return loss, output, sorted_adjacency_matrix


def train_epoch(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: torch.utils.data.DataLoader,
    device: torch.device = "cpu",
) -> float:
    average_training_loss = 0
    encoder.train()
    decoder.train()
    print("Training")
    for batch in tqdm(loader):
        loss, _, _ = process_batch(encoder, decoder, batch, device)

        average_training_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return average_training_loss / len(loader)


def validate_epoch(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device = "cpu",
) -> tuple:
    average_validation_loss = 0
    average_wrong_bonds = []
    encoder.eval()
    decoder.eval()
    print("Validation")
    for batch in tqdm(loader):
        loss, output, target = process_batch(encoder, decoder, batch, device)
        average_validation_loss += loss.item()
        metric = wrong_bonds(output, target)["wrong"].tolist()
        average_wrong_bonds.extend(metric)

    return average_validation_loss / len(loader), mean(average_wrong_bonds)


def train(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    epochs: int,
    training_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    learning_rate: float = 6e-4,
    device: torch.device = "cpu",
    save: bool = True,
    save_path: str = "./training_log",
) -> None:
    """
    The main training function
    :param encoder: Encoder module
    :param decoder: Decoder module
    :param epochs: Number of epochs to train for
    :param training_loader: Loader object for training dataset
    :param validation_loader: Loader object for validation dataset
    :param learning_rate: Starting learning rate
    :param device: Device
    :param save: If True - encoder and decoder weights
                along with the states of optimizer and scheduler are saved after each epoch
    :param save_path: the path to the directory, where logs, weights and states should be saved
    :return: None
    """
    encoder.to(device)
    decoder.to(device)

    os.makedirs(save_path, exist_ok=True)

    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=32, eta_min=8e-5
    )

    total_time = 0
    for epoch_index in range(epochs):
        print(f"EPOCH {epoch_index + 1} out of {epochs}")
        log = open(f"{save_path}/train_log.txt", "a+")
        epoch_start = time.time()

        train_loss = train_epoch(encoder, decoder, optimizer, training_loader, device)

        if epoch_index < 32:
            scheduler.step()

        current_lr = scheduler.get_last_lr()

        val_loss, metric = validate_epoch(encoder, decoder, validation_loader, device)

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        total_time += epoch_time
        average_time_per_epoch = round(total_time / (epoch_index + 1))

        if save:
            torch.save(
                encoder.state_dict(),
                f"{save_path}/Encoder_epoch_{epoch_index + 1}.weights",
            )
            torch.save(
                decoder.state_dict(),
                f"{save_path}/Decoder_epoch_{epoch_index + 1}.weights",
            )
            torch.save(
                optimizer.state_dict(),
                f"{save_path}/Optimizer_epoch_{epoch_index + 1}.state",
            )
            torch.save(
                scheduler.state_dict(),
                f"{save_path}/Scheduler_epoch_{epoch_index + 1}.state",
            )

        status = (
            f"EPOCH {epoch_index + 1} AVERAGED LOSS - TRAINING = {round(train_loss, 5)}"
            f" AVERAGED LOSS - VALIDATION = {round(val_loss, 5)}"
            f" AVERAGED WRONG BONDS = {round(metric * 100, 2)}%"
            f" AVERAGED TIME = {round(average_time_per_epoch)} sec"
            f" LR: {current_lr[0]} \n"
        )

        log.write(status)
        print(status)
        log.close()

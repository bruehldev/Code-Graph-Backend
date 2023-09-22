from random import choices

import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models_neu.model_definitions import DynamicUmap


class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        embedding = torch.tensor(self.dataframe.iloc[idx]["embedding"], dtype=torch.float32)
        label = torch.tensor(self.dataframe.iloc[idx]["label"], dtype=torch.long)

        sample = {"embedding": embedding, "label": label}

        return sample


def form_triplets(outputs, labels, num_triplets=10):
    anchors, positives, negatives = [], [], []
    unique_labels = torch.unique(labels)

    for label in unique_labels:
        label_mask = labels == label
        other_label_mask = labels != label
        label_indices = torch.nonzero(label_mask).squeeze(1)
        other_label_indices = torch.nonzero(other_label_mask).squeeze(1)

        anchor_choices = choices(label_indices.tolist(), k=num_triplets)
        positive_choices = choices(label_indices.tolist(), k=num_triplets)
        negative_choices = choices(other_label_indices.tolist(), k=num_triplets)

        anchors.extend(outputs[torch.tensor(anchor_choices)])
        positives.extend(outputs[torch.tensor(positive_choices)])
        negatives.extend(outputs[torch.tensor(negative_choices)])

    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)

    return anchors, positives, negatives


def train_clusters(data, model: DynamicUmap, focus_ids):
    # create data loader
    dataset = CustomDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # right now only for dynamic umap
    # TODO add get neural net:
    neural_net = model._model.model.encoder

    optimizer = optim.Adam(params=neural_net.parameters(), lr=0.0003)
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    for batch in tqdm(dataloader):
        outputs = neural_net(batch["embedding"])
        anchors, positives, negatives = form_triplets(outputs, batch["label"])
        loss = triplet_loss(anchors, positives, negatives)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model._model.model.encoder = neural_net
    return model


class CustomDatasetPoint(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        embedding = torch.tensor(self.dataframe.iloc[idx]["embedding"], dtype=torch.float32)
        label = torch.tensor(self.dataframe.iloc[idx]["label"], dtype=torch.long)
        id = torch.tensor(self.dataframe.iloc[idx]["id"], dtype=torch.long)

        sample = {"id": id, "embedding": embedding, "label": label}

        return sample


def custom_loss(output, target_dicts, original, lam=0.1):
    # Initialize MSE loss function
    mse_loss = nn.MSELoss()

    # Extract the IDs and positions from the target list of dictionaries
    target_ids = [d["batch_id"] for d in target_dicts]
    target_pos = [d["pos"] for d in target_dicts]

    # Create tensors from the extracted values
    target_ids_tensor = torch.tensor(target_ids, dtype=torch.long)
    target_pos_tensor = torch.tensor(target_pos, dtype=output.dtype)

    # Gather the corresponding output values using the IDs
    selected_output = torch.index_select(output, 0, target_ids_tensor)

    # Calculate MSE loss for selected indices
    mse_loss_selected = mse_loss(selected_output, target_pos_tensor)

    # Create a mask to find indices that are NOT in the target list
    full_indices = torch.arange(output.size(0))
    mask_non_target = torch.ones(output.size(0), dtype=torch.bool)
    mask_non_target[target_ids_tensor] = 0

    non_target_indices = full_indices[mask_non_target]

    # Gather the corresponding original and output values using the non-target IDs
    selected_original = torch.index_select(original, 0, non_target_indices)
    selected_output_non_target = torch.index_select(output, 0, non_target_indices)

    # Calculate MSE loss for non-target indices with regularization
    mse_loss_non_target = mse_loss(selected_output_non_target, selected_original)

    # Combine both losses with lambda
    total_loss = mse_loss_selected + lam * mse_loss_non_target

    return total_loss


def collate_fn(batch, corrections):
    relevant_corrections = []

    for idx, item in enumerate(batch):
        for correction in corrections:
            if correction["id"] == item.get("id"):
                correction["batch_idx"] = idx
                relevant_corrections.append(correction)
    return {"id": batch["id"], "embedding": batch["embedding"], "label": batch["label"], "corrections": relevant_corrections}


def train_points(data, model, correction):
    dataset = CustomDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda batch: collate_fn(batch, correction))
    # TODO add get neural net:
    neural_net = model._model.model.encoder
    optimizer = optim.Adam(params=neural_net.parameters(), lr=0.00005)
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    for batch in tqdm(dataloader):
        outputs = neural_net(batch["embedding"])
        loss = custom_loss(outputs, batch["corrections"], batch["embedding"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model._model.model.encoder = neural_net
    return model

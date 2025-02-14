import numpy as np
import torch
import os
import json
import torch.nn.functional as F
from pytorch_metric_learning import miners, losses
from sklearn.metrics.pairwise import cosine_similarity
from .evaluator import measurement
from tqdm import tqdm


def knn_monitor(net, device, memory_data_loader, test_data_loader, num_classes, k=200, t=0.1):
    """
    Perform k-Nearest Neighbors (kNN) monitoring.

    Parameters:
    net (nn.Module): The neural network model.
    device (torch.device): The device to run the computations on.
    memory_data_loader (DataLoader): DataLoader for the memory bank.
    test_data_loader (DataLoader): DataLoader for the test data.
    num_classes (int): Number of classes.
    k (int): Number of nearest neighbors to use.
    t (float): Temperature parameter for scaling.

    Returns:
    tuple: True labels and predicted labels.
    """
    net.eval()
    total_num = 0
    feature_bank, feature_labels = [], []
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        # Generate feature bank
        for data, target in memory_data_loader:
            feature = net(data.to(device))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            feature_labels.append(target)

        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous().to(device)
        feature_labels = torch.cat(feature_labels, dim=0).t().contiguous().to(device)

        # Loop through test data to predict the label by weighted kNN search
        for data, target in test_data_loader:
            data, target = data.to(device), target.to(device)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            pred_labels = knn_predict(feature, feature_bank, feature_labels, num_classes, k, t)
            total_num += data.size(0)
            y_pred.append(pred_labels[:, 0].cpu().numpy())
            y_true.append(target.cpu().numpy())
    
    y_true = np.concatenate(y_true).flatten()
    y_pred = np.concatenate(y_pred).flatten()
    
    return y_true, y_pred

def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    """
    Predict labels using k-Nearest Neighbors (kNN) with cosine similarity.

    Parameters:
    feature (Tensor): Feature tensor.
    feature_bank (Tensor): Feature bank tensor.
    feature_labels (Tensor): Labels corresponding to the feature bank.
    classes (int): Number of classes.
    knn_k (int): Number of nearest neighbors to use.
    knn_t (float): Temperature parameter for scaling.

    Returns:
    Tensor: Predicted labels.
    """
    sim_matrix = torch.mm(feature, feature_bank)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    
    return pred_labels

def fast_count_burst(arr):
    """
    Count bursts of continuous values in an array.

    Parameters:
    arr (ndarray): Input array.

    Returns:
    ndarray: Length of bursts.
    """
    diff = np.diff(arr)
    change_indices = np.nonzero(diff)[0]
    segment_starts = np.insert(change_indices + 1, 0, 0)
    segment_ends = np.append(change_indices, len(arr) - 1)
    segment_lengths = segment_ends - segment_starts + 1
    segment_signs = np.sign(arr[segment_starts])
    adjusted_lengths = segment_lengths * segment_signs
    
    return adjusted_lengths


def adjust_learning_rate(optimizer, echo, learning_rate, epoch, annealing_rate = 0.2): # i added this myself to this code
    lr = learning_rate * (annealing_rate ** (echo / epoch)) # this was initially 0.2
    for para_group in optimizer.param_groups:
        para_group['lr'] = lr

def model_train(
    model,
    optimizer,
    train_iter,
    valid_iter,
    loss_name,
    save_metric,
    eval_metrics,
    train_epochs,
    out_file,
    num_classes,
    num_tabs,
    device,
    original_lr = None,
    adjust_lr = False
):
    if loss_name in ["CrossEntropyLoss", "BCEWithLogitsLoss"]:
        criterion = eval(f"torch.nn.{loss_name}")()
    elif loss_name == "TripletMarginLoss":
        criterion = losses.TripletMarginLoss(margin=0.1)
        miner = miners.TripletMarginMiner(margin=0.1, type_of_triplets="semihard")
    elif loss_name == "SupConLoss":
        criterion = losses.SupConLoss(temperature=0.1)
    elif loss_name == "MultiCrossEntropyLoss":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Loss function {loss_name} is not matched.")
    
    assert save_metric in eval_metrics, f"save_metric {save_metric} should be included in {eval_metrics}"
    metric_best_value = 0

    for epoch in range(train_epochs):
        model.train()
        sum_loss = 0
        sum_count = 0
        print(f'epoch: {epoch}')
        if adjust_lr:
            
            adjust_learning_rate(optimizer, echo= epoch, learning_rate= original_lr, epoch= train_epochs)
        
        
        for index, cur_data in enumerate(tqdm(train_iter, desc= 'going through batches for holmes training')):
            cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
            optimizer.zero_grad()
            outs = model(cur_X)
            
            if loss_name == "TripletMarginLoss":
                hard_pairs = miner(outs, cur_y)
                loss = criterion(outs, cur_y, hard_pairs)
            elif loss_name == "SupConLoss":
                loss = criterion(outs, cur_y)
            elif loss_name == "MultiCrossEntropyLoss":
                loss = 0
                cur_indices = torch.nonzero(cur_y)
                cur_indices = cur_indices[:,1].view(-1, num_tabs)
                for ct in range(num_tabs):
                    loss_ct = criterion(outs[:, ct], cur_indices[:, ct])
                    loss = loss + loss_ct
            else:
                loss = criterion(outs, cur_y)
            # print(f'loss is {loss.item()}')
            loss.backward()
            optimizer.step()
            sum_loss += loss.data.cpu().numpy() * outs.shape[0]
            sum_count += outs.shape[0]

        train_loss = round(sum_loss / sum_count, 3)
        print(f"epoch {epoch}: train_loss = {train_loss}")

        if loss_name in ["TripletMarginLoss", "SupConLoss"]:
            valid_true, valid_pred = knn_monitor(model, device, train_iter, valid_iter, num_classes, 10)
        else:
            with torch.no_grad():
                model.eval()
                sum_loss = 0
                sum_count = 0
                valid_pred = []
                valid_true = []

                for index, cur_data in enumerate(valid_iter):
                    cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
                    outs = model(cur_X)
                    
                    if loss_name == "BCEWithLogitsLoss":
                        cur_pred = torch.sigmoid(outs)
                    elif loss_name == "CrossEntropyLoss":
                        cur_pred = torch.argsort(outs, dim=1, descending=True)[:,0]
                    elif loss_name == "MultiCrossEntropyLoss":
                        cur_indices = torch.argmax(outs, dim=-1).cpu()
                        cur_pred = torch.zeros((cur_indices.shape[0], num_classes))
                        for cur_tab in range(cur_indices.shape[1]):
                            row_indices = torch.arange(cur_pred.shape[0])
                            cur_pred[row_indices,cur_indices[:,cur_tab]] += 1
                    else:
                        raise ValueError(f"Loss function {loss_name} is not matched.")

                    valid_pred.append(cur_pred.cpu().numpy())
                    valid_true.append(cur_y.cpu().numpy())
                
                valid_pred = np.concatenate(valid_pred)
                valid_true = np.concatenate(valid_true)
        
        valid_result = measurement(valid_true, valid_pred, eval_metrics, num_tabs)
        print(f"{epoch}: {valid_result}")
        
        if valid_result[save_metric] > metric_best_value:
            metric_best_value = valid_result[save_metric]
            torch.save(model.state_dict(), out_file)

def model_eval(
        model, 
        test_iter, 
        valid_iter, 
        eval_method, 
        eval_metrics, 
        out_file, 
        num_classes, 
        ckp_path, 
        scenario,
        num_tabs,
        device,
        save_path = None,
        save_predictions = False,  # New parameter to control saving
        filename_predict = None,
        filename_true = None,
    ):
    if eval_method == "common":
        with torch.no_grad():
            model.eval()
            y_pred = []
            y_true = []

            for index, cur_data in enumerate(tqdm(test_iter, desc= f'evaluating model with {eval_method}')):
                cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
                outs = model(cur_X)
                if num_tabs == 1:
                    cur_pred = torch.argsort(outs, dim=1, descending=True)[:,0]
                else:
                    if len(outs.shape) <= 2:
                        cur_pred = torch.sigmoid(outs)
                    else:
                        cur_indices = torch.argmax(outs, dim=-1).cpu()
                        cur_pred = torch.zeros((cur_indices.shape[0], num_classes))
                        for cur_tab in range(cur_indices.shape[1]):
                            row_indices = torch.arange(cur_pred.shape[0])
                            cur_pred[row_indices,cur_indices[:,cur_tab]] += 1

                y_pred.append(cur_pred.cpu().numpy())
                y_true.append(cur_y.cpu().numpy())

            y_pred = np.concatenate(y_pred)
            y_true = np.concatenate(y_true)
    elif eval_method == "kNN":
        y_true, y_pred = knn_monitor(model, device, valid_iter, test_iter, num_classes, 10)
    elif eval_method == "Holmes":
        open_threshold = 1e-2
        spatial_dist_file = os.path.join(ckp_path, "spatial_distribution.npz")
        assert os.path.exists(spatial_dist_file), f"{spatial_dist_file} does not exist, please run spatial_analysis.py first"
        spatial_data = np.load(spatial_dist_file)
        webs_centroid = spatial_data["centroid"]
        webs_radius = spatial_data["radius"]

        with torch.no_grad():
            model.eval()
            y_pred = []
            y_true = []

            for index, cur_data in enumerate(tqdm(test_iter, desc= f'evaluating model with {eval_method}')):
                cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
                embs = model(cur_X).cpu().numpy()
                cur_y = cur_y.cpu().numpy()

                all_sims = 1 - cosine_similarity(embs, webs_centroid)
                all_sims -= webs_radius
                outs = np.argmin(all_sims, axis=1)

                if scenario == "Open-world":
                    outs_d = np.min(all_sims, axis=1)
                    open_indices = np.where(outs_d > open_threshold)[0]
                    outs[open_indices] = num_classes - 1

                y_pred.append(outs)
                y_true.append(cur_y)
            y_pred = np.concatenate(y_pred).flatten()
            y_true = np.concatenate(y_true).flatten()
    else:
        raise ValueError(f"Evaluation method {eval_method} is not matched.")
    
    # Save predictions and true labels if requested
    if save_predictions and save_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save predictions and true labels separately
        if filename_predict is None:
            filename_predict = f"{eval_method}_predictions.npy"
        if filename_true is None:
            filename_true = f"{eval_method}_true_labels.npy"
        pred_path = os.path.join(save_path, filename_predict)
        true_path = os.path.join(save_path, filename_true)
        
        np.save(pred_path, y_pred)
        np.save(true_path, y_true)
        print(f"Saved predictions to {pred_path}")
        print(f"Saved true labels to {true_path}")
    
    result = measurement(y_true, y_pred, eval_metrics, num_tabs)
    print(result)

    
    with open(out_file, "w") as fp:
        json.dump(result, fp, indent=4)

def info_nce_loss(features, batch_size, device):
    """
    Compute the InfoNCE loss.

    Parameters:
    features (Tensor): Feature tensor.
    batch_size (int): Batch size.
    device (torch.device): The device to run the computations on.

    Returns:
    tuple: Logits and labels.
    """
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    logits = logits / 0.5

    return logits, labels

def pretrian_accuracy(output, target):
    """
    Compute the accuracy over the top predictions.

    Parameters:
    output (Tensor): Model output.
    target (Tensor): Target labels.

    Returns:
    float: Computed accuracy.
    """
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        res = correct_k.mul_(100.0 / batch_size)

        return res.cpu().numpy()[0]

def model_pretrian(model, optimizer, train_iter, train_epochs, out_file, batch_size, device):
    """
    Pretrain the model.

    Parameters:
    model (nn.Module): The neural network model.
    optimizer (torch.optim.Optimizer): Optimizer for training.
    train_iter (DataLoader): DataLoader for training data.
    train_epochs (int): Number of training epochs.
    out_file (str): Output file to save the model.
    batch_size (int): Batch size.
    device (torch.device): The device to run the computations on.
    """
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(train_epochs):
        model.train()
        mean_acc = 0
        iter_count = 0

        for index, cur_data in enumerate(train_iter):
            cur_X, cur_y = cur_data[0], cur_data[1]
            cur_X = torch.cat(cur_X, dim=0)
            cur_X = cur_X.view(cur_X.size(0), 1, cur_X.size(1)).float().to(device)

            optimizer.zero_grad()
            features = model(cur_X)
            logits, labels = info_nce_loss(features, batch_size, device)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            iter_count += 1
            mean_acc += pretrian_accuracy(logits, labels)

        mean_acc /= iter_count
        print(f"epoch {epoch}: {mean_acc}")

    torch.save(model.state_dict(), out_file)

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score


def train(model, optimizer, criterion, loader, device, use_meta=False):
    model.train()
    losses, correct, total = [], 0, 0
    for image, meta, label in loader:
        image, meta, label = image.to(device), meta.to(device), label.to(device)
        optimizer.zero_grad()
        output  = model(image, meta) if use_meta else model(image)
        loss    = criterion(output, label)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        correct += (output.argmax(1) == label).sum().item()
        total   += label.size(0)
    return np.mean(losses), correct / total


def test(model, criterion, loader, device, use_meta=False):
    model.eval()
    losses, correct, total = [], 0, 0
    with torch.no_grad():
        for image, meta, label in loader:
            image, meta, label = image.to(device), meta.to(device), label.to(device)
            output  = model(image, meta) if use_meta else model(image)
            losses.append(criterion(output, label).item())
            correct += (output.argmax(1) == label).sum().item()
            total   += label.size(0)
    return np.mean(losses), correct / total


def score(model, loader, device, mel_idx, use_meta=False):
    model.eval()
    TP, FP, FN, TN = 0, 0, 0, 0
    all_probs, all_labels   = [], []
    conf_when_true, conf_when_false = [], []

    with torch.no_grad():
        for image, meta, label in loader:
            image, meta, label = image.to(device), meta.to(device), label.to(device)
            output   = model(image, meta) if use_meta else model(image)
            probs    = F.softmax(output, dim=1)
            preds    = output.argmax(1)

            prob_mel = probs[:, mel_idx]
            true_mel = (label == mel_idx)

            conf_when_true.extend(prob_mel[true_mel].cpu().numpy())
            conf_when_false.extend(prob_mel[~true_mel].cpu().numpy())

            all_probs.extend(prob_mel.cpu().numpy())
            all_labels.extend(true_mel.cpu().numpy())

            pred_mel = (preds == mel_idx)
            TP += ( pred_mel &  true_mel).sum().item()
            FP += ( pred_mel & ~true_mel).sum().item()
            FN += (~pred_mel &  true_mel).sum().item()
            TN += (~pred_mel & ~true_mel).sum().item()

    auc             = roc_auc_score(all_labels, all_probs)
    conf_true_mean  = float(np.mean(conf_when_true))  if conf_when_true  else 0.0
    conf_false_mean = float(np.mean(conf_when_false)) if conf_when_false else 0.0

    return TP, FP, FN, TN, auc, conf_true_mean, conf_false_mean
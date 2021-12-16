import copy
import torch
import uuid
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def validate(model, cont_data, categ_data, target_data, device="cuda", val_batch_size=1, save_metrics=True):
    model = model.eval()
    results = np.zeros((categ_data.shape[0], 1))

    for i in range(categ_data.shape[0] // val_batch_size):
        x_categ = torch.tensor(categ_data[val_batch_size*i:val_batch_size*i+val_batch_size]).to(dtype=torch.int64, device=device)
        x_cont = torch.tensor(cont_data[val_batch_size*i:val_batch_size*i+val_batch_size]).to(dtype=torch.float32, device=device)

        pred = model(x_categ, x_cont)
        results[val_batch_size*i:val_batch_size*i+val_batch_size, 0] = torch.sigmoid(pred).squeeze().cpu().detach().numpy()

    fpr, tpr, _ = metrics.roc_curve(target_data[:results.shape[0]], results[:, 0])
    if save_metrics:
        fig, ax = plt.subplots(1, 1)
        plt.xlim([0,1])
        plt.ylim([0,1])
        ax.plot(fpr, tpr)
        plt.savefig(f'{uuid.uuid4()}.png')

    area = metrics.auc(fpr, tpr)
    model = model.train()
    return area


def train(
    model,
    criterion,
    optimizer,
    scheduler,
    train_cont,
    train_categ,
    train_target,
    val_cont,
    val_categ,
    val_target,
    device="cuda",
    batch_size=64,
    max_epochs=100,
    patience=10,
    save_best_model_dict=True,
    save_metrics=True,
    log_interval=10
):
    running_loss = 0.0
    max_score = 0
    best_model_dict = None
    waiting = 0

    for epoch in range(max_epochs):
        for i in range(train_categ.shape[0] // batch_size):
            optimizer.zero_grad()

            x_categ = torch.tensor(train_categ[batch_size*i:batch_size*i+batch_size]).to(dtype=torch.int64, device=device)
            x_cont = torch.tensor(train_cont[batch_size*i:batch_size*i+batch_size]).to(dtype=torch.float32, device=device)
            y_target = torch.tensor(train_target[batch_size*i:batch_size*i+batch_size]).to(dtype=torch.float32, device=device)

            pred = model(x_categ, x_cont)
            loss = criterion(pred, y_target)
            running_loss += loss

            loss.backward()
            optimizer.step()

            if i % log_interval == log_interval - 1:
                print(f"epoch: {epoch + 1}, it: {i + 1}, loss: {running_loss / log_interval}")
                running_loss = 0.0

        running_loss = 0.0
        current_score = validate(model, val_cont, val_categ, val_target, device=device, save_metrics=save_metrics)
        print("Validation score: ", current_score)
        scheduler.step()

        if current_score > max_score:
            max_score = current_score
            best_model_dict = copy.deepcopy(model.state_dict())
            waiting = 0
        else:
            waiting += 1

            if waiting >= patience:
                break

    if save_best_model_dict:
        torch.save(best_model_dict, f"./models/model_{uuid.uuid4()}")

    return best_model_dict

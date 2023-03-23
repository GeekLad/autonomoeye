import sys
import os
import yaml
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm


import torch
import torch.utils.data as data
import torchvision

from process_waymo_dataset import ProcessWaymoDataset
from autonomoeye.utils.train_utils import get_fast_rcnn, track_metrics, collate_fn, \
    get_custom_backbone_fast_rcnn, calc_precision_recall, bb_intersection_over_union

import sklearn.metrics
from sklearn.metrics import average_precision_score, recall_score, auc

import wandb


def evaluate(model, valid_dataloader, iou_vals, nms_thresh):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    gt = []  # list of ground truth boxes per image
    preds = []  # list of predictions  per image
    with torch.no_grad():
        for imgs, annotations in tqdm(valid_dataloader):
            gt.extend([{k: v.to(device) for k, v in t.items()} for t in annotations])

            keeps = []  # nms indices to keep
            imgs = [img.to(device) for img in imgs]
            model_preds = model(imgs)
            # perform nms over predicted bounding boxes
            for entry in model_preds:
                keeps.append(torchvision.ops.nms(
                    entry['boxes'], entry['scores'], nms_thresh).to('cpu').numpy())

            # filtering out predictions not included after nms
            tmp_pred = []
            for img_idx, model_pred in enumerate(model_preds):
                tmp_img = {}
                for entry in model_pred:
                    tmp_img[entry] = model_pred[entry][keeps[img_idx]].to(
                        'cpu').numpy()
                tmp_pred.append(tmp_img)
            preds.extend(tmp_pred)

    # create a pandas dataframe with values needed to calculate metrics
    final_vals = []
    for img_idx in range(len(gt)):
        for pred_idx in range(len(preds[img_idx]['boxes'])):
            ious = []
            pred_box = preds[img_idx]['boxes'][pred_idx]
            for gtb in gt[img_idx]['boxes']:
                ious.append(
                    np.float(bb_intersection_over_union(pred_box, gtb)))
            gt_box = gt[img_idx]['boxes'][np.argmax(ious)]
            gt_label = gt[img_idx]['labels'][np.argmax(ious)]
            label = preds[img_idx]['labels'][pred_idx]
            score = preds[img_idx]['scores'][pred_idx]
            final_vals.append([img_idx, label, gt_label.detach().to("cpu").numpy(), score, pred_box[0], pred_box[1], pred_box[2],
                              pred_box[3], gt_box[0].detach().to("cpu").numpy(), gt_box[1].detach().to("cpu").numpy(),
                              gt_box[2].detach().to("cpu").numpy(), gt_box[3].detach().to("cpu").numpy(), np.max(ious)])

    eval_df = pd.DataFrame(final_vals, columns=['image_id', 'pred_label', 'gt_label', 'confidence_score',
                           'pred_x1', 'pred_y1', 'pred_x2', 'pred_y2', 'gt_x1', 'gt_y1', 'gt_x2', 'gt_y2', 'iou'])

    vehicle_aps = []
    pedestrian_aps = []
    cyclist_aps = []
    for iou in iou_vals:
        precision_vehicles, recall_vehicles = calc_precision_recall(
            eval_df, 1, iou)
        precision_pedestrians, recall_pedestrians = calc_precision_recall(
            eval_df, 2, iou)
        precision_cyclists, recall_cyclists = calc_precision_recall(
            eval_df, 3, iou)

        if precision_vehicles is not None:
            vehicle_aps.append(auc(recall_vehicles, precision_vehicles))
        if precision_pedestrians is not None:
            pedestrian_aps.append(
                auc(recall_pedestrians, precision_pedestrians))
        if precision_cyclists is not None:
            cyclist_aps.append(auc(recall_cyclists, precision_cyclists))

    vehicles_map = np.mean(vehicle_aps)
    pedestrians_map = np.mean(pedestrian_aps)
    cyclist_map = np.mean(cyclist_aps)
    total_map = np.mean(vehicle_aps + pedestrian_aps + cyclist_aps)

    wandb.log({'vehicles_map': vehicles_map,
               'pedestrians_map': pedestrians_map,
               'cyclist_map': cyclist_map,
               'total_map': total_map
               })


def train(model, optimizer, lr_scheduler, train_dataloader, valid_dataloader, wandb_config):

    base_path = '/home/jasierra/autonomoeye/model'
    iou_vals = [0.2, 0.4, 0.6, 0.8]
    nms_threshold = 0.1

    print('Starting to train model...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(wandb_config.num_epochs):
        model.train()
        print('Starting Epoch_{}'.format(epoch))
        model.train()
        total_losses = []
        classifier_losses = []
        box_reg_losses = []
        objectness_losses = []
        rpn_losses = []
        for imgs, annotations in tqdm(train_dataloader):
            # Push data to device
            train_imgs = [img.to(device) for img in imgs]
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

            # Perform forward pass
            loss_dict = model(train_imgs, annotations)
            print(loss_dict)
            losses = sum(loss for loss in loss_dict.values())

            # Back propagate errors
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # Track loss
            total_losses.append(
                sum(loss_dict.values()).detach().to('cpu').numpy())
            classifier_losses.append(
                loss_dict['loss_classifier'].detach().to('cpu').numpy())
            box_reg_losses.append(
                loss_dict['loss_box_reg'].detach().to('cpu').numpy())
            objectness_losses.append(
                loss_dict['loss_objectness'].detach().to('cpu').numpy())
            rpn_losses.append(
                loss_dict['loss_rpn_box_reg'].detach().to('cpu').numpy())

        lr_scheduler.step()

        # Average mertrics over epoch and track
        loss = np.mean(total_losses)
        classifier_loss = np.mean(classifier_losses)
        box_reg_loss = np.mean(box_reg_losses)
        objectness_loss = np.mean(objectness_losses)
        rpn_loss = np.mean(rpn_losses)
        track_metrics(loss, classifier_loss, box_reg_loss,
                      objectness_loss, rpn_loss, epoch)

        print('Saving model weights...')
        torch.save(model.state_dict(), base_path +
                   "/model_weights_{}.pth".format(epoch))
        wandb.save(base_path + "/model_weights_{}.pth".format(epoch))

        # Evaluation on validation data
        print('Evaluating model on validation set...')
        evaluate(model, valid_dataloader, iou_vals, nms_threshold)

# Setup weights and biases and gcp connection
hyperparameter_defaults = dict(
    num_epochs=25,
    num_classes=4,
    learning_rate=0.01,
    momentum=0.9,
    weight_decay=0.0005,
    batch_size=8,
    area_limit=100
)

wandb.init(config=hyperparameter_defaults,
            project='autonomoeye',
            entity='jasierra',
            name='train_val')

wandb_config = wandb.config

# Training and Validation datasets

CATEGORY_NAMES = ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST']
CATEGORY_IDS = [1, 2, 3]
RESIZE = [1152, 768]
AREA_LIMIT = 100

print("Processing training data")
train_dataset = ProcessWaymoDataset('/home/jasierra/autonomoeye/data/train', CATEGORY_NAMES, CATEGORY_IDS, RESIZE, AREA_LIMIT)
print("Processing validation data")
val_dataset = ProcessWaymoDataset('/home/jasierra/autonomoeye/data/validation', CATEGORY_NAMES, CATEGORY_IDS, RESIZE, AREA_LIMIT)

train_dataloader = data.DataLoader(
    train_dataset, batch_size=6, collate_fn=collate_fn)
valid_dataloader = data.DataLoader(
    val_dataset, batch_size=6, collate_fn=collate_fn)


# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_custom_backbone_fast_rcnn(4)

model = model.to(device)

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=wandb_config.learning_rate,
                            momentum=wandb_config.momentum, weight_decay=wandb_config.weight_decay)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.95)

train(model, optimizer, lr_scheduler,
      train_dataloader, valid_dataloader, wandb_config)
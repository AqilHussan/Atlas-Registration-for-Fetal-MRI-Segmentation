"""
@brief  Marginalization functions.
        Plug this between the output of your neural network and your loss function.
        This should replace either the softmax or log_softmax function.
        Please see src/loss/marginalized_dice_loss.py for an example with the mean Dice loss.
@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
@date   May 2021
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
EPS = 1e-10


def marginalize(flat_proba, flat_partial_seg, labels_superset_map):
    """
    Compute the marginalization operation.
    :param flat_proba: tensor; Class probability map. It is assumed that the spatial dimensions have been flattened.
        Expected shape: (nb batches, nb classes, nb voxels)
    :param flat_partial_seg: tensor; Partial segmentation. It is assumed that the spatial dimensions have been flattened.
        Expected shape: (nb batches, nb voxels)
    :param labels_superset_map: dict; mapping from set labels to list of labels.
    :return: marginalized probability map and marginalized one-hot segmentation map.
    """
    assert flat_proba.dim() == 1 + flat_partial_seg.dim(), "Only compatible with label map ground-truth segmentation"
    # Get the nb of classes that are not supersets.
    # This should correspond to the number of output channels.
    num_out_classes = flat_proba.size(1)  # nb of classes that are not supersets
    num_total_classes = max(labels_superset_map.keys()) + 1  # total number of classes (with supersets)

    # Reorient the proba
    flat_proba = flat_proba.permute(0, 2, 1)  # b,s,c

    # Initialize the marginalized one-hot probability map associated with the partial segmentation
    # Warning: here we assume that the superset label numbers are higher
    # than the (singleton) label number
    flat_partial_seg = flat_partial_seg.long()  # make sure that the target is a long before using one_hot
    marg_onehot_seg = F.one_hot(flat_partial_seg, num_classes=num_total_classes).float()  # b,s,c+
    # Remove the supersets
    marg_onehot_seg = marg_onehot_seg[:, :, :num_out_classes]  # b,s,c

    # Marginalize the predicted and target probability maps
    for super_class in list(labels_superset_map.keys()):
        with torch.no_grad():  # only constants in this part so we don't need to compute gradients
            # Compute the mask to use for the marginalization wrt super class super_class
            super_class_size = len(labels_superset_map[super_class])
            super_class_mask = (flat_partial_seg == super_class)  # b,s
            if torch.sum(super_class_mask) == 0:  # super_class is not present in flat_target; skip.
                continue
            w = torch.zeros(num_out_classes, device=flat_proba.device)  # c,
            w_bool = torch.zeros(num_out_classes, device=flat_proba.device)  # c,
            mask = torch.zeros_like(flat_proba, device=flat_proba.device, requires_grad=False)  # b,s,c
            for c in labels_superset_map[super_class]:
                w[c] = 1. / super_class_size
                w_bool[c] = 1
            mask[super_class_mask, :] = w_bool[None, :]

            # Marginalize the one hot probabilities of the partial segmentation for the super class super_class
            marg_onehot_seg[super_class_mask, :] = w[None, :]

        # Marginalize the predicted proba for the super class super_class
        marginal_map_full = torch.sum(w[None, None, :] * flat_proba, dim=2)  # b,s
        # This uses a lot of memory...
        flat_proba = (1 - mask) * flat_proba + mask * marginal_map_full[:, :, None]

    # Transpose to match PyTorch convention
    marg_proba = flat_proba.permute(0, 2, 1)  # b,c,s
    marg_onehot_seg = marg_onehot_seg.permute(0, 2, 1)  # b,c,s

    return marg_proba, marg_onehot_seg


def softmax_marginalize(flat_input, flat_target, labels_superset_map):
    assert flat_input.dim() == 1 + flat_target.dim(), "Only compatible with label map ground-truth segmentation"
    pred_proba = F.softmax(flat_input, dim=1)  # b,c,s
    pred_proba, target_proba = marginalize(
        flat_proba=pred_proba,
        flat_partial_seg=flat_target,
        labels_superset_map=labels_superset_map,
    )
    return pred_proba, target_proba


def log_softmax_marginalize(flat_input, flat_target, labels_superset_map):
    """
    Compute
    log(1/Omega x \sum_{gt classes set} exp(out_c)) - log(1/Omega x \sum_{all classes} exp(out_c))

    This is useful with loss functions like the cross entropy loss that use the log proba rather than the proba.

    :param flat_input:
    :param flat_target:
    :param labels_superset_map:
    :return:
    """
    assert flat_input.dim() == 1 + flat_target.dim(), "Only compatible with label map ground-truth segmentation"

    # Get the nb of classes that are not supersets.
    # This should correspond to the number of output channels.
    num_out_classes = flat_input.size(1)  # nb of classes that are not supersets
    num_total_classes = max(labels_superset_map.keys()) + 1  # total number fo classes (with supersets)

    # Normalize flat_input for stability
    max_flat_input, _ = torch.max(flat_input, dim=1, keepdim=True)
    flat_input = flat_input - max_flat_input  # substract the max per-voxel activation

    # Initialize the batch log probability map prediction
    pred_logproba = F.log_softmax(flat_input, dim=1).permute(0, 2, 1)  # b,s,c
    # We need the exponential of the output map
    pred_explogit = torch.exp(flat_input).permute(0, 2, 1)  # b,s,c
    pred_explogit = pred_explogit + EPS  # add epsilon to avoid to get -inf out of torch.log later

    # Initialize the target probability map
    # Warning: here we assume that the superset label numbers are higher
    # than the (singleton) label number
    flat_target = flat_target.long()  # make sure that the target is a long before using one_hot
    target_proba = F.one_hot(flat_target, num_classes=num_total_classes).float()  # b,s,c+
    # Remove the supersets
    target_proba = target_proba[:, :, :num_out_classes]  # b,s,c

    # Marginalize the predicted and target probability / log-probability maps
    for super_class in list(labels_superset_map.keys()):
        with torch.no_grad():  # only constants in this part so we don't need to compute gradients
            super_class_size = len(labels_superset_map[super_class])
            super_class_mask = (flat_target == super_class)  # b,s
            if torch.sum(super_class_mask) == 0:  # super_class is not present in flat_target; skip.
                continue
            w = torch.zeros(num_out_classes, device=flat_input.device)  # c,
            w_bool = torch.zeros(num_out_classes, device=flat_input.device)  # c,
            mask = torch.zeros_like(pred_logproba, device=flat_input.device)  # b,s,c
            for c in labels_superset_map[super_class]:
                w[c] = 1. / super_class_size
                w_bool[c] = 1
            mask[super_class_mask, :] = w_bool[None, :]  # 1 if part of the ground-truth superset else 0
            target_proba[super_class_mask, :] = w[None, :]
        # Most important lines of code; Compute for all voxels:
        # log(1/Omega x \sum_{super_class} exp(out_c)) - log(\sum_{all classes} exp(out_c))
        superclass_logproba = torch.log(torch.sum(w[None, None, :] * pred_explogit, dim=2))  # b,s
        superclass_logproba -= torch.log(torch.sum(pred_explogit, dim=2))
        # This uses a lot of memory...
        # Copy the logproba to all the class proba of the voxels labeled as super_class
        pred_logproba = (1 - mask) * pred_logproba + mask * superclass_logproba[:, :, None]

    # Transpose to match PyTorch convention
    pred_logproba = pred_logproba.permute(0, 2, 1)  # b,c,s
    target_proba = target_proba.permute(0, 2, 1)  # b,c,s

    return pred_logproba, target_proba

"""
@brief  Pytorch implementation of the mean marginalized Dice loss + marginalized Focal loss.

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
@date   May 2021
"""

def check_label_set_map(labels_superset_map):
    if labels_superset_map is None:
        raise TypeError('labels_superset_map=None. Please indicate a labels_superset_map.\nSee README for more details.')

    # Check that the super class labels are higher than for the class labels
    for labelset in list(labels_superset_map):
        set = labels_superset_map[labelset]
        err_msg = 'Superset label %d is lower than at least one class label in %s' % (labelset, str(set))
        np.testing.assert_array_less(set, labelset, err_msg=err_msg)

from torch.nn.modules.loss import _WeightedLoss
EPSILON = 1e-5

class MarginalizedDiceFocalLoss(_WeightedLoss):
    def __init__(self, labels_superset_map, gamma=2., weight=None, squared=True, reduction='mean'):
        """
        :param gamma: (float) value of the exponent gamma in the definition
            of the Focal loss.
        :param weight: (tensor) weights to apply to the
            voxels of each class. If None no weights are applied.
        :param squared: bool. If True, squared the terms in the denominator of the Dice loss.
        :param labels_superset_map: dict to handle superclasses
        :param reduction: str.
        """
        super(MarginalizedDiceFocalLoss, self).__init__(weight=weight, reduction=reduction)
        self.gamma = gamma
        self.squared = squared
        check_label_set_map(labels_superset_map)
        self.labels_superset_map = labels_superset_map

    def _prepare_data(self, input_batch, target):
        # Flatten the predicted class score map and the segmentation ground-truth
        flat_input = torch.reshape(input_batch, (input_batch.size(0), input_batch.size(1), -1))  # b,c,s
        flat_target = torch.reshape(target, (target.size(0), -1))
        # Compute the marginalization
        log_pred_proba, target_proba = log_softmax_marginalize(
            flat_input=flat_input,
            flat_target=flat_target,
            labels_superset_map=self.labels_superset_map,
        )
        return log_pred_proba, target_proba

    def forward(self, input, target):
        logpt, target_proba = self._prepare_data(input, target) # b,c,s

        # Get the proba
        pt = torch.exp(logpt)  # b,c,s

        # Compute the batch of marginalized Focal loss values
        if self.weight is not None:
            if self.weight.type() != logpt.data.type():
                self.weight = self.weight.type_as(logpt.data)
            # Expand the weight to a map of shape b,c,s
            w_class = self.weight[None, :, None] # c => 1,c,1
            w_class = w_class.expand((logpt.size(0), -1, logpt.size(2)))  # 1,c,1 => b,c,s
            logpt = logpt * w_class
        weight = torch.pow(-pt + 1., self.gamma)
        # Sum over classes and mean over voxels
        per_voxel_focal_loss = torch.sum(-weight * target_proba * logpt, dim=1)  # b,s
        focal_loss = torch.mean(per_voxel_focal_loss, dim=1)  # b,

        # Compute the batch marginalized mean Dice loss
        num = pt * target_proba  # b,c,s --p*g
        num = torch.sum(num, dim=2)  # b,c
        if self.squared:
            den1 = torch.sum(pt * pt, dim=2)  # b,c
            den2 = torch.sum(target_proba * target_proba, dim=2)  # b,c
        else:
            den1 = torch.sum(pt, dim=2)  # b,c
            den2 = torch.sum(target_proba, dim=2)  # b,c
        dice = (2. * num) / (den1 + den2 + EPSILON)
        # Get the mean of the dices over all classes
        dice_loss = 1. - torch.mean(dice, dim=1)  # b,

        loss = dice_loss + focal_loss

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        # Default is mean reduction
        else:
            return loss.mean()


EPSILON = 1e-5


class MeanDiceLoss(nn.Module):
    def __init__(self, reduction='mean', squared=True):
        """
        :param reduction: str. Mode to merge the batch of sample-wise loss values.
        :param squared: bool. If True, squared the terms in the denominator of the Dice loss.
        """
        super(MeanDiceLoss, self).__init__()
        self.squared = squared
        self.reduction = reduction

    def _prepare_data(self, input_batch, target):
        num_out_classes = input_batch.size(1)
        # Prepare the batch prediction
        flat_input = torch.reshape(input_batch, (input_batch.size(0), input_batch.size(1), -1))  # b,c,s
        flat_target = torch.reshape(target, (target.size(0), -1))  # b,s
        pred_proba = F.softmax(flat_input, dim=1)  # b,c,s
        target_proba = F.one_hot(flat_target, num_classes=num_out_classes)
        target_proba = target_proba.permute(0, 2, 1).float()  # b,c,s
        return pred_proba, target_proba


    def forward(self, input_batch, target):
        """
        compute the mean dice loss between input_batch and target.
        :param input_batch: tensor array. This is the output of the deep neural network (before softmax).
        Dimensions should be in the order:
        1D: num batch, num classes, num voxels
        2D: num batch, num classes, dim x, dim y
        3D: num batch, num classes, dim x, dim y, dim z
        :param target: tensor array. This is the ground-truth segmentation.
        Dimensions should be in the order:
        1D: num batch, num voxels or num batch, 1, num voxels
        2D: num batch, dim x, dim y or num batch, 1, dim x, dim y
        3D: num batch, dim x, dim y, dim z or num batch, 1, dim x, dim y, dim z
        :return: tensor array.
        """
        pred_proba, target_proba = self._prepare_data(input_batch, target)

        # Compute the dice for all classes
        num = pred_proba * target_proba  # b,c,s --p*g
        num = torch.sum(num, dim=2)  # b,c
        if self.squared:
            den1 = torch.sum(pred_proba * pred_proba, dim=2)  # b,c
            den2 = torch.sum(target_proba * target_proba, dim=2)  # b,c
        else:
            den1 = torch.sum(pred_proba, dim=2)  # b,c
            den2 = torch.sum(target_proba, dim=2)  # b,c

        # I choose not to add an epsilon on the numerator.
        # This is because it can lead to unstabilities in the gradient.
        # As a result, for empty prediction and empty target the Dice loss value
        # is 1 and not 0. But it is fine because for the optimization of a
        # deep neural network it is the gradient not the loss that is relevant.
        dice = (2. * num) / (den1 + den2 + EPSILON)

        # Get the mean of the dices over all classes
        loss = 1. - torch.mean(dice, dim=1)  # b,

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        # Default is mean reduction
        else:
            return loss.mean()
                     
class OneHotDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(OneHotDiceLoss, self).__init__()

    def forward(self, predictions, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        num_classes = 2
        smooth= 1e-5
        
        predictions = F.sigmoid(predictions)     
        targets_onehot = F.one_hot(targets,num_classes)
        targets_onehot = targets_onehot.view(-1,num_classes,256,256)
        
        num = 2*(predictions * targets_onehot).sum(dim=(0,2,3))
        den = (predictions.sum(dim=(0,2,3)) + targets_onehot.sum(dim=(0,2,3)) + smooth)
#         weights = targets_onehot.sum(dim=(0,2,3))
#         weights = 1/(weights*weights)
        loss = num/den
        return torch.mean(loss)
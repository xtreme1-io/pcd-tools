import numpy as np
from easydict import EasyDict as edict
from .iou_3d import iou_bev, iou_3d
from scipy.optimize import linear_sum_assignment


types = ['bev', '3d']

def build_iou_matrix(pred_bboxes, target_bboxes):
    """
    :param pred_bboxes: a ndarray of ["x", "y", "z", "dx", "dy", "dz", "rotZ"]
    :param target_bboxes: a ndarray of ["x", "y", "z", "dx", "dy", "dz", "rotZ"]
    :return: a ndarray of shape (2, len(pred_bboxes), len(target_bboxes)), for bev and 3d iou
    """
    result = np.zeros((2, len(pred_bboxes), len(target_bboxes)))
    for i, bbox1 in enumerate(pred_bboxes):
        for j, bbox2 in enumerate(target_bboxes):
            result[0, i, j] = iou_bev(bbox1, bbox2)
            result[1, i, j] = iou_3d(bbox1, bbox2)[0]
    return result

def best_matches(preds, targets):
    """
    :param preds: a ndarray of ["x", "y", "z", "dx", "dy", "dz", "rotZ", "confidence"]
    :param targets: a ndarray of ["x", "y", "z", "dx", "dy", "dz", "rotZ"]
    :return: a dict of best matches for each iou threshold
        {
            "bev": {
                "row_ind": [0, 1, 2, ...],
                "col_ind": [0, 1, 2, ...],
                "best_match_iou": [0.5, 0.6, 0.9, ...],
                "confidences": [0.5, 0.6, 0.9, ...],
            },
            "3d": {
                ...
            }       
        }
    """
    iou_matrices = build_iou_matrix(preds[:, :7], targets)
    matches = edict()
    for type_idx, type in enumerate(types):
        iou_matrix = iou_matrices[type_idx]
        # find the best match for each target
        row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True) # Hungarian algorithm
        best_match_iou = iou_matrix[row_ind, col_ind]
        confidences = preds[row_ind, 7]
        matches[type] = edict(
            row_ind = row_ind,
            col_ind = col_ind,
            best_match_iou = best_match_iou,
            confidences = confidences,
        )
    return matches


def build_metrics_for_label(preds_list, targets_list, iou_thresholds):
    """
    :param preds: a list of ndarray with fields ["x", "y", "z", "dx", "dy", "dz", "rotZ", "confidence"]
    :param targets: a list of ndarray with fields ["x", "y", "z", "dx", "dy", "dz", "rotZ"]
    :param iou_thresholds: a ndarray of threshold
    """
    matches = [
        best_matches(preds, targets)
        for preds, targets in zip(preds_list, targets_list)
    ]

    def build_confidence_thresholds(confidences, n_thresholds=11):
        # sort confidences in descending order
        confidences = np.sort(confidences)[::-1]
        thresholds = []
        min_thresholds = np.linspace(1, 0, n_thresholds)[1:]

        # find the largest confidence that is in each max_thresholds
        start_idx = 0
        for max_threshold in min_thresholds:
            for idx in range(start_idx, len(confidences)):
                confidence = confidences[idx]
                if (confidence >= max_threshold 
                    and (idx == len(confidences) - 1 
                         or confidences[idx + 1] < max_threshold)):
                    thresholds.append(confidence)
                    start_idx = idx + 1
                    break
        return np.array(thresholds)


    def calc_ap(precisions, recalls):
        # calculate average precision
        assert len(precisions) == len(recalls)
        for i in range(len(precisions)):
            precisions[i] = precisions[i:].max(axis=-1)
            
        # calculate area under precision-recall curve
        recalls = np.concatenate([[0], recalls])
        ap = ((recalls[1:] - recalls[:-1]) * precisions).sum()
        return ap

    # (bev, 3d), iou_thresholds
    aps = np.zeros((2, len(iou_thresholds))) 
    
    # collect all confidences for bev and 3d
    for type_idx, type in enumerate(types):
        confidences = np.concatenate([
            match[type].confidences
            for match in matches
        ])
        confidence_thresholds = build_confidence_thresholds(confidences)

        # build metrics for each iou threshold
        for iou_idx, iou_threshold in enumerate(iou_thresholds):
            precisions = np.zeros(len(confidence_thresholds))
            recalls = np.zeros(len(confidence_thresholds))

            for conf_idx, conf_threshold in enumerate(confidence_thresholds):
                tp = 0
                fp = 0
                fn = 0
                for i, match in enumerate(matches):
                    iou_mask = match[type].best_match_iou >= iou_threshold
                    confidence_mask = match[type].confidences >= conf_threshold

                    tp_ = (iou_mask & confidence_mask).sum()
                    tp += tp_
                    fp += (preds_list[i][:, 7] >= conf_threshold).sum() - tp_
                    fn += len(targets_list[i]) - tp_
                precisions[conf_idx] = tp / (tp + fp) if tp + fp > 0 else 0
                recalls[conf_idx] = tp / (tp + fn) if tp + fn > 0 else 0
            
            aps[type_idx, iou_idx] = calc_ap(precisions, recalls)

    return edict({
        type: edict({
            # "aps": aps[type_idx].tolist(),
            # "iou_thresholds": iou_thresholds.tolist(),
            "AP": aps[type_idx].mean(),
        })
        for type_idx, type in enumerate(types)
    })
        
def build_labels(preds_list, targets_list):
    all_labels = set()
    for data in preds_list + targets_list:
        for obj in data:
            if "label" in obj:
                all_labels.add(obj["label"])
            else:
                return []
                
    return sorted(all_labels)

def build_metrics(preds_list, targets_list):
    """
    :params preds_list: a list of list of objects with fields: 
        label, confidence, x, y, z, dx, dy, dz, rotX, rotY, rotZ
    :params targets_list: a list of list of objects with fields
    """
    assert len(preds_list) == len(targets_list), f"the count of preds_list({len(preds_list)}) and targets_list({len(targets_list)}) must be equal"
    all_labels = build_labels(preds_list, targets_list)
    has_label = len(all_labels) > 0
    if not has_label:
        all_labels = [""]

    def filter_by_label(label, objects, is_target=False, filter=True):
        objects_for_label = [
            [o[v] for v in ["x", "y", "z", "dx", "dy", "dz", "rotZ"]] + ([o["confidence"]] if not is_target else [])
            for o in objects 
            if not filter or o["label"] == label
        ]
        if len(objects_for_label) == 0:
            return np.empty((0, 7 if is_target else 8))
        else:
            return np.array(objects_for_label)
    
    thresholds = np.linspace(0.5, 0.95, 10)
    metrics = edict({
        label: build_metrics_for_label([
            filter_by_label(label, preds, is_target=False, filter=has_label)
            for preds in preds_list
        ], [
            filter_by_label(label, targets, is_target=True, filter=has_label)
            for targets in targets_list
        ], thresholds)
        for label in all_labels
    })

    for type in types:
        metrics[f"{'mAP' if has_label else 'AP'}-{type}"] = np.mean([metrics[label][type].AP for label in all_labels])

    if not has_label:
        del metrics[""]

    return metrics


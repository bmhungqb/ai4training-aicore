import torch
import numpy as np

# def accuracy(outputs, targets, topk=(1, )):
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = targets.size(0)
#
#         _, pred = outputs.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(targets.view(1, -1).expand_as(pred))
#
#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(1 / batch_size))
#     return res

# def accuracy(outputs, targets, topk=(1, )):
#     topk = (1,)
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = targets.size(0)
#
#         _, pred = outputs.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(targets.view(1, -1).expand_as(pred))
#         k = 1
#         correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#         res = correct_k.mul_(1 / batch_size)
#
#         # res = []
#         # for k in topk:
#         #     correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#         #     res.append(correct_k.mul_(1 / batch_size))
#     return res

def accuracy(pred, targets):
    with torch.no_grad():
        correct = pred.eq(targets)
        num_correct = correct.float().sum(0)
        acc = correct.float().mean(0)# * 100
    return acc, num_correct

def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    frame_wise_labels = frame_wise_labels.cpu().numpy()
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:# find next label
            if frame_wise_labels[i] not in bg_class: # and not the bg
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = torch.zeros((m_row + 1, n_col + 1), dtype=torch.float64)
    # D = np.zeros([m_row + 1, n_col + 1], np.float)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1,
                              D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score

def edit_score(dataset, recognized, ground_truth):
    # recognized, ground_truth, norm = True, bg_class = ["background"]
    if dataset == 'gtea':
        bg_class = [10]
    elif dataset == '50salads' or dataset == 'breakfast':
        bg_class = []
    elif dataset == 'tas_instance':
        bg_class = []  # tas_instance has NO background class; all 4 classes are valid actions
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, True)

def f_score(dataset, recognized, ground_truth, overlap):
    if dataset == 'gtea':
        bg_class = [10]
    elif dataset == '50salads' or dataset == 'breakfast':
        bg_class = []
    elif dataset == 'tas_instance':
        bg_class = []  # tas_instance has NO background class; all 4 classes are valid actions
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start) # remove bg in all to get start and end
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start) # U

        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))]) #???
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def compute_metrics( dataset, outputs, targets):
    with torch.no_grad():
        pred = outputs.argmax(dim=1)
        accs, num_correct = accuracy(pred, targets)

        # if state == 'train':
        #     return num_correct, accs
        # elif state == 'val':
        #     edit = edit_score(dataset, pred, targets) # just bs= 1
        #
        #     overlap = [.1, .25, .5]
        #     tp, fp, fn = torch.zeros(3), torch.zeros(3), torch.zeros(3)
        #     for s in range(len(overlap)):
        #         tp1, fp1, fn1  = f_score( dataset, pred, targets, overlap[s])
        #         tp[s] = tp1 #tp[s] += tp1
        #         fp[s] = fp1 #fp[s] += fp1
        #         fn[s] = fn1 #fn[s] += fn1
        #     f_scores = tp, fp, fn

        edit = edit_score(dataset, pred, targets)  # just bs= 1， compute each sample

        overlap = [.1, .25, .5]
        tp, fp, fn = torch.zeros(3), torch.zeros(3), torch.zeros(3)
        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(dataset, pred, targets, overlap[s])
            tp[s] = tp1  # tp[s] += tp1
            fp[s] = fp1  # fp[s] += fp1
            fn[s] = fn1  # fn[s] += fn1
        f_scores = [tp, fp, fn]
        return num_correct, accs, edit, f_scores

def compute_metrics_from_action( dataset, outputs, targets):
    with torch.no_grad():
        pred = outputs
        accs, num_correct = accuracy(pred, targets)

        # if state == 'train':
        #     return num_correct, accs
        # elif state == 'val':
        #     edit = edit_score(dataset, pred, targets) # just bs= 1
        #
        #     overlap = [.1, .25, .5]
        #     tp, fp, fn = torch.zeros(3), torch.zeros(3), torch.zeros(3)
        #     for s in range(len(overlap)):
        #         tp1, fp1, fn1  = f_score( dataset, pred, targets, overlap[s])
        #         tp[s] = tp1 #tp[s] += tp1
        #         fp[s] = fp1 #fp[s] += fp1
        #         fn[s] = fn1 #fn[s] += fn1
        #     f_scores = tp, fp, fn

        edit = edit_score(dataset, pred, targets)  # just bs= 1， compute each sample

        overlap = [.1, .25, .5]
        tp, fp, fn = torch.zeros(3), torch.zeros(3), torch.zeros(3)
        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(dataset, pred, targets, overlap[s])
            tp[s] = tp1  # tp[s] += tp1
            fp[s] = fp1  # fp[s] += fp1
            fn[s] = fn1  # fn[s] += fn1
        f_scores = [tp, fp, fn]
        return num_correct, accs, edit, f_scores

def compute_dense_acc(outputs, targets):
    with torch.no_grad():
        pred = outputs.argmax(dim=1) # (batchsize, cls_num)
        accs, num_correct = accuracy(pred, targets) # acc is same as the class_predict, which is average over a sample
    return accs, num_correct

def top_accuracy(outputs, targets, topk=1): #topk=(1, )
    with torch.no_grad():
        # maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        ##-------for multi-top
        # res = []
        # for k in topk:
        #     correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        #     res.append(correct_k.mul_(1 / batch_size))
        #     res.append(correct_k)
        # return res

        ##----- for single top
        correct_k = correct[:topk].reshape(-1).float().sum(0, keepdim=True)

    return correct_k


def evaluate_instance_metrics(pred_instances, gt_instances, thresholds=(0.25, 0.5, 0.75)):
    """Direct instance-level evaluation using temporal IoU (tIoU), operating on
    predicted/ground-truth instance tuples instead of dense frame labels.

    This avoids collapsing adjacent same-class instances into a single segment,
    which is the main limitation of get_labels_start_end_time()-based metrics
    (see problem_definition.md, Instance-Level TAS protocol).

    Args:
        pred_instances: list of (start, end, class_id) predicted instances.
        gt_instances: list of (start, end, class_id) ground-truth instances.
        thresholds: tIoU thresholds at which to report precision/recall/F1.

    Returns:
        dict mapping threshold -> {'tp', 'fp', 'fn', 'precision', 'recall', 'f1'}
    """
    def tiou(a, b):
        s1, e1, _ = a
        s2, e2, _ = b
        inter = min(e1, e2) - max(s1, s2)
        if inter <= 0:
            return 0.0
        union = max(e1, e2) - min(s1, s2)
        return inter / union if union > 0 else 0.0

    results = {}
    for tau in thresholds:
        matched_gt = [False] * len(gt_instances)
        # score predictions by best tIoU against an unclaimed, same-class GT instance
        candidates = []
        for pi, p in enumerate(pred_instances):
            for gi, g in enumerate(gt_instances):
                if p[2] != g[2]:
                    continue
                score = tiou(p, g)
                if score >= tau:
                    candidates.append((score, pi, gi))
        candidates.sort(key=lambda x: x[0], reverse=True)

        matched_pred = set()
        tp = 0
        for score, pi, gi in candidates:
            if pi in matched_pred or matched_gt[gi]:
                continue
            matched_pred.add(pi)
            matched_gt[gi] = True
            tp += 1

        fp = len(pred_instances) - tp
        fn = len(gt_instances) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[tau] = {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}

    return results
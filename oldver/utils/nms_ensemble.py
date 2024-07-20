import torch


def box_iou(box1, box2):
    """Compute IoU between two sets of boxes."""
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union = box1_area + box2_area - intersection

    return intersection / (union + 1e-6)


def nms(boxes, scores, iou_threshold):
    """Non-maximum suppression."""
    _, order = scores.sort(0, descending=True)
    keep = []

    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        i = order[0].item()
        keep.append(i)

        iou = box_iou(boxes[i].unsqueeze(0), boxes[order[1:]])
        idx = (iou <= iou_threshold).nonzero().squeeze()
        order = order[idx + 1]

    return torch.tensor(keep, dtype=torch.long)


def nms_ensemble(detections, iou_threshold=0.5, score_threshold=0.05):
    """Perform NMS ensemble on detections from multiple views."""
    all_boxes = []
    all_scores = []
    all_labels = []

    for detection in detections:
        boxes = detection["boxes"]
        scores = detection["scores"]
        labels = detection["labels"]

        mask = scores > score_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    keep = nms(all_boxes, all_scores, iou_threshold)

    final_boxes = all_boxes[keep]
    final_scores = all_scores[keep]
    final_labels = all_labels[keep]

    return {"boxes": final_boxes, "scores": final_scores, "labels": final_labels}

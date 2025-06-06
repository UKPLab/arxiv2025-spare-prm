import torch


def collate_tag_positions(input_ids, step_tag_id, padding_side="right"):
    tag_pos_ = [torch.where(i == step_tag_id)[0] for i in input_ids]
    max_len = max(len(i) for i in tag_pos_)
    
    tag_pos = []
    for pos in tag_pos_:
        if len(pos) < max_len:
            padding = torch.full(size=(max_len-len(pos),), fill_value=0).to(pos.device)
            if padding_side == "right":
                pos = torch.cat((pos, padding))
            else:
                pos = torch.cat((padding, pos))
        tag_pos.append(pos)

    tag_pos = torch.stack(tag_pos)
    return tag_pos


def collate_labels(labels, targets_label_pad_id=-100, padding_side="right"):
    # labels are already contiguous so just scrap off based on min/max position value
    if padding_side == "right":
        cut_off_pos = max(torch.where(labels != targets_label_pad_id)[1])
        labels = labels[:,:cut_off_pos+1]

    else:
        cut_off_pos = min(torch.where(labels != targets_label_pad_id)[1])
        labels = labels[:,cut_off_pos:]

    return labels


def collate_logits(
    logits, 
    input_ids, 
    step_tag_id, 
    step_target_ids=None, 
    padding_side="left",
):
    tag_pos = collate_tag_positions(input_ids, step_tag_id, padding_side)
    logits = logits[torch.arange(tag_pos.size(0)).unsqueeze(1), tag_pos]
    if step_target_ids:
        return logits[:,:,step_target_ids], tag_pos
    return logits, tag_pos

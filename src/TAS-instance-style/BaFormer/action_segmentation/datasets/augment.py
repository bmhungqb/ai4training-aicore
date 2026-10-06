import random



def augment_crop(data, target):
    percent = random.uniform(0.7, 1)
    len_video = data.shape[2]
    len_video_clip = round(data.shape[2] * percent)
    # if len_video- len_video_clip == 0:
    #     start = 0
    # else:
    #     start = random.sample(range(0, len_video- len_video_clip +1), 1)[0]# +1
    # data = data[:,:,start: start + len_video_clip]
    # target = target[:, start: start + len_video_clip]
    # return data, target

    start = random.sample(range(0, len_video- len_video_clip +1), 1)[0]# +1
    data = data[:,:,start: start + len_video_clip]
    target = target[:, start: start + len_video_clip]
    return data, target


def augment_crop_instances(data, target, instances):
    """Instance-aware variant of `augment_crop` (proposal_imprv_baformer04, Pillar 4).

    Crops `data` along the time axis the same way as `augment_crop`, and remaps the
    `(start_frame, end_frame, class_id)` rows in `instances` into the cropped window
    (clipping to the window and dropping instances left with zero length). Also crops
    `target` (frame-wise labels) along the same temporal window so metrics and loss
    remain fully synchronized.
    """
    percent = random.uniform(0.7, 1)
    len_video = data.shape[2]
    len_video_clip = round(data.shape[2] * percent)
    start = random.sample(range(0, len_video - len_video_clip + 1), 1)[0]
    end = start + len_video_clip

    # `instances` arrives as (1, N, 3) via default_collate (batch_size=1); operate on
    # the middle (N, 3) dim regardless of whether a batch dim is present.
    squeeze_back = instances.dim() == 3
    inst = instances.squeeze(0) if squeeze_back else instances

    new_inst = inst.clone()
    new_inst[:, 0] = (new_inst[:, 0] - start).clamp(min=0, max=len_video_clip)
    new_inst[:, 1] = (new_inst[:, 1] - start).clamp(min=0, max=len_video_clip)
    keep = new_inst[:, 1] > new_inst[:, 0]
    new_inst = new_inst[keep]

    # Guard: if crop happens to eliminate all instances (extremely rare), return original
    if len(new_inst) == 0:
        return data, target, instances

    data = data[:, :, start:end]
    if target is not None:
        target = target[:, start:end]

    new_instances = new_inst.unsqueeze(0) if squeeze_back else new_inst
    return data, target, new_instances


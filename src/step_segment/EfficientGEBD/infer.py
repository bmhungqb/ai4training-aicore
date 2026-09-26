"""Run inference with a trained EfficientGEBD checkpoint on either:
  - a single new video file (.mp4/.avi/.mov/.mkv/.webm)
  - a single pre-extracted frame folder (frame1.jpg, frame2.jpg, ...)
  - a folder containing several such frame folders (e.g. the val split's
    images/val/ directory) or several raw video files (batch mode)

and visualize the predicted step-boundary scores/timestamps:
  - <out>/<vid>/score_curve.png   score-vs-time curve with predicted (red) /
                                  GT (green, if --gt given) boundary markers
  - <out>/<vid>/boundaries.json   predicted (+ GT) boundary frame idx / timestamps
  - <out>/<vid>/annotated.mp4     original video played normally; at each predicted
                                  boundary it freezes on that frame (red border +
                                  'STEP BOUNDARY' label, ~5s by default via
                                  --freeze-seconds) before continuing (skip with --no-video)

Examples:
    # a brand new video, no ground truth
    python infer.py --config-file config-files/sewing_resnet50.yaml \\
        --checkpoint output/sewing/resnet50/model_best.pth \\
        --input /path/to/new_video.mp4 --output-dir output/infer/new_video

    # the whole val split, overlaying GT + printing F1
    python infer.py --config-file config-files/sewing_resnet50.yaml \\
        --checkpoint output/sewing/resnet50/model_best.pth \\
        --input data/efficient_gebd_dataset/images/val \\
        --gt data/efficient_gebd_dataset/val_annotation.pkl \\
        --output-dir output/infer/val
"""
import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from datasets.dataset import image_loader
from modeling import cfg, build_model
from utils.eval import eval_f1, get_idx_from_score_by_threshold

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
TEMPLATE = 'frame{:d}.jpg'


# --------------------------------------------------------------------------- #
# Input resolution: video file(s) / frame folder(s) -> list of (vid, frame_dir, fps)
# --------------------------------------------------------------------------- #
def extract_frames_cv2(video_path: Path, out_dir: Path, overwrite: bool = False):
    existing = sorted(out_dir.glob('frame*.jpg'))
    if existing and not overwrite:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        return len(existing), fps

    out_dir.mkdir(parents=True, exist_ok=True)
    for f in existing:
        f.unlink()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        cv2.imwrite(str(out_dir / TEMPLATE.format(idx)), frame)
    cap.release()
    return idx, fps


def resolve_inputs(input_path: Path, cache_dir: Path, overwrite_cache: bool = False):
    entries = []  # (vid, frame_dir, known_fps_or_None)
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTS:
            raise ValueError(f'Unsupported video extension: {input_path}')
        vid = input_path.stem
        vlen, fps = extract_frames_cv2(input_path, cache_dir / vid, overwrite_cache)
        entries.append((vid, cache_dir / vid, fps))
        return entries

    if not input_path.is_dir():
        raise ValueError(f'--input not found: {input_path}')

    if list(input_path.glob('frame*.jpg')):
        entries.append((input_path.name, input_path, None))
        return entries

    for d in sorted(p for p in input_path.iterdir() if p.is_dir()):
        if list(d.glob('frame*.jpg')):
            entries.append((d.name, d, None))
    for f in sorted(p for p in input_path.iterdir() if p.is_file()):
        if f.suffix.lower() in VIDEO_EXTS:
            vid = f.stem
            vlen, fps = extract_frames_cv2(f, cache_dir / vid, overwrite_cache)
            entries.append((vid, cache_dir / vid, fps))

    if not entries:
        raise ValueError(f'No frame folders or video files found under {input_path}')
    return entries


def get_video_info(vid, frame_dir, gt_dict, known_fps, fallback_fps):
    if gt_dict is not None and vid in gt_dict:
        info = gt_dict[vid]
        return info['num_frames'], info['fps'], info['video_duration']
    vlen = len(list(frame_dir.glob('frame*.jpg')))
    fps = known_fps or fallback_fps or 25.0
    duration = vlen / fps if fps else 0.0
    return vlen, fps, duration


# --------------------------------------------------------------------------- #
# Slicing (mirrors datasets/dataset.py::prepare_gebd_annotations, SEWING branch)
# and model forward pass
# --------------------------------------------------------------------------- #
def build_slices(vlen, duration, seq_len):
    num_slices = int(duration // 10 + 1)
    num_valid = max(1, min(int(duration / 0.1), seq_len * num_slices))
    selected_valid = np.linspace(1, vlen, num_valid, dtype=int)
    last_idx = selected_valid[-1]
    total = seq_len * num_slices
    selected_all = np.full(total, last_idx, dtype=int)
    selected_all[:num_valid] = selected_valid
    frame_masks_all = np.zeros(total, dtype=bool)
    frame_masks_all[:num_valid] = True
    return [(selected_all[s * seq_len:(s + 1) * seq_len], frame_masks_all[s * seq_len:(s + 1) * seq_len])
            for s in range(num_slices)]


def load_slice_tensor(frame_dir, indices, transform, size):
    imgs = torch.zeros(len(indices), 3, size, size, dtype=torch.float32)
    for i, idx in enumerate(indices):
        img = image_loader(str(frame_dir / TEMPLATE.format(int(idx))))
        imgs[i] = transform(img)
    return imgs


@torch.no_grad()
def run_inference_for_video(model, device, frame_dir, slices, transform, size, num_heads, batch_size):
    frame_idx_out = []
    scores_out = [[] for _ in range(num_heads)]

    for start in range(0, len(slices), batch_size):
        chunk = slices[start:start + batch_size]
        imgs_batch = torch.stack(
            [load_slice_tensor(frame_dir, idxs, transform, size) for idxs, _ in chunk], dim=0
        ).to(device)
        scores = model({'imgs': imgs_batch})  # (b, num_heads, T)
        scores = scores.cpu().numpy()
        for i, (idxs, mask) in enumerate(chunk):
            valid_idx = np.asarray(idxs)[mask]
            frame_idx_out.extend(valid_idx.tolist())
            for h in range(num_heads):
                scores_out[h].extend(scores[i, h][mask].tolist())

    frame_idx_arr = np.array(frame_idx_out)
    _, uniq_pos = np.unique(frame_idx_arr, return_index=True)
    keep = uniq_pos[np.argsort(frame_idx_arr[uniq_pos])]
    frame_idx_final = frame_idx_arr[keep].tolist()
    scores_final = [np.asarray(scores_out[h])[keep].tolist() for h in range(num_heads)]
    return frame_idx_final, scores_final


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def plot_scores(vid, frame_idx, scores_per_head, fps, pred_by_head, gt_frame_idx, threshold, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    times = [(i - 1) / fps for i in frame_idx]
    num_heads = len(scores_per_head)
    fig, axes = plt.subplots(num_heads, 1, figsize=(12, 3 * num_heads), squeeze=False, sharex=True)
    for h in range(num_heads):
        ax = axes[h][0]
        ax.plot(times, scores_per_head[h], color='C0', linewidth=1, label=f'head{h} score')
        ax.axhline(threshold, color='gray', linestyle=':', label=f'threshold={threshold}')
        for b in pred_by_head[h]:
            ax.axvline((b - 1) / fps, color='red', linestyle='--', alpha=0.8)
        if gt_frame_idx:
            for g in gt_frame_idx:
                ax.axvline((g - 1) / fps, color='green', linestyle='-', alpha=0.5)
        ax.set_ylabel('score')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='upper right', fontsize=8)
    axes[-1][0].set_xlabel('time (s)')
    fig.suptitle(f'{vid}  (red dashed = predicted, green = GT)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_annotated_video(frame_dir, vlen, fps, pred_frame_idx, gt_frame_idx, out_path,
                            freeze_s=5.0, gt_match_tol_s=0.5):
    """Play the video normally, and at each predicted boundary freeze on that frame
    (with a red border + label) for `freeze_s` seconds before resuming playback.
    """
    first = cv2.imread(str(frame_dir / TEMPLATE.format(1)))
    if first is None:
        print(f'Could not read frames in {frame_dir}, skipping video render.')
        return
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    freeze_frames = max(1, int(round(freeze_s * fps)))
    gt_match_tol_frames = max(1, int(round(gt_match_tol_s * fps)))
    pred_sorted = sorted(int(p) for p in pred_frame_idx)
    gt_sorted = sorted(int(g) for g in gt_frame_idx) if gt_frame_idx else []
    next_boundary = 0

    for i in tqdm(range(1, vlen + 1), desc=f'rendering {out_path.parent.name}/{out_path.name}'):
        frame = cv2.imread(str(frame_dir / TEMPLATE.format(i)))
        if frame is None:
            continue
        cv2.putText(frame, f't={(i - 1) / fps:.2f}s', (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)

        if next_boundary < len(pred_sorted) and i >= pred_sorted[next_boundary]:
            boundary_idx = pred_sorted[next_boundary]
            next_boundary += 1
            freeze_frame = frame.copy()
            cv2.rectangle(freeze_frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
            cv2.putText(freeze_frame, 'STEP BOUNDARY', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            if gt_sorted:
                matched = any(abs(boundary_idx - g) <= gt_match_tol_frames for g in gt_sorted)
                label = 'matches GT' if matched else 'no GT match nearby'
                color = (0, 255, 0) if matched else (0, 165, 255)
                cv2.putText(freeze_frame, label, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            for _ in range(freeze_frames):
                writer.write(freeze_frame)
    writer.release()


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True,
                         help='video file, a single frame folder, or a folder of frame folders / video files')
    parser.add_argument('--gt', default=None,
                         help='optional annotation pkl (e.g. val_annotation.pkl) to overlay GT boundaries and print F1')
    parser.add_argument('--output-dir', default='output/infer')
    parser.add_argument('--threshold', type=float, default=None, help='default: cfg.TEST.THRESHOLD')
    parser.add_argument('--min-peak-dist', type=int, default=None, help='default: cfg.TEST.MIN_PEAK_DIST')
    parser.add_argument('--fps', type=float, default=None, help='fallback fps if not known/in --gt (default 25.0)')
    parser.add_argument('--batch-size', type=int, default=4, help='number of 10s slices per forward pass')
    parser.add_argument('--no-video', action='store_true', help='skip annotated.mp4 (plot + json still produced)')
    parser.add_argument('--freeze-seconds', type=float, default=5.0,
                         help='freeze the video on each predicted boundary frame for this many seconds before continuing')
    parser.add_argument('--overwrite-cache', action='store_true', help='re-extract frames even if cached')
    parser.add_argument('--device', default=None)
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    threshold = args.threshold if args.threshold is not None else cfg.TEST.THRESHOLD
    min_peak_dist = args.min_peak_dist if args.min_peak_dist is not None else cfg.TEST.MIN_PEAK_DIST
    num_heads = len(cfg.MODEL.HEAD_CHOICE)

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print('Loaded checkpoint {} (epoch {}), threshold={}, min_peak_dist={}'.format(
        args.checkpoint, ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?', threshold, min_peak_dist))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class TeeLogger:
        def __init__(self, filepath, stream):
            self.file = open(filepath, "a", encoding="utf-8", buffering=1)
            self.stream = stream
        def write(self, data):
            self.stream.write(data)
            self.stream.flush()
            self.file.write(data)
            self.file.flush()
        def flush(self):
            self.stream.flush()
            self.file.flush()

    sys.stdout = TeeLogger(output_dir / "infer.log", sys.stdout)
    sys.stderr = TeeLogger(output_dir / "infer.log", sys.stderr)

    cache_dir = output_dir / '_frames_cache'


    gt_dict = None
    if args.gt:
        with open(args.gt, 'rb') as f:
            gt_dict = pickle.load(f, encoding='latin1')

    entries = resolve_inputs(Path(args.input), cache_dir, args.overwrite_cache)
    print(f'Found {len(entries)} video(s) to process.')

    transform = transforms.Compose([
        transforms.Resize((cfg.INPUT.RESOLUTION, cfg.INPUT.RESOLUTION)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model_pred_dict = {}
    video_meta = {}
    for vid, frame_dir, known_fps in entries:
        vlen, fps, duration = get_video_info(vid, frame_dir, gt_dict, known_fps, args.fps)
        if vlen == 0:
            print(f'Skip {vid}: no frames found in {frame_dir}')
            continue
        slices = build_slices(vlen, duration, cfg.INPUT.SEQUENCE_LENGTH)
        frame_idx, scores = run_inference_for_video(
            model, device, frame_dir, slices, transform, cfg.INPUT.RESOLUTION, num_heads, args.batch_size)
        model_pred_dict[vid] = {'frame_idx': frame_idx, 'scores': scores}
        video_meta[vid] = {'frame_dir': frame_dir, 'vlen': vlen, 'fps': fps}
        print(f'[{vid}] scored {len(frame_idx)} frames over {len(slices)} slice(s) (fps={fps:.2f}, dur={duration:.1f}s)')

    if not model_pred_dict:
        print('Nothing to process.')
        return

    # Resolve final predicted boundaries per video (+ GT-based F1 report if --gt given)
    pred_by_vid = {}
    if gt_dict is not None:
        results, pred_dict, _ = eval_f1(model_pred_dict, args.gt, num_heads=num_heads, threshold=threshold,
                                         return_pred_dict=True, rel_dis_thres=[cfg.TEST.RELDIS_THRESHOLD],
                                         min_peak_dist=min_peak_dist)
        n_matched = len(pred_dict)
        for head in range(num_heads):
            f1, rec, prec = results[cfg.TEST.RELDIS_THRESHOLD][head]
            print(f'[GT metrics] head{head}: F1={f1:.4f} Rec={rec:.4f} Prec={prec:.4f} '
                  f'(over {n_matched} video(s) with GT)')
        for vid, entry in model_pred_dict.items():
            if vid in pred_dict:
                pred_by_vid[vid] = pred_dict[vid]
            else:
                # not present in the GT pkl (e.g. a genuinely new video mixed into the
                # same run) -> fall back to direct threshold/peak detection.
                pred_by_vid[vid] = [
                    get_idx_from_score_by_threshold(threshold=threshold, seq_indices=entry['frame_idx'],
                                                     seq_scores=entry['scores'][head], min_peak_dist=min_peak_dist)[0]
                    for head in range(num_heads)
                ]
    else:
        for vid, entry in model_pred_dict.items():
            pred_by_vid[vid] = [
                get_idx_from_score_by_threshold(threshold=threshold, seq_indices=entry['frame_idx'],
                                                 seq_scores=entry['scores'][head], min_peak_dist=min_peak_dist)[0]
                for head in range(num_heads)
            ]

    # Visualize
    for vid, entry in model_pred_dict.items():
        vid_out = output_dir / vid
        vid_out.mkdir(parents=True, exist_ok=True)
        fps = video_meta[vid]['fps']
        vlen = video_meta[vid]['vlen']
        frame_dir = video_meta[vid]['frame_dir']
        pred_by_head = pred_by_vid[vid]
        gt_frame_idx = [int(x) for x in gt_dict[vid]['substages_myframeidx'][0]] if (gt_dict and vid in gt_dict) else None

        result = {
            'vid': vid, 'fps': float(fps), 'num_frames': int(vlen), 'threshold': float(threshold),
            'predicted_boundaries': {
                f'head{h}': {
                    'frame_idx': [int(p) for p in pred_by_head[h]],
                    'time_s': [round((int(p) - 1) / fps, 3) for p in pred_by_head[h]],
                }
                for h in range(num_heads)
            },
        }
        if gt_frame_idx is not None:
            result['gt_boundaries'] = {
                'frame_idx': [int(g) for g in gt_frame_idx],
                'time_s': [round((int(g) - 1) / fps, 3) for g in gt_frame_idx],
            }
        with open(vid_out / 'boundaries.json', 'w') as f:
            json.dump(result, f, indent=2)

        try:
            plot_scores(vid, entry['frame_idx'], entry['scores'], fps, pred_by_head, gt_frame_idx, threshold,
                        vid_out / 'score_curve.png')
        except ImportError:
            print('matplotlib not installed, skipping score_curve.png (pip install matplotlib)')

        if not args.no_video:
            render_annotated_video(frame_dir, vlen, fps, pred_by_head[num_heads - 1], gt_frame_idx,
                                    vid_out / 'annotated.mp4', freeze_s=args.freeze_seconds)

        print(f'[{vid}] saved -> {vid_out}')

    # AI Research Loop: Save aggregated predictions.json
    combined_preds = {}
    for vid, meta in video_meta.items():
        if vid in pred_by_vid:
            final_head_preds = pred_by_vid[vid][num_heads - 1]
            fps = meta['fps']
            combined_preds[vid] = [round((int(p) - 1) / fps, 3) for p in final_head_preds]

    preds_file = output_dir / 'predictions.json'
    with open(preds_file, 'w', encoding='utf-8') as pf:
        json.dump(combined_preds, pf, indent=2)
    print(f'Aggregated predictions saved to {preds_file}')

    # Also save efficient_gebd_preds.json for legacy benchmark script
    with open(output_dir / 'efficient_gebd_preds.json', 'w', encoding='utf-8') as pf:
        json.dump(combined_preds, pf, indent=2)

    # Standardized evaluation against GT
    try:
        import sys
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from tools.eval_step_segment_predictions import load_ground_truth, evaluate_predictions
        gt = load_ground_truth(repo_root / "data")
        eval_gt = {k: v for k, v in gt.items() if k in combined_preds} or gt
        metrics_report = evaluate_predictions(eval_gt, combined_preds, thresholds=[0.25, 0.5, 1.0], primary_window=0.5)
        metrics_report["model"] = "EfficientGEBD"
        metrics_report["checkpoint"] = str(args.checkpoint)
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as mf:
            json.dump(metrics_report, mf, indent=2)
        print(f"Standardized metrics report -> {metrics_file}")
        macro_f1 = metrics_report["primary_metrics"].get("macro_f1", 0.0)
        print(f"EfficientGEBD Macro F1@0.5s: {macro_f1:.4f}")
    except Exception as e:
        print(f"Notice: Standardized evaluation skipped: {e}")


if __name__ == '__main__':
    main()


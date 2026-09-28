import argparse
import hashlib
import os
import pickle
import time
from collections import defaultdict
from contextlib import suppress
from datetime import timedelta
import random
import numpy as np
import torch
import torch.distributed as dist
from tabulate import tabulate
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from datasets import build_dataloader
from modeling import cfg, build_model
from solver.solver_utils import build_optimizer, build_scheduler
from utils.distribute import synchronize, all_gather, is_main_process
from utils.eval import eval_f1
from utils.misc import SmoothedValue, MetricLogger
try:
    import wandb
except ImportError:
    wandb = None

def make_inputs(inputs, device):
    if isinstance(inputs, list):
        return [img.to(device) for img in inputs]
    else:
        return inputs.to(device)



def make_targets(cfg, inputs, device):
    targets = inputs['labels'].to(device)
    return targets


def train_one_epoch(cfg, args, model, device, optimizer, scheduler, data_loader, summary_writer, auto_cast, loss_scaler, epoch, nan_streak=0):
    model.train()

    max_consecutive_nan = cfg.SOLVER.MAX_CONSECUTIVE_NAN
    clip_grad = cfg.SOLVER.CLIP_GRAD if cfg.SOLVER.CLIP_GRAD > 0 else 1e9  # 1e9 == effectively "no clipping", just probe the norm

    def _abort_on_nan_streak(i):
        raise RuntimeError(
            f"Aborting: {nan_streak} consecutive non-finite loss/gradient steps at epoch {epoch} step {i}. "
            "Model weights are very likely permanently NaN/Inf at this point and will not recover on their "
            "own (every subsequent forward pass will keep producing non-finite loss too). Resume training "
            "from the last known-good checkpoint (model_last_good.pth or model_best.pth in the output dir) "
            "with a lower learning rate, e.g.:\n"
            f"  --resume {os.path.join(args.output_dir, 'model_last_good.pth')} --reset-lr SOLVER.LR <lower_lr>\n"
            "Also consider lowering SOLVER.CLIP_GRAD further to prevent recurrence."
        )

    start = time.time()
    for i, inputs in enumerate(data_loader):
        
        samples = make_inputs(inputs['imgs'], device)
        targets = inputs['labels'].to(device)
        masks = None

        model_start = time.time()
        with auto_cast():
            loss_dict = model(samples, targets, masks)
            total_loss = sum(loss_dict.values())

        if not torch.isfinite(total_loss):
            nan_streak += 1
            if is_main_process():
                print(f"WARNING: non-finite loss {total_loss} at epoch {epoch} step {i}, skipping optimizer step! (consecutive={nan_streak})", flush=True)
            optimizer.zero_grad()
            if max_consecutive_nan > 0 and nan_streak >= max_consecutive_nan:
                _abort_on_nan_streak(i)
            continue

        optimizer.zero_grad()
        if cfg.SOLVER.AMPE:
            loss_scaler.scale(total_loss).backward()
            loss_scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            grad_finite = torch.isfinite(grad_norm)
            # IMPORTANT: always call scaler.step()/update() every iteration in
            # that exact order, regardless of grad_finite. GradScaler tracks
            # the inf/nan it saw during unscale_() internally and
            # automatically no-ops the underlying optimizer.step() itself -
            # manually skipping scaler.step() here (while still calling
            # update()) desyncs its per-optimizer state machine and crashes
            # on a *later*, unrelated iteration (e.g. "unscale_() has already
            # been called" / assertion inside GradScaler.step()).
            loss_scaler.step(optimizer)
            loss_scaler.update()
            if grad_finite:
                scheduler.step()
        else:
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            grad_finite = torch.isfinite(grad_norm)
            if grad_finite:
                optimizer.step()
                scheduler.step()

        if not grad_finite:
            nan_streak += 1
            if is_main_process():
                print(f"WARNING: non-finite grad norm {grad_norm} at epoch {epoch} step {i}, skipping optimizer step! (consecutive={nan_streak})", flush=True)
            if max_consecutive_nan > 0 and nan_streak >= max_consecutive_nan:
                _abort_on_nan_streak(i)
            continue

        nan_streak = 0

        if is_main_process():
            summary_writer.global_step += 1

            if len(loss_dict) > 1:
                summary_writer.update(**loss_dict)

            summary_writer.update(lr=optimizer.param_groups[0]['lr'], total_loss=total_loss,
                                  total_time=time.time() - start, model_time=time.time() - model_start)
            start = time.time()

            speed = summary_writer.total_time.avg
            eta = str(timedelta(seconds=int((len(data_loader) - i - 1) * speed)))
            if i % 10 == 0:
                print('Epoch{:02d} ({:04d}/{:04d}): {}, Eta:{}'.format(epoch,
                                                                       i,
                                                                       len(data_loader),
                                                                       str(summary_writer),
                                                                       eta
                                                                       ), flush=True)

    return nan_streak


@torch.no_grad()
def validate(cfg, args, model, device, data_loader, epoch):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if cfg.INPUT.END_TO_END:
        return validate_end_to_end(cfg, args, model, device, data_loader, epoch)

    model_pred_dict = {}
    model.eval()
    start_time = time.time()
    num_frames = 0
    for i, inputs in enumerate(tqdm(data_loader, total=len(data_loader))):
        samples = make_inputs(inputs, device)
        num_frames += samples['imgs'].shape[0]

        vids = inputs['vid']
        frame_idxs = inputs['frame_idx']
        scores = model(samples)

        scores = scores.cpu().numpy()
        for vid, frame_idx, score in zip(vids, frame_idxs, scores):
            if vid not in model_pred_dict.keys():
                model_pred_dict[vid] = {}
                model_pred_dict[vid]['frame_idx'] = []
                model_pred_dict[vid]['scores'] = []
            model_pred_dict[vid]['frame_idx'].append(frame_idx)
            model_pred_dict[vid]['scores'].append(score)

    synchronize()
    metrics = {}
    data_list = all_gather(model_pred_dict)
    if not is_main_process():
        metrics['F1'] = 0.00
        return metrics
    total_time = time.time() - start_time
    print('Cost {:.2f}s for evaluating {} videos, {:.4f}s/video'.format(total_time, len(data_loader.dataset), total_time / len(data_loader.dataset)))
    print('{:.9f}ms/frame'.format(total_time * 1000 / num_frames))

    model_pred_dict = defaultdict(dict)
    for p in data_list:
        for vid in p:
            if 'frame_idx' not in model_pred_dict[vid]:
                model_pred_dict[vid]['frame_idx'] = []
                model_pred_dict[vid]['scores'] = []

            model_pred_dict[vid]['frame_idx'].extend(p[vid]['frame_idx'])
            model_pred_dict[vid]['scores'].extend(p[vid]['scores'])

    for vid in model_pred_dict:
        frame_idx = np.array(model_pred_dict[vid]['frame_idx'])
        scores = np.array(model_pred_dict[vid]['scores'])
        _, indices = np.unique(frame_idx, return_index=True)
        frame_idx = frame_idx[indices]
        scores = scores[indices]

        indices = np.argsort(frame_idx)
        model_pred_dict[vid]['frame_idx'] = frame_idx[indices].tolist()
        model_pred_dict[vid]['scores'] = scores[indices].tolist()

    gt_path = f'data/k400_mr345_{data_loader.dataset.split}_min_change_duration0.3.pkl'
    f1, rec, prec = eval_f1(model_pred_dict, gt_path, threshold=cfg.TEST.THRESHOLD, protocol=cfg.TEST.PROTOCOL)
    print('F1: {:.4f}, Rec: {:.4f}, Prec: {:.4f}'.format(f1, rec, prec))
    metrics['F1'] = f1
    metrics['Rec'] = rec
    metrics['Prec'] = prec
    return metrics


@torch.no_grad()
def validate_end_to_end(cfg, args, model, device, data_loader, epoch):
    metrics = {}
    if cfg.TEST.PRED_FILE:
        if not is_main_process():
            return metrics, metrics
        with open(cfg.TEST.PRED_FILE, 'rb') as f:
            model_pred_dict = pickle.load(f)
        print(f'Load results from {cfg.TEST.PRED_FILE}.')
    else:
        model_pred_dict = defaultdict(dict)
        model.eval()

        start_time = time.time()
        num_frames = 0
        for i, inputs in enumerate(tqdm(data_loader, total=len(data_loader))):
            samples = make_inputs(inputs['imgs'], device)
            targets = inputs['labels'].to(device)
            
            num_frames += (targets.shape[0] * targets.shape[1])

            sample_seeds = None
            if getattr(cfg.DIFFUSION, 'DETERMINISTIC_SAMPLE_SEEDING', False):
                # exp_004_deterministic_noise_seeding: derive a stable,
                # sample-identifying seed per batch row from hash(video_id,
                # chunk_index) -- independent of global RNG stream order /
                # total chunk count, so it no longer depends on
                # --val-overlap-seconds. chunk_index falls back to the
                # starting frame index when unavailable (still stable per vid).
                sample_seeds = []
                for batch_idx, vid in enumerate(inputs['vid']):
                    chunk_idx = inputs.get('chunk_idx', [None] * len(inputs['vid']))[batch_idx] \
                        if 'chunk_idx' in inputs else int(inputs['frame_indices'][batch_idx][0])
                    key = f"{vid}::{chunk_idx}"
                    seed = int(hashlib.sha256(key.encode('utf-8')).hexdigest()[:8], 16)
                    sample_seeds.append(seed)

            outputs = model(samples, targets, sampling_timesteps=cfg.DIFFUSION.SAMPLING_TIMESTEPS,
                             sample_seeds=sample_seeds)  # (b, t)
            for batch_idx, frame_indices in enumerate(inputs['frame_indices']):
                vid = inputs['vid'][batch_idx]
                scores = outputs[batch_idx]
                if 'frame_masks' in inputs:
                    frame_mask = inputs['frame_masks'][batch_idx]
                    frame_indices = frame_indices[frame_mask]
                    scores = scores[frame_mask]

                model_pred_dict[vid]['frame_idx'] = frame_indices.tolist()
                model_pred_dict[vid]['scores'] = scores.tolist()


        synchronize()
        data_list = all_gather(model_pred_dict)
        if not is_main_process():
            metrics['F1'] = 0.00
            return metrics, metrics
        total_time = time.time() - start_time
        print('Cost {:.2f}s for evaluating {} videos, {:.4f}s/video'.format(total_time, len(data_loader.dataset), total_time / len(data_loader.dataset)))
        print('{:.9f}ms/frame'.format(total_time * 1000 / num_frames))

        model_pred_dict = {}
        for p in data_list:
            model_pred_dict.update(p)

    if not cfg.TEST.PRED_FILE:
        save_path = os.path.join(args.output_dir, 'model_pred_dict_ep{}.pkl'.format(epoch))
        with open(save_path, 'wb') as f:
            pickle.dump(model_pred_dict, f)
        print(f'Saved results to {save_path}.')

    if data_loader.dataset.name == 'GEBD':
        gt_path = os.path.join('data', f'k400_mr345_{data_loader.dataset.split}_min_change_duration0.3.pkl')
    elif data_loader.dataset.name == 'TAPOS':
        gt_path = os.path.join('data', f'TAPOS_for_GEBD_{data_loader.dataset.split}.pkl')
    elif data_loader.dataset.name in ('SEWING', 'SEWING_CHUNKED'):
        suffix = '_chunked' if data_loader.dataset.name == 'SEWING_CHUNKED' else ''
        gt_path = getattr(data_loader.dataset, 'ann_path', None) or os.path.join('data', 'diff_gebd_dataset', f'{data_loader.dataset.split}_annotation{suffix}.pkl')
        if not os.path.exists(gt_path):
            alt_gt = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'diff_gebd_dataset', f'{data_loader.dataset.split}_annotation{suffix}.pkl'))
            if os.path.exists(alt_gt):
                gt_path = alt_gt
    else:
        raise NotImplemented

    if args.all_thres:
        rel_dis_thres = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    else:
        rel_dis_thres = [0.05]

    results, pred_dict, gt_dict = eval_f1(model_pred_dict, gt_path,
                                                        threshold=cfg.TEST.THRESHOLD,
                                                        return_pred_dict=True,
                                                        rel_dis_thres=rel_dis_thres,
                                                        protocol=cfg.TEST.PROTOCOL)
    list_rec = []
    list_prec = []
    list_f1 = []
    
    for th in rel_dis_thres:
        f1, rec, prec = results[th]
        list_rec.append(rec)
        list_prec.append(prec)
        list_f1.append(f1)

    headers = rel_dis_thres + ['Avg']

    avg_rec = np.mean(list_rec)
    avg_prec = np.mean(list_prec)
    avg_F1 = np.mean(list_f1)
    
    
    tabulate_data = [
        ['Recall'] + list_rec + [avg_rec],
        ['Precision'] + list_prec + [avg_prec],
        ['F1'] + list_f1 + [avg_F1],
    ]
    print(f'Results for {data_loader.dataset.name}_{data_loader.dataset.split}:')
    print(tabulate(tabulate_data, headers=headers, floatfmt='.4f'))

    f1, rec, prec = results[0.05]
    print('___F1@0.05: {:.4f}, Rec: {:.4f}, Prec: {:.4f}'.format(f1, rec, prec))

    return results


def main(cfg, args):
    train_data_loader = build_dataloader(cfg, args, cfg.DATASETS.TRAIN, train=True)
    val_data_loader = build_dataloader(cfg, args, cfg.DATASETS.TEST, train=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = build_model(cfg)
    model = model.to(device)

    if is_main_process() and not args.test_only:
        print(model)

    start_epoch = -1
    if args.resume:
        state_dict = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(state_dict['model'], strict=False)
        start_epoch = state_dict['epoch']
        if is_main_process():
            print('Loaded from {}, Epoch: {}'.format(args.resume, start_epoch), flush=True)

    exp_name = '_ann{}_dim{}_len{}'.format(cfg.INPUT.ANNOTATORS,
                                            cfg.MODEL.DIMENSION,
                                            cfg.INPUT.SEQUENCE_LENGTH,   
                                            )
    output_dir = cfg.OUTPUT_DIR + exp_name
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    
    with open(os.path.join(output_dir,'config.yaml'), 'w') as f:
        f.write(cfg.dump())

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
        torch.multiprocessing.set_start_method('spawn')
    
    if args.test_only:
        results = validate(cfg, args, model, device, val_data_loader, epoch=-1)
        return
    
    optimizer_config = {}
    optimizer_config['warmup'] = cfg.SOLVER.WARMUP
    optimizer_config['epochs'] = cfg.SOLVER.MAX_EPOCHS
    optimizer_config['warmup_epochs'] = cfg.SOLVER.WARMUP_EPOCHS
    optimizer_config['schedule_type'] = cfg.SOLVER.SCHEDULER
    optimizer_config["schedule_steps"] = cfg.SOLVER.MILESTONES
    optimizer_config["gamma"] = cfg.SOLVER.GAMMA
    optimizer = build_optimizer(cfg, [p for p in model.parameters() if p.requires_grad])
    scheduler = build_scheduler(optimizer, optimizer_config, len(train_data_loader))
     
    if args.resume:
        for name, obj in [('optimizer', optimizer), ('scheduler', scheduler)]:
            if name in state_dict:
                obj.load_state_dict(state_dict[name])
                if is_main_process():
                    print('Loaded {} from {}'.format(name, args.resume), flush=True)

        if args.reset_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg.SOLVER.LR
                param_group['initial_lr'] = cfg.SOLVER.LR
            if is_main_process():
                print(f'Overrode optimizer learning rate to {cfg.SOLVER.LR} via --reset-lr', flush=True)

    summary_writer = MetricLogger(log_dir=os.path.join(output_dir, 'logs')) if is_main_process() else None
    if summary_writer is not None:
        summary_writer.add_meter('lr', SmoothedValue(fmt='{value:.5f}'))
        summary_writer.add_meter('total_time', SmoothedValue(fmt='{avg:.3f}s'))
        summary_writer.add_meter('model_time', SmoothedValue(fmt='{avg:.3f}s'))

    if cfg.SOLVER.AMPE:
        auto_cast = (lambda: torch.amp.autocast('cuda')) if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast
        loss_scaler = (torch.amp.GradScaler('cuda') if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler') else torch.cuda.amp.GradScaler())
    else:
        auto_cast = suppress
        loss_scaler = None

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    best_f1 = 0.
    early_stop_patience = cfg.SOLVER.EARLY_STOP_PATIENCE
    early_stop_min_delta = cfg.SOLVER.EARLY_STOP_MIN_DELTA
    epochs_no_improve = 0
    nan_streak = 0
    for epoch in range(start_epoch + 1, cfg.SOLVER.MAX_EPOCHS):
        
        nan_streak = train_one_epoch(cfg, args, model, device, optimizer, scheduler, train_data_loader, summary_writer, auto_cast, loss_scaler, epoch, nan_streak)

        if is_main_process() and nan_streak == 0:
            # Best-effort safety net: only overwrite model_last_good.pth when
            # the epoch's final step was healthy (finite loss + grad), so a
            # NaN blow-up never gets saved as "good". This is at most one
            # epoch stale and is always safe to --resume from, unlike the
            # potentially-corrupted state torch.save'd right when a NaN
            # streak hits SOLVER.MAX_CONSECUTIVE_NAN.
            model_state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
            torch.save({
                'model': model_state_dict,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, os.path.join(output_dir, 'model_last_good.pth'))

        if len(cfg.DIFFUSION.VALIDATION_TIMESTEPS) == 1 or cfg.DIFFUSION.DETERMINISTIC:
            results = validate(cfg, args, model, device, val_data_loader, epoch)

            if is_main_process():
                if args.wandb:
                    wandb.log({'Val/F1@0.05' : results[0.05][0],
                            'Val/Rec@0.05' : results[0.05][1],
                            'Val/Prec@0.05' : results[0.05][2],}, step=epoch)
        
        else:
            for sampling_steps in cfg.DIFFUSION.VALIDATION_TIMESTEPS:
                cfg.DIFFUSION.SAMPLING_TIMESTEPS = sampling_steps
                results = validate(cfg, args, model, device, val_data_loader, epoch)
                if is_main_process():
                    if args.wandb:
                        wandb.log({f'Val_st{sampling_steps}/F1@0.05' : results[0.05][0],
                            f'Val_st{sampling_steps}/Rec@0.05' : results[0.05][1],
                            f'Val_st{sampling_steps}/Prec@0.05' : results[0.05][2],}, step=epoch)
                        
        should_stop = False
        if is_main_process():
            if args.wandb:
                wandb.log({'Train/loss' : summary_writer.meters['total_loss'].global_avg,
                        'Train/lr' : summary_writer.meters['lr'].value
                    }, step=epoch)
                    
            model_state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
            
            save_path = None
            if results[0.05][0] >= best_f1 + early_stop_min_delta:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if results[0.05][0] >= best_f1:
                best_f1 = results[0.05][0]
                save_path = os.path.join(output_dir, f'model_best.pth')
                torch.save({
                    'model': model_state_dict,
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, save_path)    
                print(f'Saved best model at {epoch}')

            if results[0.05][0] >= 0.73:
                save_path = os.path.join(output_dir, f'model_epoch{epoch:02d}.pth')
                torch.save({
                    'model': model_state_dict,
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, save_path)
            with open(os.path.join(output_dir, 'metrics.txt'), 'a') as f:
                f.write('Epoch: {:02d},Rel@0.05 F1: {:.4f}, Rec: {:.4f}, Prec: {:.4f}\n'.format(epoch, results[0.05][0], results[0.05][1], results[0.05][2]))
            
            if save_path:
                print('Saved to {}'.format(save_path))

            if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
                should_stop = True
                print(f'Early stopping at epoch {epoch}: no improvement in F1@0.05 '
                      f'(best={best_f1:.4f}) for {epochs_no_improve} epochs '
                      f'(patience={early_stop_patience}).')

        if args.distributed:
            # Broadcast rank 0's stop decision so every rank breaks the loop
            # together (DDP forward/backward requires all ranks to stay in
            # lock-step; only rank 0 computes val metrics / epochs_no_improve).
            stop_tensor = torch.tensor([1 if should_stop else 0], device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())

        if should_stop:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", help="path to config file", type=str)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--reset-lr", action='store_true', help='Override checkpoint optimizer LR with cfg.SOLVER.LR')
    parser.add_argument("--test-only", action='store_true')
    parser.add_argument("--train-test", action='store_true')
    parser.add_argument("--all-thres", action='store_true', help='test using all thresholds [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]')
    parser.add_argument('--gebd_data_dir', default='your_kinetics_folder', type=str)
    parser.add_argument('--tapos_data_dir', default='your_tapos_folder', type=str)
    parser.add_argument('--sewing_data_dir', default='data/diff_gebd_dataset/images', type=str)
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--proj_name", type=str, default='your_proj')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--cuda_deterministic', action='store_true', default=False)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.local_rank:
        args.local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    args.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = args.num_gpus > 1

    if torch.cuda.is_available():
        if args.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            dist.barrier()
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if is_main_process():
        print('Args: \n{}'.format(args))
        print('Configs: \n{}'.format(cfg))

        if args.wandb:
            assert wandb is not None, 'wandb is not installed; run `pip install wandb` or drop --wandb'
            exp_name = cfg.OUTPUT_DIR.replace('output/', '') + '_' + args.exp_name
            run = wandb.init(entity='user_wandb', 
                            project=args.proj_name,
                            name=exp_name)
    
    seed = args.seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    np.random.seed(seed)
    random.seed(seed) 
    if args.cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    main(cfg, args)

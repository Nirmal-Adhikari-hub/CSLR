import os
import sys
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from evaluation.slr_eval.wer_calculation import evaluate
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

import torch.distributed as dist

def is_main_process():
    # Returns True on either non-distributed runs or on rank 0 in DDP
    return not dist.is_initialized() or dist.get_rank() == 0

def seq_train(loader, model, optimizer, device, epoch_idx, recoder):
    model.train()
    scaler = GradScaler()
    lr = optimizer.optimizer.param_groups[0]['lr']

    # only rank 0 shows the progress bar
    disable_tqdm = dist.is_initialized() and dist.get_rank() != 0
    pbar = tqdm(loader, ncols=100, disable=disable_tqdm)

    for batch_idx, data in enumerate(pbar):
        vid, vid_lgt, label, label_lgt = data[0], data[1], data[2], data[3]
        vid = vid.to(device); vid_lgt = vid_lgt.to(device)
        label = label.to(device); label_lgt = label_lgt.to(device)

        optimizer.zero_grad()
        with autocast():
            ret = model(vid, vid_lgt, label=label, label_lgt=label_lgt)
            loss = model.module.criterion_calculation(ret, label, label_lgt)

        # 1) Detect NaN/Inf and log once
        if torch.isnan(loss) or torch.isinf(loss):
            if is_main_process():
                recoder.print_log(
                    f"⚠️  NaN/Inf loss at epoch {epoch_idx}, batch {batch_idx}. "
                    f"Lowering LR from {lr:.2e} to {lr*0.5:.2e}"
                )
            # lower LR by half
            for g in optimizer.optimizer.param_groups:
                g['lr'] = lr * 0.5
            loss = torch.zeros_like(loss)

        # 2) Backward + gradient clipping + optimizer step
        scaler.scale(loss).backward()

        # unscale first, then clip
        scaler.unscale_(optimizer.optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        scaler.step(optimizer.optimizer)
        scaler.update()

        # logging
        if batch_idx % recoder.log_interval == 0 and is_main_process():
            recoder.print_log(
                f"\tEpoch: {epoch_idx}, Batch({batch_idx}/{len(loader)}) "
                f"Loss: {loss.item():.6f}  lr:{optimizer.optimizer.param_groups[0]['lr']:.6f}"
            )
        pbar.set_postfix({'Loss': loss.item()})

    # step scheduler once per epoch
    optimizer.scheduler.step()
    if is_main_process():
        recoder.print_log(f"\tEpoch {epoch_idx} finished. Current LR: {optimizer.optimizer.param_groups[0]['lr']:.2e}")


def seq_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder,
             evaluate_tool="python"):
    model.eval()
    total_sent = []
    total_info = []
    total_conv_sent = []
    stat = {i: [0, 0] for i in range(len(loader.dataset.dict))}

    # initialize here so that later `del conv_ret` is safe
    conv_ret = None
    lstm_ret = None

    for batch_idx, data in enumerate(tqdm(loader, ncols=100)):
        if is_main_process():
            recoder.record_timer("device")

        vid = data[0].to(device)
        vid_lgt = data[1].to(device)
        label = data[2].to(device)
        label_lgt = data[3].to(device)

        with torch.no_grad():
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)

        total_info += [f.split("|")[0] for f in data[-1]]
        total_sent += ret_dict['recognized_sents']
        total_conv_sent += ret_dict['conv_sents']

    try:
        python_eval = (evaluate_tool == "python")

        write2file(f"{work_dir}output-hypothesis-{mode}.ctm",
                   total_info, total_sent)
        write2file(f"{work_dir}output-hypothesis-{mode}-conv.ctm",
                   total_info, total_conv_sent)

        conv_ret = evaluate(
            prefix=work_dir,
            mode=mode,
            output_file=f"output-hypothesis-{mode}-conv.ctm",
            evaluate_dir=cfg.dataset_info['evaluation_dir'],
            evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
            output_dir=f"epoch_{epoch}_result/",
            python_evaluate=python_eval,
        )
        lstm_ret = evaluate(
            prefix=work_dir,
            mode=mode,
            output_file=f"output-hypothesis-{mode}.ctm",
            evaluate_dir=cfg.dataset_info['evaluation_dir'],
            evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
            output_dir=f"epoch_{epoch}_result/",
            python_evaluate=python_eval,
            triplet=True,
        )
    except Exception as e:
        if is_main_process():
            print(f"Unexpected error during evaluation: {e}")
        lstm_ret = 100.0

    # clean up safely
    if conv_ret is not None:
        del conv_ret
    del total_sent, total_info, total_conv_sent, vid, vid_lgt, label, label_lgt

    if is_main_process():
        recoder.print_log(f"Epoch {epoch}, {mode} {lstm_ret:2.2f}%", f"{work_dir}/{mode}.txt")

    return lstm_ret



def seq_feature_generation(loader, model, device, mode, work_dir, recoder):
    model.eval()

    src_path = os.path.abspath(f"{work_dir}{mode}")
    tgt_path = os.path.abspath(f"./features/{mode}")
    if not os.path.exists("./features/"):
        os.makedirs("./features/")

    if os.path.islink(tgt_path):
        curr_path = os.readlink(tgt_path)
        if work_dir[1:] in curr_path and os.path.isabs(curr_path):
            return
        else:
            os.unlink(tgt_path)
    else:
        if os.path.exists(src_path) and len(loader.dataset) == len(os.listdir(src_path)):
            os.symlink(src_path, tgt_path)
            return

    for batch_idx, data in tqdm(enumerate(loader)):
        recoder.record_timer("device")
        vid = data[0].to(device, non_blocking=True)
        vid_lgt = data[1].to(device, non_blocking=True)
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt)
        if not os.path.exists(src_path):
            os.makedirs(src_path)
        start = 0
        for sample_idx in range(len(vid)):
            end = start + data[3][sample_idx]
            filename = f"{src_path}/{data[-1][sample_idx].split('|')[0]}_features.npy"
            save_file = {
                "label": data[2][start:end],
                "features": ret_dict['framewise_features'][sample_idx][:, :vid_lgt[sample_idx]].T.cpu().detach(),
            }
            np.save(filename, save_file)
            start = end
        assert end == len(data[2])
    os.symlink(src_path, tgt_path)


def write2file(path, info, output):
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                "{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx],
                                                 word_idx * 1.0 / 100,
                                                 (word_idx + 1) * 1.0 / 100,
                                                 word[0]))

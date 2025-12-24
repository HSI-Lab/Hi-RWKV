# train_with_logging.py
import os
import csv
import time
import json
import math
import shutil
import random
import argparse
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from torch import nn
from torchvision import transforms
from PIL import Image

# ==== your project deps ====
import utils.data_load_operate as data_load_operate
from utils.Loss import head_loss, resize
from utils.evaluation import Evaluator
from utils.HSICommonUtils import ImageStretching
from utils.setup_logger import setup_logger
from utils.visual_predict import visualize_predict
from model.HiRWKV_model import HiRWKV
from calflops import calculate_flops

torch.autograd.set_detect_anomaly(True)


# ---------------------------
# Utils
# ---------------------------
def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def orthoreg_sum(model: nn.Module) -> torch.Tensor:
    device = next(model.parameters()).device
    total = torch.tensor(0.0, device=device)
    for m in model.modules():
        if hasattr(m, "regularization_loss") and callable(m.regularization_loss):
            total = total + m.regularization_loss()
    return total


def global_grad_norm(model: nn.Module) -> float:
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            v = p.grad.detach().data
            total_sq += float(v.norm(2).item()) ** 2
    return math.sqrt(total_sq)


def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)


def vis_a_image(gt_vis, pred_vis, save_single_predict_path, save_single_gt_path, only_vis_label=False):
    visualize_predict(gt_vis, pred_vis, save_single_predict_path, save_single_gt_path, only_vis_label=only_vis_label)
    visualize_predict(gt_vis, pred_vis, save_single_predict_path.replace('.png', '_mask.png'),
                      save_single_gt_path, only_vis_label=True)


def measure_inference(model, x, label_map, evaluator: Evaluator, loss_fn, device,
                      include_reg: bool = False, lambda_reg: float = 0.0):
    """
    Return (val_ce, val_total, reg_val, OA, mIoU, Kappa, mAcc, IOU, Acc, it_seconds)
    - include_reg=False: val_total=val_ce, reg_val=0
    - include_reg=True : val_total=val_ce + lambda_reg * reg_sum(model)
    """
    model.eval()
    evaluator.reset()
    with torch.no_grad():
        t0 = time.time()
        out = model(x)
        it = time.time() - t0

        y = label_map.unsqueeze(0)
        seg_logits = resize(input=out, size=y.shape[1:], mode='bilinear', align_corners=True)
        val_ce = head_loss(loss_fn, seg_logits, y.long())

        pred = torch.argmax(seg_logits, dim=1).cpu().numpy()
        Y_np = label_map.cpu().numpy()
        Y_255 = np.where(Y_np == -1, 255, Y_np)
        evaluator.add_batch(np.expand_dims(Y_255, axis=0), pred)
        OA = evaluator.Pixel_Accuracy()
        mIOU, IOU = evaluator.Mean_Intersection_over_Union()
        mAcc, Acc = evaluator.Pixel_Accuracy_Class()
        Kappa = evaluator.Kappa()

        
        if include_reg and lambda_reg != 0.0:
            reg_val = orthoreg_sum(model)
            val_total = val_ce + lambda_reg * reg_val
        else:
            reg_val = torch.tensor(0.0, device=val_ce.device)
            val_total = val_ce

    return (float(val_ce.detach().cpu().item()),
            float(val_total.detach().cpu().item()),
            float(reg_val.detach().cpu().item()),
            float(OA), float(mIOU), float(Kappa), float(mAcc), IOU, Acc, float(it))

def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True,
                   help="Path to yaml config. e.g. configs/hanchuan.yaml")
    return p

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = get_parser().parse_args()

    # 1) Load config (required)
    if (not hasattr(args, "config")) or (args.config is None) or (str(args.config).strip() == ""):
        raise ValueError("You must specify --config <path_to_yaml>")

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # required fields check
    if "dataset" not in cfg:
        raise ValueError("Config missing field: dataset")
    if "rwkv" not in cfg or not isinstance(cfg["rwkv"], list) or len(cfg["rwkv"]) == 0:
        raise ValueError("Config missing field: rwkv (non-empty list)")

    # 2) Resolve dataset name
    all_names = ['UP', 'HanChuan', 'HongHu', 'Houston', 'IndianPines', 'XuZhou']

    ds_cfg = cfg.get("dataset", {})
    ds_name = ds_cfg.get("name", None)
    ds_index = ds_cfg.get("index", None)

    if ds_name is None:
        if ds_index is None:
            raise ValueError("Config must provide dataset.name or dataset.index")
        ds_index = int(ds_index)
        if not (0 <= ds_index < len(all_names)):
            raise ValueError(f"dataset.index out of range: {ds_index}, allowed [0,{len(all_names)-1}]")
        ds_name = all_names[ds_index]
    else:
        # normalize / validate
        if ds_name not in all_names:
            raise ValueError(f"Unknown dataset.name='{ds_name}'. Allowed: {all_names}")

    split_image = ds_name in ['HanChuan', 'Houston']

    # 3) Resolve IO / train / model configs
    io_cfg = cfg.get("io", {})
    tr_cfg = cfg.get("train", {})
    m_cfg = cfg.get("model", {})

    data_set_path = str(io_cfg.get("data_set_path", "./data"))
    work_dir = str(io_cfg.get("work_dir", "./"))
    exp_name = str(io_cfg.get("exp_name", "RUNS"))
    log_dir = str(io_cfg.get("log_dir", "./logs_curve"))
    save_splits = bool(io_cfg.get("save_splits", False))

    lr = float(tr_cfg.get("lr", 2.5e-4))
    max_epoch = int(tr_cfg.get("max_epoch", 500))
    train_ratio = float(tr_cfg.get("train_ratio", 0.01))
    train_samples = int(tr_cfg.get("train_samples", 30))
    val_samples = int(tr_cfg.get("val_samples", 10))
    seed_list_str = str(tr_cfg.get("seed_list", "0"))
    seed_list = [int(x) for x in seed_list_str.split(",") if x.strip() != ""]

    lambda_reg = float(tr_cfg.get("lambda_reg", 1.0))
    weight_decay = float(tr_cfg.get("weight_decay", 0.0))
    label_smoothing = float(tr_cfg.get("label_smoothing", 0.0))
    use_dropout = bool(tr_cfg.get("use_dropout", False))
    dropout_p = float(tr_cfg.get("dropout_p", 0.0))
    use_aug = bool(tr_cfg.get("use_aug", False))

    val_with_reg = bool(tr_cfg.get("val_with_reg", False))
    no_early_stop = bool(tr_cfg.get("no_early_stop", False))
    early_stop_patience = int(tr_cfg.get("early_stop_patience", 30))
    log_train_oa = bool(tr_cfg.get("log_train_oa", False))
    test_each_epoch = bool(tr_cfg.get("test_each_epoch", False))
    test_every = int(tr_cfg.get("test_every", 1))

    record_computecost = bool(tr_cfg.get("record_computecost", True))

    group_num = int(m_cfg.get("group_num", 1))
    hidden_dim = int(m_cfg.get("hidden_dim", 128))

    rwkv_spec = cfg["rwkv"]

    # 4) dirs & logger
    #    include config name to avoid overwrite
    cfg_stem = Path(args.config).stem
    save_folder = os.path.join(work_dir, exp_name, 'HiRWKV',
                               f"{ds_name}_{cfg_stem}_Train_{train_ratio}")
    ensure_dir(save_folder)

    current_time = datetime.now().strftime("%m%d%H%M%S")
    # backup model file
    if os.path.exists("model/HiRWKV_model.py"):
        shutil.copy("model/HiRWKV_model.py", os.path.join(save_folder, f"{current_time}_HiRWKV_model.py"))

    # save effective config for reproducibility
    with open(os.path.join(save_folder, f"run_config_{current_time}.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    save_log_path = os.path.join(save_folder, f"train_tr{train_samples}_val{val_samples}.log")
    logger = setup_logger(name=f'{ds_name}', logfile=save_log_path)
    ensure_dir(log_dir)

    # 5) dataset load
    data, gt = data_load_operate.load_data(ds_name, data_set_path)
    H, W, C = data.shape
    gt_reshape = gt.reshape(-1)
    img = ImageStretching(data)
    class_count = int(max(np.unique(gt)))

    # 6) loss fn
    if label_smoothing > 0:
        loss_func = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=label_smoothing)
    else:
        loss_func = nn.CrossEntropyLoss(ignore_index=-1)

    # 7) transform
    transform = transforms.Compose([transforms.ToTensor()])
    x = transform(np.array(img)).unsqueeze(0).float().to(device)

    # 8) aggregators
    OA_ALL, AA_ALL, KPP_ALL, EACH_ACC_ALL = [], [], [], []

    # 9) record run config (json)
    paras = {
        "net_name": "HiRWKV",
        "dataset": ds_name,
        "config_path": str(args.config),
        "lr": lr,
        "max_epoch": max_epoch,
        "train_ratio": train_ratio,
        "seeds": seed_list,
        "lambda_reg": lambda_reg,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "use_dropout": int(use_dropout),
        "dropout_p": dropout_p,
        "use_aug": int(use_aug),
        "val_with_reg": int(val_with_reg),
        "no_early_stop": int(no_early_stop),
        "early_stop_patience": early_stop_patience,
        "test_each_epoch": int(test_each_epoch),
        "test_every": test_every,
        "record_computecost": int(record_computecost),
        "group_num": group_num,
        "hidden_dim": hidden_dim,
        "rwkv_spec": rwkv_spec,
    }
    logger.info(json.dumps(paras, indent=2, ensure_ascii=False))

    # 10) split sampling
    flag_list = [0, 0]  # ratio mode
    ratio_list = [train_ratio, 0.0050]  # val_ratio placeholder
    num_list = [train_samples, val_samples]

    for exp_idx, curr_seed in enumerate(seed_list):
        setup_seed(curr_seed)

        single_dir = os.path.join(save_folder, f'run{exp_idx}_seed{curr_seed}')
        vis_dir = os.path.join(single_dir, 'vis')
        ensure_dir(single_dir)
        ensure_dir(vis_dir)

        save_weight_path = os.path.join(single_dir, f"best_tr{num_list[0]}_val{num_list[1]}.pth")
        results_save_path = os.path.join(single_dir, f"result_tr{num_list[0]}_val{num_list[1]}.txt")
        predict_save_path = os.path.join(single_dir, f"pred_vis_tr{num_list[0]}_val{num_list[1]}.png")
        gt_save_path = os.path.join(single_dir, f"gt_vis_tr{num_list[0]}_val{num_list[1]}.png")

        # sampling
        train_idx, val_idx, test_idx, _ = data_load_operate.sampling(
            ratio_list, num_list, gt_reshape, class_count, flag_list[0]
        )
        train_label, val_label, test_label = data_load_operate.generate_image_iter(
            data, H, W, gt_reshape, (train_idx, val_idx, test_idx)
        )

        if save_splits:
            np.save(os.path.join(save_folder, f'train_idx_seed{curr_seed}.npy'), train_idx)
            np.save(os.path.join(save_folder, f'val_idx_seed{curr_seed}.npy'), val_idx)
            np.save(os.path.join(save_folder, f'test_idx_seed{curr_seed}.npy'), test_idx)

        train_label = train_label.to(device)
        val_label = val_label.to(device)
        test_label = test_label.to(device)

        # 11) model (config-driven rwkv)
        net = HiRWKV(
            in_channels=C,
            hidden_dim=hidden_dim,
            num_classes=class_count,
            group_num=group_num,
            rwkv_spec=rwkv_spec,
            dataset_name=ds_name,
        ).to(device)

        # complexity
        if record_computecost:
            net.eval()
            flops, macs1, para = calculate_flops(model=net, input_shape=(1, x.shape[1], x.shape[2], x.shape[3]))
            n_params = sum(p.numel() for p in net.parameters())
            logger.info(f"params:{n_params}, flops:{flops}")
        else:
            flops, n_params = 0.0, sum(p.numel() for p in net.parameters())

        # optimizer & scheduler
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99875)

        # 12) CSV header
        csv_path = os.path.join(log_dir, f"{ds_name}_{cfg_stem}_seed{curr_seed}.csv")
        csv_header = [
            "epoch","seed","lr",
            "train_loss","cls_loss","reg_loss","total_loss",
            "val_ce","val_total","reg_val","OA","mIoU","Kappa","MACC",
            "gap_ce","gap_total","grad_norm","inference_time","params","FLOPs","train_ratio",
            "lambda_reg","weight_decay","dropout_p","label_smoothing","use_aug","peak_mem_MB"
        ]
        if log_train_oa:
            csv_header += ["train_OA","gap_OA"]
        if test_each_epoch:
            csv_header += ["test_ce","test_total","test_reg","test_OA","test_mIoU","test_Kappa","test_MACC","test_infer_time"]

        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(csv_header)

        evaluator_val = Evaluator(num_class=class_count)
        evaluator_train = Evaluator(num_class=class_count) if log_train_oa else None
        best_metric, best_epoch = -1.0, -1
        best_state = None
        bad = 0

        # 13) training
        for epoch in range(max_epoch):
            net.train()
            y_train = train_label.unsqueeze(0)

            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # forward
            try:
                y_pred = net(x)
                cls_loss = head_loss(loss_func, y_pred, y_train.long())
                reg_train = orthoreg_sum(net)
                loss = cls_loss + lambda_reg * reg_train
                optimizer.zero_grad()
                loss.backward()
                gn = global_grad_norm(net)
                optimizer.step()
            except RuntimeError:
                # OOM fallback
                if not split_image:
                    split_image = True
                x1 = x[:, :, :x.shape[2] // 2 + 5, :]
                y1 = y_train[:, :x.shape[2] // 2 + 5, :]
                x2 = x[:, :, x.shape[2] // 2 - 5:, :]
                y2 = y_train[:, x.shape[2] // 2 - 5:, :]

                # part1
                y1_pred = net(x1)
                reg1 = orthoreg_sum(net)
                ls1 = head_loss(loss_func, y1_pred, y1.long()) + lambda_reg * reg1
                optimizer.zero_grad(); ls1.backward(); gn = global_grad_norm(net); optimizer.step()

                # part2
                y2_pred = net(x2)
                reg2 = orthoreg_sum(net)
                ls2 = head_loss(loss_func, y2_pred, y2.long()) + lambda_reg * reg2
                optimizer.zero_grad(); ls2.backward(); _ = global_grad_norm(net); optimizer.step()

                # merge logs
                cls_loss = (ls1 + ls2 - lambda_reg * (reg1 + reg2))
                reg_train = (reg1 + reg2)
                loss = ls1 + ls2

            # optional train OA
            train_OA = None
            if log_train_oa:
                net.eval()
                evaluator_train.reset()
                with torch.no_grad():
                    out_train = net(x)
                    y = y_train
                    logits_t = resize(input=out_train, size=y.shape[1:], mode='bilinear', align_corners=True)
                    pred_t = torch.argmax(logits_t, dim=1).cpu().numpy()
                    Yt_np = train_label.cpu().numpy()
                    Yt_255 = np.where(Yt_np == -1, 255, Yt_np)
                    evaluator_train.add_batch(np.expand_dims(Yt_255, axis=0), pred_t)
                    train_OA = float(evaluator_train.Pixel_Accuracy())

            val_ce, val_total, reg_val, OA, mIoU, Kappa, mAcc, IOU, Acc, it = measure_inference(
                net, x, val_label, evaluator_val, loss_func, device,
                include_reg=val_with_reg, lambda_reg=lambda_reg
            )

            # record memory
            if torch.cuda.is_available():
                peak_mem_MB = torch.cuda.max_memory_allocated() / (1024 ** 2)
            else:
                peak_mem_MB = 0.0

            gap_ce = float(val_ce - float(cls_loss.detach().cpu().item()))
            train_total = float(loss.detach().cpu().item())
            gap_total = float(val_total - train_total)

            lr_now = optimizer.param_groups[0]['lr']

            # optional: test each epoch
            test_cols = []
            if test_each_epoch and ((epoch + 1) % max(1, test_every) == 0):
                evaluator_test = Evaluator(num_class=class_count)
                test_ce, test_total, test_reg, test_OA, test_mIoU, test_Kappa, test_MACC, _, _, test_it = measure_inference(
                    net, x, test_label, evaluator_test, loss_func, device,
                    include_reg=val_with_reg, lambda_reg=lambda_reg
                )
                test_cols = [test_ce, test_total, test_reg, test_OA, test_mIoU, test_Kappa, test_MACC, test_it]

            # write CSV
            row = [
                epoch, curr_seed, lr_now,
                float(loss.detach().cpu().item()),
                float(cls_loss.detach().cpu().item()),
                float((lambda_reg * reg_train).detach().cpu().item()),
                float(train_total),
                float(val_ce), float(val_total), float(reg_val),
                float(OA), float(mIoU), float(Kappa), float(mAcc),
                float(gap_ce), float(gap_total),
                float(gn), float(it), int(n_params), str(flops), float(train_ratio),
                float(lambda_reg), float(weight_decay), float(dropout_p),
                float(label_smoothing), int(use_aug), round(float(peak_mem_MB), 2)
            ]
            if log_train_oa:
                gap_OA = float(train_OA - OA) if train_OA is not None else None
                row += [train_OA, gap_OA]
            if test_cols:
                row += test_cols

            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

            # logging
            logger.info(
                f"Epoch {epoch} | LR {lr_now:.5e} | "
                f"train(total/cls/reg) {train_total:.5f}/{float(cls_loss):.5f}/{float(lambda_reg * reg_train):.5f} | "
                f"val(ce/total/OA/mAcc/mIoU/Kappa) {val_ce:.5f}/{val_total:.5f}/{OA:.4f}/{mAcc:.4f}/{mIoU:.4f}/{Kappa:.4f} | "
                f"gap_ce {gap_ce:.5f} gap_total {gap_total:.5f} | grad_norm {gn:.3f} | mem {peak_mem_MB:.1f} MB | IT {it:.4f}s"
            )

            # best checkpoint by Val OA
            if OA > best_metric + 1e-7:
                best_metric = OA
                best_epoch = epoch
                best_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
                torch.save(net.state_dict(), save_weight_path)
                bad = 0
            else:
                bad += 1

            # optional early stop (kept as your original style, commented block could be enabled by config)
            # if (not no_early_stop) and bad >= max(1, early_stop_patience):
            #     logger.info(f"[EarlyStop] epoch={epoch}, best_val_OA={best_metric:.4f} @ {best_epoch}")
            #     if best_state is not None:
            #         net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            #     break

            scheduler.step()

            if (epoch + 1) % 50 == 0:
                net.eval()
                with torch.no_grad():
                    out_val = net(x)
                    yv = val_label.unsqueeze(0)
                    seg_logits = resize(input=out_val, size=yv.shape[1:], mode='bilinear', align_corners=True)
                    pred_val = torch.argmax(seg_logits, dim=1).cpu().numpy()
                vis_a_image(gt, pred_val, os.path.join(vis_dir, f'predict_{epoch+1}.png'),
                            os.path.join(vis_dir, 'gt.png'))

        if best_state is not None:
            net.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        # ---- Final TEST on best ----
        evaluator_test_final = Evaluator(num_class=class_count)
        test_ce, test_total, test_reg, OA_test, mIOU_test, Kappa_test, mAcc_test, IOU_test, Acc_test, _ = measure_inference(
            net, x, test_label, evaluator_test_final, loss_func, device,
            include_reg=val_with_reg, lambda_reg=lambda_reg
        )

        net.eval()
        with torch.no_grad():
            out_test = net(x)
            yt = test_label.unsqueeze(0)
            seg_logits_t = resize(input=out_test, size=yt.shape[1:], mode='bilinear', align_corners=True)
            pred_test = torch.argmax(seg_logits_t, dim=1).cpu().numpy()
        vis_a_image(gt, pred_test, predict_save_path, gt_save_path)

        np.save(os.path.join(single_dir, 'confmat_test.npy'), evaluator_test_final.confusion_matrix)
        np.save(os.path.join(single_dir, 'class_acc_test.npy'), Acc_test)

        with open(results_save_path, 'a+', encoding="utf-8") as f:
            f.write('\n======================' +
                    f" exp_idx={exp_idx} seed={curr_seed} lr={lr} epochs={max_epoch} train ratio={train_ratio} " +
                    "======================\n" +
                    f"OA={OA_test}\nAA={mAcc_test}\nkpp={Kappa_test}\n" +
                    f"mIOU_test:{mIOU_test}\nIOU_test:{IOU_test}\nAcc_test:{Acc_test}\n" +
                    f"test_ce={test_ce} test_total={test_total} test_reg={test_reg}\n")

        logger.info(f"[FINAL TEST] seed={curr_seed} | OA={OA_test:.4f} | mIoU={mIOU_test:.4f} | "
                    f"Kappa={Kappa_test:.4f} | mAcc={mAcc_test:.4f} "
                    f"| test_ce {test_ce:.5f} test_total {test_total:.5f} test_reg {test_reg:.5f}")

        OA_ALL.append(OA_test); AA_ALL.append(mAcc_test); KPP_ALL.append(Kappa_test); EACH_ACC_ALL.append(Acc_test)
        torch.cuda.empty_cache()

    # ---- Multi-seed summary ----
    OA_ALL = np.array(OA_ALL); AA_ALL = np.array(AA_ALL); KPP_ALL = np.array(KPP_ALL); EACH_ACC_ALL = np.array(EACH_ACC_ALL)
    mean_result_path = os.path.join(save_folder, 'mean_result.txt')
    with open(mean_result_path, 'a', encoding="utf-8") as f:
        f.write('\n\n***************Mean result of ' + str(len(seed_list)) + ' runs ********************\n' +
                'List of OA:' + str(list(OA_ALL)) + '\n' +
                'List of AA:' + str(list(AA_ALL)) + '\n' +
                'List of KPP:' + str(list(KPP_ALL)) + '\n' +
                'OA=' + str(round(np.mean(OA_ALL) * 100, 2)) + '+-' + str(round(np.std(OA_ALL) * 100, 2)) + '\n' +
                'AA=' + str(round(np.mean(AA_ALL) * 100, 2)) + '+-' + str(round(np.std(AA_ALL) * 100, 2)) + '\n' +
                'Kpp=' + str(round(np.mean(KPP_ALL) * 100, 2)) + '+-' + str(round(np.std(KPP_ALL) * 100, 2)) + '\n' +
                'Acc per class=\n' + str(np.round(np.mean(EACH_ACC_ALL, 0) * 100, 2)) + '+-' +
                str(np.round(np.std(EACH_ACC_ALL, 0) * 100, 2)) + '\n')

    # save json config snapshot (for your previous habit)
    with open(os.path.join(save_folder, 'run_config.json'), 'w', encoding="utf-8") as fp:
        json.dump(paras, fp, indent=2, ensure_ascii=False)

    print("Done.")



if __name__ == "__main__":
    main()

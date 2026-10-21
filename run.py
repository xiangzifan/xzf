import os
import sys
import time
import random
import logging
import argparse

import numpy as np
import torch

from utils import save_config, load_config
from evaluation import evalrank
from co_train import main


def run():

    # current_time
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    # Hyper Parameters
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

    parser.add_argument(
        "--data_path",
        default=os.environ.get("IN2R_DATA_PATH", "/202201052/IN2R_xzf/data"),
        help="path to datasets",
    )
    parser.add_argument(
        "--data_name", default="f30k_precomp", help="{coco,f30k,cc152k}_precomp"
    )
    parser.add_argument(
        "--vocab_path",
        default=os.environ.get("IN2R_VOCAB_PATH", "/202201052/IN2R_xzf/vocab"),
        help="Path to saved vocabulary json files.",
    )

    # ----------------------- training setting ----------------------#
    parser.add_argument(
        "--batch_size", default=128, type=int, help="Size of a training mini-batch."
    )
    parser.add_argument(
        "--num_epochs", default=40, type=int, help="Number of training epochs."
    )
    parser.add_argument(
        "--lr_update",
        default=15,
        type=int,
        help="Number of epochs to update the learning rate (step schedule only).",
    )
    parser.add_argument(
        "--lr_scheduler",
        default="cosine",
        choices=["step", "cosine"],
        help="LR schedule for co-training: step (original code) or cosine (paper).",
    )
    parser.add_argument(
        "--lambda_intra",
        default=0.5,
        type=float,
        help="Weight for intra-modal clean-pair losses (paper L_clean = L_inter + lambda_intra * (L_img + L_txt)).",
    )
    parser.add_argument(
        "--learning_rate", default=0.0005, type=float, help="Initial learning rate."
    )
    parser.add_argument(
        "--workers", default=0, type=int, help="Number of data loader workers."
    )
    parser.add_argument(
        "--log_step",
        default=500,
        type=int,
        help="Number of steps to print and record the log.",
    )
    parser.add_argument(
        "--grad_clip", default=2.0, type=float, help="Gradient clipping threshold."
    )
    parser.add_argument("--margin", default=0.2, type=float, help="Rank loss margin.")

    # ------------------------- model setting -----------------------#
    parser.add_argument(
        "--img_dim",
        default=2048,
        type=int,
        help="Dimensionality of the image embedding.",
    )
    parser.add_argument(
        "--dropout_rate",
        default=0.3,
        type=float,
        help="dropout rate",
    )
    parser.add_argument(
        "--word_dim",
        default=300,
        type=int,
        help="Dimensionality of the word embedding.",
    )
    parser.add_argument(
        "--embed_size",
        default=1024,
        type=int,
        help="Dimensionality of the joint embedding.",
    )
    parser.add_argument(
        "--num_layers", default=1, type=int, help="Number of GRU layers."
    )
    parser.add_argument("--bi_gru", action="store_false", help="Use bidirectional GRU.")
    parser.add_argument(
        "--no_imgnorm",
        action="store_true",
        help="Do not normalize the image embeddings.",
    )
    parser.add_argument(
        "--no_txtnorm",
        action="store_true",
        help="Do not normalize the text embeddings.",
    )

    # noise settings
    parser.add_argument("--noise_file", default="", help="noise_file")
    parser.add_argument("--noise_ratio", default=0.2, type=float, help="Noisy ratio")

    # NCR Settings
    parser.add_argument(
        "--no_co_training", action="store_true", help="No co-training for noisy label."
    )
    parser.add_argument("--warmup_epoch", default=5, type=int, help="warm up epochs")
    parser.add_argument("--warmup_model_path", default="", help="warm up models")
    parser.add_argument(
        "--p_threshold", default=0.5, type=float, help="clean probability threshold"
    )
    parser.add_argument(
        "--soft_margin", default="exponential", help="linear|exponential|sin"
    )

    # Runing Settings
    parser.add_argument("--gpu", default="0", help="Which gpu to use.")
    parser.add_argument(
        "--seed", default=42, type=int, help="Random seed."
    )
    parser.add_argument(
        "--output_dir", default=os.path.join("output", current_time), help="Output dir."
    )

    parser.add_argument(
        "--queue_size", default=65536, type=int, help="Memory queue size M."
    )
    parser.add_argument(
        "--k", default=5, type=int, help="Number of intra-modal neighbors K."
    )
    parser.add_argument(
        "--memory_mode",
        default="quality_diverse",
        choices=["fifo", "id_refresh", "quality_diverse"],
        help=(
            "Memory ablation: original A5 FIFO, ID-aware refresh, or full "
            "reliability/freshness/diversity support selection."
        ),
    )
    parser.add_argument(
        "--memory_candidate_pool_size",
        default=20,
        type=int,
        help="Raw-similarity candidate pool P used by quality_diverse mode.",
    )
    parser.add_argument(
        "--memory_quality_weight",
        default=0.25,
        type=float,
        help="Weight of standardized peer clean probability in reranking.",
    )
    parser.add_argument(
        "--memory_freshness_weight",
        default=0.10,
        type=float,
        help="Weight of standardized feature freshness in reranking.",
    )
    parser.add_argument(
        "--memory_freshness_tau",
        default=2.0,
        type=float,
        help="Exponential freshness time constant measured in epochs.",
    )
    parser.add_argument(
        "--memory_mmr_balance",
        default=0.80,
        type=float,
        help="MMR balance in [0,1]: larger values favor relevance/quality.",
    )
    parser.add_argument(
        "--memory_alpha_min",
        default=0.20,
        type=float,
        help="Minimum current-feature weight during an ID refresh.",
    )
    parser.add_argument(
        "--memory_alpha_max",
        default=0.80,
        type=float,
        help="Maximum current-feature weight during an ID refresh.",
    )
    parser.add_argument(
        "--memory_quality_momentum",
        default=0.50,
        type=float,
        help="Old-quality weight used when refreshing an existing record.",
    )
    parser.add_argument(
        "--i2t", dest="i2t", action="store_true", default=True,
        help="Enable image-to-text rectification (default: on)."
    )
    parser.add_argument(
        "--no_i2t", dest="i2t", action="store_false",
        help="Disable image-to-text rectification."
    )
    parser.add_argument(
        "--t2i", dest="t2i", action="store_true", default=True,
        help="Enable text-to-image rectification (default: on)."
    )
    parser.add_argument(
        "--no_t2i", dest="t2i", action="store_false",
        help="Disable text-to-image rectification."
    )

    # load arguments
    opt = parser.parse_args()

    if opt.k < 1 or opt.queue_size < max(opt.k, opt.batch_size):
        parser.error("--queue_size must hold at least K and one full batch")
    if opt.memory_candidate_pool_size < opt.k:
        parser.error("--memory_candidate_pool_size must be >= --k")
    if (
        not np.isfinite(opt.memory_quality_weight)
        or not np.isfinite(opt.memory_freshness_weight)
        or opt.memory_quality_weight < 0
        or opt.memory_freshness_weight < 0
    ):
        parser.error("memory reranking weights must be finite and nonnegative")
    if not np.isfinite(opt.memory_freshness_tau) or opt.memory_freshness_tau <= 0:
        parser.error("--memory_freshness_tau must be finite and positive")
    if not np.isfinite(opt.memory_mmr_balance) or not 0 <= opt.memory_mmr_balance <= 1:
        parser.error("--memory_mmr_balance must be in [0, 1]")
    if (
        not np.isfinite(opt.memory_alpha_min)
        or not np.isfinite(opt.memory_alpha_max)
        or not 0 <= opt.memory_alpha_min <= opt.memory_alpha_max <= 1
    ):
        parser.error("require 0 <= memory_alpha_min <= memory_alpha_max <= 1")
    if (
        not np.isfinite(opt.memory_quality_momentum)
        or not 0 <= opt.memory_quality_momentum <= 1
    ):
        parser.error("--memory_quality_momentum must be in [0, 1]")

    # Output dir
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    if not opt.noise_file:
        # 优先使用固定的 20% 噪声文件，保证与历史实验同一份噪声索引
        if opt.noise_ratio == 0.2 and os.environ.get("IN2R_NOISE_02"):
            opt.noise_file = os.environ["IN2R_NOISE_02"]
        else:
            opt.noise_file = os.path.join(
                opt.output_dir, opt.data_name + "_" + str(opt.noise_ratio) + ".npy"
            )

    if opt.data_name == "cc152k_precomp":
        opt.noise_ratio = 0
        opt.noise_file = ""

    print("\n*-------- Experiment Config --------*")
    print(opt)

    # CUDA env
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    # set random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.random.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = True

    # save config
    save_config(opt, os.path.join(opt.output_dir, "config.json"))

    # traing and evaluation
    print("\n*-------- Training --------*")
    main(opt)

    print("\n*-------- Testing --------*")
    if opt.data_name == "coco_precomp":
        print("5 fold validation")
        evalrank(
            os.path.join(opt.output_dir, "model_best.pth.tar"),
            split="testall",
            fold5=True,
        )
        print("full validation")
        evalrank(os.path.join(opt.output_dir, "model_best.pth.tar"), split="testall")
    else:
        evalrank(os.path.join(opt.output_dir, "model_best.pth.tar"), split="test")


if __name__ == "__main__":
    run()

"""Training script"""

import os
import time
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.mixture import GaussianMixture

from data import get_loader, get_dataset
from model import SGRAF
from vse import VSE, infonce
from vocab import Vocabulary, deserialize_vocab
from evaluation import i2t, t2i, encode_data, shard_attn_scores
from memory_bank import MemoryQueue
import logging
from torch.nn.utils.clip_grad import clip_grad_norm_

# 创建 logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # 设置全局日志级别

from utils import (
    AverageMeter,
    ProgressMeter,
    save_checkpoint,
    adjust_learning_rate,
)


_VALID_REFINER_MODES = {"mean", "dua_pool", "qdmp_mean", "tugr_full"}


class GraphRefiner(nn.Module):
    """Task-utility-guided support graph refinement.

    Modes deliberately form a 2x2 factorial design:

    ``mean``
        Original support self-attention followed by equal mean pooling.
    ``dua_pool``
        Original messages followed by discriminative-utility pooling.
    ``qdmp_mean``
        Query-conditioned discriminative message passing followed by mean.
    ``tugr_full``
        QDMP messages and DUA pooling (the proposed full model).

    The utility is derived only from train-visible query/support embeddings.  It
    never consumes synthetic clean/noisy ground-truth labels.  By default it is
    detached before it changes messages or pooling, which prevents the encoder
    from manipulating the routing score as a shortcut while preserving normal
    gradients through the refined support representations.
    """

    def __init__(
        self,
        dim=1024,
        k=5,
        n_heads=4,
        dropout=0.1,
        mode="tugr_full",
        utility_temperature=0.1,
        pool_temperature=0.1,
        message_prior_strength=0.5,
        detach_utility=True,
        eps=1e-6,
    ):
        super(GraphRefiner, self).__init__()
        if dim < 1 or k < 1 or n_heads < 1:
            raise ValueError("dim, k, and n_heads must be positive")
        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if mode not in _VALID_REFINER_MODES:
            raise ValueError("unsupported refiner mode: {}".format(mode))
        if utility_temperature <= 0 or pool_temperature <= 0 or eps <= 0:
            raise ValueError("temperatures and eps must be positive")
        if message_prior_strength < 0:
            raise ValueError("message_prior_strength must be nonnegative")

        self.dim = int(dim)
        self.k = int(k)
        self.n_heads = int(n_heads)
        self.d_k = self.dim // self.n_heads
        self.mode = mode
        self.utility_temperature = float(utility_temperature)
        self.pool_temperature = float(pool_temperature)
        self.message_prior_strength = float(message_prior_strength)
        self.detach_utility = bool(detach_utility)
        self.eps = float(eps)

        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)

        # Retained, but frozen, for state/init compatibility with A5-Delete.
        # The released A5 path declared edge_nn but never used it in forward.
        self.edge_nn = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.dim // 2, 1),
            nn.Sigmoid(),
        )
        self.edge_nn.requires_grad_(False)

        self.output_layer = nn.Linear(self.dim, self.dim)
        self.layer_norm = nn.LayerNorm(self.dim)
        self.dropout = nn.Dropout(dropout)
        self._diagnostic_sums = defaultdict(float)
        self._diagnostic_counts = defaultdict(int)

    @property
    def uses_utility(self):
        return self.mode != "mean"

    @property
    def uses_qdmp(self):
        return self.mode in {"qdmp_mean", "tugr_full"}

    @property
    def uses_dua_pool(self):
        return self.mode in {"dua_pool", "tugr_full"}

    def reset_diagnostics(self):
        self._diagnostic_sums.clear()
        self._diagnostic_counts.clear()

    def _record_scalar(self, name, value, weight=1):
        self._diagnostic_sums[name] += float(value.detach().item()) * int(weight)
        self._diagnostic_counts[name] += int(weight)

    def diagnostics(self):
        result = {
            "mode": self.mode,
            "calls": int(self._diagnostic_counts.get("calls", 0)),
            "queries": int(self._diagnostic_counts.get("queries", 0)),
        }
        for name, total in self._diagnostic_sums.items():
            count = self._diagnostic_counts.get(name, 0)
            if count:
                result[name] = total / count
        return result

    def _validate_inputs(self, neighbors, query_feats):
        if neighbors.ndim != 3:
            raise ValueError("neighbors must have shape [batch, k, dim]")
        batch_size, support_count, dim = neighbors.shape
        if support_count != self.k:
            raise ValueError(
                "expected {} supports, received {}".format(self.k, support_count)
            )
        if dim != self.dim:
            raise ValueError(
                "expected feature dimension {}, received {}".format(self.dim, dim)
            )
        if not torch.isfinite(neighbors).all():
            raise ValueError("neighbors must be finite")
        if self.uses_utility:
            if query_feats is None:
                raise ValueError("query_feats are required by mode {}".format(self.mode))
            if query_feats.shape != (batch_size, self.dim):
                raise ValueError("query_feats must have shape [batch, dim]")
            if query_feats.device != neighbors.device:
                raise ValueError("query_feats and neighbors must share a device")
            if query_feats.dtype != neighbors.dtype:
                raise ValueError("query_feats and neighbors must share a dtype")
            if not torch.isfinite(query_feats).all():
                raise ValueError("query_feats must be finite")

    def compute_discriminative_utility(self, query_feats, neighbors):
        """Return DUA utility with shape [batch, k].

        For support ``j`` retrieved for query ``i`` the positive term is its
        cosine similarity to query ``i``.  The negative term is a temperature-
        scaled LogMeanExp over all other queries in the current mini-batch.
        Batch size one has no negatives and therefore falls back to the positive
        cosine term instead of producing an undefined value.
        """
        normalized_queries = F.normalize(
            query_feats, p=2, dim=-1, eps=self.eps
        )
        normalized_supports = F.normalize(
            neighbors, p=2, dim=-1, eps=self.eps
        )
        batch_size = query_feats.size(0)
        # One GEMM is substantially faster than a Python loop/einsum for the
        # [source_query, owner_query, support] similarity tensor.
        all_similarities = torch.matmul(
            normalized_queries,
            normalized_supports.reshape(batch_size * self.k, self.dim).t(),
        ).view(batch_size, batch_size, self.k)
        owner = torch.arange(batch_size, device=query_feats.device)
        positive = all_similarities[owner, owner]
        if batch_size == 1:
            utility = positive
        else:
            negative_mask = ~torch.eye(
                batch_size, dtype=torch.bool, device=query_feats.device
            )
            negative_logits = all_similarities.masked_fill(
                ~negative_mask[:, :, None], -torch.inf
            )
            negative_lme = self.utility_temperature * (
                torch.logsumexp(
                    negative_logits / self.utility_temperature, dim=0
                )
                - np.log(batch_size - 1)
            )
            utility = positive - negative_lme
        if self.detach_utility:
            utility = utility.detach()
        return utility

    def _standardize_utility(self, utility):
        centered = utility - utility.mean(dim=-1, keepdim=True)
        variance = centered.square().mean(dim=-1, keepdim=True)
        return centered / variance.add(self.eps).sqrt()

    def _record_diagnostics(self, utility, pool_weights, message_weights):
        batch_size = pool_weights.size(0)
        tiny = torch.finfo(pool_weights.dtype).tiny
        pool_entropy = -(
            pool_weights * pool_weights.clamp_min(tiny).log()
        ).sum(dim=-1)
        effective_k = pool_weights.square().sum(dim=-1).clamp_min(
            self.eps
        ).reciprocal()
        pool_sum_error = (pool_weights.sum(dim=-1) - 1.0).abs()

        self._diagnostic_counts["calls"] += 1
        self._diagnostic_counts["queries"] += batch_size
        self._record_scalar(
            "pool_top1", pool_weights.max(dim=-1).values.mean(), batch_size
        )
        self._record_scalar("pool_entropy", pool_entropy.mean(), batch_size)
        self._record_scalar("effective_k", effective_k.mean(), batch_size)
        self._record_scalar("pool_sum_error", pool_sum_error.max())
        if utility is not None:
            self._record_scalar("utility_mean", utility.mean(), batch_size)
            self._record_scalar(
                "utility_std", utility.std(unbiased=False), batch_size
            )
        if message_weights is not None:
            message_tiny = torch.finfo(message_weights.dtype).tiny
            message_entropy = -(
                message_weights
                * message_weights.clamp_min(message_tiny).log()
            ).sum(dim=-1)
            self._record_scalar(
                "message_entropy", message_entropy.mean(), batch_size
            )

    def forward(self, neighbors, query_feats=None):
        """Refine a retrieved support set into one normalized prototype."""
        self._validate_inputs(neighbors, query_feats)
        batch_size, support_count, dim = neighbors.size()

        utility = None
        if self.uses_utility:
            utility = self.compute_discriminative_utility(query_feats, neighbors)

        projected_queries = self.q_proj(neighbors).view(
            batch_size, support_count, self.n_heads, self.d_k
        ).transpose(1, 2)
        projected_keys = self.k_proj(neighbors).view(
            batch_size, support_count, self.n_heads, self.d_k
        ).transpose(1, 2)
        projected_values = self.v_proj(neighbors).view(
            batch_size, support_count, self.n_heads, self.d_k
        ).transpose(1, 2)

        attention_logits = torch.matmul(
            projected_queries, projected_keys.transpose(-2, -1)
        ) / (self.d_k ** 0.5)
        if self.uses_qdmp:
            standardized_utility = self._standardize_utility(utility)
            attention_logits = attention_logits + (
                self.message_prior_strength
                * standardized_utility[:, None, None, :]
            )

        message_weights = F.softmax(attention_logits, dim=-1)
        context = torch.matmul(message_weights, projected_values)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, support_count, dim
        )
        refined_nodes = self.layer_norm(
            neighbors + self.dropout(self.output_layer(context))
        )

        if self.uses_dua_pool:
            pool_weights = F.softmax(
                utility / self.pool_temperature, dim=-1
            )
            refined_feat = (
                pool_weights.unsqueeze(-1) * refined_nodes
            ).sum(dim=1)
        else:
            pool_weights = torch.full(
                (batch_size, support_count),
                1.0 / support_count,
                dtype=neighbors.dtype,
                device=neighbors.device,
            )
            refined_feat = refined_nodes.mean(dim=1)

        self._record_diagnostics(utility, pool_weights, message_weights)
        return F.normalize(refined_feat, p=2, dim=-1, eps=self.eps)


def build_graph_refiner(opt, device):
    return GraphRefiner(
        dim=opt.embed_size,
        k=opt.k,
        n_heads=opt.refiner_heads,
        dropout=opt.refiner_dropout,
        mode=opt.refiner_mode,
        utility_temperature=opt.refiner_utility_temperature,
        pool_temperature=opt.refiner_pool_temperature,
        message_prior_strength=opt.refiner_message_prior_strength,
        detach_utility=not opt.refiner_end_to_end_utility,
        eps=opt.refiner_eps,
    ).to(device)


def build_memory_queue(opt, device=None):
    """Construct the selected Memory ablation on an explicit runtime device."""
    return MemoryQueue(
        queue_size=opt.queue_size,
        dim=opt.embed_size,
        mode=opt.memory_mode,
        candidate_pool_size=opt.memory_candidate_pool_size,
        quality_weight=opt.memory_quality_weight,
        freshness_weight=opt.memory_freshness_weight,
        freshness_tau=opt.memory_freshness_tau,
        mmr_balance=opt.memory_mmr_balance,
        alpha_min=opt.memory_alpha_min,
        alpha_max=opt.memory_alpha_max,
        quality_momentum=opt.memory_quality_momentum,
        device=device,
    )


def _model_device(net):
    return next(net.img_enc.parameters()).device


def _trainable_parameters(module):
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def _optimizer_parameters(optimizer):
    """Return every unique parameter owned by an optimizer."""
    parameters = []
    seen = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            identity = id(parameter)
            if identity not in seen:
                seen.add(identity)
                parameters.append(parameter)
    return parameters


def ensure_rectification_modules(opt, net):
    """Create Memory and both refiners exactly once and register optimizer groups."""
    device = _model_device(net)
    if not hasattr(net, "memory_queue"):
        net.memory_queue = build_memory_queue(opt, device=device)
    if net.memory_queue.device != device:
        raise RuntimeError("memory queue and encoder must share a device")

    for attribute in ("i2t_refiner_node", "t2i_refiner_node"):
        if hasattr(net, attribute):
            continue
        refiner = build_graph_refiner(opt, device)
        setattr(net, attribute, refiner)
        parameters = _trainable_parameters(refiner)
        if parameters:
            net.optimizer.add_param_group(
                {"params": parameters, "lr": opt.learning_rate}
            )


def rectification_state_dict(net):
    """Serialize state that VSE.state_dict() does not include."""
    required = ("memory_queue", "i2t_refiner_node", "t2i_refiner_node")
    missing = [name for name in required if not hasattr(net, name)]
    if missing:
        raise RuntimeError("missing rectification modules: {}".format(missing))
    return {
        "schema": 1,
        "memory_queue": net.memory_queue.state_dict(),
        "i2t_refiner": net.i2t_refiner_node.state_dict(),
        "t2i_refiner": net.t2i_refiner_node.state_dict(),
    }


def load_rectification_state_dict(net, state):
    if state.get("schema") != 1:
        raise ValueError("unsupported rectification-state schema")
    net.memory_queue.load_state_dict(state["memory_queue"])
    net.i2t_refiner_node.load_state_dict(state["i2t_refiner"], strict=True)
    net.t2i_refiner_node.load_state_dict(state["t2i_refiner"], strict=True)


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _validate_resume_config(opt, saved_opt):
    critical = (
        "data_name",
        "noise_ratio",
        "seed",
        "batch_size",
        "num_epochs",
        "learning_rate",
        "lr_scheduler",
        "lambda_intra",
        "margin",
        "embed_size",
        "k",
        "queue_size",
        "memory_mode",
        "memory_candidate_pool_size",
        "memory_quality_weight",
        "memory_freshness_weight",
        "memory_freshness_tau",
        "memory_mmr_balance",
        "memory_alpha_min",
        "memory_alpha_max",
        "memory_quality_momentum",
        "refiner_mode",
        "refiner_heads",
        "refiner_dropout",
        "refiner_utility_temperature",
        "refiner_pool_temperature",
        "refiner_message_prior_strength",
        "refiner_end_to_end_utility",
        "refiner_eps",
        "i2t",
        "t2i",
    )
    mismatches = []
    for name in critical:
        current = getattr(opt, name, None)
        previous = getattr(saved_opt, name, None)
        if current != previous:
            mismatches.append("{}: current={!r}, saved={!r}".format(
                name, current, previous
            ))
    if mismatches:
        raise ValueError(
            "resume configuration mismatch:\n  " + "\n  ".join(mismatches)
        )


def main(opt):

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 创建文件处理器
    file_handler = logging.FileHandler(f'{opt.output_dir}/log.txt', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 将处理器添加到 logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # load Vocabulary Wrapper
    logger.info("load and process dataset ...")
    vocab = deserialize_vocab(
        os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)
    )
    vocab.add_word('<mask>')  # add the mask, for testing cloze

    word2idx = vocab.word2idx

    logger.info('Add <mask> token into the vocab')
    opt.vocab_size = len(vocab)

    # load dataset
    captions_train, images_train = get_dataset(
        opt.data_path, opt.data_name, "train"
    )
    captions_dev, images_dev = get_dataset(opt.data_path, opt.data_name, "dev")

    # data loader
    noisy_trainloader, data_size, clean_labels = get_loader(
        captions_train,
        images_train,
        vocab,
        "warmup",
        opt.batch_size,
        opt.workers,
        opt.noise_ratio,
        opt.noise_file,
    )
    val_loader = get_loader(
        captions_dev, images_dev, vocab, "dev", opt.batch_size, opt.workers
    )

    # create models
    model_A = VSE(opt, word2idx)
    model_B = VSE(opt, word2idx)

    state_dict = model_A.state_dict()
    model_B.load_state_dict(state_dict)

    best_rsum = 0
    start_epoch = 0
    all_loss = [[], []]

    if opt.resume:
        if not os.path.isfile(opt.resume):
            raise FileNotFoundError("resume checkpoint not found: {}".format(opt.resume))
        checkpoint = torch.load(opt.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("schema") != "tugr_checkpoint_v1":
            raise ValueError(
                "--resume requires a strict TUGR checkpoint with runtime state"
            )
        _validate_resume_config(opt, checkpoint["opt"])
        model_A.load_state_dict(checkpoint["model_A"])
        model_B.load_state_dict(checkpoint["model_B"])
        ensure_rectification_modules(opt, model_A)
        ensure_rectification_modules(opt, model_B)
        load_rectification_state_dict(model_A, checkpoint["rectification_A"])
        load_rectification_state_dict(model_B, checkpoint["rectification_B"])
        model_A.optimizer.load_state_dict(checkpoint["optimizer_A"])
        model_B.optimizer.load_state_dict(checkpoint["optimizer_B"])
        best_rsum = float(checkpoint["best_rsum"])
        start_epoch = int(checkpoint["next_epoch"])
        all_loss = checkpoint["all_loss"]
        restore_rng_state(checkpoint["rng_state"])
        logger.info(
            "=> strictly resumed '{}' at co-training epoch {} (best rSum {:.4f})",
            opt.resume,
            start_epoch,
            best_rsum,
        )
    else:
        logger.info("\n* Warmup")
        if opt.warmup_model_path:
            if os.path.isfile(opt.warmup_model_path):
                checkpoint = torch.load(
                    opt.warmup_model_path, map_location="cpu", weights_only=False
                )
                model_A.load_state_dict(checkpoint["model_A"])
                model_B.load_state_dict(checkpoint["model_B"])
                logger.info(
                    "=> load warmup checkpoint '{}' (epoch {})".format(
                        opt.warmup_model_path, checkpoint["epoch"]
                    )
                )
                logger.info("\nValidation ...")
                validate(opt, val_loader, [model_A, model_B])
            else:
                raise FileNotFoundError(
                    "warmup checkpoint not found: {}".format(opt.warmup_model_path)
                )
        else:
            for epoch in range(0, opt.warmup_epoch):
                logger.info(
                    "[{}/{}] Warmup model_A".format(epoch + 1, opt.warmup_epoch)
                )
                warmup(opt, noisy_trainloader, model_A, epoch)
                logger.info(
                    "[{}/{}] Warmup model_B".format(epoch + 1, opt.warmup_epoch)
                )
                warmup(opt, noisy_trainloader, model_B, epoch)

                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_A": model_A.state_dict(),
                        "model_B": model_B.state_dict(),
                        "opt": opt,
                    },
                    is_best=False,
                    filename="warmup_model_{}.pth.tar".format(epoch),
                    prefix=opt.output_dir + "/",
                )
                logger.info("\nValidation ...")
                validate(opt, val_loader, [model_A, model_B])

        ensure_rectification_modules(opt, model_A)
        ensure_rectification_modules(opt, model_B)

    training_end_epoch = opt.co_train_epochs or opt.num_epochs
    if start_epoch >= training_end_epoch:
        raise ValueError(
            "resume next_epoch {} is not below training end epoch {}".format(
                start_epoch, training_end_epoch
            )
        )
    logger.info("\n* Co-training")

    # Train the Model
    for epoch in range(start_epoch, training_end_epoch):
        logger.info("\nEpoch [{}/{}]".format(epoch, opt.num_epochs))
        adjust_learning_rate(opt, model_A.optimizer, epoch)
        adjust_learning_rate(opt, model_B.optimizer, epoch)

        # # Dataset split (labeled, unlabeled)
        logger.info("Split dataset ...")
        prob_A, prob_B, all_loss = eval_train(
            opt,
            model_A,
            model_B,
            noisy_trainloader,
            data_size,
            all_loss,
            clean_labels,
            epoch,
        )

        pred_A = split_prob(prob_A, opt.p_threshold)
        pred_B = split_prob(prob_B, opt.p_threshold)

        clean_probs_A = prob_A[pred_A]

        if len(clean_probs_A) > 0:
            avg_clean_prob_A = clean_probs_A.mean()
        else:
            avg_clean_prob_A = 0.0

        num_above_avg = (prob_A > avg_clean_prob_A).sum()
        ratio_above_avg = (num_above_avg / len(prob_A)) * 100

        logger.info(f"Average probability of Predicted Clean samples: {avg_clean_prob_A:.4f}")
        logger.info(f"Number of samples with prob > avg: {num_above_avg} ({ratio_above_avg:.2f}%)")

        model_A.avg_clean_prob = avg_clean_prob_A

        clean_probs_B = prob_B[pred_B]

        if len(clean_probs_B) > 0:
            avg_clean_prob_B = clean_probs_B.mean()
        else:
            avg_clean_prob_B = 0.0

        num_above_avg = (prob_B > avg_clean_prob_B).sum()
        ratio_above_avg = (num_above_avg / len(prob_B)) * 100

        logger.info(f"Average probability of Predicted Clean samples: {avg_clean_prob_B:.4f}")
        logger.info(f"Number of samples with prob > avg: {num_above_avg} ({ratio_above_avg:.2f}%)")

        model_B.avg_clean_prob = avg_clean_prob_B

        logger.info("\nModel A training ...")
        # train model_A
        labeled_trainloader, unlabeled_trainloader = get_loader(
            captions_train,
            images_train,
            vocab,
            "train",
            opt.batch_size,
            opt.workers,
            opt.noise_ratio,
            opt.noise_file,
            pred=pred_B,
            prob=prob_B,
        )
        train(opt, model_A, model_B, labeled_trainloader, unlabeled_trainloader, epoch)

        logger.info("\nModel B training ...")
        # train model_B
        labeled_trainloader, unlabeled_trainloader = get_loader(
            captions_train,
            images_train,
            vocab, 
            "train",
            opt.batch_size,
            opt.workers,
            opt.noise_ratio,
            opt.noise_file,
            pred=pred_A,
            prob=prob_A,
        )
        train(opt, model_B, model_A, labeled_trainloader, unlabeled_trainloader, epoch)

        logger.info("\nValidattion ...")
        # evaluate on validation set
        rsum = validate(opt, val_loader, [model_A, model_B])

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        evaluation_checkpoint = {
            "schema": "tugr_eval_checkpoint_v1",
            "epoch": epoch,
            "model_A": model_A.state_dict(),
            "model_B": model_B.state_dict(),
            "i2t_refiner_A": model_A.i2t_refiner_node.state_dict(),
            "t2i_refiner_A": model_A.t2i_refiner_node.state_dict(),
            "i2t_refiner_B": model_B.i2t_refiner_node.state_dict(),
            "t2i_refiner_B": model_B.t2i_refiner_node.state_dict(),
            "best_rsum": best_rsum,
            "opt": opt,
        }
        save_checkpoint(
            evaluation_checkpoint,
            is_best,
            filename="checkpoint_{}.pth.tar".format(epoch),
            prefix=opt.output_dir + "/",
        )

        # Runtime Memory and Adam moments are large.  Keep one strict, complete
        # epoch-boundary snapshot instead of duplicating them in every numbered
        # evaluation checkpoint.
        resume_checkpoint = {
            key: value
            for key, value in evaluation_checkpoint.items()
            if key
            not in {
                "i2t_refiner_A",
                "t2i_refiner_A",
                "i2t_refiner_B",
                "t2i_refiner_B",
            }
        }
        resume_checkpoint.update(
            {
                "schema": "tugr_checkpoint_v1",
                "next_epoch": epoch + 1,
                "rectification_A": rectification_state_dict(model_A),
                "rectification_B": rectification_state_dict(model_B),
                "optimizer_A": model_A.optimizer.state_dict(),
                "optimizer_B": model_B.optimizer.state_dict(),
                "all_loss": all_loss,
                "rng_state": capture_rng_state(),
            }
        )
        save_checkpoint(
            resume_checkpoint,
            is_best=False,
            filename="resume_latest.pth.tar",
            prefix=opt.output_dir + "/",
        )


def train(opt, net, net2, labeled_trainloader, unlabeled_trainloader=None, epoch=None):
    """
    One epoch training.
    """
    losses_l = AverageMeter("Loss L", ":.4e")  
    losses_u = AverageMeter("Loss U", ":.4e")
    losses_u_i2t = AverageMeter("Loss U-i2t", ":.4e")
    losses_u_t2i = AverageMeter("Loss U-t2i", ":.4e")
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    
    max_len = max(len(labeled_trainloader), 0)

    progress = ProgressMeter(
        # len(labeled_trainloader),
        max_len, 
        [
            batch_time,
            data_time,
            losses_l,
            losses_u,
            losses_u_i2t,
            losses_u_t2i,
        ],
        prefix=f"Epoch[{epoch}/{opt.num_epochs}] Training Step",
    )

    ensure_rectification_modules(opt, net)
    ensure_rectification_modules(opt, net2)
    net.i2t_refiner_node.train()
    net.t2i_refiner_node.train()
    net.i2t_refiner_node.reset_diagnostics()
    net.t2i_refiner_node.reset_diagnostics()
    net2.i2t_refiner_node.eval()
    net2.t2i_refiner_node.eval()
    criterion = net.criterion
    
    # fix one network and train the other
    net.train_start()
    net2.val_start()

    unlabeled_train_iter = iter(unlabeled_trainloader)
    labeled_train_iter = iter(labeled_trainloader)
    labels_l = []
    pred_labels_l = []
    labels_u = []
    pred_labels_u = []
    end = time.time()
    for i in range(max_len):        
        try: 
            (
                batch_images_l,
                batch_img_len_l, 
                batch_text_l,
                batch_lengths_l,
                batch_ids_l,
                batch_labels_l,
                batch_prob_l,
                batch_clean_labels_l,
            ) = next(labeled_train_iter)
        except StopIteration:
            labeled_train_iter = iter(labeled_trainloader)
            (
                batch_images_l,
                batch_img_len_l, 
                batch_text_l,
                batch_lengths_l,
                batch_ids_l,
                batch_labels_l,
                batch_prob_l,
                batch_clean_labels_l,
            ) = next(labeled_train_iter)
        batch_size = batch_images_l.size(0)
        labels_l.append(batch_clean_labels_l)

        # unlabeled data
        try:
            (
                batch_images_u,
                batch_img_len_u,
                batch_text_u,
                batch_lengths_u,
                batch_ids_u,
                batch_clean_labels_u,
            # ) = unlabeled_train_iter.next()
            ) = next(unlabeled_train_iter)

        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (
                batch_images_u,
                batch_img_len_u,
                batch_text_u,
                batch_lengths_u,
                batch_ids_u,
                batch_clean_labels_u,
            # ) = unlabeled_train_iter.next()
            ) = next(unlabeled_train_iter)
        labels_u.append(batch_clean_labels_u)

        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            batch_prob_l = batch_prob_l.cuda()
            batch_labels_l = batch_labels_l.cuda()

        net.optimizer.zero_grad()

        img_emb_l, cap_emb_l, cap_len_l = net.forward_emb(batch_images_l, batch_text_l, batch_lengths_l, imgae_lengths=batch_img_len_l)

        # 计算标准 Contrastive Loss
        sims_l = net.forward_sim(img_emb_l, cap_emb_l, cap_len_l)
        loss_1 = criterion(sims_l, labels=None) # Standard loss

        img_emb_l2, cap_emb_l2, cap_len_l2 = net.forward_emb(batch_images_l, batch_text_l, batch_lengths_l, imgae_lengths=batch_img_len_l)
        img_sims = img_emb_l.mm(img_emb_l2.t())
        cap_sims = cap_emb_l.mm(cap_emb_l2.t())
        # A5: intra-modal margin 0.3 -> 0.2（唯一变量；inter margin 仍由 opt.margin 控制）
        loss_2 = criterion(img_sims, margin=0.2)
        loss_3 = criterion(cap_sims, margin=0.2)

        loss_l = loss_1 + opt.lambda_intra * (loss_2 + loss_3)

        if torch.isfinite(loss_l).item():
            loss_l.backward()
        else:
            logger.warning(
                f"Step {i}: labeled loss is not finite ({loss_l.item()}); skip its backward"
            )

        high_conf_idx = (batch_prob_l > net.avg_clean_prob).nonzero(as_tuple=True)[0]
        if high_conf_idx.numel() > 0:
            batch_ids_l_tensor = torch.as_tensor(
                batch_ids_l, dtype=torch.long, device=img_emb_l.device
            )
            net.memory_queue.enqueue(
                img_emb_l[high_conf_idx],
                cap_emb_l[high_conf_idx],
                record_ids=batch_ids_l_tensor[high_conf_idx],
                clean_prob=batch_prob_l[high_conf_idx],
                current_epoch=int(epoch or 0),
            )

        img_emb_u, cap_emb_u, cap_len_u = net.forward_emb(batch_images_u, batch_text_u, batch_lengths_u, imgae_lengths=batch_img_len_u)

        loss_u_i2t = torch.zeros((), device=img_emb_u.device)
        loss_u_t2i = torch.zeros((), device=img_emb_u.device)

        if opt.i2t:

            corrected_txt_emb = net2.memory_queue.retrieve_nearest_text(
                img_emb_u,
                k=opt.k,
                refiner_node=net.i2t_refiner_node,
                img_refiner_node=net.t2i_refiner_node,
                query_ids=batch_ids_u,
                current_epoch=int(epoch or 0),
            )
            # corrected_txt_emb = net2.memory_queue.retrieve_nearest_text(img_emb_u, k=opt.k, refiner_node=None)

            if corrected_txt_emb is not None:
                sims_u = img_emb_u.mm(corrected_txt_emb.t())
                loss_u_i2t = infonce(sims_u)

                correction_weight = 0.5
                loss_u_i2t = loss_u_i2t * correction_weight
                if torch.isfinite(loss_u_i2t).item():
                    loss_u_i2t.backward()
                else:
                    logger.warning(
                        f"Step {i}: i2t unlabeled loss is not finite ({loss_u_i2t.item()}); skip its backward"
                    )
                    loss_u_i2t = torch.zeros((), device=img_emb_u.device)
            
        if opt.t2i:
            corrected_img_emb = net2.memory_queue.retrieve_nearest_img(
                cap_emb_u,
                k=opt.k,
                refiner_node=net.t2i_refiner_node,
                txt_refiner_node=net.i2t_refiner_node,
                query_ids=batch_ids_u,
                current_epoch=int(epoch or 0),
            )
            # corrected_img_emb = net2.memory_queue.retrieve_nearest_img(cap_emb_u, k=opt.k, refiner_node=None)

            if corrected_img_emb is not None:
                sims_u = cap_emb_u.mm(corrected_img_emb.t())
                loss_u_t2i = infonce(sims_u)

                correction_weight = 0.5
                loss_u_t2i = loss_u_t2i * correction_weight
                if torch.isfinite(loss_u_t2i).item():
                    loss_u_t2i.backward()
                else:
                    logger.warning(
                        f"Step {i}: t2i unlabeled loss is not finite ({loss_u_t2i.item()}); skip its backward"
                    )
                    loss_u_t2i = torch.zeros((), device=img_emb_u.device)

        loss_u = loss_u_i2t.detach() + loss_u_t2i.detach()

        # 梯度有限性检查 + 梯度裁剪和更新
        # 注意：GraphRefiner 参数是通过 add_param_group 动态加入 optimizer 的，
        # 不在 net.params 中，必须遍历所有 param_groups 才能覆盖 refiner 梯度
        grads_finite = all(
            torch.isfinite(p.grad).all().item()
            for group in net.optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        )
        if grads_finite:
            if net.grad_clip > 0:
                clip_grad_norm_(
                    _optimizer_parameters(net.optimizer), net.grad_clip
                )
            net.optimizer.step()
        else:
            logger.warning(f"Step {i}: non-finite gradients detected; skipping optimizer step")
            net.optimizer.zero_grad()

        # loss = loss_l + loss_u
        losses_l.update(loss_l.item(), batch_images_l.size(0))

        losses_u.update(loss_u.item(), batch_images_u.size(0))
        losses_u_i2t.update(loss_u_i2t.detach().item(), batch_images_u.size(0))
        losses_u_t2i.update(loss_u_t2i.detach().item(), batch_images_u.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logger.info log info
        if i % opt.log_step == 0:
            progress.display(i)

    logger.info("Writer memory diagnostics: %s", net.memory_queue.diagnostics())
    logger.info("Peer memory diagnostics: %s", net2.memory_queue.diagnostics())
    logger.info(
        "Writer i2t GraphRefiner diagnostics: %s",
        net.i2t_refiner_node.diagnostics(),
    )
    logger.info(
        "Writer t2i GraphRefiner diagnostics: %s",
        net.t2i_refiner_node.diagnostics(),
    )


def warmup(opt, train_loader, model, epoch):
    # average meters to record the training statistics
    losses = AverageMeter("loss", ":.4e")
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(train_loader), [batch_time, data_time, losses], prefix="Warmup Step"
    )

    end = time.time()
    for i, (images, image_lengths, captions, lengths, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # drop last batch if only one sample (batch normalization require)
        if images.size(0) == 1:
            break

        model.train_start()

        # Update the model
        loss = model.train(images, captions, lengths, image_lengths=image_lengths, mode="warmup")
        losses.update(loss, images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.log_step == 0:
            progress.display(i)


def validate(opt, val_loader, models=[]):
    # compute the encoding for all the validation images and captions
    if opt.data_name == "cc152k_precomp":
        per_captions = 1
    elif opt.data_name in ["coco_precomp", "f30k_precomp"]:
        per_captions = 5

    Eiters = models[0].Eiters
    sims_mean = 0
    count = 0
    for ind in range(len(models)):
        count += 1
        logger.info("Encoding with model {}".format(ind))
        img_embs, cap_embs, cap_lens = encode_data(
            models[ind], val_loader, opt.log_step
        )
        logger.info(f"image shape:{img_embs.shape}, cap shape: {cap_embs.shape}")
        # clear duplicate 5*images and keep 1*images FIXME
        img_embs = np.array(
            [img_embs[i] for i in range(0, len(img_embs), per_captions)]
        )

        # record computation time of validation
        start = time.time()
        logger.info("Computing similarity from model {}".format(ind))
        sim = shard_attn_scores(
            models[ind], img_embs, cap_embs, cap_lens, opt, shard_size=100
        )
        end = time.time()
        logger.info(
            "Calculate similarity time with model {}: {:.2f} s".format(ind, end - start)
        )
        logger.info(
            f"Model_{ind}:"
        )
        # caption retrieval
        (r1, r5, r10, medr, meanr) = i2t(img_embs.shape[0], sim, per_captions)
        logger.info(
            "Image to text: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
                r1, r5, r10, medr, meanr
            )
        )

        # image retrieval
        (r1i, r5i, r10i, medri, meanr) = t2i(img_embs.shape[0], sim, per_captions)
        logger.info(
            "Text to image: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
                r1i, r5i, r10i, medri, meanr
            )
        )

        sims_mean += sim
    # average the sims
    sims_mean = sims_mean / count

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs.shape[0], sims_mean, per_captions)
    logger.info(
        "Image to text: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
            r1, r5, r10, medr, meanr
        )
    )

    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(img_embs.shape[0], sims_mean, per_captions)
    logger.info(
        "Text to image: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
            r1i, r5i, r10i, medri, meanr
        )
    )

    # sum of recalls to be used for early stopping
    r_sum = r1 + r5 + r10 + r1i + r5i + r10i
    logger.info(f"R_sum: {r_sum}")
    return r_sum


def eval_train(
    opt, model_A, model_B, data_loader, data_size, all_loss, clean_labels, epoch
):
    """
    Compute per-sample loss and prob
    """
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(data_loader), [batch_time, data_time], prefix="Computinng losses"
    )

    model_A.val_start()
    model_B.val_start()
    losses_A = torch.zeros(data_size)
    losses_B = torch.zeros(data_size)

    end = time.time()
    for i, (images, image_lengths, captions, lengths, ids) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        with torch.no_grad():
            # compute the loss
            loss_A = model_A.train(images, captions, lengths, image_lengths=image_lengths, mode="eval_loss")
            loss_B = model_B.train(images, captions, lengths, image_lengths=image_lengths, mode="eval_loss")
            for b in range(images.size(0)):
                losses_A[ids[b]] = loss_A[b]
                losses_B[ids[b]] = loss_B[b]

            batch_time.update(time.time() - end)
            end = time.time()
            if i % opt.log_step == 0:
                progress.display(i)

    losses_A = (losses_A - losses_A.min()) / (losses_A.max() - losses_A.min())
    all_loss[0].append(losses_A)
    losses_B = (losses_B - losses_B.min()) / (losses_B.max() - losses_B.min())
    all_loss[1].append(losses_B)

    input_loss_A = losses_A.reshape(-1, 1)
    input_loss_B = losses_B.reshape(-1, 1)

    logger.info("\nFitting GMM ...")
    # fit a two-component GMM to the loss
    gmm_A = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm_A.fit(input_loss_A.cpu().numpy())
    prob_A = gmm_A.predict_proba(input_loss_A.cpu().numpy())
    prob_A = prob_A[:, gmm_A.means_.argmin()]

    gmm_B = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm_B.fit(input_loss_B.cpu().numpy())
    prob_B = gmm_B.predict_proba(input_loss_B.cpu().numpy())
    prob_B = prob_B[:, gmm_B.means_.argmin()]

    return prob_A, prob_B, all_loss


def split_prob(prob, threshld):
    print(prob.min(), prob.max(), prob.mean())
    if prob.min() > threshld:
        # If prob are all larger than threshld, i.e. no noisy data, we enforce 1/100 unlabeled data
        logger.info(
            "No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled."
        )
        threshld = np.sort(prob)[len(prob) // 100]

    pred = prob > threshld
    return pred

#!/usr/bin/env/python3
"""Recipe for training a sequence-to-sequence ASR system with librispeech.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with beamsearch coupled with a neural
language model.

To run this recipe, do the following:
> python train.py hparams/train_BPE1000.yaml

With the default hyperparameters, the system employs a CRDNN encoder.
The decoder is based on a standard  GRU. Beamsearch coupled with a RNN
language model is used  on the top of decoder probabilities.

The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE)
are used as basic recognition tokens. Training is performed on the full
LibriSpeech dataset (960 h).

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE),
training split (e.g, train-clean 100 rather than the full one), and many
other possible variations.

This recipe assumes that the tokenizer and the LM are already trained.
To avoid token mismatches, the tokenizer used for the acoustic model is
the same use for the LM.  The recipe downloads the pre-trained tokenizer
and LM.

If you would like to train a full system from scratch do the following:
1- Train a tokenizer (see ../../Tokenizer)
2- Train a language model (see ../../LM)
3- Train the acoustic model (with this code).



Authors
 * Ju-Chieh Chou 2020
 * Mirco Ravanelli 2020
 * Abdel Heba 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020
 * Andreas Nautsch 2021
"""

import csv
import logging
import os
import sys
from enum import Enum, auto
from itertools import combinations

import numpy as np
import speechbrain as sb
import torch
import torch.nn as nn
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from recipes.CommonVoice.common_voice_prepare import prepare_common_voice
from speechbrain.dataio import audio_io
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.autocast import AMPConfig, TorchAutocast
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.edit_distance import wer_details_for_batch
from torch.utils.data import DataLoader
from tqdm.contrib import tqdm

from defense import Detector

logger = logging.getLogger(__name__)


class Stage(Enum):
    """Simple enum to track stage of experiments."""

    ATTACK = auto()


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # print(f"precision {hparams["precision"]}, eval precision {hparams["eval_precision"]}")
        # print(f"[wavs] dtype: {wavs.dtype}, autocast: {torch.is_autocast_enabled()}")
        # print(next(self.modules.wav2vec2.parameters()).dtype, next(self.modules.enc.parameters()).dtype, next(self.modules.ctc_lin.parameters()).dtype)

        # Downsample the inputs if specified
        if hasattr(self.modules, "downsampler"):
            wavs = self.modules.downsampler(wavs)

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
        # print(f"[wavs] dtype: {wavs.dtype}, autocast: {torch.is_autocast_enabled()}")
        # Forward pass

        # Handling SpeechBrain vs HuggingFace pretrained models
        if hasattr(self.modules, "extractor"):  # SpeechBrain pretrained model
            latents = self.modules.extractor(wavs)
            feats = self.modules.encoder_wrapper(latents, wav_lens=wav_lens)[
                "embeddings"
            ]
        else:  # HuggingFace pretrained model
            feats = self.modules.wav2vec2(wavs, wav_lens)
        # print(f"[feats] dtype: {feats.dtype}, autocast: {torch.is_autocast_enabled()}")
        ## x change precision and logits!!!!
        ## Autocast only changes the dtype of eligible ops, not tensors by themselves — and many speech front-end ops are explicitly kept in FP32
        ## enc is usually nn.Linear, nn.LSTM, or nn.Transformer
        ## These ops are on PyTorch’s autocast allowlist
        ## Autocast casts their outputs
        ## ctc_lin changes for the same reason, nn.Linear autocast
        x = self.modules.enc(feats)
        # print(f"[x] dtype: {x.dtype}, autocast: {torch.is_autocast_enabled()}")
        # Compute outputs
        p_tokens = None
        logits = self.modules.ctc_lin(x)
        # print(f"[logits] dtype: {logits.dtype}, autocast: {torch.is_autocast_enabled()}")
        # Upsample the inputs if they have been highly downsampled
        if hasattr(self.hparams, "upsampling") and self.hparams.upsampling:
            logits = logits.view(logits.shape[0], -1, self.hparams.output_neurons)

        p_ctc = self.hparams.log_softmax(logits)
        # print(f"[p_ctc] dtype: {p_ctc.dtype}, autocast: {torch.is_autocast_enabled()}")
        if stage == sb.Stage.VALID:
            p_tokens = sb.decoders.ctc_greedy_decode(
                p_ctc.detach(), wav_lens, blank_id=self.hparams.blank_index
            )

        elif stage == sb.Stage.TEST:
            p_tokens = test_searcher(p_ctc.detach(), wav_lens)

        return p_ctc, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        p_ctc, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens, tokens_lens = batch.tokens
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        # Labels must be extended if parallel augmentation or concatenated
        # augmentation was performed on the input (increasing the time dimension)
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            tokens = self.hparams.wav_augment.replicate_labels(tokens)
            tokens_lens = self.hparams.wav_augment.replicate_labels(tokens_lens)

        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        loss = loss_ctc

        if stage == sb.Stage.VALID:
            # Decode token terms to words
            predicted_words = self.tokenizer(predicted_tokens, task="decode_from_list")

        elif stage == sb.Stage.TEST:
            if hasattr(self.hparams, "rescorer"):
                predicted_words = [hyp[0].split(" ") for hyp in predicted_tokens]
            else:
                predicted_words = [hyp[0].text.split(" ") for hyp in predicted_tokens]

        if stage != sb.Stage.TRAIN and stage != Stage.ATTACK:
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer(target_words, task="decode_from_list")
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(self.model_optimizer, new_lr_model)
            if not self.hparams.wav2vec2.freeze:
                sb.nnet.schedulers.update_learning_rate(
                    self.wav2vec_optimizer, new_lr_wav2vec
                )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    "lr_wav2vec": old_lr_wav2vec,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]},
                min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.test_wer_file, "w", encoding="utf-8") as w:
                    self.wer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"

        # If the wav2vec encoder is unfrozen, we create the optimizer
        if not self.hparams.wav2vec2.freeze:
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.wav2vec2.parameters()
            )
            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("wav2vec_opt", self.wav2vec_optimizer)

        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)

        if not self.hparams.wav2vec2.freeze:
            self.optimizers_dict = {
                "wav2vec_optimizer": self.wav2vec_optimizer,
                "model_optimizer": self.model_optimizer,
            }
        else:
            self.optimizers_dict = {"model_optimizer": self.model_optimizer}

    def _initialize_vars(self):
        self.eps = 1  # 0.05
        self.max_iter_1 = 1000  # 4000  # 4000 # 10
        self.learning_rate_1 = 0.002  # 0.001
        self.global_max_length = 562480  # Need to check max length file!
        self.initial_rescale = 1.0
        self.decrease_factor_eps = 0.5  # 0.8
        self.num_iter_decrease_eps = (
            1  # 10  # In the code is 1, check it!, with one it checks every time
        )
        self.clip_min = None
        self.clip_max = None
        self.const = 10  # 1.0
        self.targeted = True
        self.optimizer = None
        self.alpha = 0.3
        self._optimizer_arg_1 = None

    ########## Initialize Detectors

    ## clean up later
    def _init_gaussian_detector_pure(self):
        """Load 3d Gaussian Detector with X=(Entropy Mean fp32, Entropy Mean fp16, Entropy Mean bf16) -> Mean, Covariance"""
        gaussian_path = getattr(
            self.hparams,
            "gaussian_detector_pure_path",
            "PLACEHOLDER",
        )

        if hasattr(self, "gaussian_mean_pure") and self.gaussian_mean_pure is not None:
            return

        if not os.path.exists(gaussian_path):
            raise FileNotFoundError(
                f"Gaussian detector not found at {gaussian_path}. "
                "Run the distriblock pipeline to fit the Gaussian first."
            )

        state = torch.load(gaussian_path, map_location=self.device, weights_only=False)
        self.gaussian_mean_pure = torch.tensor(
            state["gaussian_mean"], dtype=torch.float32, device=self.device
        )
        self.gaussian_cov_pure = torch.tensor(
            state["gaussian_cov"], dtype=torch.float32, device=self.device
        )
        self.gaussian_precision_pure = torch.inverse(self.gaussian_cov_pure)

        logger.info(
            f"Loaded 3D Gaussian detector: mean={self.gaussian_mean_pure.cpu().numpy()}, "
            f"cov={self.gaussian_cov_pure.cpu().numpy()}"
        )

    def _init_detector_separate_gaussian(self):
        """Load 2d Gaussian Detector with X=(PVP score, Mean Entropy) -> Mean, Covariance"""
        gaussian_path = getattr(
            self.hparams,
            "gaussian_detector_separate_path",
            "PLACEHOLDER",
        )

        if (
            hasattr(self, "gaussian_mean_separate")
            and self.gaussian_mean_separate is not None
        ):
            return

        if not os.path.exists(gaussian_path):
            raise FileNotFoundError(
                f"Separate Gaussian detector not found at {gaussian_path}. "
                "Run PVP_original.py with adv_type=hybrid_separate_gaussian first."
            )

        state = torch.load(gaussian_path, map_location=self.device, weights_only=False)
        self.gaussian_mean_separate = torch.tensor(
            state["gaussian_mean"], dtype=torch.float32, device=self.device
        )
        self.gaussian_cov_separate = torch.tensor(
            state["gaussian_cov"], dtype=torch.float32, device=self.device
        )
        self.gaussian_cov_separate += torch.eye(2, device=self.device) * 1e-6
        self.gaussian_precision_separate = torch.inverse(self.gaussian_cov_separate)
        logger.info("Loaded 2D Gaussian detector (separate)")

    def _init_detector_separate_nn(self):
        """Load nn with input X=(PVP score, Mean Entropy)"""
        detector_path = getattr(
            self.hparams,
            "nn_detector_separate_path",
            "PLACEHOLDER",
        )

        if (
            hasattr(self, "detector_separate_nn")
            and self.detector_separate_nn is not None
        ):
            return

        self.detector_separate_nn = Detector(input_dim=2, hidden_dim=8).to(self.device)
        self.detector_separate_nn.eval()
        for p in self.detector_separate_nn.parameters():
            p.requires_grad_(False)

        if os.path.exists(detector_path):
            state = torch.load(
                detector_path, map_location=self.device, weights_only=False
            )
            self.detector_separate_nn.load_state_dict(state["model"])
            self.detector_separate_threshold = state.get("threshold", 0.5)
            logger.info(
                f"Loaded NN detector (separate, input_dim=2) from {detector_path}"
            )
        else:
            logger.warning(
                f"NN detector (separate) not found at {detector_path}. Using untrained."
            )
            self.detector_separate_threshold = 0.5

    def _init_detector_pure_nn(self):
        """Load nn with input X=(Entropy Mean fp32, Entropy Mean fp16, Entropy Mean bf16)"""
        detector_path = getattr(
            self.hparams,
            "nn_detector_pure_path",
            "PLACEHOLDER",
        )

        if hasattr(self, "detector_pure_nn") and self.detector_pure_nn is not None:
            return

        self.detector_pure_nn = Detector(input_dim=3, hidden_dim=8).to(self.device)
        self.detector_pure_nn.eval()
        for p in self.detector_pure_nn.parameters():
            p.requires_grad_(False)

        if os.path.exists(detector_path):
            state = torch.load(
                detector_path, map_location=self.device, weights_only=False
            )
            self.detector_pure_nn.load_state_dict(state["model"])
            self.detector_pure_threshold = state.get("threshold", 0.5)
            logger.info(f"Loaded NN detector (pure, input_dim=3) from {detector_path}")
        else:
            logger.warning(
                f"NN detector (pure) not found at {detector_path}. Using untrained."
            )
            self.detector_pure_threshold = 0.5

    ####################

    def attack(
        self,
        train_set,
        max_key=None,
        min_key=None,
        hparams=None,
        progressbar=None,
        train_loader_kwargs={},
    ):

        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (
            isinstance(train_set, DataLoader) or isinstance(train_set, LoopedLoader)
        ):
            train_loader_kwargs["ckpt_prefix"] = None
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TEST, **train_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()

        self.sample_rate = hparams["sample_rate"]

        # Determine attack type from hparams and initialize the detector
        self.attack_type = hparams.get("adv_type", "pure_gaussian")
        if self.attack_type == "hybrid_pure_gaussian":
            self._init_gaussian_detector_pure()
        elif self.attack_type == "hybrid_pure":
            self._init_detector_pure_nn()
        elif self.attack_type == "hybrid_separate_gaussian":
            self._init_detector_separate_gaussian()
        elif self.attack_type == "hybrid_separate":
            self._init_detector_separate_nn()

        for batch in tqdm(train_set, dynamic_ncols=True, disable=not progressbar):
            # for batch in train_set:
            self._initialize_vars()
            batch = batch.to(self.device)
            # First reset delta
            global_optimal_delta = torch.zeros(
                batch.batchsize, self.global_max_length
            ).to(self.device)
            self.global_optimal_delta = nn.Parameter(global_optimal_delta)
            # Next, reset optimizers
            if self._optimizer_arg_1 is None:
                self.optimizer_1 = torch.optim.Adam(
                    params=[self.global_optimal_delta], lr=self.learning_rate_1
                )
            else:
                self.optimizer_1 = self._optimizer_arg_1(
                    params=[self.global_optimal_delta], lr=self.learning_rate_1
                )
            # Then calculate the adversarial sample
            for i in batch.path:
                root_path = hparams["path_adapt"]
                os.makedirs(root_path, exist_ok=True)
                file_name = os.path.basename(i)
                save_dirct = os.path.join(root_path, file_name)
                save_dirct = save_dirct.replace(".flac", ".wav")

                if not os.path.exists(save_dirct):
                    self.attack_1st_stage(batch, hparams, save_dirct)

    def attack_1st_stage(self, batch, hparams, save_dirct):
        """
        The first stage of the attack.
        """
        # Compute local shape
        local_batch_size = batch.batchsize
        real_lengths = (
            (batch.sig[1] * batch.sig[0].size(1)).long().detach().cpu().numpy()
        )
        local_max_length = np.max(real_lengths)
        # Initialize rescale
        rescale = (
            np.ones([local_batch_size, local_max_length], dtype=np.float32)
            * self.initial_rescale
        )
        # Reformat input
        input_mask = np.zeros([local_batch_size, local_max_length], dtype=np.float32)
        original_input = torch.clone(batch.sig[0])

        for local_batch_size_idx in range(local_batch_size):
            input_mask[local_batch_size_idx, : real_lengths[local_batch_size_idx]] = 1
        # Optimization loop
        almost_successful = [None] * local_batch_size
        successful_adv_input_2 = [None] * local_batch_size
        first_hit = [None] * local_batch_size
        best_hit = [None] * local_batch_size
        best_eta = [None] * local_batch_size
        token_lenghts = (
            (batch.tokens[1] * batch.tokens[0].size(1)).long().detach().cpu().numpy()
        )
        count_succs = [None] * local_batch_size
        best_loss_2nd_stage = [np.inf] * local_batch_size
        best_score = [None] * local_batch_size

        for iter_1st_stage_idx in range(self.max_iter_1):
            self.optimizer_1.zero_grad()

            forward_fn_map = {
                "hybrid_pure_gaussian": self.forward_1st_stage_adaptive_pure_gaussian,
                "hybrid_pure": self.forward_1st_stage_adaptive_pure_nn,
                "hybrid_separate_gaussian": self.forward_1st_stage_adaptive_separate_gaussian,
                "hybrid_separate": self.forward_1st_stage_adaptive_separate_nn,
            }
            forward_fn = forward_fn_map.get(
                self.attack_type, self.forward_1st_stage_adaptive_pure_gaussian
            )

            (
                loss,
                loss_2,
                characteristic,
                local_delta,
                masked_adv_input,
                _,
            ) = forward_fn(
                original_input=original_input,
                batch=batch,
                local_batch_size=local_batch_size,
                local_max_length=local_max_length,
                rescale=rescale,
                input_mask=input_mask,
                hparams=hparams,
                real_lengths=real_lengths,
            )
            loss.backward()
            self.global_optimal_delta.grad = torch.sign(self.global_optimal_delta.grad)
            self.optimizer_1.step()

            for local_batch_size_idx in range(local_batch_size):
                almost_successful[local_batch_size_idx] = masked_adv_input[
                    local_batch_size_idx
                ]
                torchaudio.save(
                    "tmp.wav",
                    almost_successful[local_batch_size_idx][
                        : real_lengths[local_batch_size_idx]
                    ]
                    .detach()
                    .cpu()[None, :],
                    self.sample_rate,
                )
                data_adv, _ = torchaudio.load("tmp.wav")
                batch.sig[0][local_batch_size_idx][
                    : real_lengths[local_batch_size_idx]
                ] = data_adv

            _, _, best_hyps = self.compute_forward(batch, stage=sb.Stage.TEST)

            for local_batch_size_idx in range(local_batch_size):
                tokens = (
                    batch.tokens[0][
                        local_batch_size_idx, 0 : token_lenghts[local_batch_size_idx]
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                pred_test = np.array(best_hyps[local_batch_size_idx])
                if len(pred_test) == len(tokens) and (pred_test == tokens).all():
                    if loss_2.detach() < best_loss_2nd_stage[local_batch_size_idx]:
                        best_loss_2nd_stage[local_batch_size_idx] = loss_2.detach()
                        self.alpha = min(self.alpha * 1.2, 0.999999999)
                        best_eta[local_batch_size_idx] = (
                            rescale[local_batch_size_idx] * self.eps
                        )

                        if iter_1st_stage_idx > 30:
                            max_local_delta = np.max(
                                np.abs(
                                    local_delta[local_batch_size_idx]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )
                            )
                            if (
                                rescale[local_batch_size_idx][0] * self.eps
                                > max_local_delta
                            ):
                                rescale[local_batch_size_idx] = (
                                    max_local_delta / self.eps
                                )
                            rescale[local_batch_size_idx] *= self.decrease_factor_eps

                        # Save the best adversarial example
                        if successful_adv_input_2[local_batch_size_idx] is None:
                            first_hit[local_batch_size_idx] = iter_1st_stage_idx
                        successful_adv_input_2[local_batch_size_idx] = masked_adv_input[
                            local_batch_size_idx
                        ]
                        best_hit[local_batch_size_idx] = iter_1st_stage_idx
                        if count_succs[local_batch_size_idx] is None:
                            count_succs[local_batch_size_idx] = 1
                        else:
                            count_succs[local_batch_size_idx] += 1
                        best_score[local_batch_size_idx] = characteristic

            # If attack is unsuccessful
            if iter_1st_stage_idx == self.max_iter_1 - 1:
                for local_batch_size_idx, dirct in enumerate(batch.path):
                    if successful_adv_input_2[local_batch_size_idx] is None:
                        successful_adv_input_2[local_batch_size_idx] = masked_adv_input[
                            local_batch_size_idx
                        ]
                        with open(hparams["unsuccesfull_adapt"], "a") as myfile:
                            wr = csv.writer(myfile)
                            wr.writerow(
                                [
                                    [dirct],
                                    [first_hit[local_batch_size_idx]],
                                    [best_hit[local_batch_size_idx]],
                                    [best_eta[local_batch_size_idx]],
                                    [count_succs[local_batch_size_idx]],
                                    [self.alpha],
                                    [characteristic.cpu().detach().item()],
                                ]
                            )
                            myfile.close()
                    else:
                        with open(hparams["succesfull_adapt"], "a") as myfile:
                            wr = csv.writer(myfile)
                            wr.writerow(
                                [
                                    [dirct],
                                    [first_hit[local_batch_size_idx]],
                                    [best_hit[local_batch_size_idx]],
                                    [best_eta[local_batch_size_idx][0]],
                                    [count_succs[local_batch_size_idx]],
                                    [self.alpha],
                                    [
                                        best_score[local_batch_size_idx]
                                        .cpu()
                                        .detach()
                                        .item()
                                    ],
                                ]
                            )
                            myfile.close()
                    torchaudio.save(
                        save_dirct,
                        successful_adv_input_2[local_batch_size_idx][
                            : real_lengths[local_batch_size_idx]
                        ]
                        .detach()
                        .cpu()[None, :],
                        self.sample_rate,
                    )

        result = torch.stack(successful_adv_input_2)
        batch.sig = original_input, batch.sig[1]
        return result

    ########### Tools for forward passes ######################

    def _get_entropy_mean_differentiable(self, predictions):
        """Differentiable entropy mean from model output log-probabilities."""

        p_ctc = predictions[0]
        p_ctc_prob = torch.exp(p_ctc)

        eps = 1e-8
        p_clamped = torch.clamp(p_ctc_prob, min=eps, max=1.0 - eps)

        log_p = torch.log(p_clamped)
        entropy_t = -torch.sum(p_clamped * log_p, dim=-1)

        return torch.mean(entropy_t)

    def _get_PVP(self, batch):
        """Compute pairwise WER between all precision combinations (PVP).

        Returns a single scalar tensor (mean WER across all utterance-pairs
        and all precision pairs). This is non-differentiable (uses token decoding).
        """
        precisions = ["fp32", "fp16", "bf16"]
        predictions_by_precision = {}

        for precision in precisions:
            eval_dtype = AMPConfig.from_name(precision).dtype
            ctx = TorchAutocast(device_type=self.device, dtype=eval_dtype)
            with torch.no_grad(), ctx:
                _, _, p_tokens = self.compute_forward(batch, stage=sb.Stage.VALID)

            predicted_words = self.tokenizer(p_tokens, task="decode_from_list")
            predictions_by_precision[precision] = predicted_words

        ids = batch.id
        batch_size = len(ids)
        pair_wers = np.zeros((batch_size, 3), dtype=np.float32)

        for pair_idx, (prec1, prec2) in enumerate(combinations(precisions, 2)):
            scores = wer_details_for_batch(
                ids,
                predictions_by_precision[prec1],
                predictions_by_precision[prec2],
                compute_alignments=True,
            )
            for utt_idx, s in enumerate(scores):
                pair_wers[utt_idx, pair_idx] = s["WER"]

        # Return a single scalar: mean across all utterances and all precision pairs
        return torch.tensor(np.mean(pair_wers), dtype=torch.float32, device=self.device)

    def _get_precision_varying_entropy(self, batch):
        """Compute differentiable entropy mean for each precision.

        Returns:
            entropy_features: [batch_size, 3] — [entropy_fp32, entropy_fp16, entropy_bf16]
        """
        characteristics = []
        for precision in ["fp32", "fp16", "bf16"]:
            eval_dtype = AMPConfig.from_name(precision).dtype
            ctx = TorchAutocast(device_type=self.device, dtype=eval_dtype)
            with ctx:
                predictions = self.compute_forward(batch, Stage.ATTACK)
                entropy_mean = self._get_entropy_mean_differentiable(predictions)
            characteristics.append(entropy_mean)

        # Stack into [3] and add batch dim → [1, 3] for Detector's BatchNorm1d(3)
        return torch.stack(characteristics, dim=-1).unsqueeze(0)  # [1, 3]

    ############ Forward passes ##############

    def forward_1st_stage_adaptive_pure_gaussian(
        self,
        original_input: np.ndarray,
        batch: sb.dataio.batch.PaddedBatch,
        local_batch_size: int,
        local_max_length: int,
        rescale: np.ndarray,
        input_mask: np.ndarray,
        hparams,
        real_lengths: np.ndarray,
    ):
        """Computes (1 - alpha) * l + alpha * Mahalanobis(gaussian)"""

        # Compute perturbed inputs
        local_delta = self.global_optimal_delta[:local_batch_size, :local_max_length]
        local_delta_rescale = torch.clamp(local_delta, -self.eps, self.eps).to(
            self.device
        )
        local_delta_rescale *= torch.tensor(rescale).to(self.device)
        adv_input = local_delta_rescale + torch.tensor(original_input).to(self.device)
        masked_adv_input = adv_input * torch.tensor(input_mask).to(self.device)
        # Compute loss and decoded output
        batch.sig = masked_adv_input, batch.sig[1]
        eval_dtype = AMPConfig.from_name("fp32").dtype
        self.evaluation_ctx = TorchAutocast(device_type=self.device, dtype=eval_dtype)
        with self.evaluation_ctx:
            predictions = self.compute_forward(batch, Stage.ATTACK)
            if hparams["eval_precision"] == "fp32":
                pred_1 = predictions
            entropy_mean_1 = self._get_entropy_mean_differentiable(predictions)
        eval_dtype = AMPConfig.from_name("fp16").dtype
        self.evaluation_ctx = TorchAutocast(device_type=self.device, dtype=eval_dtype)
        with self.evaluation_ctx:
            predictions = self.compute_forward(batch, Stage.ATTACK)
            if hparams["eval_precision"] == "fp16":
                pred_1 = predictions
            entropy_mean_2 = self._get_entropy_mean_differentiable(predictions)

        eval_dtype = AMPConfig.from_name("bf16").dtype
        self.evaluation_ctx = TorchAutocast(device_type=self.device, dtype=eval_dtype)
        with self.evaluation_ctx:
            predictions = self.compute_forward(batch, Stage.ATTACK)
            if hparams["eval_precision"] == "bf16":
                pred_1 = predictions
            entropy_mean_3 = self._get_entropy_mean_differentiable(predictions)

        loss_cw = self.compute_objectives(pred_1, batch, Stage.ATTACK)

        loss_1 = self.const * loss_cw + torch.norm(local_delta_rescale)

        entropy_vec = torch.stack(
            [entropy_mean_1, entropy_mean_2, entropy_mean_3], dim=-1
        )

        # Mahalanobis distance on 3d-Gaussian
        delta = entropy_vec - self.gaussian_mean_pure
        mahalanobis_sq = torch.sum(
            delta.unsqueeze(0) @ self.gaussian_precision_pure * delta.unsqueeze(0),
            dim=-1,
        ).squeeze(0)
        loss_2 = mahalanobis_sq

        loss = (1 - self.alpha) * loss_1 + self.alpha * loss_2.to(self.device)

        return (
            loss,
            loss_2,
            torch.mean(entropy_vec),
            local_delta,
            masked_adv_input,
            local_delta_rescale,
        )

    def forward_1st_stage_adaptive_pure_nn(
        self,
        original_input: np.ndarray,
        batch: sb.dataio.batch.PaddedBatch,
        local_batch_size: int,
        local_max_length: int,
        rescale: np.ndarray,
        input_mask: np.ndarray,
        hparams,
        real_lengths: np.ndarray,
    ):
        """Calculates L = (1 - alpha) * l + alpha * MSE(nn_pred, threshold)"""
        local_delta = self.global_optimal_delta[:local_batch_size, :local_max_length]
        local_delta_rescale = torch.clamp(local_delta, -self.eps, self.eps).to(
            self.device
        )
        local_delta_rescale *= torch.tensor(rescale).to(self.device)
        adv_input = local_delta_rescale + torch.tensor(original_input).to(self.device)
        masked_adv_input = adv_input * torch.tensor(input_mask).to(self.device)
        batch.sig = masked_adv_input, batch.sig[1]

        batch.sig = masked_adv_input, batch.sig[1]
        eval_dtype = AMPConfig.from_name("fp32").dtype
        self.evaluation_ctx = TorchAutocast(device_type=self.device, dtype=eval_dtype)
        with self.evaluation_ctx:
            predictions = self.compute_forward(batch, Stage.ATTACK)
            if hparams["eval_precision"] == "fp32":
                pred_1 = predictions
            entropy_mean_1 = self._get_entropy_mean_differentiable(predictions)
        eval_dtype = AMPConfig.from_name("fp16").dtype
        self.evaluation_ctx = TorchAutocast(device_type=self.device, dtype=eval_dtype)
        with self.evaluation_ctx:
            predictions = self.compute_forward(batch, Stage.ATTACK)
            if hparams["eval_precision"] == "fp16":
                pred_1 = predictions
            entropy_mean_2 = self._get_entropy_mean_differentiable(predictions)

        eval_dtype = AMPConfig.from_name("bf16").dtype
        self.evaluation_ctx = TorchAutocast(device_type=self.device, dtype=eval_dtype)
        with self.evaluation_ctx:
            predictions = self.compute_forward(batch, Stage.ATTACK)
            if hparams["eval_precision"] == "bf16":
                pred_1 = predictions
            entropy_mean_3 = self._get_entropy_mean_differentiable(predictions)

        loss_cw = self.compute_objectives(pred_1, batch, Stage.ATTACK)

        loss_1 = self.const * loss_cw + torch.norm(local_delta_rescale)

        entropy_vec = torch.stack(
            [entropy_mean_1, entropy_mean_2, entropy_mean_3], dim=-1
        )

        detector_out = self.detector_pure_nn(entropy_vec)
        boundary_target = torch.full_like(detector_out, self.detector_pure_threshold)
        detector_loss = torch.nn.functional.mse_loss(detector_out, boundary_target)

        total_loss = (1.0 - self.alpha) * loss_1 + self.alpha * detector_loss
        return (
            total_loss,
            detector_loss,
            torch.mean(entropy_vec),
            local_delta,
            masked_adv_input,
            local_delta_rescale,
        )

    def forward_1st_stage_adaptive_separate_nn(
        self,
        original_input: np.ndarray,
        batch: sb.dataio.batch.PaddedBatch,
        local_batch_size: int,
        local_max_length: int,
        rescale: np.ndarray,
        input_mask: np.ndarray,
        hparams,
        real_lengths: np.ndarray,
    ):
        """Calculates L = (1 - alpha) * l + alpha * MSE(nn_pred, threshold)"""
        local_delta = self.global_optimal_delta[:local_batch_size, :local_max_length]
        local_delta_rescale = torch.clamp(local_delta, -self.eps, self.eps).to(
            self.device
        )
        local_delta_rescale *= torch.tensor(rescale).to(self.device)
        adv_input = local_delta_rescale + torch.tensor(original_input).to(self.device)
        masked_adv_input = adv_input * torch.tensor(input_mask).to(self.device)
        batch.sig = masked_adv_input, batch.sig[1]

        # CW loss using default precision
        eval_dtype = AMPConfig.from_name(hparams["eval_precision"]).dtype
        self.evaluation_ctx = TorchAutocast(device_type=self.device, dtype=eval_dtype)
        with self.evaluation_ctx:
            predictions = self.compute_forward(batch, Stage.ATTACK)
            ctc_loss = self.compute_objectives(predictions, batch, Stage.ATTACK)
        cw_loss = self.const * ctc_loss + torch.norm(local_delta_rescale)

        # Compute entropy from the already-computed predictions
        entropy_mean = self._get_entropy_mean_differentiable(predictions)

        pvp_wer = self._get_PVP(batch)

        combined_features = torch.stack([pvp_wer, entropy_mean]).unsqueeze(0)

        detector_out = self.detector_separate_nn(combined_features)
        boundary_target = torch.full_like(
            detector_out, self.detector_separate_threshold
        )
        detector_loss = torch.nn.functional.mse_loss(detector_out, boundary_target)

        total_loss = (1.0 - self.alpha) * cw_loss + self.alpha * detector_loss
        return (
            total_loss,
            detector_loss,
            entropy_mean,
            local_delta,
            masked_adv_input,
            local_delta_rescale,
        )

    def forward_1st_stage_adaptive_separate_gaussian(
        self,
        original_input: np.ndarray,
        batch: sb.dataio.batch.PaddedBatch,
        local_batch_size: int,
        local_max_length: int,
        rescale: np.ndarray,
        input_mask: np.ndarray,
        hparams,
        real_lengths: np.ndarray,
    ):
        """Computes (1 - alpha) * l + alpha * Malahanobis (gaussian)"""

        local_delta = self.global_optimal_delta[:local_batch_size, :local_max_length]
        local_delta_rescale = torch.clamp(local_delta, -self.eps, self.eps).to(
            self.device
        )
        local_delta_rescale *= torch.tensor(rescale).to(self.device)
        adv_input = local_delta_rescale + torch.tensor(original_input).to(self.device)
        masked_adv_input = adv_input * torch.tensor(input_mask).to(self.device)
        batch.sig = masked_adv_input, batch.sig[1]

        # CW loss using default precision
        eval_dtype = AMPConfig.from_name(hparams["eval_precision"]).dtype
        self.evaluation_ctx = TorchAutocast(device_type=self.device, dtype=eval_dtype)
        with self.evaluation_ctx:
            predictions = self.compute_forward(batch, Stage.ATTACK)
            ctc_loss = self.compute_objectives(predictions, batch, Stage.ATTACK)
        cw_loss = self.const * ctc_loss + torch.norm(local_delta_rescale)

        entropy_mean = self._get_entropy_mean_differentiable(predictions)

        pvp_wer = self._get_PVP(batch).to(self.device)

        combined_features = torch.stack([pvp_wer, entropy_mean]).unsqueeze(0)

        # Malahanobis
        delta = combined_features - self.gaussian_mean_separate
        mahalanobis_sq = torch.sum(
            delta @ self.gaussian_precision_separate * delta, dim=-1
        )
        detector_loss = torch.mean(mahalanobis_sq)

        total_loss = (1.0 - self.alpha) * cw_loss + self.alpha * detector_loss
        return (
            total_loss,
            detector_loss,
            entropy_mean,
            local_delta,
            masked_adv_input,
            local_delta_rescale,
        )


############ dataio ################


def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """

    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError("sorting must be random, ascending or descending")

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"],
        replacements={"data_root": data_folder},
    )

    # We also sort the validation data so it is faster to validate
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        info = audio_io.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate,
            hparams["sample_rate"],
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "tokens_bos", "tokens_eos", "tokens"],
    )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams_train = hparams["dynamic_batch_sampler_train"]
        dynamic_hparams_valid = hparams["dynamic_batch_sampler_valid"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_train,
        )
        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_valid,
        )

    return (
        train_data,
        valid_data,
        test_data,
        train_batch_sampler,
        valid_batch_sampler,
    )


def dataio_prepare_2(hparams, file_path, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    Adapted for CommonVoice data: adds audio resampling and uses SentencePiece
    tokenizer interface.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=file_path,
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(sort_key="duration", reverse=True)
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError("sorting must be random, ascending or descending")

    datasets = [train_data]

    # 2. Define audio pipeline:
    # Adapt from CommonVoice: read audio, downmix to mono, resample to target sample rate
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig", "path")
    def audio_pipeline(wav):
        info = audio_io.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        if info.num_channels > 1:
            sig = torch.mean(sig, dim=1)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate,
            hparams["sample_rate"],
        )(sig)
        yield resampled
        yield wav

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    # Adapt from CommonVoice: use SentencePiece tokenizer interface (tokenizer.sp)
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    # Keep "wrd" and "path" as they are required by the attack code
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens", "path"],
    )
    train_batch_sampler = None

    if hparams["dynamic_batching"]:
        from speechbrain.dataio.batch import PaddedBatch  # noqa
        from speechbrain.dataio.dataloader import SaveableDataLoader  # noqa
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]
        hop_size = hparams["feats_hop_size"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            length_func=lambda x: x["duration"] * (1 / hop_size),
            **dynamic_hparams,
        )

    return (
        train_data,
        train_batch_sampler,
    )


if __name__ == "__main__":
    print("AA ", torch.cuda.device_count())
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    if use_cuda:
        print("__CUDNN VERSION:", torch.backends.cudnn.version())
        print("__Number CUDA Devices:", torch.cuda.device_count())
        print("__CUDA Device Name:", torch.cuda.get_device_name(0))
        print(
            "__CUDA Device Total Memory [GB]:",
            torch.cuda.get_device_properties(0).total_memory,
        )

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_common_voice,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "train_tsv_file": hparams["train_tsv_file"],
            "dev_tsv_file": hparams["dev_tsv_file"],
            "test_tsv_file": hparams["test_tsv_file"],
            "accented_letters": hparams["accented_letters"],
            "language": hparams["language"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        annotation_train=hparams["train_csv"],
        annotation_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
    )

    # here we create the datasets objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_datasets,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # We load the pretrained wav2vec2 model
    if "pretrainer" in hparams.keys():
        hparams["pretrainer"].collect_files()
        hparams["pretrainer"].load_collected()

    # We dynamically add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    asr_brain.tokenizer = tokenizer

    vocab_list = [tokenizer.sp.id_to_piece(i) for i in range(tokenizer.sp.vocab_size())]

    from speechbrain.decoders.ctc import CTCBeamSearcher

    test_searcher = CTCBeamSearcher(
        **hparams["test_beam_search"],
        vocab_list=vocab_list,
    )

    cw_data, _ = dataio_prepare_2(
        hparams, hparams["cw_audio_adv_transcripts"], tokenizer
    )
    asr_brain.attack(
        cw_data,
        hparams=hparams,
        train_loader_kwargs=hparams["test_dataloader_options"],
    )

    if not os.path.exists(hparams["output_wer_folder"]):
        os.makedirs(hparams["output_wer_folder"])
    adv_test_data, label_encoder = dataio_prepare_2(
        hparams, hparams["adv_audio_adv_transcripts"], tokenizer
    )

    asr_brain.hparams.test_wer_file = os.path.join(
        hparams["output_wer_folder"], hparams["adv_WER"]
    )
    print(
        f"PRECISION {hparams['precision']}, EVAL PRECISION {hparams['eval_precision']}"
    )
    asr_brain.evaluate(
        adv_test_data,
        test_loader_kwargs=hparams["test_dataloader_options"],
        min_key="WER",
    )

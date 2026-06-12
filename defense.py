#!/usr/bin/env/python3
"""Recipe for training a wav2vec-based ctc ASR system with librispeech.
The system employs wav2vec as its encoder. Decoding is performed with
ctc greedy decoder during validation and a beam search with an optional
language model during test. The test searcher can be chosen from the following
options: CTCBeamSearcher, CTCPrefixBeamSearcher, TorchAudioCTCPrefixBeamSearcher.

To run this recipe, do the following:
> python train_with_wav2vec.py hparams/train_{hf,sb}_wav2vec.yaml
The neural network is trained on CTC likelihood target and character units
are used as basic recognition tokens.

Authors
 * Rudolf A Braun 2022
 * Titouan Parcollet 2022
 * Sung-Lin Yeh 2021
 * Ju-Chieh Chou 2020
 * Mirco Ravanelli 2020
 * Abdel Heba 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020
 * Adel Moumen 2023
"""

import os
import pickle
import statistics
import sys
from collections import defaultdict

# Added by me
from enum import Enum, auto
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import speechbrain as sb
import torch
import torch.nn as nn
from hyperpyyaml import load_hyperpyyaml
from matplotlib import pyplot as plt
from scipy.stats import entropy, multivariate_normal, norm
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from speechbrain.dataio import audio_io
from speechbrain.dataio.dataio import read_audio, split_word
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.dataio.dataset import (
    DynamicItemDataset,
    add_dynamic_item,
    set_output_keys,
)
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.autocast import AMPConfig, TorchAutocast
from speechbrain.utils.data_pipeline import provides, takes
from speechbrain.utils.distributed import if_main_process
from speechbrain.utils.edit_distance import _str_equals, wer_details_for_batch
from speechbrain.utils.logger import get_logger
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample
from tqdm.contrib import tqdm

logger = get_logger(__name__)


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
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )

        elif stage == sb.Stage.TEST:
            p_tokens = test_searcher(p_ctc, wav_lens)
            # print(p_tokens)
            # print(f"[p_ctc] dtype: {p_ctc.dtype}, autocast: {torch.is_autocast_enabled()}")
            # print(f"[p_tokens] dtype: {p_tokens.dtype}, autocast: {torch.is_autocast_enabled()}")
            candidates = []
            scores = []

            for batch in p_tokens:
                candidates.append([hyp.text for hyp in batch])
                scores.append([hyp.score for hyp in batch])

            if hasattr(self.hparams, "rescorer"):
                p_tokens, _ = self.hparams.rescorer.rescore(candidates, scores)

        return p_ctc, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        p_ctc, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens, tokens_lens = batch.tokens

        # Labels must be extended if parallel augmentation or concatenated
        # augmentation was performed on the input (increasing the time dimension)
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            (
                tokens,
                tokens_lens,
            ) = self.hparams.wav_augment.replicate_multiple_labels(tokens, tokens_lens)

        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        loss = loss_ctc

        if stage == sb.Stage.VALID:
            # Decode token terms to words
            predicted_words = [
                "".join(self.tokenizer.decode_ndim(utt_seq)).split(" ")
                for utt_seq in predicted_tokens
            ]
        elif stage == sb.Stage.TEST:
            if hasattr(self.hparams, "rescorer"):
                predicted_words = [hyp[0].split(" ") for hyp in predicted_tokens]
            else:
                predicted_words = [hyp[0].text.split(" ") for hyp in predicted_tokens]

        if stage != sb.Stage.TRAIN and stage != Stage.ATTACK:
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def compute_objectives_2(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        p_ctc, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens, tokens_lens = batch.tokens

        # Labels must be extended if parallel augmentation or concatenated
        # augmentation was performed on the input (increasing the time dimension)
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            (
                tokens,
                tokens_lens,
            ) = self.hparams.wav_augment.replicate_multiple_labels(tokens, tokens_lens)

        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        loss = loss_ctc

        if stage == sb.Stage.VALID:
            # Decode token terms to words
            predicted_words = [
                "".join(self.tokenizer.decode_ndim(utt_seq)).split(" ")
                for utt_seq in predicted_tokens
            ]
        elif stage == sb.Stage.TEST:
            if hasattr(self.hparams, "rescorer"):
                predicted_words = [hyp[0].split(" ") for hyp in predicted_tokens]
            else:
                predicted_words = [hyp[0].text.split(" ") for hyp in predicted_tokens]

        if stage != sb.Stage.TRAIN and stage != Stage.ATTACK:
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return ids, predicted_words

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

        if stage == sb.Stage.TEST:
            if hasattr(self.hparams, "rescorer"):
                self.hparams.rescorer.move_rescorers_to_device()

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
        # Handling SpeechBrain vs HuggingFace pretrained models
        if hasattr(self.modules, "extractor"):  # SpeechBrain pretrained model
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.encoder_wrapper.parameters()
            )

        else:  # HuggingFace pretrained model
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.wav2vec2.parameters()
            )

        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        # save the optimizers in a dictionary
        # the key will be used in `freeze_optimizers()`
        self.optimizers_dict = {
            "model_optimizer": self.model_optimizer,
        }
        if not self.hparams.freeze_wav2vec:
            self.optimizers_dict["wav2vec_optimizer"] = self.wav2vec_optimizer

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("wav2vec_opt", self.wav2vec_optimizer)
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)

    def calculate_wer(
        self,
        test_set,
        precision_types=("fp32", "fp16", "bf16"),
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        if progressbar is None:
            progressbar = not self.noprogressbar

        # Only show progressbar if requested and main_process
        enable = progressbar and sb.utils.distributed.if_main_process()

        if not (isinstance(test_set, DataLoader) or isinstance(test_set, LoopedLoader)):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, sb.Stage.TEST, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()

        pred_words = {}

        for p in precision_types:
            eval_dtype = AMPConfig.from_name(p).dtype
            self.evaluation_ctx = TorchAutocast(
                device_type=self.device,
                dtype=eval_dtype,
            )
            pred_words_2 = []
            with torch.no_grad():
                for batch in tqdm(
                    test_set,
                    dynamic_ncols=True,
                    disable=enable,  # not enable,
                    colour=self.tqdm_barcolor["test"],
                ):
                    # self.step += 1
                    # loss = self.evaluate_batch(batch, stage=Stage.TEST)
                    with self.evaluation_ctx:
                        out = self.compute_forward(batch, stage=sb.Stage.TEST)
                        ids, predicted_words = self.compute_objectives_2(
                            out, batch, stage=sb.Stage.TEST
                        )
                        pred_words_2.append((ids, predicted_words))
            pred_words[p] = pred_words_2
            # stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            # stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            # self.on_stage_end(Stage.TEST, avg_test_loss, None)
        # self.step = 0
        return pred_words

    def PVP_characteristics(
        self,
        train_set,
        path_file,
        max_key=None,
        min_key=None,
        hparams=None,
        progressbar=None,
        train_loader_kwargs={},
    ):
        """
        Function that calculates the 24 scores
        (resulting from combing each of the 4 aggregation methods with the 6 characteristics).
        """

        # Characteristics
        precisions = ["fp32", "fp16", "bf16"]
        measurements = {"Entropy mean": 0, "Median mean": 0}
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

        measurements_dict = {}

        for precision in precisions:
            entr_avg, med_avg = [], []
            self.evaluation_ctx = TorchAutocast(
                device_type=self.device, dtype=AMPConfig.from_name(precision).dtype
            )
            with torch.no_grad():
                for batch in tqdm(
                    train_set, dynamic_ncols=True, disable=not progressbar
                ):
                    with self.evaluation_ctx:
                        predictions = self.compute_forward(batch, stage=sb.Stage.TEST)
                    p_ctc = torch.squeeze(predictions[0], dim=0)
                    p_ctc_prob = torch.exp(p_ctc).detach().cpu()

                    p_ctc_prob = np.array(p_ctc_prob)
                    # Remove extreme cases which lead to undefined characteristic values
                    p_ctc_prob = np.delete(
                        p_ctc_prob, np.where((p_ctc_prob == 0))[0], axis=0
                    )
                    p_ctc_prob = np.delete(
                        p_ctc_prob, np.where((p_ctc_prob == 1))[0], axis=0
                    )
                    # Entropy
                    entropy_1 = entropy(p_ctc_prob, axis=1)
                    entr_avg.append(np.mean(entropy_1))
                    # Median
                    median_prob = np.log(np.median(p_ctc_prob, axis=1))
                    med_avg.append(np.mean(median_prob))

            # Save an independent snapshot per precision.
            measurements_dict[precision] = {
                "Entropy mean": list(entr_avg),
            }

        with open(path_file, "wb") as file:
            pickle.dump(measurements_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
        pass

    def characteristics(
        self,
        train_set,
        path_file,
        max_key=None,
        min_key=None,
        hparams=None,
        progressbar=None,
        train_loader_kwargs={},
    ):
        """
        Function that calculates the 24 scores
        (resulting from combing each of the 4 aggregation methods with the 6 characteristics).
        """

        # Characteristics
        measurements = {"Entropy mean": 0, "Median mean": 0}
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

        entr_avg, med_avg = [], []

        with torch.no_grad():
            for batch in tqdm(train_set, dynamic_ncols=True, disable=not progressbar):
                with self.evaluation_ctx:
                    predictions = self.compute_forward(batch, stage=sb.Stage.TEST)
                p_ctc = torch.squeeze(predictions[0], dim=0)
                p_ctc_prob = torch.exp(p_ctc).detach().cpu()

                p_ctc_prob = np.array(p_ctc_prob)
                # Remove extreme cases which lead to undefined characteristic values
                p_ctc_prob = np.delete(
                    p_ctc_prob, np.where((p_ctc_prob == 0))[0], axis=0
                )
                p_ctc_prob = np.delete(
                    p_ctc_prob, np.where((p_ctc_prob == 1))[0], axis=0
                )
                # Entropy
                entropy_1 = entropy(p_ctc_prob, axis=1)
                entr_avg.append(np.mean(entropy_1))
                # Median
                median_prob = np.log(np.median(p_ctc_prob, axis=1))
                med_avg.append(np.mean(median_prob))

        # Saving the Characteristics
        measurements["Entropy mean"] = entr_avg
        # measurements['Median mean'] = med_avg

        with open(path_file, "wb") as file:
            pickle.dump(measurements, file, protocol=pickle.HIGHEST_PROTOCOL)
        pass


def dataio_prepare_2(hparams, file_path, tokenizer):
    data_folder = hparams["data_folder"]

    train_data = DynamicItemDataset.from_csv(
        csv_path=file_path,
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        train_data = train_data.filtered_sorted(sort_key="duration")
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(sort_key="duration", reverse=True)
        hparams["dataloader_options"]["shuffle"] = False

    else:
        raise NotImplementedError("sorting must be random, ascending or descending")

    datasets = [train_data]

    @takes("wav")
    @provides("sig", "path")
    def audio_pipeline(wav):
        info = audio_io.info(wav)
        sig = read_audio(wav)
        resampled = Resample(
            info.sample_rate,
            hparams["sample_rate"],
        )(sig)
        yield resampled
        yield wav

    add_dynamic_item(datasets, audio_pipeline)

    @takes("wrd")
    @provides("wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens")
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + tokens_list)
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    add_dynamic_item(datasets, text_pipeline)

    set_output_keys(
        datasets,
        ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens", "path"],
    )

    return train_data


def to_utt_dict(results):
    utt_dict = {}
    for ids, words in results:
        utt_id = ids[0]
        utt_dict[utt_id] = words[0]  # list of tokens
    return utt_dict


def pairwise_wer_cer(preds, asr_brain, space="_"):
    results = {}
    results_scores = {}

    for ref_p, hyp_p in combinations(preds.keys(), 2):
        wer_metric = asr_brain.hparams.error_rate_computer()
        cer_metric = asr_brain.hparams.cer_computer()

        ref_dict = preds[ref_p]
        hyp_dict = preds[hyp_p]

        common_utts = ref_dict.keys() & hyp_dict.keys()

        wer_list = []
        cer_list = []

        for utt_id in common_utts:
            ref = ref_dict[utt_id]
            hyp = hyp_dict[utt_id]

            # WER (word-level)
            _, wer = error_score(
                [utt_id],
                [hyp],
                [ref],
                split_tokens=False,
            )

            # CER (char-level)
            _, cer = error_score(
                [utt_id],
                [hyp],
                [ref],
                split_tokens=True,
            )

            wer_list.append((utt_id, wer))
            cer_list.append((utt_id, cer))
            # Sort by error (descending = worst first)
            # wer_list.sort(key=lambda x: x[0], reverse=True)
            # cer_list.sort(key=lambda x: x[0], reverse=True)

            # error_score([utt_id], [hyp], [ref], False)

            wer_metric.append(
                ids=[utt_id],
                predict=[hyp],
                target=[ref],
            )

            cer_metric.append(
                ids=[utt_id],
                predict=[hyp],
                target=[ref],
            )

        results[(ref_p, hyp_p)] = {
            "WER": wer_metric.summarize("error_rate"),
            "CER": cer_metric.summarize("error_rate"),
        }

        results_scores[(ref_p, hyp_p)] = {
            "WER": wer_list,
            "CER": cer_list,
        }

    return results, results_scores


def error_score(ids, predict, target, split_tokens=False, space="_"):

    if split_tokens:
        predict = split_word(predict, space="_")
        target = split_word(target, space="_")
    """
    if self.extract_concepts_values:
        predict = extract_concepts_values(
            predict,
            self.keep_values,
            self.tag_in,
            self.tag_out,
            space=self.space_token,
        )
        target = extract_concepts_values(
            target,
            self.keep_values,
            self.tag_in,
            self.tag_out,
            space=self.space_token,
        )
    """
    equality_comparator: Callable[[str, str], bool] = _str_equals

    scores = wer_details_for_batch(
        ids,
        target,
        predict,
        compute_alignments=True,
        equality_comparator=equality_comparator,
    )
    s = scores[0]
    return s["key"], s["WER"]


def build_char_dataset(values, benign=True, key_name="WER_max"):
    return {key_name: values, "benign_flg": [1 if benign else 0] * len(values)}


def merge(dict_1, dict_2, key):
    """
    Merge two dictionaries based on specific keys.

    :param dict_1: Dictionary variable.
    :param dict_2: Dictionary variable.
    :param key: Keys to use during the merge.
    :return: A dictionary merged based on specific keys.
    """
    dict_all = {x: dict_1[x] + dict_2[x] for x in key}
    return dict_all


def fit_gaussian(train_set, test_set, adv_set, key):
    """
    Gaussian distribution-based adversarial detector
    """

    char_key = [key, "benign_flg"]

    test_metrics = {x: test_set[x] for x in char_key}
    adv_metrics = {x: adv_set[x] for x in char_key}

    test_all = merge(test_metrics, adv_metrics, char_key)

    mean = np.mean(train_set[key])
    print("mean", mean)
    std = np.std(train_set[key])
    std = max(std, 1e-6)  # critical fix

    fitted_norm = norm.pdf(test_all[key], loc=mean, scale=std)

    # train_vals = np.array(train_set[key])
    # test_vals  = np.array(test_all[key])

    # # log transform
    # train_vals = np.log(train_vals + 1e-8)
    # test_vals  = np.log(test_vals + 1e-8)

    # mean = np.mean(train_vals)
    # std  = np.std(train_vals)
    # std  = max(std, 1e-6)

    # fitted_norm = norm.pdf(test_vals, loc=mean, scale=std)

    fpr, tpr, _ = roc_curve(test_all["benign_flg"], fitted_norm)

    roc_auc = auc(fpr, tpr)
    # print(roc_auc)
    # Ignorar primer punto (0,0) que siempre aparece
    fpr = fpr[1:]
    tpr = tpr[1:]
    # Remove (1,1) if present
    if len(tpr) > 0 and tpr[-1] == 1.0 and fpr[-1] == 1.0:
        fpr = fpr[:-1]
        tpr = tpr[:-1]

    fnr = 1 - tpr
    tnr = 1 - fpr
    worst_benign_idx = np.argmax(fnr)  # equivalente a np.argmin(tpr)
    worst_benign = {
        "tpr": float(tpr[worst_benign_idx]),
        "tnr": float(tnr[worst_benign_idx]),
        "fpr": float(fpr[worst_benign_idx]),
        "fnr": float(fnr[worst_benign_idx]),
    }

    # ============================
    # TPR @ FPR < 1% (or closest)
    # ============================
    target_fpr = 0.01  # FPR constraint
    target_tpr = 0.95  # desired minimum TPR

    mask = fpr < target_fpr

    if np.any(mask):
        # among FPR < 1%, choose the TPR closest to 0.95 but not smaller if possible
        candidate_tpr = tpr[mask]
        candidate_fpr = fpr[mask]
        # find the one closest to target_tpr
        idx = np.argmin(np.abs(candidate_tpr - target_tpr))
        selected_tpr = candidate_tpr[idx]
        selected_fpr = candidate_fpr[idx]
    else:
        # no TPR under FPR < 1%, choose the TPR closest to 0.95 globally
        idx = np.argmin(np.abs(tpr - target_tpr))
        selected_tpr = tpr[idx]
        selected_fpr = fpr[idx]

    tpr_at_fpr = float(selected_tpr)
    fpr_at_fpr = float(selected_fpr)
    fnr_at_fpr = 1 - tpr_at_fpr
    tnr_at_fpr = 1 - fpr_at_fpr

    return {
        "roc_auc": float(roc_auc),
        "worst_benign": worst_benign,
        "tpr_at_fpr": {
            "tpr": tpr_at_fpr,
            "fpr": fpr_at_fpr,
            "tnr": tnr_at_fpr,
            "fnr": fnr_at_fpr,
        },
    }


def precision_robustness_stats(
    asr_brain,
    data,
    # precision_types=("fp32", "fp16", "bf16"),
    dataloader_opts=None,
):
    """
    Runs ASR in multiple precisions, computes pairwise WER/CER differences,
    and returns aggregated max/min/median/mean per-utterance statistics.
    """

    # 1) Run ASR + collect predictions
    # pred_words = {}

    # for p in precision_types:
    #     eval_dtype = AMPConfig.from_name(p).dtype
    #     asr_brain.evaluation_ctx = TorchAutocast(
    #         device_type=asr_brain.device,
    #         dtype=eval_dtype,
    #     )

    #     with asr_brain.evaluation_ctx:
    #         pred_words[p] = asr_brain.calculate_wer(
    #             data,
    #             test_loader_kwargs=dataloader_opts,
    #             min_key="WER",
    #         )

    pred_words = asr_brain.calculate_wer(
        data,
        test_loader_kwargs=dataloader_opts,
        min_key="WER",
    )

    preds = {p: to_utt_dict(v) for p, v in pred_words.items()}

    # 2) Pairwise WER/CER
    _, pairwise_scores = pairwise_wer_cer(preds, asr_brain)

    # 3) Reorganize per utterance
    per_utt = defaultdict(lambda: {"WER": [], "CER": []})

    for metrics in pairwise_scores.values():
        for metric in ["WER", "CER"]:
            for utt_id, score in metrics[metric]:
                per_utt[utt_id][metric].append(score)

    # 4) Aggregate stats
    aggregated = {}

    for utt_id, metrics in per_utt.items():
        aggregated[utt_id] = {
            metric: {
                "max": max(values),
                "min": min(values),
                "median": statistics.median(values),
                "mean": sum(values) / len(values),
            }
            for metric, values in metrics.items()
        }

    # 5) Extract lists
    stats = {
        "WER": {
            "max": [v["WER"]["max"] for v in aggregated.values()],
            "min": [v["WER"]["min"] for v in aggregated.values()],
            "median": [v["WER"]["median"] for v in aggregated.values()],
            "mean": [v["WER"]["mean"] for v in aggregated.values()],
        },
        "CER": {
            "max": [v["CER"]["max"] for v in aggregated.values()],
            "min": [v["CER"]["min"] for v in aggregated.values()],
            "median": [v["CER"]["median"] for v in aggregated.values()],
            "mean": [v["CER"]["mean"] for v in aggregated.values()],
        },
    }

    return stats, aggregated


def all_metrics(
    stats_gaussian,
    stats_test,
    stats_adv,
):
    """
    Compute AUROC for all WER/CER characteristics using
    Gaussian precision-instability modeling.
    """

    metrics = ["WER", "CER"]
    aggregations = ["mean"]  # ['max', 'min', 'median', 'mean']

    results = {}

    for metric in metrics:
        results[metric] = {}

        for agg in aggregations:
            key = f"{metric}_{agg}"

            train_set = build_char_dataset(
                stats_gaussian[metric][agg], benign=True, key_name=key
            )

            test_set = build_char_dataset(
                stats_test[metric][agg], benign=True, key_name=key
            )

            adv_set = build_char_dataset(
                stats_adv[metric][agg], benign=False, key_name=key
            )

            roc_auc_dict = fit_gaussian(
                train_set=train_set, test_set=test_set, adv_set=adv_set, key=key
            )

            results[metric][agg] = roc_auc_dict  # auc

    return results


def load_csv(csv_path):
    return pd.read_csv(csv_path)


def sample_csv(df, n, seed=42):
    if len(df) < n:
        raise ValueError(f"Requested {n} samples but CSV has only {len(df)}")
    return df.sample(n=n, random_state=seed)


def combine_csvs(
    csv_paths,
    sample_sizes,
    output_csv,
    shuffle=True,
    seed=42,
):
    """
    csv_paths: list of paths to csv files
    sample_sizes: list of sample counts per csv (same length)
    output_csv: where to save the merged csv
    """

    assert len(csv_paths) == len(sample_sizes)

    dfs = []
    cnt = 0
    for csv_path, n in zip(csv_paths, sample_sizes):
        df = load_csv(csv_path)
        df = sample_csv(df, n, seed)

        # Rename ID column conditionally
        if Path(csv_path).name == "adv_audio_adv_transcripts.csv":
            if cnt == 0:
                cnt += 1
            else:
                df = df.reset_index(drop=True)
                df["ID"] = pd.Series(range(1, len(df) + 1)).astype(str).str.zfill(3)

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    if shuffle:
        combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)

    return str(output_csv)


def resolve_csv(spec):
    """
    spec can be:
    - string → already a csv path
    - dict → needs to be combined
    """
    if isinstance(spec, str):
        return spec

    return combine_csvs(
        csv_paths=spec["csvs"],
        sample_sizes=spec["sizes"],
        output_csv=spec["out"],
    )


def load_meas_data(file_path, benign_flg):
    """
    Load a file containing the Characteristics.

    :param file_path: *.pickle file path.
    :param benign_flg: Set 1 to Benign data and 0 to Adversarial data.
    :return: A dictionary containing the Characteristics.
    """
    with open(file_path, "rb") as file:
        measurements = pickle.load(file)
    # Get first key automatically
    first_key = list(measurements.keys())[0]
    total_length = len(measurements[first_key])
    measurements["benign_flg"] = [1 if benign_flg else 0] * total_length
    return measurements


def load_meas_data_pvp(file_path, benign_flg):
    """
    Load a PVP characteristics file containing one dictionary per precision.

    Each precision entry is expected to contain the same characteristic lists
    as the flat format, and this helper appends a precision-local benign flag.
    """
    with open(file_path, "rb") as file:
        measurements = pickle.load(file)

    for precision_key, precision_measurements in measurements.items():
        precision_measurements = dict(precision_measurements)
        first_key = next(iter(precision_measurements))
        total_length = len(precision_measurements[first_key])
        precision_measurements["benign_flg"] = [1 if benign_flg else 0] * total_length
        measurements[precision_key] = precision_measurements

    return measurements


def create_precision_feature_matrix(
    distriblock,
    feature_key="Entropy mean",
):

    precision_features = []
    expected_length = None

    for precision in ["fp32", "fp16", "bf16"]:
        precision_measurements = distriblock[precision]
        values = precision_measurements[feature_key]

        if expected_length is None:
            expected_length = len(values)
        elif len(values) != expected_length:
            raise ValueError(
                f"Mismatched length for {precision}:{feature_key} "
                f"(expected {expected_length}, got {len(values)})"
            )

        precision_features.append(values)

    data = torch.tensor(list(zip(*precision_features)), dtype=torch.float32)
    labels = torch.tensor(
        distriblock["fp32"]["benign_flg"], dtype=torch.float32
    )

    return data, labels


def distriblock_gaussians(train_set, test_set, adv_set, key):
    """
    Fit a Gaussian distribution to each Characteristic score computed for the utterances from a training set of benign data.
    If the probability of a new audio sample is below a chosen threshold under the Gaussian model,
    this example is classified as adversarial.

    :param train_set: Training set of benign data.
    :param test_set: Testing set of benign data.
    :param adv_set: Testing set of adversarial data.
    :param key: Characteristic to fit the gaussian.
    :return: Classifier performance in terms of AUROC.
    """
    char_key = []
    char_key.append(key)
    char_key.append("benign_flg")
    test_metrics = {x: test_set[x] for x in char_key}
    adv_metrics = {x: adv_set[x] for x in char_key}
    # print(np.mean(train_set[key]), np.mean(test_metrics[key]), np.mean(adv_metrics[key]))
    test_all = merge(test_metrics, adv_metrics, char_key)
    # mean, std = norm.fit(train_set[key])

    # train_vals = np.array(train_set[key])
    # test_vals  = np.array(test_all[key])

    # log transform
    # train_vals = np.log(train_vals + 1e-8)
    # test_vals  = np.log(test_vals + 1e-8)

    # mean = np.mean(train_vals)
    # std  = np.std(train_vals)
    # std  = max(std, 1e-6)
    # fitted_norm = norm.pdf(test_vals, loc=mean, scale=std)
    mean = np.mean(train_set[key])
    print("mean", mean)
    std = np.std(train_set[key])
    std = max(std, 1e-6)  # critical fix

    fitted_norm = norm.pdf(test_all[key], loc=mean, scale=std)

    # fitted_norm = norm.pdf(test_all[key], loc=mean, scale=std)
    fpr, tpr, threshold = roc_curve(test_all["benign_flg"], fitted_norm)
    roc_auc = auc(fpr, tpr)

    return roc_auc





def plot_gaussian_fit_vs_adversarial(
    train_values, benign_values, adv_values, save_path=None, show=False, title=None
):
    train_values = np.asarray(train_values, dtype=float)
    adv_values = np.asarray(adv_values, dtype=float)
    clean_values = np.asarray(benign_values, dtype=float)

    train_mean = float(np.mean(train_values))
    train_std = float(max(np.std(train_values), 1e-6))

    x_min = float(min(np.min(train_values), np.min(adv_values)))
    x_max = float(max(np.max(train_values), np.max(adv_values)))
    x_values = np.linspace(x_min, x_max, 400)
    gaussian_fit = norm.pdf(x_values, loc=train_mean, scale=train_std)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        x_values,
        gaussian_fit,
        color="black",
        linewidth=2,
        label="Train Gaussian fit",
    )

    combined = np.concatenate([clean_values, adv_values])
    bin_count = min(40, max(8, len(combined) // 3))
    bins = np.linspace(np.min(combined), np.max(combined), bin_count)

    ax.hist(
        clean_values,
        bins=bins,
        density=True,
        alpha=0.35,
        color="blue",
        label="Benign data",
    )

    ax.hist(
        adv_values,
        bins=bins,
        density=True,
        alpha=0.35,
        color="red",
        label="Adversarial data",
    )
    ax.set_xlabel("Entropy mean")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


############### NN Detector ######################

class Detector(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=8):
        super().__init__()

        self.norm = nn.BatchNorm1d(input_dim)

        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.linear(self.norm(features))


def create_feature_Matrix(pvp, distriblock):
    """Create the input feature matrix of shape [PVP score, Entropy Mean]"""

    wer_stats = pvp["WER"]
    if isinstance(wer_stats, dict) and "mean" in wer_stats:
        wer_values = wer_stats["mean"]
    else:
        wer_values = [metrics["WER"]["mean"] for metrics in pvp.values()]

    entropy_values = distriblock["Entropy mean"]
    benign_flags = distriblock["benign_flg"]

    data = torch.tensor(
        [
            [float(wer), float(entropy)]
            for wer, entropy in zip(wer_values, entropy_values)
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor(benign_flags, dtype=torch.float32)

    return data, labels


def train_detector(model, data, labels, epochs=50, lr=1e-3):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        logits = model(data)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()


def eval_detector(model, data, labels, threshold):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        acc = (preds == labels).float().mean()
    try:
        return float(acc.item())
    except Exception:
        return float(acc)


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        annotation_train=hparams["train_csv"],
        annotation_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
    )

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    asr_brain.tokenizer = tokenizer

    vocab_list = [tokenizer.sp.id_to_piece(i) for i in range(tokenizer.sp.vocab_size())]

    # We load the pretrained wav2vec2 model
    if "pretrainer" in hparams.keys():
        hparams["pretrainer"].collect_files()
        hparams["pretrainer"].load_collected()

    # Make all randomness reproducible with a configurable seed
    seed = hparams.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set higher-level seeds for reproducibility
    import random

    random.seed(seed)

    print(
        f"PRECISION {hparams['precision']}, EVAL PRECISION {hparams['eval_precision']}"
    )
    if hparams["adv_type"] == "dist":
        EXPERIMENTS = [
            {
                "name": "baseline_cw",
                "benign": hparams["benign_audio_train"],
                "adv": hparams["adv_audio_adv_transcripts"],
                "test": hparams["clean_audio_clean_transcripts"],
            }
        ]
        # adding objects to trainer:
        benign_csv = resolve_csv(EXPERIMENTS[0]["benign"])
        adv_csv = resolve_csv(EXPERIMENTS[0]["adv"])
        # adv_adapt_csv    = resolve_csv(EXPERIMENTS[0]["adv_adapt"])
        test_csv = resolve_csv(EXPERIMENTS[0]["test"])

        # # here we create the datasets objects as well as tokenization and encoding
        benign_data_test = dataio_prepare_2(hparams, benign_csv, tokenizer)
        adv_data = dataio_prepare_2(hparams, adv_csv, tokenizer)
        # adv_adapt_data, _ = dataio_prepare_2(hparams, adv_adapt_csv)
        test_data = dataio_prepare_2(hparams, test_csv, tokenizer)

        # We dynamically add the tokenizer to our brain class.
        # NB: This tokenizer corresponds to the one used for the LM!!

        from speechbrain.decoders.ctc import CTCBeamSearcher

        test_searcher = CTCBeamSearcher(
            **hparams["test_beam_search"],
            vocab_list=vocab_list,
        )

        # file_names = ["train.pickle", "val.pickle", "test.pickle", "adv_train.pickle", "adv_test.pickle", ]
        # data_sets = [train_set, val_set, test_set, adv_val, adv_test]
        file_names = ["train_1.pickle", "test_1.pickle", "adv_test_1.pickle"]
        data_sets = [benign_data_test, test_data, adv_data]

        characteristic_folder = hparams["distriblock_folder"]
        if not os.path.exists(characteristic_folder):
            os.makedirs(characteristic_folder)

        for i, data in enumerate(data_sets):
            if not os.path.exists(f"{characteristic_folder}/{file_names[i]}"):
                print(f"Saving characteristics in file: {file_names[i]}!")
                asr_brain.characteristics(
                    data,
                    f"{characteristic_folder}/{file_names[i]}",
                    train_loader_kwargs=hparams["test_dataloader_options"],
                    min_key="WER",
                )
        if os.path.exists(f"{characteristic_folder}/{file_names[0]}"):
            train_meas = load_meas_data(
                f"{characteristic_folder}/{file_names[0]}", benign_flg=True
            )
            keys = []
            for i in train_meas:
                keys.append(i)
        if os.path.exists(f"{characteristic_folder}/{file_names[1]}"):
            test_meas = load_meas_data(
                f"{characteristic_folder}/{file_names[1]}", benign_flg=True
            )
        if os.path.exists(f"{characteristic_folder}/{file_names[2]}"):
            adv_meas = load_meas_data(
                f"{characteristic_folder}/{file_names[2]}", benign_flg=False
            )
        if keys == ["Entropy mean", "Median mean", "benign_flg"]:
            plot_gaussian_fit_vs_adversarial(
                train_meas[keys[0]],
                test_meas[keys[0]],
                adv_meas[keys[0]],
                save_path=f"{characteristic_folder}/{keys[0].replace(' ', '_').lower()}_train_fit_vs_adv.png",
                title="Entropy mean: train Gaussian fit vs adversarial test data",
            )
            print(" ")
            print(
                "------------------------- Gaussian Classifiers results: ---------------------------"
            )
            print(keys)
            auroc = distriblock_gaussians(train_meas, test_meas, adv_meas, keys[0])
            print('Characteristic: "{}". AUROC: {:.4f}'.format(keys[0], auroc))
        else:
            sys.exit(
                "-------------Error when Characteristics were calculated-------------"
            )

    elif hparams["adv_type"] == "hybrid_separate":
        EXPERIMENTS = [
            {
                "name": "baseline_cw",
                "benign": hparams["train_benign"],
                # "val_benign": hparams["val_benign"],
                "val_adv": hparams["val_adv"],
                "test_benign": hparams["clean_audio_clean_transcripts"],
                "test_adv": hparams["adv_audio_adv_transcripts"],
            }
        ]
        benign_csv = resolve_csv(EXPERIMENTS[0]["benign"])
        # val_benign_csv = resolve_csv(EXPERIMENTS[0]["val_benign"])
        val_adv_csv = resolve_csv(EXPERIMENTS[0]["val_adv"])
        test_csv = resolve_csv(EXPERIMENTS[0]["test_benign"])
        adv_test_csv = resolve_csv(EXPERIMENTS[0]["test_adv"])

        benign_data_train = dataio_prepare_2(hparams, benign_csv, tokenizer)
        # benign_data_val = dataio_prepare_2(hparams, val_benign_csv, tokenizer)
        adv_data_val = dataio_prepare_2(hparams, val_adv_csv, tokenizer)
        benign_data_test = dataio_prepare_2(hparams, test_csv, tokenizer)
        adv_data_test = dataio_prepare_2(hparams, adv_test_csv, tokenizer)

        print(" ")
        print(
            "--------------------- Combined PVP + Distriblock detector Separate---------------------"
        )

        from speechbrain.decoders.ctc import CTCBeamSearcher

        test_searcher = CTCBeamSearcher(
            **hparams["test_beam_search"],
            vocab_list=vocab_list,
        )

        file_names = [
            "train_hs.pickle",
            "val_adv_hs.pickle",
            "test_hs.pickle",
            "test_adv_hs.pickle",
        ]
        data_sets = [benign_data_train, adv_data_val, benign_data_test, adv_data_test]

        characteristic_folder = hparams["distriblock_folder"]
        if not os.path.exists(characteristic_folder):
            os.makedirs(characteristic_folder)

        for i, data in enumerate(data_sets):
            if not os.path.exists(f"{characteristic_folder}/{file_names[i]}"):
                print(f"Saving characteristics in file: {file_names[i]}!")
                asr_brain.characteristics(
                    data,
                    f"{characteristic_folder}/{file_names[i]}",
                    train_loader_kwargs=hparams["test_dataloader_options"],
                    min_key="WER",
                )
        if os.path.exists(f"{characteristic_folder}/{file_names[0]}"):
            train_meas = load_meas_data(
                f"{characteristic_folder}/{file_names[0]}", benign_flg=True
            )
            keys = []
            for i in train_meas:
                keys.append(i)
        # if os.path.exists(f"{characteristic_folder}/{file_names[1]}"):
        #    val_meas = load_meas_data(f"{characteristic_folder}/{file_names[1]}", benign_flg=True)
        if os.path.exists(f"{characteristic_folder}/{file_names[1]}"):
            val_adv_meas = load_meas_data(
                f"{characteristic_folder}/{file_names[1]}", benign_flg=False
            )
        if os.path.exists(f"{characteristic_folder}/{file_names[2]}"):
            test_meas = load_meas_data(
                f"{characteristic_folder}/{file_names[2]}", benign_flg=True
            )
        if os.path.exists(f"{characteristic_folder}/{file_names[3]}"):
            test_adv_meas = load_meas_data(
                f"{characteristic_folder}/{file_names[3]}", benign_flg=False
            )

        stats_train, _ = precision_robustness_stats(
            asr_brain,
            benign_data_train,
            dataloader_opts=hparams["test_dataloader_options"],
        )
        # stats_val, _ = precision_robustness_stats(asr_brain, benign_data_val, dataloader_opts=hparams["test_dataloader_options"])
        stats_adv_val, _ = precision_robustness_stats(
            asr_brain, adv_data_val, dataloader_opts=hparams["test_dataloader_options"]
        )
        stats_test, _ = precision_robustness_stats(
            asr_brain,
            benign_data_test,
            dataloader_opts=hparams["test_dataloader_options"],
        )
        stats_adv_test, _ = precision_robustness_stats(
            asr_brain, adv_data_test, dataloader_opts=hparams["test_dataloader_options"]
        )

        train_feats, train_labels = create_feature_Matrix(stats_train, train_meas)
        # val_feats, val_labels = create_feature_Matrix(stats_val, val_meas)
        adv_feats_val, adv_labels_val = create_feature_Matrix(
            stats_adv_val, val_adv_meas
        )
        test_feats, test_labels = create_feature_Matrix(stats_test, test_meas)
        adv_test_feats, adv_test_labels = create_feature_Matrix(
            stats_adv_test, test_adv_meas
        )

        train_data = torch.cat([train_feats, adv_feats_val], dim=0)
        train_labels = torch.cat([train_labels, adv_labels_val], dim=0)
        train_labels = torch.unsqueeze(train_labels, 1)

        train_data, validate_data, train_labels, validate_labels = train_test_split(
            train_data.cpu().numpy(),
            train_labels.cpu().numpy(),
            train_size=0.8,
            stratify=train_labels.cpu().numpy(),
            random_state=seed,
        )

        train_data = torch.tensor(train_data, dtype=torch.float32)
        validate_data = torch.tensor(validate_data, dtype=torch.float32)

        train_labels = torch.tensor(train_labels, dtype=torch.float32)
        validate_labels = torch.tensor(validate_labels, dtype=torch.float32)

        # --- Model existence check: hybrid_separate Detector ---
        detector_path = Path(hparams["distriblock_folder"]) / "detector_separate"
        if detector_path.exists():
            print(
                f"Detector model found at {detector_path}. Loading and skipping training."
            )
            checkpoint = torch.load(
                detector_path, map_location=asr_brain.device, weights_only=False
            )
            model = Detector(input_dim=2, hidden_dim=8)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            best_thresh = checkpoint["threshold"]
        else:
            print(
                f"No existing detector model at {detector_path}. Training new model..."
            )
            model = Detector(input_dim=2, hidden_dim=8)
            train_detector(model, train_data, train_labels, epochs=100, lr=1e-3)

            with torch.no_grad():
                probs = torch.sigmoid(model(validate_data)).cpu().numpy()
                fpr, tpr, threshold = roc_curve(validate_labels.cpu().numpy(), probs)
                auc_score = auc(fpr, tpr)

                plt.figure()
                plt.plot(fpr, tpr, label="ROC curve (area = %0.3f)" % auc_score)
                plt.plot([0, 1], [0, 1], "k--", label="No Skill")
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend()
                plt.show()
                plt.savefig("results/auroc_hybrid", dpi=200, bbox_inches="tight")

                print("ROC-AUC:", auc_score)
                best_idx = np.argmax(tpr - fpr)
                best_thresh = threshold[best_idx]

            state = {"model": model.state_dict(), "threshold": best_thresh}
            torch.save(state, detector_path)
            print(f"Detector saved to {detector_path}")

        eval_benign_acc = eval_detector(model, test_feats, test_labels, best_thresh)
        print("Eval Acc (benign-only):", eval_benign_acc, len(test_feats))

        eval_adv_acc = eval_detector(
            model, adv_test_feats, adv_test_labels, best_thresh
        )
        print("Eval Acc (adv-only):", eval_adv_acc, len(adv_test_feats))

        eval_combined_data = torch.cat([test_feats, adv_test_feats], dim=0)
        eval_combined_labels = torch.cat([test_labels, adv_test_labels], dim=0)
        eval_combined_acc = eval_detector(
            model, eval_combined_data, eval_combined_labels, best_thresh
        )
        print("Eval Acc (benign + adv):", eval_benign_acc)

    elif hparams["adv_type"] == "hybrid_separate_gaussian":
        EXPERIMENTS = [
            {
                "name": "baseline_cw",
                "benign": hparams["train_benign"],
                "val_benign": hparams["val_benign"],
                "val_adv": hparams["val_adv"],
                "test_benign": hparams["clean_audio_clean_transcripts"],
                "test_adv": hparams["adv_audio_adv_transcripts"],
            }
        ]
        benign_csv = resolve_csv(EXPERIMENTS[0]["benign"])
        val_benign_csv = resolve_csv(EXPERIMENTS[0]["val_benign"])
        val_adv_csv = resolve_csv(EXPERIMENTS[0]["val_adv"])
        test_csv = resolve_csv(EXPERIMENTS[0]["test_benign"])
        adv_test_csv = resolve_csv(EXPERIMENTS[0]["test_adv"])

        benign_data_train = dataio_prepare_2(hparams, benign_csv, tokenizer)
        benign_data_val = dataio_prepare_2(hparams, val_benign_csv, tokenizer)
        adv_data_val = dataio_prepare_2(hparams, val_adv_csv, tokenizer)
        benign_data_test = dataio_prepare_2(hparams, test_csv, tokenizer)
        adv_data_test = dataio_prepare_2(hparams, adv_test_csv, tokenizer)

        print(" ")
        print(
            "--------------------- Combined PVP + Distriblock Gaussian detector ---------------------"
        )

        from speechbrain.decoders.ctc import CTCBeamSearcher

        test_searcher = CTCBeamSearcher(
            **hparams["test_beam_search"],
            vocab_list=vocab_list,
        )

        file_names = [
            "train_hsg.pickle",
            "val_hsg.pickle",
            "val_adv_hsg.pickle",
            "test_hsg.pickle",
            "test_adv_hsg.pickle",
        ]
        data_sets = [
            benign_data_train,
            benign_data_val,
            adv_data_val,
            benign_data_test,
            adv_data_test,
        ]

        characteristic_folder = hparams["distriblock_folder"]
        if not os.path.exists(characteristic_folder):
            os.makedirs(characteristic_folder)

        if not os.path.exists(characteristic_folder):
            os.makedirs(characteristic_folder)

        for i, data in enumerate(data_sets):
            if not os.path.exists(f"{characteristic_folder}/{file_names[i]}"):
                print(f"Saving characteristics in file: {file_names[i]}!")
                asr_brain.characteristics(
                    data,
                    f"{characteristic_folder}/{file_names[i]}",
                    train_loader_kwargs=hparams["test_dataloader_options"],
                    min_key="WER",
                )
        if os.path.exists(f"{characteristic_folder}/{file_names[0]}"):
            train_meas = load_meas_data(
                f"{characteristic_folder}/{file_names[0]}", benign_flg=True
            )
            keys = []
            for i in train_meas:
                keys.append(i)
        if os.path.exists(f"{characteristic_folder}/{file_names[1]}"):
            val_meas = load_meas_data(
                f"{characteristic_folder}/{file_names[1]}", benign_flg=True
            )
        if os.path.exists(f"{characteristic_folder}/{file_names[2]}"):
            val_adv_meas = load_meas_data(
                f"{characteristic_folder}/{file_names[2]}", benign_flg=False
            )
        if os.path.exists(f"{characteristic_folder}/{file_names[3]}"):
            test_meas = load_meas_data(
                f"{characteristic_folder}/{file_names[3]}", benign_flg=True
            )
        if os.path.exists(f"{characteristic_folder}/{file_names[4]}"):
            test_adv_meas = load_meas_data(
                f"{characteristic_folder}/{file_names[4]}", benign_flg=False
            )

        stats_train, _ = precision_robustness_stats(
            asr_brain,
            benign_data_train,
            dataloader_opts=hparams["test_dataloader_options"],
        )
        stats_val, _ = precision_robustness_stats(
            asr_brain,
            benign_data_val,
            dataloader_opts=hparams["test_dataloader_options"],
        )
        stats_adv_val, _ = precision_robustness_stats(
            asr_brain, adv_data_val, dataloader_opts=hparams["test_dataloader_options"]
        )
        stats_test, _ = precision_robustness_stats(
            asr_brain,
            benign_data_test,
            dataloader_opts=hparams["test_dataloader_options"],
        )
        stats_adv_test, _ = precision_robustness_stats(
            asr_brain, adv_data_test, dataloader_opts=hparams["test_dataloader_options"]
        )

        train_feats, train_labels = create_feature_Matrix(stats_train, train_meas)
        val_feats, val_labels = create_feature_Matrix(stats_val, val_meas)
        adv_feats_val, adv_labels_val = create_feature_Matrix(
            stats_adv_val, val_adv_meas
        )
        test_feats, test_labels = create_feature_Matrix(stats_test, test_meas)
        adv_test_feats, adv_test_labels = create_feature_Matrix(
            stats_adv_test, test_adv_meas
        )

        gaussian_path = (
            Path(hparams["distriblock_folder"]) / "gaussian_detector_separate"
        )
        if gaussian_path.exists():
            print(
                f"Gaussian model found at {gaussian_path}. Loading and skipping Gaussian fit."
            )
            state = torch.load(
                gaussian_path, map_location=asr_brain.device, weights_only=False
            )
            gaussian_mean = state["gaussian_mean"]
            gaussian_cov = state["gaussian_cov"]
            roc_auc = state["roc_auc"]
            worst_benign = state["worst_benign"]
            selected_tpr = state["tpr_at_fpr"]["tpr"]
            selected_fpr = state["tpr_at_fpr"]["fpr"]
        else:
            print(
                f"No existing Gaussian model at {gaussian_path}. Fitting new model..."
            )
            gaussian_mean = np.mean(train_feats.numpy(), axis=0)
            gaussian_cov = np.cov(train_feats.numpy(), rowvar=False)
            gaussian_cov += np.eye(2) * 1e-6

            print(
                f"2D Multivariate Gaussian fit on {len(train_feats)} benign training samples"
            )
            print(f"  Mean: {gaussian_mean}")
            print(f"  Covariance diagonal: {np.diag(gaussian_cov)}")

            val_scores = multivariate_normal.logpdf(
                torch.cat([val_feats, adv_feats_val], dim=0).numpy(),
                mean=gaussian_mean,
                cov=gaussian_cov,
            )
            val_labels = torch.cat([val_labels, adv_labels_val], dim=0)

            fpr, tpr, threshold = roc_curve(val_labels.numpy(), val_scores)
            roc_auc = auc(fpr, tpr)

            fpr = fpr[1:]
            tpr = tpr[1:]
            if len(tpr) > 0 and tpr[-1] == 1.0 and fpr[-1] == 1.0:
                fpr = fpr[:-1]
                tpr = tpr[:-1]

            fnr = 1 - tpr
            tnr = 1 - fpr
            worst_benign_idx = np.argmax(fnr)
            worst_benign = {
                "tpr": float(tpr[worst_benign_idx]),
                "tnr": float(tnr[worst_benign_idx]),
                "fpr": float(fpr[worst_benign_idx]),
                "fnr": float(fnr[worst_benign_idx]),
            }

            target_fpr = 0.01
            target_tpr = 0.95
            mask = fpr < target_fpr
            if np.any(mask):
                candidate_tpr = tpr[mask]
                candidate_fpr = fpr[mask]
                idx = np.argmin(np.abs(candidate_tpr - target_tpr))
                selected_tpr = candidate_tpr[idx]
                selected_fpr = candidate_fpr[idx]
            else:
                idx = np.argmin(np.abs(tpr - target_tpr))
                selected_tpr = tpr[idx]
                selected_fpr = fpr[idx]

            print(" ")
            print(
                "--------------------- 2D Multivariate Gaussian Classifier results: ---------------------------"
            )
            print(f"ROC-AUC: {roc_auc:.4f}")
            print(
                f"Worst benign (max FNR): TPR={worst_benign['tpr']:.4f}, "
                f"FPR={worst_benign['fpr']:.4f}, TNR={worst_benign['tnr']:.4f}, FNR={worst_benign['fnr']:.4f}"
            )
            print(f"TPR@FPR<1%: TPR={selected_tpr:.4f}, FPR={selected_fpr:.4f}")

            plt.figure()
            plt.plot(fpr, tpr, label="ROC curve (area = %0.4f)" % roc_auc)
            plt.plot([0, 1], [0, 1], "k--", label="No Skill")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.savefig(
                "results/auroc_hybrid_separate_gaussian", dpi=200, bbox_inches="tight"
            )
            plt.show()

            state = {
                "gaussian_mean": gaussian_mean,
                "gaussian_cov": gaussian_cov,
                "roc_auc": roc_auc,
                "worst_benign": worst_benign,
                "tpr_at_fpr": {"tpr": float(selected_tpr), "fpr": float(selected_fpr)},
            }
            torch.save(state, gaussian_path)
            print(f"Gaussian model saved to {gaussian_path}")

        test_benign_scores = multivariate_normal.logpdf(
            test_feats.numpy(), mean=gaussian_mean, cov=gaussian_cov
        )

        test_adv_scores = multivariate_normal.logpdf(
            adv_test_feats.numpy(), mean=gaussian_mean, cov=gaussian_cov
        )

        test_scores = np.concatenate([test_benign_scores, test_adv_scores])
        test_labels_final = np.concatenate(
            [test_labels.numpy(), adv_test_labels.numpy()]
        )

        test_fpr, test_tpr, _ = roc_curve(test_labels_final, test_scores)
        test_auc = auc(test_fpr, test_tpr)

        best_idx = np.argmax(tpr - fpr)
        best_threshold = threshold[best_idx]

        test_preds = (test_scores >= best_threshold).astype(int)

        TP = np.sum((test_preds == 1) & (test_labels_final == 1))
        TN = np.sum((test_preds == 0) & (test_labels_final == 0))
        FP = np.sum((test_preds == 1) & (test_labels_final == 0))
        FN = np.sum((test_preds == 0) & (test_labels_final == 1))

        acc = (TP + TN) / (TP + TN + FP + FN)
        tpr = TP / (TP + FN + 1e-8)
        fpr = FP / (FP + TN + 1e-8)
        precision = TP / (TP + FP + 1e-8)
        recall = tpr
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        print("\n---------------- TEST SET RESULTS ----------------")
        print(f"Test ROC-AUC: {test_auc:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"TPR: {tpr:.4f} | FPR: {fpr:.4f}")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    elif hparams["adv_type"] == "hybrid_pure":
        EXPERIMENTS = [
            {
                "name": "baseline_cw",
                "benign": hparams["train_benign"],
                # "val_benign": hparams["val_benign"],
                "val_adv": hparams["val_adv"],
                "test_benign": hparams["clean_audio_clean_transcripts"],
                "test_adv": hparams["adv_audio_adv_transcripts"],
            }
        ]
        benign_csv = resolve_csv(EXPERIMENTS[0]["benign"])
        # val_benign_csv = resolve_csv(EXPERIMENTS[0]["val_benign"])
        val_adv_csv = resolve_csv(EXPERIMENTS[0]["val_adv"])
        test_csv = resolve_csv(EXPERIMENTS[0]["test_benign"])
        adv_test_csv = resolve_csv(EXPERIMENTS[0]["test_adv"])

        benign_data_train = dataio_prepare_2(hparams, benign_csv, tokenizer)
        # benign_data_val = dataio_prepare_2(hparams, val_benign_csv, tokenizer)
        adv_data_val = dataio_prepare_2(hparams, val_adv_csv, tokenizer)
        benign_data_test = dataio_prepare_2(hparams, test_csv, tokenizer)
        adv_data_test = dataio_prepare_2(hparams, adv_test_csv, tokenizer)

        print(" ")
        print(
            "--------------------- Combined PVP + Distriblock detector Separate---------------------"
        )

        from speechbrain.decoders.ctc import CTCBeamSearcher

        test_searcher = CTCBeamSearcher(
            **hparams["test_beam_search"],
            vocab_list=vocab_list,
        )

        file_names = [
            "train_hs.pickle",
            "val_adv_hs.pickle",
            "test_hs.pickle",
            "test_adv_hs.pickle",
        ]
        data_sets = [benign_data_train, adv_data_val, benign_data_test, adv_data_test]

        characteristic_folder = hparams["distriblock_folder"]
        if not os.path.exists(characteristic_folder):
            os.makedirs(characteristic_folder)

        if not os.path.exists(characteristic_folder):
            os.makedirs(characteristic_folder)

        for i, data in enumerate(data_sets):
            if not os.path.exists(f"{characteristic_folder}/{file_names[i]}"):
                print(f"Saving characteristics in file: {file_names[i]}!")
                asr_brain.PVP_characteristics(
                    data,
                    f"{characteristic_folder}/{file_names[i]}",
                    train_loader_kwargs=hparams["test_dataloader_options"],
                    min_key="WER",
                )
        if os.path.exists(f"{characteristic_folder}/{file_names[0]}"):
            train_meas = load_meas_data_pvp(
                f"{characteristic_folder}/{file_names[0]}", benign_flg=True
            )
        if os.path.exists(f"{characteristic_folder}/{file_names[1]}"):
            adv_meas = load_meas_data_pvp(
                f"{characteristic_folder}/{file_names[1]}", benign_flg=False
            )
        if os.path.exists(f"{characteristic_folder}/{file_names[2]}"):
            test_meas = load_meas_data_pvp(
                f"{characteristic_folder}/{file_names[2]}", benign_flg=True
            )
        if os.path.exists(f"{characteristic_folder}/{file_names[3]}"):
            adv_test_meas = load_meas_data_pvp(
                f"{characteristic_folder}/{file_names[3]}", benign_flg=False
            )

        train_feats, train_labels = create_precision_feature_matrix(train_meas)
        test_feats, test_labels = create_precision_feature_matrix(test_meas)
        adv_feats, adv_labels = create_precision_feature_matrix(adv_meas)
        adv_test_feats, adv_test_labels = create_precision_feature_matrix(adv_test_meas)

        data = torch.cat([train_feats, adv_feats], dim=0)
        labels = torch.cat([train_labels, adv_labels], dim=0)
        labels = torch.unsqueeze(labels, 1)

        benign_mask = labels.squeeze() == 1.0
        adv_mask = labels.squeeze() == 0.0
        benign_indices = torch.where(benign_mask)[0]
        adv_indices = torch.where(adv_mask)[0]

        gen = torch.Generator()
        gen.manual_seed(seed)

        benign_perm = torch.randperm(len(benign_indices), generator=gen)
        adv_perm = torch.randperm(len(adv_indices), generator=gen)
        val_benign_idx = benign_indices[benign_perm[:20]]
        val_adv_idx = adv_indices[adv_perm[:20]]
        validate_idx = torch.cat([val_benign_idx, val_adv_idx])

        all_indices = torch.arange(data.shape[0])
        train_mask = torch.ones(data.shape[0], dtype=torch.bool)
        train_mask[validate_idx] = False
        train_idx = all_indices[train_mask]

        train_perm = torch.randperm(len(train_idx), generator=gen)
        train_idx = train_idx[train_perm]

        train_data = data[train_idx]
        train_labels = labels[train_idx]
        validate_data = data[validate_idx]
        validate_labels = labels[validate_idx]

        # --- Model existence check: hybrid_pure Detector ---
        detector_path = Path(hparams["distriblock_folder"]) / "detector_pure"
        if detector_path.exists():
            print(
                f"Detector model found at {detector_path}. Loading and skipping training."
            )
            checkpoint = torch.load(detector_path, map_location=asr_brain.device)
            model = Detector(input_dim=3, hidden_dim=8)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            best_thresh = checkpoint["threshold"]
        else:
            print(
                f"No existing detector model at {detector_path}. Training new model..."
            )
            model = Detector(input_dim=3, hidden_dim=8)
            train_detector(model, train_data, train_labels, epochs=100, lr=1e-3)

            with torch.no_grad():
                probs = torch.sigmoid(model(validate_data)).cpu().numpy()
                fpr, tpr, threshold = roc_curve(validate_labels.cpu().numpy(), probs)
                auc_score = auc(fpr, tpr)

                plt.figure()
                plt.plot(fpr, tpr, label="ROC curve (area = %0.3f)" % auc_score)
                plt.plot([0, 1], [0, 1], "k--", label="No Skill")
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend()
                plt.show()
                plt.savefig("results/auroc_hybrid", dpi=200, bbox_inches="tight")

                print("ROC-AUC:", auc_score)
                best_idx = np.argmax(tpr - fpr)
                best_thresh = threshold[best_idx]

            state = {"model": model.state_dict(), "threshold": best_thresh}
            torch.save(state, detector_path)
            print(f"Detector saved to {detector_path}")

        test_feats = torch.cat([adv_test_feats, test_feats])
        test_labels = torch.cat([adv_test_labels, test_labels])
        print(len(test_feats), len(test_labels))
        test_labels = torch.unsqueeze(test_labels, 1)
        gen2 = torch.Generator()
        gen2.manual_seed(seed)
        indices = torch.randperm(test_feats.shape[0], generator=gen2)
        test_data = test_feats[indices]
        test_labels = test_labels[indices]
        eval_acc = eval_detector(model, test_data, test_labels, best_thresh)
        print("Eval Acc:", eval_acc)

    elif hparams["adv_type"] == "hybrid_pure_gaussian":
        EXPERIMENTS = [
            {
                "name": "baseline_cw",
                "benign": hparams["train_benign"],
                "val_benign": hparams["val_benign"],
                "val_adv": hparams["val_adv"],
                "test_benign": hparams["clean_audio_clean_transcripts"],
                "test_adv": hparams["adv_audio_adv_transcripts"],
            }
        ]
        benign_csv = resolve_csv(EXPERIMENTS[0]["benign"])
        val_benign_csv = resolve_csv(EXPERIMENTS[0]["val_benign"])
        val_adv_csv = resolve_csv(EXPERIMENTS[0]["val_adv"])
        test_csv = resolve_csv(EXPERIMENTS[0]["test_benign"])
        adv_test_csv = resolve_csv(EXPERIMENTS[0]["test_adv"])

        benign_data_train = dataio_prepare_2(hparams, benign_csv, tokenizer)
        benign_data_val = dataio_prepare_2(hparams, val_benign_csv, tokenizer)
        adv_data_val = dataio_prepare_2(hparams, val_adv_csv, tokenizer)
        benign_data_test = dataio_prepare_2(hparams, test_csv, tokenizer)
        adv_data_test = dataio_prepare_2(hparams, adv_test_csv, tokenizer)

        print(" ")
        print(
            "--------------------- PVP + Multivariate Gaussian detector ---------------------"
        )

        from speechbrain.decoders.ctc import CTCBeamSearcher

        test_searcher = CTCBeamSearcher(
            **hparams["test_beam_search"],
            vocab_list=vocab_list,
        )

        file_names = [
            "train_hs.pickle",
            "val_hs.pickle",
            "val_adv_hs.pickle",
            "test_hs.pickle",
            "test_adv_hs.pickle",
        ]
        data_sets = [
            benign_data_train,
            benign_data_val,
            adv_data_val,
            benign_data_test,
            adv_data_test,
        ]

        characteristic_folder = hparams["distriblock_folder"]
        if not os.path.exists(characteristic_folder):
            os.makedirs(characteristic_folder)

        if not os.path.exists(characteristic_folder):
            os.makedirs(characteristic_folder)

        for i, data in enumerate(data_sets):
            if not os.path.exists(f"{characteristic_folder}/{file_names[i]}"):
                print(f"Saving characteristics in file: {file_names[i]}!")
                asr_brain.PVP_characteristics(
                    data,
                    f"{characteristic_folder}/{file_names[i]}",
                    train_loader_kwargs=hparams["test_dataloader_options"],
                    min_key="WER",
                )
        if os.path.exists(f"{characteristic_folder}/{file_names[0]}"):
            train_meas = load_meas_data_pvp(
                f"{characteristic_folder}/{file_names[0]}", benign_flg=True
            )
            keys = []
            for i in train_meas:
                keys.append(i)
        if os.path.exists(f"{characteristic_folder}/{file_names[1]}"):
            val_meas = load_meas_data_pvp(
                f"{characteristic_folder}/{file_names[1]}", benign_flg=True
            )
        if os.path.exists(f"{characteristic_folder}/{file_names[2]}"):
            val_adv_meas = load_meas_data_pvp(
                f"{characteristic_folder}/{file_names[2]}", benign_flg=False
            )
        if os.path.exists(f"{characteristic_folder}/{file_names[3]}"):
            test_meas = load_meas_data_pvp(
                f"{characteristic_folder}/{file_names[3]}", benign_flg=True
            )
        if os.path.exists(f"{characteristic_folder}/{file_names[4]}"):
            test_adv_meas = load_meas_data_pvp(
                f"{characteristic_folder}/{file_names[4]}", benign_flg=False
            )

        train_feats, train_labels = create_precision_feature_matrix(train_meas)
        val_feats, val_labels = create_precision_feature_matrix(val_meas)
        val_adv_feats, val_adv_labels = create_precision_feature_matrix(val_adv_meas)
        test_feats, test_labels = create_precision_feature_matrix(test_meas)
        test_adv_feats, test_adv_labels = create_precision_feature_matrix(test_adv_meas)

        benign_mask = train_labels == 1.0
        benign_feats = train_feats[benign_mask].numpy()

        gaussian_path = Path(hparams["distriblock_folder"]) / "gaussian_detector_pure"
        if gaussian_path.exists():
            print(
                f"Gaussian model found at {gaussian_path}. Loading and skipping Gaussian fit."
            )
            state = torch.load(gaussian_path, map_location=asr_brain.device)
            gaussian_mean = state["gaussian_mean"]
            gaussian_cov = state["gaussian_cov"]
            roc_auc = state["roc_auc"]
            worst_benign = state["worst_benign"]
            selected_tpr = state["tpr_at_fpr"]["tpr"]
            selected_fpr = state["tpr_at_fpr"]["fpr"]
        else:
            print(
                f"No existing Gaussian model at {gaussian_path}. Fitting new model..."
            )
            gaussian_mean = np.mean(benign_feats, axis=0)
            gaussian_cov = np.cov(benign_feats, rowvar=False)
            gaussian_cov += np.eye(3) * 1e-6

            print(
                f"Multivariate Gaussian fit on {len(benign_feats)} benign training samples"
            )
            print(f"  Mean: {gaussian_mean}")
            print(f"  Covariance diagonal: {np.diag(gaussian_cov)}")

            train_scores = multivariate_normal.logpdf(
                train_feats.numpy(), mean=gaussian_mean, cov=gaussian_cov
            )
            val_scores = multivariate_normal.logpdf(
                torch.cat([val_feats, val_adv_feats], dim=0).numpy(),
                mean=gaussian_mean,
                cov=gaussian_cov,
            )
            val_labels = torch.cat([val_labels, val_adv_labels], dim=0)

            fpr, tpr, threshold = roc_curve(val_labels.numpy(), val_scores)
            roc_auc = auc(fpr, tpr)

            fpr = fpr[1:]
            tpr = tpr[1:]
            if len(tpr) > 0 and tpr[-1] == 1.0 and fpr[-1] == 1.0:
                fpr = fpr[:-1]
                tpr = tpr[:-1]

            fnr = 1 - tpr
            tnr = 1 - fpr
            worst_benign_idx = np.argmax(fnr)
            worst_benign = {
                "tpr": float(tpr[worst_benign_idx]),
                "tnr": float(tnr[worst_benign_idx]),
                "fpr": float(fpr[worst_benign_idx]),
                "fnr": float(fnr[worst_benign_idx]),
            }

            target_fpr = 0.01
            target_tpr = 0.95
            mask = fpr < target_fpr
            if np.any(mask):
                candidate_tpr = tpr[mask]
                candidate_fpr = fpr[mask]
                idx = np.argmin(np.abs(candidate_tpr - target_tpr))
                selected_tpr = candidate_tpr[idx]
                selected_fpr = candidate_fpr[idx]
            else:
                idx = np.argmin(np.abs(tpr - target_tpr))
                selected_tpr = tpr[idx]
                selected_fpr = fpr[idx]

            print(" ")
            print(
                "--------------------- Multivariate Gaussian Classifier results: ---------------------------"
            )
            print(f"ROC-AUC: {roc_auc:.4f}")
            print(
                f"Worst benign (max FNR): TPR={worst_benign['tpr']:.4f}, "
                f"FPR={worst_benign['fpr']:.4f}, TNR={worst_benign['tnr']:.4f}, FNR={worst_benign['fnr']:.4f}"
            )
            print(f"TPR@FPR<1%: TPR={selected_tpr:.4f}, FPR={selected_fpr:.4f}")

            plt.figure()
            plt.plot(fpr, tpr, label="ROC curve (area = %0.4f)" % roc_auc)
            plt.plot([0, 1], [0, 1], "k--", label="No Skill")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.savefig(
                "results/auroc_hybrid_pure_gaussian", dpi=200, bbox_inches="tight"
            )
            plt.show()

            state = {
                "gaussian_mean": gaussian_mean,
                "gaussian_cov": gaussian_cov,
                "roc_auc": roc_auc,
                "worst_benign": worst_benign,
                "tpr_at_fpr": {"tpr": float(selected_tpr), "fpr": float(selected_fpr)},
            }
            torch.save(state, gaussian_path)
            print(f"Gaussian model saved to {gaussian_path}")

            test_benign_scores = multivariate_normal.logpdf(
                test_feats.numpy(), mean=gaussian_mean, cov=gaussian_cov
            )

        test_adv_scores = multivariate_normal.logpdf(
            test_adv_feats.numpy(), mean=gaussian_mean, cov=gaussian_cov
        )

        test_scores = np.concatenate([test_benign_scores, test_adv_scores])
        test_labels_final = np.concatenate(
            [test_labels.numpy(), test_adv_labels.numpy()]
        )

        test_fpr, test_tpr, _ = roc_curve(test_labels_final, test_scores)
        test_auc = auc(test_fpr, test_tpr)

        best_idx = np.argmax(tpr - fpr)
        best_threshold = threshold[best_idx]

        test_preds = (test_scores >= best_threshold).astype(int)

        TP = np.sum((test_preds == 1) & (test_labels_final == 1))
        TN = np.sum((test_preds == 0) & (test_labels_final == 0))
        FP = np.sum((test_preds == 1) & (test_labels_final == 0))
        FN = np.sum((test_preds == 0) & (test_labels_final == 1))

        acc = (TP + TN) / (TP + TN + FP + FN)
        tpr = TP / (TP + FN + 1e-8)
        fpr = FP / (FP + TN + 1e-8)
        precision = TP / (TP + FP + 1e-8)
        recall = tpr
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        print("\n---------------- TEST SET RESULTS ----------------")
        print(f"Test ROC-AUC: {test_auc:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"TPR: {tpr:.4f} | FPR: {fpr:.4f}")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    else:
        EXPERIMENTS = [
            {
                "name": "baseline_cw",
                "benign": hparams["train_benign"],
                "adv": hparams["adv_audio_adv_transcripts"],
                "test": hparams["clean_audio_clean_transcripts"],
            }
        ]
        for exp in EXPERIMENTS:
            print(f"\n===== Running experiment: {exp['name']} =====")

            benign_csv = resolve_csv(exp["benign"])
            adv_csv = resolve_csv(exp["adv"])
            test_csv = resolve_csv(exp["test"])

            # # here we create the datasets objects as well as tokenization and encoding
            benign_data_test = dataio_prepare_2(hparams, benign_csv, tokenizer)
            adv_data = dataio_prepare_2(hparams, adv_csv, tokenizer)
            test_data = dataio_prepare_2(hparams, test_csv, tokenizer)

            # We dynamically add the tokenizer to our brain class.
            # NB: This tokenizer corresponds to the one used for the LM!!

            from speechbrain.decoders.ctc import CTCBeamSearcher

            test_searcher = CTCBeamSearcher(
                **hparams["test_beam_search"],
                vocab_list=vocab_list,
            )

            stats_gaussian, aggregated_gaussian = precision_robustness_stats(
                asr_brain,
                benign_data_test,
                dataloader_opts=hparams["test_dataloader_options"],
            )

            stats_test, aggregated_test = precision_robustness_stats(
                asr_brain,
                test_data,
                dataloader_opts=hparams["test_dataloader_options"],
            )

            stats_adv, aggregated_adv = precision_robustness_stats(
                asr_brain,
                adv_data,
                dataloader_opts=hparams["test_dataloader_options"],
            )

            auc_results = all_metrics(
                stats_gaussian=stats_gaussian,
                stats_test=stats_test,
                stats_adv=stats_adv,
            )
            # roc_auc, min_fnr, corresponding_tpr, corresponding_fpr
            # print(auc_results)

            agg = "mean"

            wer_vals = auc_results["WER"][agg]
            cer_vals = auc_results["CER"][agg]

            # AUROC
            wer_auc = wer_vals["roc_auc"]
            cer_auc = cer_vals["roc_auc"]

            # Worst benign
            wer_wb = wer_vals["worst_benign"]
            cer_wb = cer_vals["worst_benign"]

            wer_at1 = wer_vals["tpr_at_fpr"]
            cer_at1 = cer_vals["tpr_at_fpr"]

            print(
                f"[{exp['name']}] "
                f"AUROC={wer_auc:.4f}/{cer_auc:.4f} | "
                f"WorstBenign("
                f"TPR={wer_wb['tpr']:.4f}/{cer_wb['tpr']:.4f}, "
                f"FPR={wer_wb['fpr']:.4f}/{cer_wb['fpr']:.4f}, "
                f"TNR={wer_wb['tnr']:.4f}/{cer_wb['tnr']:.4f}, "
                f"FNR={wer_wb['fnr']:.4f}/{cer_wb['fnr']:.4f}"
                f") | "
                f"TPR@FPR<1%("
                f"TPR={wer_at1['tpr']:.4f}/{cer_at1['tpr']:.4f}, "
                f"FPR={wer_at1['fpr']:.4f}/{cer_at1['fpr']:.4f}, "
                f"TNR={wer_at1['tnr']:.4f}/{cer_at1['tnr']:.4f}, "
                f"FNR={wer_at1['fnr']:.4f}/{cer_at1['fnr']:.4f}"
                f")"
            )

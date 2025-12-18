import os
import random
import re
import unicodedata
import logging

import torch
import torch.utils.data

import commons
from mel_processing import spectrogram_torch, spec_to_mel_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cleaned_text_to_sequence

logger = logging.getLogger(__name__)

# ----------------------------
# Text cleaning
# ----------------------------
def clean_tamil_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)

    replacements = {
        "\u201c": '"', "\u201d": '"',
        "\u2018": "'", "\u2019": "'",
        "\u2013": "-", "\u2014": "-",
        "\u2026": "...",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # remove zero-width chars
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================
# MULTI-SPEAKER DATASET
# ============================
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    Metadata format:
    wav_path | text | speaker_id
    """

    def __init__(self, metadata_path, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(metadata_path)

        # -------- Audio params --------
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length

        # -------- Mel params --------
        # self.n_mel_channels = hparams.n_mel_channels
        # self.mel_fmin = hparams.mel_fmin
        # self.mel_fmax = hparams.mel_fmax

        # -------- Text params --------
        self.text_cleaners = hparams.text_cleaners
        self.cleaned_text = getattr(hparams, "cleaned_text", False)
        self.add_blank = hparams.add_blank
        self.min_text_len = hparams.min_text_len
        self.max_text_len = hparams.max_text_len

        # -------- Training guards --------
        self.min_training_frames = getattr(hparams, "min_training_frames", 80)

        # -------- Speaker mapping (STABLE) --------
        speaker_ids = sorted(set(str(x[2]) for x in self.audiopaths_sid_text))
        self.speaker_map = {spk: i for i, spk in enumerate(speaker_ids)}

        # random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)

        self._filter()

        if len(self.audiopaths_sid_text) == 0:
            raise RuntimeError("All audio files were filtered out. Check dataset.")

        logger.info("TRAIN DATA LOADED — %d samples", len(self.audiopaths_sid_text))

    # ---------------------------
    # Filtering
    # ---------------------------
    def _filter(self):
        filtered = []
        for wav, text, spk in self.audiopaths_sid_text:
            if not os.path.exists(wav):
                continue
            if not (self.min_text_len <= len(text) <= self.max_text_len):
                continue
            try:
                audio, sr = load_wav_to_torch(wav)
                if sr == self.sampling_rate and audio.numel() > self.filter_length:
                    filtered.append([wav, text, spk])
            except:
                pass
        self.audiopaths_sid_text = filtered

    # ---------------------------
    # Dataset interface
    # ---------------------------
    def __len__(self):
        return len(self.audiopaths_sid_text)

    def __getitem__(self, idx):
        wav_path, text, spk = self.audiopaths_sid_text[idx]

        sid = torch.LongTensor([self.speaker_map[str(spk)]])
        text = self.get_text(text)

        spec, wav = self.get_audio(wav_path)
        if spec is None:
            return None

        return (
            text,
            torch.LongTensor([text.size(0)]),
            spec,
            torch.LongTensor([spec.size(1)]),
            wav,
            torch.LongTensor([wav.size(1)]),
            sid,
        )


    # ---------------------------
    # Audio → Mel
    # ---------------------------
    def get_audio(self, filename):
        try:
            audio, sr = load_wav_to_torch(filename)
            if sr != self.sampling_rate:
                return None, None

            if audio.dim() == 2:
                audio = audio.mean(dim=0)

            wav = audio.unsqueeze(0).float() / self.max_wav_value

            spec = spectrogram_torch(
                wav,
                self.filter_length,
                self.sampling_rate,
                self.hop_length,
                self.win_length,
                center=False,
            )

            spec = torch.abs(spec).squeeze(0)  # [513, T]

            if spec.size(1) < self.min_training_frames:
                return None, None

            return spec, wav

        except Exception as e:
            logger.warning("Audio load failed %s: %s", filename, e)
            return None, None



    # ---------------------------
    # Text → IDs
    # ---------------------------
    def get_text(self, text):
        text = clean_tamil_text(text)

        if self.cleaned_text:
            seq = cleaned_text_to_sequence(text)
        else:
            seq = text_to_sequence(text, self.text_cleaners)

        if self.add_blank:
            seq = commons.intersperse(seq, 0)

        return torch.LongTensor(seq)


# ============================
# Collate
# ============================
class TextAudioSpeakerCollate:
    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None

        texts, text_lens, mels, mel_lens, wavs, wav_lens, sids = zip(*batch)

        max_text = max(t.size(0) for t in texts)
        max_mel = max(m.size(1) for m in mels)
        max_wav = max(w.size(1) for w in wavs)

        text_pad = torch.zeros(len(batch), max_text, dtype=torch.long)
        mel_pad = torch.zeros(len(batch), mels[0].size(0), max_mel)
        wav_pad = torch.zeros(len(batch), 1, max_wav)

        for i in range(len(batch)):
            text_pad[i, :texts[i].size(0)] = texts[i]
            mel_pad[i, :, :mels[i].size(1)] = mels[i]
            wav_pad[i, 0, :wavs[i].size(1)] = wavs[i]

        return (
            text_pad,
            torch.cat(text_lens),
            mel_pad,
            torch.cat(mel_lens),
            wav_pad,
            torch.cat(wav_lens),
            torch.cat(sids),
        )

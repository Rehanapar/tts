import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import logging
import sys
from mel_processing import spectrogram_torch, mel_spectrogram_torch

import commons
import utils
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn, MultiPeriodDiscriminator
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
import torch
torch.cuda.empty_cache()
# torch.backends.cudnn.benchmark = True


torch.backends.cudnn.benchmark = True
global_step = 0


# ---------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------
def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


# ---------------------------------------------------------------
# CREATE GENERATOR
# ---------------------------------------------------------------
def create_synthesizer(hps, device):
    spec_channels = hps.data.filter_length // 2 + 1  # 513 for 1024 FFT

    #segment size in frames
    segment_size = hps.train.segment_size // hps.data.hop_length
    net = SynthesizerTrn(
        hps.model.n_vocab,
        spec_channels,
        segment_size,
        hps.model.inter_channels,
        hps.model.hidden_channels,
        hps.model.filter_channels,
        hps.model.n_heads,
        hps.model.n_layers,
        hps.model.kernel_size,
        hps.model.p_dropout,
        hps.model.resblock,
        hps.model.resblock_kernel_sizes,
        hps.model.resblock_dilation_sizes,
        hps.model.upsample_rates,
        hps.model.upsample_initial_channel,
        hps.model.upsample_kernel_sizes,
        n_speakers=hps.model.n_speakers,
        gin_channels=hps.model.gin_channels,
        use_sdp=hps.model.use_sdp,
    )
    return net.to(device)


# ---------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------
def run(hps, model_dir):
    global global_step
    setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "ERROR: VITS requires CUDA GPU."

    os.makedirs(model_dir, exist_ok=True)
    logger = utils.get_logger(model_dir)
    logger.info(f"MODEL DIR = {os.path.abspath(model_dir)}")
    logger.info(hps)

    writer = SummaryWriter(model_dir)

    # -----------------------------------------------------------
    # DATASET
    # -----------------------------------------------------------
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    collate = TextAudioSpeakerCollate()

    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        shuffle=True,
        num_workers=hps.data.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
    )

    steps_per_epoch = len(train_loader)
    logger.info(f"TRAIN DATA LOADED — {steps_per_epoch} batches per epoch")

    # if steps_per_epoch == 0:
    #     logger.error("ERROR: Dataset empty or filtering removed all items.")
    #     return

    # -----------------------------------------------------------
    # MODELS
    # -----------------------------------------------------------
    net_g = create_synthesizer(hps, device)
    net_d = MultiPeriodDiscriminator(
        getattr(hps.model, "use_spectral_norm", False)
    ).to(device)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        lr=hps.train.learning_rate,
        betas=tuple(hps.train.betas),
        eps=hps.train.eps,
    )

    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        lr=hps.train.learning_rate,
        betas=tuple(hps.train.betas),
        eps=hps.train.eps,
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    # -----------------------------------------------------------
    # RESUME TRAINING
    # -----------------------------------------------------------
    latest_g = utils.latest_checkpoint_path(model_dir, "G_*.pth")
    latest_d = utils.latest_checkpoint_path(model_dir, "D_*.pth")
    epoch_start = 1
    global_step = 0

    if latest_g and latest_d:
        _, _, _, global_step = utils.load_checkpoint(latest_g, net_g, optim_g)
        utils.load_checkpoint(latest_d, net_d, optim_d)
        epoch_start = global_step // steps_per_epoch + 1
        logger.info(f"RESUMED @ step {global_step}")
   

    # -----------------------------------------------------------
    # SCHEDULERS
    # -----------------------------------------------------------
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay)

    # -----------------------------------------------------------
    # TRAIN LOOP
    # -----------------------------------------------------------
    for epoch in range(epoch_start, hps.train.epochs + 1):
        net_g.train()
        net_d.train()

        for batch in train_loader:
            if batch is None:
                continue

            x, x_l, spec, spec_l, y, y_l, sid = batch
            x, x_l, spec, spec_l, y, sid = [
                t.to(device) for t in (x, x_l, spec, spec_l, y, sid)
            ]
            # ======================= DISCRIMINATOR =========================
            # spec = spectrogram_torch(y, hps.data.filter_length, hps.data.sampling_rate,
            #              hps.data.hop_length, hps.data.win_length, center=False, device=device)

            # spec_l = spectrogram_torch(y_l, hps.data.filter_length, hps.data.sampling_rate,
            #                         hps.data.hop_length, hps.data.win_length, center=False, device=device)

            spec = spectrogram_torch(
                y,                              # waveform [B, 1, T]
                hps.data.filter_length,         # 1024
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                center=False,
                device=device
            )
            
            if global_step == 0:
                print("DEBUG SHAPES")
                print("spec shape:", spec.shape)
                print("expected: [B, 513, T]")

            with autocast(hps.train.fp16_run):
                (
                    y_hat,
                    l_length,
                    _,
                    ids_slice,
                    _,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(x, x_l, spec, spec_l, sid)

                y_mel = mel_spectrogram_torch(
                    y.squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )

                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )

                y_slice = commons.slice_segments(
                    y,
                    ids_slice * hps.data.hop_length,
                    hps.train.segment_size,
                )

                y_real, y_fake, _, _ = net_d(y_slice, y_hat.detach())
                loss_d, _, _ = discriminator_loss(y_real, y_fake)

            optim_d.zero_grad()
            scaler.scale(loss_d).backward()
            scaler.step(optim_d)

            # ======================= GENERATOR =============================
            with autocast(hps.train.fp16_run):
                y_real, y_fake, fmap_r, fmap_f = net_d(y_slice, y_hat)

                loss_mel = torch.nn.functional.l1_loss(
                    y_mel[..., : y_hat_mel.size(2)], y_hat_mel
                )
                loss_klv = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
                loss_fm = feature_loss(fmap_r, fmap_f)
                loss_g, _ = generator_loss(y_fake)

                loss_g_total = (
                    loss_g + loss_fm + loss_mel + loss_klv + l_length.sum()
                )

            optim_g.zero_grad()
            scaler.scale(loss_g_total).backward()
            scaler.step(optim_g)
            scaler.update()

            global_step += 1

            # -----------------------------------------------------------
            # CHECKPOINT SAVE
            # -----------------------------------------------------------
            if global_step % hps.train.save_every_steps == 0:
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    optim_g.param_groups[0]["lr"],
                    global_step,
                    os.path.join(model_dir, f"G_{global_step}.pth"),
                )
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    optim_d.param_groups[0]["lr"],
                    global_step,
                    os.path.join(model_dir, f"D_{global_step}.pth"),
                )

        scheduler_g.step()
        scheduler_d.step()
        logger.info(f"EPOCH {epoch} COMPLETE")

    logger.info("TRAINING COMPLETE")
    writer.close()


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-m", "--model_dir", required=True)
    args = parser.parse_args()

    hps = utils.get_hparams_from_file(args.config)
    run(hps, args.model_dir)


if __name__ == "__main__":
    main()


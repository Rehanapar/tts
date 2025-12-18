import argparse
import torch
import soundfile as sf

from models import SynthesizerTrn
from text import text_to_sequence
from utils import get_hparams_from_file
from commons import intersperse


def synthesize(hps, checkpoint_path, text, speaker_id, out_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    net_g = SynthesizerTrn(
        hps.model.n_vocab,
        hps.model.spec_channels,
        hps.model.segment_size,
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
        n_speakers=hps.data.n_speakers,
        gin_channels=hps.model.gin_channels,
        use_sdp=hps.model.use_sdp
    ).to(device)

    net_g.eval()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net_g.load_state_dict(checkpoint["model"], strict=True)

    # Text processing (Tamil)
    text_norm = text_to_sequence(text, hps.data.text_cleaners)

    if hps.data.add_blank:
        text_norm = intersperse(text_norm, 0)

    text_tensor = torch.LongTensor(text_norm).unsqueeze(0).to(device)
    text_lengths = torch.LongTensor([text_tensor.size(1)]).to(device)

    # Speaker ID
    sid = torch.LongTensor([speaker_id]).to(device)

    with torch.no_grad():
        audio = net_g.infer(
            text_tensor,
            text_lengths,
            sid=sid,
            noise_scale=0.667,
            noise_scale_w=1.0,
            length_scale=1.0
        )[0][0, 0].cpu().numpy()

    peak = max(abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9
    print("Audio min/max:", audio.min(), audio.max())

    sf.write(out_path, audio, hps.data.sampling_rate)
    print(f"Saved audio to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker_id", type=int, default=0)
    parser.add_argument("--out", default="output.wav")

    args = parser.parse_args()
    hps = get_hparams_from_file(args.config)

    synthesize(
        hps,
        args.checkpoint,
        args.text,
        args.speaker_id,
        args.out
    )


if __name__ == "__main__":
    main()

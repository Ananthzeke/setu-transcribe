import time
import glob
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

class IndicASRMultiNeMo:

    def __init__(self, checkpoint: str, device: str = "cpu", language: str = None, trainer=None):
        self.trainer = trainer
        self.language = language
        self.model = self._load_model(checkpoint, device)

    def _load_model(self, checkpoint: str, device: str = "cpu"):
        model = EncDecHybridRNNTCTCBPEModel.restore_from(checkpoint, map_location=device,trainer=self.trainer)
        model.freeze()
        return model

    def transcribe(self, audio_filepaths: list, batch_size: int = 32, num_workers: int = 4):
        print(f"Transcribing {len(audio_filepaths)} files...")
        transcripts = self.model.transcribe(audio_filepaths, batch_size=batch_size, num_workers=num_workers,language_id=self.language )
        return transcripts
    
    def get_ctc_logits(self, audio_filepaths: list, batch_size: int = 32, num_workers: int = 4):
        self.model.cur_decoder = "ctc"
        logits = self.model.transcribe(audio_filepaths, batch_size=batch_size, logprobs=True, num_workers=num_workers, )
        return logits

if __name__ == "__main__":

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="models/ai4b_pratinidhi_hi.nemo",
        required=True,
        help="Path to .nemo file",
    )
    parser.add_argument(
        "-f",
        "--audio_filepaths",
        type=str,
        default=None,
        required=True,
        help="Glob pattern for audio filepaths",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        required=True,
        help="Device (cpu/gpu)",
    )
    parser.add_argument(
        "-l",
        "--language_code",
        type=str,
        required=True,
        help="Language Code (eg. hi)",
    )
    parser.add_argument(
        "-g",
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "-w",
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for transcription",
    )
    args = parser.parse_args()

    # Use glob to find audio file paths
    audio_filepaths = glob.glob(args.audio_filepaths)
    if not audio_filepaths:
        print(f"No audio files found matching the pattern: {args.audio_filepaths}")
        exit()

    print(f"Found {len(audio_filepaths)} audio files.")

    # Set up the trainer
    trainer_kwargs = {
        'devices': args.gpus,
        'accelerator': 'gpu' if args.device == 'cuda' else 'cpu',
    }
    if args.gpus > 1:
        trainer_kwargs['strategy'] = 'ddp'
    else:
        trainer_kwargs['strategy'] = 'auto'

    trainer = Trainer(**trainer_kwargs)

    # Load the model
    print("Loading model..")
    model = IndicASRMultiNeMo(args.checkpoint, args.device, args.language_code, trainer)
    print("Transcribing..")
    t_start = time.time()
    output = model.transcribe(audio_filepaths, batch_size=args.batch_size, num_workers=args.num_workers)
    logits = model.get_ctc_logits(audio_filepaths, batch_size=args.batch_size, num_workers=args.num_workers)
    t_end = time.time()

    # Check if the output is correct
    if not output:
        print("No transcriptions were returned.")
    else:
        print("Transcriptions received.")

    print("-" * 10)
    for i, transcript in enumerate(output):
        print(f"Transcript {i+1}: {transcript}")
    print("Logits shape:", [logit.shape for logit in logits])
    print(f"Took {t_end-t_start:.2f} seconds.")
    print("-" * 10)

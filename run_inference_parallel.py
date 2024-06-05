import os
import time
import glob
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from datasets import Dataset
import logging
logging.disable(logging.CRITICAL)

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
    
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="transcriptions/",
        help="Output path for transcriptions",
    )
    args = parser.parse_args()

    # Use glob to find audio file paths
    audio_filepaths = sorted(set(glob.glob(args.audio_filepaths)))

    if not audio_filepaths:
        print(f"No audio files found matching the pattern: {args.audio_filepaths}")
        exit()

    print(f"Found {len(audio_filepaths)} audio files.")
    ds=Dataset.from_dict({'audio_filepaths':audio_filepaths})

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
    t_start = time.time()
    def transcribe_batch(batch):
        transcriptions = model.transcribe(batch['audio_filepaths'], batch_size=args.batch_size, num_workers=args.num_workers)
        return {'transcriptions': transcriptions[0]}

    ds = ds.map(transcribe_batch, batched=True, batch_size=args.batch_size, desc=f"Transcribing {len(audio_filepaths)} files")

    def group_and_join_text(file_text_dict):
        paths=list({os.path.dirname(path) for path in file_text_dict['audio_filepaths']})
        new_text=[]
        for path in paths:
            joined_text=""
            for file_path,text in zip(file_text_dict['audio_filepaths'],file_text_dict['transcriptions']):
                if file_path.startswith(path):
                    joined_text+=text+" "
            new_text.append(joined_text)
        return {'audio_path':paths,'transcription':new_text}
    ds=ds.map(group_and_join_text, batched=True, batch_size=ds.num_rows,remove_columns=ds.column_names)
    ds.save_to_disk(args.output_path,num_proc=args.num_workers)
    t_end = time.time()

    print(ds)
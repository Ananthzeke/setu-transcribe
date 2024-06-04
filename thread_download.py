import os
import time
import argparse
import subprocess
import shutil
from tqdm import tqdm

# Argument parser
parser = argparse.ArgumentParser(description="Download audio from YouTube videos")
parser.add_argument("lang", type=str, help="Language identifier")
parser.add_argument("n_lines", type=int, help="Number of lines to process")
parser.add_argument("--threads", type=int, default=4, help="Number of threads to use for downloading")
args = parser.parse_args()

lang = args.lang
n_lines = args.n_lines
threads = args.threads

try:
    print(lang, '...........')
    root = 'datasets/raw/' + lang + '/'
    save_dir = root + 'wavs/'
    os.makedirs(save_dir, exist_ok=True)
    txt_path = 'vids_from_wavs_mahadhwani_hindi.txt'
    dwn_path = root + 'downloaded_new.txt'
    st = time.time()

    # Count the number of lines in the txt_path for the total progress bar
    total_lines = sum(1 for _ in open(txt_path))
    lines_to_process = min(total_lines, n_lines)

    # Check if yt-dlp is installed
    if not shutil.which("yt-dlp"):
        raise FileNotFoundError("yt-dlp is not installed or not found in the system PATH.")

    with tqdm(total=lines_to_process, desc=f"num_proc({threads})", unit=" video") as pbar:
        cmd = (
            f"head -n {lines_to_process} {txt_path} | xargs -I '{{}}' -P {threads} "
            f"yt-dlp -f \"bestaudio/best\" -ciw -o "
            f"{save_dir}\"%(id)s.%(ext)s\" --extract-audio --audio-format wav "
            f"--download-archive {dwn_path} --audio-quality 0 --no-playlist "
            "https://youtu.be/{} --no-abort-on-error --ppa "
            "\"ffmpeg:-ac 1 -ar 16000\""
        )

        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Update the progress bar as lines are processed
        for _ in process.stdout:
            pbar.update(1)  # Update the progress bar

        process.stdout.close()
        process.stderr.close()
        process.wait()
        pbar.close()  # Close the progress bar explicitly

    et = time.time()
    print("Time taken: ", round((et - st) * 1.0 / 3600, 2), " hours")

except FileNotFoundError as fnf_error:
    print(f"Error: {fnf_error}. Please check if the file {txt_path} exists or if yt-dlp is installed.")
except Exception as e:
    print(f"An error occurred: {e}")

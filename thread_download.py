import os
import time
import sys
from tqdm import tqdm

lang = sys.argv[1]
print(lang,'...........')
root = 'datasets/raw/'+lang+'/'
threads = 4
save_dir = root + 'wavs/'
os.makedirs(save_dir, exist_ok=True) 
txt_path = 'vids_from_wavs_mahadhwani_hindi.txt'
dwn_path = root+'downloaded_new.txt'
st = time.time()

# Count the number of lines in the txt_path for the total progress bar
total_lines = sum(1 for _ in open(txt_path))

with tqdm(total=total_lines, desc="Downloading", unit=" video") as pbar:
    cmd = (
        f"cat  {txt_path} | xargs -I '{{}}' -P {threads} "
        f"yt-dlp -f \"bestaudio/best\" -ciw -o "
        f"{save_dir}\"%(id)s.%(ext)s\" --extract-audio --audio-format wav "
        f"--download-archive {dwn_path} --audio-quality 0 --no-playlist "
        "https://youtu.be/{} --no-abort-on-error --ppa "
        "\"ffmpeg:-ac 1 -ar 16000\""
    )

    process = os.popen(cmd)

    # Update the progress bar as lines are processed
    for _ in process:
        pbar.update(1)  # Update the progress bar

    process.close()  # Close the process after reading all lines
    pbar.close()     # Close the progress bar explicitly

et = time.time()
print("Time taken: ", round((et-st)*1.0/3600,2), " hours")

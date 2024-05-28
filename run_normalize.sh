# #!/bin/bash

# # Check if the correct number of arguments is provided
# if [ "$#" -ne 2 ]; then
#     echo "Usage: $0 input_file output_file"
#     exit 1
# fi

# # Assign input arguments to variables
# input_file="$1"
# output_file="$2"

# # Run ffmpeg command to convert the audio file
# ffmpeg -loglevel warning -hide_banner -stats -i "$input_file" -af "aresample=16000" -ac 1 -y "$output_file" 

# # Notify the user that the process has started
# echo "Processing $input_file to $output_file in the background."

path="$1"
ext=".$2"
for f in $(find "$path" -type f -name "*$ext")
do
ffmpeg -loglevel warning -hide_banner -stats -i "$f" -ar 16000 -ac 1 "$f$ext" && rm "$f" && mv "$f$ext" "$f" 

done

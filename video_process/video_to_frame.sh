#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo "usage: $0 <video_input> <target_dir>"
    exit 1
fi

ffmpeg -i $1 -vf scale=640:360 $2/frame%05d.png


#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

fileList=$(ls -1 $1)

for f in $fileList
do
  echo $f
  convert $1/$f -resize 640x360 $2/$f
done


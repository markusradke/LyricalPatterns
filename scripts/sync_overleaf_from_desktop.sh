#!/bin/bash

DOWNLOADED_PATH="/mnt/c/Users/marku/Desktop/_More_than_words___Genre_Discriminating_Lyrical_Patterns_in_Anglophone_German_Popular_Music.zip"

REPORT_PATH="/mnt/n/Materialien/Promotion/LyricsGenreRecognition/reports/paper_ismir/"

if [ -f $DOWNLOADED_PATH ]; then
  echo "Unzipping file to project folder..."
  unzip -o $DOWNLOADED_PATH -d $REPORT_PATH
  rm $DOWNLOADED_PATH
  echo "Done."
else
  echo "Downloaded File does not exist. Please export the file from Overleaf with 'File > Download as source (.zip)' Aborting script..."
fi

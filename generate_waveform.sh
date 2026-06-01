#!/bin/bash

eval set -- "$(getopt -o "" \
--long config:,output_dir:,author:,affiliation:,submission_number: \
-- "$@")"

while true; do
  case "$1" in
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --author) AUTHOR="$2"; shift 2 ;;
    --affiliation) AFFILIATION="$2"; shift 2 ;;
    --submission_number) SUBMISSION_NUMBER="$2"; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done

# Configuration variables
EVAL_SET_DIR="data/eval_set"
SOUNDSCAPE_DIR="$EVAL_SET_DIR/soundscape"

OUTPUT_FOLDER_NAME="${AUTHOR}_${AFFILIATION}_task4_${SUBMISSION_NUMBER}"
ZIP_NAME="${OUTPUT_FOLDER_NAME}.zip"
OUTPUT_FOLDER="${OUTPUT_DIR}/${OUTPUT_FOLDER_NAME}"

# Generate waveform
python -m src.evaluation.generate_waveform \
  -c "$CONFIG_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --output_name "$OUTPUT_FOLDER_NAME"

# Create zip file for submission
if [ -d "$OUTPUT_FOLDER" ]; then
  echo "Zipping: $OUTPUT_FOLDER_NAME"
  cd "$OUTPUT_DIR"
  zip -r "$ZIP_NAME" "$OUTPUT_FOLDER_NAME"
  cd -
  echo "Zipped to $ZIP_NAME"
else
  echo "Error: Output folder $OUTPUT_FOLDER not found."
  exit 1
fi

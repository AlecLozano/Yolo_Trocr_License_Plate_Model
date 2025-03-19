

## Description
A fine tuned [hugging face Trocr model](https://huggingface.co/microsoft/trocr-base-printed) that detects and reads license plates. 

When given unprocessed raw images, images are first passed through a [YOLO](https://huggingface.co/nickmuchi/yolos-small-finetuned-license-plate-detection) to detect and crop license plates from the image which are then passed through the Trocr model.

Work In Progress

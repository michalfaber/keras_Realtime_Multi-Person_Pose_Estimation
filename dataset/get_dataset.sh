#!/usr/bin/env bash

# Install gsutil which provides tools for efficiently accessing datasets
# without unzipping large files.
# Install gsutil via:curl https://sdk.cloud.google.com | bash

mkdir train2017
mkdir val2017
mkdir test2017
mkdir annotations

echo "Downloading train2017..."
gsutil -m rsync gs://images.cocodataset.org/train2017 train2017

echo "Downloading val2017..."
gsutil -m rsync gs://images.cocodataset.org/val2017 val2017

echo "Downloading test2017..."
gsutil -m rsync gs://images.cocodataset.org/test2017 test2017

echo "Downloading annotations..."
gsutil -m rsync gs://images.cocodataset.org/annotations annotations


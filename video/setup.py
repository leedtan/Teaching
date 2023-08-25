import os
import ssl
import zipfile
from urllib.request import urlretrieve

ssl._create_default_https_context = ssl._create_unverified_context


def download_and_extract_dataset():
    dataset_url = "https://motchallenge.net/data/MOT16.zip"
    zip_path = "MOT16.zip"
    extract_path = "MOT16"

    # Check if the directory already exists
    if os.path.exists(extract_path):
        print(f"{extract_path} already exists. No need to download again.")
        return

    # Download the zip file
    print(f"Downloading {dataset_url}...")
    urlretrieve(dataset_url, zip_path)

    # Extract the zip file
    print(f"Extracting {zip_path} to {extract_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    # Optionally, remove the zip file after extraction
    os.remove(zip_path)
    print(f"Dataset downloaded and extracted to {extract_path}.")


# Run the function to download and extract the dataset
download_and_extract_dataset()


def test_download_and_extract_dataset():
    download_and_extract_dataset()
    assert os.path.exists(
        "MOT16"
    ), "MOT16 directory should exist after download and extraction."

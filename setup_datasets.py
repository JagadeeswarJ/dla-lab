"""
DLA Lab - Automated Dataset Setup and Verification Script
Use this script to verify or download any missing datasets for the DLA Lab.
"""

import os
import sys
import zipfile
import urllib.request
import subprocess

def log(msg):
    print(f"[DATASET SETUP] {msg}")

def setup_driving_data():
    if os.path.exists("data") and os.path.exists(os.path.join("data", "driving_log.csv")):
        log("Self-driving car dataset ('data/') is present.")
    else:
        log("Cloning self-driving car dataset (Aman-py/data)...")
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/Aman-py/data.git"], check=True)
        log("Downloaded driving dataset successfully.")

def setup_uci_har():
    har_dir = os.path.join("har_data", "uci-har-dataset-master")
    if os.path.exists(har_dir):
        log("UCI HAR dataset ('har_data/uci-har-dataset-master') is present.")
        return

    har_zip = "uci-har-dataset-master.zip"
    if not os.path.exists(har_zip):
        log("Downloading UCI HAR dataset zip...")
        url = "https://github.com/greenglobal/uci-har-dataset/archive/refs/heads/master.zip"
        urllib.request.urlretrieve(url, har_zip)
        log("Downloaded UCI HAR zip.")

    log("Extracting UCI HAR dataset...")
    os.makedirs("har_data", exist_ok=True)
    with zipfile.ZipFile(har_zip, "r") as zf:
        zf.extractall("har_data")
    log("Extracted UCI HAR dataset successfully.")

def setup_quora_train():
    if os.path.exists("train.csv") and os.path.getsize("train.csv") > 1000:
        log("Quora Question Pairs dataset ('train.csv') is present.")
    else:
        log("Downloading Quora Question Pairs train.csv...")
        train_zip = "train.csv.zip"
        train_url = "https://raw.githubusercontent.com/Koorimikiran369/Quora-Question-Pairing/main/train.csv.zip"
        urllib.request.urlretrieve(train_url, train_zip)
        with zipfile.ZipFile(train_zip, "r") as zf:
            zf.extractall(".")
        if os.path.exists(train_zip):
            os.remove(train_zip)
        log("Quora train.csv downloaded and extracted successfully.")

def setup_nottingham():
    if os.path.exists("nottingham-dataset") and os.path.isdir("nottingham-dataset"):
        log("Nottingham ABC folk music dataset ('nottingham-dataset/') is present.")
    else:
        log("Cloning Nottingham ABC dataset...")
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/jukedeck/nottingham-dataset.git"], check=True)
        log("Nottingham dataset cloned successfully.")

def setup_maestro():
    if os.path.exists("maestro") and os.path.isdir("maestro"):
        log("MAESTRO MIDI dataset ('maestro/') is present.")
    else:
        log("Downloading MAESTRO MIDI dataset...")
        maestro_url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
        maestro_zip = "maestro-v3.0.0-midi.zip"
        urllib.request.urlretrieve(maestro_url, maestro_zip)
        log("Extracting MAESTRO MIDI dataset...")
        with zipfile.ZipFile(maestro_zip, "r") as zf:
            zf.extractall("maestro")
        if os.path.exists(maestro_zip):
            os.remove(maestro_zip)
        log("MAESTRO dataset downloaded and extracted successfully.")

def verify_all():
    print("=" * 60)
    print("DLA LAB DATASET VERIFICATION & SETUP")
    print("=" * 60)
    setup_driving_data()
    setup_uci_har()
    setup_quora_train()
    setup_nottingham()
    setup_maestro()
    print("=" * 60)
    log("ALL DATASETS ARE FULLY READY FOR EXAM EXECUTION!")
    print("=" * 60)

if __name__ == "__main__":
    verify_all()

import os
import gdown

os.mkdir("model")
# For storing mc model checkpoints
os.mkdir("model/mc")
# For storing qa model checkpoints
os.mkdir("model/qa")

# Download the dataset only if you don't have it already
if not os.path.exists("./dataset"):
    print("Downloading dataset...")
    dataset_url = "https://drive.google.com/drive/folders/1MoWExudAkSWXwibTrLPIaCaN6JScRaTo?usp=share_link"
    gdown.download_folder(dataset_url)
    print("Dataset downloaded!")
else:
    print("Dataset already exists!")
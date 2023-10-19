import os
import gdown
os.mkdir("outputs")
os.mkdir("outputs/mc")
os.mkdir("outputs/qa")
os.mkdir("outputs/result")
dataset_url = "https://drive.google.com/drive/folders/1MoWExudAkSWXwibTrLPIaCaN6JScRaTo?usp=share_link"
gdown.download_folder(dataset_url)
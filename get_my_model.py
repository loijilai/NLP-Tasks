import gdown
dataset_url = "https://drive.google.com/drive/folders/1MoWExudAkSWXwibTrLPIaCaN6JScRaTo?usp=share_link"
my_model_url = "https://drive.google.com/drive/folders/1mecwGvJzeQj3PjB7nGI1aMh0pkGi8rFm?usp=share_link"
gdown.download_folder(dataset_url)
gdown.download_folder(my_model_url)
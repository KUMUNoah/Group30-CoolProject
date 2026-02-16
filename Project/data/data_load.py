import kagglehub

# Download latest version
path = kagglehub.dataset_download("mahdavi1202/skin-cancer")

print("Path to dataset files:", path)
import kagglehub

# Download latest version
path = kagglehub.dataset_download("erniromauli/global-happiness-indicators-owid")

print("Path to dataset files:", path)
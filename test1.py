import os

# # Original dataset folders
# SOURCE_DIR = 'input'
# DEST_DIR = 'data/binary_split'
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        pass

# tumor_labels = ['glioma', 'meningioma', 'pituitary']
# no_tumor_label = 'notumor'

# # Create the target folders
# for split in ['train', 'test']:
#     for category in ['tumor', 'no_tumor']:
#         os.makedirs(os.path.join(DEST_DIR, split, category), exist_ok=True)
def main():
    load_dotenv()
    model_path = os.getenv("MODEL_PATH", "notebooks/api/model/brain_mri_model.h5")
    if os.path.exists(model_path):
        print(f"Model path exists: {model_path}")
    else:
        print(f"Model path NOT found: {model_path}")

# # Function to copy images to new binary structure
# def copy_images(split, original_label, target_label):
#     src_folder = os.path.join(SOURCE_DIR, 'Training' if split == 'train' else 'Testing', original_label)
#     dest_folder = os.path.join(DEST_DIR, split, target_label)
#     for img_name in os.listdir(src_folder):
#         src = os.path.join(src_folder, img_name)
#         dst = os.path.join(dest_folder, img_name)
#         if os.path.isfile(src):
#             shutil.copy(src, dst)

# # Process Training and Testing
# for split in ['train', 'test']:
#     for label in tumor_labels:
#         copy_images(split, label, 'tumor')
#     copy_images(split, no_tumor_label, 'no_tumor')

# print("✅ Dataset reorganized for binary classification (Tumor / No Tumor).")
import requests
with open(r"D:\ML_Projects\brainTumorDetection\input\Testing\notumor\Te-no_0040.jpg", "rb") as f:
    r = requests.post("http://localhost:8080/predict", files={"file": f})
print(r.json())

if __name__ == "__main__":
    main()
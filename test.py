import os
import shutil

# Original dataset folders
SOURCE_DIR = 'input'
DEST_DIR = 'data/binary_split'

tumor_labels = ['glioma', 'meningioma', 'pituitary']
no_tumor_label = 'notumor'

# Create the target folders
for split in ['train', 'test']:
    for category in ['tumor', 'no_tumor']:
        os.makedirs(os.path.join(DEST_DIR, split, category), exist_ok=True)

# Function to copy images to new binary structure
def copy_images(split, original_label, target_label):
    src_folder = os.path.join(SOURCE_DIR, 'Training' if split == 'train' else 'Testing', original_label)
    dest_folder = os.path.join(DEST_DIR, split, target_label)
    for img_name in os.listdir(src_folder):
        src = os.path.join(src_folder, img_name)
        dst = os.path.join(dest_folder, img_name)
        if os.path.isfile(src):
            shutil.copy(src, dst)

# Process Training and Testing
for split in ['train', 'test']:
    for label in tumor_labels:
        copy_images(split, label, 'tumor')
    copy_images(split, no_tumor_label, 'no_tumor')

print("✅ Dataset reorganized for binary classification (Tumor / No Tumor).")

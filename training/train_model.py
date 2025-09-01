from pathlib import Path


test_dir = Path("../data/binary_split/test")  # your test dir
# your model path
model_path = Path("../notebooks/api/model/brain_mri_model.h5")

print("Test dir exists? ", test_dir.exists())
print("Model file exists? ", model_path.exists())

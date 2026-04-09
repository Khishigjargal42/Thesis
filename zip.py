import zipfile
import os

# ZIP хийх файлууд
files_to_zip = [
    "spec.npy",
    "spec_labels.npy"
]

zip_name = "spec.zip"

# ZIP үүсгэх
with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for file in files_to_zip:
        if os.path.exists(file):
            z.write(file)
            print(f"Added: {file}")
        else:
            print(f"⚠️ File not found: {file}")

print(f"\n✅ ZIP created: {zip_name}")
import os, shutil, numpy as np
from pathlib import Path
# Build tiny dataset where all data lives under a single category
base = Path('d:/ASL/tmp_train_test')
if base.exists():
    shutil.rmtree(base)
base.mkdir(parents=True)
labels = ['LABEL_A', 'LABEL_B']
for lbl in labels:
    lbl_dir = base/ lbl
    lbl_dir.mkdir()
    seq_dir = lbl_dir/ '0'
    seq_dir.mkdir()
    for i in range(30):
        np.save(seq_dir/ f"{i}.npy", (np.random.randn(1662).astype(np.float32)))

import importlib.util
spec = importlib.util.spec_from_file_location('train', 'd:/ASL/train.py')
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)

# override dataset and categories
train_mod.DATA_PATH = str(base)
train_mod.categories = [labels]
train_mod.category_labels = ['category_0']

# make training lighter
train_mod.EPOCHS = 1
train_mod.BATCH_SIZE = 2

# Run main (should train category-level and one fine-grained model quickly)
train_mod.main()

# cleanup
shutil.rmtree(base)
print('Test run finished')

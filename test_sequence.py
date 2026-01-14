import os, shutil, numpy as np
from pathlib import Path
# prepare small dummy sequence
base = Path('d:/ASL/tmp_test_seq')
if base.exists():
    shutil.rmtree(base)
base.mkdir(parents=True)
(seq:=base/ '0').mkdir()
for i in range(30):
    np.save(seq/ f"{i}.npy", np.zeros(1662, dtype=np.float32))

# import SequenceGenerator
import importlib.util
spec = importlib.util.spec_from_file_location('train', 'd:/ASL/train.py')
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)

sg = train_mod.SequenceGenerator([( 'LABEL', str(seq) )], {'LABEL': 0}, batch_size=1)
X, y = sg[0]
print('X shape:', X.shape, 'y shape:', y.shape)
# cleanup
shutil.rmtree(base)

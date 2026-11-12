"""Validate the canonical archive; no model fitting and no Google Drive dependency.
Usage: python Code/src/verify_archive.py Data/fi2010_canonical_raw40_labels.npz
"""
import sys, hashlib
from pathlib import Path
import numpy as np

EXPECTED_SHA256='e55943477e294b853ea032137ab5907a53a207ff36fcbc710d7024f6d613d696'
EXPECTED_DAY_ROWS=[39512,38397,28535,37023,34785,39152,37346,55478,52172,31937]
p=Path(sys.argv[1] if len(sys.argv)>1 else 'Data/fi2010_canonical_raw40_labels.npz')
actual=hashlib.sha256(p.read_bytes()).hexdigest()
assert actual==EXPECTED_SHA256, f'Unexpected data digest: {actual}'
z=np.load(p,allow_pickle=False)
assert z['X'].shape==(394337,40) and z['Y'].shape==(394337,5)
assert z['X'].dtype==np.dtype('float32') and z['Y'].dtype==np.dtype('int8')
assert set(np.unique(z['Y']).tolist())=={1,2,3}
assert len(z['raw_feature_names'])==40 and len(z['label_names'])==5
assert [int((z['day']==d).sum()) for d in range(1,11)]==EXPECTED_DAY_ROWS
assert [int(((z['day']==d)&z['is_segment_start']).sum()) for d in range(1,11)]==[5]*10
assert int(z['is_segment_start'].sum())==50
print('PASS: SHA256, 394337x40 inputs, 394337x5 publisher labels, 10 day counts and 50 segment starts')
print('Archive SHA-256:',actual)

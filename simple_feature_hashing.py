#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
Use the binary representation of a variable as features.
Probably useful for shorter strings.
"""


import numpy as np
import pandas as pd
import sys
from time import time

t0 = time()
# Create data
n_phones = 100_000 if len(sys.argv) == 1 else int(sys.argv[1])
phones = """Iphone X
Iphone 8 Max
Samsung Galaxy S10 Plus
Huawei P30 Pro
OnePlus 7 Pro
Huawei Mate 20
Google Pixel 3
LG V40 ThinQ
LG G7 Plus ThinQ
Google Pixel 2 XL
OnePlus 6T""".splitlines()
ser = pd.Series([phones[i] for i in np.random.randint(0, len(phones), (n_phones,))])

# This is an optimization, but the problem seems to be memory bound rather than CPU bound (the hashing must be cheap)
if n_phones < 10_000:
    hsh = ser.apply(hash).values
else:
    unique_phones = ser.unique()
    hsh_dct = {el: hash(el) for el in unique_phones}
    hsh = np.array([hsh_dct[ph] for ph in ser])

assert pd.unique(hsh).shape[0] == ser.nunique()

hshv = np.unpackbits(
    hsh.view(np.uint8)
)  # open a view into the continious memory, interpret as uint8

# This is our new feature
feat = hshv.reshape(ser.shape[0], 64)  # 64 because the hast values are really int64

print(feat)
assert len({tuple(el) for el in feat}) == len(phones)
t1 = time()
print(t1 - t0)

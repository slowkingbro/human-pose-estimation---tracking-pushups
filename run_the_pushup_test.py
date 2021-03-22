from importlib import reload
import sys
#sys.path.append('/content/drive/My Drive/cs231a/Project/VIBE')
print(sys.path)
import pandas as pd
import lib.utils.get_functions as gf
from matplotlib import pyplot as plt
#reload(gf)
import numpy as np


name_root = 'Jordan Syatt - BRO PUSH-UPS  THE 3 WORST PUSH-UP MISTAKES ft Johnny Bag-O-Donuts'
vibe_output = gf.get_vibe_output(name_root, is_good_pushup = 'BAD')
time_start, time_end = 17, 20
df_count_pushup = gf.count_pushup(name_root, time_start, time_end, 'BAD', 2, True, 25)
df_count_pushup = pd.DataFrame.from_dict(df_count_pushup)


pred_result = {}
is_good_pushup = False
key = name_root
for i in range(time_start*25, time_end*25):
    pred_result[i] = (gf.is_straight_back(vibe_output, i))
        is_spike, error_min = gf.is_error_spike(vibe_output, i)
            is_moving = gf.structure_test(vibe_output, 18)
                if is_spike and is_moving:
pred_result[i] = min

pred_result = pd.DataFrame.from_dict(pred_result, orient = 'index')
pred_result = pred_result.reset_index()
pred_result = pred_result.rename(columns = {0:'pred_error', 'index':'frame_number'})
df_result = pd.merge(df_count_pushup, pred_result, on = 'frame_number', how = 'outer')
plt.figure()
plt.plot(df_result['pred_error'][df_result['pred_error'] != 1000000])
plt.plot(df_result['direction'][df_result['pred_error'] != 1000000])
plt.xlabel('Frame Number')
plt.ylabel('Adjusted Delta')
plt.title('Delta vs. Frame Number')
plt.show()



#data labels: 1 means correct, 0 means incorrect
bad_example_gt = {'The Worst Push-Up Mistakes (and How to Fix Them)  Health':{9:0, 13:1, 17:1, 19:1, 22:0}, #9 is not detect appropriately because of two people
               'Push Up Right vs Wrong':{4:0, 7:1, 11:0, 16:1, 20:1, 32:1, 47:0, 51:0, 52:0}, #32 is front view completely wrong
               'Jordan Syatt - BRO PUSH-UPS  THE 3 WORST PUSH-UP MISTAKES ft Johnny Bag-O-Donuts':{36:0, 38:1, 45:1, 60+2:0, 60+7:1,60+9:1, 60+34:1, 60+56:0, 2*60+5:0, 2*60+18:0},
               'How To Do A Push-Up  The Right Way  Well+Good':{1:0, 20:0, 22:0, 24:0, 28:0, 33:0, 60+14:0, 60+19:1, 60+59:1, 60*2+1:1, 60*2+6:1, 60*2+10:1, 60*2+15:1, 60*2+22:0, 60*2+24:1},
               'Common push-up mistakes':{19:0, 24:1, 29:1, 33:1, 36:0, 37:0, 42:0, 46:0, 60+22:1, 60+26:1, 60+30:1, 60+37:1, 60+48:1, 60+53:1},
               'Bad Push-up':{1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1},
               '6 Pushup mistakes and how to fix them':{19:0, 23:0, 33:0, 38:0, 60+3:1, 60+9:0, 60+30:1, 60+35:1, 60*2+33:0, 60*2+36:1, 60*2+38:0, 60*2+40:1},
               '5 Ways People Do Push Ups Wrong':{8:0, 14:0, 54:1, 58:1, 60:0, 60+3:0, 60+7:1, 60+11:1, 60+17:1, 60+46:0, 60+52:0, 60*2:1, 60*2+5:1, 60*2+9:1, 60*2+12:1, 60*2+25:0, 60*2+32:1, 60*2+38:1},
               '3 Biggest Pushup Mistakes Most People Make & And How To Fix Them Instantly - Sixpackfactory':{4:1, 60:0, 60+1:1}
}

df_gt = pd.DataFrame([(k, k1, v1) for k, v in bad_example_gt.items() for k1, v1 in v.items()],
                   columns=['video_name','timestamp','gt_is_good_pushup'])
df_gt['gt_is_good_pushup_adjusted_label'] = [1 if x == 0 else 0 for x in df_gt['gt_is_good_pushup']]


#prediction result
pred_result = {}
is_good_pushup = False
for key in bad_example_gt.keys():
  name_root = key
  vibe_output = get_vibe_output(name_root, is_good_pushup = 'BAD')
  pred_result[key] = {}
  for i in bad_example_gt[name_root].keys():
    print(name_root, "timestamp = ", i)
    timestamp = i
    frame_number = get_frame_number(timestamp)
    pred_result[key][i] = (is_straight_back(vibe_output, frame_number))
    is_spike, min = is_error_spike(vibe_output, i)
    if is_spike:
       pred_result[key][i] = min

df_pred = pd.DataFrame([(k, k1, v1) for k, v in pred_result.items() for k1, v1 in v.items()],
                   columns=['video_name','timestamp','pred_is_good_pushup'])

df_pred2 = df_pred[df_pred['pred_is_good_pushup'] != 1000000] #filter out 10000 (frame out) 
df_merge = pd.merge(df_pred2, df_gt, on=['video_name', 'timestamp'], how = 'left')


# ROC curve
test = df_merge['gt_is_good_pushup_adjusted_label']
pred = df_merge['pred_is_good_pushup']
fpr, tpr, thresholds = roc_curve(test, pred)

plt.figure()
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()

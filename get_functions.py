import joblib
import os
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from IPython.display import HTML
from base64 import b64encode
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix


def get_vibe_output(name_root, is_good_pushup = 'BAD'): #enter 'GOOD' 'BAD' or your own path
  name = name_root + '_vibe_result'
  #img_name = name + str(frame_number)+'.jpg'
  if is_good_pushup == 'GOOD':
    path = './output/good_pushup/'+ name_root + '/'
  elif is_good_pushup == 'BAD':
    path = './output/bad_pushup/'+ name_root + '/'
  else:
    path = path
  vibe_output = joblib.load(path + 'vibe_output.pkl')
  return vibe_output


def get_frame_number(timestamp, fps = 25):
  frame_number = timestamp*fps
  return frame_number


def get_frame_pic(timestamp, name_root, is_good_pushup, fps = 25):
  frame_number = timestamp * fps
  name = name_root + '_vibe_result'
  if is_good_pushup == 'GOOD':
    path = './output/good_pushup/'+ name_root + '/'
  elif is_good_pushup == 'BAD':
    path = './output/bad_pushup/'+ name_root + '/'
  else:
    path = path
  vidcap = cv2.VideoCapture(path + name + '.mp4')
  vidcap.set(1, frame_number)
  res, frame = vidcap.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  vibe_output = get_vibe_output(name_root, is_good_pushup)
  kp_2d = get_pkl_item(vibe_output, get_next_available_frame_num(vibe_output, frame_number), item='joints2d_img_coord')

  for idx in range(49):
    frame = cv2.circle(frame, (int(kp_2d[idx][0]), int(kp_2d[idx][1])), 5, (255, 0, 0), -1)  

  return frame   #plt.imshow(frame)


# Play the generated video
def video(path):
  mp4 = open(path,'rb').read()
  data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
  return HTML('<video width=500 controls loop> <source src="%s" type="video/mp4"></video>' % data_url)


# If the back is the same plane
def is_straight_back(vibe_output, frame_number, if_print = True): #check both hip and the right knee on the same plane as right/left shoulder and left knee

  people_key = identify_person_key(vibe_output, frame_number)
  if people_key != 0:
    pred_j3ds = get_pkl_item(vibe_output, frame_number, 'joints3d')
    rshoulder = pred_j3ds[33, :]
    lshoulder = pred_j3ds[34, :]
    lknee = pred_j3ds[29, :]
    lhip = pred_j3ds[28, :]
    rknee = pred_j3ds[24, :]

    v1 = lknee - rshoulder
    v2 = lshoulder - rshoulder
    cp = np.cross(v1, v2)
    a, b, c = cp
    d = np.dot(cp, lknee)

    X, Y, Z = lhip
    Z_pred = (d - a * X - b * Y) / c
    lhip_error = abs(Z_pred - Z)
    
    error = lhip_error
  else:
    error = 1000000

  if if_print:
    print('error actual vs. pred=', error)
  return error


def identify_person_key(vibe_output, frame_number):  # identify_person_key return 0 when no frames are found
  people_key = 0
  for i in vibe_output.keys():
    if frame_number in vibe_output[i]['frame_ids']:
      people_key = i
      break
  return people_key


def get_pkl_item (vibe_output, frame_number, item = 'joints3d'):
  #joints reference github https://github.com/mkocabas/VIBE/blob/6e70f5fcba8809fb64e4199bbfd44fb324ac47e4/lib/data_utils/kp_utils.py
  people_key = identify_person_key(vibe_output, frame_number)
  if people_key != 0:
    tensor = vibe_output[people_key][item]
    row_num = vibe_output[people_key]['frame_ids']==frame_number
    if item in ('joints3d'):
      result = tensor[row_num, :, :][0]
    else:
      result = tensor[row_num][0]
  else:
    return 
  return result
  

def get_next_available_frame_num(vibe_output, frame_number):
  #frame_number = timestamp*25
  people_key = identify_person_key(vibe_output, frame_number)
  while people_key == 0 and frame_number <= get_max_frame(vibe_output):
    people_key = identify_person_key(vibe_output, frame_number)
    #print(frame_number)
    frame_number += 1
  return frame_number


def get_max_frame(vibe_output):
  person_max = 0
  for i in vibe_output.keys():
      person_max = max(vibe_output[i]['frame_ids'].max(), person_max)
  return person_max


def draw_tracks(frame_num, frame, mask, points_prev, points_curr, color):
    """Draw the tracks and create an image.
    """
    for i, (p_prev, p_curr) in enumerate(zip(points_prev, points_curr)):
        a, b = p_curr.ravel()
        c, d = p_prev.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(
            frame, (a, b), 3, color[i].tolist(), -1)

    img = cv2.add(frame, mask)

    cv2.imwrite('./output/optical_flow/frame_%d.png'%frame_num,img)
    return img


def count_pushup(name_root, timestamp_start, timestamp_end, is_good_pushup = 'BAD', tracking_joint = 2, save_optical_flow = False, fps = 25): # by default tracking joint = 2 to track rshoulder
  frame_range = np.arange(timestamp_start*fps, timestamp_end*fps)

  name = name_root + '_vibe_result'
  if is_good_pushup == 'GOOD':
    path = './output/good_pushup/'+ name_root + '/'
  elif is_good_pushup == 'BAD':
    path = './output/bad_pushup/'+ name_root + '/'
  else:
    path = path
  vidcap = cv2.VideoCapture(path + name + '.mp4')
  vidcap.set(1, timestamp_start)    
  _, frame = vidcap.read()

  vibe_output = get_vibe_output(name_root, is_good_pushup)

  # params for ShiTomasi corner detection
  feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

  # Parameters for lucas kanade optical flow
  lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

  # Create some random colors
  color = np.random.randint(0,255,(100,3))

  # Take first frame and find corners in it
  old_frame = frame
  old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

  kp_2d = get_pkl_item(vibe_output, get_next_available_frame_num(vibe_output, frame_range[0]), item='joints2d_img_coord')
  #print('kp_2d', kp_2d)
  p0 = kp_2d.reshape(kp_2d.shape[0], 1, -1)
  direction = {}
  direction['frame_number'] = []
  direction['direction'] = []

  # Create a mask image for drawing purposes
  mask = np.zeros_like(old_frame)

  for fn in frame_range:
      direction['frame_number'].append(fn)
      vidcap.set(1, fn)
      ret,frame = vidcap.read()
      frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # calculate optical flow
      p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
      # Select good points
      if p1 is not None:
          good_new = p1[st==1]
          good_old = p0[st==1]
      #draw the tracks
      for i,(new,old) in enumerate(zip(good_new, good_old)): #iterate through all tracking points, here we only track chest 
          if i == tracking_joint: # only examine one joint's motion
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1) 
            # only examine one joint's motion
            if b > d:
              direction['direction'].append(-1) 
            else:
              direction['direction'].append(1)
      img = cv2.add(frame,mask)
      #plt.imshow(img)
      if save_optical_flow:
        draw_tracks(i, frame, mask, good_old, good_new, color)

      k = cv2.waitKey(30) & 0xff
      if k == 100:
          break
      #Now update the previous frame and previous points
      old_gray = frame_gray.copy()
      p0 = good_new.reshape(-1,1,2)
      
      df = pd.DataFrame.from_dict(direction)
  return direction
   

def plot_joint_time_duration(name_root, timestamp_start, timestamp_end, joint_id = 33, is_good_pushup = 'BAD'): #default is rshoulder
  vibe_output = get_vibe_output(name_root, is_good_pushup)
  single_joint_list = {}
  single_joint_list['frame_number'] = []
  single_joint_list['timestamp'] = []
  single_joint_list['x_coordinate'] = []
  single_joint_list['y_coordinate'] = []
  single_joint_list['z_coordinate'] = []
  
  for i in range(get_frame_number(timestamp_start), get_frame_number(timestamp_end)):
    
    people_key = identify_person_key(vibe_output, i)
    
    if people_key > 0:
      joints3d = get_pkl_item(vibe_output, frame_number=i)
      single_joint_list['x_coordinate'].append(joints3d[joint_id, 0])
      single_joint_list['y_coordinate'].append(joints3d[joint_id, 1])
      single_joint_list['z_coordinate'].append(joints3d[joint_id, 2])
      single_joint_list['frame_number'].append(i)
      single_joint_list['timestamp'].append(i//25) #fps = 25

  df = pd.DataFrame.from_dict(single_joint_list)

  ax = plt.gca()
  ax.plot('frame_number', 'x_coordinate', data = df, marker = 'o', color = 'skyblue')
  ax.plot('frame_number', 'y_coordinate', data = df, marker = 'o', color = 'olive')
  ax.plot('frame_number', 'z_coordinate', data = df, marker = 'o', color = 'green')
  ax.legend()
  ax.set_xlabel('frame_number')
  ax.set_ylabel('joints 3d coordinate')
  ax.set_title('timeframe between {0} and {1}. video name ='.format(get_frame_number(timestamp_start), get_frame_number(timestamp_end)) + name_root)
  return ax


def factorization_method(points_im1, points_im2):
    c1 = np.array([np.sum(points_im1[:, 0])/points_im1.shape[0], np.sum(points_im1[:, 1])/points_im1.shape[0]])
    c2 = np.array([np.sum(points_im2[:, 0])/points_im2.shape[0], np.sum(points_im2[:, 1])/points_im2.shape[0]])
    D = np.vstack((points_im1[:,:2].T, points_im2[:,:2].T))
    D[0,:] = D[0,:] - c1[0]
    D[1,:] = D[1,:] - c1[1]
    D[2,:] = D[2,:] - c2[0]
    D[3,:] = D[3,:] - c2[1]
    
    U,S,V = np.linalg.svd(D)    
    S = np.diag(S)
    S3 = S[:3, :3]
    V3 = V[:3, :]
    structure = np.dot(S3, V3)
    motion = U[:, :3]
    return (structure, motion)


def is_error_spike(vibe_output, timestamp):
  error_list = {}
  error_list['frame_number'] = []
  error_list['error'] = []

  for i in range((timestamp)*25, (timestamp+1)*25):
    error_list['error'].append(is_straight_back(vibe_output, i, if_print = False))
    error_list['frame_number'].append(i)

  df_error = pd.DataFrame.from_dict(error_list)
  min = df_error['error'].min()
  mean = df_error['error'].mean() #is_error_spike(vibe_output, 107)['error'].mean()
  std = df_error['error'].std()   #is_error_spike(vibe_output, 107)['error'].std()
  return (df_error['error'].max()>3*std+mean), min #, df_error # if the error is outlier of the second (that includes the frame), return min predction of the second
  
  
def structure_test(vibe_output, timestamp):
  structure_list = {}
  structure_list['frame_number'] = []
  structure_list['x'] = []
  structure_list['y'] = []

  for i in range((timestamp)*25, (timestamp+1)*25):
    t1 = i
    points_im1 = get_pkl_item(vibe_output, get_next_available_frame_num(vibe_output, t1), item = 'joints2d_img_coord')
    t2 = i+5
    points_im2 = get_pkl_item(vibe_output, get_next_available_frame_num(vibe_output, t2), item = 'joints2d_img_coord')
    structure, motion = factorization_method(points_im1, points_im2) 
    structure_list['x'].append(structure.T[22][0])
    structure_list['y'].append(structure.T[22][1])
    structure_list['frame_number'].append(i)
  df_structure = pd.DataFrame.from_dict(structure_list)
  df_structure['y_change'] = df_structure['y'].shift(1)/df_structure['y']-1
  df_structure['x_change'] = df_structure['x'].shift(1)/df_structure['x']-1
  return (max(abs(df_structure['y_change']).max(), abs(df_structure['x_change']).max()) >= 2) #, df_structure # if x or y coordinate change is exceed 200% from previous frame


  

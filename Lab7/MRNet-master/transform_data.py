import numpy as np
import os
from scipy import ndimage as nd

def apply_three_slice_trans(data_path, trans_path, dilation=2):
    npy_files = [f for f in sorted(os.listdir(data_path)) if f[-4:]=='.npy']
    temp = []
    new_stacks = []
    for file in npy_files:
        orig_stack=np.load(data_path+'/'+file)
        mid_slice_idx=orig_stack.shape[0]//2
        mid_slice = orig_stack[mid_slice_idx]
        lower_slice = orig_stack[mid_slice_idx-dilation]
        upper_slice = orig_stack[mid_slice_idx+dilation]
        temp.extend([lower_slice,mid_slice,upper_slice])
        new_stacks.append(np.array(temp))
        temp.clear()
    np.save(trans_path,np.array(new_stacks))

def apply_five_class_labels(abn_label_path, acl_label_path, men_label_path, trans_path):
    abnormal = np.genfromtxt(abn_label_path, delimiter=',')
    acl = np.genfromtxt(acl_label_path, delimiter=',')
    meniscus = np.genfromtxt(men_label_path, delimiter=',')
    labels = np.zeros((abnormal.shape[0],5))
    for i in range(labels.shape[0]):
        if(abnormal[i,1] == 1 and acl[i,1] == 1 and meniscus[i,1] == 1):
            labels[i,4] = 1
        elif(abnormal[i,1] == 1 and meniscus[i,1] == 1):
            labels[i,3] = 1
        elif(abnormal[i,1] == 1 and acl[i,1] == 1):
            labels[i,2] = 1
        elif(abnormal[i,1] == 1):
            labels[i,1] = 1
        elif(abnormal[i,1] == 0 and acl[i,1] == 0 and meniscus[i,1] == 0):
            labels[i,0] = 1
    np.save(trans_path,labels)

def apply_interpolation_transformation(data_path, trans_path,slices):
    npy_files = [f for f in sorted(os.listdir(data_path)) if f[-4:]=='.npy']
    new_stacks=[]
    for file in npy_files:
      orig_file_path=data_path+'/'+file
      print('Converting {}'.format(orig_file_path))
      orig_stack=np.load(orig_file_path)
      print('Number of slices for this scan: {}'.format(orig_stack.shape[0]))
      interpolation_factors = [w/float(f) for w,f in zip([slices,256,256], orig_stack.shape)]
      print('Interpolation factors for this scan: {}'.format(interpolation_factors))
      interpolated_scan = nd.interpolation.zoom(orig_stack, zoom=interpolation_factors)
      new_stacks.append(interpolated_scan)
      
    np.save(trans_path,new_stacks)
    print('===============================')
    print('===============================')

    
    

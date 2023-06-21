import numpy as np
import pandas as pd
import os
from numpy.linalg import inv
from functions import geometric_median_old
np.set_printoptions(suppress=True)
from pycpd import RigidRegistration
from ct2imf import compute_ct2vtk,read_ini,CamRoboRegistration
from scipy.spatial.transform import Rotation as R

path = r"D:\Navigation\CT_sphere_detection\Dataset-3-20230610T054412Z-001"

# metal and Oarmdf
file = os.path.join(path,"4330_collect_batch3_metal_Oarm.csv")
ct_cmm_file = os.path.join(path,"ct_cmm.txt")
df = pd.read_csv(file)

def transform_pts(cmm,heli2ref_tf_filt):
    cmm_homo = np.hstack((cmm,np.ones((len(cmm),1))))

    cmm_ref_homo = heli2ref_tf_filt@cmm_homo.transpose()

    cmm_ref = cmm_ref_homo.transpose()[:,:3]
    return cmm_ref

def filter_data(pos,quat):
    
    #L2 Chordal Mean for Quat
    quat_filt = R.from_quat(quat).mean().as_quat()
    median_pos = geometric_median_old(pos)
    marker2cam_r=R.from_quat(quat_filt).as_matrix().transpose()
    marker2cam=rot2tf(marker2cam_r,median_pos)

    return marker2cam

def read_mat_fromtxt(file_name):
    with open(file_name, 'r') as f:
        l = [[float(num) for num in line.split(',')] for line in f]
    return l


def rot2tf(rot,pos):
    
    pos_s = np.array(pos)
    rot_matrix = rot
    temp= np.column_stack((rot_matrix,pos_s))
    tf= np.vstack((temp,[0,0,0,1]))
    return tf


Oarm_pos = df[['refx','refy','refz']].to_numpy()
Oarm_quat = df [['ref_qx','ref_qy','ref_qz','ref_qw']].to_numpy()
heli_pos = df[['toolx','tooly','toolz']].to_numpy()
heli_quat = df [['tool_qx','tool_qy','tool_qz','tool_qw']].to_numpy()
heli2ref_tf = []
heli2ref_pos = []



Oarm2cam_filt= filter_data(Oarm_pos,Oarm_quat)
cam2Oarm_filt = inv(Oarm2cam_filt)
heli2cam_filt= filter_data(heli_pos,heli_quat)

helical2Oarm_tf = cam2Oarm_filt@heli2cam_filt

cmm_heli = pd.read_csv("heli_cmm.csv")
cmm_Oarm = transform_pts(cmm_heli,helical2Oarm_tf)
cmm_heli = cmm_heli.to_numpy()
cmm_heli_homo = np.hstack((cmm_heli,np.ones((len(cmm_heli),1))))
cmm_Oarm_homo = helical2Oarm_tf@cmm_heli_homo.transpose()
cmm_Oarm = cmm_Oarm_homo.transpose()[:,:3]
ct_cmm = np.array(read_mat_fromtxt(ct_cmm_file))

trans,error=CamRoboRegistration(ct_cmm,cmm_Oarm)
ct2Oarm = trans

np.save('cmm_Oarm',cmm_Oarm)
np.save('heli2cam_tf_filt', heli2cam_filt)
np.save('ct_cmm',ct_cmm)
np.save("ct2Oarm", ct2Oarm)











    





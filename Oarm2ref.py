import numpy as np
import pandas as pd
import os
from numpy.linalg import inv
from functions import geometric_median_old
np.set_printoptions(suppress=True)
from functions import plot_fids

from scipy.spatial.transform import Rotation as R
from ct2imf import compute_ct2vtk,read_ini,CamRoboRegistration


path = r"D:\Navigation\CT_sphere_detection\Dataset-4-20230610T054436Z-001"

file = os.path.join(path,"54320_collect_batch4_ref_Oarm.csv")

df = pd.read_csv(file)
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

def transform_pts(cmm,heli2ref_tf_filt):
    cmm_homo = np.hstack((cmm,np.ones((len(cmm),1))))

    cmm_ref_homo = heli2ref_tf_filt@cmm_homo.transpose()

    cmm_ref = cmm_ref_homo.transpose()[:,:3]
    return cmm_ref


def rot2tf(rot,pos):
    
    pos_s = np.array(pos)
    rot_matrix = rot
    temp= np.column_stack((rot_matrix,pos_s))
    tf= np.vstack((temp,[0,0,0,1]))
    return tf


Oarm_pos = df[['refx','refy','refz']].to_numpy()
Oarm_quat = df [['ref_qx','ref_qy','ref_qz','ref_qw']].to_numpy()
ref_pos = df[['toolx','tooly','toolz']].to_numpy()
ref_quat = df [['tool_qx','tool_qy','tool_qz','tool_qw']].to_numpy()
Oarm2ref_tf = []
Oarm2ref_pos = []



ref2cam_filt= filter_data(ref_pos,ref_quat)
Oarm2cam_filt= filter_data(Oarm_pos,Oarm_quat)
cam2ref_filt = inv(ref2cam_filt)

Oarm2ref_tf_filt = cam2ref_filt@Oarm2cam_filt
ref2Oarm = inv(Oarm2ref_tf_filt)

ct2Oarm = np.load("ct2Oarm.npy")
Oarm2ct = inv(ct2Oarm)

ref2Ct = Oarm2ct@ref2Oarm
ref2ct_round = np.around(ref2Ct, decimals=5)
np.save("ref2ct_Oarm",ref2Ct)
np.savetxt('ref2Ct_Oarm.txt',ref2ct_round ,fmt='%1.3f', delimiter=',') 

print(ref2ct_round)


geo_path = r"D:\Navigation\CT_sphere_detection\GeometryFiles"
geometry_ref = "geometry54320.ini"
geometry_hel = "geometry4330.ini"
geometry_Oarm = "geometry11005.ini"

ir_ref_geo_val = read_ini(geo_path,geometry_ref)
ir_heli_geo_val = read_ini(geo_path,geometry_hel)
ir_Oarm_geo_val = read_ini(geo_path,geometry_Oarm)
heli2cam_tf_filt = np.load('heli2cam_tf_filt.npy')

heli2ref_tf_filt = cam2ref_filt@heli2cam_tf_filt
ir_heli_ref = transform_pts(ir_heli_geo_val,heli2ref_tf_filt)
ct_cmm = np.load('ct_cmm.npy')
ct2ref = inv(ref2Ct)
cmm_ct_inRef = transform_pts(ct_cmm,ct2ref)
cmm_Oarm = np.load('cmm_Oarm.npy')
cmm_ref = transform_pts(cmm_Oarm,Oarm2ref_tf_filt)
ir_Oarm_ref = transform_pts(ir_Oarm_geo_val,Oarm2ref_tf_filt)




fids_plot = transform_pts(np.vstack((ir_ref_geo_val,ir_heli_ref,cmm_ref,cmm_ct_inRef,ir_Oarm_ref)),ref2cam_filt)
fig = plot_fids(fids_plot)
fig.show()













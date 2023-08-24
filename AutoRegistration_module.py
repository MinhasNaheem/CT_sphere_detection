import numpy as np
import pandas as pd
import os
from numpy.linalg import inv
from functions import geometric_median_old
from pycpd import RigidRegistration
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycpd import DeformableRegistration,AffineRegistration
from ct2imf import compute_ct2vtk,read_ini,CamRoboRegistration
from functions import plot_fids



np.set_printoptions(suppress=True)

from scipy.spatial.transform import Rotation as R


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
def transform_pts(cmm,heli2ref_tf_filt):
    cmm_homo = np.hstack((cmm,np.ones((len(cmm),1))))

    cmm_ref_homo = heli2ref_tf_filt@cmm_homo.transpose()

    cmm_ref = cmm_ref_homo.transpose()[:,:3]
    return cmm_ref

def main():
    path = r"D:\Navigation\Carm_registration\Dataset-2"

    file = os.path.join(path,"4330_collect_batch2_autoreg_hel2Referecnce.csv")
    ct_cmm_file = os.path.join(path,"ct_cmm.txt")
    df = pd.read_csv(file)
    ref_pos = df[['refx','refy','refz']].to_numpy()
    ref_quat = df [['ref_qx','ref_qy','ref_qz','ref_qw']].to_numpy()
    heli_pos = df[['toolx','tooly','toolz']].to_numpy()
    heli_quat = df [['tool_qx','tool_qy','tool_qz','tool_qw']].to_numpy()
    heli2ref_tf = []
    heli2ref_pos = []



    ref2cam_filt= filter_data(ref_pos,ref_quat)
    heli2cam_filt= filter_data(heli_pos,heli_quat)
    cam2ref_filt = inv(ref2cam_filt)

    heli2ref_tf_filt = cam2ref_filt@heli2cam_filt
    ref2heli = inv(heli2ref_tf_filt)

    cmm_heli = pd.read_csv("heli_cmm.csv")
    cmm = cmm_heli.to_numpy()
    cmm_ref = transform_pts(cmm,heli2ref_tf_filt)
    
    ct_cmm = np.array(read_mat_fromtxt(ct_cmm_file))
    
    X = np.vstack((ct_cmm[0],ct_cmm[1],ct_cmm[5]))
    Y = np.vstack((cmm_ref[0],cmm_ref[1],cmm_ref[5]))
    X=ct_cmm
    Y=cmm_ref
    
    trans,error=CamRoboRegistration(X,Y)
    ct2ref = trans
    print(f'error: {error}')
 
    ref2ct = inv(ct2ref)
    np.save("ref2ct", ref2ct )
    print(f'ref2ct {ref2ct}')
    file_path = r"D:\Navigation\Carm_registration\Dataset-2\DICOM\PA0\ST0\SE1"
    ct2imf = compute_ct2vtk(file_path,plot_flag=False)
    imf2ct = inv(ct2imf)
    print(f'ct2imf {ct2imf}')
    ref2imf = ct2imf @ref2ct
    print(f'ref2img : {ref2imf}')
    # print(inv(ref2imf))
    geo_path = r"D:\Navigation\Carm_registration\GeometryFiles"
    geometry_ref = "geometry54320.ini"
    geometry_hel = "geometry4330.ini"
    
    ir_ref_geo_val = read_ini(geo_path,geometry_ref)
    ir_heli_geo_val = read_ini(geo_path,geometry_hel)

    
    ir_heli_ref = transform_pts(ir_heli_geo_val,heli2ref_tf_filt)
    cmm_ct_inRef = transform_pts(ct_cmm,ct2ref)



    fids_plot = transform_pts(np.vstack((ir_ref_geo_val,ir_heli_ref,cmm_ref,cmm_ct_inRef)),ref2cam_filt)
    fig = plot_fids(fids_plot)
    fig.show()






    


if __name__ == '__main__':
    main()









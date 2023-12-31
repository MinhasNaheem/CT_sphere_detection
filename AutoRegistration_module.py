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
from MathFunctions import registrationWithoutCorrespondence



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
    path = r"D:\Navigation\Carm_registration\DICOM"

    file = os.path.join(path,"Fid_Set4_temp.csv")
    ct_cmm_file = os.path.join(path,"ct_cmm_auto_reg.txt")
    ct_cmm = np.loadtxt(ct_cmm_file)
    df = pd.read_csv(file)
    ref_pos = df[['refx','refy','refz']].to_numpy()
    ref_quat = df [['ref_qx','ref_qy','ref_qz','ref_qw']].to_numpy()
    heli_pos = df[['phanx','phany','phanz']].to_numpy()
    heli_quat = df [['phan_qx','phan_qy','phan_qz','phan_qw']].to_numpy()
    heli2ref_tf = []
    heli2ref_pos = []



    ref2cam_filt= filter_data(ref_pos,ref_quat)
    heli2cam_filt= filter_data(heli_pos,heli_quat)
    cam2ref_filt = inv(ref2cam_filt)

    heli2ref_tf_filt = cam2ref_filt@heli2cam_filt
    ref2heli = inv(heli2ref_tf_filt)

    cmm = np.loadtxt(os.path.join(path,'auto_regCmm.txt')) 
    cmm_ref = transform_pts(cmm,heli2ref_tf_filt)
    
    
    ref2ct, error = registrationWithoutCorrespondence(cmm_ref,ct_cmm)
    ct2ref = inv(ref2ct)
    print(f'error: {error}')

    np.save("ref2ct", ref2ct )
    print(f'ref2ct {ref2ct}')
    file_path = os.path.join(path,'PA0\ST0\SE1')
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









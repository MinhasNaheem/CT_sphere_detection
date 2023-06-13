
import SimpleITK as sitk
import numpy as np
# import json
from numpy.linalg import inv
np.set_printoptions(suppress=True)
from functions import plot_fids
import configparser
import os
import open3d as o3d

file_path = "D:\\imfusion\\ImFusion_registration_sdk\\MatlabData\\CT\\SET2\\SE0"
def compute_ct2imf(file_path,plot_flag=False):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(file_path)

    reader.SetFileNames(dicom_names)
    image = reader.Execute()


    image_dim = np.array(list(image.GetSize()))
    image_dir = np.array(list(image.GetDirection()))
    image_size = np.array(list(image.GetSpacing()))
    volume = image_dim*image_size
    # print(f'imfusion centre {volume}')
    centre2corner = image.TransformContinuousIndexToPhysicalPoint(volume/2)


    origin_ct = np.array(image.GetOrigin())
    origin_imf = np.array(list(image.TransformContinuousIndexToPhysicalPoint([(image_dim[0]-1)/2, (image_dim[1]-1)/2, (image_dim[2]-1)/2])))


    # Dicom Original 2 imfusion
    trans_ct2imf = np.zeros(3)
    trans_ct2imf[2] = -origin_imf[2]
    rot_ct2imf = np.array(image_dir).reshape(3,3)
    # rot_ct2imf = np.identity(3)
    tf_ct2imf = np.identity(4)
    tf_ct2imf[:3, :3] = rot_ct2imf
    tf_ct2imf[:3, 3] = trans_ct2imf
    


    
    if plot_flag == True:
        np.hstack((origin_ct,origin_imf))


        bounds = np.array([[1,1,1],
                    [1,512,1],
                    [512,512,1],
                    [512,512,651],
                    [1,512,651],
                    [1,1,651]])
        bounds_ct = image.TransformContinuousIndexToPhysicalPoint(np.array([0,0,0]))

        vol = np.vstack((bounds_ct,origin_ct,origin_imf))
        fig = plot_fids(vol)
        fig.show()
    return tf_ct2imf

def read_ini(path,geometry_path):
    config = configparser.ConfigParser()

    file_name = os.path.join(path,geometry_path)
    config.read(file_name)

    marker_id = path[8:12]
    keys = config.sections()
    marker_count = len(keys)-1
    # inlineTop = 1
    # inlineBottom = 3
    xCoord = []
    yCoord = []
    zCoord = []
    for k in keys:
        if k != 'geometry':
            xCoord.append (float(config[k]['x']))
            yCoord.append (float(config[k]['y']))
            zCoord.append (float(config[k]['z']))
            


    xx= np.array(xCoord)
    yy= np.array(yCoord)
    zz= np.array(zCoord)
    fids = np.vstack((xx,yy,zz)).transpose()
    return fids

def CamRoboRegistration(CT_Space,Robot_Space):
    CT_Space_pointCloud = o3d.geometry.PointCloud()
    CT_Space_pointCloud.points = o3d.utility.Vector3dVector(CT_Space)
    Robot_Space_pointCloud = o3d.geometry.PointCloud()
    Robot_Space_pointCloud.points = o3d.utility.Vector3dVector(Robot_Space)
    corres_mat = np.asarray([[0,0],[1,1],[2,2],[3,3]])
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    transformation_mat = p2p.compute_transformation(CT_Space_pointCloud, Robot_Space_pointCloud,o3d.utility.Vector2iVector(corres_mat))
    ErrorCalc_vec = []
    Src_CtPts = np.asarray(CT_Space_pointCloud.points)
    for i in range(len(Src_CtPts)):
        errCal_vec = list(Src_CtPts[i])
        errCal_vec.append(1)
        errCal_vec2 = np.dot(transformation_mat,errCal_vec)
        errCal_vec3 = np.asarray(errCal_vec2[0:3])
        ErrorCalc_vec.append(errCal_vec3)
    ErrorCalc = o3d.geometry.PointCloud()
    ErrorCalc.points = o3d.utility.Vector3dVector(ErrorCalc_vec)
    error = p2p.compute_rmse(ErrorCalc,Robot_Space_pointCloud,o3d.utility.Vector2iVector(corres_mat))
    print("Registration Error : {}".format(error))
    return transformation_mat,error
    
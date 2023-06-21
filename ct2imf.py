
import SimpleITK as sitk
import numpy as np
# import json
from numpy.linalg import inv
np.set_printoptions(suppress=True)
from functions import plot_fids
import configparser
import os
import open3d as o3d

def rot2tf(rot,pos):
        
    pos_s = np.array(pos)
    rot_matrix = rot
    temp= np.column_stack((rot_matrix,pos_s))
    tf= np.vstack((temp,[0,0,0,1]))
    return tf

def compute_ct2vtk(file_path,plot_flag=False):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(file_path)

    reader.SetFileNames(dicom_names)
    image = reader.Execute()


    image_dim = np.array(list(image.GetSize()))-1
    image_dir = np.array(list(image.GetDirection()))
    image_size = np.array(list(image.GetSpacing()))
    volume = image_dim*image_size 
    # print(f'imfusion centre {volume}')
    centre2corner = image.TransformContinuousIndexToPhysicalPoint(volume/2)


    origin_ct = np.array(image.GetOrigin())
    origin_imf = np.array(list(image.TransformContinuousIndexToPhysicalPoint([(image_dim[0]-1)/2, (image_dim[1]-1)/2, (image_dim[2]-1)/2])))

    image_orient = np.array(image_dir).reshape(3,3)

    
    pos = volume +  image_orient@origin_ct
    pos = np.array([volume[0]-origin_ct[0],volume[1]-origin_ct[1], volume[2]+origin_ct[2]])
    image_orient_temp = np.array([-1,0,0,0,1,0,0,0,-1]).reshape(3,3)
    ct2corner = rot2tf(image_orient_temp,pos)
    
    
    



    return ct2corner

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
    
import configparser
from json import tool
import logging
import numpy as np
from numpy.linalg import norm 
from scipy.spatial.transform import Rotation as R
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import urllib3
import json
from sklearn import linear_model
from itertools import combinations
import plotly.graph_objects as go
from scipy.ndimage import median_filter
from scipy.spatial.distance import cdist, euclidean
import time
import os
import math
import socket
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , mean_squared_error
# from log_function import setup_logger
from scipy.optimize import minimize
# from kalman_filter import ukf_position,ukf_orientation
http = urllib3.PoolManager(maxsize=10)
#logging
dir=os.getcwd()
log_path= os.path.join(dir,'Log')
#logging
if os.path.exists(log_path)==False:
    
    direc="Log"
    folder=os.path.join(dir,direc)
    fld=os.mkdir(folder)
# logging.basicConfig(filename='Log\\function.log',format='%(asctime)s || %(levelname)s || %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.DEBUG)
# connection_log = setup_logger('Connection','Log\\Connection.log')
# tool_registration_log = setup_logger('ToolRegistration','Log\\ToolRegistration.log')
def marker_loaded(geo_status,tool_marker,reference_marker):
        marker_count = 0
        for markers in geo_status:
            if markers[0][8:-4] == reference_marker or markers[0][8:-4] == tool_marker:
                # print(f'{markers[0][8:-4]} marker loaded')
                marker_count += 1
            
        # print('count of existing',marker_count,tool_marker,reference_marker)
        if marker_count < 2:   
            return 0
        else:
            return 1

def marker_loaded_one(geo_status,markername):
    marker_ct=0
    for marker in geo_status:
        if marker[0][8:-4] == markername:
            marker_ct+=1
    if marker_ct == 1:
        return 1
    else:
        return 0

def guard_detect(tool_tip,reference_marker,delay):
    guard_pos = []
    ref_pos = []
    global dist_arr
    dist_arr = []
    dirname=os.path.dirname('rest_atracsys')
    filename=os.path.join(dirname,'env\port.txt')
    fd=os.open(filename, os.O_RDONLY)
    red=os.read(fd,20)
    port= red.decode('utf-8')
    print(port)
    
    r = http.request('GET','http://'+port+':8081/GetCameraData')
    logging.info(r)
    json_dict = json.loads(r.data)
    # print(json_dict)
    
    RegisteredMarkerCount =  len(json_dict['RegisteredMarkersList'])
    FiducialDataCount = len(json_dict['FiducialDataList'])
    print(f'fiducial data count : {FiducialDataCount}')
    logging.info(f"Fiducial Data Count: {FiducialDataCount}")
    time.sleep(delay)
    if  RegisteredMarkerCount != 0: 
        for i in range(RegisteredMarkerCount):      
            if json_dict['RegisteredMarkersList'][i]["MarkerName"] == reference_marker:
                Marker1 = {}
                Marker1 = json_dict['RegisteredMarkersList'][i]
                ref_position = Marker1['Top']['point'] 
                ref_pos = [ref_position['x'],ref_position['y'],ref_position['z'] ]
                # print(f'reference marker pos {ref_pos}')

            if json_dict['RegisteredMarkersList'][i]["MarkerName"] == tool_tip:
                Marker0 = {}
                Marker0 = json_dict['RegisteredMarkersList'][i]
                pos = Marker0['Top']['point'] 
                position = [pos['x'],pos['y'],pos['z'] ] 
                # print(type(position))
                # position 1 and position2 are random fidicial data of all the retro balls
                for i in range(FiducialDataCount): 
                    
                    fid_array = json_dict['FiducialDataList']
                    global fiducials 
                    fiducials = fid_array
                    min_guard_dist =  norm(np.array(fid_array[i]) - np.array(position))
                    dist_arr.append(min_guard_dist)
                    # print(f'stray fiducial list {fiducials}')

        if len(dist_arr) > 0:
            arr = dist_arr
            minElement = np.amin(arr)
            print(f'The closest fiducials is at {minElement} mm distance from the tip: ')
            logging.info(f'The closest fiducials is at {minElement} mm distance from the tip: ')
            global result
            result = np.where(arr == np.amin(arr))
            # print('index',result[0][0])
            # print(fiducials[result[0][0]])
            if minElement > 1 and minElement < 14:
                guard_pos = fid_array[result[0][0]]
                print(f'guard pos {guard_pos}')
        

    if len(ref_pos)>0 and len(guard_pos)>0:
        ref_guard_dist = norm(np.array(ref_pos) - np.array(guard_pos))
        print(f'reference and guard marker distance {ref_guard_dist}')
        logging.info(f'reference and guard marker distance {ref_guard_dist}')

    return guard_pos,ref_pos

def clean_ini(filepath):
    def readfile(filepath): 
        with open(filepath, "r") as f: 
            for line in f:
                yield line

    lines = readfile(filepath)

    n_lines = ["\n%s" % line if "[Sect" in line else line for line in lines if line.strip()]

    f = open(filepath, "w")
    f.write("".join(n_lines).lstrip())
    f.close()
    
def data_fetch(method,tool_marker,reference_marker,delay,maxPoints):
    if method == 'baseplate' :
        vel_thresh = 0.1
        sample_size = 500
        lenFlag = True
        print('Please enter the aproximate length or place the tool in pivot point')

    elif method == 'static' :
        vel_thresh = 1.5
        sample_size = maxPoints
        lenFlag = True
        
    else:
        vel_thresh = 1.5
        sample_size = maxPoints
        lenFlag = True

    needle_marker  = tool_marker
    geometry = [needle_marker,reference_marker]
    needle_marker_pos = []
    needle_marker_quat = []
    reference_marker_pos = []
    reference_marker_quat = []
    fiducial1 = np.zeros(3)
    fiducial2 = np.zeros(3)
    fiducial3 = np.zeros(3)
    fiducial4 = np.zeros(3)

    # from get_json import Get_camera_quats
    #######################################
    global reg_counter 
    reg_counter = 0
    end_time = time.time() + 40
    ref_pos_list = []
    ref_quat_list = []
    needle_pos_list = []
    needle_quat_list = []

    fiducial1 = []
    fiducial2 = []
    fiducial3 = []
    fiducial4 = []

    tool = []
    vel_list = []
    
    tool_pos = []
    ref_pos = []
    tool_quat = []
    ref_quat = []
    fid1 = []
    fid2 = []
    fid3 = []
    fid4 = []

    length = 0
    err = 2
    time.sleep(delay)
    while reg_counter < sample_size:            
        if time.time() < end_time  :   
            camera_data = Get_camera_quats(geometry)
            # print(camera_data)
            if reference_marker in camera_data :
                reference_marker_quat =  camera_data[reference_marker][0]
                reference_marker_pos = camera_data[reference_marker][1]
                # print(' pos ',reference_marker_pos,"  quat ",reference_marker_quat)

            if needle_marker in camera_data :
                needle_marker_quat =  camera_data[needle_marker][0]
                needle_marker_pos = camera_data[needle_marker][1]


            if (needle_marker in camera_data) and (reference_marker in camera_data) :
                disp = norm(np.array([reference_marker_pos]) - np.array([needle_marker_pos]))
                # print(disp)
                tool.append(needle_marker_pos)
                if len(tool) == 2:   
                    velocity = norm(np.diff(np.array(tool),axis = 0),axis = 1)
                    tool.reverse()
                    tool.pop()
                    vel_list.append(velocity)
                    if ((disp < length+err) or lenFlag) and ((disp > length -err) or lenFlag) and velocity < vel_thresh:
                        needle_pos_list.append(needle_marker_pos)
                        needle_quat_list.append(needle_marker_quat)
                        ref_pos_list.append(reference_marker_pos)
                        ref_quat_list.append(reference_marker_quat)
                        fiducial1.append(camera_data[needle_marker][2])
                        fiducial2.append(camera_data[needle_marker][3])
                        fiducial3.append(camera_data[needle_marker][4])
                        fiducial4.append(camera_data[needle_marker][5])
                        # print('tool_marker',needle_marker_pos)
                
              
                        reg_counter = reg_counter + 1
                        print(f'count {reg_counter}')
        else:
            print('Time out')
            break
            
    tool_pos = np.array(needle_pos_list)
    ref_pos = np.array(ref_pos_list) 
    tool_quat = np.array(needle_quat_list)
    ref_quat = np.array(ref_quat_list)
    fid1 = np.array(fiducial1)
    fid2 = np.array(fiducial2)
    fid3 = np.array(fiducial3)
    fid4 = np.array(fiducial4)

    
    return(tool_pos,tool_quat,ref_pos,ref_quat,fid1,fid2,fid3,fid4,reg_counter)

def geometric_median_old(X, eps=0.05):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1
def geometric_median(X, eps,numIter = 100):
    """
    Compute the geometric median of a point sample.
    The geometric median coordinates will be expressed in the Spatial Image reference system (not in real world metrics).
    We use the Weiszfeld's algorithm (http://en.wikipedia.org/wiki/Geometric_median)

    :Parameters:
     - `X` (list|np.array) - voxels coordinate (3xN matrix)
     - `numIter` (int) - limit the length of the search for global optimum

    :Return:
     - np.array((x,y,z)): geometric median of the coordinates;
    """
    # -- Initialising 'median' to the centroid
    y = np.mean(X,0)
    # -- If the init point is in the set of points, we shift it:
    while (y[0] in X[0]) and (y[1] in X[1]) and (y[2] in X[2]):
        y+=0.1

    convergence=False # boolean testing the convergence toward a global optimum
    dist=[] # list recording the distance evolution

    # -- Minimizing the sum of the squares of the distances between each points in 'X' and the median.
    i=0
    while ( (not convergence) and (i < numIter) ):
        num_x, num_y, num_z = 0.0, 0.0, 0.0
        denum = 0.0
        m = X.shape[1]
        d = 0
        for j in range(0,m):
            div = np.linalg.norm(X[j]-y)
            num_x += X[0,j] -y[0]/ div
            num_y += X[1,j]-y[1] / div
            num_z += X[2,j] -y[2]/ div
            denum += 1./div
            d += div**2 # distance (to the median) to miminize
        dist.append(d) # update of the distance evolution

        if denum == 0.:
            # warnings.warn( "Couldn't compute a geometric median, please check your data!" )
            return [0,0,0]

        y = [num_x/denum, num_y/denum, num_z/denum] # update to the new value of the median
        if i > 3:
            convergence=(abs(dist[i]-dist[i-2])<eps) # we test the convergence over three steps for stability
            #~ print abs(dist[i]-dist[i-2]), convergence
        i += 1
    if i == numIter:
        raise ValueError( "The Weiszfeld's algoritm did not converged after"+str(numIter)+"iterations !!!!!!!!!" )
    # -- When convergence or iterations limit is reached we assume that we found the median.

    return np.array(y)



def distanceFromLine(p1,p2,p3):
    d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
    return d

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def sphere(radius,centre,angle):
# The function will rotate the points and translate to the centre point
    vec = np.array([1,1,1])
    vec = (vec/norm(vec))*radius
    rot = R.from_quat([0,0,np.sin(np.pi/8),np.cos(np.pi/8)])
    rot2 = R.from_rotvec([np.random.randint(10)*0.1,np.random.randint(10)*0.1,np.deg2rad(angle)])
    rot_vec = rot2.apply(vec)
    rot_vec = rot_vec + centre

    return rot_vec

def rotate_vec(r,p_vec):
    rotated_vec = []
    for i in range (len(p_vec)):
        rot_vec =r[i].apply(p_vec[i])
        rotated_vec.append(rot_vec)  
    return np.array(rotated_vec)


def plot_vec(p_vec):
    xdata = np.array(p_vec).transpose()[0]
    ydata = np.array(p_vec).transpose()[1]
    zdata = np.array(p_vec).transpose()[2]

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
    



geometry = ['1500', '4200']
def Get_camera_quats(geometry):
    RegisteredMarkerCount = 0
    data = {}
    # r = http.request('GET','http://localhost:8081')
    # port = 'http://172.16.101.138'
    dirname=os.path.dirname('rest_atracsys')
    filename=os.path.join(dirname,'env\port.txt')
    fd=os.open(filename, os.O_RDONLY)
    red=os.read(fd,20)
    port= red.decode('utf-8')
        # print(port)
    try:
        r = http.request('GET',port+':8081/GetCameraData')
        
        json_dict = json.loads(r.data)
        # print(json_dict)
        RegisteredMarkerCount =  len(json_dict['RegisteredMarkersList'])
    # print(json_dict)
    except:
        print('Connection Error')
    

    if  RegisteredMarkerCount != 0: 
        for i in range(RegisteredMarkerCount):
            for Markers in geometry: 
                if json_dict['RegisteredMarkersList'][i]["MarkerName"] == Markers:
                    Marker0 = {}
                    Marker0 = json_dict['RegisteredMarkersList'][i]
                    rot = Marker0['Top']['rotation']
                    pos = Marker0['Top']['point']
                    pos1 = Marker0['Top']['point1']
                    pos2 = Marker0['Top']['point2']
                    pos3 = Marker0['Top']['point3']
                    pos4 = Marker0['Top']['point4']
                    position = [pos['x'],pos['y'],pos['z'] ] 
                    position1 =[pos1['x'],pos1['y'],pos1['z'] ]
                    position2 =[pos2['x'],pos2['y'],pos2['z'] ]
                    position3 =[pos3['x'],pos3['y'],pos3['z'] ]
                    position4 =[pos4['x'],pos4['y'],pos4['z'] ]
                    quat = [ rot['x'],rot['y'],rot['z'],rot['w'] ]
                    #position 1 and position2 are random fidicial data of all the retro balls

                    data[Markers] = (quat,position,position1,position2,position3,position4) 
                    logging.info(f"{data[Markers]}")
                    

    else:
        print("Marker not visible")
        logging.error("Marker Not Visibile")
        
    
    return data


def ransac(n,pivot_len):
    X=np.linspace(1,n,n)[:,np.newaxis]
    y = pivot_len
    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',
                # label='Inliers')
    # plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.',
                # label='Outliers')


    return inlier_mask

def major_minor_angle(pivot_vector):
    angle = []
    base_normal = np.mean(pivot_vector,axis = 0)
    for i in range(len(pivot_vector)):
        angle.append(angle_between(base_normal,pivot_vector[i]))

    # plt.ticklabel_format(useOffset=False)
    # plt.plot(angle)

    # plt.xlabel("sample")
    # plt.ylabel("angle in degrees")
    # plt.show(block=False)
    # plt.pause(2) # 3 seconds, I use 1 usually
    # plt.close("all")

    minor_angle = np.min(angle)
    major_angle = np.max(angle)
    return minor_angle,major_angle

def nor(v):
    return v/norm(v)

def ang(vector_1,vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle

def combination(marker_count,f):
    comb = combinations([x for x in range(marker_count)],2)

    imd = []
    for indices in comb:
    #     print(indices)
        dist = norm(f[indices[0]]-f[indices[1]])
        imd.append([indices[0],indices[1],dist])
        # print(np.rad2deg(ang(f[indices[0]],f[indices[1]])))
        # print(f[indices[0]],f[indices[1]])

    imd_array = np.array(imd)
    return imd_array

def rotateInline2X(fids,inlineTop,inlineBottom):
    v_inline = fids[inlineTop]-fids[inlineBottom]
    vector1 = v_inline
    if norm(vector1[1:]) != 0:
        a = vector1 / norm(vector1)
        b = np.array([-1,0,0])
        axis = np.cross(a,b)
        print(axis)
        axis = axis/norm(axis)

        angle = ang(a,b)
        r = R.from_rotvec(axis*angle)
        rot_fid = r.apply(fids-fids[inlineBottom,:])
        tip_vec = r.apply(-fids[inlineBottom,:])
        rot_fid_piv = rot_fid-tip_vec
    else:
        print("it is already inline with x axis")
        logging.info("It is already inline with x axis")
        rot_fid_piv = fids

    return rot_fid_piv
def rotateInline2theta(fids,inlineTop,inlineBottom,theta):
    v_inline = fids[inlineTop]-fids[inlineBottom]
    v_plane = fids[2]-fids[inlineBottom]
    vec_plane = v_plane/norm(v_plane)
    

    # if norm(v_inline[1:]) != 0:
    vec_diag = v_inline / norm(v_inline)
    vec_m_normal = np.cross(vec_plane,vec_diag)
    vec_m_normal = vec_m_normal/norm(vec_m_normal)  
    b = np.array([-1,0,0])
    
    if (b[0] == vec_diag[0]) & (b[1] == vec_diag[1]) & (b[2] == vec_diag[2]): 
        axis = np.array([0,0,-1])
    else:
        axis = np.cross(vec_diag,b)

    axis = axis/norm(axis)
    rot_tool = R.from_rotvec(vec_m_normal*np.deg2rad(theta))
    tool_axis = rot_tool.apply(vec_diag)
    org = [0,0,0]
    root = fids[inlineBottom]
    to_plot = np.vstack((org,root,root+vec_diag*50,root,root+tool_axis*50,root,fids[inlineTop]))
    fig = plot_fids(to_plot)
    # fig.show()
    print(np.rad2deg(ang(tool_axis,vec_diag)))

    angle = ang(tool_axis,b)
    print(f'angle to be rotated{np.deg2rad(angle)}')
    r = R.from_rotvec(axis*-angle)
    rot_fid = r.apply(fids-fids[inlineBottom,:])
    tip_vec = r.apply(-fids[inlineBottom,:])
    rot_fid_piv = rot_fid-tip_vec

    # else:
    #     print("it is already inline with x axis")
    #     rot_fid_piv = fids

    return rot_fid_piv

def rotateInline2Z(fids,inlineTop,inlineBottom):
    v_inline = fids[inlineTop]-fids[inlineBottom]
    vector1 = v_inline
    if norm(vector1[1:]) != 0:
        a = vector1 / norm(vector1)
        b = np.array([0,0,-1])
        axis = np.cross(a,b)
        print(axis)
        axis = axis/norm(axis)

        angle = ang(a,b)
        r = R.from_rotvec(axis*angle)
        rot_fid = r.apply(fids-fids[inlineBottom,:])
        tip_vec = r.apply(-fids[inlineBottom,:])
        rot_fid_piv = rot_fid-tip_vec
    else:
        print("it is already inline with y axis")
        logging.info("It is already inline with y axis")
        rot_fid_piv = fids

    return rot_fid_piv

def plot_fids(fids):
    fids=np.vstack((fids,[0, 0, 0]))
    fiducials = go.Scatter3d(
        x=fids[:,0], y=fids[:,1], z=fids[:,2],
        marker=dict(
            size=4,
            colorscale='Viridis',
        ),
        line=dict(
            color='darkblue',
            width=2
        )
    )
    axes   = go.Scatter3d( x = [0, 0,   0  , 100, 0, 0  ],
                           y = [0, 100, 0  , 0,   0, 0  ],
                           z = [0, 0,   0  , 0,   0, 100],
                           marker = dict( size = 1,
                                          color = "rgb(84,48,5)"),
                           line = dict( color = "rgb(84,48,5)",
                                        width = 6)
                         )
    data = [fiducials,axes]
    name = 'default'
# Default parameters which are used when `layout.scene.camera` is not provided
    camera = dict(
        up=dict(x=-1, y=0, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=1.25)
        )

    fig = go.Figure(data=data)



    fig.update_layout(scene_camera=camera, title=name)

    fig.update_layout(
        scene = dict(
            xaxis = dict(nticks=4, range=[-4000,4000],),
            yaxis = dict(nticks=4, range=[-4000,4000],),
            zaxis = dict(nticks=4, range=[-4000,4000],),
            ),
            width=700,
            margin=dict(r=10, l=10, b=10, t=10))
    return fig

def median3d(a,n,m):
    return median_filter(a,footprint = np.ones((n,m)))

def unit(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def getdivot_x_axis(df,TransformationMatrix):
        
        
        
        
        divot_pos = df[['refx','refy','refz']].to_numpy()
        divot_quat = df [['ref_qx','ref_qy','ref_qz','ref_qw']].to_numpy()
        tool_poss = df[['toolx','tooly','toolz']].to_numpy()
        tool_quatt = df [['tool_qx','tool_qy','tool_qz','tool_qw']].to_numpy()
        tool_fid1 = df [['fid1x','fid1y','fid1z']].to_numpy()
        tool2divot_tf=[]
        divot2tool_tf=[]
        x_axis=[]
        
        Guide2GuideNew_tf = TransformationMatrix
        inline_stack = []
        pos_stack = []
        # divot_pos = ukf_position(divot_pos,dt=0.1,P=0.2,Z=0.1)
        # divot_quat = ukf_orientation(divot_quat)
        # tool_poss = ukf_position(tool_poss,dt=0.1,P=0.2,Z=0.1)
        # tool_quatt = ukf_orientation(tool_quatt)
        
        # divot_pos = np.array(divot_pos)
        # divot_quat = np.array(divot_quat)
        # tool_poss = np.array(tool_poss)
        # tool_quatt = np.array(tool_quatt)
        #divot2cam
        for i in range(len(divot_pos)):
            divot2cam_r=R.from_quat(divot_quat[i]).as_matrix().transpose()
            tool2cam_r= R.from_quat(tool_quatt[i]).as_matrix().transpose()
            
            divot2cam=rot2tf(divot2cam_r,divot_pos[i])
            cam2divot = np.linalg.inv(divot2cam)
            tool2cam= rot2tf(tool2cam_r,tool_poss[i])
            cam2tool = np.linalg.inv(tool2cam)
            divot2tool = cam2tool@divot2cam
            tool2divot = np.linalg.inv(divot2tool)
            print(cam2tool@np.hstack((tool_fid1[0],[1])))
            
            tool2divot_tf.append(tool2divot)
            divot2tool_tf.append(divot2tool)

            if np.linalg.norm(TransformationMatrix) != 0:
                GuideNew2Tool_tf = divot2tool @ np.linalg.inv(Guide2GuideNew_tf)
                inline_stack.append(GuideNew2Tool_tf[:3,0])
            # x_divot = tool2divot@[1,0,0,1]
            # x_tool = divot2tool@[1,0,0,1]
            # print(x_tool,x_divot)
            # x_axis.append(x_divot[i])  
            else:
                inline_stack.append(divot2tool[:3,0])
            # pos_stack.append(divot2tool[:3,3])
            
        inline= np.mean(np.array(inline_stack),axis=0)   
        return inline
    
def rot2tf(rot,pos):
    
    pos_s = np.array(pos)
    rot_matrix = rot
    temp= np.column_stack((rot_matrix,pos_s))
    tf= np.vstack((temp,[0,0,0,1]))
    return tf

def rotateInline2X_baseplate(fids,divot_x):
    
    vector1 = -divot_x
    if norm(vector1[1:]) != 0:
        a = vector1 / norm(vector1)
        b = np.array([-1,0,0])
        axis = np.cross(a,b)
        print(axis)
        axis = axis/norm(axis)

        angle = ang(a,b)
        r = R.from_rotvec(axis*angle)
        rot_fid_piv = r.apply(fids)
        
    else:
        print("it is already inline with x axis")
        logging.info("It is already inline with x axis")
        rot_fid_piv = fids

    return rot_fid_piv

def connectionCheck():
    
    atarc = configparser.ConfigParser()
    atarc.read('Config\\NavigationConfig.ini')
    atarc_ip = atarc.get("Atracsys","IP")
    atarc_port = atarc.get("Atracsys","Port")
    IP=atarc_ip
    PORT = int(atarc_port)
    
    s= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   
    time.sleep(1)
    res = s.connect_ex((IP,PORT))
    # connection_log.info(f'Response Code: {res}')
    if res == 0:
        s.close()
        return 1
    elif res == 10061:
        return 0
    else:
        return 0
         
   
    
        

def distance(p1,p2):
    squared_dist = np.sum((p1-p2)**2, axis=0)
    dist = np.sqrt(squared_dist)
    # print(dist)
 
    return dist   

def write_geo_calib(geo_path,calibplate_name,radius):

    #New Geometry File
    newGeo = configparser.ConfigParser()
    
    
    #Taking Values from tool config
    # toolConfig = configparser.ConfigParser()
    # toolConfig.read("Config\\toolConfig.ini")
    # tool_id = toolConfig.sections()
    theta = 45.26066
    # for t in tool_id:
    #     if t == tool_name:
    #         r = float(toolConfig.get(t,'diameter'))/2
            
    l = radius / np.sin(np.deg2rad(theta))
    
    # print(l)
    
    # l=0
    
    config = configparser.ConfigParser()
    
    #Taking Values from Original Geometry File(Calib Plate)
    config.read(geo_path+"\\SeedGeometry\\geometry"+calibplate_name+'.ini')
    keys = config.sections()
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
    keys = config.sections()
    marker_count = len(keys)-1
    
    
    for i in range(marker_count):
        # section.append('fiducial'+str(i))
        newGeo.add_section('fiducial'+str(i))
    newGeo.add_section('geometry')
    value = np.vstack((xx,yy,zz)).transpose()
    offset = np.array([0,l,0])
    
    num = 0
    # offs_values = []
    for k in keys:
        if k != 'geometry':
            
            xval,yval,zval =  value[num,:] - offset
            # offs_values.append([xval,yval,zval])
            newGeo.set(k,'x',str(xval))
            newGeo.set(k,'y',str(yval))
            newGeo.set(k,'z',str(zval))
            num = num + 1
    
    ov = value -offset
    vo = np.vstack((value,ov))
    fig = plot_fids(vo)
    # fig.show()      
    id=calibplate_name
    newGeo.set('geometry','count',str(marker_count))
    newGeo.set('geometry','id',id)
    toolConfig = configparser.ConfigParser()
    toolConfig.read('Config\\toolConfig.ini')
    template_name = int(toolConfig.get('Template','name'))                     
    path_geo = geo_path+"\\geometry"+calibplate_name+".ini"
    if os.path.exists(path_geo) == True:
        
        os.remove(path_geo)
    else:
        pass
    cfgfile = open(geo_path+"\\geometry"+calibplate_name+'.ini','w')
    newGeo.write(cfgfile,space_around_delimiters=False)
    cfgfile.close()  
    clean_ini(geo_path+"\\geometry"+calibplate_name+'.ini')
    
    return offset


def compute_tre(vector,target):
    vector = np.array(vector)



    # add noise to system
    # vector = vector + noise

    
    centroid = np.mean(vector,axis=0)

    axis =[]
    temp_vec =  unit(vector[1] -  centroid)

    temp_vec2 = np.cross(temp_vec,unit(vector[2]-centroid))
    temp_vec3 = np.cross(temp_vec,temp_vec2)
    axis.append(temp_vec)
    axis.append(temp_vec2)
    axis.append(temp_vec3)


    # fiducials to principal axis
    # target from principal axis 
    tp = []
    tp.append(distanceFromLine(centroid,centroid + axis[0],target))
    tp.append(distanceFromLine(centroid,centroid + axis[1],target))
    tp.append(distanceFromLine(centroid,centroid + axis[2],target))

    fp = []
    f = np.zeros(3)
    for i in range(3):
        sum = 0
        for j in range(len(vector)):
            fp = np.around(distanceFromLine(centroid,centroid+axis[i],vector[j]),3)
            # print(fp)
            sum = sum + fp**2
        if np.round(sum/len(vector)) == 0:
            f[i]=10000
            # print(f[i])
        else:
            f[i] = tp[i]**2/(sum/len(vector))
            
        
        

    fle = 0.3
    tre = (fle**2/len(vector))*(np.mean(f)+1)
    return np.sqrt(np.around(tre,3))

def compute_tre_geo(geo_path,geometry_id, target):
    geo = configparser.ConfigParser()

    geo.read(geo_path+"\\"+geometry_id+".ini")

    keys = geo.sections()
    Xval =[]
    Yval = []
    Zval = []
    for k in keys:
        if k != 'geometry':
            Xval.append(float(geo[k]['x']))
            Yval.append(float(geo[k]['y']))
            Zval.append(float(geo[k]['z']))
            
    arr = np.vstack((Xval,Yval,Zval)).transpose()

    
    tre = compute_tre(arr,target)
    return tre

        
def adaptive_cost_estimation(input_data): 
           
        data_features = pd.read_csv('ToolCalibrationCameraData\\Pivot_data.csv')
        drop = ['Tool Length','Optimality','VelocityPerUnitLength','Cost','Velocity']
        data_features = data_features.drop(drop,axis=1)
        
        
        inp = pd.DataFrame([input_data],columns=data_features.columns)
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(data_features)
        test_data = scaler.transform(inp)
        reg_coeff = 1.55746575
        reg_inter = 2.5
        predicted_cost = reg_coeff*test_data[0] + reg_inter
        
        
        return predicted_cost

def check_folder(directory,folder_name):
    check_status = 0
    path = os.path.join(directory,folder_name)
    
    if os.path.exists(path) != True:
        print('Folder not exists')
        check_status = 0
    else:
        print('Folder already exists')
        check_status = 1
    return check_status


def binning(df_in):
    
        df_in.loc[df_in['tool_angle'].between(0,10,'both'),'Weights']= 5
        df_in.loc[df_in['tool_angle'].between(10,20,'both'),'Weights']= 4
        df_in.loc[df_in['tool_angle'].between(20,30,'both'),'Weights']= 3
        df_in.loc[df_in['tool_angle'].between(30,40,'both'),'Weights']= 2
        df_in.loc[df_in['tool_angle'].between(40,180,'both'),'Weights']= 1
    
    
        
        return df_in

def svd(tool_pos):
    #Plane Fitting
    arr = tool_pos
    centeroid = np.mean(arr,axis =1,keepdims=True)
    arr_centered = arr - centeroid
    U,s,VT = np.linalg.svd(arr_centered)
    # U is the basis and the normal vector of the plane 
    normal_vector = U[:,-1]
    
    #After fitting plane use arr_centered for projecting the points onto plane
    
    return U,centeroid,arr_centered

def weight_average_dict(dict):
    dividend =[]
    divisor = []
    dict_weights = dict.keys()
    for w in dict_weights:
        divid = np.mean(dict[w])*w*len(dict[w])
        dividend.append(divid)
        divis = len(dict[w])*w
        divisor.append(divis)
        
    avg = np.sum(dividend)/np.sum(divisor)
    
    return avg

def rotational_geometric_median(X,numIter=200):
    y = R.from_quat(X).mean()
    y = y.as_quat()
    # y=np.mean(X,0)
    # -- If the init point is in the set of points, we shift it:
    while (y[0] in X[0]) and (y[1] in X[1]) and (y[2] in X[2]):
        y+=0.01

    convergence=False # boolean testing the convergence toward a global optimum
    dist=[] # list recording the distance evolution

    # -- Minimizing the sum of the squares of the distances between each points in 'X' and the median.
    i=0
    while ( (not convergence) and (i < numIter) ):
        num_x, num_y, num_z ,num_w= 0.0, 0.0, 0.0,0.0
        denum = 0.0
        m = X.shape[1]
        d = 0
        for j in range(0,m):
            div = np.minimum(norm(X[j]-y[0]),norm(X[j]+y[0]))
            # if div == 0.0:
            #     div = 1.0
            num_x += X[0,j] / div
            num_y += X[1,j] / div
            num_z += X[2,j] / div
            num_w += X[3,j] / div
            denum += 1./div
            d += div**2 # distance (to the median) to miminize
        dist.append(d) # update of the distance evolution

        if denum == 0.:
            # warnings.warn( "Couldn't compute a geometric median, please check your data!" )
            return [0,0,0]

        y = [num_x/denum, num_y/denum, num_z/denum,num_w/denum] # update to the new value of the median
        if i > 3:
            convergence=(abs(dist[i]-dist[i-2])<0.1) # we test the convergence over three steps for stability
            #~ print abs(dist[i]-dist[i-2]), convergence
        i += 1
    if i == numIter:
        raise ValueError( "The Weiszfeld's algoritm did not converged after"+str(numIter)+"iterations !!!!!!!!!" )
    # -- When convergence or iterations limit is reached we assume that we found the median.

    return np.array(y)

def geometric_median_rotation(X, eps=0.05):
    y = R.from_quat(X).mean()
    y = y.as_quat()

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            Rr = (T - y) * Dinvs
            r = np.linalg.norm(Rr)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1

def time_bounded_velocity(pos,timestamp):
    v = np.linalg.norm(np.diff((pos[0],pos[1]),axis=0))/np.diff((timestamp[0][0],timestamp[1][0]))
    return v[0]


def ClassObj2JSON(ClassObj):
    #Input is Class Obj , Returns JSON Formatted Data
    return json.loads(json.dumps(ClassObj.__dict__))


def TakeTransformationMatrixTemplate(template_name,type):
    
    if type == 'Template':
        MatrixFileName = 'Transformation'
        currentDir = os.getcwd()
        FolderDirectory = currentDir+'\\TransformationMatrices\\'
        folder_status = check_folder(currentDir,'TransformationMatrices')
        
        if folder_status == 1:
            TemplateName = configparser.ConfigParser()
            TemplateName.read('Config\\toolConfig.ini')
            
            TempName = int(TemplateName.get('Template','name'))
            
            for i in range(2,5):
                
                if int(template_name) == TempName+i:
                    
                    try:
                    
                        fileName = MatrixFileName+str(i-1)+'.txt'
                        path = FolderDirectory+fileName
                        TransformationMatrix = np.loadtxt(path)
                    except:
                        tool_registration_log.error('Template Transformation file not found')
                        return 0
        else:
            tool_registration_log.error('Transformation Matrices folder not found')
            return 0
    
    elif type == 'PRM':
        MatrixFileName = 'PRMTransformation'
        currentDir = os.getcwd()
        folder_status = check_folder(currentDir,'TransformationMatrices')
        
        if folder_status == 1:
           try: 
                FolderDirectory = currentDir+'\\TransformationMatrices\\'
                path = FolderDirectory+MatrixFileName+'.txt'
                TransformationMatrix = np.loadtxt(path)
           except:
               
               tool_registration_log.error('PRM transformation file not found')
               return 0
        else:
            tool_registration_log.error('Transformation Matrices folder not found')
            return 0
                 
    return TransformationMatrix

def OffsetAdditionTemplate(TransformationMatrix, ToolName):
    
    toolData = configparser.ConfigParser()
    toolData.read('Config\\toolConfig.ini')
    
    theta = float(toolData.get('theta','theta'))
    
    toolNames = toolData.sections()
    
    for i in toolNames:
        
        if i == ToolName:
            
            radius = float(toolData.get(i,'radius'))
            
            if radius !=0 :
                l = radius / np.sin(np.deg2rad(theta))
                offset = np.array([0,l,0])
                
                TransformationMatrix[:3,3]=TransformationMatrix[:3,3] - offset
            else:
                tool_registration_log.error('Offset Addition Failed: Check the radius value in config')
                TransformationMatrix = 0
            
    
    return TransformationMatrix
            
            
            
            
            
            
    
    
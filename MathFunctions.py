from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist, euclidean
from scipy.spatial import distance_matrix
import numpy as np
import SimpleITK as sitk
import open3d as o3d
    
def createPoseMatrix(rotationMatrix:np.array, 
                        positionVector:np.array):
    
    temp = np.column_stack((rotationMatrix,positionVector))
    transformationMatrix = np.vstack((temp,[0,0,0,1]))
    
    return transformationMatrix

def createTransformationMatrix(pos:np.array, quat:np.array):
    
    r_marker2cam = R.from_quat(quat).as_matrix().transpose()
    tf_marker2cam = createPoseMatrix(r_marker2cam,pos)
    
    return tf_marker2cam

def transformPoints(points:np.array, transformationMatrix:np.array):
    
    pointsHomogenous = np.hstack((points,np.ones((len(points),1))))
    transformedPointsHm = (transformationMatrix @ 
                            pointsHomogenous.transpose())
    transformedPoints = transformedPointsHm.transpose()[:,:3]
    
    return transformedPoints

def CTtoVTK(dicomFilesPath:str):
    
    dicomReader = sitk.ImageSeriesReader()
    dicomNames = dicomReader.GetGDCMSeriesFileNames(dicomFilesPath)
    dicomReader.SetFileNames(dicomNames)
    dicomImage = dicomReader.Execute()
    
    ImageDim = np.array(list(dicomImage.GetSize()))
    ImageDir = np.array(list(dicomImage.GetDirection()))
    ImageOrient = np.array(ImageDir).reshape(3,3)
    ImageSpacing = np.array(list(dicomImage.GetSpacing()))
    volumeCT = (ImageDim-1)*ImageSpacing
    CTOrigin = np.array(dicomImage.GetOrigin())
    
    positionVector = volumeCT + ImageOrient @ CTOrigin
    alteredImageOrient = ImageOrient @ np.array([1,0,0,
                                                    0,-1,0,
                                                    0,0,-1]).reshape(
                                                            3,3)    
    transformationCTtoVTK = createPoseMatrix(alteredImageOrient,
                                             positionVector)
    
    
    return transformationCTtoVTK

def geometricMedianFilter(X, eps=0.05):
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

def filterData(position:np.array, quaternion:np.array):
    if position.shape[0] == 1:
        return position[0], quaternion[0]
    else:
        quaternionFiltered = R.from_quat(quaternion).mean().as_quat()
        positionMedian = geometricMedianFilter(position)
    
        return positionMedian, quaternionFiltered

def estimatePointToPointError(sourcePoints:np.array, 
                              targetPoints:np.array):
    error = []
    for i in range(len(sourcePoints)):
        min_error = float("inf")
        relative_difference = np.abs(targetPoints - sourcePoints[i])  
        for k in range(len(relative_difference)):
            if relative_difference[k] < min_error:
                min_error = relative_difference[k]
        error.append(min_error)
    return np.array(error)

def pointBasedRegistration(sourcePoints:np.array, 
                           targetPoints:np.array):
    src_pointCloud = o3d.geometry.PointCloud()
    src_pointCloud.points = o3d.utility.Vector3dVector(sourcePoints)
    tgt_pointCloud = o3d.geometry.PointCloud()
    tgt_pointCloud.points = o3d.utility.Vector3dVector(targetPoints)
    ind = np.arange(len(sourcePoints))
    cor = np.hstack((ind,ind))
    
    corres_mat = np.vstack((ind,ind)).transpose()
    p2p = (o3d.pipelines.registration
           .TransformationEstimationPointToPoint())
    transformation_mat = p2p.compute_transformation(
                                src_pointCloud, 
                                tgt_pointCloud,
                                o3d.utility.Vector2iVector(corres_mat))
    ErrorCalc_vec = []
    Src_CtPts = np.asarray(src_pointCloud.points)
    for i in range(len(Src_CtPts)):
        errCal_vec = list(Src_CtPts[i])
        errCal_vec.append(1)
        errCal_vec2 = np.dot(transformation_mat,errCal_vec)
        errCal_vec3 = np.asarray(errCal_vec2[0:3])
        ErrorCalc_vec.append(errCal_vec3)
    ErrorCalc = o3d.geometry.PointCloud()
    ErrorCalc.points = o3d.utility.Vector3dVector(ErrorCalc_vec)
    error = p2p.compute_rmse(ErrorCalc,tgt_pointCloud,
                             o3d.utility.Vector2iVector(corres_mat))
    return transformation_mat,error

def recursive_combinations(input):
    
    if len(input) == 1:
        return [[a] for a in range(input[0])]
    outcomes = []
    num_of_a_in_current_set= input[0]
    remaining_sets = input[1:]
    for a in range(num_of_a_in_current_set):
        sub_outcomes = recursive_combinations(remaining_sets)
        for sub_outcome in sub_outcomes:
            outcomes.append([a] + sub_outcome)
            
    return outcomes 
def check_size(src,tgt):
    if src.shape == tgt.shape:
        rows_diff = 0

    else:
        if src.size > tgt.size:
            small_matrix = tgt
            big_matrix = src
        else:
            small_matrix = src
            big_matrix = tgt
            
        rows_diff = big_matrix.shape[0] - small_matrix.shape[0]

        if rows_diff > 0:
            zeros = np.zeros((rows_diff, small_matrix.shape[1]))
            small_matrix = np.concatenate((small_matrix, zeros), axis=0)
        
        if src.size > tgt.size:
            tgt = small_matrix
        else:
            src = small_matrix

    return src, tgt, rows_diff

def registrationWithoutCorrespondence(src:np.array, 
                                      tgt:np.array):
    
    # check for stray points
    src,tgt,rows_diff = check_size(src,tgt) 

    # intermarker distance calculation
    dist_src = distance_matrix(src,src,p=2)
    dist_tgt = distance_matrix(tgt,tgt,p=2)

    corres = {}
    intersec_ip = {}
    iterr_i = 0
    dimensions = 3
    reg_error = np.inf

    for i in range(len(dist_src)):
        corres[i] = []
        for j in range(len(dist_tgt)):
            err = estimatePointToPointError(dist_src[i],dist_tgt[j])
            src_count = len(dist_src)-rows_diff
            if np.sum(err<1) >= dimensions and i <= src_count:
                for iterr_i in range(len(dist_src)):
                    for iterr_j in range(len(dist_tgt)):
                        element_err = np.abs(dist_src[i, iterr_i] - dist_tgt[j, iterr_j])
                        if element_err < 1 and i <= len(dist_src)-rows_diff :
                            if iterr_i in corres.keys():
                                corres[iterr_i].append(iterr_j)
                            else:
                                corres[iterr_i] = [iterr_j]
        
        intersec_ip[i] = corres
        print(corres)

        #backtrack src and tgt points
        src_ind = []
        tgt_ind = []
        src_ind_count = []
        tgt_ind_count = []

        for inp in corres.keys():
            
            src_ind.append(inp)
            src_ind_count.append(np.sum(inp==inp))

        for inp in corres.values():
            tgt_ind.append(inp[0] if len(inp)!=0 else [])
            tgt_ind_count.append(len(inp))

        if rows_diff != 0 :
            src_in = np.array(src)[:-rows_diff]
        else:
            src_in = np.array(src)
        
        tgt_in = np.array(tgt)

        if np.sum(src_ind_count)==np.sum(tgt_ind_count):
            val = [x for x in src_ind]
            val1 = []
            for y in range(len(list(tgt_ind))):
                val1.append(list(tgt_ind)[y])
            src_points = src_in[val,:]
            tgt_points = tgt_in[val1,:]
            tf_x2y,err = pointBasedRegistration(src_points,tgt_points)
            if err < reg_error:
                reg_error=err
                tf_x2y_minErr = tf_x2y

        else:
            all_possible_outcomes = recursive_combinations(tgt_ind_count)
            val = [x for x in src_ind if x<len(src_in)]
            
            src_points = src_in[val,:]
            
                
            for tgt_combination, outcome in enumerate(all_possible_outcomes):
                # tgt_points=[]
                val1 = []
                for i in range(len(corres.values())):
                    indexeOfY = list(corres.values())[i][outcome[i]]
                    val1.append(indexeOfY)
                tgt_points = tgt[val1,:]
                tf_x2y,err = pointBasedRegistration(src_points,tgt_points)
                if err < reg_error:
                    reg_error=err              
                    tf_x2y_minErr = tf_x2y

        corres = {}

    print("Corresponding indices of SRC and TGT")
    print("SRC Index \n", val)
    print("TGT Index \n", val1)
    print("Source Points \n", src_points)
    print("Target Points \n", np.asarray(tgt_points))
    print("Registration Error \n", reg_error)
    print("Transformation src2tgt \n", tf_x2y_minErr)

    # if plot == True and len(tf_x2y_minErr)!=0:
    #     plot_points = plot3d(src_points,tgt_points)

    #     tgt_pts_homogenous = np.column_stack((np.matrix(tgt_points), np.ones((len(tgt_points), 1))))
    #     src_transformed_pts = tgt_pts_homogenous@tf_x2y_minErr
    #     src_transformed_pts[:,:3]

    #     fig1 = plt.figure()
    #     ax1 = fig1.add_subplot(111, projection='3d')
    #     ax1.scatter(src_points[:,0], src_points[:,1], src_points[:,2], c='green', label='X_source')
    #     ax1.scatter(src_transformed_pts[:,0], src_transformed_pts[:,1], src_transformed_pts[:,2], c='yellow', label='X_transformed')
    #     plt.show()

    return tf_x2y_minErr, reg_error
            
            
             
        
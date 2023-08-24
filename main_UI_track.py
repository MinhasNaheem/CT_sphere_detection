#!/usr/bin/env python

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
import vtk
from vtkmodules.vtkCommonCore import (
    VTK_VERSION_NUMBER,
    vtkVersion
)
from vtkmodules.vtkFiltersCore import (
    vtkFlyingEdges3D,
    vtkMarchingCubes,
    vtkStripper
)
from vtkmodules.vtkFiltersModeling import vtkOutlineFilter
from vtkmodules.vtkIOImage import vtkMetaImageReader
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCamera,
    vtkPolyDataMapper,
    vtkProperty,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
from numpy.linalg import inv,norm
from vtk.util import numpy_support
from ct2imf import compute_ct2vtk
import numpy as np

from cam_parser import CameraAquisition
def main():
    tool_marker = "11005"
    reference_marker = "54320"
    # dicom_dir = r"D:\Navigation\CT_sphere_detection\Dataset-4-20230610T054436Z-001\Dataset-4\DICOM\PA0\ST0\SE1"
    # ref2ct_tf = np.load('ref2ct_Oarm.npy')
    # dicom_dir = r"D:\Navigation\Carm_registration\Dataset-2\DICOM\PA0\ST0\SE1"
    dicom_dir = r"D:\Navigation\Carm_registration\DICOM\PA0\ST0\SE1"

    ct2vtk_tf = compute_ct2vtk(dicom_dir)

    ref2ct_tf = np.load('ref2CT.npy')
    cam = CameraAquisition(tool_marker,reference_marker)

    global rotation_angle, rotation_increment
    # vtkFlyingEdges3D was introduced in VTK >= 8.2
    use_flying_edges = vtk_version_ok(8, 2, 0)
    colors = vtkNamedColors()
    colors.SetColor('SkinColor', [240, 184, 160, 255])
    colors.SetColor('BackfaceColor', [255, 229, 200, 255])
    colors.SetColor('BkgColor', [51, 77, 102, 255])
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(dicom_dir)



    # An isosurface, or contour value of 500 is known to correspond to the
    # skin of the patient.
    # The triangle stripper is used to create triangle strips from the
    # isosurface these render much faster on many systems.
    if use_flying_edges:
        try:
            skin_extractor = vtkFlyingEdges3D()
        except AttributeError:
            skin_extractor = vtkMarchingCubes()
    else:
        skin_extractor = vtkMarchingCubes()
    skin_extractor.SetInputConnection(reader.GetOutputPort())
    skin_extractor.SetValue(-1024, 500)

    skin_stripper = vtkStripper()
    skin_stripper.SetInputConnection(skin_extractor.GetOutputPort())

    skin_mapper = vtkPolyDataMapper()
    skin_mapper.SetInputConnection(skin_stripper.GetOutputPort())
    skin_mapper.ScalarVisibilityOff()

    skin = vtkActor()
    skin.SetMapper(skin_mapper)
    skin.GetProperty().SetDiffuseColor(colors.GetColor3d('SkinColor'))
    skin.GetProperty().SetSpecular(0.3)
    skin.GetProperty().SetSpecularPower(20)
    skin.GetProperty().SetOpacity(0.5)

    back_prop = vtkProperty()
    back_prop.SetDiffuseColor(colors.GetColor3d('BackfaceColor'))
    skin.SetBackfaceProperty(back_prop)

    # An isosurface, or contour value of 1150 is known to correspond to the
    # bone of the patient.
    # The triangle stripper is used to create triangle strips from the
    # isosurface these render much faster on may systems.
    if use_flying_edges:
        try:
            bone_extractor = vtkFlyingEdges3D()
        except AttributeError:
            bone_extractor = vtkMarchingCubes()
    else:
        bone_extractor = vtkMarchingCubes()
    bone_extractor.SetInputConnection(reader.GetOutputPort())
    # bone_extractor.SetValue(-100, 3000)

    bone_stripper = vtkStripper()
    bone_stripper.SetInputConnection(bone_extractor.GetOutputPort())

    bone_mapper = vtkPolyDataMapper()
    bone_mapper.SetInputConnection(bone_stripper.GetOutputPort())
    bone_mapper.ScalarVisibilityOff()

    bone = vtkActor()
    bone.SetMapper(bone_mapper)
    bone.GetProperty().SetDiffuseColor(colors.GetColor3d('Ivory'))

    # An outline provides context around the data.
    #
    outline_data = vtkOutlineFilter()
    outline_data.SetInputConnection(reader.GetOutputPort())

    map_outline = vtkPolyDataMapper()
    map_outline.SetInputConnection(outline_data.GetOutputPort())

    outline = vtkActor()
    outline.SetMapper(map_outline)
    outline.GetProperty().SetColor(colors.GetColor3d('Black'))

    # It is convenient to create an initial view of the data. The FocalPoint
    # and Position form a vector direction. Later on (ResetCamera() method)
    # this vector is used to position the camera to look at the data in
    # this direction.
    a_camera = vtkCamera()
    a_camera.SetViewUp(0, 1, 0)
    a_camera.SetPosition(0, 0, 1)
    # a_camera.SetFocalPoint(0, 0, 0)
    a_camera.ComputeViewPlaneNormal()
    a_camera.Azimuth(-60.0)
    a_camera.Elevation(30.0)


    # Create an STL reader
    stl_reader = vtk.vtkSTLReader()
    stl_reader.SetFileName("NeedleTracker.stl")
    stl_reader.Update()

    # Create a mapper
    stl_mapper = vtk.vtkPolyDataMapper()
    stl_mapper.SetInputConnection(stl_reader.GetOutputPort())

    # Create an actor
    stl_actor = vtk.vtkActor()
    stl_actor.SetPosition(0, 0, 0)
    stl_actor.SetMapper(stl_mapper)


    ren2 = vtkRenderer()
    ren2.AddActor(outline)
    ren2.AddActor(skin)
    ren2.AddActor(bone)
    ren2.AddActor(stl_actor)
    ren2.SetActiveCamera(a_camera)
    ren2.ResetCamera()
    a_camera.Dolly(1.5)
    ren2.SetBackground(colors.GetColor3d('BkgColor'))
    ren2.ResetCameraClippingRange()

    
    # Create a renderer
    ren1 = vtk.vtkRenderer()
    ren1.AddActor(stl_actor)
    ren1.SetBackground(0.2, 0.3, 0.4)


    rotation_angle = 0.0
    rotation_increment = 5.0 
    ren_win = vtkRenderWindow()
    ren_win.AddRenderer(ren2)
    # ren_win.AddRenderer(ren1)

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)

    ren_win.SetSize(1000, 800)
    ren_win.SetWindowName('Minnumumumumumumumumumum')
        # Start the interactor
    iren.Initialize()
    


    # Define a function to update the actor's orientation
    def update_orientation():
        json_data = cam.parse_cam()
    
    
        marker_data = cam.Get_camera_quats(json_data)
        tool_data = marker_data.get(tool_marker,0)
        ref_data = marker_data.get(reference_marker,0)
        ref2cam_tf = []
        if ref_data != 0  and tool_data !=0:
                ref2cam_tf = cam.transform_data(ref_data[1],ref_data[0])
                tool2cam_tf = cam.transform_data(tool_data[1],tool_data[0])
                ref2tool_tf = inv(tool2cam_tf)@ref2cam_tf
                tool2ref_tf = inv(ref2tool_tf)
                tool2vtk_tf = ct2vtk_tf@ref2ct_tf@tool2ref_tf  
                tool_position = tool2vtk_tf[:3,3]
                print(f'position of tool in vtk frame is {tool_position}')
                
        else:
                tool2vtk_tf = ct2vtk_tf
  
                
        global rotation_angle
        transform = vtk.vtkTransform()
        if norm(ref2cam_tf) !=0 :

            transform.SetMatrix(list(tool2vtk_tf.ravel()))
        # print(rotation_angle)
        # transform.SetMatrix(list(corner2ct_tf.ravel()))

        # Apply the transformation to the actor's polydata
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputConnection(stl_reader.GetOutputPort())
        transformFilter.SetTransform(transform)
        transformFilter.Update()

        # Set the transformed polydata as the input for the actor
        stl_actor.GetMapper().SetInputConnection(transformFilter.GetOutputPort())
        ren_win.Render()
  
        

    # Define a timer callback
    def timer_callback(obj, event):
        global rotation_angle, rotation_increment
        rotation_angle += rotation_increment
        # print(rotation_angle)
        update_orientation()

    

    
    # Create a timer and assign the callback function
    timer_id = iren.CreateRepeatingTimer(10)  # Timer interval in milliseconds
    iren.AddObserver(vtk.vtkCommand.TimerEvent, timer_callback)
    iren.Start()

    


   


def get_program_parameters():

    return "hello"


def vtk_version_ok(major, minor, build):
    """
    Check the VTK version.

    :param major: Major version.
    :param minor: Minor version.
    :param build: Build version.
    :return: True if the requested VTK version is greater or equal to the actual VTK version.
    """
    needed_version = 10000000000 * int(major) + 100000000 * int(minor) + int(build)
    try:
        vtk_version_number = VTK_VERSION_NUMBER
    except AttributeError:  # as error:
        ver = vtkVersion()
        vtk_version_number = 10000000000 * ver.GetVTKMajorVersion() + 100000000 * ver.GetVTKMinorVersion() \
                             + ver.GetVTKBuildVersion()
    if vtk_version_number >= needed_version:
        return True
    else:
        return False


if __name__ == '__main__':
    main()
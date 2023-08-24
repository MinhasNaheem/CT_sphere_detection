import vtk
dicom_dir = "D:/Navigation//CT_sphere_detection//SAWBONES_AUTOREG_Oarm//455//461//538/AXIAL"

# Load your data using VTK's reader, for example, a DICOM reader
reader = vtk.vtkDICOMImageReader()
reader.SetDirectoryName(dicom_dir)
reader.Update()
threshold_value = 3000
# Apply the Marching Cubes algorithm
marching_cubes = vtk.vtkMarchingCubes()
marching_cubes.SetInputConnection(reader.GetOutputPort())
marching_cubes.SetValue(0, threshold_value)  # Set the desired threshold value
marching_cubes.Update()


# Apply any desired filtering, for example, smoothing or decimation
smooth_filter = vtk.vtkSmoothPolyDataFilter()
smooth_filter.SetInputConnection(marching_cubes.GetOutputPort())
smooth_filter.SetNumberOfIterations(100)  # Increase the number of iterations for better smoothing
smooth_filter.SetRelaxationFactor(0.5)  # Adjust the relaxation factor (0.0-1.0) for more or less smoothing
smooth_filter.SetFeatureEdgeSmoothing(False)  # Disable feature edge smoothing for more isotropic smoothing
smooth_filter.BoundarySmoothingOff()  # Disable boundary smoothing
smooth_filter.Update()


smooth_filter.SetNumberOfIterations(50)  # Set the desired number of smoothing iterations
smooth_filter.Update()

# Create a surface mapper
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(smooth_filter.GetOutputPort())

# Create an actor to visualize the surface
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Create a renderer and add the actor to it
renderer = vtk.vtkRenderer()
renderer.SetBackground(1, 1, 1)  # Set background to white
renderer.AddActor(actor)

# Create a render window and set the renderer
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

# Create an interactor
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)



# Get the output image data from the filter
output_image_data = smooth_filter.GetOutput()

# Create a new VTK image data
vtk_image_data = vtk.vtkImageData()
vtk_image_data.DeepCopy(output_image_data)

# Access the voxel values
dimensions = vtk_image_data.GetDimensions()
voxel_values = vtk_image_data.GetPointData().GetScalars()



# Start the visualization
interactor.Initialize()
render_window.Render()
interactor.Start()

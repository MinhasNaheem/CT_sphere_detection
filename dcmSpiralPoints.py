import SimpleITK as sitk
import numpy as np
import numpy
from numpy import linalg as la
import glob

if True:
	reader = sitk.ImageSeriesReader()

	dcmdir = 'Dataset-2\DICOM\PA0\ST0\SE1'
	series_found = reader.GetGDCMSeriesIDs(dcmdir)
	print(series_found)
	if len(series_found)>0:
		dicom_names = reader.GetGDCMSeriesFileNames( dcmdir, series_found[0] )
	else: #if len(dicom_names)==0:
		dicom_names = glob.glob(dcmdir+'/I*')
		
	reader.SetFileNames(dicom_names)

	image = reader.Execute()

else:
	image = sitk.ReadImage('S20.mhd')

shaping=image.GetSize()
spacing=image.GetSpacing()
space_np = np.array([space for space in spacing ])
shape_np = np.array([shape for shape in shaping ])
print(shaping)
print(spacing)
print(image.GetDirection())
print(image.GetOrigin())
origin = image.GetOrigin()
pos_shift = origin - (shape_np/2 * space_np)
print(pos_shift)

#image.SetOrigin((shape[0]*spacing[0]/2, shape[1]*spacing[1]/2, shape[2]*spacing[2]/2))
# so that the centroids are in vox 
# image.SetOrigin((54,-57,113)) # rough good value
#~ image.SetSpacing((1,1,1)) 
# image.SetDirection((-1,0,0, 0,1,0, 0,0,-1))

thr_image = image > 850 #1080 # balls are 3071, skin markers are 1081 to 1109, screws are ...

stats = sitk.LabelShapeStatisticsImageFilter()

stats.Execute(sitk.ConnectedComponent(thr_image))

balls = []
#~ skinmarks = []
#~ screws = []

for lab in stats.GetLabels():
	
	cen = stats.GetCentroid(lab)
	npix = stats.GetNumberOfPixels(lab)
	idx = image.TransformPhysicalPointToIndex(cen)
	intv = image[idx]
	coord = image.TransformIndexToPhysicalPoint(idx)
	elong = stats.GetElongation(lab)
	bb = stats.GetOrientedBoundingBoxSize(lab)
	
	#~ print(lab,cen,npix,intv,elong, bb)
	
	if True: #cen[1]< 350:
		#print(cen,npix,intv,elong)
		if npix > 100  and npix < 400 and elong >0.5 and elong <1.5 and intv >3000:  
			print(lab,cen,npix,intv,elong)
			balls.append(cen)
		#~ else:
			#~ if intv < 2700:
				#~ screws.append(cen)
			#~ else:
				#~ skinmarks.append(cen)

print(len(balls))

thr_image2 = (image > 3000) 

stats.Execute(sitk.ConnectedComponent(thr_image2))

spheres = []
for lab in stats.GetLabels():
	
	cen = stats.GetCentroid(lab)
	npix = stats.GetNumberOfPixels(lab)
	idx = image.TransformPhysicalPointToIndex(cen)
	intv = image[idx]
	coord = image.TransformIndexToPhysicalPoint(idx)
	elong = stats.GetElongation(lab)
	bb = stats.GetOrientedBoundingBoxSize(lab)
	if npix > 100:
		print(lab,cen,npix,intv,elong)
		spheres.append(cen)

dims = len(spacing)

#origin is ~min(x) and ~max(y) - can be found by elimination
sph = numpy.array(spheres)
el1 = numpy.argmax(sph[:,0])
el2 = numpy.argmin(sph[:,1])

orgidx = numpy.setdiff1d([0,1,2],[el1,el2])[0]
print(orgidx)
org = sph[orgidx,:]
xsph = sph[el1,:]
ysph = sph[el2,:]

xvec = xsph-org
xaxis = xvec/la.norm(xvec)

plvec = org-ysph
plvec = plvec/la.norm(plvec)

zaxis = numpy.cross(xaxis,plvec)

yaxis = numpy.cross(zaxis,xaxis)

#print(xdir,ydir,zdir)
T = numpy.eye(4)
T[:-1,0]=xaxis
T[:-1,1]=yaxis
T[:-1,2]=zaxis
T[:-1,3]=org
#print(M) # world to face

M = la.inv(T)
#trans[:3,-1]=-org

balls_array = numpy.hstack( (numpy.array(balls),numpy.ones((10,1)) )).T

#~ print(balls_array.shape)
centered_balls = numpy.dot(M,balls_array)
print(centered_balls.T)

sph_array = numpy.hstack( (sph, numpy.ones((3,1)) )).T

centered_spheres = numpy.dot(M,sph_array)
print(centered_spheres.T)



if False: #len(balls) == 10:
	orgv = numpy.array(balls).mean(axis=0)
	print("isocenter vox:",orgv)
	org = [orgv[i]*spacing[i] for i in range(dims)]
	print("isocenter mm:", org)

	#image.SetOrigin(org)
	#image.SetSpacing(spacing)

	print("\nskin markers")
	for sm in skinmarks:	
		mmc = [sm[i]*spacing[i] - org[i] for i in range(dims)]
		print(mmc)
	#	print(image.TransformIndexToPhysicalPoint(sm))

	print("\nscrews")
	screws_iso = []
	for sc in screws:
		mmc = [sc[i]*spacing[i] - org[i] for i in range(dims)]
		screws_iso.append(mmc)
		print(mmc)	
	#	print(image.TransformIndexToPhysicalPoint(sc))
		
	print("\nph_centered")
	balls_iso = []
	for b in balls:
		mmc = [b[i]*spacing[i] - org[i] for i in range(dims)]
		balls_iso.append(mmc)
		
	balls_iso = numpy.array(balls_iso)
	print(balls_iso)
	
	height = la.norm(balls_iso[0]-balls_iso[-1])/10.0
	
	delta_ang = 2*numpy.pi/9.0
	
	dia = la.norm(balls_iso[0,:-1]-screws_iso[0][:-1])
	
	print(dia)
	
	ideal = []
	
	#for z in numpy.linspace(-height/2.0,height/2.0):
		
	
	

import cv2, sys, numpy, os, time, json
from PIL import Image

def record_faces(fn_name):
	count = 0
	size = 4
	fn_haar = 'haarcascade_frontalface_alt.xml'
	fn_dir = 'database'
	
	path = os.path.join ( fn_dir, fn_name )

	if not os.path.isdir ( path ):
		os.mkdir ( path )

	(im_width, im_height) =(224,224)

	haar_cascade = cv2.CascadeClassifier ( fn_haar )

	cap = cv2.VideoCapture(0)
	print("Record your face. Please look into the camera.")

	while count < 45:
		try:
			(rval, im) = cap.read ()

			im = cv2.flip ( im, 1, 0 )
			gray = cv2.cvtColor ( im, cv2.COLOR_BGR2GRAY )
			x=(int(gray.shape[ 1 ] / size))
			y=(int(gray.shape[ 0 ] / size))

			mini = cv2.resize ( gray, (x, y) )

			faces = haar_cascade.detectMultiScale ( mini )
			faces = sorted ( faces, key=lambda x: x[ 3 ] )

			cv2.imshow('Input', im)

			if faces:
				face_i = faces[0]
				(x, y, w, h) = [v * size for v in face_i]
				face = gray[y:y + h, x:x + w]
				face_resize = cv2.resize(face, (im_width, im_height))
				pin=sorted([int(n[:n.find('.')])for n in os.listdir(path)
					   if n[0]!='.' ]+[0])[-1] + 1
				cv2.imwrite ( '%s/%s.png' % (path, pin), face_resize )

				cv2.rectangle ( im, (x, y), (x + w, y + h), (0, 255, 0), 3 )
				cv2.putText ( im, fn_name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0) )

				time.sleep ( 0.38 )

				count = count+1
				print("Progress : ", int((100/45)*count), "%")


		except:
			pass

def retrain_model():
	size = 4
	fn_haar = 'haarcascade_frontalface_default.xml'
	fn_dir = 'database'

	legend = {}

	folder_pointer=0
	folder_name='xxxxxx'

	print('Retraining Model')

	(images, lables, names, id) = ([], [], {}, 0)
	for (subdirs, dirs, files) in os.walk(fn_dir):
		for subdir in dirs:
			names[id] = subdir

			subjectpath = os.path.join(fn_dir, subdir)
			for filename in os.listdir(subjectpath):
				path = subjectpath + '/' + filename

				if (folder_name != subdir):
					folder_name=subdir
					folder_pointer+=2

				lable = id
				mat_point=folder_pointer
				imgFile = cv2.imread ( path,0 )

				imread=Image.open(path)

				imResize=cv2.resize(imgFile,(224,224))
				images.append(imResize)

				lables.append(int(mat_point))
				id += 1

			print("subdir : ", subdir)
			print("folder_pointer : ", folder_pointer)
			legend[folder_pointer] = subdir

	(images, lables) = [numpy.array(lis) for lis in [images, lables]]

	model = cv2.face.FisherFaceRecognizer_create()

	model.train(images, lables)
	model.write('test.yml')
	print('Model Saved.')

	with open('legend.json', 'w') as f:
		json.dump(legend, f)

	print("LEGEND : ", legend)

correct_name = False
while correct_name is False:
	fn_name = input("Enter your Name : ")

	if(type(fn_name) is str):
		fn_name = fn_name.replace(" ", "_")
		print(fn_name)
		correct_name = True

record_faces(fn_name)
retrain_model()


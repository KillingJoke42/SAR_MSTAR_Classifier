import subprocess
import os

traindata_path = "/home/killingjoke42/projects/awp/dataset/TARGETS/TEST/15_DEG/T72/SN_S7/"

image_count = 0

for index in range(3333,5685):
	subprocess.call(["./mstar2jpeg", 
						"-i", os.path.join(traindata_path, 
						"HB0{}.017".format(index)), 
						"-o", 
						os.path.join(traindata_path, "jpeg/{}.jpeg".format(image_count)),
						])
	image_count += 1

#mstar2jpeg -i <MSTAR File> -o <JPEG File> [-e] -q] qf [-h] [-v]
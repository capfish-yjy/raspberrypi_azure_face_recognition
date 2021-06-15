import os
import sys
import datetime
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))
print('[', datetime.datetime.now(), '] ', "Connecting to Blob Storage with connection string...")
connect_str = "<connectionstring_azure_storage_blob>"
print("[", datetime.datetime.now(), "] ","Connection Established!")

# print("Required directory structure:\ntest\n --download\n --output")
print("[", datetime.datetime.now(), "] ","Checking if directory structure exists...")
cwd = os.getcwd()
dir1 = 'test'
dir2 = 'input'
dir3 = 'output'
dir4 = 'known'
fullpath1 = os.path.join(cwd, dir1)
fullpath2 = os.path.join(fullpath1, dir2)
fullpath3 = os.path.join(fullpath1, dir3)
fullpath4 = os.path.join(fullpath1, dir4)

if os.path.isdir(fullpath1):
    print("[", datetime.datetime.now(), "] ","test folder exists!")
    if not os.path.isdir(fullpath2):
        print("[", datetime.datetime.now(), "] ","Creating input folder...")
        os.makedirs(fullpath2)
        print("[", datetime.datetime.now(), "] ","Done!")
    else:
        print("[", datetime.datetime.now(), "] ","input folder exists!")
    if not os.path.isdir(fullpath3):
        print("[", datetime.datetime.now(), "] ","Creating output folder...")
        os.makedirs(fullpath3)
        print("[", datetime.datetime.now(), "] ","Done!")
    else:
        print("[", datetime.datetime.now(), "] ","output folder exists!")
    if not os.path.isdir(fullpath4):
        print("[", datetime.datetime.now(), "] ","Creating known folder...")
        os.makedirs(fullpath4)
        print("[", datetime.datetime.now(), "] ","Done!")
    else:
        print("[", datetime.datetime.now(), "] ","known folder exists!")
else:
    print("[", datetime.datetime.now(), "] ","Directory structure does not exist.")
    print("[", datetime.datetime.now(), "] ","Creating Directory Structure...")
    os.makedirs(fullpath1)
    os.makedirs(fullpath2)
    os.makedirs(fullpath3)
    os.makedirs(fullpath4)
    print("[", datetime.datetime.now(), "] ","Directory structure created!")

print("[", datetime.datetime.now(), "] ","Creating Blob Service Client")
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
print("[", datetime.datetime.now(), "] ","Blob Service Client created successfully!")

#access known person container
localPathDownload = fullpath4

print("[", datetime.datetime.now(), "] ","Listing files present in 'fileupload' container...")
container_client = blob_service_client.get_container_client('known')
blobs = container_client.list_blobs()
knownPersonDic = {}
for blob in blobs:
    print("[", datetime.datetime.now(), "] ","Find known person picture in storage!", " ", blob.name)
    knownPersonFileName = str(blob.name)
    knownPersonFilePath = os.path.join(localPathDownload, knownPersonFileName)
    user = knownPersonFileName.split(".")[0]
    if "_" in user :
        user_name = " ".join(user.split("_"))
        print("[", datetime.datetime.now(), "] ","Creating a blob client using the file name obtained previously...")
        blob_client1 = blob_service_client.get_blob_client(container='known', blob=knownPersonFileName)
        print("[", datetime.datetime.now(), "] ","Writing data from blob to known folder...")
        with open(knownPersonFilePath, "wb") as knwonPerson_file:
            knwonPerson_file.write(blob_client1.download_blob().readall())
        knownPersonDic[user_name] = knownPersonFilePath
        print("[", datetime.datetime.now(), "] ","Download known person picture successful!", " ", user_name)
print("[", datetime.datetime.now(), "] ","Download known person pictures successful!", " ", knownPersonDic)
#access input container
# print("[", datetime.datetime.now(), "] ","Listing files present in 'fileupload' container...")
# container_client = blob_service_client.get_container_client('input')

fileNameDownload =sys.argv[1]
print("[", datetime.datetime.now(), "] ","File: ", fileNameDownload, '\n')
    
localPathDownload = fullpath2
downloadFilePath = os.path.join(localPathDownload, fileNameDownload)

print("[", datetime.datetime.now(), "] ","Creating a blob client using the file name obtained previously...")
blob_client1 = blob_service_client.get_blob_client(container='input', blob=fileNameDownload)
print("[", datetime.datetime.now(), "] ","Writing data from blob to download folder...")
with open(downloadFilePath, "wb") as download_file:
    download_file.write(blob_client1.download_blob().readall())
    print("[", datetime.datetime.now(), "] ","Download successful!")#access input container

# print("[", datetime.datetime.now(), "] ","Listing files present in 'fileupload' container...")
# container_client = blob_service_client.get_container_client('input')

#------------------------------------------------------------------------------    
# This is an example of running face recognition on a single image
# and drawing a box around each person that was identified.

# Load a sample picture and learn how to recognize it.
known_face_encodings = []
known_face_names = []
# Load a sample picture and learn how to recognize it.
for knownPerson in knownPersonDic.keys(): 
    known_face_names.append(knownPerson)
    my_image = face_recognition.load_image_file(str(knownPersonDic[knownPerson]))
    my_face_encoding = face_recognition.face_encodings(my_image)[0]
    known_face_encodings.append(my_face_encoding)


# Load an image with an unknown face
unknown_image = face_recognition.load_image_file(downloadFilePath)

# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
# See http://pillow.readthedocs.io/ for more about PIL/Pillow
pil_image = Image.fromarray(unknown_image)
# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)

meIsContained = False
detected_known_name = []
# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"
    # If a match was found in known_face_encodings, just use the first one.
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
        detected_known_name.append(name)
        meIsContained = True
    # Or instead, use the known face with the smallest distance to the new face
    # face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    # best_match_index = np.argmin(face_distances)
    # if matches[best_match_index]:
    #    name = known_face_names[best_match_index]
    #    detected_known_name.append(name)
    #    meIsContained = True

    # Draw a box around the face using the Pillow module
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
    # Draw a label with a name below the face
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

if meIsContained:
    print("It's a picture contains me!")
else:
    print("It's not a picture contains me!")
# Remove the drawing library from memory as per the Pillow docs
del draw
# Create a file in local Documents directory to upload from
localPathOutput = fullpath3
localFilenameOutput = 'processed_' + fileNameDownload
outputFilePath = os.path.join(localPathOutput, localFilenameOutput)

print("[", datetime.datetime.now(), "] ","Copying downloaded file to output folder...")
pil_image.save(outputFilePath)
print("[", datetime.datetime.now(), "] ","Success!")

# Create a blob client using the local file name as the name for the blob
print("[", datetime.datetime.now(), "] ","Creating a blob client using the local file name as the name for the blob...")
blob_client2 = blob_service_client.get_blob_client(container='output', blob=localFilenameOutput)
print("[", datetime.datetime.now(), "] ","Writing data from output folder to the blob storage...")
with open(outputFilePath, "rb") as data:
    blob_client2.upload_blob(data, overwrite=True)
print("[", datetime.datetime.now(), "] ","Upload successful!")
print("[", datetime.datetime.now(), "] ","Please find the output file in the output folder of the Azure blob storage account.")

out_dict = {}
out_dict["outputfile"] = str(localFilenameOutput)
out_dict["result"] = str(meIsContained)
out_dict["names"] = ';'.join(detected_known_name)
with open('outputs.json','w') as file:
    file.write(str(out_dict))


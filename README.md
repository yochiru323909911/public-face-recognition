# public-face-recognition

Look at "face_recognition.py" file.

Registration is the registration function. 
You need to send it a list of images or paths to images that are images of that person's
face and it will return the vector that represents it.
Anlock is the identification function that receives an image or a path to an image of
a person's face and a vector representing someone's face and returns true if it is the 
same person and false if not and also the updated representative vector of that person 
if indeed he was identified as this person.
Restrictions on the photos: the photos should be frontal (the face is exactly in front 
of the camera and you see the ears in the same place - symmetry according to the Y axis) 
and without a smile or anything like that.

import cv2

# Load pre-trained models

   # Load the cascade

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')

gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')

def detect_faces(img_path):

    # Load the image

    img = cv2.imread(img_path)

    # Convert the image to grayscale

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

    # Detect faces in the image

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    return faces

def predict_age_and_gender(img_path):

    #calling detect_faces function 

    faces = detect_faces(img_path)

    # Iterate over detected faces

    for (x, y, w, h) in faces:

        # Extract the face

        face = img[y:y+h, x:x+w]

        # Preprocess the face for gender prediction

        face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603,       

        87.7689143744, 114.895847746), swapRB=False)

        

        # Set the blob as input to the gender network and perform a forward pass

        gender_net.setInput(face_blob)

        gender_prediction = gender_net.forward()

        

        # Preprocess the face for age prediction

        face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603,

        87.7689143744, 114.895847746), swapRB=True)

        

        # Set the blob as input to the age network and perform a forward pass

        age_net.setInput(face_blob)

        age_prediction = age_net.forward()

        

        # Get the predicted age and gender

        age = age_prediction[0].argmax() * 5

        gender = 'Male' if gender_prediction[0][0] > gender_prediction[0][1] else 'Female'

        

        # Draw box around the face and predicted age and gender on the image

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(img, f'Age: {age}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255,  

        0), 2)

        cv2.putText(img, f'Gender: {gender}', (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,

        (0, 255, 0), 2)

    

    # Display the image with bounding boxes and predictions

    cv2.imshow('Face Detection', img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    

# Example usage

img_path = 'path_to_image.jpg'

predict_age_and_gender(img_path)

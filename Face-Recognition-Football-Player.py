import cv2
import os
import math
import numpy as np
from sklearn.model_selection import train_test_split

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

def preprocess_image(image):
    resized_image = cv2.resize(image, (100, 100))
    # blur = cv2.blur(resized_image, (3, 3))
    # gaussian_blur = cv2.GaussianBlur(blur, (3,3), 2)
    equalized_image = cv2.equalizeHist(resized_image)
    return equalized_image

def augment_data(images, labels, num_augmentations=2):
    augmented_images = []
    augmented_labels = []
    
    for i in range(len(images)):
        for _ in range(num_augmentations):
            transformed_image = images[i] 
            augmented_images.append(transformed_image)
            augmented_labels.append(labels[i])
    
    return augmented_images, augmented_labels

def train_and_test_model():
    train_path = "Dataset"
    train_dir = os.listdir(train_path)

    face_list = []
    class_list = []
    print("Training and Testing")

    for idx, tdir in enumerate(train_dir):
        images = os.listdir(os.path.join(train_path, tdir))

        
        for filename in images:
            path = os.path.join(train_path, tdir, filename)
            igray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            

            faces = classifier.detectMultiScale(igray, scaleFactor=1.5, minNeighbors=5)

            if len(faces) == 1:
                x, y, w, h = faces[0]
                face_image = igray[y:y + h, x:x + w]
                preprocess_face = preprocess_image(face_image)
                face_list.append(preprocess_face)
                class_list.append(idx)

    face_list, class_list = augment_data(face_list, class_list)

    X_train, X_test, y_train, y_test = train_test_split(face_list, class_list, test_size=0.25, random_state=42)

    face_recognizer.train(X_train, np.array(y_train))

    correct_predictions = 0

    for idx, face_image in enumerate(X_test):
        res, conf = face_recognizer.predict(face_image)
        conf = math.floor(conf * 100) / 100

        # print(f"{res} Predicted: {train_dir[res]}, Actual: {train_dir[y_test[idx]]}, Confidence: {conf}%")

        if res == y_test[idx]:
            correct_predictions += 1

    accuracy = correct_predictions / len(X_test)
    print("Training and Testing Finished")
    print(f"Average Accuracy: {accuracy * 100}%")

    save_path = "model_trained.yml"
    face_recognizer.save(save_path)


def predict():

    train_path = "Dataset"
    train_dir = os.listdir(train_path)

    save_path = "model_trained.yml"
    if os.path.exists(save_path):
        face_recognizer.read(save_path)
    else:
        print("Error: Model not found. Please train the model first.")
        return

    input_path = input("Input the absolute path for image to predict: ")

    if(input_path.endswith('.jpg' or '.png')):
        image = cv2.imread(input_path)
        igray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(igray, scaleFactor=1.5, minNeighbors=5)

        for face_rect in faces:
            x, y, w, h = face_rect
            face_image = igray[y:y + h, x:x + w]

            res, conf = face_recognizer.predict(face_image)
            conf = math.floor(conf * 100) / 100

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            image_text = f"{train_dir[res]}: {str(conf)}%"
            cv2.putText(image, image_text, (x - 20, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Prediction Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Please input the correct directories")

# Main menu
while True:
    print("1. Train and Test Model")
    print("2. Predict")
    print("3. Exit")
    
    choice = input(">> ")

    if choice == "1":
        train_and_test_model()
    elif choice == "2":
        predict()
    elif choice == "3":
        break
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

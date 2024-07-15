import os  #the libraries needed for this script.
import cv2

#Create and access the folder where the collected images will be stored.
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 30 #for the number of letters
dataset_size = 70   #for the number of pictures taken per letter.

#open the camera to collect images
cap = cv2.VideoCapture(0)


for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    #keep in mind counting starts from 0.
    print('Collecting data for class {}'.format(j))

    done = False
    #added a stopper between each class so you're not overwhelmed and have time to prepare the move for the next class.
    #press Q to start collecting the next class.
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()

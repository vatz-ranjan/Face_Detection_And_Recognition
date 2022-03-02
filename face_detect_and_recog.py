'''
******************************************************************

--- WELCOME TO THIS FACE DETECTION AND RECOGNITION SYSTEM ---

******************************************************************
'''


'''
******************************************************************

Made by- Ranjan Kumar 087-CSE-B-19, Shikhar 059-CSE-B-19

******************************************************************
'''

import numpy as np
import cv2
import os
import sqlite3
import face_recognition
import pickle
# import imutils


def get_folder():
    '''
    Return: folder_name: Return the folder name
    Description: This function will create a new folder (if not present) and return the folder path
    The folder created will contain all the pickles files having face encodings and database file.
    '''
    folder_name = 'Faces'
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def get_file_loc(file_no):
    '''
    Return: file_loc - File location for the particular file location
    Description: This function will get the folder path and then will return the file location for the particular file no.
    '''
    folder_name = get_folder()
    file_name = str(file_no) + '.pkl'
    file_loc = os.path.join(folder_name, file_name)
    return file_loc


def create_pickle_file(file_no, face_encoding):
    '''
    Parameter: file_no - File no to create a new pickle file for new face encoding, face_encoding - Face Encoding for the new face
    Description: This function will simply create a new file for a new face and will dump the face encoding into it.
    '''
    file_loc = get_file_loc(file_no)
    pickle.dump(face_encoding, open(file_loc, 'wb'))
    # print("File Created for a new Face")


def connect_database():
    '''
    Return: connection - connection with the database, cursor - Cursor for execution and fetching
    Description: This function will establish a connection with the table present in the database
    '''
    folder_name = get_folder()
    sqlite_file_name = 'Recognized_Faces.sqlite'
    file_loc = os.path.join(folder_name, sqlite_file_name)
    connection_established = False
    try:
        connection = sqlite3.connect(file_loc)
        connection_established = True

    except:
        print("Unable to connect to the database !!!")
        connection_established = False

    return connection, connection_established


def close_connection(connection, cursor=None):
    '''
    Description: Close the connection with the database (if present)
    '''
    try: cursor.close()
    except: pass

    try: connection.close()
    except: pass


def delete_table():
    '''
    Description: Delete the table (if present) from the database
    '''
    connection, connection_established = connect_database()
    if connection_established:
        sql_query = '''
        DROP TABLE IF EXISTS Face
                '''
        cursor = connection.cursor()
        cursor.executescript(sql_query)
        connection.commit()
        close_connection(connection, cursor)
        # print('\nTable Deleted')


def create_table():
    '''
    Description: Create a table (if not present) in the database
    '''
    connection, connection_established = connect_database()
    if connection_established:
        sql_query = '''CREATE TABLE IF NOT EXISTS Face
        (
        Name TEXT
        )
        '''
        cursor = connection.cursor()
        cursor.executescript(sql_query)
        connection.commit()
        close_connection(connection, cursor)
        print('\nTable Created')


def add_face(name, face_encoding):
    '''
    Parameter: name - name of the person not in the database, face_encoding - face encoding of the person's face
    Description: Connect to the database, add the name of the person to the database and call the function to create a corresponding pickle file of face encoding
    '''
    connection, connection_established = connect_database()
    if connection_established:
        cursor = connection.cursor()
        sql_query = '''INSERT INTO Face(Name) VALUES(?)'''
        cursor.execute(sql_query, (name,))
        user_no = cursor.lastrowid
        create_pickle_file(user_no, face_encoding)
        connection.commit()
        close_connection(connection, cursor)
        print("\nFace Added")


def face_identification(face_encoding):
    '''
    Parameter: face_encoding - Face Encoding (NumPy.array)
    Return: Name of the person if present in database else ask the user for name
    Description: Connect to the database of names and read all the pickle files having known face encodings
    Will check for the each face encoding and if matched then return the name
    '''
    name = 'UNKNOWN'
    connection, connection_established = connect_database()

    if connection_established:
        matched = False
        cursor = connection.cursor()
        sql_query = '''SELECT * FROM Face'''
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        for i, row in enumerate(rows):
            known_face_encodings = []
            file_no = i+1
            file_loc = get_file_loc(file_no)
            read_file = open(file_loc, 'rb')
            known_face_encoding = pickle.load(read_file)
            # print("FE",face_encoding)
            # print("KFE", known_face_encoding)
            # print(face_encoding.shape)
            # print(known_face_encoding.shape)
            known_face_encodings = [known_face_encoding]
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)

            if True in matches:
                matched = True
                name = row[0]
                return name

        close_connection(connection, cursor)

        if not matched:
            name = input("\nEnter name of the person : ")
            add_face(name, face_encoding)

    return name

# def detect_face(o_frame):
def detect_face(frame):
    '''
    Parameter: frame - frame of video or image
    Output: Display the frame/image with the rectangle boxes for each face
    Description: Check for the face locations
    For each face locations gives a face encoding
    Display the frame/image with a rectangle box around each face
    '''
    print("\nDetecting Faces ---")
    # frame = imutils.resize(o_frame, width=1000, height=1000)
    new_frame = frame[:, :, ::-1]
    face_locations = []
    face_encodings = []

    face_locations = face_recognition.face_locations(new_frame)
    face_encodings = face_recognition.face_encodings(new_frame, face_locations)

    print("Face location", face_locations, len(face_locations))
    # print("Face encodings", face_encodings, len(face_encodings))
    # print(type(face_encodings))

    for ((top, right, bottom, left), face_encoding) in zip(face_locations, face_encodings):
        name = face_identification(face_encoding)
        cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)

        cv2.rectangle(frame, (left, bottom-35), (right, bottom), (0,0,255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left+6, bottom-6), font, 1.0, (255,255,255), 1)

    # resized_frame = cv2.resize(frame)
    cv2.imshow("Face", frame)


def image_webcam():
    '''
    Description: This function is to open the webcam for photo
    And call for face detection and recognition function
    '''
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret_val, img = video.read()

    if ret_val:
        detect_face(img)
        cv2.waitKey(0)
        video.release()
        cv2.destroyAllWindows()

    else: print("--- Image not found ---")
    return


def image_path(img_path):
    '''
    Parameter: img_path - Path of image
    Description: This function is to read an image with the path provided by the user
    And call for face detection and recognition function
    '''
    img = cv2.imread(img_path, 1)
    detect_face(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def video_webcam():
    '''
    Description: This function is to open the webcam for video
    And call for face detection and recognition function
    '''
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while video.isOpened():
        check, frame = video.read()
        detect_face(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    return


def video_path(vid_path):
    '''
    Parameter: vid_path - Path of video
    Description: This function is to read a video with the path provided by the user
    And call for face detection and recognition function
    '''
    video = cv2.VideoCapture(vid_path)

    if not video.isOpened():
        print("\n-- Unable to open the video --\n")
        return

    while video.isOpened():
        check, frame = video.read()
        if check:
            detect_face(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        else: break

    video.release()
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    '''
    MAIN FUNCTION
    Ask the user for processing image and video
    And for both the case it will ask for path or for webcam
    Call the respective function!!!
    '''
    intro = "--- WELCOME TO THIS FACE DETECTION SYSTEM ---"
    print("*"*len(intro))
    print("\n{}\n".format(intro))
    print("*" * len(intro))

    # delete_table()
    create_table()
    again = True

    while again:
        img_or_vid = int(input("\n-- Image or Video (1. Image / 2. Video) : "))
        # img_or_vid = 1
        path = input("-- Enter path (Y for WebCam) : ")
        # path = 'C:\\Users\RK\Desktop\images.jfif'

        if path in ['Y', 'y']:
            if img_or_vid == 1: image_webcam()
            elif img_or_vid == 2: video_webcam()

        else:
            path = r'{}'.format(path)
            if img_or_vid == 1: image_path(path)
            elif img_or_vid == 2: video_path(path)

        again_stat = "WANT TO DO MORE (Y/N) : "

        print("*"*len(intro))
        again = input("{}".format(again_stat))
        print("*" * len(intro))

        if again in ['y', 'Y']: again = True
        else: again = False

    at_last = "--- THANKS, Hope you enjoyed this ---"

    print("*" * len(intro))
    print("\n{}\n".format(at_last))
    print("*" * len(intro))



# C:\Users\RK\Desktop\images.jfif
# C:\Users\RK\Desktop\fdar.mp4
# E:\Data Structures\HashMap.mp4



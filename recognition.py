import face_recognition
import cv2
import os
import numpy

# -------------------------------
# Loads the known faces from the known_faces folder and returns them as a list of encodings and names
# -------------------------------
KNOWN_FACES_DIR = "known_faces"
def load_known_faces():
    known_faces = []
    known_names = []
    print("Loading known faces...")
    for name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        if not os.path.isdir(person_dir):
            continue
        for filename in os.listdir(person_dir):
            filepath = os.path.join(person_dir, filename)
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_faces.append(encodings[0])
                known_names.append(name)
                print(f"Loaded {filename} for {name}")
            else:
                print(f"Warning: No faces found in {filepath}")

    print(f"Finished loading {len(known_faces)} known faces.\n")
    return known_faces, known_names

# -------------------------------
# Recognizes faces in an image and returns the name of the person 
# -------------------------------
def recognize_image(known_faces, known_names):
    print("🔍 Checking image...")
    image = face_recognition.load_image_file("input.jpg")
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb_image)
    encodings = face_recognition.face_encodings(rgb_image, locations)

    for face_encoding, face_location in zip(encodings, locations):
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        name = "Unknown"

        if len(face_distances) > 0:
            best_match_index = numpy.argmin(face_distances)
            if face_distances[best_match_index] < 0.6:
                name = known_names[best_match_index]
    #draws a rectangle around the face and writes the name of the person
        top, right, bottom, left = face_location
        cv2.rectangle(rgb_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(rgb_image, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print(f"Found: {name}")

    cv2.imshow("Image Recognition", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------------------
# Runs the camera and recognizes faces
# -------------------------------
def run_camera(known_faces, known_names):
    video = cv2.VideoCapture(0)
    print("Press 'q' to quit the camera.")

    while True:
        ret, frame = video.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, locations)

        for face_encoding, face_location in zip(encodings, locations):
            face_distances = face_recognition.face_distance(known_faces, face_encoding)

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                name = known_names[best_match_index]
                distance = face_distances[best_match_index]
                if distance > 0.6:
                    name = "Unknown"
                print(f"Detected: {name} (distance: {distance:.2f})")
            else:
                name = "Unknown"

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()

# -------------------------------
# User chooses what mode to use and the program continues to ask until the user chooses to exit
# -------------------------------
def main():
    known_faces, known_names = load_known_faces() # Loads the known faces from the known_faces folder and returns them as a list of encodings and names

    while True:
        print("\nWhat do you want to do?")
        print("1. Run camera")
        print("2. Run image")
        print("3. Exit")
        choice = input("Enter choice: ")

        if choice == "1":
            run_camera(known_faces, known_names)
        elif choice == "2":
            recognize_image(known_faces, known_names)
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice, try again.")

if __name__ == "__main__":
    main()
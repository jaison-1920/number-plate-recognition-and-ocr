import cv2
import easyocr
import csv
from datetime import datetime



def save_img(img_roi,frame):
    global count

    cv2.imwrite("plates/scanned_img_"+str(count+1)+".jpg",img_roi)
    cv2.rectangle(frame,(0,200),(640,300),(0,255,0),cv2.FILLED)
    cv2.putText(frame,"Number Saved",(150,265),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(255,0,255),2)
    cv2.imshow("saved",frame)
    ocr_and_save_to_csv(img_roi) #calling the fn for ocr and saving to csv
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    count+=1

def ocr_and_save_to_csv(img_roi):
    state_codes = {
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chattisgarh",
    "DL": "Delhi",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JK": "Jammu and Kashmir",
    "JH": "Jharkhand",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LD": "Lakshadweep Islands",
    "MP": "Madhya Pradesh",
    "MH": "Maharashtra",
    "MN": "Manipur",
    "ML": "Meghalaya",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "OR": "Odisha",
    "PY": "Pondicherry",
    "PB": "Punjab",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TS": "Telangana",
    "TR": "Tripura",
    "UP": "Uttar Pradesh",
    "UK": "Uttarakhand",
    "UA": "Uttarakhand",
    "WB": "West Bengal",
    "AN": "Andaman and Nicobar Islands",
    "CH": "Chandigarh",
    "DN": "Dadra & Nagar Haveli",
    "DD": "Daman & Diu",
    "LA": "Ladakh"
}
    #ocr of number plate
    reader = easyocr.Reader(['en'])
    output = reader.readtext(img_roi)
    plate_number = output[0][1] #retrieving the vehicle's number
    plate_number_upper = plate_number.upper()
    code = plate_number_upper[:2] #extracting the state code
    print(plate_number_upper)
    print(code)
    state = state_codes[code] #identifying the state
   
    

    #saving the file to csv 
    captured_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") #getting the  live time 

    #csv file
    csv_file = "vehicle's_details.csv"

    with open(csv_file, 'a', newline='') as csvfile:
        fieldnames = ["Vehicle Number", "Registration State", "Captured Time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
        if csvfile.tell() == 0:  # Check if the file is empty
            writer.writeheader()

        writer.writerow({
        "Vehicle Number": plate_number_upper,
        "Registration State": state,
        "Captured Time": captured_time
        })

    


def main():
    haarcascade = "haarcascade_russian_plate_number.xml"
    global count
    count=0
    cam = cv2.VideoCapture(0)
    cam.set(3,640) #height of capturing img
    cam.set(4,480) #width of capturing img

    while True:
        ret,frame = cam.read()
        if ret:
            cascade = cv2.CascadeClassifier(haarcascade)
            img_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)# haarcascade need grayscale img

            #for getting the coordinates
            plates = cascade.detectMultiScale(img_gray,1.1,4)

            for (x,y,w,h) in plates:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(frame,"Number Plate",(x+10,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2)

                #cropping the roi and displaying
                img_roi = frame[y:y+h,x:x+w]
                cv2.imshow('number',img_roi)

            cv2.imshow('Detected number plate',frame)
            
            #to save the region of interest
            if cv2.waitKey(1) & 0xFF == ord('s'):
                save_img(img_roi,frame) #calling the save_img()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

main()






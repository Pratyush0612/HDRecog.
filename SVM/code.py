!pip install opencv.python
!pip install pandas
!pip install scikit-learn

#Generating dataset
import cv2
import csv
import glob

header = ["label"]
for i in range(0,784):
  header.append("pixel"+str(i))
with open('dataset.csv','a') as f:
  writer = csv.writer(f)
  writer.writerow(header)

for label in range(10):
  dirList = glob.glob("/content/sample_data/captured_images/"+str(label)+"/*.png")

  for img_path in dirList:
    im = cv2.imread(img_path)
    im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray,(15,15), 0)
    roi = cv2.resize(im_gray,(28,28),interpolation = cv2.INTER_AREA)

    data =[]
    data.append(label)
    rows,cols = roi.shape

    #add pixel one by one into data array
    for i in range(rows):
      for j in range(cols):
        k=roi[i,j]
        if k>100:
          k=1
        else:
          k=0
        data.append(k)
    with open('dataset.csv','a') as f:
      writer = csv.writer(f)
      writer.writerow(data)


data = pd.read_csv("/content/dataset.csv")
X= data.drop(["label"],axis = 1)
Y= data["label"]
%matplotlib inline
import matplotlib.pyplot as plt
import cv2
idx = 1666
image = X.loc[idx].values.reshape(28,28)
print(Y[idx])
plt.imshow(image)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y, test_size = 0.2)

import joblib
from sklearn.svm import SVC
classifier = SVC(kernel="linear",random_state=10)
classifier.fit(X_train,y_train)
joblib.dump(classifier,"/content/sample_data/Model/SVC_digit_recogniser")
from sklearn import metrics
prediction = classifier.predict(X_test)
print("Accuracy = ",metrics.accuracy_score(prediction,y_test))

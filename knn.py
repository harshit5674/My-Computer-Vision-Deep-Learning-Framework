from datasets import SimpleDatasetLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imutils import paths
import argparse
from preprocessing import SimplePreprocessor

ar=argparse.ArgumentParser()
ar.add_argument("-d","--dataset",required=True,help="path to input dataset")
ar.add_argument("-k","--neighbors",type=int,default=1,help="# of nearest neighbors")

args=vars(ar.parse_args())
print("[INFO] loading images...")
imagePaths=list(paths.list_images(args["dataset"]))
sp=SimplePreprocessor(32,32)
sdl=SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))


print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))


#trainig the model

le = LabelEncoder()
labels = le.fit_transform(labels)
(X_train, X_test, y_train, y_test) = train_test_split(data, labels,test_size=0.25, random_state=42)

print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"])
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test),target_names=le.classes_))



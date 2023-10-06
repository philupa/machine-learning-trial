from sklearn import datasets, svm
digits = datasets.load_digits()
import matplotlib.pyplot as plt

clf = svm.SVC(gamma=0.001, C=100.)
fit_method = clf.fit(digits.data[:-2], digits.target[:-2])

prediction = clf.predict(digits.data[-2:])
print(prediction)

plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[-2], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()


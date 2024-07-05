#!/usr/bin/env python

##############
#### Your name: Alexis Wilmot
##############

import numpy as np
import re, math
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color
import ransac_score
#from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt

class ImageClassifier:
    
    def __init__(self):
        self.classifier = None

    def imread_convert(self, f):
        return io.imread(f).astype(np.uint8)

    def load_data_from_folder(self, dir):
        # read all images into an image collection
        ic = io.ImageCollection(dir+"*.bmp", load_func=self.imread_convert)
        
        #create one large array of image data
        data = io.concatenate_images(ic)
        
        #extract labels from image names
        labels = np.array(ic.files)
        for i, f in enumerate(labels):
            m = re.search("_", f)
            labels[i] = f[len(dir):m.start()]
        
        return(data,labels)

    def extract_image_features(self, data):
        # Please do not modify the header above

        # extract feature vector from image data

        ########################
        ######## YOUR CODE HERE
        # skimage.feature.hog(image, orientations=9, pixels_per_cell=(8, 8), 
        #                       cells_per_block=(3, 3), block_norm='L2-Hys', visualize=False,
        #                       transform_sqrt=False, feature_vector=True, *, channel_axis=None)
        # extract histogram of oriented gradients (HOG) for a given image - flattens into a feature vector
        # print(data.shape)
        # return
        # feature_data = [np.empty(data.shape[0], dtype=object)]
        feature_data = []

        # loop through each photo
        for i in range(data.shape[0]):
            #image = color.rgb2gray(data[i])
            image = data[i]
            oneColor = image[:, :, 0]
            #image = data[i]
            #print(image)
            # add a gaussian filter
            gauss = filters.gaussian(oneColor, sigma=2.0)
            #print(gauss.shape)
            # adjust exposure to correct it
            #processed = exposure.equalize_hist(gauss)
            #print('done processing')
            #print(processed.shape)
            features = feature.hog(gauss, orientations=8, pixels_per_cell=(16, 16), 
                                              cells_per_block=(3,3), block_norm='L2-Hys', visualize=False,
                                              transform_sqrt=False, feature_vector=True)
            #print(features)
            #feature_data[i] = features
            #print(features.shape)
            feature_data.append(features)
        #print(feature_data.shape)
        feature_data = np.array(feature_data)
        #print(feature_data.shape)
        # Please do not modify the return type below
        #print('done extracting')
        return(feature_data)

    def train_classifier(self, train_data, train_labels):
        # Please do not modify the header above
        
        # train model and save the trained model to self.classifier
        ########################
        ######## YOUR CODE HERE
        
        # create classifier
        classy = svm.SVC()

        # train the classifier
        #print('training')
        classy.fit(train_data, train_labels)
        #print('done training')
        self.classifier = classy
        ########################
        pass

    def predict_labels(self, data):
        # Please do not modify the header

        # predict labels of test data using trained model in self.classifier
        # the code below expects output to be stored in predicted_labels
        
        ########################
        ######## YOUR CODE HERE
        #print('predicting')
        predicted_labels = self.classifier.predict(data)
        #print('done')
        ########################

        # Please do not modify the return type below
        return predicted_labels
    


    def line_fitting(self, data):
        # Please do not modify the header

        # fit a line the to arena wall using RANSAC
        # return two lists containing slopes and y intercepts of the line

        ########################
        ######## YOUR CODE HERE
        ########################
        #newData = preprocess(data)
        slope = np.empty(shape=(data.shape[0], 1))
        intercept = np.empty(shape=(data.shape[0], 1))
        for i in range(data.shape[0]):
            # outputs a binary image
            #image = color.rgb2gray(data[i])
            # increase exposure
            expose = exposure.adjust_gamma(data[i], gamma=.5)
            # gaussian blur
            gauss = filters.gaussian(expose, sigma=3.0)
            # contrast
            #image2 = exposure.equalize_hist(gauss)
            image2 = gauss
            #print(image2.shape)
            plt.imshow(image2)
            plt.show()
            # binary of edges on plot
            can = feature.canny(image2[:, :, 0], sigma=3)
            # coordinates of the edges, WHERE the edges are
            #print(np.where(can))
            where = np.column_stack(np.where(can))
            #print(max(where[0]))
            # model

            best_slope = None
            best_int = None
            best_inliers = 0
            for e in range(40): # 5 iterations, might change it tho
                # get indices of the randomly chosen points
                inds = np.random.choice(len(where), size=2, replace=False)
                # these are the actual points
                p1, p2 = where[inds]
                line = np.polyfit([p1[1], p2[1]], [p1[0], p2[0]], 1)
                # find the slope of the points
                #m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                # y intercept
                #b = p1[1] - (m * p1[0])

                m = line[0]
                b = line[1]
                 # get distances between line and points
                #distances = np.abs(where[:, 1] - (m * where[:, 0] + b))
                #distances = np.abs((where[:, 1] - (m * where[:, 0] + b)) / np.sqrt(1 + m**2))
                #distances = np.sqrt((where[:, 0] - p1[0])**2 + (where[:, 1] - (m * where[:, 0] + b))**2)
                distances = np.sqrt((where[:, 0] - where[:, 0])**2 + (where[:, 1] - where[:, 1])**2)
                # get standard deviation
                #stdev = np.std(distances)
                # if within 1 standard deviation, keep
                #rint(stdev)
                inliers = where[distances < 8]
                inlier_count = len(inliers)
               

                if inlier_count > best_inliers:
                    best_inliers = inlier_count
                    best_slope = m
                    best_int = b
            #print(distances)        
            slope[i] = best_slope
            intercept[i] = best_int
            #print(best_inliers)
            #plt.plot(np.array([0, image.shape[1]]))
            plt.imshow(can, cmap='gray')
            plt.plot(np.array([0, image2.shape[1]]), best_slope * np.array([0, image2.shape[1]]) + best_int)
            print(slope)
            plt.show()


            # ransac = RANSACRegressor()
            # # fit(X, y), where x is training data, y is target values
            # ransac.fit(where[:, 1].reshape(-1, 1), where[:, 0])
            # slope.append(ransac.estimator_.coef_[0])
            # intercept.append(ransac.estimator_.intercept_)

        # Please do not modify the return type below
        return slope, intercept
# def preprocess(self, data):
        
#     return newData
def main():

    img_clf = ImageClassifier()

    # load images
    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')
    (test_raw, test_labels) = img_clf.load_data_from_folder('./test/')
    (wall_raw, _) = img_clf.load_data_from_folder('./wall/')
    
    # convert images into features
    # train_data = img_clf.extract_image_features(train_raw)
    # test_data = img_clf.extract_image_features(test_raw)
    
    # train model and test on training data
    # img_clf.train_classifier(train_data, train_labels)
    # predicted_labels = img_clf.predict_labels(train_data)
    # print("\nTraining results")
    # print("=============================")
    # print("Confusion Matrix:\n1",metrics.confusion_matrix(train_labels, predicted_labels))
    # print("Accuracy: ", metrics.accuracy_score(train_labels, predicted_labels))
    # print("F1 score: ", metrics.f1_score(train_labels, predicted_labels, average='micro'))
    
    # # test model
    # predicted_labels = img_clf.predict_labels(test_data)
    # print("\nTest results")
    # print("=============================")
    # print("Confusion Matrix:\n",metrics.confusion_matrix(test_labels, predicted_labels))
    # print("Accuracy: ", metrics.accuracy_score(test_labels, predicted_labels))
    # print("F1 score: ", metrics.f1_score(test_labels, predicted_labels, average='micro'))

    # ransac
    print("\nRANSAC results")
    print("=============================")
    s, i = img_clf.line_fitting(wall_raw)
    print(f"Line Fitting Score: {ransac_score.score(s,i)}/10")

if __name__ == "__main__":
    main()
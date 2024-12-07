#include <iostream> 
#include <cstdlib> 
#include <ctime> 
#include <iomanip> 
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <algorithm>
#include <chrono>

using namespace std; 
using namespace chrono; // allows for time and durations 

/* Part 1 function start 
// Stub evaluation function that generates a random accuracy 
double randomEvaluation() {
    return static_cast<double>(rand()) / RAND_MAX * 100.0;
}

// Forward Selection Algorithm 
void forwardSelection(int totalFeatures) {
    cout << "\nForward Selection Algorithm\n";
    vector<int> currentSet;
    double bestOverallAccuracy = 0.0;
    vector<int> bestOverallSet;

    for (int i = 1; i <= totalFeatures; ++i) {
        cout << "\nLevel " << i << " of the search\n";
        int bestFeature = -1;
        double bestAccuracyThisLevel = 0.0;

        // Evaluate each feature not already in the current set
        for (int feature = 1; feature <= totalFeatures; ++feature) {
            if (find(currentSet.begin(), currentSet.end(), feature) == currentSet.end()) {
                vector<int> tempSet = currentSet;
                tempSet.push_back(feature);
                double accuracy = randomEvaluation();

                cout << "Using feature(s) { ";
                for (int f : tempSet) cout << f << " ";
                cout << "}, accuracy is " << fixed << setprecision(1) << accuracy << "%\n";

                // Track the best feature in this level
                if (accuracy > bestAccuracyThisLevel) {
                    bestAccuracyThisLevel = accuracy;
                    bestFeature = feature;
                }
            }
        }

        // If we found a valid best feature, add it to the current set
        if (bestFeature != -1) {
            currentSet.push_back(bestFeature);
            cout << "Feature set { ";
            for (int f : currentSet) cout << f << " ";
            cout << "} was best, accuracy is " << fixed << setprecision(1) << bestAccuracyThisLevel << "%\n";

            // Update the overall best subset and accuracy only if it's better
            if (bestAccuracyThisLevel > bestOverallAccuracy) {
                bestOverallAccuracy = bestAccuracyThisLevel;
                bestOverallSet = currentSet;
            } else {
                cout << "(Warning, Accuracy has decreased!)\n";
            }
        }
    }

    // Report the best subset and accuracy found over all levels
    cout << "\nFinished search!! The best feature subset is { ";
    for (int f : bestOverallSet) cout << f << " ";
    cout << "}, which has an accuracy of " << fixed << setprecision(1) << bestOverallAccuracy << "%\n";
}


// Backward Elimination Algorithm 
void backwardElimination(int totalFeatures) {
    cout << "\nBackward Elimination Algorithm\n";
    vector<int> currentSet(totalFeatures);
    for (int i = 0; i < totalFeatures; ++i) {
        currentSet[i] = i + 1;
    }

    double bestOverallAccuracy = 0.0;
    vector<int> bestOverallSet = currentSet;

    for (int level = totalFeatures; level > 0; --level) {
        cout << "\nLevel " << level << " of the search\n";
        int worstFeature = -1;
        double bestAccuracyThisLevel = 0.0;

        for (int feature : currentSet) {
            vector<int> tempSet;
            for (int f : currentSet) {
                if (f != feature) tempSet.push_back(f);
            }
            double accuracy = randomEvaluation();
            cout << "Using feature(s) { ";
            for (int f : tempSet) cout << f << " ";
            cout << "}, accuracy is " << fixed << setprecision(1) << accuracy << "%\n";

            if (accuracy > bestAccuracyThisLevel) {
                bestAccuracyThisLevel = accuracy;
                worstFeature = feature;
            }
        }

        if (worstFeature != -1) {
            currentSet.erase(remove(currentSet.begin(), currentSet.end(), worstFeature), currentSet.end());
            cout << "Feature set { ";
            for (int f : currentSet) cout << f << " ";
            cout << "} was best, accuracy is " << fixed << setprecision(1) << bestAccuracyThisLevel << "%\n";

            if (bestAccuracyThisLevel > bestOverallAccuracy) {
                bestOverallAccuracy = bestAccuracyThisLevel;
                bestOverallSet = currentSet;
            } else {
                cout << "(Warning, Accuracy has decreased!)\n";
            }
        }
    }

    cout << "\nFinished search!! The best feature subset is { ";
    for (int f : bestOverallSet) cout << f << " ";
    cout << "}, which has an accuracy of " << fixded << setprecision(1) << bestOverallAccuracy << "%\n";
}
Part 1 function end 
*/

// Part 2 Function Start 
class NearestNeighborClassifier {
private:
    vector<vector<double>> trainingData; // Stores the training data
    vector<int> trainingLabels;          // Stores class labels for the training data


public:
    // Method to train the classifier with data and labels
    void Train(const vector<vector<double>> &data, const vector<int> &labels) {
        trainingData = data;
        trainingLabels = labels;
    }

    // Method to predict the label of a given test instance 
    int Test(const vector<double> &testInstance) {
        double minDistance = numeric_limits<double>::max();
        int predictedLabel = -1;

        // Compare the test instance to all training data 
        for (size_t i = 0; i < trainingData.size(); ++i) {
            double distance = euclideanDistance(testInstance, trainingData[i]);
            if (distance < minDistance) { // Update minimum distance and predicted label 
                minDistance = distance;
                predictedLabel = trainingLabels[i];
            }
        }

        // Return the predicted class label
        return predictedLabel; 
    }


private:
    // Helper function to calculate Euclidean distance between two vectors
    double euclideanDistance(const vector<double> &a, const vector<double> &b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            sum += pow(a[i] - b[i], 2); // Sum of squared differences 
        }
        return sqrt(sum); // Return the predicted class label 
    }
};

// Function to normalize dataset values to a range of 0 to 1 
void normalizeDataset(vector<vector<double>> &data) { // Number of features in each instance 
    size_t numFeatures = data[0].size();

    // Loop through each feature to normalize 
    for (size_t j = 0; j < numFeatures; ++j) {
        double minVal = numeric_limits<double>::max();
        double maxVal = numeric_limits<double>::lowest();


        // Find the minimum and maximum values for the current feature
        for (size_t i = 0; i < data.size(); ++i) {
            minVal = min(minVal, data[i][j]);
            maxVal = max(maxVal, data[i][j]);
        }


        // Normalize the feature values for all instances 
        for (size_t i = 0; i < data.size(); ++i) {
            if (maxVal != minVal) { // Avoid division by zero 
                data[i][j] = (data[i][j] - minVal) / (maxVal - minVal);
            } else {
                data[i][j] = 0.0; // If max equals min, set normalized value to 0
            }
        }
    }
}

// Function to read dataset from a file and extract the selected feautures
void readDataset(const string &filename, vector<vector<double>> &data, vector<int> &labels, const vector<int> &featureSubset) {
    ifstream file(filename);

    // Check if the file was successfully opened 
    if (!file.is_open()) {
        cout << "Error: Could not open file " << filename << endl;
        exit(1);    // Exit program if file is missing 
    }


    string line;
    // Read each line of the dataset 
    while (getline(file, line)) {
        stringstream ss(line); // Use stringstream to parse the line 
        vector<double> instance;
        double value;
        int label;


        ss >> label; // Extract the class label
        labels.push_back(label);

        vector<double> features; // Extract feature values 
        while (ss >> value) {
            features.push_back(value);
        }


        vector<double> selectedFeatures;
        // Select only the features specified in the subset
        for (int index : featureSubset) {
            selectedFeatures.push_back(features[index]);
        }


        data.push_back(selectedFeatures); // Add the selected features to the dataset
    }


    file.close();
}

// Class to handle validation tasks 
class Validator {
public:
    // Leave-One-Out Validation
    double LeaveOneOutValidation(NearestNeighborClassifier &classifier, const vector<vector<double>> &data, const vector<int> &labels) {
        int correctPredictions = 0; // Count of correct predictions
        int totalInstances = data.size(); // Total number of instances

        // Loop through each instance, Leaving one out for testing 
        for (size_t i = 0; i < totalInstances; ++i) {
            auto start = high_resolution_clock::now(); // Record start time


            vector<vector<double>> trainingData;
            vector<int> trainingLabels;

            // Use all instances except the current one for training 
            for (size_t j = 0; j < totalInstances; ++j) {
                if (i != j) {
                    trainingData.push_back(data[j]);
                    trainingLabels.push_back(labels[j]);
                }
            }


            // Train the classifier on the remaining data
            classifier.Train(trainingData, trainingLabels);


            // Test the classifier on the excluded instance
            int predictedLabel = classifier.Test(data[i]);


            auto end = high_resolution_clock::now(); // Record end time
            double duration = duration_cast<microseconds>(end - start).count(); // Calculate duration


            // Trace the steps
            cout << "Instance " << i << ": Actual Label = " << labels[i]
                 << ", Predicted Label = " << predictedLabel
                 << ", Time Taken = " << duration << "microseconds" << endl;

            // Check if prediction is correct
            if (predictedLabel == labels[i]) {
                correctPredictions++;
            }
        }

        // return accuracy
        return static_cast<double>(correctPredictions) / totalInstances;
    }
};
// Part 2 function end 

int main() {


    /* Part 1 main start 
    srand(static_cast<unsigned>(time(0)));

    cout << "Welcome to the feature Selection Algorithm.\n";
    cout << "Please enter the total number of features: ";
    int totalFeatures; 
    cin >> totalFeatures; 

    cout << "Type the number of the algorithm you want to run.\n"; 
    cout << "1. Forward Selection\n";
    cout << "2. Backward Elimination\n";
    int choice; 
    cin >> choice; 

    switch (choice) {
        case 1: 
            forwardSelection(totalFeatures);
            break;
        case 2: 
            backwardElimination(totalFeatures);
            break;
        default:
            cout << "Invalid choice, Please enter 1 or 2.\n";
            break;
    }

    return 0;
    Part 1 main end     
     */

    // Part 2 main start
    vector<vector<double>> dataset;
    vector<int> labels;


    // Specify the feature subset 
    vector<int> featureSubset = {3, 5, 7};
    //vector<int> featureSubset = {1, 15, 27};


    // Read the dataset from the file
    auto startRead = high_resolution_clock::now();
    readDataset("small-test-dataset.txt", dataset, labels, featureSubset);
    //readDataset("large-test-dataset.txt", dataset, labels, featureSubset);
    auto endRead = high_resolution_clock::now();
    cout << "Dataset reading completed in "
         << duration_cast<microseconds>(endRead - startRead).count() << "microseconds" << endl;


    // Normalize the dataset
    auto startNormalize = high_resolution_clock::now();
    normalizeDataset(dataset);
    auto endNormalize = high_resolution_clock::now();
    cout << "Dataset normalization completed in "
         << duration_cast<microseconds>(endNormalize - startNormalize).count() << "microseconds" << endl;


    // Initialize the classifier
    NearestNeighborClassifier nnClassifier;


    // Initialize the validator
    Validator validator;


    // Perform Leave-one-out validation
    auto startValidation = high_resolution_clock::now();
    double accuracy = validator.LeaveOneOutValidation(nnClassifier, dataset, labels);
    auto endValidation = high_resolution_clock::now();


    cout << "Leave-One-Out Validation completed in "
         << duration_cast<microseconds>(endValidation - startValidation).count() << "microseconds" << endl;


    // Print final accuracy
    cout << "Accuracy using leave-one-out validation: " << accuracy * 100 << "%" << endl;


    return 0;

    // Part 2 main end

}
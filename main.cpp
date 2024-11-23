#include <iostream> 
#include <vector> 
#include <cstdlib> 
#include <ctime> 
#include <iomanip> 
#include <algorithm>

using namespace std; 

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


int main() {
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
}
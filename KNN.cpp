#include <Rcpp.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>

using namespace std;

// A simple struct to represent a data point
struct DataPoint {
    double x, y; // Features
    int classLabel; // Class label
    double distance; // Distance from test point

    // Constructor
    DataPoint(double x, double y, int classLabel) : x(x), y(y), classLabel(classLabel), distance(0.0) {}
};

// Function to calculate Euclidean distance between two points
double euclideanDistance(const DataPoint& a, const DataPoint& b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

// Function to get the k-nearest neighbors
vector<DataPoint> getKNearestNeighbors(vector<DataPoint>& trainingSet, DataPoint& testPoint, int k) {
    // Calculate distance from test point to all training points
    for (auto& data : trainingSet) {
        data.distance = euclideanDistance(data, testPoint);
    }

    // Sort the training set by distance from test point
    sort(trainingSet.begin(), trainingSet.end(), [](const DataPoint& a, const DataPoint& b) {
        return a.distance < b.distance;
    });

    // Select the first k elements after sorting
    vector<DataPoint> neighbors(trainingSet.begin(), trainingSet.begin() + k);
    return neighbors;
}

// Function to predict the class of the test point
int predictClass(vector<DataPoint>& neighbors) {
    map<int, int> classVotes;

    // Count votes for each class
    for (const auto& neighbor : neighbors) {
        classVotes[neighbor.classLabel]++;
    }

    // Find the class with the most votes
    int maxVotes = 0;
    int predictedClass = -1;
    for (const auto& vote : classVotes) {
        if (vote.second > maxVotes) {
            predictedClass = vote.first;
            maxVotes = vote.second;
        }
    }

    return predictedClass;
}

// Wrapper function for Rcpp
// [[Rcpp::export]]
int knnRcpp(Rcpp::NumericMatrix trainingSet, double x, double y, int k) {
    // Convert R matrix to C++ vector
    vector<DataPoint> trainingData;
    for (int i = 0; i < trainingSet.nrow(); ++i) {
        trainingData.push_back(DataPoint(trainingSet(i, 0), trainingSet(i, 1), trainingSet(i, 2)));
    }

    // Test point
    DataPoint testPoint(x, y, 0); // Class label is not used for prediction

    // Find the k-nearest neighbors
    vector<DataPoint> neighbors = getKNearestNeighbors(trainingData, testPoint, k);

    // Predict the class of the test point
    int predictedClass = predictClass(neighbors);

    return predictedClass;
}

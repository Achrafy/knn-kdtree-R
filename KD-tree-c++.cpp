#include <Rcpp.h>
#include <cmath>
#include <limits>
#include <queue>
#include <vector>
#include <algorithm>

using namespace std;

class Point {
public:
  double x, y;
  int cls; // Class label
  
  Point(double x = 0, double y = 0, int cls = 0) : x(x), y(y), cls(cls) {}
};

class Node {
public:
  Point point;
  Node* left;
  Node* right;
  
  Node(Point p) : point(p), left(NULL), right(NULL) {}
};

class KDTree {
private:
  Node* root;
  
  Node* buildTree(vector<Point>& points, int depth) {
    if (points.empty()) return NULL;
    
    int axis = depth % 2;
    int medianIdx = points.size() / 2;
    
    nth_element(points.begin(), points.begin() + medianIdx, points.end(), [axis](const Point& a, const Point& b) {
      return axis == 0 ? a.x < b.x : a.y < b.y;
    });
    
    Node* node = new Node(points[medianIdx]);
    vector<Point> leftPoints(points.begin(), points.begin() + medianIdx);
    vector<Point> rightPoints(points.begin() + medianIdx + 1, points.end());
    
    node->left = buildTree(leftPoints, depth + 1);
    node->right = buildTree(rightPoints, depth + 1);
    
    return node;
  }
  
  void kNearestNeighborsUtil(Node* node, Point& target, priority_queue<pair<double, int>>& pq, int k, int depth) {
    if (node == NULL) return;
    
    double dist = pow(node->point.x - target.x, 2) + pow(node->point.y - target.y, 2);
    
    if (pq.size() < k || dist < pq.top().first) {
      pq.push(make_pair(dist, node->point.cls));
      if (pq.size() > k) pq.pop();
    }
    
    int axis = depth % 2;
    
    Node* next = (axis == 0 ? target.x : target.y) < (axis == 0 ? node->point.x : node->point.y) ? node->left : node->right;
    Node* other = next == node->left ? node->right : node->left;
    
    kNearestNeighborsUtil(next, target, pq, k, depth + 1);
    
    double dx = (axis == 0 ? target.x - node->point.x : target.y - node->point.y);
    if (pq.size() < k || dx * dx < pq.top().first) {
      kNearestNeighborsUtil(other, target, pq, k, depth + 1);
    }
  }
  
public:
  KDTree(const vector<Point>& points) {
    vector<Point> pts = points;
    root = buildTree(pts, 0);
  }
  
  int predictClass(Point target, int k) {
    priority_queue<pair<double, int>> pq;
    kNearestNeighborsUtil(root, target, pq, k, 0);
    
    unordered_map<int, int> classCounts;
    while (!pq.empty()) {
      classCounts[pq.top().second]++;
      pq.pop();
    }
    
    int maxCount = 0;
    int predictedClass = -1;
    for (auto& count : classCounts) {
      if (count.second > maxCount) {
        maxCount = count.second;
        predictedClass = count.first;
      }
    }
    
    return predictedClass;
  }
};

// [[Rcpp::export]]
int predictClassForPoint(Rcpp::NumericMatrix pointsData, double x, double y, int k) {
  vector<Point> points;
  for (int i = 0; i < pointsData.nrow(); i++) {
    points.push_back(Point(pointsData(i, 0), pointsData(i, 1), pointsData(i, 2)));
  }
  
  KDTree tree(points);
  return tree.predictClass(Point(x, y), k);
}


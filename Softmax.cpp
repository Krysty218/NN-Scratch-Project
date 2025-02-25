#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

using namespace std;

typedef vector<double> vector2;
typedef vector<vector<double>> vector3;

class Utilities {
public:
    // Function to generate random numbers with a normal distribution
    static double generateRandomNumber() {
        static random_device rd;
        static mt19937 generator(rd());
        static normal_distribution<double> distribution(0.0, 1.0); // Mean 0, standard deviation 1
        return distribution(generator);
    }

    // Function to print a matrix
    static void printMatrix(const vector3& matrix) {
        for (const auto& row : matrix) {
            for (double val : row) {
                cout << val << " ";
            }
            cout << endl;
        }
    }

    //Create Spiral Data function (based on corrections and code in image)
    static vector3 createSpiralData(int points, int classes, vector2& labels) {
      vector3 X;
      labels.resize(points * classes);

      for (int class_number = 0; class_number < classes; ++class_number) {
          for (int i = 0; i < points; ++i) {
              double angle = (double)i / points * 2 * M_PI; // Angle
              double radius = (double)class_number / 3.0;    // Radius
              double x = radius * cos(angle);
              double y = radius * sin(angle);
              X.push_back({x, y});
              labels[class_number * points + i] = class_number;
          }
      }
      return X;
    }

    // Function to perform matrix multiplication
    static vector3 matrixMultiply(const vector3& matrix1, const vector3& matrix2) {
        size_t rows1 = matrix1.size();
        size_t cols1 = (matrix1.empty() ? 0 : matrix1[0].size());
        size_t rows2 = matrix2.size();
        size_t cols2 = (matrix2.empty() ? 0 : matrix2[0].size());

        if (cols1 != rows2) {
            cerr << "Error: Incompatible matrix dimensions for multiplication." << endl;
            return {}; // Return an empty matrix
        }

        vector3 result(rows1, vector2(cols2, 0.0));
        for (size_t i = 0; i < rows1; ++i) {
            for (size_t j = 0; j < cols2; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < cols1; ++k) {
                    sum += matrix1[i][k] * matrix2[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    // Function to add biases (adding a vector to each row of a matrix)
    static vector3 addBiases(const vector3& matrix, const vector2& biases) {
        size_t rows = matrix.size();
        size_t cols = (matrix.empty() ? 0 : matrix[0].size()); // Ensure matrix is not empty
        size_t bias_size = biases.size();

        vector3 result(rows, vector2(cols));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i][j] = matrix[i][j] + biases[j];
            }
        }
        return result;
    }
};

class Activation_ReLU {
public:
    vector3 forward(const vector3& inputs) {
        size_t rows = inputs.size();
        size_t cols = (inputs.empty() ? 0 : inputs[0].size());
        output_.resize(rows, vector2(cols));

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                output_[i][j] = max(0.0, inputs[i][j]); // ReLU activation
            }
        }
        return output_;
    }

    vector3 getOutput() const {
        return output_;
    }

private:
    vector3 output_;
};

class Activation_Softmax {
public:
    vector3 forward(const vector3& inputs) {
        size_t rows = inputs.size();
        size_t cols = (inputs.empty() ? 0 : inputs[0].size());
        output_.resize(rows, vector2(cols));


        for (size_t i = 0; i < rows; ++i) {
            // Find the maximum value in the row
            double max_val = inputs[i][0];
            for (size_t j = 1; j < cols; ++j) {
                if (inputs[i][j] > max_val) {
                    max_val = inputs[i][j];
                }
            }

            // Calculate exponential values and normalize
            double sum_exp = 0.0;
            vector2 exp_values(cols);
            for (size_t j = 0; j < cols; ++j) {
                exp_values[j] = exp(inputs[i][j] - max_val); //subtract max to prevent overflow
                sum_exp += exp_values[j];
            }

            // Calculate probabilities
            for (size_t j = 0; j < cols; ++j) {
                output_[i][j] = exp_values[j] / sum_exp;
            }
        }
        return output_;
    }

    vector3 getOutput() const {
        return output_;
    }

private:
    vector3 output_;
};

class Layer_Dense {
public:
    Layer_Dense(int n_inputs, int n_neurons) : n_inputs_(n_inputs), n_neurons_(n_neurons) {
        // Initialize weights with small random numbers
        weights_.resize(n_inputs_, vector2(n_neurons_));
        for (int i = 0; i < n_inputs_; ++i) {
            for (int j = 0; j < n_neurons_; ++j) {
                weights_[i][j] = 0.10 * Utilities::generateRandomNumber();
            }
        }

        // Initialize biases to zero
        biases_.resize(n_neurons_, 0.0);
    }

    vector3 forward(const vector3& inputs) {
        // Multiply inputs by weights and add biases
        vector3 output = Utilities::matrixMultiply(inputs, weights_);

        // Create a bias matrix and add biases
          vector3 biasMatrix(inputs.size(), vector2(biases_.size()));

            for(size_t i = 0; i < inputs.size(); i++){
                for(size_t j = 0; j < biases_.size(); j++){
                     biasMatrix[i][j] = biases_[j];
                }
            }


        vector3 biased_output = Utilities::addBiases(output, biases_);
        output_ = biased_output; // Store the output

        return output_;
    }

    vector3 getOutput() const {
        return output_;
    }

private:
    int n_inputs_;
    int n_neurons_;
    vector3 weights_;
    vector2 biases_;
    vector3 output_;
};


int main() {
  // Create spiral data
    int num_points = 100;
    int num_classes = 3;
    vector2 y;
    vector3 X = Utilities::createSpiralData(num_points, num_classes, y);

    Layer_Dense dense1(2, 3);
    Activation_ReLU activation1;
    Layer_Dense dense2(3, 3);
    Activation_Softmax activation2;

    vector3 dense1_outputs = dense1.forward(X);
    vector3 activation1_outputs = activation1.forward(dense1_outputs);
    vector3 dense2_outputs = dense2.forward(activation1_outputs);
    vector3 activation2_outputs = activation2.forward(dense2_outputs);


    // Print the output
    cout << "Output after Softmax (first 5 samples):" << endl;
    for (size_t i = 0; i < min((size_t)5, activation2_outputs.size()); ++i) {
        Utilities::printMatrix({activation2_outputs[i]}); // Print each row as a matrix
    }


    return 0;
}

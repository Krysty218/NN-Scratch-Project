#include <iostream>
#include <vector>
#include <random>
#include <cmath>

class Utilities {
public:
    // Function to generate random numbers with a normal distribution
    static double generateRandomNumber() {
        static std::random_device rd;
        static std::mt19937 generator(rd());
        static std::normal_distribution<double> distribution(0.0, 1.0); // Mean 0, standard deviation 1
        return distribution(generator);
    }

    // Function to print a matrix
    static void printMatrix(const std::vector<std::vector<double>>& matrix) {
        for (const auto& row : matrix) {
            for (double val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }

    // Function to create spiral data, based on the Python from the images you provided
    static std::vector<std::vector<double>> createSpiralData(int points, int classes, std::vector<double>& labels) {
        std::vector<std::vector<double>> X;
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
    static std::vector<std::vector<double>> matrixMultiply(const std::vector<std::vector<double>>& matrix1,
                                                           const std::vector<std::vector<double>>& matrix2) {
        size_t rows1 = matrix1.size();
        size_t cols1 = matrix1[0].size();
        size_t rows2 = matrix2.size();
        size_t cols2 = matrix2[0].size();

        if (cols1 != rows2) {
            std::cerr << "Error: Incompatible matrix dimensions for multiplication." << std::endl;
            return {}; // Return an empty matrix
        }

        std::vector<std::vector<double>> result(rows1, std::vector<double>(cols2, 0.0));
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
    static std::vector<std::vector<double>> addBiases(const std::vector<std::vector<double>>& matrix,
                                                       const std::vector<double>& biases) {
        size_t rows = matrix.size();
        size_t cols = matrix[0].size();
        size_t bias_size = biases.size();

        std::vector<std::vector<double>> result(rows, std::vector<double>(cols));
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
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& inputs) {
        output_.resize(inputs.size(), std::vector<double>(inputs[0].size()));
        for (size_t i = 0; i < inputs.size(); ++i) {
            for (size_t j = 0; j < inputs[0].size(); ++j) {
                output_[i][j] = relu(inputs[i][j]);
            }
        }
        return output_;
    }

    std::vector<std::vector<double>> getOutput() const {
        return output_;
    }

private:
    std::vector<std::vector<double>> output_;

    // ReLU activation function
    double relu(double x) {
        return std::max(0.0, x);
    }
};

class Layer_Dense {
public:
    Layer_Dense(int n_inputs, int n_neurons) : n_inputs_(n_inputs), n_neurons_(n_neurons) {
        // Initialize weights with small random numbers
        weights_.resize(n_inputs_, std::vector<double>(n_neurons_));
        for (int i = 0; i < n_inputs_; ++i) {
            for (int j = 0; j < n_neurons_; ++j) {
                weights_[i][j] = 0.10 * Utilities::generateRandomNumber();
            }
        }

        // Initialize biases to zero
        biases_.resize(n_neurons_, 0.0);
    }

    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& inputs) {
        // Multiply inputs by weights and add biases
        std::vector<std::vector<double>> output = Utilities::matrixMultiply(inputs, weights_);


        std::vector<std::vector<double>> biased_output = Utilities::addBiases(output, biases_);
        output_ = biased_output; // Store the output

        return output_;
    }

    std::vector<std::vector<double>> getOutput() const {
        return output_;
    }

private:
    int n_inputs_;
    int n_neurons_;
    std::vector<std::vector<double>> weights_;
    std::vector<double> biases_;
    std::vector<std::vector<double>> output_;
};



int main() {
    // Create spiral data
    int num_points = 100;
    int num_classes = 3;
    std::vector<double> y;
    std::vector<std::vector<double>> X = Utilities::createSpiralData(num_points, num_classes, y);

    Layer_Dense layer1(2, 5);
    Activation_ReLU activation1;

    std::vector<std::vector<double>> layer1_outputs = layer1.forward(X);
    std::vector<std::vector<double>> activation1_outputs = activation1.forward(layer1_outputs);

    // Print the output after ReLU activation
    std::cout << "Output after ReLU:" << std::endl;
    Utilities::printMatrix(activation1_outputs);


    return 0;
}

#include <iostream>
#include <vector>
#include <random>

// Function to generate random numbers with a normal distribution
double generateRandomNumber() {
    static std::random_device rd;
    static std::mt19937 generator(rd());
    static std::normal_distribution<double> distribution(0.0, 1.0); // Mean 0, standard deviation 1
    return distribution(generator);
}

// Function to transpose a matrix
std::vector<std::vector<double>> transposeMatrix(const std::vector<std::vector<double>>& matrix) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    std::vector<std::vector<double>> transposed_matrix(cols, std::vector<double>(rows));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            transposed_matrix[j][i] = matrix[i][j];
        }
    }
    return transposed_matrix;
}

// Function to perform matrix multiplication
std::vector<std::vector<double>> matrixMultiply(const std::vector<std::vector<double>>& matrix1,
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
std::vector<std::vector<double>> addBiases(const std::vector<std::vector<double>>& matrix,
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


// Function to print a matrix
void printMatrix(const std::vector<std::vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

class Layer_Dense {
public:
    Layer_Dense(int n_inputs, int n_neurons) : n_inputs_(n_inputs), n_neurons_(n_neurons) {
        // Initialize weights with small random numbers
        weights_.resize(n_inputs_, std::vector<double>(n_neurons_));
        for (int i = 0; i < n_inputs_; ++i) {
            for (int j = 0; j < n_neurons_; ++j) {
                weights_[i][j] = 0.10 * generateRandomNumber();
            }
        }

        // Initialize biases to zero
        biases_.resize(n_neurons_, 0.0);
    }

    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& inputs) {
        // Multiply inputs by weights and add biases
        std::vector<std::vector<double>> output = matrixMultiply(inputs, weights_);

         //Create Bias Matrix

        std::vector<std::vector<double>> biasMatrix(inputs.size(), std::vector<double>(biases_.size()));

        for(size_t i = 0; i < inputs.size(); i++){
            for(size_t j = 0; j < biases_.size(); j++){
                 biasMatrix[i][j] = biases_[j];
            }
        }

        std::vector<std::vector<double>> biased_output = addBiases(output, biases_);
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
    // Define inputs
    std::vector<std::vector<double>> X = {
        {1, 2, 3, 2.5},
        {2.0, 5.0, -1.0, 2.0},
        {-1.5, 2.7, 3.3, -0.8}
    };


    Layer_Dense layer1(4, 5);
    Layer_Dense layer2(5, 2);

    std::vector<std::vector<double>> layer1_outputs = layer1.forward(X);
    std::vector<std::vector<double>> layer2_outputs = layer2.forward(layer1_outputs);

    // Print the output of the second layer
    std::cout << "Layer 2 Output:" << std::endl;
    printMatrix(layer2_outputs);

    return 0;
}

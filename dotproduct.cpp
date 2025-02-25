#include <iostream>
#include <vector>

class Layer {
public:
    Layer(const std::vector<std::vector<double>>& weights, const std::vector<double>& biases)
        : weights_(weights), biases_(biases) {}

    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& inputs) {
        // Transpose weights
        std::vector<std::vector<double>> transposed_weights = transposeMatrix(weights_);

        // Perform matrix multiplication
        std::vector<std::vector<double>> outputs = matrixMultiply(inputs, transposed_weights);

        // Add biases
        addBiases(outputs, biases_);

        return outputs;
    }

private:
    std::vector<std::vector<double>> weights_;
    std::vector<double> biases_;

    // Helper function to transpose a matrix
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

    // Helper function to perform matrix multiplication
    std::vector<std::vector<double>> matrixMultiply(const std::vector<std::vector<double>>& matrix1,
                                                     const std::vector<std::vector<double>>& matrix2) {
        size_t rows1 = matrix1.size();
        size_t cols1 = matrix1[0].size();
        size_t rows2 = matrix2.size();
        size_t cols2 = matrix2[0].size();

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

    // Helper function to add biases to a matrix
    void addBiases(std::vector<std::vector<double>>& matrix, const std::vector<double>& biases) {
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[0].size(); ++j) {
                matrix[i][j] += biases[j];
            }
        }
    }
};

// Function to print a matrix
void printMatrix(const std::vector<std::vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Define inputs, weights, and biases
    std::vector<std::vector<double>> inputs = {
        {1, 2, 3, 2.5},
        {2.0, 5.0, -1.0, 2.0},
        {-1.5, 2.7, 3.3, -0.8}
    };

    std::vector<std::vector<double>> weights = {
        {0.2, 0.8, -0.5, 1.0},
        {0.5, -0.91, 0.26, -0.5},
        {-0.26, -0.27, 0.17, 0.87}
    };

    std::vector<double> biases = {2, 3, 0.5};

     std::vector<std::vector<double>> weights2 = {
        {0.1, -0.14, 0.5},
        {-0.5, 0.12, -0.33},
        {-0.44, 0.73, -0.13}
    };

    std::vector<double> biases2 = {-1, 2, -0.5};

    // Create layers
    Layer layer1(weights, biases);
    Layer layer2(weights2, biases2);


    // Perform forward pass through the layers
    std::vector<std::vector<double>> layer1_outputs = layer1.forward(inputs);
    std::vector<std::vector<double>> layer2_outputs = layer2.forward(layer1_outputs);


    // Print the final output
    std::cout << "Layer 2 Outputs:" << std::endl;
    printMatrix(layer2_outputs);

    return 0;
}

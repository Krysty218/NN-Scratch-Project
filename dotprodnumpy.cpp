#include <iostream>
#include <Eigen/Dense>

int main() {
    // Define inputs as an Eigen vector
    Eigen::VectorXd inputs(4);
    inputs << 1.0, 2.0, 3.0, 2.5;

    // Define weights as an Eigen matrix
    Eigen::MatrixXd weights(3, 4);
    weights << 0.2, 0.8, -0.5, 1.0,
               0.5, -0.91, 0.26, -0.5,
               -0.26, -0.27, 0.17, 0.87;

    // Define biases as an Eigen vector
    Eigen::VectorXd biases(3);
    biases << 2.0, 3.0, 0.5;

    // Compute the output using matrix-vector multiplication
    Eigen::VectorXd output = weights * inputs + biases;

    // Print the output
    std::cout << "Output:\n" << output << std::endl;

    return 0;
}

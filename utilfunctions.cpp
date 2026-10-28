#include "utilfunctions.hpp"


using namespace std;

namespace transformer {
    ublas::matrix<double> softmax(const ublas::matrix<double>& x) {
        ublas::matrix<double> result(x.size1(), x.size2());

        // process each row
        //#pragma omp parallel for
        for (size_t i = 0; i < x.size1(); ++i) {
            
            double max_val = -std::numeric_limits<double>::infinity();
            
            //find max value
            for (size_t j = 0; j < x.size2(); ++j)
                max_val = max(max_val, x(i, j));

            double sum = 0.0;
            for (size_t j = 0; j < x.size2(); ++j) {
                result(i, j) = exp(x(i, j) - max_val); // prevents large exponentials that could cause overflow
                sum += result(i, j);
            }

            // normalize
            for (size_t j = 0; j < x.size2(); ++j)
                result(i, j) /= sum;
        }

        return result;
    }
    
    std::mt19937& global_generator() {
        static std::mt19937 gen(std::random_device{}());
        return gen;
    }

    ublas::matrix<double> one_matrix(int rows, int cols) {
        ublas::matrix<double> mat(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                mat(i, j) = 1.0;
        return mat;
    }

    ublas::matrix<double> random_weight(int rows, int cols) {
        ublas::matrix<double> mat(rows, cols);
        std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / rows)); // He initialization
        auto& gen = global_generator();

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                mat(i, j) = dist(gen);
            }
        }

        return mat;
    }

    ublas::matrix<double> random_weight_parallel(int rows, int cols) {
        ublas::matrix<double> mat(rows, cols);
        auto& gen = global_generator();

        #pragma omp parallel
        {
            std::mt19937 local_gen();

            #pragma omp critical
            {
                local_gen.seed(gen());
            }
            std::normal_distribution<double> local_dist(0.0, std::sqrt(2.0 / rows)); // He initialization
            
            #pragma omp for collapse(2)
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    mat(i, j) = local_dist(local_gen);
                }
            }
        }


        return mat;
    }

    ublas::matrix<double> layer_norm(
        const ublas::matrix<double>& input,
        const ublas::matrix<double>& gamma,
        const ublas::matrix<double>& beta,
        double epsilon
    ) {
        int batch = input.size1();
        int dim = input.size2();
        ublas::matrix<double> output(batch, dim);
        
        //#pragma omp parallel for
        for (int i = 0; i < batch; ++i) {
            double mean = 0.0;
            double var = 0.0;

            // mean
            for (int j = 0; j < dim; ++j) 
                mean += input(i, j);
            mean /= dim;

            // variance
            for (int j = 0; j < dim; ++j)
                var += (input(i, j) - mean) * (input(i, j) - mean);
            var /= dim;

            // normalize
            for (int j = 0; j < dim; ++j) {
                double norm = (input(i, j) - mean) / std::sqrt(var + epsilon);
                output(i, j) = norm * gamma(j,0) + beta(j,0);
            }
        }

        return output;
    }

}

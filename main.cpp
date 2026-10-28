#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <omp.h>
#include "DataProcessor.hpp"
#include "transformer.hpp"

using namespace transformer;

int argmax(const ublas::matrix<double>& row) {
    int idx = 0;
    double max_val = row(0, 0);
    for (std::size_t i = 1; i < row.size2(); ++i) {
        if (row(0, i) > max_val) {
            max_val = row(0, i);
            idx = i;
        }
    }
    return idx;
}

double cross_entropy_loss(const ublas::matrix<double>& predictions, const std::vector<int>& target_ids) {
    double loss = 0.0;
    for (std::size_t i = 0; i < predictions.size1(); ++i) {
        int label = target_ids[i];
        double prob = predictions(i, label);
        
        loss -= std::log(prob + 1e-9); // avoid log(0)
    }
    return loss / predictions.size1();
}

int main() {
    omp_set_num_threads(4);
    const std::string filename = "data.txt";
    const int embed_dim = 32;
    const int num_heads = 4;
    const int num_layers = 2;
    const int max_epochs = 20;

    // read and tokenize
    std::string text = read_file(filename);
    auto tokens = tokenize(text);
    tokens.resize(80);  // test with small sequence, CHANGE LATER

    auto vocab = build_vocab(tokens);

    const int num_epochs = (max_epochs >= vocab.size()) ? vocab.size()-1 : max_epochs;

    auto rev_vocab = build_reverse_vocab(vocab);

    // convert to token ids
    auto token_ids = tokens_to_ids(tokens, vocab);
    if (token_ids.size() < 3) {
        std::cerr << "Need at least 3 tokens for training.\n";
        return 1;
    }

    if (token_ids.size() < 2) {
        std::cerr << "Need at least 2 tokens for decoder/target.\n";
        return 1;
    }
    
    // prepare inputs
    std::vector<int> encoder_ids = token_ids;
    std::vector<int> decoder_ids(token_ids.begin(), token_ids.end() - 1);
    std::vector<int> target_ids(token_ids.begin() + 1, token_ids.end());

    std::cout << "vocab size: " << vocab.size() << "\n";

    // initialize model
    Transformer transformer(num_layers, embed_dim, num_heads, vocab.size());

    auto embedding_matrix = transformer.create_embedding_matrix(vocab.size(), embed_dim);
    
    const double learning_rate = 0.01;

    // vectors to store the results
    std::vector<double> losses;
    std::vector<std::string> predictions;
    std::vector<std::string> actual_words;


    double epoch_start_time = omp_get_wtime(); 

    const int window_size = 8;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        double total_loss = 0.0;
        int num_windows = 0;
    
        for (std::size_t i = 0; i + window_size < token_ids.size(); ++i) {
            std::vector<int> encoder_ids(token_ids.begin() + i, token_ids.begin() + i + window_size);
            std::vector<int> decoder_ids(encoder_ids.begin(), encoder_ids.end() - 1);
            std::vector<int> target_ids(encoder_ids.begin() + 1, encoder_ids.end());
    
            auto output = transformer.forward(encoder_ids, decoder_ids);
            double loss = cross_entropy_loss(output, target_ids);
            total_loss += loss;
            ++num_windows;
    
            transformer.backward(output, target_ids, decoder_ids);
            transformer.sgd_step(learning_rate);
    
            // predict one token
            ublas::matrix<double> last_row(1, output.size2());
            for (std::size_t j = 0; j < output.size2(); ++j)
                last_row(0, j) = output(output.size1() - 1, j);
            int pred_token = argmax(last_row);

            predictions.push_back(rev_vocab[pred_token]);
            actual_words.push_back(rev_vocab[target_ids[0]]);
            
        }
    
        losses.push_back(total_loss / num_windows);
    }

    double epoch_end_time = omp_get_wtime();  // Capture the end time
    double epoch_duration = epoch_end_time - epoch_start_time;  // Calculate the duration for the epoch

    // output the collected results after all epochs
    std::cout << "\ntraining completed!\n";




    std::cout << " time: " << epoch_duration << " seconds\n";

    std::ofstream output_file("output.txt"); 

    if (!output_file.is_open()) {
        std::cerr << "Failed to open file.\n";
        return 1;
    }

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        output_file << std::left <<  "[epoch " << epoch + 1 << std::setw(20) << "]"  
        << " loss: " << std::setw(20) << losses[epoch] 
        << "token: " << std::setw(20) << predictions[epoch] 
        <<  "actual word: " << std::setw(20) << actual_words[epoch]  
        << "\n";
    }
    output_file.close();
    std::cout << "predictions and actual written to 'output.txt'.\n";

    return 0;
}

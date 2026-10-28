#pragma once
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

std::string read_file(const std::string& filename) {
    std::ifstream file(filename);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::vector<std::string> tokenize(const std::string& text) {
    std::istringstream stream(text);
    std::vector<std::string> tokens;
    std::string token;
    while (stream >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

std::unordered_map<std::string, int> build_vocab(const std::vector<std::string>& tokens) {
    std::unordered_map<std::string, int> vocab;
    int index = 0;
    for (const auto& tok : tokens) {
        if (vocab.find(tok) == vocab.end()) {
            vocab[tok] = index++;
        }
    }
    return vocab;
}


std::vector<std::string> build_reverse_vocab(const std::unordered_map<std::string, int>& vocab) {
    std::vector<std::string> reverse_vocab(vocab.size());
    for (const auto& [word, id] : vocab) {
        reverse_vocab[id] = word;
    }
    return reverse_vocab;
}

std::vector<int> tokens_to_ids(
    const std::vector<std::string>& tokens,
    const std::unordered_map<std::string, int>& vocab
) {
    std::vector<int> ids;
    for (const auto& tok : tokens) {
        auto it = vocab.find(tok);
        if (it != vocab.end()) {
            ids.push_back(it->second);
        } else {
            // handle unknowns if needed
            ids.push_back(0);
        }
    }
    return ids;
}
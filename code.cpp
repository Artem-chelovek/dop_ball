#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>
using namespace std;



struct Matrix { // Упрощённая структура матрицы
    vector<vector<double>> data;
    int rows, cols;
    
    Matrix(int r, int c) : rows(r), cols(c), data(r, vector<double>(c)) {}

    void operator+=(const Matrix& other) {
        if (rows != other.rows || cols != other.cols) throw runtime_error("Размеры матриц не совпадают");
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                this->data[i][j] += other.data[i][j];
            }
        }
    }
};

class UNet {
private:
    struct ConvLayer {
        Matrix weights;
        double bias;
        
        ConvLayer(int in_channels, int out_channels, int kernel_size=3) :
            weights(out_channels, in_channels * kernel_size * kernel_size),
            bias(0) {
            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<> dis(-1.0, 1.0);
            for (auto& row : weights.data) {
                for (auto& elem : row) {
                    elem = dis(gen); // случайная инициализация веса
                }
            }
        }
            
        Matrix convolve(const Matrix &input) {
            return input; // Упростили реализацию свёрточного слоя
        }
    };

    struct PoolLayer {
        Matrix pool(const Matrix &input) {
            return input; // MaxPooling упрощённо
        }
    };

    struct UpConvLayer {
        Matrix upconv(const Matrix &input) {
            return input; // Transposed convolution упрощённо
        }
    };

    vector<ConvLayer> encoders;
    vector<PoolLayer> pools;
    vector<UpConvLayer> decoders;

public:
    UNet(int depth) {
        for (int i = 0; i < depth; ++i) {
            encoders.emplace_back(i+1, i+2);   // Увеличение количества каналов
            pools.emplace_back();
            decoders.emplace_back();           // Декодеры симметричны кодерам
        }
    }

    pair<Matrix, vector<Matrix>> encode(const Matrix& input) {
        Matrix current = input;
        vector<Matrix> features;
        for (size_t i = 0; i < encoders.size(); ++i) {
            current = encoders[i].convolve(current);
            features.push_back(current);       // Сохраняем карту признаков
            current = pools[i].pool(current);  // Downsample
        }
        return make_pair(current, features);   // Возвращаем конечную закодированную матрицу и карты признаков
    }

    Matrix decode(const Matrix& encoded, const vector<Matrix>& skip_connections) {
        Matrix current = encoded;
        for (int i = static_cast<int>(decoders.size()) - 1; i >= 0; --i) {
            current = decoders[i].upconv(current); // Upsample
            if (!skip_connections.empty())
                current += skip_connections[i];     // Добавляем пропущенное соединение
        }
        return current;
    }

    Matrix forward(const Matrix& input) {
        auto result = encode(input);              // Получаем закодированное представление и карты признаков
        auto encoded = result.first;
        auto features = result.second;
        return decode(encoded, features);         // Передаём карты признаков
    }

    double dice_loss(const Matrix& pred, const Matrix& target) {
        double intersection = 0.;
        double sum_pred = 0., sum_target = 0.;
        for (int i = 0; i < pred.rows; ++i) {
            for (int j = 0; j < pred.cols; ++j) {
                intersection += pred.data[i][j] * target.data[i][j];
                sum_pred += pred.data[i][j];
                sum_target += target.data[i][j];
            }
        }
        return 1. - (2.*intersection)/(sum_pred + sum_target + 1e-8); // Эпсильон для предотвращения деления на ноль
    }
};

int main() {
    // Создаем экземпляр сети
    UNet net(5); // Например, глубина равна 5
    
    // Остальные ваши тесты или логика программы
    return 0;
}

#include <vector>

class Matrix{
public:
    typedef std::vector<std::vector<double>> matrix;

    Matrix(int cols, int rows);
    
    static matrix multiple(matrix a, matrix b);
    void transparent();

    void random_fill(double start, double stop);
    void print_matrix();

    
    matrix m;
};
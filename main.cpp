#include <nndescent.hpp>
#include <cmath>
#include <random>

namespace plt = matplotlibcpp;

typedef float DataType;
typedef std::vector<DataType> point;

template <typename ValueType>
class Similarity
{
public:
    typedef ValueType value_type;
    virtual ValueType operator()(point p1, point p2) = 0;
};

class EuclideanSimilarity : public Similarity<DataType>
{
public:
    virtual DataType operator()(point p1, point p2)
    {
        DataType dist = 0;
        int dim = p1.size();
        assert(dim == p2.size());
        for (int i = 0; i < dim; i++)
        {
            float tmp = p1[i] - p2[i];
            dist += tmp * tmp;
        }
        return -std::sqrt(dist);
    }
};
void draw(std::vector<DataType> x, std::vector<DataType> y, const char *filename)
{
    plt::figure_size(1200, 1200);
    plt::scatter(x, y);
    // Set x-axis to interval [0,1000000]
    plt::xlim(0.0, 1.0);
    plt::ylim(0.0, 1.0);

    // Add graph title
    plt::title("Initial Points");
    // Enable legend.
    plt::legend();
    plt::save(filename);

    //plt::xlabel('Numbers',fontsize=14)
    plt::close();
}
int main(int argc, char **argv)
{
    // std::vector<int> v = {1, 10, 3, 8, 23, 7};
    // std::sort(v.begin(), v.end(), [](int a, int b) { return a < b; });
    // std::for_each(v.begin(), v.end(), [](int a) { std::cout << a << ","; });

    // std::cout << "\n";
    // std::size_t d = std::upper_bound(v.begin(), v.end(), 23, [](int a, int b) { return a > b; }) - v.begin();
    // std::cout << d << "," << v.back() << "\n";

    int dim = 2;
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<DataType> rand(0.0, 1.0);

    std::vector<DataType> x, y;
    std::vector<point> datas(100);
    for (int i = 0; i < datas.size(); i++)
    {
        for (int j = 0; j < dim; j++)
        {
            DataType v = rand(gen);
            datas[i].push_back(v);
            (j % dim == 0) ? x.push_back(v) : y.push_back(v);
        }
    }
    draw(x, y, "initial_point.jpg");
    nng::NNDescent<point, EuclideanSimilarity> nn(4);
    nn(datas);
    return 0;
}
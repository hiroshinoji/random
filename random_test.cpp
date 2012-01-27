#include "random.hpp"
#include <iostream>
#include <algorithm>
#include <map>
#include <memory>
#include <iterator>

using namespace std;
int main(int argc, char *argv[])
{
  unique_ptr<RandomBase> r(new RandomMT());

  cout << "some randoms" << endl;
  for (size_t i = 0; i < 3; ++i) {
    cout << r->NextDouble() << endl;
  }
  cout << endl;
  std::vector<double> alpha(10, 0.1);
  
  cout << "draw from dirichlet simplex" << endl;
  std::vector<double> dirichletSample = r->NextDirichlet(alpha);
  cout << "sum(normalizing constant) = " << std::accumulate(dirichletSample.begin(), dirichletSample.end(), 0.0) << endl;
  cout << "drawn sample:"<< endl;
  std::cout << "[ ";
  std::copy(dirichletSample.begin(), dirichletSample.end(),
            std::ostream_iterator<double>(cout, ", "));
  std::cout << "]" << endl << endl;

  cout << "draw from unnormalized dicrete" << endl;
  std::vector<double> pdf(5);
  pdf[0] = 20.0; pdf[1] = 10.0; pdf[2] = 3.0; pdf[3] = 1.0; pdf[4] = 1.0;
  
  map<int, int> drawCounts;
  cout << "1000 trial from the following multinomial" << endl;
  std::cout << "[ ";
  for (size_t i = 0; i < pdf.size(); ++i) {
    cout << i << ":" << pdf[i] << ", ";
  }
  std::cout << "]" << endl;
  cout << "result:" << endl;
  for (size_t i = 0; i < 10000000; ++i) {
    drawCounts[r->SampleUnnormalizedPdf(pdf)]++;
  }
  for (const auto &count : drawCounts) {
    cout << count.first << " : " << count.second << endl;
  }
  return 0;
}


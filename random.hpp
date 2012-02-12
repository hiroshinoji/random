#ifndef _RANDOM_H_
#define _RANDOM_H_

#include <iostream>
#include <functional>
#include <random>
#include <climits>
#include <cassert>
#include <cmath>
#include <unordered_map>

namespace {

double fast_logsumexp(double a, double b) {
  if (a == -INFINITY && b == -INFINITY) return -INFINITY;
  if (a>b) {
    return log(1+exp(b-a)) + a;
  } else {
    return log(1+exp(a-b)) + b;
  }
}

double fast_logsumexp(std::vector<double> a) {
  if (!a.empty()) {
    double z = a[0];
    for (size_t i = 1; i < a.size(); ++i) {
      z = fast_logsumexp(z, a[i]);
    }
    return z;
  } else {
    return 1;
  }
}

}


class RandomBase {
public:
  virtual ~RandomBase() {}

  virtual double NextDouble() = 0;
  virtual double NextGaussian(double mean, double stddev) = 0;

  double operator()() {
    return NextDouble();
  }
  
  long int operator()(long int max) {
    return NextMult(max);
  }

  long int NextMult(long int max) {
    return NextDouble() * max;
  }

  bool NextBernoille(double trueProb) {
    return trueProb > NextDouble();
  }

  double NextGamma(double a) {
    double x, y, z;
    double u, v, w, b, c, e;
    int accept = 0;
    if (a < 1)
      {
        /* Johnk's generator. Devroye (1986) p.418 */
        e = -log(NextDouble());
        do {
          x = pow(NextDouble(), 1 / a);
          y = pow(NextDouble(), 1 / (1 - a));
        } while (x + y > 1);
        return (e * x / (x + y));
      } else {
      /* Best's rejection algorithm. Devroye (1986) p.410 */
      b = a - 1;
      c = 3 * a - 0.75;
      do {
        /* generate */
        u = NextDouble();
        v = NextDouble();
        w = u * (1 - u);
        y = sqrt(c / w) * (u - 0.5);
        x = b + y;
        if (x >= 0)
          {
            z = 64 * w * w * w * v * v;
            if (z <= 1 - (2 * y * y) / x)
              {
                accept = 1;
              } else {
              if (log(z) < 2 * (b * log(x / b) - y))
                accept = 1;
            }
          }
      } while (accept != 1);
      return x;
    }
  }

  double NextGamma(double a, double b) {
    return NextGamma(a) / b;
  }

  double NextBeta(double a, double b) {
    double x = NextGamma(a), y = NextGamma(b);
    return x / (x + y);
  }
  
  template <typename ValueType>
  std::unordered_map<int, double> NextDirichlet(const std::unordered_map<int, ValueType>& alpha, double prec = 0) {
    std::unordered_map<int, double> theta;
    double z = 0;
    /* theta must have been allocated */
    for (const auto& item : alpha) {
      const int key = item.first;
      const ValueType value = item.second;
      if (prec != 0) {
        theta[key] = NextGamma(value * prec);
      } else {
        theta[key] = NextGamma(value);
      }
      if (theta[key] < 0.00000001) theta[key] = 0.00000001;
      z += theta[key];
    }
    for (auto& item : theta) {
      item.second /= z;
    }
    return theta;
  }

  template <typename ValueType>
  std::vector<double> NextDirichlet(const std::vector<ValueType>& alpha, double prec = 0) {
    std::vector<double> theta(alpha.size());
    double z = 0;
    /* theta must have been allocated */
    for (size_t i = 0; i < alpha.size(); i++) {
      if (alpha[i] != 0) {
        if (prec != 0) {
          theta[i] = NextGamma(alpha[i] * prec);
        } else {
          theta[i] = NextGamma(alpha[i]);
        }
        z += theta[i];
      }
    }
    for (size_t i = 0; i < alpha.size(); i++) {
      if (alpha[i] != 0) {
        theta[i] /= z;
      }
    }
    return theta;
  }
  
  int SampleUnnormalizedPdf(std::vector<double> pdf, int endPos = 0) {
    assert(pdf.size() > 0);
    assert(endPos < (int)pdf.size());
    assert(endPos >= 0);

    // if endPos == 0, use entire vector
    if (endPos == 0) {
      endPos = pdf.size()-1;
    }
    
    // compute CDF (inplace)
    for (int i = 0; i < endPos; ++i) {
      if (pdf[i] < 0) {
        std::cerr << "pdf[i] " << pdf[i] << std::endl;
      }
      assert(pdf[i] >= 0);
      pdf[i+1] += pdf[i];
    }

    assert(pdf[endPos] > 0);

    // sample pos ~ Uniform(0,Z)
    double z = NextDouble() * pdf[endPos];

    assert((z >= 0) && (z <= pdf[endPos]));
    
    int x = std::lower_bound(pdf.begin(), pdf.begin() + endPos + 1, z)
      - pdf.begin();

    assert(x == 0 || pdf[x-1] != pdf[x]);
    return x;
  }

  // sample from vector of log probabilities
  int SampleLogPdf(std::vector<double> lpdf) {
    double logSum = fast_logsumexp(lpdf);

    for (size_t i = 0; i < lpdf.size(); ++i) {
      lpdf[i] = exp(lpdf[i] - logSum);
    }
    return SampleUnnormalizedPdf(lpdf);
  }
  
  int SamplePdfAnnealing(std::vector<double> pdf, double templature = 1, int endPos = 0) {
    double s = 0;
    if (endPos == 0) {
      endPos = pdf.size() - 1;
    }
    for (int i = 0; i <= endPos; ++i) {
      s += pdf[i];
    }
    for (int i = 0 ; i <= endPos; ++i) {
      pdf[i] = pow((pdf[i] / s), templature);
    }
    return SampleUnnormalizedPdf(move(pdf), endPos);
  }
  
  template <typename KeyType, typename ValueType>
  KeyType SampleUnnormalizedPdf(const std::unordered_map<KeyType, ValueType>& pdf) {
    double r;
    double s = 0;
    for (const auto& atom : pdf) {
      s += atom.second;
    }
    r = NextDouble() * s;
    s = 0;
    KeyType last = (*pdf.begin()).first;
    for (const auto& atom : pdf) {
      s += atom.second;
      last = atom.first;
      if (r <= s) {
        return last;
      }
    }
    return last;
  }
};

class RandomMT : public RandomBase {
public:
  RandomMT(int seed = 10)
    : gen(std::bind(std::uniform_real_distribution<double>(0.0, 1.0),
                    std::mt19937(seed))) {
  }
  virtual double NextDouble() {
    return gen();
  }
  virtual double NextGaussian(double mean, double stddev) {
    return std::bind(std::normal_distribution<>(mean, stddev), std::mt19937())();
  }
private:
  std::function<double(void)> gen;
};

class RandomRand : public RandomBase {
public:
  RandomRand(int seed = 10)
    : gen(rand) { srand(seed); }
  virtual double NextDouble() {
    return static_cast<double>(gen()) / static_cast<double>(INT_MAX);
  }
  virtual double NextGaussian(double, double) {
    throw std::string("RandomRand does not support NextGaussian()!!!");
  }
private:
  std::function<double(void)> gen;
};

#endif /* _RANDOM_H_ */

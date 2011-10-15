#include <random>
#include <cassert>
#include <cmath>

class Random {
public:
  Random(int seed = 10)
    : gen(std::bind(std::uniform_real_distribution<double>(0.0, 1.0),
                    std::mt19937(seed))) {}

  double NextDouble() {
    return gen();
  }

  int NextInt(long int max) {
    return NextDouble() * (max-1);
  }

  bool NextBernoille(double trueProb) {
    return trueProb > NextDouble();
  }

  // from mots quotidiens. http://chasen.org/~daiti-m/diary/?20051228
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
  
  // from mots quotidiens. http://chasen.org/~daiti-m/diary/?20051228
  // modified a little.
  // This is the clone of the MATLAB dirichlet simplex generator
  std::vector<double> NextDirichlet(const std::vector<double>& alpha, double prec = 0) {
    std::vector<double> theta(alpha.size());
    int i;
    double z = 0;
    /* theta must have been allocated */
    for (i = 0; i < (int)alpha.size(); i++)
      if (prec != 0)
        theta[i] = NextGamma(alpha[i] * prec);
      else
        theta[i] = NextGamma(alpha[i]);
    for (i = 0; i < (int)alpha.size(); i++)
      z += theta[i];
    for (i = 0; i < (int)alpha.size(); i++)
      theta[i] /= z;
    return theta;
  }

  
  /**
   * Sample from a discrete distribution on 0,...,MAX with the given PDF.
   *
   * The probability if returning x for x \in 0,...,MAX is given by 
   *   pdf[x] / (\sum_{i=0,...,end_pos} pdf[i])
   * i.e. pdf is normalized so that the sum of all elements up to and including
   * element end_pos is 1.
   *  
   * Algorithm:
   *   1) Compute CDF; normalizing constant Z = CDF[end_pos]
   *   2) Sample z ~ Uniform(0,Z)
   *   3) find the smallest element x of CDF that is larger than i using binary
   *      search
   *   4) return x
   *
   * Complexity: O(log MAX)
   */
  int SampleUnnormalizedPdf(std::vector<double> pdf, int endPos = 0) {
    assert(pdf.size() > 0);
    assert(endPos < pdf.size());
    assert(endPos >= 0);

    // if endPos == 0, use entire vector
    if (endPos == 0) {
      endPos = pdf.size()-1;
    }
    
    // compute CDF (inplace)
    for (int i = 0; i < endPos; ++i) {
      assert(pdf[i] >= 0);
      pdf[i+1] += pdf[i];
    }

    assert(pdf[endPos] > 0);

    // sample pos ~ Uniform(0,Z)
    double z = NextDouble() * pdf[endPos];

    assert((z >= 0) && (z <= pdf[endPos]));
    
    // Perform binary search for z using std::lower_bound.
    // lower_bound(begin, end, x) returns the first element within [begin,end)
    // that is equal or larger than x.
    int x = std::lower_bound(pdf.begin(), pdf.begin() + endPos + 1, z)
      - pdf.begin();

    assert(x == 0 || pdf[x-1] != pdf[x]);
    return x;
  }

private:
  std::function<double(void)> gen;
};

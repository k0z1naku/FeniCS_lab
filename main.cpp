//
//     - div grad u(x, y) = f(x, y)
//
// on the unit square with source f given by
//
//     f(x, y) = heat transfer function
//
// and boundary conditions given by
//
//     u(x, y)     = u0 on x = 0 and x = 1
//     u(x, y) + du/dn(x, y) = g  on y = 0 and y = 1 
// using a discontinuous Galerkin formulation (interior penalty method).
#include <dolfin.h>
#include <cmath>
#include "Poisson.h"

using namespace dolfin;

// Source term (right-hand side)
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    double dx = x[0] - 0.5;
    double dy = x[1] - 0.5;
    values[0] = exp(-(dx*dx + dy*dy) / 0.1) - cos(2 * M_PI * x[1]);
  }
};


// Sub domain for Dirichlet boundary condition, x = 0, x = 1, and y = 0
class DirichletBoundary : public SubDomain
{
public:
    DirichletBoundary(std::function<double(double)> g1, std::function<double(double)> g2, std::function<double(double)> g3)
        : g1(g1), g2(g2), g3(g3) {}

    bool inside(const Array<double>& x, bool on_boundary) const override
    {
        return on_boundary and ((near(x[0], 0.0) && near(x[1], 0.0)) || near(x[0], 1.0));
    }

    void value(const Array<double>& x, Array<double>& values) const
    {
        if (near(x[1], 0.0)) // y = 0
            values[0] = g3(x[0]);
        else if (near(x[0], 0.0)) // x = 0
            values[0] = g1(x[1]);
        else if (near(x[0], 1.0)) // x = 1
            values[0] = g2(x[1]);
    }

private:
    std::function<double(double)> g1; // Function g1(y)
    std::function<double(double)> g2; // Function g2(y)
    std::function<double(double)> g3; // Function g3(x)
};

class RobinBoundary : public Expression {
public:
    RobinBoundary(double alpha, double beta) : alpha(alpha), beta(beta) {}

    void eval(Array<double>& values, const Array<double>& x) const override {
        const double dx = x[0] - 0.5;
        const double dy = x[1] - 0.5;
        const double neumann_term = dx * dy; // Example Neumann term
        const double dirichlet_term = sin(2 * M_PI * x[0]); // Example Dirichlet term

        values[0] = alpha * neumann_term + beta * dirichlet_term;
    }

private:
    double alpha; // Coefficient for Neumann term
    double beta;  // Coefficient for Dirichlet term
};


int main()
{      
  // Create mesh and function space
  auto mesh = std::make_shared<Mesh>(
    UnitSquareMesh::create({{32, 32}}, CellType::Type::triangle));
  auto V = std::make_shared<Poisson::FunctionSpace>(mesh);

  // Define boundary condition
auto g1 = [](double y) { return sin(2 * M_PI * y) * sin(2 * M_PI * y); }; // Example function g1(y)
auto g2 = [](double y) { return cos(2 * M_PI * y) * cos(2 * M_PI * y); }; // Example function g2(y)
auto g3 = [](double x) { return x*x; }; // Example function g3(x)

  auto u0 = std::make_shared<Constant>(0.0);
  auto boundary = std::make_shared<DirichletBoundary>(g1, g2, g3);
  DirichletBC bc(V, u0, boundary);

  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);
  auto f = std::make_shared<Source>();
  auto g = std::make_shared<RobinBoundary>(1, 1);
  L.f = f;
  L.g = g;

  // Compute solution
  Function u(V);
  solve(a == L, u, bc);

  // Save solution in VTK format
  File file("RobinandDirichlet.pvd");
  file << u;

  return 0;
}

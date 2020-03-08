#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <sys/time.h>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;

// camera intrinsic
const double uc = 1600;
const double vc = 900;
const double fu = 2000;
const double fv = 2000;
const double img_width = 3200;
const double img_height = 1800;
const double img_bound = 40;

// map point perturbation
const double x_perturbation = 0.01; //m
const double y_perturbation = 0.01; //m
const double z_perturbation = 0.01; //m

double rand(double min, double max)
{
  return min + (max - min) * double(rand()) / RAND_MAX;
}

void random_pose(Eigen::Matrix4d & T_cw)
{
  const double range = 1;

  double phi   = rand(0, range * 3.14159 * 2);
  double theta = rand(0, range * 3.14159);
  double psi   = rand(0, range * 3.14159 * 2);

  double R[3][3];
  double t[3];

  R[0][0] = cos(psi) * cos(phi) - cos(theta) * sin(phi) * sin(psi);
  R[0][1] = cos(psi) * sin(phi) + cos(theta) * cos(phi) * sin(psi);
  R[0][2] = sin(psi) * sin(theta);

  R[1][0] = -sin(psi) * cos(phi) - cos(theta) * sin(phi) * cos(psi);
  R[1][1] = -sin(psi) * sin(phi) + cos(theta) * cos(phi) * cos(psi);
  R[1][2] = cos(psi) * sin(theta);

  R[2][0] = sin(theta) * sin(phi);
  R[2][1] = -sin(theta) * cos(phi);
  R[2][2] = cos(theta);

  t[0] = rand(0.0f, 1.0f);
  t[1] = rand(0.0f, 1.0f);
  t[2] = rand(0.0f, 1.0f);

  T_cw(0,0) = R[0][0]; T_cw(0,1) = R[0][1]; T_cw(0,2) = R[0][2];
  T_cw(1,0) = R[1][0]; T_cw(1,1) = R[1][1]; T_cw(1,2) = R[1][2];
  T_cw(2,0) = R[2][0]; T_cw(2,1) = R[2][1]; T_cw(2,2) = R[2][2];
  T_cw(0,3) = t[0];
  T_cw(1,3) = t[1];
  T_cw(2,3) = t[2];
}

void random_point(Eigen::Vector3d & map_point, double R)
{
    map_point = Eigen::Vector3d::Zero();

    double Xw, Yw, Zw;
    double theta = rand(0, 3.14159), phi = rand(0, 2 * 3.14159);

    Xw =  sin(theta) * sin(phi) * R;
    Yw = -sin(theta) * cos(phi) * R;
    Zw =  cos(theta) * R;

    map_point[0] = Xw;
    map_point[1] = Yw;
    map_point[2] = Zw;
}

bool project_without_noise(const Eigen::Matrix4d & T_cw,
                            const Eigen::Vector3d & map_points,
                            double &img_u, double & img_v)
{
    if(T_cw.hasNaN() || map_points.hasNaN())
    {
        std::cout << "[project_without_noise] Invalid input matrix or vector!\n";
        return false;
    }

    const double eps = std::numeric_limits<double>::epsilon();
    Eigen::Vector3d projected_point = (T_cw * map_points.homogeneous()).hnormalized();
    if(projected_point[2] <= eps)
    {
        std::cout << "[project_without_noise] Invalid projection!\n";
        return false;
    }

    img_u = uc + fu * projected_point[0] / projected_point[2];
    img_v = vc + fv * projected_point[1] / projected_point[2];

    if(img_bound < img_u && img_u < img_width-img_bound && img_bound < img_v && img_v < img_height-img_bound)
        return true;
    else
        return false;
}

int main(int /*argc*/, char ** /*argv*/)
{

  srand(time(0));

  Eigen::Matrix4d T_cw = Eigen::Matrix4d::Identity();
  Eigen::Vector3d world_point = Eigen::Vector3d::Zero();
  double world_point_radius = 1.0; // m

//  random_pose(T_cw);
  T_cw << -0.0536623, -0.0151586, -0.998444, -0.301355,
          -0.0562737, -0.99825, 0.0181801, -0.288975,
          -0.996972, 0.0571618, 0.0527154, -0.271971,
          0, 0, 0, 1;

  double total_pixel_error = 0.0;

  size_t it_num = 100;
  for(size_t i = 0; i < it_num; ++i ){
      bool b_valid_projection = false;
      double true_img_x = 0, true_img_y = 0;

      while(!b_valid_projection){

          random_point(world_point, world_point_radius);

          b_valid_projection =  project_without_noise(T_cw, world_point, true_img_x, true_img_y);

      }

      std::cout << "Chosen map point : " << world_point.transpose() << "\n";
      std::cout << "Before perturb: projected pixel : " << true_img_x << ", " << true_img_y << "\n";

      // add x/z perturbation on world_point
      world_point[0] += x_perturbation;//m

      double perturb_img_x = 0, perturb_img_y = 0;
      project_without_noise(T_cw, world_point, perturb_img_x, perturb_img_y);
      std::cout << "After perturb: projected pixel : " << perturb_img_x << ", " << perturb_img_y << "\n";
      double x_err = true_img_x - perturb_img_x;
      double y_err = true_img_y- perturb_img_y;
      std::cout << "X error : " << x_err << ", Y error : " << y_err << "\n";
      double error = std::sqrt( x_err * x_err + y_err * y_err);
      std::cout << " Distance in pixel : " << error << "\n";

      total_pixel_error += error;
  }

  std::cout << "Average pixel error under the x perturbation of map point : " << total_pixel_error/it_num << "\n";

  return 0;
}

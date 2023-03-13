#include <Eigen/Dense>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>
#include <set>
#include <vector>
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

using namespace std;
namespace py = pybind11;

namespace Eigen {
    typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;
}




namespace grid_map_raycasting {
    Eigen::MatrixXb rayCastGridMap(Eigen::MatrixXd grid_map){

    Eigen::MatrixXb mask(grid_map.rows(), grid_map.cols());
    mask.setConstant(false);

    octomap::ColorOcTree* ground_truth_model = new octomap::ColorOcTree(1.0);
    ground_truth_model->setClampingThresMax(1.0);   //设置地图节点最大值，初始0.971
    ground_truth_model->setClampingThresMin(0.0);   //设置地图节点最小值，初始0.1192
    ground_truth_model->setOccupancyThres(0.7);   //设置节点占用阈值，初始0.5

    for (int i = 0; i < 434; i++) {
       for (int j = 0; j < 480; j++) {
          octomap::OcTreeKey key_sp;  bool key_have_sp = ground_truth_model->coordToKeyChecked(octomap::point3d(1.0*i, 1.0*j, 0.0), key_sp);
          if (key_have_sp) {
             octomap::ColorOcTreeNode* voxel_sp = ground_truth_model->search(key_sp);
             if (voxel_sp == NULL) {
                ground_truth_model->setNodeValue(key_sp, octomap::logodds(grid_map(i, j)), true);
                if(grid_map(i, j)>0.7) ground_truth_model->setNodeColor(key_sp, 0, 0, 255);
                else if(grid_map(i, j)<0.3) ground_truth_model->setNodeColor(key_sp, 0, 255, 0);
                else ground_truth_model->setNodeColor(key_sp, 128, 128, 128);
             }
          }
       }
    }
    ground_truth_model->updateInnerOccupancy();
    //ground_truth_model->write("C:\\Users\\ALIENWARE\\OneDrive\\桌面\\raycasting_2D\\texting_img.ot");

    unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash>* hit_mask;
   hit_mask = new unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash>();

   octomap::point3d origin_mid = octomap::point3d(1.0, 240.0, 0.0);
   octomap::point3d origin_left = octomap::point3d(1.0, 60.0, 0.0);
   octomap::point3d origin_right = octomap::point3d(1.0, 350.0, 0.0);
   //mid view
   for (double j = 49; j < 428; j += 0.5) {
      //反向投影找到终点
      octomap::point3d end = octomap::point3d(429.0, 1.0*j, 0.0);
      octomap::OcTreeKey key_end;
      octomap::point3d direction = end - origin_mid;
      octomap::point3d end_point;
      //越过未知区域，找到终点
      bool found_end_point = ground_truth_model->castRay(origin_mid, direction, end_point, true, 650.0);
      if (!found_end_point) {//未找到终点，无观测数据
         continue;
      }
      if (end_point == origin_mid) {
         continue;
      }
      //检查一下末端是否在地图限制范围内
      bool key_end_have = ground_truth_model->coordToKeyChecked(end_point, key_end);
      if (key_end_have) {
         octomap::ColorOcTreeNode* node = ground_truth_model->search(key_end);
         if (node != NULL) {
            hit_mask->insert(key_end);
            //cout << end_point.x() << " " << end_point.y() << endl;
            ground_truth_model->setNodeColor(key_end, 255, 0, 0);
         }
      }
   }
   //left view
   for (double j = 49; j < 428; j += 0.5) {
      //反向投影找到终点
      octomap::point3d end = octomap::point3d(429.0, 1.0*j, 0.0);
      octomap::OcTreeKey key_end;
      octomap::point3d direction = end - origin_left;
      octomap::point3d end_point;
      //越过未知区域，找到终点
      bool found_end_point = ground_truth_model->castRay(origin_left, direction, end_point, true, 650.0);
      if (!found_end_point) {//未找到终点，无观测数据
         continue;
      }
      if (end_point == origin_left) {
         continue;
      }
      //检查一下末端是否在地图限制范围内
      bool key_end_have = ground_truth_model->coordToKeyChecked(end_point, key_end);
      if (key_end_have) {
         octomap::ColorOcTreeNode* node = ground_truth_model->search(key_end);
         if (node != NULL) {
            hit_mask->insert(key_end);
            //cout << end_point.x() << " " << end_point.y() << endl;
            ground_truth_model->setNodeColor(key_end, 255, 0, 0);
         }
      }
   }
   //right view
   for (double j = 49; j < 428; j += 0.5) {
      //反向投影找到终点
      octomap::point3d end = octomap::point3d(429.0, 1.0*j, 0.0);
      octomap::OcTreeKey key_end;
      octomap::point3d direction = end - origin_right;
      octomap::point3d end_point;
      //越过未知区域，找到终点
      bool found_end_point = ground_truth_model->castRay(origin_right, direction, end_point, true, 650.0);
      if (!found_end_point) {//未找到终点，无观测数据
         continue;
      }
      if (end_point == origin_right) {
         continue;
      }
      //检查一下末端是否在地图限制范围内
      bool key_end_have = ground_truth_model->coordToKeyChecked(end_point, key_end);
      if (key_end_have) {
         octomap::ColorOcTreeNode* node = ground_truth_model->search(key_end);
         if (node != NULL) {
            hit_mask->insert(key_end);
            //cout << end_point.x() << " " << end_point.y() << endl;
            ground_truth_model->setNodeColor(key_end, 255, 0, 0);
         }
      }
   }

   ground_truth_model->updateInnerOccupancy();
   //ground_truth_model->write("C:\\Users\\ALIENWARE\\OneDrive\\桌面\\raycasting_2D\\texting_img_mask.ot");

   //vector<vector<bool>> mask;
   //for (int i = 0; i < 434; i++) {
   //   vector<bool> row;
   //   for (int j = 0; j < 480; j++) {
   //      row.push_back(false);
   //   }
   //   mask.push_back(row);
   //}

   for (int i = 0; i < 434; i++) {
      for (int j = 0; j < 480; j++) {
         octomap::OcTreeKey key_sp;  bool key_have_sp = ground_truth_model->coordToKeyChecked(octomap::point3d(1.0 * i, 1.0 * j, 0.0), key_sp);
         if (key_have_sp) {
            if (hit_mask->count(key_sp)) {
                if(i>400) continue;
               mask(i, j) = true;
               //cout << i << " " << j << endl;
            }
         }
      }
   }
  return mask;
  }

 Eigen::MatrixXb rayCastGridMapReal(Eigen::MatrixXd grid_map){

    Eigen::MatrixXb mask(grid_map.rows(), grid_map.cols());
    mask.setConstant(false);

    octomap::ColorOcTree* ground_truth_model = new octomap::ColorOcTree(1.0);
    ground_truth_model->setClampingThresMax(1.0);   //设置地图节点最大值，初始0.971
    ground_truth_model->setClampingThresMin(0.0);   //设置地图节点最小值，初始0.1192
    ground_truth_model->setOccupancyThres(0.7);   //设置节点占用阈值，初始0.5

    for (int i = 0; i < 300; i++) {
       for (int j = 0; j < 400; j++) {
          octomap::OcTreeKey key_sp;  bool key_have_sp = ground_truth_model->coordToKeyChecked(octomap::point3d(1.0*i, 1.0*j, 0.0), key_sp);
          if (key_have_sp) {
             octomap::ColorOcTreeNode* voxel_sp = ground_truth_model->search(key_sp);
             if (voxel_sp == NULL) {
                ground_truth_model->setNodeValue(key_sp, octomap::logodds(grid_map(i, j)), true);
                if(grid_map(i, j)>0.7) ground_truth_model->setNodeColor(key_sp, 0, 0, 255);
                else if(grid_map(i, j)<0.3) ground_truth_model->setNodeColor(key_sp, 0, 255, 0);
                else ground_truth_model->setNodeColor(key_sp, 128, 128, 128);
             }
          }
       }
    }
    ground_truth_model->updateInnerOccupancy();
    //ground_truth_model->write("C:\\Users\\ALIENWARE\\OneDrive\\桌面\\raycasting_2D\\texting_img.ot");

    unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash>* hit_mask;
   hit_mask = new unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash>();

   octomap::point3d origin_mid = octomap::point3d(-99.0, 200.0, 0.0);
   octomap::point3d origin_left = octomap::point3d(-99.0, 20.0, 0.0);
   octomap::point3d origin_right = octomap::point3d(-99.0, 375.0, 0.0);
   //mid view
   for (double j = 15; j < 382; j += 0.5) {
      //反向投影找到终点
      octomap::point3d end = octomap::point3d(285.0, 1.0*j, 0.0);
      octomap::OcTreeKey key_end;
      octomap::point3d direction = end - origin_mid;
      octomap::point3d end_point;
      //越过未知区域，找到终点
      bool found_end_point = ground_truth_model->castRay(origin_mid, direction, end_point, true, 650.0);
      if (!found_end_point) {//未找到终点，无观测数据
         continue;
      }
      if (end_point == origin_mid) {
         continue;
      }
      //检查一下末端是否在地图限制范围内
      bool key_end_have = ground_truth_model->coordToKeyChecked(end_point, key_end);
      if (key_end_have) {
         octomap::ColorOcTreeNode* node = ground_truth_model->search(key_end);
         if (node != NULL) {
            hit_mask->insert(key_end);
            //cout << end_point.x() << " " << end_point.y() << endl;
            ground_truth_model->setNodeColor(key_end, 255, 0, 0);
         }
      }
   }
   //left view
   for (double j = 15; j < 382; j += 0.5) {
      //反向投影找到终点
      octomap::point3d end = octomap::point3d(285.0, 1.0*j, 0.0);
      octomap::OcTreeKey key_end;
      octomap::point3d direction = end - origin_left;
      octomap::point3d end_point;
      //越过未知区域，找到终点
      bool found_end_point = ground_truth_model->castRay(origin_left, direction, end_point, true, 650.0);
      if (!found_end_point) {//未找到终点，无观测数据
         continue;
      }
      if (end_point == origin_left) {
         continue;
      }
      //检查一下末端是否在地图限制范围内
      bool key_end_have = ground_truth_model->coordToKeyChecked(end_point, key_end);
      if (key_end_have) {
         octomap::ColorOcTreeNode* node = ground_truth_model->search(key_end);
         if (node != NULL) {
            hit_mask->insert(key_end);
            //cout << end_point.x() << " " << end_point.y() << endl;
            ground_truth_model->setNodeColor(key_end, 255, 0, 0);
         }
      }
   }
   //right view
   for (double j = 15; j < 382; j += 0.5) {
      //反向投影找到终点
      octomap::point3d end = octomap::point3d(285.0, 1.0*j, 0.0);
      octomap::OcTreeKey key_end;
      octomap::point3d direction = end - origin_right;
      octomap::point3d end_point;
      //越过未知区域，找到终点
      bool found_end_point = ground_truth_model->castRay(origin_right, direction, end_point, true, 650.0);
      if (!found_end_point) {//未找到终点，无观测数据
         continue;
      }
      if (end_point == origin_right) {
         continue;
      }
      //检查一下末端是否在地图限制范围内
      bool key_end_have = ground_truth_model->coordToKeyChecked(end_point, key_end);
      if (key_end_have) {
         octomap::ColorOcTreeNode* node = ground_truth_model->search(key_end);
         if (node != NULL) {
            hit_mask->insert(key_end);
            //cout << end_point.x() << " " << end_point.y() << endl;
            ground_truth_model->setNodeColor(key_end, 255, 0, 0);
         }
      }
   }

   ground_truth_model->updateInnerOccupancy();
   //ground_truth_model->write("C:\\Users\\ALIENWARE\\OneDrive\\桌面\\raycasting_2D\\texting_img_mask.ot");

   //vector<vector<bool>> mask;
   //for (int i = 0; i < 434; i++) {
   //   vector<bool> row;
   //   for (int j = 0; j < 480; j++) {
   //      row.push_back(false);
   //   }
   //   mask.push_back(row);
   //}

   for (int i = 0; i < 300; i++) {
      for (int j = 0; j < 400; j++) {
         octomap::OcTreeKey key_sp;  bool key_have_sp = ground_truth_model->coordToKeyChecked(octomap::point3d(1.0 * i, 1.0 * j, 0.0), key_sp);
         if (key_have_sp) {
            if (hit_mask->count(key_sp)) {
                if(i<30) continue;
                if(i>250) continue;
                if(j>350) continue;
                if(j<40) continue;
                mask(i, j) = true;
                //cout << i << " " << j << endl;
            }
         }
      }
   }
  return mask;
  }
}


PYBIND11_MODULE(grid_map_raycasting, m) {
    m.doc() = R"pbdoc(
        C++ component including Python bindings to raycast a gridmap from a viewpoint to check for occlusions
        -----------------------

        .. currentmodule:: grid_map_raycasting

        .. autosummary::
           :toctree: _generate
    )pbdoc";

    m.def("rayCastGridMap", &grid_map_raycasting::rayCastGridMap, R"pbdoc(
        Raycast every cell on the grid from a constant origin of the ray.

        It returns a grid map of booleans which signify weather the grid cell is visible from the vantage point of the robot or if its hidden by the terrain.
        Formulated alternatively, it creates an occlusion mask for a given Digital Elevation Map (DEM) which stores true for occluded and false for visible.
    )pbdoc", py::arg("grid_map"));

    m.def("rayCastGridMapReal", &grid_map_raycasting::rayCastGridMapReal, R"pbdoc(
        Raycast every cell on the grid from a constant origin of the ray.

        It returns a grid map of booleans which signify weather the grid cell is visible from the vantage point of the robot or if its hidden by the terrain.
        Formulated alternatively, it creates an occlusion mask for a given Digital Elevation Map (DEM) which stores true for occluded and false for visible.
    )pbdoc", py::arg("grid_map"));


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

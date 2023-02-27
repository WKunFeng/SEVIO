#ifndef ESVO_CORE_MAPPING_H
#define ESVO_CORE_MAPPING_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Imu.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <tf2_ros/transform_broadcaster.h>

#include <esvo_core/container/CameraSystem.h>
#include <esvo_core/container/DepthMap.h>
#include <esvo_core/container/EventMatchPair.h>
#include <esvo_core/core/DepthFusion.h>
#include <esvo_core/core/DepthRegularization.h>
#include <esvo_core/core/DepthProblem.h>
#include <esvo_core/core/DepthProblemSolver.h>
#include <esvo_core/core/EventBM.h>
#include <esvo_core/tools/utils.h>
#include <esvo_core/tools/Visualization.h>

#include <dynamic_reconfigure/server.h>
#include <esvo_core/DVS_MappingStereoConfig.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <map>
#include <deque>
#include <mutex>
#include <future>

#include <cv_bridge/cv_bridge.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>

#include <sophus/so3.hpp>

namespace esvo_core
{
using namespace core;

/***********************************************imu****************************************************/

struct IMUData
{
  double time = 0.0;
  Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
};

/***********************************************imu****************************************************/


class esvo_Mapping
{
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  esvo_Mapping(const ros::NodeHandle& nh,
               const ros::NodeHandle& nh_private);
  virtual ~esvo_Mapping();

  // mapping
  void MappingLoop(std::promise<void> prom_mapping, std::future<void> future_reset);
  void MappingAtTime(const ros::Time& t);
  bool InitializationAtTime(const ros::Time& t);
  bool dataTransferring();

  // callback functions
  void stampedPoseCallback(const geometry_msgs::PoseStampedConstPtr &ps_msg);
  void eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg, EventQueue& EQ);
  void timeSurfaceCallback(
    const sensor_msgs::ImageConstPtr& time_surface_left,
    const sensor_msgs::ImageConstPtr& time_surface_right);
  void onlineParameterChangeCallback(DVS_MappingStereoConfig &config, uint32_t level);

  // utils
  bool getPoseAt(const ros::Time& t, Transformation& Tr, const std::string& source_frame);
  void clearEventQueue(EventQueue& EQ);
  void reset();

  /*** publish results ***/
  void publishMappingResults(
    DepthMap::Ptr depthMapPtr,
    Transformation tr,
    ros::Time t);
  void publishPointCloud(
    DepthMap::Ptr& depthMapPtr,
    Transformation & tr,
    ros::Time& t);
  void publishImage(
    const cv::Mat &image,
    const ros::Time & t,
    image_transport::Publisher & pub,
    std::string encoding = "bgr8");

  /*** event processing ***/
  void createEdgeMask(
    std::vector<dvs_msgs::Event *>& vEventsPtr,
    PerspectiveCamera::Ptr& camPtr,
    cv::Mat& edgeMap,
    std::vector<std::pair<size_t, size_t> >& vEdgeletCoordinates,
    bool bUndistortEvents = true,
    size_t radius = 0);

  void createDenoisingMask(
    std::vector<dvs_msgs::Event *>& vAllEventsPtr,
    cv::Mat& mask,
    size_t row, size_t col);// reserve in this file

  void extractDenoisedEvents(
    std::vector<dvs_msgs::Event *> &vCloseEventsPtr,
    std::vector<dvs_msgs::Event *> &vEdgeEventsPtr,
    cv::Mat& mask,
    size_t maxNum = 5000);

  /***********************************************imu****************************************************/
  enum MeasurementType {
    POSE = 0,
    POSE_VEL,
    POSI,
    POSI_VEL,
    POSI_MAG,
    POSI_VEL_MAG,
    NUM_TYPES
  };
  
  /**
   * @class Measurement
   * @brief Kalman filter measurement data
   */
  struct Measurement {
    // timestamp:
    double time;
    // a. pose observation, lidar/visual frontend:
    Eigen::Matrix4d T_nb;
    // b. body frame velocity observation, odometer:
    Eigen::Vector3d v_b;
    // c. body frame angular velocity, needed by motion constraint:
    Eigen::Vector3d w_b;
    // d. magnetometer:
    Eigen::Vector3d B_b;
  };
  
  void imuCallback(const sensor_msgs::ImuConstPtr& imu_msg_ptr);

  void IMULoop();

  /**
   * @brief  Kalman update
   * @param  imu_data, input IMU measurements
   * @return true if success false otherwise
   */
  //bool Update(const IMUData &imu_data);
  bool Update();

  /**
   * @brief  Kalman correction, pose measurement
   * @param  measurement_type, input measurement type
   * @param  measurement, input measurement
   * @return void
   */
  bool Correct(const MeasurementType &measurement_type,
               const Measurement &measurement);

  /**
   * @brief  get unbiased angular velocity in body frame
   * @param  angular_vel, angular velocity measurement
   * @param  R, corresponding orientation of measurement
   * @return unbiased angular velocity in body frame
   */
  Eigen::Vector3d GetUnbiasedAngularVel(const Eigen::Vector3d &angular_vel);
  /**
   * @brief  get unbiased linear acceleration in navigation frame
   * @param  linear_acc, linear acceleration measurement
   * @param  R, corresponding orientation of measurement
   * @return unbiased linear acceleration in navigation frame
   */
  Eigen::Vector3d GetUnbiasedLinearAcc(const Eigen::Vector3d &linear_acc,
                                       const Eigen::Matrix3d &R);

  /**
   * @brief  get angular delta
   * @param  index_curr, current imu measurement buffer index
   * @param  index_prev, previous imu measurement buffer index
   * @param  angular_delta, angular delta output
   * @return true if success false otherwise
   */
  bool GetAngularDelta(const size_t index_curr, const size_t index_prev,
                       Eigen::Vector3d &angular_delta,
                       Eigen::Vector3d &angular_vel_mid);
  /**
   * @brief  get velocity delta
   * @param  index_curr, current imu measurement buffer index
   * @param  index_prev, previous imu measurement buffer index
   * @param  R_curr, corresponding orientation of current imu measurement
   * @param  R_prev, corresponding orientation of previous imu measurement
   * @param  velocity_delta, velocity delta output
   * @param  linear_acc_mid, mid-value unbiased linear acc
   * @return true if success false otherwise
   */
  bool GetVelocityDelta(const size_t index_curr, const size_t index_prev,
                        const Eigen::Matrix3d &R_curr,
                        const Eigen::Matrix3d &R_prev, double &T,
                        Eigen::Vector3d &velocity_delta,
                        Eigen::Vector3d &linear_acc_mid);
  /**
   * @brief  update orientation with effective rotation angular_delta
   * @param  angular_delta, effective rotation
   * @param  R_curr, current orientation
   * @param  R_prev, previous orientation
   * @return void
   */
  void UpdateOrientation(const Eigen::Vector3d &angular_delta,
                         Eigen::Matrix3d &R_curr, Eigen::Matrix3d &R_prev);
  /**
   * @brief  update orientation with effective velocity change velocity_delta
   * @param  velocity_delta, effective velocity change
   * @return void
   */
  void UpdatePosition(const double &T, const Eigen::Vector3d &velocity_delta);
  /**
   * @brief  update IMU odometry estimation
   * @param  linear_acc_mid, output mid-value unbiased linear acc
   * @return void
   */
  void UpdateOdomEstimation(Eigen::Vector3d &linear_acc_mid,
                            Eigen::Vector3d &angular_vel_mid);
  

  /**
   * @brief  set process equation
   * @param  C_nb, rotation matrix, body frame -> navigation frame
   * @param  f_n, accel measurement in navigation frame
   * @return void
   */
  void SetProcessEquation(const Eigen::Matrix3d &C_nb,
                          const Eigen::Vector3d &f_n,
                          const Eigen::Vector3d &w_n);
  /**
   * @brief  update process equation
   * @param  linear_acc_mid, input mid-value unbiased linear acc
   * @return void
   */
  void UpdateProcessEquation(const Eigen::Vector3d &linear_acc_mid,
                             const Eigen::Vector3d &angular_vel_mid);

  /**
   * @brief  update error estimation
   * @param  linear_acc_mid, input mid-value unbiased linear acc
   * @return void
   */
  void UpdateErrorEstimation(const double &T,
                             const Eigen::Vector3d &linear_acc_mid,
                             const Eigen::Vector3d &angular_vel_mid);

  /**
   * @brief  correct error estimation using pose measurement
   * @param  T_nb, input pose measurement
   * @return void
   */
  void CorrectErrorEstimationPose(const Eigen::Matrix4d &T_nb,
                                  Eigen::VectorXd &Y, Eigen::MatrixXd &G,
                                  Eigen::MatrixXd &K);

  /**
   * @brief  correct error estimation
   * @param  measurement_type, measurement type
   * @param  measurement, input measurement
   * @return void
   */
  void CorrectErrorEstimation(const MeasurementType &measurement_type,
                              const Measurement &measurement);

  /**
   * @brief  eliminate error
   * @param  void
   * @return void
   */
  void EliminateError(void);

  /**
   * @brief  reset filter state
   * @param  void
   * @return void
   */
  void ResetState(void);
  /**
   * @brief  reset filter covariance
   * @param  void
   * @return void
   */
  void ResetCovariance(void);


  /***********************************************imu****************************************************/

  /************************ member variables ************************/
  private:
  ros::NodeHandle nh_, pnh_;

  // Subscribers
  ros::Subscriber events_left_sub_, events_right_sub_;
  ros::Subscriber stampedPose_sub_;
  message_filters::Subscriber<sensor_msgs::Image> TS_left_sub_, TS_right_sub_;

  // Publishers
  ros::Publisher pc_pub_, gpc_pub_;
  image_transport::ImageTransport it_;
  double t_last_pub_pc_;

  // Time-Surface sync policy
  typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image> ExactSyncPolicy;
  message_filters::Synchronizer<ExactSyncPolicy> TS_sync_;

  // dynamic configuration (modify parameters online)
  boost::shared_ptr<dynamic_reconfigure::Server<DVS_MappingStereoConfig> > server_;
  dynamic_reconfigure::Server<DVS_MappingStereoConfig>::CallbackType dynamic_reconfigure_callback_;

  /***********************************************imu****************************************************/
  ros::Subscriber imu_;
  ros::Publisher imu_pose_pub_;

  std::deque<IMUData> imu_data_buff_;
  double time_;
  bool imuInitialization;
  int imunum;
  int imu_initial_num_max_;
  MeasurementType measurement_type;
  Measurement measurement;

  //EKF parameters
  //P
  double initial_covariance_translation_;
  double initial_covariance_velocity_;
  double initial_covariance_rotation_;
  double initial_covariance_acc_bias_;
  double initial_covariance_gyro_bias_;
  //Q
  double noise_acc_;
  double noise_gyro_;
  double noise_acc_bias_;
  double noise_gyro_bias_;
  //R
  double noise_observation_translation_;
  double noise_observation_rotation_;

  // indices:指数
  static constexpr int kDimState{15};

  static constexpr int kIndexErrorPos{0};
  static constexpr int kIndexErrorVel{3};
  static constexpr int kIndexErrorOri{6};
  static constexpr int kIndexErrorAccel{9};
  static constexpr int kIndexErrorGyro{12};

  static constexpr int kDimProcessNoise{12};

  static constexpr int kIndexNoiseAccel{0};
  static constexpr int kIndexNoiseGyro{3};
  static constexpr int kIndexNoiseBiasAccel{6};
  static constexpr int kIndexNoiseBiasGyro{9};

  // dimensions:维度
  static constexpr int kDimMeasurementPose{6};
  static const int kDimMeasurementPoseNoise{6};



  // state:
  using VectorX=Eigen::Matrix<double, kDimState, 1>;
  using MatrixP=Eigen::Matrix<double, kDimState, kDimState>;

  // process equation:
  using MatrixF=Eigen::Matrix<double, kDimState, kDimState>;
  using MatrixB=Eigen::Matrix<double, kDimState, kDimProcessNoise>;
  using MatrixQ=Eigen::Matrix<double, kDimProcessNoise, kDimProcessNoise>;

  // measurement equation:
  using MatrixGPose=Eigen::Matrix<double, kDimMeasurementPose,kDimState> ;
  using MatrixCPose=Eigen::Matrix<double, kDimMeasurementPose,kDimMeasurementPoseNoise>;
  using MatrixRPose=Eigen::Matrix<double, kDimMeasurementPoseNoise,kDimMeasurementPoseNoise>;

  // measurement:
  using VectorYPose=Eigen::Matrix<double, kDimMeasurementPose, 1>;

  // Kalman gain:
  using MatrixKPose=Eigen::Matrix<double, kDimState, kDimMeasurementPose>;

  // state observality matrix:
  using MatrixQPose=Eigen::Matrix<double, kDimState * kDimMeasurementPose, kDimState>;

  //gravity constant
  Eigen::Vector3d g_;

  // odometry estimation from IMU integration:
  Eigen::Matrix4d init_pose_ = Eigen::Matrix4d::Identity();

  //trans
  Eigen::Matrix4d T_imu_lcam;

  //
  Eigen::Matrix4d pose_ = Eigen::Matrix4d::Identity();
  Eigen::Vector3d vel_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_bias_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d accl_bias_ = Eigen::Vector3d::Zero();

  //ini
  Eigen::Vector3d sum_angular_vel = Eigen::Vector3d::Zero();
  Eigen::Vector3d sum_linear_acc = Eigen::Vector3d::Zero();

  // state:
  VectorX X_ = VectorX::Zero();
  MatrixP P_ = MatrixP::Identity();
  // process & measurement equations:
  MatrixF F_ = MatrixF::Zero();
  MatrixB B_ = MatrixB::Zero();
  MatrixQ Q_ = MatrixQ::Zero();

  MatrixGPose GPose_ = MatrixGPose::Zero();
  MatrixCPose CPose_ = MatrixCPose::Zero();
  MatrixRPose RPose_ = MatrixRPose::Zero();
  MatrixQPose QPose_ = MatrixQPose::Zero();

  // measurement:
  VectorYPose YPose_;
  
  /***********************************************imu****************************************************/


  // offline data
  std::string dvs_frame_id_;
  std::string world_frame_id_;
  std::string calibInfoDir_;
  CameraSystem::Ptr camSysPtr_;

  // online data
  EventQueue events_left_, events_right_;
  TimeSurfaceHistory TS_history_;
  StampedTimeSurfaceObs TS_obs_;
  StampTransformationMap st_map_;
  std::shared_ptr<tf::Transformer> tf_;
  size_t TS_id_;
  ros::Time tf_lastest_common_time_;

  // system
  std::string ESVO_System_Status_;
  DepthProblemConfig::Ptr dpConfigPtr_;
  DepthProblemSolver dpSolver_;
  DepthFusion dFusor_;
  DepthRegularization dRegularizor_;
  Visualization visualizor_;
  EventBM ebm_;

  // data transfer
  std::vector<dvs_msgs::Event *> vALLEventsPtr_left_;// for BM
  std::vector<dvs_msgs::Event *> vCloseEventsPtr_left_;// for BM
  std::vector<dvs_msgs::Event *> vDenoisedEventsPtr_left_;// for BM
  size_t totalNumCount_;// count the number of events involved
  std::vector<dvs_msgs::Event *> vEventsPtr_left_SGM_;// for SGM

  // result
  PointCloud::Ptr pc_, pc_near_, pc_global_;
  DepthFrame::Ptr depthFramePtr_;
  std::deque<std::vector<DepthPoint> > dqvDepthPoints_;

  // inter-thread management
  std::mutex data_mutex_;
  std::promise<void> mapping_thread_promise_, reset_promise_;
  std::future<void> mapping_thread_future_, reset_future_;

  /**** mapping parameters ***/
  // range and visualization threshold
  double invDepth_min_range_;
  double invDepth_max_range_;
  double cost_vis_threshold_;
  size_t patch_area_;
  double residual_vis_threshold_;
  double stdVar_vis_threshold_;
  size_t age_max_range_;
  size_t age_vis_threshold_;
  int fusion_radius_;
  std::string FusionStrategy_;
  int maxNumFusionFrames_;
  int maxNumFusionPoints_;
  size_t INIT_SGM_DP_NUM_Threshold_;
  // module parameters
  size_t PROCESS_EVENT_NUM_;
  size_t TS_HISTORY_LENGTH_;
  size_t mapping_rate_hz_;
  // options
  bool changed_frame_rate_;
  bool bRegularization_;
  bool resetButton_;
  bool bDenoising_;
  bool bVisualizeGlobalPC_;
  // visualization parameters
  double visualizeGPC_interval_;
  double visualize_range_;
  size_t numAddedPC_threshold_;
  // Event Block Matching (BM) parameters
  double BM_half_slice_thickness_;
  size_t BM_patch_size_X_;
  size_t BM_patch_size_Y_;
  size_t BM_min_disparity_;
  size_t BM_max_disparity_;
  size_t BM_step_;
  double BM_ZNCC_Threshold_;
  bool   BM_bUpDownConfiguration_;

  // SGM parameters (Used by Initialization)
  int num_disparities_;
  int block_size_;
  int P1_;
  int P2_;
  int uniqueness_ratio_;
  cv::Ptr<cv::StereoSGBM> sgbm_;

  /**********************************************************/
  /******************** For test & debug ********************/
  /**********************************************************/
  image_transport::Publisher invDepthMap_pub_, stdVarMap_pub_, ageMap_pub_, costMap_pub_;

  // For counting the total number of fusion
  size_t TotalNumFusion_;
};
}

#endif //ESVO_CORE_MAPPING_H

#define _CRT_SECURE_NO_WARNINGS
#define SPDLOG_HEADER_ONLY
#define FMT_HEADER_ONLY

#define CURL_STATICLIB
#pragma comment(lib, "libcurld.lib")
#pragma comment(lib, "wldap32.lib")
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "Crypt32.lib")


#include <spdlog/spdlog.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <curl/curl.h>

//#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <depthai/depthai.hpp>

#include <iostream>
#include <sstream>
#include <fstream>

#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

#include <memory>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <string>
#include <map>
#include <chrono>
#include <limits>

#include <Eigen/Dense>
#include <Eigen/Geometry>

extern "C" {
#include "apriltag/apriltag.h"
#include "apriltag/tag36h11.h"
#include "apriltag/tagStandard41h12.h"
#include "apriltag/common/zarray.h"
#include "apriltag/common/image_u8.h"
}
using json = nlohmann::json;



struct PoseData {
	double x, y, z;
	double qx, qy, qz, qw;
	int marker_id; // �±� ID
	double marker_rel_x; // ī�޶� ��� x
	double marker_rel_z; // ī�޶� ��� z
	int marker_zx_change; // �±� ��ġ �� ZX �� ���� ��(0, 90, 180, 270)
	double computed_angle;
};

struct MarkerInfo {
	int use; // ��� ���� (1 ���)
	double size_m; // ��Ŀ ũ��
	double position_x; // ������ �۷ι� ���� ��ġ
	double position_y;
	double position_z;
	double axis_x; // ���� ȸ�� (roll, deg)
	double axis_y; // ���� ȸ�� (pitch, deg)
	double axis_z; // ���� ȸ�� (yaw, deg)
	int zx_change; // ��Ŀ ��ġ �� ZX �� ���� ��(90, 180, 270)
};

struct PoseEstimationOutput {
	cv::Mat annotated_image;         // �±� ���� ����� �׷��� �̹���
	std::vector<PoseData> raw_poses;   // �� �±׿� ���� solvePnP�� ������ ���� ���� (tvec, quat)
	std::vector<PoseData> final_poses; // ����/��Ȱȭ ���� �� ���� ���� (��: ���õ� �±�)
	int selected_marker_id;          // ī�޶� ��ǥ�迡�� ���� ����� �±��� ID
};

struct BasicInfoModel {
	std::string pidx;
	std::string vidx;
};





#include <zmq.hpp>
zmq::context_t context(1);
using SubscriberSocket = zmq::socket_t;

nlohmann::json cfg;
std::string markerCSVPath;
std::string backendInitURL;
std::string backendPoseURL;
int mapSize;
double scale;
cv::Point mapOrigin;
double arrowLength;
std::string frontCameraMxId;
std::string rearCameraMxId;
std::string frontCameraAddress;
std::string rearCameraAddress;
double smoothingAlpha;
double translationThreshold;

SubscriberSocket _frontFrameSocket(context, zmq::socket_type::sub);
SubscriberSocket _rearFrameSocket(context, zmq::socket_type::sub);

cv::Mat cameraMatrixFront, distCoeffsFront, cameraMatrixRear, distCoeffsRear;
apriltag_detector_t* td = nullptr;
std::map<int, MarkerInfo> markerDictPlatform;

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
	size_t totalSize = size * nmemb;
	std::string* str = static_cast<std::string*>(userp);
	str->append(static_cast<char*>(contents), totalSize);
	return totalSize;
}

// ���� �ʱ�ȭ �Լ�: config���� ���� �ּҷ� ������ �����ϰ� ��� ���� ����
SubscriberSocket InitializeSocket(const std::string& address)
{
	std::cout << "[INFO] Trying to connect to " << address << "..." << std::endl;
	SubscriberSocket socket(context, zmq::socket_type::sub);
	try {
		socket.connect(address);
		socket.set(zmq::sockopt::subscribe, "");
		std::cout << "[DEBUG] Socket connected to " << address << std::endl;
	}
	catch (const std::exception& e) {
		std::cerr << "[ERROR] InitializeSocket failed: " << e.what() << std::endl;
		// ���α׷��� �����ϰų�, ���ܸ� �ٽ� throw ���� ����
		throw;
	}
	return socket;
}

// ZMQ �޽����� �����Ͽ� ������ �����͸� cv::Mat���� ���ڵ�(�׷��̽�����)
cv::Mat receiveFrame(SubscriberSocket& socket) {
	zmq::message_t msg;
	socket.recv(msg, zmq::recv_flags::none);
	std::vector<uchar> data(static_cast<uchar*>(msg.data()),
		static_cast<uchar*>(msg.data()) + msg.size());
	cv::Mat frame = cv::imdecode(data, cv::IMREAD_GRAYSCALE);
	return frame;
}

std::queue<cv::Mat> frontFrameQueue, rearFrameQueue;
std::mutex queueMutex;
std::condition_variable frameAvailable;
std::atomic<bool> running(true);

void frameReceiver(SubscriberSocket& socket, std::queue<cv::Mat>& frameQueue) {
	while (running) {
		cv::Mat frame = receiveFrame(socket);
		if (!frame.empty()) {
			std::lock_guard<std::mutex> lock(queueMutex);
			frameQueue.push(frame);
			frameAvailable.notify_one();
		}
	}
}


// �鿣�忡�� �⺻ ������ �޾ƿ��� �Լ� (���� ���)
BasicInfoModel getBasicInfoFromBackend(const std::string& url) {
	BasicInfoModel info;
	CURL* curl = curl_easy_init();
	if (curl) {
		std::string responseStr;
		curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
		curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);  // ȣ��Ʈ ���� ��Ȱ��ȭ

		curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
		curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseStr);

		CURLcode res = curl_easy_perform(curl);
		if (res != CURLE_OK) {
			std::cerr << "GET failed: " << curl_easy_strerror(res) << std::endl;
		}
		else {
			try {
				// ���� JSON �Ľ� (��: m_basicInfoModel.data[0].pidx, vidx �� �ִٰ� ����)
				nlohmann::json j = nlohmann::json::parse(responseStr);
				// JSON ������ ���� �鿣�� ���信 ���� ���� �ʿ�
				if (j.contains("data") && j["data"].is_array() && !j["data"].empty()) {
					info.pidx = std::to_string(j["data"][0].value("pidx", 0));  // ���ڸ� ���ڿ��� ��ȯ
					info.vidx = std::to_string(j["data"][0].value("vidx", 0));  // ���ڸ� ���ڿ��� ��ȯ
				}
			}
			catch (std::exception& e) {
				std::cerr << "JSON faild: " << e.what() << std::endl;
			}
		}
		curl_easy_cleanup(curl);
	}
	return info;
}

std::pair<long, std::string> sendGlobalPoseToBackend(const PoseData& pose,
	int resultForBackend, const BasicInfoModel& basicInfo, double final_angle) {
	json j;

	j["mapId"] = "5544cb62bf6b4f9b9504fe1aacffa3fc";
	j["workLocationId"] = "WATA_KGWANGJU_1F";
	j["pidx"] = basicInfo.pidx;
	j["vidx"] = basicInfo.vidx;
	j["vehicleId"] = "WTA_FORKLIFT_001";
	j["x"] = static_cast<long>(pose.x) * 1000.0;
	j["y"] = static_cast<long>(pose.z) * 1000.0;
	j["t"] = final_angle;
	j["rotate"] = 1; // ������
	j["height"] = 0;
	j["move"] = 1; // �̵�����
	j["load"] = 0; // �������
	j["action"] = 0;
	j["result"] = resultForBackend; // nav��������
	j["loadId"] = "";
	j["epc"] = "DP";
	j["errorCode"] = "0000";

	std::string jsonStr = j.dump(); // JSON object -> string

	CURL* curl = curl_easy_init(); // libcurl �ʱ�ȭ �� POST ����
	long responseCode = 0;
	std::string responseStr;

	if (curl) {
		CURLcode res;
		struct curl_slist* headers = nullptr;
		headers = curl_slist_append(headers, "Content-Type: application/json");

		curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
		curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);  // ȣ��Ʈ ���� ��Ȱ��ȭ

		curl_easy_setopt(curl, CURLOPT_URL, backendPoseURL.c_str());
		curl_easy_setopt(curl, CURLOPT_POST, 1L);
		// ASCII ���ڵ��� �����ϰ� ��������, �⺻������ C++���� ���ڿ��� ASCII (�Ǵ� UTF-8)�� �Ǿ������Ƿ� �״�� ���
		curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonStr.c_str());
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
		curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 30000L);
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseStr);

		res = curl_easy_perform(curl);
		if (res != CURLE_OK) {
			std::cerr << "POST failed: " << curl_easy_strerror(res) << std::endl;
		}
		else {
			curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &responseCode);
			std::cout << "POST success!" << std::endl;
		}
		curl_slist_free_all(headers);
		curl_easy_cleanup(curl);
	}
	return { responseCode, responseStr };
}

std::map<int, MarkerInfo> loadMarkerInfoFromCSV(const std::string& filename) {
	std::map<int, MarkerInfo> markerDict;
	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Failed to open " << filename << std::endl;
		return markerDict;
	}

	std::string line;
	if (!std::getline(file, line)) {
		std::cerr << "CSV ������ ����ֽ��ϴ�." << std::endl;
		return markerDict;
	}

	while (std::getline(file, line)) {
		if (line.empty()) continue;
		std::stringstream ss(line);
		std::vector<std::string> tokens;
		std::string token;
		while (std::getline(ss, token, ',')) {
			tokens.push_back(token);
		}
		if (tokens.size() < 6) { continue; }

		try {
			int vision_poi_id = std::stoi(tokens[3]);
			double vision_size = std::stod(tokens[4]);
			double vision_rotation = std::stod(tokens[5]);
			double xcoord = std::stod(tokens[8]);
			double ycoord = std::stod(tokens[9]);

			MarkerInfo info;
			info.use = 1;
			info.size_m = vision_size;
			info.position_x = (xcoord - 14160189.210642364);
			info.position_y = 0.0;
			info.position_z = (ycoord - 4491954.475227742);
			info.axis_x = 0.0;
			info.axis_y = 0.0;
			info.axis_z = vision_rotation;
			//if (vision_rotation == 90) {
			//	info.axis_x = 0.0;
			//	info.axis_y = 90.0;
			//	info.axis_z = 0.0;
			//}
			//else if (vision_rotation == 180) {
			//	info.axis_x = 180.0;
			//	info.axis_y = 0.0;
			//	info.axis_z = 0.0;
			//}
			//else if (vision_rotation == 270) {
			//	info.axis_x = 0.0;
			//	info.axis_y = 90.0;
			//	info.axis_z = 0.0;
			//}
			//else if (vision_rotation == 360) {
			//	info.axis_x = 0.0;
			//	info.axis_y = 0.0;
			//	info.axis_z = 0.0;
			//}

			info.zx_change = static_cast<int>(vision_rotation);

			markerDict[vision_poi_id] = info;

			std::cout << "Marker ID: " << vision_poi_id << ", Size: " << vision_size
				<< ", Position: (" << xcoord << ", " << ycoord << "), Rotation: " << vision_rotation << std::endl;
		}
		catch (...) {
			// ��ȯ ���� �� ����
			continue;
		}
	}

	file.close();
	return markerDict;
}

// cv::Mat �� apriltag�� image_u8_t �� ��ȯ
image_u8_t* matToImageU8(const cv::Mat& mat) {
	image_u8_t* im = (image_u8_t*)calloc(1, sizeof(image_u8_t));
	if (!im) {
		std::cerr << "Failed to allocate image_u8_t" << std::endl;
		return nullptr;
	}
	im->width = mat.cols;
	im->height = mat.rows;
	im->stride = mat.cols;
	im->buf = const_cast<uint8_t*>(mat.data);
	return im;
}

// ȸ������� ��ȿ�� �˻�
bool isRotationMatrix(const cv::Mat& R) {
	cv::Mat Rt;
	cv::transpose(R, Rt);
	cv::Mat shouldBeIdentity = Rt * R;
	cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());
	return cv::norm(I - shouldBeIdentity) < 1e-2;
}

// ȸ������� Euler ������ ��ȯ (x,y,z ����)
cv::Vec3d rotationMatrixToEulerAngles(const cv::Mat& R) {
	CV_Assert(isRotationMatrix(R));

	double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));
	bool singular = sy < 1e-6; // ���������� ������ ������ Ȯ��
	double x, y, z;
	if (!singular) {
		x = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
		y = std::atan2(-R.at<double>(2, 0), sy);
		z = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
	}
	else {
		x = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
		y = std::atan2(-R.at<double>(2, 0), sy);
		z = 0;
	}
	return cv::Vec3d(x, y, z);
}

// Euler ���� ���ʹϾ����� ��ȯ
cv::Vec4d getQuaternionFromEuler(double roll, double pitch, double yaw) {
	double cy = cos(yaw * 0.5), sy = sin(yaw * 0.5);
	double cp = cos(pitch * 0.5), sp = sin(pitch * 0.5);
	double cr = cos(roll * 0.5), sr = sin(roll * 0.5);
	cv::Vec4d q;
	q[0] = sr * cp * cy - cr * sp * sy; // x
	q[1] = cr * sp * cy + sr * cp * sy; // y
	q[2] = cr * cp * sy - sr * sp * cy; // z
	q[3] = cr * cp * cy + sr * sp * sy; // w
	double norm = cv::norm(q);
	return q / norm;
}

// �±��� 3D ���� ���� (solvePnP ���)
void my_estimatePoseSingleMarkers(const std::vector<cv::Point2f>& corners,
	double marker_size,
	const cv::Mat& cameraMatrix,
	const cv::Mat& distCoeffs,
	cv::Vec3d& rvec,
	cv::Vec3d& tvec,
	cv::Vec4d& quat)
{
	std::vector<cv::Point3f> objectPoints = {
		cv::Point3f(static_cast<float>(-marker_size / 2), static_cast<float>(marker_size / 2), 0),
		cv::Point3f(static_cast<float>(marker_size / 2), static_cast<float>(marker_size / 2), 0),
		cv::Point3f(static_cast<float>(marker_size / 2), static_cast<float>(-marker_size / 2), 0),
		cv::Point3f(static_cast<float>(-marker_size / 2), static_cast<float>(-marker_size / 2), 0)
	};
	cv::solvePnP(objectPoints, corners, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
	cv::Mat R;
	cv::Rodrigues(rvec, R);
	cv::Vec3d euler = rotationMatrixToEulerAngles(R);
	quat = getQuaternionFromEuler(euler[0], euler[1], euler[2]);
}

double getYawFromQuaternion(const cv::Vec4d& quat) {
	double siny_cosp = 2 * (quat[3] * quat[2] + quat[0] * quat[1]);
	double cosy_cosp = 1 - 2 * (quat[1] * quat[1] + quat[2] * quat[2]);
	return std::atan2(siny_cosp, cosy_cosp);
}


double getHeadingFromQuaternionY(const cv::Vec4d& quat) {
	// y�� ���� ���
	double siny = 2.0 * (quat[3] * quat[1] - quat[0] * quat[2]);
	double cosy = 1.0 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2]);
	return std::atan2(siny, cosy);
}

double calcDistance(const PoseData& p1, const PoseData& p2) {
	double dx = p1.x - p2.x;
	double dy = p1.y - p2.y;
	double dz = p1.z - p2.z;
	return std::sqrt(dx * dx + dy * dy + dz * dz);
}

PoseEstimationOutput poseEstimationFunction(
	int camera_id,
	const cv::Mat& gray_frame,
	const cv::Mat& cameraMatrix,
	const cv::Mat& distCoeffs,
	const std::map<int, MarkerInfo>& markerDictPlatform,
	bool use_real_position,
	apriltag_detector_t* td
) {
	PoseEstimationOutput output;
	cv::Mat grayClone = gray_frame.clone();
	cv::cvtColor(grayClone, output.annotated_image, cv::COLOR_GRAY2BGR);
	output.selected_marker_id = 0;
	double min_distance = (std::numeric_limits<double>::max)();

	// �̹����� apriltag �Է¿� image_u8_t�� ��ȯ
	image_u8_t* im = matToImageU8(gray_frame);
	if (!im) {
		std::cerr << "Failed to convert image to image_u8_t" << std::endl;
		return output;
	}

	zarray_t* detections = apriltag_detector_detect(td, im);
	int numDetections = zarray_size(detections);
	std::cout << "Camera " << camera_id << " - Detected " << numDetections << " tags." << std::endl;

	// 180�� ȸ����� (x�� ����): R_flip = diag(1, -1, -1)
	cv::Mat R_flip = cv::Mat::eye(3, 3, CV_64F);
	R_flip.at<double>(1, 1) = -1.0;
	R_flip.at<double>(2, 2) = -1.0;

	for (int i = 0; i < numDetections; i++) {
		apriltag_detection_t* det;
		zarray_get(detections, i, &det);
		int id = det->id;
		auto it = markerDictPlatform.find(id);
		if (it == markerDictPlatform.end())
			continue;
		MarkerInfo m_info = it->second;
		if (m_info.use != 1)
			continue;
		double markerSize = m_info.size_m;

		// 2D �ڳ� ���� (top-left, top-right, bottom-right, bottom-left)
		std::vector<cv::Point2f> corners;
		for (int j = 0; j < 4; j++) {
			corners.push_back(cv::Point2f(static_cast<float>(det->p[j][0]), static_cast<float>(det->p[j][1])));
			cv::line(output.annotated_image,
				cv::Point(static_cast<int>(det->p[j][0]), static_cast<int>(det->p[j][1])),
				cv::Point(static_cast<int>(det->p[(j + 1) % 4][0]), static_cast<int>(det->p[(j + 1) % 4][1])),
				cv::Scalar(0, 255, 0), 2);
		}

		// �ڼ� ����: solvePnP�� rvec, tvec, ���ʹϾ� ���
		cv::Vec3d rvec, tvec;
		cv::Vec4d quat;
		my_estimatePoseSingleMarkers(corners, markerSize, cameraMatrix, distCoeffs, rvec, tvec, quat);

		// ī�޶� ��ǥ�迡�� �±� ��ġ ���: pos_camera = -R^T * tvec
		cv::Mat R_tmp;
		cv::Rodrigues(rvec, R_tmp);
		cv::Mat R_tc = R_tmp.t();
		cv::Mat tvec_mat = (cv::Mat_<double>(3, 1) << tvec[0], tvec[1], tvec[2]);
		cv::Mat pos_camera = -R_tc * tvec_mat;

		double pos_x = pos_camera.at<double>(0, 0);
		double pos_y = pos_camera.at<double>(1, 0);
		double pos_z = pos_camera.at<double>(2, 0);



		double distance = std::sqrt(pos_x * pos_x + pos_y * pos_y + pos_z * pos_z);
		if (distance < min_distance) {
			min_distance = distance;
			output.selected_marker_id = id;
		}
		std::cout << "Camera " << camera_id << ", Marker " << id << ", pos_camera: ("
			<< pos_x << ", " << pos_y << ", " << pos_z << ")" << std::endl;

		std::cout << "Camera " << camera_id << ", Marker " << id << ", pos_camera: ("
			<< pos_x << ", " << pos_y << ", " << pos_z << ")" << std::endl;

		// ���� ������ ���� R_corr ���
		cv::Mat R_corr = cv::Mat::eye(3, 3, CV_64F);
		cv::Mat R_x180 = (cv::Mat_<double>(3, 3) <<
			-1, 0, 0,
			0, 1, 0,
			0, 0, 1);
		cv::Mat R_z180 = (cv::Mat_<double>(3, 3) <<
			1, 0, 0,
			0, 1, 0,
			0, 0, -1);



		if (m_info.zx_change == 0) {
			double theta = -CV_PI / 2;  // -90�� Y in radians
			R_corr = (cv::Mat_<double>(3, 3) <<
				-std::cos(theta), 0, std::sin(theta),
				0, 1, 0,
				-std::sin(theta), 0, -std::cos(theta));

			double g_angle3 = 360 - (std::atan2(pos_x, -pos_z) * 180 / CV_PI + 270);
			double angle3 = std::fmod(g_angle3, 360.0);
			if (angle3 < 0) {
				angle3 += 360.0;
			}
			std::cout << "[HEADING] angle3: " << angle3 << std::endl;

			cv::putText(output.annotated_image, "angle(0deg): " + std::to_string(angle3),
				cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

			PoseData currentPose;
			currentPose.marker_id = id;
			currentPose.marker_rel_x = pos_x;
			currentPose.marker_rel_z = pos_z;
			currentPose.marker_zx_change = m_info.zx_change;
			currentPose.computed_angle = angle3;
		}

		else if (m_info.zx_change == 90) {
			R_corr = (cv::Mat_<double>(3, 3) <<
				-std::cos(-CV_PI / 2), -std::sin(-CV_PI / 2), 0,
				std::sin(-CV_PI / 2), std::cos(-CV_PI / 2), 0,
				0, 0, -1);


			double g_angle3 = 360 - (std::atan2(pos_x, -pos_z) * 180 / CV_PI + 270 + 90);
			double angle3 = std::fmod(g_angle3, 360.0);
			if (angle3 < 0) {
				angle3 += 360.0;
			}
			std::cout << "[HEADING] angle3: " << angle3 << std::endl;

			cv::putText(output.annotated_image, "angle3(90deg): " + std::to_string(angle3),
				cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

			PoseData currentPose;
			currentPose.marker_id = id;
			currentPose.marker_rel_x = pos_x;
			currentPose.marker_rel_z = pos_z;
			currentPose.marker_zx_change = m_info.zx_change;
			currentPose.computed_angle = angle3;
		}

		else if (m_info.zx_change == 270) {
			double theta = CV_PI / 2;  // -90��
			R_corr = (cv::Mat_<double>(3, 3) <<
				std::cos(theta), -std::sin(theta), 0,
				std::sin(theta), std::cos(theta), 0,
				0, 0, 1);

			double g_angle3 = 360 - (std::atan2(pos_x, -pos_z) * 180 / CV_PI + 270 + 270);
			double angle3 = std::fmod(g_angle3, 360.0);
			if (angle3 < 0) {
				angle3 += 360.0;
			}
			std::cout << "[HEADING] angle3: " << angle3 << std::endl;

			cv::putText(output.annotated_image, "angle(270deg): " + std::to_string(angle3),
				cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

			PoseData currentPose;
			currentPose.marker_id = id;
			currentPose.marker_rel_x = pos_x;
			currentPose.marker_rel_z = pos_z;
			currentPose.marker_zx_change = m_info.zx_change;
			currentPose.computed_angle = angle3;
		}
		else if (m_info.zx_change == 180) {
			double theta = CV_PI / 2;  // -90�� in radians
			R_corr = (cv::Mat_<double>(3, 3) <<
				std::cos(theta), 0, std::sin(theta),
				0, 1, 0,
				-std::sin(theta), 0, std::cos(theta));
			R_corr *= R_x180;

			double g_angle3 = 360 - (std::atan2(pos_x, -pos_z) * 180 / CV_PI + 270 + 180);
			double angle3 = std::fmod(g_angle3, 360.0);
			if (angle3 < 0) {
				angle3 += 360.0;
			}
			std::cout << "[HEADING] angle3: " << angle3 << std::endl;

			cv::putText(output.annotated_image, "angle(180deg): " + std::to_string(angle3),
				cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

			PoseData currentPose;
			currentPose.marker_id = id;
			currentPose.marker_rel_x = pos_x;
			currentPose.marker_rel_z = pos_z;
			currentPose.marker_zx_change = m_info.zx_change;
			currentPose.computed_angle = angle3;

		}

		cv::Mat R_marker;
		cv::Rodrigues(cv::Vec3d(m_info.axis_x * CV_PI / 180, m_info.axis_y * CV_PI / 180, m_info.axis_z * CV_PI / 180), R_marker);


		R_marker = R_corr * R_marker;


		cv::Mat t_marker = (cv::Mat_<double>(3, 1) << m_info.position_x, m_info.position_y, m_info.position_z);
		// 1. �±��� �۷ι� ȸ�� ���(R_marker)�� ��ġ(t_marker)

		// 2. �۷ι� ī�޶� ��ġ: t_marker + R_marker * (camera position in marker frame)
		cv::Mat global_cam_pos = t_marker + R_marker * pos_camera;

		// 3. ī�޶� �۷ι� ����: 
//    - ī�޶��� ��� ȸ��: R_cam = R_tc (�±� ��ǥ�迡���� ī�޶� ȸ��)
//    - �±��� �۷ι� ȸ��: R_marker (�̹� ����)
//    => R_global_cam = R_marker * R_cam
		cv::Mat R_global_cam = R_marker * R_tc;
		cv::Vec3d euler_global = rotationMatrixToEulerAngles(R_global_cam);
		cv::Vec4d quat_global = getQuaternionFromEuler(euler_global[0], euler_global[1], euler_global[2]);



		// ���� ī�޶� �۷ι� ��� ����
		PoseData finalPose;
		finalPose.x = global_cam_pos.at<double>(0, 0);
		finalPose.y = global_cam_pos.at<double>(1, 0);
		finalPose.z = global_cam_pos.at<double>(2, 0);
		finalPose.qx = quat_global[0];
		finalPose.qy = quat_global[1];
		finalPose.qz = quat_global[2];
		finalPose.qw = quat_global[3];

		double scaleFactor = 1;
		finalPose.x *= scaleFactor;
		finalPose.y *= scaleFactor;
		finalPose.z *= scaleFactor;

		static bool firstFrame = true;
		static PoseData prevGlobalPose;
		double alpha = smoothingAlpha; // smoothing factor (0.0�̸� ���, 1.0�̸� ���� �� ����)

		// outlier ������ ���� �Ӱ谪 (��: 0.5 ���� �̻� ��ȭ�ϸ� �̻�ġ�� ����)
		double effectiveAlpha = alpha;

		PoseData smoothedPose;
		if (firstFrame) {
			smoothedPose = finalPose;
			prevGlobalPose = finalPose;
			firstFrame = false;
		}
		else {
			// ���� ������� ��ȭ�� ���
			double diff = calcDistance(finalPose, prevGlobalPose);
			// ��ȭ���� �Ӱ谪���� ũ�� alpha�� ���߾� ��ȭ���� ������ ����
			if (diff > translationThreshold) {
				effectiveAlpha = alpha * 0.5;
			}
			// translation smoothing (EMA)
			smoothedPose.x = effectiveAlpha * finalPose.x + (1 - effectiveAlpha) * prevGlobalPose.x;
			smoothedPose.y = effectiveAlpha * finalPose.y + (1 - effectiveAlpha) * prevGlobalPose.y;
			smoothedPose.z = effectiveAlpha * finalPose.z + (1 - effectiveAlpha) * prevGlobalPose.z;

			// orientation smoothing: SLERP (Eigen �̿�)
			Eigen::Quaterniond q_prev(prevGlobalPose.qw, prevGlobalPose.qx, prevGlobalPose.qy, prevGlobalPose.qz);
			Eigen::Quaterniond q_curr(finalPose.qw, finalPose.qx, finalPose.qy, finalPose.qz);
			Eigen::Quaterniond q_smoothed = q_prev.slerp(effectiveAlpha, q_curr);
			smoothedPose.qw = q_smoothed.w();
			smoothedPose.qx = q_smoothed.x();
			smoothedPose.qy = q_smoothed.y();
			smoothedPose.qz = q_smoothed.z();

			// ������Ʈ: �̹� �������� smoothed ���� ���� �����ӿ� ����� ���� ������ ����
			prevGlobalPose = smoothedPose;
		}

		// ���������� ���õ� �±�(���� ����� �±�)���� ���� �۷ι� ����� smoothedPose�� ���
		if (id == output.selected_marker_id) {
			output.final_poses.push_back(smoothedPose);
		}

		// ���� Pose: tvec, quat (�򰡿�)
		PoseData rawPose;
		rawPose.x = tvec[0];
		rawPose.y = tvec[1];
		rawPose.z = tvec[2];
		rawPose.qx = quat[0];
		rawPose.qy = quat[1];
		rawPose.qz = quat[2];
		rawPose.qw = quat[3];
		output.raw_poses.push_back(rawPose);

		cv::drawFrameAxes(output.annotated_image, cameraMatrix, distCoeffs, rvec, tvec, float(m_info.size_m * 0.5));
		cv::putText(output.annotated_image, std::to_string(id),
			cv::Point(static_cast<int>(det->c[0]), static_cast<int>(det->c[1])),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
	}

	apriltag_detections_destroy(detections);
	free(im);


	std::ostringstream oss;
	oss << "Final poses for camera " << camera_id << ": " << std::endl;
	for (const auto& pose : output.final_poses) {
		oss << "\nPose: (" << pose.x << ", " << pose.y << ", " << pose.z << std::endl;
	}
	cv::putText(output.annotated_image, oss.str(), cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
	return output;
}

cv::Mat create2DMap(const std::vector<PoseData>& markerPoses,
	const PoseData& cameraPose,
	bool isRearPose,
	bool drawCamera,
	int mapSize, double scale,
	const cv::Point& mapOrigin, double arrowLength = 20.0) {

	cv::Mat mapImage = cv::Mat::zeros(mapSize, mapSize, CV_8UC3);

	mapImage.setTo(cv::Scalar(255, 255, 255));

	cv::line(mapImage, cv::Point(0, mapOrigin.y), cv::Point(mapSize, mapOrigin.y), cv::Scalar(200, 200, 200), 1);
	cv::line(mapImage, cv::Point(mapOrigin.x, 0), cv::Point(mapOrigin.x, mapSize), cv::Scalar(200, 200, 200), 1);

	// �±� ��ġ (��)
	for (const auto& pose : markerPoses) {
		int pixelX = static_cast<int>(mapOrigin.x + pose.x * scale);
		int pixelY = static_cast<int>(mapOrigin.y - pose.z * scale); // z ���� ������ ����� ��� ����
		cv::circle(mapImage, cv::Point(pixelX, pixelY), 5, cv::Scalar(0, 0, 255), -1);
		double yaw = getYawFromQuaternion(cv::Vec4d(pose.qx, pose.qy, pose.qz, pose.qw));
		int arrowEndX = static_cast<int>(pixelX + arrowLength * cos(yaw));
		int arrowEndY = static_cast<int>(pixelY - arrowLength * sin(yaw)); // y ���� ����
		// ȭ��ǥ �׸���
		cv::arrowedLine(mapImage, cv::Point(pixelX, pixelY), cv::Point(arrowEndX, arrowEndY), cv::Scalar(0, 0, 0), 2, cv::LINE_AA, 0, 0.3);
		std::ostringstream oss;
		oss << "(" << pose.x << ", " << pose.z << ")";
		cv::putText(mapImage, oss.str(), cv::Point(pixelX - 50, pixelY - 8), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0), 1);
	}

	if (drawCamera) {
		int camPx = static_cast<int>(mapOrigin.x + cameraPose.x * scale);
		int camPy = static_cast<int>(mapOrigin.y - cameraPose.z * scale);
		cv::circle(mapImage, cv::Point(camPx, camPy), 5, cv::Scalar(255, 0, 0), -1);

		double angle_deg = cameraPose.computed_angle;
		double camYaw = angle_deg * CV_PI / 180.0;


		int arrowX = static_cast<int>(camPx + arrowLength * std::cos(camYaw));
		int arrowY = static_cast<int>(camPy - arrowLength * std::sin(camYaw));
		cv::arrowedLine(mapImage, cv::Point(camPx, camPy), cv::Point(arrowX, arrowY),
			cv::Scalar(255, 0, 0), 2, cv::LINE_AA, 0, 0.3);

		cv::putText(mapImage, "Forklift", cv::Point(camPx + 10, camPy),
			cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 0, 0), 1);
	}

	return mapImage;
}




static bool hasPrevGlobalPose = false;
static PoseData prevGlobalPose = { 0, 0, 0, 0, 0, 0, 1 }; // �ʱⰪ

int main(int argc, char* argv[])
{
	std::cout << "[INFO] Starting program..." << std::endl;

	std::ifstream configFile("config/config.json");
	if (!configFile.is_open()) {
		std::cerr << "Failed to open config" << std::endl;
		return -1;
	}
	configFile >> cfg;
	configFile.close();
	std::cout << "[INFO] Config file loaded." << std::endl;


	// ���� ���� �ʱ�ȭ
	markerCSVPath = cfg.value("marker_csv_path", "marker/visionData (6).csv");
	backendInitURL = cfg["backend"].value("init_url", "");
	backendPoseURL = cfg["backend"].value("pose_url", "");
	mapSize = cfg["map"].value("size", 900);
	scale = cfg["map"].value("scale", 18.0);
	mapOrigin = cv::Point(cfg["map"].value("origin_x", mapSize / 2),
		cfg["map"].value("origin_y", mapSize / 2));
	arrowLength = cfg["map"].value("arrow_length", 20.0);
	frontCameraMxId = cfg["camera"].value("front_mxid", "19443010C180962E00");
	rearCameraMxId = cfg["camera"].value("rear_mxid", "19443010C142962E00");
	frontCameraAddress = cfg["camera"].value("front_address", "tcp://localhost:5571");
	rearCameraAddress = cfg["camera"].value("rear_address", "tcp://localhost:5580");

	smoothingAlpha = cfg["smoothing"].value("alpha", 0.15);
	translationThreshold = cfg["smoothing"].value("translation_threshold", 1.0);

	// ī�޶� ��� (tcp �Ǵ� mxid)
	std::string cameraMode = cfg["camera"].value("mode", "tcp");
	std::cout << "[INFO] Camera mode: " << cameraMode << std::endl;

	std::cout << "[INFO] Config parameters:" << std::endl;
	std::cout << "  Marker CSV: " << markerCSVPath << std::endl;
	std::cout << "  Backend Init URL: " << backendInitURL << std::endl;
	std::cout << "  Front Address: " << frontCameraAddress << std::endl;
	std::cout << "  Rear Address: " << rearCameraAddress << std::endl;


	BasicInfoModel basicInfo = getBasicInfoFromBackend(backendInitURL);
	std::cout << "[INFO] Retrieved basic info: pidx=" << basicInfo.pidx
		<< ", vidx=" << basicInfo.vidx << std::endl;

	std::map<int, MarkerInfo> markerDictPlatform = loadMarkerInfoFromCSV(markerCSVPath);
	std::cout << "[INFO] Loaded marker CSV. Total markers loaded: " << markerDictPlatform.size() << std::endl;



	// Front Camera Calibration (Camera Matrix & Distortion Coefficients)
	cv::Mat cameraMatrixFront = (cv::Mat_<double>(3, 3) <<
		567.65905762, 0.0, 639.76306152,
		0.0, 567.18048096, 412.91934204,
		0.0, 0.0, 1.0);
	cv::Mat distCoeffsFront = (cv::Mat_<double>(1, 14) <<
		4.45412493e+00, 1.27883708e+00, 4.78481670e-05, -2.38865032e-04,
		1.81006361e-02, 4.82238388e+00, 2.62705803e+00, 1.66893706e-01,
		0.0, 0.0, 0.0, 0.0, 1.21197454e-03, -4.45877854e-03);

	// Rear Camera Calibration (Camera Matrix & Distortion Coefficients)
	cv::Mat cameraMatrixRear = (cv::Mat_<double>(3, 3) <<
		571.80609131, 0.0, 630.0435791,
		0.0, 571.5758667, 400.2718811,
		0.0, 0.0, 1.0);
	cv::Mat distCoeffsRear = (cv::Mat_<double>(1, 14) <<
		3.55751157e-01, -3.65746990e-02, 8.70321674e-05, -3.60699669e-05,
		-1.43852108e-03, 6.89936757e-01, -2.05448060e-03, -9.03619174e-03,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	std::cout << "[INFO] Camera calibrations set." << std::endl;


	apriltag_family_t* tf36h11 = tag36h11_create();
	apriltag_family_t* tf41h12 = tagStandard41h12_create();

	apriltag_detector_t* td = apriltag_detector_create();
	apriltag_detector_add_family(td, tf36h11);
	apriltag_detector_add_family(td, tf41h12);
	td->quad_decimate = 1.0;
	td->quad_sigma = 0.0;
	std::cout << "[INFO] Apriltag detector initialized." << std::endl;


	bool use_real_position = true;

	// (1-1) �±׸� 2D�ʿ� ��� ���� PoseData�� ��ȯ
	std::vector<PoseData> markerPoses;
	for (auto& kv : markerDictPlatform) {
		const int mid = kv.first;
		const MarkerInfo& info = kv.second;
		if (info.use != 1) continue; // ��� �� ��

		// ȸ��(roll, pitch, yaw) = (axis_x, axis_y, axis_z) (deg->rad)
		double roll = info.axis_x * CV_PI / 180.0;
		double pitch = info.axis_y * CV_PI / 180.0;
		double yaw = info.axis_z * CV_PI / 180.0;
		cv::Vec4d mq = getQuaternionFromEuler(roll, pitch, yaw);

		PoseData mp;
		mp.x = info.position_x;
		mp.y = info.position_y; // 0�� ���� ����
		mp.z = info.position_z;
		mp.qx = mq[0];
		mp.qy = mq[1];
		mp.qz = mq[2];
		mp.qw = mq[3];

		markerPoses.push_back(mp);
	}

	// ��忡 ���� ī�޶� ���� �� ���� ���� ����
	if (cameraMode == "mxid") {
		std::cout << "[INFO] Running in MXID mode (DepthAI)." << std::endl;
		// DepthAI ���: ��ġ ������ �˻��Ͽ� front�� rear ī�޶� ã��
		auto deviceInfos = dai::Device::getAllAvailableDevices();
		if (deviceInfos.size() < 2) {
			std::cerr << "[ERROR] Not enough devices found: " << deviceInfos.size() << std::endl;
			return -1;
		}
		// config���� front�� rear�� MXID ���� �����´ٰ� ����
		std::string frontCameraMxId = cfg["camera"].value("front_mxid", "");
		std::string rearCameraMxId = cfg["camera"].value("rear_mxid", "");
		int frontCameraIndex = -1, rearCameraIndex = -1;
		for (size_t i = 0; i < deviceInfos.size(); ++i) {
			std::string mxid = deviceInfos[i].getMxId();
			if (mxid == frontCameraMxId) frontCameraIndex = i;
			if (mxid == rearCameraMxId) rearCameraIndex = i;
		}
		if (frontCameraIndex == -1 || rearCameraIndex == -1) {
			std::cerr << "[ERROR] Failed to find both front and rear cameras by MXID." << std::endl;
			return -1;
		}
		for (size_t i = 0; i < deviceInfos.size(); ++i) {
			std::string label = (i == frontCameraIndex) ? "Front" : (i == rearCameraIndex ? "Rear" : "Unknown");
			std::cout << "[INFO] Device " << i << " (" << label << "): MXID = " << deviceInfos[i].getMxId() << std::endl;
		}

		// DepthAI ���������� ���� �� ��ġ ���� (���� �ڵ�)
		dai::Pipeline pipeline1, pipeline2;
		auto mono1 = pipeline1.create<dai::node::MonoCamera>();
		mono1->setResolution(dai::MonoCameraProperties::SensorResolution::THE_720_P);
		mono1->setBoardSocket(dai::CameraBoardSocket::CAM_B);
		auto xout1 = pipeline1.create<dai::node::XLinkOut>();
		xout1->setStreamName("mono1");
		mono1->out.link(xout1->input);

		auto mono2 = pipeline2.create<dai::node::MonoCamera>();
		mono2->setResolution(dai::MonoCameraProperties::SensorResolution::THE_720_P);
		mono2->setBoardSocket(dai::CameraBoardSocket::CAM_A);
		auto xout2 = pipeline2.create<dai::node::XLinkOut>();
		xout2->setStreamName("mono2");
		mono2->out.link(xout2->input);

		// Front ī�޶� ���� (unique_ptr ��� �� �ݵ�� -> �����ڸ� ���)
		std::unique_ptr<dai::Device> frontCamera;
		try {
			int state = static_cast<int>(deviceInfos[frontCameraIndex].state);
			std::cout << "[DEBUG] Front device state: " << state << std::endl;
			frontCamera = std::make_unique<dai::Device>(pipeline1, deviceInfos[frontCameraIndex]);
			std::cout << "Front camera successfully connected." << std::endl;
		}
		catch (const std::runtime_error& e) {
			std::cerr << "Front camera connection failed: " << e.what() << std::endl;
			return -1;
		}
		std::unique_ptr<dai::Device> rearCamera;
		bool rearCameraConnected = false;
		while (!rearCameraConnected) {
			try {
				rearCamera = std::make_unique<dai::Device>(pipeline2, deviceInfos[rearCameraIndex], dai::UsbSpeed::SUPER_PLUS);
				rearCameraConnected = true;
				std::cout << "[INFO] Rear camera successfully connected." << std::endl;
			}
			catch (const std::runtime_error& e) {
				std::cerr << "[WARNING] Rear camera not ready (" << e.what() << "). Waiting and retrying..." << std::endl;
				std::this_thread::sleep_for(std::chrono::seconds(1));
				// �ʿ� ��, deviceInfos ���� ���
				// deviceInfos = dai::Device::getAllAvailableDevices();
			}
		}
		auto queue1 = frontCamera->getOutputQueue("mono1", 8, false);
		auto queue2 = rearCamera->getOutputQueue("mono2", 8, false);



		while (true) {
			std::cout << "-------------------------------------------------" << std::endl;
			std::cout << "[INFO] (MXID mode) Waiting for DepthAI frames..." << std::endl;
			auto inFrame1 = queue1->get<dai::ImgFrame>();
			auto inFrame2 = queue2->get<dai::ImgFrame>();
			cv::Mat frontFrame = cv::Mat(inFrame1->getHeight(), inFrame1->getWidth(), CV_8UC1, inFrame1->getData().data());
			cv::Mat rearFrame = cv::Mat(inFrame2->getHeight(), inFrame2->getWidth(), CV_8UC1, inFrame2->getData().data());

			// �ĸ� ī�޶� ������ 180�� ȸ��
			cv::rotate(rearFrame, rearFrame, cv::ROTATE_180);

			if (frontFrame.empty() || rearFrame.empty()) {
				std::cerr << "[ERROR] (MXID mode) Received empty frame(s), skipping iteration." << std::endl;
				continue;
			}
			else {
				std::cout << "[INFO] (MXID mode) Front frame size: " << frontFrame.cols << "x" << frontFrame.rows << std::endl;
				std::cout << "[INFO] (MXID mode) Rear frame size: " << rearFrame.cols << "x" << rearFrame.rows << std::endl;
			}
			PoseEstimationOutput outFront, outRear;
			outFront = poseEstimationFunction(1, frontFrame, cameraMatrixFront, distCoeffsFront, markerDictPlatform, use_real_position, td);
			outRear = poseEstimationFunction(2, rearFrame, cameraMatrixRear, distCoeffsRear, markerDictPlatform, use_real_position, td);



			// (2) ī�޶� ��ġ ������
			//std::vector<PoseData> allCameraPoses;
			//allCameraPoses.insert(allCameraPoses.end(), outFront.final_poses.begin(), outFront.final_poses.end());
			//allCameraPoses.insert(allCameraPoses.end(), outRear.final_poses.begin(), outRear.final_poses.end());

			size_t frontCount = outFront.final_poses.size();

			cv::Mat resizedFront, resizedRear;
			cv::resize(outFront.annotated_image, resizedFront, cv::Size(), 0.5, 0.5);
			cv::resize(outRear.annotated_image, resizedRear, cv::Size(), 0.5, 0.5);
			cv::Mat combinedFrontRear;
			cv::vconcat(resizedFront, resizedRear, combinedFrontRear);



			cv::Mat mapImage;

			bool noMarkersDetected = 0;
			PoseData chosenCameraPose;
			bool isRearPose = false;
			int resultForBackend = 1;







			if (noMarkersDetected) {
				if (hasPrevGlobalPose) {
					chosenCameraPose = prevGlobalPose;
				}
				else {
					chosenCameraPose = { 0,0,0,0,0,0,1 };
				}
				resultForBackend = 0;

				// �⺻ ��(��Ŀ��)���� mapImage�� �ٽ� ����
				mapImage = create2DMap(markerPoses, PoseData{}, false,
					/*drawCamera=*/false, mapSize, scale, mapOrigin, arrowLength);

				std::pair<long, std::string> backendResponse = sendGlobalPoseToBackend(chosenCameraPose, resultForBackend, basicInfo, final_angle);
				// Combined �信 �鿣�� ���� �α� ǥ�� (��: �� ���ʿ� ǥ��)
				std::ostringstream backendOss;
				backendOss << backendResponse.second;
				cv::putText(combinedFrontRear, backendOss.str(), cv::Point(10, 30),
					cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1);
				std::cout << "Res: " << backendResponse.second << std::endl;


			}
			else {
				// 1. �� ī�޶��� raw pose���� �±׿��� ��� �Ÿ��� ���ϰ�, �� �ε����� �����մϴ�.
				double frontLocalMinDist = std::numeric_limits<double>::max();
				size_t bestFrontIndex = 0;
				for (size_t i_f = 0; i_f < outFront.raw_poses.size(); i_f++) {
					const auto& pose = outFront.raw_poses[i_f];
					double d_f = std::sqrt(pose.x * pose.x + pose.y * pose.y + pose.z * pose.z);
					std::cout << "[DEBUG] (MXID mode) Front raw pose[" << i_f << "] distance: " << d_f << std::endl;
					if (d_f < frontLocalMinDist) {
						frontLocalMinDist = d_f;
						bestFrontIndex = i_f;
					}
				}

				double rearLocalMinDist = std::numeric_limits<double>::max();
				size_t bestRearIndex = 0;
				for (size_t i_r = 0; i_r < outRear.raw_poses.size(); i_r++) {
					const auto& pose = outRear.raw_poses[i_r];
					double d_r = std::sqrt(pose.x * pose.x + pose.y * pose.y + pose.z * pose.z);
					std::cout << "[DEBUG] (MXID mode) Rear raw pose[" << i_r << "] distance: " << d_r << std::endl;
					if (d_r < rearLocalMinDist) {
						rearLocalMinDist = d_r;
						bestRearIndex = i_r;
					}
				}

				std::cout << "[DEBUG] (MXID mode) Final front raw distance: " << frontLocalMinDist
					<< ", Rear raw distance: " << rearLocalMinDist << std::endl;

				// 2. raw pose �Ÿ��� ���Ͽ�, ��� ī�޶� �±׿� �� ������� �Ǻ��մϴ�.
				// ���� ���� �� �⺻���� �־� �ʱ�ȭ�մϴ�.
				//PoseData chosenCameraPose = { 0, 0, 0, 0, 0, 0, 1 };
				//bool isRearPose = false;

				if (rearLocalMinDist < frontLocalMinDist) {
					// Rear ī�޶� �� ������, Rear ī�޶��� �۷ι� ��� �����մϴ�.

					chosenCameraPose = outRear.final_poses[bestRearIndex];
					std::cout << "rear global pos: " << chosenCameraPose.x << ", " << chosenCameraPose.y << ", " << chosenCameraPose.z << std::endl;

					std::ostringstream oss;
					oss << "Rear global pos: (" << chosenCameraPose.x << ", " << chosenCameraPose.y << ", " << chosenCameraPose.z << ")";

					cv::putText(combinedFrontRear, oss.str(), cv::Point(10, combinedFrontRear.rows - 40),
						cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);

					isRearPose = true;
					std::cout << "[INFO] (MXID mode) Rear camera result selected (closer to tag)." << std::endl;
				}
				else {
					// �׷��� ������ Front ī�޶��� �۷ι� ��� �����մϴ�.

					chosenCameraPose = outFront.final_poses[bestFrontIndex];
					std::cout << "front global pos: " << chosenCameraPose.x << ", " << chosenCameraPose.y << ", " << chosenCameraPose.z << std::endl;
					isRearPose = false;
					std::cout << "[INFO] (MXID mode) Front camera result selected." << std::endl;

					std::ostringstream oss;
					oss << "Front global pos: (" << chosenCameraPose.x << ", " << chosenCameraPose.y << ", " << chosenCameraPose.z << ")";
					cv::putText(combinedFrontRear, oss.str(), cv::Point(10, combinedFrontRear.rows - 70),
						cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
				}
				double selectedTagYaw = getYawFromQuaternion(cv::Vec4d(chosenCameraPose.qx, chosenCameraPose.qy, chosenCameraPose.qz, chosenCameraPose.qw));
				std::cout << "[INFO] Selected tag yaw: " << selectedTagYaw << " rad ("
					<< selectedTagYaw * 180.0 / CV_PI << " deg)" << std::endl;

				prevGlobalPose = chosenCameraPose;
				hasPrevGlobalPose = true;

				resultForBackend = 1;

				double computed_angle = chosenCameraPose.computed_angle;

				if (isRearPose) {
					computed_angle = std::fmod(computed_angle + 180.0, 360.0);
					if (computed_angle < 0) {
						computed_angle += 360.0;
					}
				}

				double final_angle = std::fmod(computed_angle * 10.0, 3600.0);
				if (final_angle < 0) {
					final_angle += 3600.0;
				}


				std::pair<long, std::string> backendResponse = sendGlobalPoseToBackend(chosenCameraPose, resultForBackend, basicInfo, final_angle);




				mapImage = create2DMap(markerPoses, chosenCameraPose, isRearPose, /*drawCamera=*/true,
					mapSize, scale, mapOrigin, arrowLength);

				std::ostringstream backendOss;
				backendOss << backendResponse.second;
				cv::putText(mapImage, backendOss.str(), cv::Point(10, 30),
					cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
				std::cout << "Res: " << backendResponse.second << std::endl;

				// Combined View�� 2D �� ���� ���� ũ�� ����
				if (combinedFrontRear.rows != mapImage.rows) {
					int maxRows = std::max(combinedFrontRear.rows, mapImage.rows);
					try {
						cv::copyMakeBorder(combinedFrontRear, combinedFrontRear, 0, maxRows - combinedFrontRear.rows, 0, 0,
							cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
						cv::copyMakeBorder(mapImage, mapImage, 0, maxRows - mapImage.rows, 0, 0,
							cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
					}
					catch (const cv::Exception& ex) {
						std::cerr << "[ERROR] Exception in copyMakeBorder: " << ex.what() << std::endl;
					}
				}

				cv::Mat combined;
				try {
					cv::hconcat(combinedFrontRear, mapImage, combined);
				}
				catch (const cv::Exception& ex) {
					std::cerr << "[ERROR] Exception in hconcat: " << ex.what() << std::endl;
				}

				if (noMarkersDetected) {
					cv::putText(combined, "No markers detected. Using previous pose.", cv::Point(10, combined.rows - 20),
						cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 230, 255), 1);
				}

				else {
					try {
						int baseY = combined.rows - 20;
						std::ostringstream oss;
						if (isRearPose) {
							oss << "Camera (Rear): (" << chosenCameraPose.x << ", " << chosenCameraPose.y << ", " << chosenCameraPose.z
								<< "), heading=" << computed_angle << " deg";
						}
						else {
							oss << "Camera (Front): (" << chosenCameraPose.x << ", " << chosenCameraPose.y << ", " << chosenCameraPose.z
								<< "), heading=" << computed_angle << " deg";
						}
						cv::putText(combined, oss.str(), cv::Point(10, baseY),
							cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 230, 255), 1);
					}
					catch (const std::exception& e) {
						std::cerr << "[ERROR] Exception during combined text generation: " << e.what() << std::endl;
						cv::putText(combined, "Error computing heading", cv::Point(10, combined.rows - 20),
							cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
					}
				}
				// ��� �̹��� â ���
				cv::imshow("Pose estimation Viewer", combined);

			}





			if (cv::waitKey(1) == 27) {
				std::cout << "[INFO] (MXID mode) ESC pressed. Exiting MXID loop." << std::endl;
				break;

			}
		}
	}

	else if (cameraMode == "tcp") {

		std::cout << "[INFO] Running in TCP mode." << std::endl;

		try {
			// ���� �ʱ�ȭ �κ��� �߰� ���� ó���� ���α�
			try {
				std::cout << "[DEBUG] Initializing sockets for front and rear cameras..." << std::endl;
				SubscriberSocket frontFrameSocket = InitializeSocket(frontCameraAddress);
				SubscriberSocket rearFrameSocket = InitializeSocket(rearCameraAddress);
				std::cout << "[INFO] Sockets initialized." << std::endl;

				std::thread frontReceiverThread(frameReceiver, std::ref(frontFrameSocket), std::ref(frontFrameQueue));
				std::thread rearReceiverThread(frameReceiver, std::ref(rearFrameSocket), std::ref(rearFrameQueue));

				auto prev_tcp_time = std::chrono::steady_clock::now();

				while (running) {
					std::cout << "-------------------------------------------------" << std::endl;
					std::cout << "[INFO] (TCP mode) Waiting for frames..." << std::endl;

					std::unique_lock<std::mutex> lock(queueMutex);
					frameAvailable.wait(lock, [] { return !frontFrameQueue.empty() && !rearFrameQueue.empty(); });


					cv::Mat frontFrame = frontFrameQueue.front();
					frontFrameQueue.pop();
					cv::Mat rearFrame = rearFrameQueue.front();
					rearFrameQueue.pop();
					lock.unlock();

					cv::rotate(rearFrame, rearFrame, cv::ROTATE_180);

					PoseEstimationOutput outFront, outRear;
					try {
						std::cout << "[INFO] (TCP mode) Starting pose estimation for front camera..." << std::endl;
						outFront = poseEstimationFunction(1, frontFrame, cameraMatrixFront, distCoeffsFront, markerDictPlatform, use_real_position, td);
						std::cout << "[INFO] (TCP mode) Front camera: Detected " << outFront.raw_poses.size()
							<< " raw poses, selected marker: " << outFront.selected_marker_id << std::endl;
					}
					catch (const std::exception& e) {
						std::cerr << "[ERROR] (TCP mode) Exception during front pose estimation: " << e.what() << std::endl;
						continue;
					}
					try {
						std::cout << "[INFO] (TCP mode) Starting pose estimation for rear camera..." << std::endl;
						outRear = poseEstimationFunction(2, rearFrame, cameraMatrixRear, distCoeffsRear, markerDictPlatform, use_real_position, td);
						std::cout << "[INFO] (TCP mode) Rear camera: Detected " << outRear.raw_poses.size()
							<< " raw poses, selected marker: " << outRear.selected_marker_id << std::endl;
					}
					catch (const std::exception& e) {
						std::cerr << "[ERROR] (TCP mode) Exception during rear pose estimation: " << e.what() << std::endl;
						continue;
					}

					cv::Mat baseMap = create2DMap(markerPoses, PoseData{}, false, false,
						mapSize, scale, mapOrigin, arrowLength);

					// �ļ� ó��: ���� ����, 2D �� ����, �鿣�� ���� ��
					std::vector<PoseData> allCameraPoses;
					allCameraPoses.insert(allCameraPoses.end(), outFront.final_poses.begin(), outFront.final_poses.end());
					allCameraPoses.insert(allCameraPoses.end(), outRear.final_poses.begin(), outRear.final_poses.end());

					size_t frontCount = outFront.final_poses.size();

					cv::Mat resizedFront, resizedRear;
					cv::resize(outFront.annotated_image, resizedFront, cv::Size(), 0.5, 0.5);
					cv::resize(outRear.annotated_image, resizedRear, cv::Size(), 0.5, 0.5);
					cv::Mat combinedFrontRear;
					cv::vconcat(resizedFront, resizedRear, combinedFrontRear);

					// 2D �� ���� �� �鿣�� ���� (���� ������ ���� �ڵ带 ����)
					cv::Mat mapImage = baseMap.clone();

					bool noMarkersDetected = allCameraPoses.empty();
					PoseData chosenCameraPose = { 0, 0, 0, 0, 0, 0, 1 };
					bool isRearPose = false;
					int resultForBackend = 1;

					// �����: combinedFrontRear�� mapImage ���� Ȯ��
					std::cout << "[DEBUG] (TCP mode) combinedFrontRear size: " << combinedFrontRear.size()
						<< ", empty: " << combinedFrontRear.empty() << std::endl;
					std::cout << "[DEBUG] (TCP mode) mapImage size: " << mapImage.size()
						<< ", empty: " << mapImage.empty() << std::endl;

					// Combined View�� 2D �� ���� ���� ũ�� ����
					if (combinedFrontRear.rows != mapImage.rows) {
						int maxRows = std::max(combinedFrontRear.rows, mapImage.rows);
						try {
							cv::copyMakeBorder(combinedFrontRear, combinedFrontRear, 0, maxRows - combinedFrontRear.rows, 0, 0,
								cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
							cv::copyMakeBorder(mapImage, mapImage, 0, maxRows - mapImage.rows, 0, 0,
								cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
						}
						catch (const cv::Exception& ex) {
							std::cerr << "[ERROR] (TCP mode) Exception in copyMakeBorder: " << ex.what() << std::endl;
						}
					}

					cv::Mat combined;
					try {
						cv::hconcat(combinedFrontRear, mapImage, combined);
					}
					catch (const cv::Exception& ex) {
						std::cerr << "[ERROR] (TCP mode) Exception in hconcat: " << ex.what() << std::endl;
					}

					if (noMarkersDetected) {
						std::cout << "[WARNING] (TCP mode) No markers detected, using previous pose if available." << std::endl;
						if (hasPrevGlobalPose) {
							chosenCameraPose = prevGlobalPose;
						}
						else {
							chosenCameraPose = { 0, 0, 0, 0, 0, 0, 1 };
							cv::putText(combined, "No markers detected. Using previous pose.", cv::Point(10, combined.rows - 80),
								cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 230, 255), 1);
						}
						resultForBackend = 0;

					}
					else {
						// �� ī�޶��� raw pose���� �±׿��� ��� �Ÿ��� ���ϰ� �ε����� ����
						double frontLocalMinDist = std::numeric_limits<double>::max();
						size_t bestFrontIndex = 0;
						for (size_t i = 0; i < outFront.raw_poses.size(); i++) {
							const auto& pose = outFront.raw_poses[i];
							double d = std::sqrt(pose.x * pose.x + pose.y * pose.y + pose.z * pose.z);
							std::cout << "[DEBUG] (TCP mode) Front raw pose[" << i << "] distance: " << d << std::endl;
							if (d < frontLocalMinDist) {
								frontLocalMinDist = d;
								bestFrontIndex = i;
							}
						}

						double rearLocalMinDist = std::numeric_limits<double>::max();
						size_t bestRearIndex = 0;
						for (size_t i = 0; i < outRear.raw_poses.size(); i++) {
							const auto& pose = outRear.raw_poses[i];
							double d = std::sqrt(pose.x * pose.x + pose.y * pose.y + pose.z * pose.z);
							std::cout << "[DEBUG] (TCP mode) Rear raw pose[" << i << "] distance: " << d << std::endl;
							if (d < rearLocalMinDist) {
								rearLocalMinDist = d;
								bestRearIndex = i;
							}
						}

						std::cout << "[DEBUG] (TCP mode) Final front raw distance: " << frontLocalMinDist
							<< ", Rear raw distance: " << rearLocalMinDist << std::endl;

						if (rearLocalMinDist < frontLocalMinDist) {
							if (!outRear.final_poses.empty() && bestRearIndex < outRear.final_poses.size()) {
								chosenCameraPose = outRear.final_poses[bestRearIndex];
							}
							isRearPose = true;
							std::cout << "[INFO] (TCP mode) Rear camera result selected (closer to tag)." << std::endl;
						}
						else {
							if (!outFront.final_poses.empty() && bestFrontIndex < outFront.final_poses.size()) {
								chosenCameraPose = outFront.final_poses[bestFrontIndex];
							}
							isRearPose = false;
							std::cout << "[INFO] (TCP mode) Front camera result selected." << std::endl;
						}

						prevGlobalPose = chosenCameraPose;
						hasPrevGlobalPose = true;
						mapImage = create2DMap(markerPoses, chosenCameraPose, isRearPose, /*drawCamera=*/true,
							mapSize, scale, mapOrigin, arrowLength);
						resultForBackend = 1;
					}
					std::pair<long, std::string> backendResponse = sendGlobalPoseToBackend(chosenCameraPose, resultForBackend, basicInfo);
					std::ostringstream backendOss;
					backendOss << backendResponse.second;
					cv::putText(combinedFrontRear, backendOss.str(), cv::Point(10, 30),
						cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1);
					std::cout << "[INFO] (TCP mode) Backend response: " << backendResponse.second << std::endl;


					try {
						int baseY = combined.rows - 20;
						std::ostringstream oss;
						if (isRearPose) {
							double yaw = getHeadingFromQuaternionY(cv::Vec4d(chosenCameraPose.qx, chosenCameraPose.qy, chosenCameraPose.qz, chosenCameraPose.qw)) + CV_PI;
							if (yaw > CV_PI) yaw -= 2 * CV_PI;
							double yawDeg = yaw * 180.0 / CV_PI;
							oss << "Camera (Rear): (" << chosenCameraPose.x << ", " << chosenCameraPose.y << ", " << chosenCameraPose.z
								<< "), heading=" << yawDeg << " deg";
						}
						else {
							double yawDeg = getHeadingFromQuaternionY(cv::Vec4d(chosenCameraPose.qx, chosenCameraPose.qy, chosenCameraPose.qz, chosenCameraPose.qw)) * 180.0 / CV_PI;
							oss << "Camera (Front): (" << chosenCameraPose.x << ", " << chosenCameraPose.y << ", " << chosenCameraPose.z
								<< "), heading=" << yawDeg << " deg";
						}
						cv::putText(combined, oss.str(), cv::Point(10, baseY),
							cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 230, 255), 1);
					}
					catch (const std::exception& e) {
						std::cerr << "[ERROR] (TCP mode) Exception during combined text generation: " << e.what() << std::endl;
						cv::putText(combined, "Error computing heading", cv::Point(10, combined.rows - 80),
							cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
					}
					// ������ ����(�и�) ����: ���� �ð��� ���� �ð��� ���� ���
					auto current_tcp_time = std::chrono::steady_clock::now();
					auto delay_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_tcp_time - prev_tcp_time).count();
					prev_tcp_time = current_tcp_time;
					std::ostringstream delayOss;
					delayOss << "Frame Delay: " << delay_ms << " ms";
					cv::putText(combined, delayOss.str(), cv::Point(10, combined.rows - 50),
						cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);

					cv::imshow("Pose estimation Viewer", combined);
					if (cv::waitKey(1) == 27) {
						std::cout << "[INFO] (TCP mode) ESC pressed. Exiting TCP loop." << std::endl;
						break;
					}

				}
				frontReceiverThread.join();
				rearReceiverThread.join();
			}
			catch (const std::exception& e) {
				std::cerr << "[ERROR] Socket initialization error: " << e.what() << std::endl;
				return -1; // Ȥ�� ������ ����
			}

		}

		catch (const std::exception& e) {
			// ���� �ʱ�ȭ�� �� ���Ŀ��� �߻��ϴ� ��� ���ܸ� ó��
			std::cerr << "[ERROR] (TCP mode) Fatal error: " << e.what() << std::endl;
		}

	}
	else {
		std::cerr << "[ERROR] Unknown camera mode: " << cameraMode << std::endl;
		return -1;
	}

	std::cout << "[INFO] Cleaning up resources..." << std::endl;
	apriltag_detector_destroy(td);
	tag36h11_destroy(tf36h11);
	tagStandard41h12_destroy(tf41h12);
	std::cout << "[INFO] Program terminated." << std::endl;
	return 0;
}

#include <depthai/depthai.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

//int main() {
//    // config 또는 설정에서 가져온 전면/후면 MxId (예시)
//    std::string frontMxId = "19443010C180962E00";
//    std::string rearMxId = "19443010C142962E00";
//
//    // 모든 장치 정보 검색
//    auto deviceInfos = dai::Device::getAllAvailableDevices();
//    if (deviceInfos.empty()) {
//        std::cerr << "No DepthAI devices found." << std::endl;
//        return -1;
//    }
//
//    int frontIndex = -1;
//    int rearIndex = -1;
//    for (size_t i = 0; i < deviceInfos.size(); i++) {
//        std::string mxid = deviceInfos[i].getMxId();
//        if (mxid == frontMxId) {
//            frontIndex = i;
//        }
//        else if (mxid == rearMxId) {
//            rearIndex = i;
//        }
//    }
//
//    if (frontIndex == -1 || rearIndex == -1) {
//        std::cerr << "Could not find both front and rear cameras." << std::endl;
//        return -1;
//    }
//
//    // 두 파이프라인을 생성 (캘리브레이션 데이터 획득용 최소 구성)
//    dai::Pipeline pipelineFront, pipelineRear;
//
//    // 전면 카메라용 MonoCamera 노드 생성 (CAM_B 사용)
//    auto monoFront = pipelineFront.create<dai::node::MonoCamera>();
//    monoFront->setBoardSocket(dai::CameraBoardSocket::CAM_B);
//    auto xoutFront = pipelineFront.create<dai::node::XLinkOut>();
//    xoutFront->setStreamName("monoFront");
//    monoFront->out.link(xoutFront->input);
//
//    // 후면 카메라용 MonoCamera 노드 생성 (CAM_A 사용)
//    auto monoRear = pipelineRear.create<dai::node::MonoCamera>();
//    monoRear->setBoardSocket(dai::CameraBoardSocket::CAM_A);
//    auto xoutRear = pipelineRear.create<dai::node::XLinkOut>();
//    xoutRear->setStreamName("monoRear");
//    monoRear->out.link(xoutRear->input);
//
//    try {
//        // 전면/후면 장치를 각각 엽니다.
//        dai::Device frontCamera(pipelineFront, deviceInfos[frontIndex]);
//        dai::Device rearCamera(pipelineRear, deviceInfos[rearIndex], dai::UsbSpeed::SUPER_PLUS);
//
//        // 전면 캘리브레이션 데이터 획득 (CAM_B 사용)
//        auto calibHandlerFront = frontCamera.readFactoryCalibration();
//        std::vector<std::vector<float>> intrinsicsVecFront = calibHandlerFront.getCameraIntrinsics(dai::CameraBoardSocket::CAM_B);
//        cv::Mat cameraMatrixFront(3, 3, CV_32F);
//        for (int i = 0; i < 3; i++) {
//            for (int j = 0; j < 3; j++) {
//                cameraMatrixFront.at<float>(i, j) = intrinsicsVecFront[i][j];
//            }
//        }
//        std::vector<float> distCoeffsVecFront = calibHandlerFront.getDistortionCoefficients(dai::CameraBoardSocket::CAM_B);
//        cv::Mat distCoeffsFront(distCoeffsVecFront);
//        distCoeffsFront = distCoeffsFront.reshape(1, 1);
//
//        // 후면 캘리브레이션 데이터 획득 (CAM_A 사용)
//        auto calibHandlerRear = rearCamera.readFactoryCalibration();
//        std::vector<std::vector<float>> intrinsicsVecRear = calibHandlerRear.getCameraIntrinsics(dai::CameraBoardSocket::CAM_A);
//        cv::Mat cameraMatrixRear(3, 3, CV_32F);
//        for (int i = 0; i < 3; i++) {
//            for (int j = 0; j < 3; j++) {
//                cameraMatrixRear.at<float>(i, j) = intrinsicsVecRear[i][j];
//            }
//        }
//        std::vector<float> distCoeffsVecRear = calibHandlerRear.getDistortionCoefficients(dai::CameraBoardSocket::CAM_A);
//        cv::Mat distCoeffsRear(distCoeffsVecRear);
//        distCoeffsRear = distCoeffsRear.reshape(1, 1);
//
//        // 캘리브레이션 데이터 출력
//        std::cout << "Front Camera Matrix:" << std::endl << cameraMatrixFront << std::endl;
//        std::cout << "Front Distortion Coeffs:" << std::endl << distCoeffsFront << std::endl;
//        std::cout << "Rear Camera Matrix:" << std::endl << cameraMatrixRear << std::endl;
//        std::cout << "Rear Distortion Coeffs:" << std::endl << distCoeffsRear << std::endl;
//    }
//    catch (std::exception& e) {
//        std::cerr << "Exception caught: " << e.what() << std::endl;
//        return -1;
//    }
//
//    return 0;
//}

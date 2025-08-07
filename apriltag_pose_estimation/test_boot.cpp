#include <iostream>
#include <chrono>
#include <thread>
#include <depthai/depthai.hpp>

using namespace std;

int main() {
    // 원하는 Front 카메라의 MXID (필요에 따라 수정)
    std::string desiredFrontMxId = "19443010C180962E00";

    // 간단한 파이프라인 생성 (예: ColorCamera 프리뷰)
    dai::Pipeline pipeline;
    auto camRgb = pipeline.create<dai::node::ColorCamera>();
    camRgb->setPreviewSize(300, 300);
    camRgb->setFps(10);
    auto xoutRgb = pipeline.create<dai::node::XLinkOut>();
    xoutRgb->setStreamName("rgb");
    camRgb->preview.link(xoutRgb->input);

    while (true) {
        try {
            // 현재 연결 가능한 모든 장치 정보를 가져옴
            auto deviceInfos = dai::Device::getAllAvailableDevices();
            cout << "[DEBUG] Found " << deviceInfos.size() << " device(s)." << endl;

            int frontIndex = -1;
            // 각 장치 정보를 출력하고, 원하는 MXID를 가진 Front 카메라를 찾음
            for (size_t i = 0; i < deviceInfos.size(); i++) {
                cout << "[DEBUG] Device " << i
                    << " - Name: " << deviceInfos[i].name
                    << ", MXID: " << deviceInfos[i].getMxId()
                    << ", State: " << static_cast<int>(deviceInfos[i].state) << endl;
                if (deviceInfos[i].getMxId() == desiredFrontMxId) {
                    frontIndex = static_cast<int>(i);
                }
            }

            if (frontIndex == -1) {
                cerr << "[ERROR] Front device with MXID " << desiredFrontMxId << " not found." << endl;
            }
            else {
                // 상태 값 출력 (여기서는 BOOTED 상태가 정수 3라고 가정)
                int state = static_cast<int>(deviceInfos[frontIndex].state);
                cout << "[DEBUG] Front device state: " << state << endl;

                if (state != 3) {
                    cout << "[INFO] Front camera not booted yet (state: " << state << "). Waiting..." << endl;
                }
                else {
                    try {
                        // BOOTED 상태이면 연결 시도
                        dai::Device frontDevice(pipeline, deviceInfos[frontIndex]);
                        cout << "[INFO] Front camera connected successfully." << endl;
                        // 'rgb' 스트림에서 프레임 하나 수신
                        auto queue = frontDevice.getOutputQueue("rgb", 4, false);
                        auto inFrame = queue->get<dai::ImgFrame>();
                        cout << "[DEBUG] Received frame with dimensions: "
                            << inFrame->getWidth() << "x" << inFrame->getHeight() << endl;
                        // 연결에 성공했으므로 루프 종료
                        break;
                    }
                    catch (const std::exception& e) {
                        cerr << "[ERROR] Exception while connecting to front camera: " << e.what() << endl;
                    }
                }
            }
        }
        catch (const std::exception& e) {
            cerr << "[ERROR] Exception while polling devices: " << e.what() << endl;
        }
        // 1초 대기 후 다시 시도
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}

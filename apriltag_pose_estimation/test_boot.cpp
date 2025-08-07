#include <iostream>
#include <chrono>
#include <thread>
#include <depthai/depthai.hpp>

using namespace std;

int main() {
    // ���ϴ� Front ī�޶��� MXID (�ʿ信 ���� ����)
    std::string desiredFrontMxId = "19443010C180962E00";

    // ������ ���������� ���� (��: ColorCamera ������)
    dai::Pipeline pipeline;
    auto camRgb = pipeline.create<dai::node::ColorCamera>();
    camRgb->setPreviewSize(300, 300);
    camRgb->setFps(10);
    auto xoutRgb = pipeline.create<dai::node::XLinkOut>();
    xoutRgb->setStreamName("rgb");
    camRgb->preview.link(xoutRgb->input);

    while (true) {
        try {
            // ���� ���� ������ ��� ��ġ ������ ������
            auto deviceInfos = dai::Device::getAllAvailableDevices();
            cout << "[DEBUG] Found " << deviceInfos.size() << " device(s)." << endl;

            int frontIndex = -1;
            // �� ��ġ ������ ����ϰ�, ���ϴ� MXID�� ���� Front ī�޶� ã��
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
                // ���� �� ��� (���⼭�� BOOTED ���°� ���� 3��� ����)
                int state = static_cast<int>(deviceInfos[frontIndex].state);
                cout << "[DEBUG] Front device state: " << state << endl;

                if (state != 3) {
                    cout << "[INFO] Front camera not booted yet (state: " << state << "). Waiting..." << endl;
                }
                else {
                    try {
                        // BOOTED �����̸� ���� �õ�
                        dai::Device frontDevice(pipeline, deviceInfos[frontIndex]);
                        cout << "[INFO] Front camera connected successfully." << endl;
                        // 'rgb' ��Ʈ������ ������ �ϳ� ����
                        auto queue = frontDevice.getOutputQueue("rgb", 4, false);
                        auto inFrame = queue->get<dai::ImgFrame>();
                        cout << "[DEBUG] Received frame with dimensions: "
                            << inFrame->getWidth() << "x" << inFrame->getHeight() << endl;
                        // ���ῡ ���������Ƿ� ���� ����
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
        // 1�� ��� �� �ٽ� �õ�
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}

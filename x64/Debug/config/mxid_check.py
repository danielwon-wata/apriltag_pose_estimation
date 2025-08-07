import depthai as dai

def list_devices_mxid():
    # 1) 모든 연결된 DepthAI 디바이스 정보 가져오기
    device_infos = dai.Device.getAllAvailableDevices()
    
    if not device_infos:
        print("⚠️ DepthAI 디바이스가 연결되어 있지 않습니다.")
        return
    
    # 2) 각 디바이스의 MxID, 제품명 출력
    for idx, device_info in enumerate(device_infos):
        name          = device_info.getName()           # e.g. "OAK-D"
        product_name  = device_info.getMxIdProductName()# e.g. "DA064A"
        mxid          = device_info.getMxId()           # 바이너리 형태 MxID
        mxid_str      = device_info.getMxIdStr()        # 문자열 형태 MxID
        print(f"[{idx}] Name: {name}, Product: {product_name}")
        print(f"     ▶ MxID: {mxid} ({mxid_str})\n")

if __name__ == "__main__":
    list_devices_mxid()

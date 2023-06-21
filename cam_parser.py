import requests
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
np.set_printoptions(suppress=True)

sample = 5000

tool_marker = "11005"
reference_marker = "54320"
# url_hit = 'http://localhost:8081/MarkerRecalibration?data={"MarkerName":"'+tool_marker+'","MaxPoints":"'+ str(sample)+'"} '
# url_hit = 'http://localhost:8081/CollectStaticData?data={"ToolMarker":"'+tool_marker+'","ReferenceMarker":"'+reference_marker+'","MaximumPoint":"'+ str(sample)+'","StartDelay":"0"}'
# url_hit = 'http://localhost:8081/Trigger?data={"PlateMarkerName":"3002","ToolMarkerName":"788598",''}'




class CameraAquisition():
    def __init__(self,tool_name,reference_name):
        if isinstance(tool_name, str) and isinstance(reference_name, str):
            self.geometry = [tool_name, reference_name]
        else:
            self.geometry = []
            print("check geometry input")
    




    def parse_cam(self):
        try:
    
            url2 = 'http://127.0.0.1:8081/GetCameraData'

            r = requests.get(url2,timeout=0.1)
        
            data = r.json()
        except ConnectionError:
            print("Camera not connected")
            data = {"RegisteredMarkersList": [{"MarkerName": "4330", "ErrorStatus": "Enabled", "ErrorValue": 0.2477419376373291, "Top": {"point": {"x": 38.313575744628906, "y": 41.298828125, "z": 2485.225830078125}, "scale": {"x": 1, "y": 2, "z": 3}, "rotation": {"x": 0.8853947374318133, "y": 0.4291197113270239, "z": 0.09403737185069844, "w": 0.15195198246181438}, "Angle": {"ang": 20.47596479551467}}, "MarkerBallsList": []}, {"MarkerName": "8881", "ErrorStatus": "Enabled", "ErrorValue": 0.028032161295413967, "Top": {"point": {"x": -7.862345218658447, "y": 256.34194946289057, "z": 2251.790771484375}, "scale": {"x": 1, "y": 2, "z": 3}, "rotation": {"x": 0.9246395419190806, "y": -0.22451911629277047, "z": 0.09507565698863626, "w": -0.2925636740727827}, "Angle": {"ang": 35.840654645836345}}, "MarkerBallsList": []}]}

        except requests.exceptions.RequestException as e:  # This is the correct syntax
            # raise SystemExit(e)
            print(e)
            data = {"RegisteredMarkersList": [{"MarkerName": "4330", "ErrorStatus": "Enabled", "ErrorValue": 0.2477419376373291, "Top": {"point": {"x": 38.313575744628906, "y": 41.298828125, "z": 2485.225830078125}, "scale": {"x": 1, "y": 2, "z": 3}, "rotation": {"x": 0.8853947374318133, "y": 0.4291197113270239, "z": 0.09403737185069844, "w": 0.15195198246181438}, "Angle": {"ang": 20.47596479551467}}, "MarkerBallsList": []}, {"MarkerName": "8881", "ErrorStatus": "Enabled", "ErrorValue": 0.028032161295413967, "Top": {"point": {"x": -7.862345218658447, "y": 256.34194946289057, "z": 2251.790771484375}, "scale": {"x": 1, "y": 2, "z": 3}, "rotation": {"x": 0.9246395419190806, "y": -0.22451911629277047, "z": 0.09507565698863626, "w": -0.2925636740727827}, "Angle": {"ang": 35.840654645836345}}, "MarkerBallsList": []}]}
            print("Camera not connected")
        
        

        return data



 
    def Get_camera_quats(self,camData):
        
        RegisteredMarkerCount = 0
        data = {}
        # self.GetCurrentMarkerData()
        # print(self.camData)
        try:
            # 
            json_dict = camData
            # json_dict = json.loads(camData)
            # print(f'printing self.cam data {json_dict}')
            RegisteredMarkerCount =  len(json_dict['RegisteredMarkersList'])
        except:
            print('json error')
            print(f'printing self.cam data {json_dict}')
        

        if  RegisteredMarkerCount != 0 and bool(self.geometry): 
            for i in range(RegisteredMarkerCount):
                for Markers in self.geometry: 
                    if json_dict['RegisteredMarkersList'][i]["MarkerName"] == Markers:
                        Marker0 = {}
                        Marker0 = json_dict['RegisteredMarkersList'][i]
                        rot = Marker0['Top']['rotation']
                        pos = Marker0['Top']['point']
                        err_fre = Marker0['ErrorValue']
                        position = [pos['x'],pos['y'],pos['z'] ] 
                        quat = [ rot['x'],rot['y'],rot['z'],rot['w'] ]

                        data[Markers] = (quat,position,err_fre) 
                        

        else:
            print("Marker not visible")
            
        
        return data
    def rot2tf(self,rot,pos):
        
        pos_s = np.array(pos)
        rot_matrix = rot
        temp= np.column_stack((rot_matrix,pos_s))
        tf= np.vstack((temp,[0,0,0,1]))
        return tf

    def transform_data(self,pos,quat):
        

        marker2cam_r=R.from_quat(quat).as_matrix().transpose()
        marker2cam=self.rot2tf(marker2cam_r,pos)

        return marker2cam

def main():
    json_dict = {"RegisteredMarkersList": [{"MarkerName": "4330", "ErrorStatus": "Enabled", "ErrorValue": 0.2477419376373291, "Top": {"point": {"x": 38.313575744628906, "y": 41.298828125, "z": 2485.225830078125}, "scale": {"x": 1, "y": 2, "z": 3}, "rotation": {"x": 0.8853947374318133, "y": 0.4291197113270239, "z": 0.09403737185069844, "w": 0.15195198246181438}, "Angle": {"ang": 20.47596479551467}}, "MarkerBallsList": []}, {"MarkerName": "8881", "ErrorStatus": "Enabled", "ErrorValue": 0.028032161295413967, "Top": {"point": {"x": -7.862345218658447, "y": 256.34194946289057, "z": 2251.790771484375}, "scale": {"x": 1, "y": 2, "z": 3}, "rotation": {"x": 0.9246395419190806, "y": -0.22451911629277047, "z": 0.09507565698863626, "w": -0.2925636740727827}, "Angle": {"ang": 35.840654645836345}}, "MarkerBallsList": []}]}
    
    tool_marker = "11005"
    reference_marker = "54320"
    cam = CameraAquisition(tool_marker,reference_marker)
    json_data = cam.parse_cam()
    
    
    marker_data = cam.Get_camera_quats(json_data)
    tool_data = marker_data.get(tool_marker,0)
    ref_data = marker_data.get(reference_marker,0)
    while True:
       if ref_data != 0  and tool_data !=0:
            ref2cam_tf = cam.transform_data(ref_data[1],ref_data[0])
            tool2cam_tf = cam.transform_data(tool_data[1],tool_data[0])
            print(ref2cam_tf)
        



  



if __name__ == "__main__":

    main()









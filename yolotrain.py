from roboflow import Roboflow



rf = Roboflow(api_key="9exY2pra0dF8kVHkFBXa")
project = rf.workspace("minigolf-qsazj").project("minigolf-v2")
version = project.version(1)
dataset = version.download("yolov8")

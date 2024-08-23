from ultralytics import YOLO

# Load the YOLO model
model = YOLO('models/yolov8l.pt')

results = model.predict('data/sample_data.mp4', save=True, stream=True)

#print(results[0])
#print('--------------------------------')
#for box in results[0].boxes:
#    print(box)
    
for result in results:
    print(result)
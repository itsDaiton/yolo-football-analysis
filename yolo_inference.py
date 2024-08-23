from ultralytics import YOLO

# Load the YOLO model fine-tuned with best.pt weights
model = YOLO('models/best.pt')

results = model.predict('data/sample_data.mp4', save=True, stream=True)

#print(results[0])
#print('--------------------------------')
#for box in results[0].boxes:
#    print(box)
    
for result in results:
    print(result)
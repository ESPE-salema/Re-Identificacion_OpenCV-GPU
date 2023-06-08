from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    # load a pretrained model (recommended for training)
    model = YOLO("YOLOv8/yolov8n-seg.pt")

    # Use the model
    #model.train(data="YOLOv8/persons.yaml", epochs=164, imgsz=640)  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # export the model to ONNX format
    success = model.export(format="onnx", opset=12)
    
__name__
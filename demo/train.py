from ultralytics import YOLO

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# # 消除异步性，但是会带来性能的损失
if __name__ == '__main__':
    # Load a COCO-pretrained YOLO11n model
    # model = YOLO("yolo11n.pt")
    model = YOLO(r"yolo11n-seg.pt",task='segment') # yolo11n-seg.pt segment
    # model = YOLO(r"D:/Desktop/XLWD/dataset/ultralytics-8.3.39/runs/obb/train3/weights/best.pt",task='obb')
    # model = YOLO(r"D:/SSJ/Work/ultralytics-main/yolo11n.pt",task='detect')
    # model = YOLO(r"D:/Desktop/XLWD/dataset/ultralytics-8.3.39/runs/segment/train11/weights/best.pt",task='segment')

    # # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data=r"D:/Desktop/XLWD/project/work1/image_data_gen/data/end/Line.yaml", epochs=200, batch=32,device=0,workers = 2,imgsz=640)# imgsz=320,
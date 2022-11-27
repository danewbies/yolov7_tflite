import tensorflow.lite as tflite
import cv2
import numpy as np
import random


def letterbox(im, new_shape=(320, 320), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def detect(image, interpreter, input_details, output_details, input_shape):
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)
    im = image.astype(np.float32)
    im /= 255
    interpreter.set_tensor(input_details[0]['index'], im)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def load_weights(file="./yolov7_model.tflite"):

    # Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=file)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    return interpreter, input_details, output_details, input_shape

def draw_image(ori_images,output_data, ratio, dwdh, names, colors):
    for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(output_data):
        image = ori_images[int(batch_id)]
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score),3)
        name = names[cls_id]
        color = colors[name]
        name += ' '+str(score)
        cv2.rectangle(image,box[:2],box[2:],color,2)
        cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)  
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

if __name__ == "__main__":
    #Name of the classes according to class indices.
    names = ['car', 'crosswalk', 'highway_entry', 'highway_exit', 'no_entry', 'onewayroad', 'parking', 'pedestrian', 'priority', 'roadblock', 'roundabout', 'stop', 'trafficlight']
    #Creating random colors for bounding box visualization.
    colors = {name:[random.randint(0, 255) for _ in range(3)] for name in names}
    # load model
    interpreter, input_details, output_details, input_shape = load_weights(file="./yolov7_model.tflite")
    # get video 
    cap = cv2.VideoCapture('./test.avi')
    while True:
        _,img = cap.read()
        # img = cv2.imread('shape.jpg')
        # convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ori_images = [img.copy()]
        image = img.copy()
        image, ratio, dwdh = letterbox(image,new_shape=(input_shape[2], input_shape[3]), auto=False)

        # detect image
        output_data = detect(image,interpreter, input_details, output_details, input_shape)

        # draw image
        image = draw_image(ori_images,output_data, ratio, dwdh, names, colors)
        image = cv2.resize(image,(720,480))
        cv2.imshow("image", image)
        if cv2.waitKey(1)& 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

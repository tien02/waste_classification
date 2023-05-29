import helper
import numpy as np
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

classifier, input_name = helper.load_classifier()

@app.get('/')
def root():
    '''
    Greeting!!!
    '''
    return {
        'message': f'Waste classfier API pretrained with {helper.PRETRAINED_MODEL}.'
    }


@app.post('/predict')
def predict(img_file: UploadFile = File(..., description="Image file")):
    '''
    Classify waste into the following classes:
        class_idx = {
            'cardboard': 0, 
            'glass': 1, 
            'metal': 2, 
            'paper': 3, 
            'plastic': 4, 
            'trash': 5
        }
    '''
    img = helper.read_image(img_file.file)
    out = classifier.run(None, {input_name: img})
    pred_probs = helper.softmax(out[0])
    pred_idx = np.argmax(pred_probs)
    pred_acc = round(pred_probs[:, pred_idx][0] * 100, 2)

    return {
        'label': helper.idx_class[pred_idx],
        'accuracy': str(pred_acc) + "%",
    }
import os
from model_load import graph_session
from preprocessing import read_dicom
from postprocessing import entropy_cal
from keras import backend as K
import numpy as np

### load_model part
model_path = ''
model_name = ['','','','','']
for rep in model_name:
    [model_SEUnet_sum, graph_SEUnet_sum,session_SEUnet_sum] = graph_session(model_SEUnet_sum,
                                                    'seunet',
                                                    graph_SEUnet_sum,
                                                    session_SEUnet_sum,
                                                    os.path.join(model_path,rep))
    
### preprocessing part
dcm_path = ''
data_input = read_dicom(dcm_path)

### model prediction part
yhat_seunet_sum = []
for k in range(5):
    K.clear_session()
    with graph_SEUnet_sum[k].as_default():
        with session_SEUnet_sum[k].as_default():
            yhat_seunet = model_SEUnet_sum[k].predict(data_input)
            yhat_seunet_sum.append(yhat_seunet)

### postprocessing part
thresholds = 0.5
entropy_sum = entropy_cal(yhat_seunet_sum)
yhat_seunet_sum = np.mean(yhat_seunet_sum,0)
yhat_seunet_sum[yhat_seunet_sum>=thresholds] = 1
yhat_seunet_sum[yhat_seunet_sum<thresholds] = 0

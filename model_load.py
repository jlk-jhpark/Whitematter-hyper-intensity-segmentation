from utils import get_unet,get_SEunet
import tensorflow as tf

def graph_session(model,type_model,graph_var,session_var,model_path):
    graph = tf.Graph()
    with graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        with session.as_default():
            if type_model=='unet':
                model_ss = get_unet()
                model_ss.load_weights(model_path)
            elif type_model=='seunet':
                model_ss = get_SEunet()
                model_ss.load_weights(model_path)
            model = model +[model_ss]
            
    graph_var = graph_var+[graph]
    session_var = session_var+[session]
    return model, graph_var, session_var
"""Small fully convolutional 3D point-and-box network; unknown labels have no loss."""


def network(tf,config):
    layers=tf.keras.layers
    x=layers.Input((None,None,None,1),name='membrane')
    def block(v,n):
        for _ in range(2):
            v=layers.Conv3D(n,3,padding='same',kernel_initializer='he_normal')(v)
            v=layers.LayerNormalization(axis=-1)(v)
            v=layers.Activation('relu')(v)
        return v
    v=block(x,config.filters)
    v=layers.MaxPool3D(2)(v);skip=block(v,config.filters*2)
    v=layers.MaxPool3D(2)(skip);v=block(v,config.filters*4)
    v=layers.UpSampling3D(2)(v);v=layers.Concatenate()([v,skip]);v=block(v,config.filters*2)
    heat=layers.Conv3D(1,1,dtype='float32',bias_initializer=tf.keras.initializers.Constant(-2.),name='center_logit')(v)
    bounds=layers.Conv3D(6,1,dtype='float32',activation='softplus',name='box_faces')(v)
    return tf.keras.Model(x,layers.Concatenate(dtype='float32')([heat,bounds]),name='tiny_cell_center_box3d')


def loss(tf,target,prediction):
    heat,known,faces,box_weight=target[...,0:1],target[...,1:2],target[...,2:8],target[...,8:9]
    axes=(1,2,3,4);p=tf.math.sigmoid(prediction[...,0:1])
    ce=tf.nn.sigmoid_cross_entropy_with_logits(labels=heat,logits=prediction[...,0:1])
    positive=heat*known;negative=(1-heat)*known
    center=tf.reduce_sum(ce*positive,axis=axes)/tf.maximum(tf.reduce_sum(positive,axis=axes),1.)
    center+=tf.reduce_sum(ce*negative,axis=axes)/tf.maximum(tf.reduce_sum(negative,axis=axes),1.)
    delta=tf.abs(prediction[...,1:]-faces)
    huber=tf.where(delta<.1,.5*delta**2/.1,delta-.05)
    boxes=tf.reduce_sum(huber*box_weight,axis=axes)/tf.maximum(tf.reduce_sum(box_weight,axis=axes)*6,1.)
    mae=tf.reduce_sum(delta*box_weight,axis=axes)/tf.maximum(tf.reduce_sum(box_weight,axis=axes)*6,1.)
    return tf.reduce_mean(center+2*boxes),tf.reduce_mean(center),tf.reduce_mean(mae)

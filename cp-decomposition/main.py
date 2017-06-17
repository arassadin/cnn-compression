import numpy as np
import sys, os, subprocess
import subprocess
import google.protobuf

CAFFE_ROOT_PATH = '/home/alexandr/distr/caffe/'
TENSORLAB_PATH = '/home/alexandr/distr/tensorlab/'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(CAFFE_ROOT_PATH, 'python'))
import caffe

NET_DIR = '/home/alexandr/develop/cnn-compression-tryout/Levi2015/'
INITIAL_PROTO_NAME = 'vgg_s_rgb.prototxt'
INITIAL_WEIGHTS_NAME = 'vgg_s_rgb.caffemodel'

# LAYERS = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
LAYERS = ['conv1']
RANK = 16


def conv_layer(h, w, n, group=1, pad_h=0, pad_w=0, stride_h=1, stride_w=1):
    layer = caffe.proto.caffe_pb2.LayerParameter()
    layer.type = 'Convolution'
    # layer.convolution_param.engine = caffe.proto.caffe_pb2.ConvolutionParameter.CUDNN
    if (h == w):
        layer.convolution_param.kernel_size.append(h)
    else:
        layer.convolution_param.kernel_h = h
        layer.convolution_param.kernel_w = w
    layer.convolution_param.num_output = n
    if (group != 1):
        layer.convolution_param.group = group
    if (pad_h != 0 or pad_w != 0):
        layer.convolution_param.pad_h = pad_h
        layer.convolution_param.pad_w = pad_w
    if (stride_h != 1 or stride_w != 1):
        layer.convolution_param.stride_h = stride_h
        layer.convolution_param.stride_w = stride_w
    return layer

def find_layer_by_name(model, layer_name):
    k = 0
    while model.layer[k].name != layer_name:
        k += 1
        if (k > len(model.layer)):
            raise IOError('layer with name %s not found' % layer_name)
    return k
 
def accelerate_model(model, layer_to_decompose, rank):
    k = layer_to_decompose
    r = rank
    new_model = caffe.proto.caffe_pb2.NetParameter()
    for i in range(k):
        new_model.layer.extend([model.layer[i]])
    decomposed_layer = model.layer[k]
    if decomposed_layer.type != 'Convolution':
        raise AttributeError('only convolution layer can be decomposed')
    param = decomposed_layer.convolution_param   
    if not hasattr(param, 'pad'):
        param.pad = [0]
    if param.pad == []:
        param.pad.append(0)
    if not hasattr(param, 'stride'):
        param.stride = [1]
    if param.stride == []:
        param.stride.append(1)
    new_model.layer.extend([conv_layer(1, 1, r)])
    new_model.layer.extend([conv_layer(param.kernel_size[0], 1, r, r, pad_h=param.pad[0], stride_h=param.stride[0])])
    new_model.layer.extend([conv_layer(1, param.kernel_size[0], r, r, pad_w=param.pad[0], stride_w=param.stride[0])])
    new_model.layer.extend([conv_layer(1, 1, param.num_output)])
    name = decomposed_layer.name
    for i in range(4):
        new_model.layer[k+i].name = name + '-' + str(i + 1)
        new_model.layer[k+i].bottom.extend([name + '-' + str(i)])
        new_model.layer[k+i].top.extend([name + '-' + str(i + 1)])
    new_model.layer[k].bottom[0] = model.layer[k].bottom[0]
    new_model.layer[k+3].top[0] = model.layer[k].top[0]
    for i in range(k+1, len(model.layer)):
        new_model.layer.extend([model.layer[i]])
    return new_model

def create_deploy_model(model, input_dim=[1, 3, 224, 224]):
    new_model = caffe.proto.caffe_pb2.NetParameter()
    new_model.input.extend(['data'])
    new_model.input_dim.extend(input_dim)
    for i in range(1, len(model.layer)):
        new_model.layer.extend([model.layer[i]])
    return new_model
    
def load_model(filename):
    model = caffe.proto.caffe_pb2.NetParameter()
    input_file = open(filename, 'r')
    google.protobuf.text_format.Merge(str(input_file.read()), model)
    input_file.close()
    return model

def save_model(model, filename):
    output_file = open(filename, 'w')
    google.protobuf.text_format.PrintMessage(model, output_file)
    output_file.close()

def prepare_models():    
    proto_initial = load_model(os.path.join(NET_DIR, INITIAL_PROTO_NAME))
    proto_acc = load_model(os.path.join(NET_DIR, INITIAL_PROTO_NAME))
    layers_ind_initial = []
    layers_ind_acc = []

    for LAYER in LAYERS:

        ind_initial = find_layer_by_name(proto_initial, LAYER)
        ind_acc = find_layer_by_name(proto_acc, LAYER)

        layers_ind_initial.append(ind_initial)
        layers_ind_acc.append(ind_acc)

        proto_acc = accelerate_model(proto_acc, ind_acc, RANK)

    save_model(proto_acc,
        os.path.join(NET_DIR, INITIAL_PROTO_NAME.replace('.prototxt', '_accelerated.prototxt')))

    net_initial = caffe.Classifier(os.path.join(NET_DIR, INITIAL_PROTO_NAME),
        os.path.join(NET_DIR, INITIAL_WEIGHTS_NAME))
    net_acc = caffe.Classifier(os.path.join(NET_DIR, INITIAL_PROTO_NAME.replace('.prototxt', '_accelerated.prototxt')),
        os.path.join(NET_DIR, INITIAL_WEIGHTS_NAME))

    for l_i, (ind_init, ind_acc) in enumerate(zip(layers_ind_initial, layers_ind_acc)):
        print 'Layer: {}'.format(LAYERS[l_i])

        w = net_initial.layers[ind_init].blobs[0].data
        g = proto_initial.layer[ind_init].convolution_param.group
        if (g > 1):
            weights = np.zeros((w.shape[0], g * w.shape[1], w.shape[2], w.shape[3]))
            for i in range(g):
                weights[
                    i * w.shape[0] / g : (i + 1) * w.shape[0] / g,
                    i * w.shape[1] : (i + 1) * w.shape[1], :, :
                ] = w[i * w.shape[0] / g : (i + 1) * w.shape[0] / g, :, :, :]
                temp = w[i * w.shape[0] / g : (i + 1) * w.shape[0] / g, :, :, :]
                np.savetxt('group' + str(i) + '.txt', temp.ravel())
        else:
            weights = w
        bias = net_initial.layers[ind_init].blobs[1]

        np.savetxt('weights.txt', weights.ravel())
        np.savetxt('biases.txt', bias.data.ravel())

        if 1:
            s = weights.shape
            command = 'addpath(\'%s\');' % (TENSORLAB_PATH)
            command = command + ' decompose(%d, %d, %d, %d, %d); exit;' % (s[3], s[2], s[1], s[0], RANK)
            subprocess.call(['matlab', '-nodesktop', '-nosplash', '-r', command])

        n = proto_initial.layer[ind_init].convolution_param.num_output
        d = proto_initial.layer[ind_init].convolution_param.kernel_size[0]
        c = weights.shape[1] # / proto_initial.layer[ind_init].convolution_param.group # i don't know what i'm doing 

        if 1:
            f_x = np.loadtxt('f_x.txt').transpose()
            f_y = np.loadtxt('f_y.txt').transpose()
            f_c = np.loadtxt('f_c.txt').transpose()
            f_n = np.loadtxt('f_n.txt')
        else:    
            f_x = np.random.standard_normal([RANK*d])
            f_y = np.random.standard_normal([RANK*d])
            f_c = np.random.standard_normal([RANK*c])
            f_n = np.random.standard_normal([RANK*n])
        
        f_y = np.reshape(f_y, [RANK, 1, d, 1])
        f_x = np.reshape(f_x, [RANK, 1, 1, d])
        f_c = np.reshape(f_c, [RANK, c, 1, 1])
        f_n = np.reshape(f_n, [n, RANK, 1, 1])

        np.copyto(net_acc.layers[ind_acc].blobs[0].data, f_c)
        np.copyto(net_acc.layers[ind_acc + 1].blobs[0].data, f_y)
        np.copyto(net_acc.layers[ind_acc + 2].blobs[0].data, f_x)
        np.copyto(net_acc.layers[ind_acc + 3].blobs[0].data, f_n)
        np.copyto(net_acc.layers[ind_acc + 3].blobs[1].data, bias.data)

    net_acc.save(os.path.join(NET_DIR,
        INITIAL_WEIGHTS_NAME.replace('.caffemodel', '_accelerated.caffemodel')))

    print 'Model successfully processed!'

prepare_models()

require 'optim'
require 'nn'

DATA_PATH = '/home/arassadin/develop/emotions/RaFD/'

local function commandLine()
    local cmd = torch.CmdLine()

    cmd:text()
    cmd:text('Options:')
    cmd:option('-gpu',          false, 'gpu usage')
    cmd:option('-seed',         1, 'fixed input seed for repeatable experiments')
    cmd:option('-hash',         false, 'enable hashing')
    cmd:option('-compression',  0.125, 'compression rate')
    cmd:option('-hash_bias',    true, 'Specify whether or not to hash bias terms')
    cmd:option('-rescale_grad', false, 'Specify whether or not divide gradients by the frequency it is used')
    cmd:option('-learningRate', 0.001, 'learning rate at t=0')
    cmd:option('-decay_lr',     1e-4, 'learning rate decay')
    cmd:option('-bsize',        20, 'mini-batch size (1 = pure stochastic)')
    cmd:option('-momentum',     0.9, 'momentum (SGD only)')
    cmd:option('-l2reg',        0, 'l2 regularization')
    cmd:option('-maxEpoch',     100, 'maximum # of epochs to train for')
    cmd:option('-xi',           true, 'Specify whether or not to use auxiliary hash function Xi')
    cmd:option('-hashSeed',     1691, 'seed for hash functions')
    cmd:option('-shuffle',      true, 'shuffle training data')
    cmd:text()

    local opt = cmd:parse(arg or {})

    torch.manualSeed(opt.seed)
    return opt
end

local function load_data_cifar10(opt)
    local cifar     = require 'cifar10'    
    local trainData = cifar.traindataset()
    local testData  = cifar.testdataset()
    local data      = {}
    -- data['xtrain']      = trainData.data:float()
    data['xtrain']      = trainData.data:float():sub(1, 1000)
    -- data['xtest']      = testData.data:float()
    data['xtest']      = testData.data:float():sub(1, 1000)
    -- data['ytrain']      = trainData.label + 1
    data['ytrain']      = trainData.label:sub(1, 1000) + 1
    -- data['ytest']      = testData.label + 1
    data['ytest']      = testData.label:sub(1, 1000) + 1
    opt.outputDim   = 10

    -- shuffle the training data
    local shuffle_idx = torch.randperm(data.xtrain:size(1),'torch.LongTensor')
    data.xtrain           = data.xtrain:index(1, shuffle_idx)
    data.ytrain           = data.ytrain:index(1, shuffle_idx)

    -- normalization
    local xMax = math.max(data.xtrain:max(), data.xtest:max())
    data.xtrain:div(xMax)
    data.xtest:div(xMax)

    return data
end

local function load_data_rafd_numpy(opt)
    npy4th = require 'npy4th'
    local X_train = npy4th.loadnpy(DATA_PATH .. 'X_train.npy')
    local X_test = npy4th.loadnpy(DATA_PATH .. 'X_test.npy')
    local y_train = npy4th.loadnpy(DATA_PATH .. 'y_train.npy')
    local y_test = npy4th.loadnpy(DATA_PATH .. 'y_test.npy')
    local data = {}
    -- data['xr']      = X_train:float()
    -- data['xe']      = X_test:float()
    -- data['yr']      = y_train + 1
    -- data['ye']      = y_test + 1
    data['xtrain'] = X_train:float()
    data['xtest'] = X_test:float()
    data['ytrain'] = y_train + 1
    data['ytest'] = y_test + 1
    opt.outputDim = 8

    -- shuffle the training data
    local shuffle_idx = torch.randperm(data.xtrain:size(1),'torch.LongTensor')
    data.xtrain = data.xtrain:index(1, shuffle_idx)
    data.ytrain = data.ytrain:index(1, shuffle_idx)

    -- normalization
    local xMax = math.max(data.xtrain:max(), data.xtest:max())
    data.xtrain:div(xMax)
    data.xtest:div(xMax)

    return data
end

local function adjust(opt, nTrain)
    ------ Learning Rate Adjustment
    opt.nBatches     = math.ceil(nTrain/opt.bsize)
    --opt.decay_lr     = opt.decay_lr / opt.nBatches
    opt.decay_lr     = opt.decay_lr
end


local function optimConfig(opt)
    opt.optim_config = {
        learningRate          = opt.learningRate,
        learningRateDecay     = opt.decay_lr,
        weightDecay           = opt.l2reg,
        momentum              = opt.momentum
    }
    opt.optimizer = optim.sgd
end

local function hashConfig(opt)
    local hash_config = {
        compression  = opt.compression,
        xi           = opt.xi,
        hbias        = opt.hash_bias, 
        hashSeed     = opt.hashSeed,
        rescale_grad = opt.rescale_grad
    }
    return hash_config

end

local function createModel_VGG_S(opt)
    local hash_config = nil
    if opt.hash then
        hash_config = hashConfig(opt)
        require 'cunn'
        -- require 'cutorch'
        -- cutorch.setDevice(5)
        require 'HashLinear'
        require 'libhashnn'
    end

    local function linear_module(inSize, outSize)
        local linear = nil
        if hash_config then
            linear = nn.HashLinear(inSize, outSize, hash_config)
        else
            linear = nn.Linear(inSize, outSize)
        end
        return linear
    end

    local vgg = nn.Sequential()

    local function ConvBNReLU(nInputPlane, nOutputPlane, ksize, stride)
      vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, ksize, ksize, stride, stride))
      vgg:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
      vgg:add(nn.ReLU())
      return vgg
    end

    -- vgg:add(nn.SpatialZeroPadding(3, 3, 3, 3))
    vgg:add(nn.SpatialZeroPadding(1, 1, 1, 1))
    -- ConvBNReLU(3, 96, 7, 1)
    ConvBNReLU(3, 96, 7, 2)
    -- vgg:add(nn.SpatialMaxPooling(2,2, 2,2):ceil())
    vgg:add(nn.SpatialMaxPooling(3,3, 3,3):ceil())

    -- vgg:add(nn.SpatialZeroPadding(2, 2, 2, 2))
    vgg:add(nn.SpatialZeroPadding(1, 1, 1, 1))
    ConvBNReLU(96, 256, 5, 1)
    vgg:add(nn.SpatialMaxPooling(2,2, 2,2):ceil())

    vgg:add(nn.SpatialZeroPadding(1, 1, 1, 1))
    ConvBNReLU(256, 512, 3, 1)
    vgg:add(nn.SpatialZeroPadding(1, 1, 1, 1))
    ConvBNReLU(512, 512, 3, 1)
    vgg:add(nn.SpatialZeroPadding(1, 1, 1, 1))
    ConvBNReLU(512, 512, 3, 1)
    -- vgg:add(nn.SpatialMaxPooling(2,2, 2,2):ceil())
    vgg:add(nn.SpatialMaxPooling(3,3, 3,3):ceil())

    -- vgg:add(nn.View(8192))
    vgg:add(nn.View(18432))

    vgg:add(linear_module(18432, 4048))
    vgg:add(nn.ReLU())
    vgg:add(nn.Dropout(0.5))

    vgg:add(linear_module(4048, 4048))
    vgg:add(nn.ReLU())
    vgg:add(nn.Dropout(0.5))

    vgg:add(linear_module(4048, opt.outputDim))

    local function MSRinit(net)
        local function init(name)
            for k,v in pairs(net:findModules(name)) do
                local n = v.kW*v.kH*v.nOutputPlane
                v.weight:normal(0,math.sqrt(2/n))
                v.bias:zero()
            end
        end
        init'nn.SpatialConvolution'
    end
    MSRinit(vgg)

    vgg:add(nn.LogSoftMax())

    local criterion = nn.ClassNLLCriterion()
    print(vgg)
    print(criterion)
    if opt.gpu then
        require 'cunn'
        vgg:cuda()
        criterion:cuda()
    end
    return vgg, criterion
end

local function train_cpu(model, criterion, W, grad, data, opt)
    model:training()

    local nTrain = data.xtrain:size(1)

    -- shuffle the data
    if opt.shuffle then
        local shuffle_idx = torch.randperm(nTrain,'torch.LongTensor')
        data.xtrain = data.xtrain:index(1, shuffle_idx)
        data.ytrain = data.ytrain:index(1, shuffle_idx)
    end

    -- Train minibatch
    for t = 1, nTrain, opt.bsize do
        ------ Minibatch generation
        local idx     = math.min(t + opt.bsize - 1, nTrain)
        local inputs  = data.xtrain:sub(t, idx)
        local targets = data.ytrain:sub(t, idx)

        -- objective function for optimization
        function feval(x)
            assert(x==W)
            grad:zero() -- reset grads

            local outputs  = model:forward(inputs)
            local f        = criterion:forward(outputs, targets)
            local df_dw    = criterion:backward(outputs, targets) -- ~= df/dW
            model:backward(inputs, df_dw)

            f = f / opt.bsize -- Adjust for batch size

            return f, grad
        end
        opt.optimizer(feval, W, opt.optim_config)
    end 

end

local function train_gpu(model, criterion, W, grad, data, opt)
    require 'cunn'

    model:training()

    local inputs_gpu = torch.CudaTensor()
    local targets_gpu = torch.CudaTensor()
    local nTrain = data.xtest:size(1)

    -- shuffle the data
    if opt.shuffle then
        local shuffle_idx = torch.randperm(nTrain,'torch.LongTensor')
        data.xtest = data.xtest:index(1, shuffle_idx)
        data.ytest = data.ytest:index(1,shuffle_idx)
    end

    -- Train minibatch
    for t = 1, nTrain, opt.bsize do
        ------ Minibatch generation
        local idx     = math.min(t+opt.bsize-1, nTrain)
        local inputs  = data.xtest:sub(t, idx)
        local targets = data.ytest:sub(t, idx)

        -- copy data from cpu to gpu
        inputs_gpu:resize(inputs:size()):copy(inputs)
        targets_gpu:resize(targets:size()):copy(targets)

        -- objective function for optimization
        function feval(x)
            assert(x==W)
            grad:zero() -- reset grads
            model:zeroGradParameters()

            local outputs  = model:forward(inputs_gpu)
            local f        = criterion:forward(outputs, targets_gpu)
            local df_dw    = criterion:backward(outputs, targets_gpu) -- ~= df/dW
            model:backward(inputs_gpu, df_dw)

            f = f / opt.bsize -- Adjust for batch size

            return f,grad
        end
        opt.optimizer(feval,W, opt.optim_config)
    end 

end

local function evaluation(X, y, model, bsize, confusion)
    model:evaluate()

    local N     = X:size(1)
    local err   = 0
    
    for k = 1, N, bsize do
        local idx         = math.min(k + bsize - 1, N)
        local inputs      = X:sub(k, idx)
        local targets     = y:sub(k, idx)

        local outputs     = model:forward(inputs)
        confusion:batchAdd(outputs, targets)
    end 

    confusion:updateValids()
    err    = 1 - confusion.totalValid
    confusion:zero()

    return err
end

local function reportErr(data, model, opt, confusion)
    local bestTest  = math.huge
    local bestTrain = math.huge
    local bestEpoch = 0
    local function report(t)
        local err_train = evaluation(data.xtrain, data.ytrain, model, opt.bsize, confusion)
        local err_test = evaluation(data.xtest, data.ytest, model, opt.bsize, confusion)
        print('---------------Epoch: ' .. t .. ' of ' .. opt.maxEpoch)
        print(string.format('Current Errors: train: %.4f | test: %.4f', err_train, err_test))

        if bestTest > err_test then
            bestTrain = err_train
            bestTest = err_test
            bestEpoch = t
            -- torch.save('/home/arassadin/develop/emotions/weights_hash.t7', model)
        end
        print(string.format('Optima achieved at epoch %d: test: %.4f', bestEpoch, bestTest))
    end

    return report
end

local function main()
    local opt = commandLine()
    torch.setdefaulttensortype('torch.FloatTensor')
    local data = load_data_rafd_numpy(opt)
    print('Data Loaded')

    local nTrain = data.xtrain:size(1)
    adjust(opt, nTrain)
    optimConfig(opt)
    local model, criterion = createModel_VGG_S(opt)

    local confusion     = optim.ConfusionMatrix(opt.outputDim)
    local W,grad        = model:getParameters()

    print('the number of paramters is ' .. W:nElement())

    local train = train_cpu
    if opt.gpu then
        train = train_gpu
    end

    local report = reportErr(data, model, opt, confusion)
    for t = 1,opt.maxEpoch do
        train(model, criterion, W, grad, data, opt) -- performs a single epoch
        report(t)
        collectgarbage()
    end
end

main()

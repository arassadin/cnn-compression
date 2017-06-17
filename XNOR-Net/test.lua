require 'optim'
require 'nn'
require 'cunn'
require 'cutorch'
require 'BinActiveZ'

DATA_PATH = '/home/arassadin/develop/emotions/RaFD/'

local function commandLine()
    local cmd = torch.CmdLine()

    cmd:text()
    cmd:text('Options:')
    cmd:option('-gpu',          false, 'gpu usage')
    cmd:option('-seed',         1, 'fixed input seed for repeatable experiments')
    cmd:option('-bsize',        20, 'mini-batch size (1 = pure stochastic)')
    cmd:option('-nint',         10, 'number of iterations')
    cmd:text()

    local opt = cmd:parse(arg or {})

    torch.manualSeed(opt.seed)
    return opt
end

local function load_data_rafd_numpy(opt)
    npy4th = require 'npy4th'
    local X_test = npy4th.loadnpy(DATA_PATH .. 'X_test.npy')
    local y_test = npy4th.loadnpy(DATA_PATH .. 'y_test.npy')
    local data = {}
    data['xtest'] = X_test:float()
    data['ytest'] = y_test + 1
    opt.outputDim = 8

    return data
end

local function evaluate(X, y, model, bsize, confusion)
    model:evaluate()

    local N = X:size(1)
    
    timer = torch.Timer()
    for k = 1, N, bsize do
        local idx         = math.min(k + bsize - 1, N)
        local inputs      = X:sub(k, idx)
        local targets     = y:sub(k, idx)

        local outputs = model:forward(inputs)
        confusion:batchAdd(outputs, targets)
    end 
    t = timer:time().real

    confusion:updateValids()
    local err = 1 - confusion.totalValid
    confusion:zero()

    return err, t
end

local function main()
    cutorch.setDevice(5)

    local opt = commandLine()
    torch.setdefaulttensortype('torch.FloatTensor')
    local data = load_data_rafd_numpy(opt)

    local model = torch.load('/home/arassadin/develop/emotions/weights_XNOR.t7')
    local W, grad = model:getParameters()
    print('the number of paramters is ' .. W:nElement())

    local confusion = optim.ConfusionMatrix(opt.outputDim)

    local avg_time = 0
    for it = 1, opt.nint do
        local err, t = evaluate(data.xtest, data.ytest, model, opt.bsize, confusion)
        print(string.format('err: %.4f, time elapsed: %.4f', err, t))
        avg_time = avg_time + t
        collectgarbage()
    end
    print(string.format('---------------------The average time: %.4f seconds', avg_time / opt.nint))
end

main()

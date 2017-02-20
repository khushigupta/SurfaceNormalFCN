require 'torch'
require 'optim'
require 'pl'
require 'paths'

local fcn = {}

local inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
-- put the labels for each batch in targets
local targets = torch.Tensor(opt.batchSize * opt.labelSize * opt.labelSize)

local sampleTimer = torch.Timer()
local dataTimer = torch.Timer()


-- training function
function fcn.train(inputsArg, targetsArg)
  cutorch.synchronize()
  epoch = epoch or 1
  local dataLoadingTime = dataTimer:time().real; sampleTimer:reset(); -- timers
  local dataBatchSize = opt.batchSize 

  -- TODO: implement the training function -- TODO: setup training functions, use fcn_train_cls.lua
  inputs:copy(inputsArg)
  targets:copy(targetsArg)
  
  if opt.gpu then
    inputs:cuda()
    targets:cuda()
  end
  
  local err, outputs
  feval = function(x)
    
    model_FCN:zeroGradParameters()
    
    predictions = model_FCN:forward(inputs)
    err = criteria:forward(predictions, targets)
    
    local dLossdOuput = criteria:backward(predictions, targets)
    model:backward(inputs, dLossdOuput)
    return err, dLossdOuput
  end

  optim.sgd(feval, parameters, optimState)


  batchNumber = batchNumber + 1
  cutorch.synchronize(); collectgarbage();
  print(('Epoch: [%d][%d/%d]\tTime %.3f DataTime %.3f'):format(epoch, batchNumber, opt.epochSize, sampleTimer:time().real, dataLoadingTime))
  dataTimer:reset()

end


return fcn



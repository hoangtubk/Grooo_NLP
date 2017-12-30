---
--- Created by tuhoangbk.
--- DateTime: 19/11/2017 20:26
---

require('nn')
require('rnn')
require('cutorch')
require('cunn')
class = require('pl.class')
include('testing.lua')

local testing = Testing()
---@class Training
Training = class()

---init Training
function Training:_init()
    ---model save after each n loop
    self.train_size = 57057
    self.batch_size = 256
end

---@param table_inputs table
---@param table_targets table
---@param criterion
---@param learning_rate number
---@param input_from number
---@param input_to number
function Training:train(model, next_batch_input, next_batch_target, criterion, learning_rate, is_use_cuda)
    ---begin training
    local timer = torch.Timer()
    local  err, sum_err, precision = 0, 0, 0
    local grad_ouputs = {}
    for step = 1, math.floor(self.train_size/self.batch_size) do
        local input = next_batch_input()
        local target = next_batch_target()
        ---use GPU with cuda
        if is_use_cuda then
            input = input:cuda()
            target = target:cuda()
        end
        local output = model:forward(input)
        model:zeroGradParameters()
        err = criterion:forward(output, target)
        grad_ouputs[step] = criterion:backward(output, target)
        model:backward(input, grad_ouputs[step])
        model:updateParameters(learning_rate)
        sum_err = sum_err + err
    end
    local time_train = timer:time().real

    return sum_err, time_train
end

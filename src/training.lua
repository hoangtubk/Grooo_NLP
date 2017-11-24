---
--- Created by tuhoangbk.
--- DateTime: 19/11/2017 20:26
---

require('nn')
require('rnn')
require('cutorch')
require('cunn')
class = require('pl.class')
include('model.lua')
include('testing.lua')

local model = Model()
local testing = Testing()

Training = class()
function Training:_init()

end

---@param table_inputs table
---@param table_targets table
---@param criterion
---@param learning_rate number
---@param number_input number @number input sequence sentent
function Training:train(table_inputs, table_targets, criterion, learning_rate, number_input)
    local mlp = model:build_brnn()
---use GPU with cuda
    criterion = criterion:cuda()
    mlp = mlp:cuda()
    local iteration = 0
    local timer = torch.Timer()
    while true do
        iteration = iteration +1
        local  err, sum_err, precision = 0, 0, 0
        local grad_ouputs = {}
        for step = 1, number_input do
        ---use GPU with cuda
            table_inputs[step] = table_inputs[step]:cuda()
            table_targets[step] = table_targets[step]:cuda()
            local output = mlp:forward(table_inputs[step])
            mlp:zeroGradParameters()
            err = criterion:forward(output, table_targets[step])
            grad_ouputs[step] = criterion:backward(output, table_targets[step])
            mlp:backward(table_inputs[step], grad_ouputs[step])
            mlp:updateParameters(learning_rate)
            sum_err = sum_err + err
        end
        ---compute precision and print result
        precision = testing:test(mlp, table_inputs, table_targets, number_input)
        print(string.format("Iteration %d ; Error = %f, Precision = %f ", iteration, sum_err, precision))
        ---Training is finished:
        if precision > 0.9 then
            break
        end
    end
    torch.save('seqbnn.t7', mlp)
    print('Time training:' .. timer:time().real .. ' seconds')
end

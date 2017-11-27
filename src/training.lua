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

Training = class()
function Training:_init()

end

---@param table_inputs table
---@param table_targets table
---@param criterion
---@param learning_rate number
---@param input_from number
---@param input_to number
function Training:train(model, table_inputs, table_targets, criterion, learning_rate, input_from, input_to, test_from, test_to)
    ---use GPU with cuda
    criterion = criterion:cuda()
    model = model:cuda()
    for i = 1, #table_inputs do
        table_inputs[i] = table_inputs[i]:cuda()
        table_targets[i] = table_targets[i]:cuda()
    end
    ---begin training
    local iteration = 0
    local timer = torch.Timer()
    while iteration < 200 do
        iteration = iteration +1
        learning_rate = learning_rate - 0,002495
        local  err, sum_err, precision = 0, 0, 0
        local grad_ouputs = {}
        for step = input_from, input_to do
        ---use GPU with cuda
            local output = model:forward(table_inputs[step])
            model:zeroGradParameters()
            err = criterion:forward(output, table_targets[step])
            grad_ouputs[step] = criterion:backward(output, table_targets[step])
            model:backward(table_inputs[step], grad_ouputs[step])
            model:updateParameters(learning_rate)
            sum_err = sum_err + err
        end
        ---compute precision and print result
        precision = testing:test(model, table_inputs, table_targets, input_from, input_to)
        print(learning_rate)
        print(string.format("Iteration %d ; Error = %f, Precision = %f ", iteration, sum_err, precision))
        ---Training is finished:
        --if sum_err < 1 then
        --    break
        --end
    end
    torch.save('seqbnn.t7', model)
    print('Time training:' .. timer:time().real .. ' seconds')
end

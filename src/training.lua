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
local model = Model()

Training = class()
function Training:_init()

end

---@param table_inputs table
---@param table_targets table
function Training:train(table_inputs, table_targets, criterion, learning_rate)
    local mlp = model:build_brnn()
---use GPU with cuda
    --criterion = criterion:cuda()
    --mlp = mlp:cuda()
    local  err, sum_err = 0, 0
    local grad_ouputs = {}
    while true do
        local  err, sum_err = 0, 0
        for step = 1, 15 do
        ---use GPU with cuda
            --table_inputs[step] = table_inputs[step]:cuda()
            --table_targets[step] = table_targets[step]:cuda()
            local output = mlp:forward(table_inputs[step])
            print(output)
            mlp:zeroGradParameters()
            print(#output)
            print(#table_targets[step])
            err = criterion:forward(output, table_targets[step])
            grad_ouputs[step] = criterion:backward(output, table_targets[step])
            assert(false)
            mlp:backward(table_inputs[step], grad_ouputs[step])
            assert(false)
            mlp:updateParameters(learning_rate)
            sum_err = sum_err + err
        end
        print(sum_err)
    end
end

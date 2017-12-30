
---
--- Created by tuhoangbk.
--- DateTime: 19/11/2017 20:25
---
---======================================================================
---run Server
local restserver = require("restserver")
local server = restserver:new():port(8080)
---======================================================================
require('rnn')
require('nn')
include('model.lua')
include('tokenizer.lua')
include('pl_utils.lua')
include('training.lua')
include('testing.lua')
include('dataset.lua')

local model = Model()
local utils = Utils()
local tknz = Tokenizer()
local training = Training()
local testing = Testing()

local batch_size = 256
local max_dim = 120
local nout = 68
local epochs = 100
local train_size = 57057

local input_path = '../output/all_input.txt'
local target_path = '../output/all_target.txt'
local model_path = '../ModelCheckpoint/model.t7'
local load_model_path = '/media/tuhoangbk/HDD/mlp_uncuda.t7'

local mlp = model:build_brnn()
print(mlp)
---======================================================================
--set Flag
local is_train = false
local is_test = true
local is_test_string = true
local is_run_server = false
local is_load_model = true
local is_use_cuda = false
---======================================================================
--- Init input
--- Khởi tạo input là 1 table chứa các DoubleTensor
--local table_inputs = {}
--local table_targets = {}
--local loop_count = 0
--for i = 1, #data_inp, batch_size do
--    loop_count = loop_count + 1
--    local input = torch.Tensor(batch_size, max_dim):zero()
--    local target = torch.Tensor(batch_size, max_dim):zero()
--
--    for n = i, batch_size*loop_count do
--        if n > #data_inp then
--            goto END_LOOP
--        end
--        local inp = torch.Tensor(max_dim):zero()
--        local tar = torch.Tensor(max_dim):zero()
--
--        local table_index_inp = tknz:split_word_only(data_inp[n])
--        local table_index_tar = tknz:split_word_only(data_tar[n])
--        for j = 1, #table_index_inp do
--            inp[j] = table_index_inp[j]
--            tar[j] = table_index_tar[j]
--        end
--        input[n - (batch_size*(loop_count-1))] = inp
--        target[n - (batch_size*(loop_count-1))] = tar
--        ::END_LOOP::
--    end
--    table.insert(table_inputs, input)
--    table.insert(table_targets, target)
--end
---======================================================================
local learning_rate = 0.01
--local input_from = 1
--local input_to = 3000
--local test_from = 1
--local test_to = 3000
--local mlp = torch.load('../ModelCheckpoint/seqbnn100_raw_beckshop_2.0.t7')
if is_load_model then
    mlp = torch.load(load_model_path)
end

if is_use_cuda then
    mlp = mlp:cuda()
end

---======================================================================
if is_train then
    ---set weights for criterion
    print("Set weight for criterion")
    local dataset_weight = DataSet(target_path)
    local next_batch_target_weight = dataset_weight:batches(batch_size)

    local weight = torch.Tensor(nout):zero()
    for step = 1, math.floor(train_size/batch_size) do
        local data = next_batch_target_weight()
        for i = 1, batch_size do
            for j = 1, max_dim do
                for iclass = 1, nout do
                    if data[i][j] == iclass then
                        weight[iclass] = weight[iclass] + 1
                    end
                end
            end
        end
    end
    print(weight)
    local sum_class = 0
    for i = 1, nout do
        sum_class = sum_class + weight[i]
    end
    for i = 1, nout do
        if weight[i] ~= 0 then
            weight[i] = sum_class/weight[i]
        end
    end
    print(weight)
    print("Begin training")
    ---Begin Training
    local criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(weight),1))
    if is_use_cuda then
        criterion = criterion:cuda()
    end
    local sum_err, time_train = 0, 0
    local prec = 0
    for epoch = 1, epochs do
        ---Create input & target
        local dataset_input = DataSet(input_path)
        local next_batch_input = dataset_input:batches(batch_size)
        local next_batch_input_test = dataset_input:batches(batch_size)

        local dataset_target = DataSet(target_path, target_t7_path)
        local next_batch_target = dataset_target:batches(batch_size)
        local next_batch_target_test = dataset_target:batches(batch_size)

        print("epoch = "..tostring(epoch)..'/'..tostring(epochs))
        sum_err, time_train = training:train(mlp, next_batch_input, next_batch_target, criterion, learning_rate, is_use_cuda)
        prec = testing:test(mlp, next_batch_input_test, next_batch_target_test)
        print(string.format("Error = %f, Precision = %f, time = %f s ", sum_err, prec, time_train))
        torch.save(model_path, mlp)
    end
end

---======================================================================
---Testing
if is_test then
    print('Testing')
    if is_test_string then
        local test_string = testing:test_string(mlp, 'phung khoang nhe', is_use_cuda)
        print(test_string)
    else
        local dataset_input = DataSet(input_path)
        local next_batch_input_test = dataset_input:batches(batch_size)
        local dataset_target = DataSet(target_path)
        local next_batch_target_test = dataset_target:batches(batch_size)

        local precition = testing:test(mlp, next_batch_input_test, next_batch_target_test, is_use_cuda)
        print(precition)
    end
end

---======================================================================
if is_run_server then
    --- Init Server
    server:add_resource("tuha", {
        {
            method = "GET",
            path = "{id:.*}",
            produces = "application/json",
            handler = function(_, content)
                print('Input: ' .. content['content'])
                local output = testing:test_string(mlp, content['content'])
                print('Output: ' .. output )
                print('====================================================')
                return restserver.response():status(200):entity(output)
            end,
        },
    })
    --- Run Server
    server:enable("restserver.xavante"):start()
end
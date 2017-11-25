
---
--- Created by tuhoangbk.
--- DateTime: 19/11/2017 20:25
---

require('rnn')
require('nn')

include('model.lua')
include('tokenizer.lua')
include('pl_utils.lua')
include('training.lua')
include('testing.lua')

local model = Model()
local utils = Utils()
local tknz = Tokenizer()
local training = Training()
local testing = Testing()

local batch_size = 10
local max_dim = 97
local nout = 68

local mlp = model:build_brnn()
print(mlp)

--- Init input
--- Khởi tạo input là 1 table chứa các DoubleTensor
local data_inp = utils:read_lines('../output/input.txt')
local data_tar = utils:read_lines('../output/target.txt')
local table_inputs = {}
local table_targets = {}
local loop_count = 0
for i = 1, #data_inp, batch_size do
    loop_count = loop_count + 1
    local input = torch.Tensor(batch_size, max_dim):zero()
    local target = torch.Tensor(batch_size, max_dim):zero()

    for n = i, batch_size*loop_count do
        if n > #data_inp then
            goto END_LOOP
        end
        local inp = torch.Tensor(max_dim):zero()
        local tar = torch.Tensor(max_dim):zero()

        local table_index_inp = tknz:split_word_only(data_inp[n])
        local table_index_tar = tknz:split_word_only(data_tar[n])
        for j = 1, #table_index_inp do
            inp[j] = table_index_inp[j]
            tar[j] = table_index_tar[j]
        end
        input[n - (batch_size*(loop_count-1))] = inp
        target[n - (batch_size*(loop_count-1))] = tar
        ::END_LOOP::
    end
    table.insert(table_inputs, input)
    table.insert(table_targets, target)
end

--- Training
local learning_rate = 0.02
local number_input = 100
local criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(),1))
training:train(table_inputs, table_targets, criterion, learning_rate, number_input)

---Testing
--testing:test(mlp, table_inputs, table_targets, number_input)

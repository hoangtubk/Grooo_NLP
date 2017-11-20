
---
--- Created by tuhoangbk.
--- DateTime: 19/11/2017 20:25
---

require('rnn')
require('nn')

include('model.lua')
include('tokenizer.lua')
include('pl_utils.lua')

local model = Model()
local utils = Utils()
local tknz = Tokenizer()

local batch_size = 10
local max_dim = 97

local mlp = model:build_brnn()
print(mlp)

--- Init input
local data_inp = utils:read_lines('../output/input.txt')
local data_tar = utils:read_lines('../output/target.txt')
local input = torch.Tensor(batch_size, max_dim):zero()
local target = torch.Tensor(batch_size, max_dim):zero()
--for i = 1, #data_inp do
for i = 1, batch_size do
    local inp = torch.Tensor(max_dim)
    local tar = torch.Tensor(max_dim)
    local table_index_inp = tknz:split_word_only(data_inp[i])
    local table_index_tar = tknz:split_word_only(data_tar[i])
    for j = 1, #table_index_inp do
    --for j = 1, max_dim do
        inp[j] = table_index_inp[j]
        tar[j] = table_index_tar[j]
    end
    input[i] = inp
end
print('Input_')
print(input)

--local test_in = torch.DoubleTensor({{10, 3 , 4, 5, 0 }, {10, 4, 5, 6, 0}})
--print(test_in)
--print(input)
local output = mlp:forward(input)
print('Output_')
print(output)


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

local mlp = model:build_brnn()
print(mlp)

--- Init input
local data_inp = utils:read_lines('../output/input.txt')
local data_tar = utils:read_lines('../output/target.txt')
local input = torch.DoubleTensor(10, 100):zero()
local target = torch.DoubleTensor(10, 100):zero()
--for i = 1, #data_inp do
for i = 1, 10 do
    local table_index_inp = tknz:split_word_only(data_inp[i])
    local table_index_tar = tknz:split_word_only(data_tar[i])
    for j = 1, #table_index_inp do
        input[i][j] = table_index_inp[j]
        target[i][j] = table_index_tar[j]
    end
end

local test_in = torch.DoubleTensor({{10, 3 , 4, 5, 0 }, {10, 4, 5, 6, 0}})
print(test_in.T)
--print(input)
local output = mlp:forward(test_in)
print(output)

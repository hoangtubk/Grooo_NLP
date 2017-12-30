---
--- Created by tuhoangbk.
--- DateTime: 10/12/2017 01:30
---

include('pl_utils.lua')
include('tokenizer.lua')
class = require('pl.class')

local batch_size = 10
local max_dim = 120

---@class DataSet
DataSet = class()

local utils = Utils()
local tknz = Tokenizer()

---Init data set
---@param input_file string @file input.txt or target.txt
---@param file_t7_data string @file after save
function DataSet:_init(input_file)
    self.file_t7_data = input_file .. '.t7'
    self.input_file = input_file
    self:write_table_to_t7()
end

function DataSet:write_table_to_t7()
    local data = utils:read_file(self.input_file)
    if data == nil then
        print('Error create file t7 in Dataset')
        return false
    end
    local table_split = tknz: split_word_only(data, '\n')
    local file = torch.DiskFile(self.file_t7_data, "w")
    for i = 1, #table_split do
        file:writeObject(table_split[i])
    end
    file:close()
end

---load input sau mỗi lần gọi
---@param batch_size number @số lượng input mỗi seq
function DataSet:batches(batch_size)
    local file = torch.DiskFile(self.file_t7_data, "r")
    file:quiet()
    local done = false

    return function()
        if done then
            return
        end
        local inputSeqs,targetSeqs = {},{}
        local maxInputSeqLen,maxTargetOutputSeqLen = 0,0
        local input_seq = {}
        local input = torch.Tensor(batch_size, max_dim):zero()
        for i = 1, batch_size do
            local one_object = file:readObject()
            if one_object == nil then
                done = true
                file:close()
                return examples
            end
            local table_data_input = tknz:split_word_only(one_object, ' ')

            for j = 1, #table_data_input  do
                input[i][j] = table_data_input[j]
            end
        end
        return input
    end
end

--local test = DataSet('../output/all_input.txt', 'a.t7')
--local nextBatch = test:batches(10)
--print('begin print input')
--local encoderInputs--[[, decoderTargets]] = nextBatch()
--print(encoderInputs)

--encoderInputs = nextBatch()
--print(encoderInputs)
--local encoderInputs, decoderInputs--[[, decoderTargets]] = nextBatch()
--print(decoderInputs)
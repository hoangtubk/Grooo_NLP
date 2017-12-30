---
--- Created by tuhoangbk.
--- DateTime: 19/11/2017 20:26
---

utf8 = require 'lua-utf8'
class = require('pl.class')
Testing = class()

function Testing:_init()
    self.dict_size = 28 ---26 chữ cái, dấu cách và dấu '<' (>-unknown)
    self.hidden_size = 200
    self.batch_size = 256 -- số lượng câu trong mỗi input
    self.nout = 68 ---68 nhãn
    self.max_dim = 120 ---seq_len
    self.train_size = 57057
end

---@param model @Neuron Network
---@param table_inputs table @input
---@param table_targets table @target
---@param number_input number @number input sequence sentent
function Testing:test(model, next_batch_input_test, next_batch_target_test, is_use_cuda)
    local sum_class_predict = 0
    local sum_class_exactly = 0
    local precision = 0

    for i = 1, math.floor(self.train_size/self.batch_size) do
        local test_input = next_batch_input_test()
        local test_target = next_batch_target_test()
        ---use GPU with cuda
        if is_use_cuda then
            test_input = test_input:cuda()
            test_target = test_target:cuda()
        end
        local tb_index_pre, size = self:get_table_index_pred(test_target)
        ---so nhan can du doan
        sum_class_predict = sum_class_predict + size
        local output = model:forward(test_input)
        ---so nhan da du doan dung
        local value, index = torch.topk(output, 1, true)
        for i = 1, self.batch_size do
            for j = 1, self.nout do
                if tb_index_pre[i][j] == index[i][j][1]
                and test_target[i] ~= 0
                and test_target[i] ~= self.nout then
                    sum_class_exactly = sum_class_exactly + 1
                end
            end
        end
    end
    ---tinh do chinh xac
    --print('True   All')
    --print(sum_class_exactly, sum_class_predict)
    precision = sum_class_exactly/sum_class_predict
    print(sum_class_predict)

    return precision
end

--Lay ra index cua cac tu can du doan
---@param table_targets table
---@param input_from number
---@param input_to number
function Testing:get_table_index_pred(test_target)
    local size = 0
    local table_index = test_target:clone():zero()
    for i = 1,  self.batch_size do
        for j = 1, self.max_dim do
            if test_target[i][j] ~= 0
            and test_target[i][j] ~= 68 then
                size = size + 1
                table_index[i][j] = test_target[i][j]
            end
        end
    end

    return table_index, size
end

function Testing:test_string(model, content, is_use_cuda)
    local list_accent = 'àáảãạăắằẵặẳâầấậẫẩđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ'
    print(content)
    ---Convert to vector
    content = utf8.lower(content)
    local new_string = ''
    local tensor_test = torch.Tensor(1, self.max_dim):zero()
    for i = 1, #content do
        local dec_char = utf8.byte(content, i) - 96
        --- Nếu kí tự không phải a-z thì set = 28
        --- Nếu là space thì set = 27
        if dec_char < 1 or dec_char > 26 then
            if dec_char == -64 then
                dec_char = 27
            else
                dec_char = 28
            end
        end

        tensor_test[1][i] = dec_char
    end
    ---use GPU with cuda
    if is_use_cuda then
        tensor_test = tensor_test:cuda()
    end
    ---forward model & get index softmax
    local output = model:forward(tensor_test)
    local value, index = torch.topk(output, 1, true)
    --print(index)
    ---create new content
    for i = 1, #content do
        local index_char = index[1][i][1]
        --new_string = new_string .. utf8.sub(content,index_char, index_char)
        if index_char ~= 68 then
            new_string = new_string .. utf8.sub(list_accent, index_char, index_char)
        else
            new_string = new_string .. utf8.sub(content,i, i)
        end
    end
    return new_string
end
--local mlp_load = torch.load('../ModelCheckpoint/seqbrnn2.0_uncuda.t7')
--testing = Testing()
--testing:test()

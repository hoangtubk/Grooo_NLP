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
    self.batch_size = 10 -- số lượng câu trong mỗi input
    self.nout = 68 ---68 nhãn
    self.max_dim = 120 ---seq_len
end

---@param model @Neuron Network
---@param table_inputs table @input
---@param table_targets table @target
---@param number_input number @number input sequence sentent
function Testing:test(model, table_inputs, table_targets, test_from, test_to)
    local sum_class_predict = 0
    local sum_class_exactly = 0
    local precision = 0
    local tb_index_pre, size = self:get_table_index_pred(table_targets, test_from, test_to)
    ---Tinh so luong nhan can du doan
    for i = 1, size do
        sum_class_predict = sum_class_predict + tb_index_pre[i]
    end
    for step = test_from, test_to do
        local output = model:forward(table_inputs[step])
        --print(output)
        --assert(false)
        ---so nhan da du doan dung
        local value, index = torch.topk(output, 1, true)
        --print(index)
        --assert(false)
        for i = 1, self.batch_size do
            for j = 1, self.max_dim do
                if table_targets[step][i][j] == index[i][j][1]
                and table_targets[step][i][j] ~= 0
                and table_targets[step][i][j] ~= self.nout then
                    sum_class_exactly = sum_class_exactly + 1
                end
            end
        end
    end
    ---tinh do chinh xac
    print('True   All')
    print(sum_class_exactly, sum_class_predict)
    precision = sum_class_exactly/sum_class_predict

    return precision
end

--Lay ra index cua cac tu can du doan
---@param table_targets table
---@param input_from number
---@param input_to number
function Testing:get_table_index_pred(table_targets, input_from, input_to)
    local size = input_to - input_from + 1
    local index_tensor = torch.Tensor(input_to - input_from + 1):zero()
    for step = input_from, input_to do
        for i = 1, self.batch_size do
            for j = 1,self.max_dim do
                if table_targets[step][i][j] ~= 0
                and table_targets[step][i][j] ~= 68 then
                    index_tensor[step - input_from + 1] = index_tensor[step - input_from + 1] + 1
                end
            end
        end
    end
    return index_tensor, size
end

function Testing:test_string(model, content)
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

--testing = Testing()
--testing:test('a', 'mp3 tim kiem tin')

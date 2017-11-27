---
--- Created by tuhoangbk.
--- DateTime: 19/11/2017 20:26
---

class = require('pl.class')
Testing = class()

function Testing:_init()
    self.dict_size = 28 ---26 chữ cái, dấu cách và dấu '<' (>-unknown)
    self.hidden_size = 100
    self.batch_size = 10 -- số lượng câu trong mỗi input
    self.nout = 68 ---68 nhãn
    self.max_dim = 97 ---seq_len
end

---@param model @Neuron Network
---@param table_inputs table @input
---@param table_targets table @target
---@param number_input number @number input sequence sentent
function Testing:test(model, table_inputs, table_targets, test_from, test_to)
    local sum_class_predict = 0
    local sum_class_exactly = 0
    local precision = 0
    local tb_index_pre, size = self:get_table_index_pred_(table_targets, test_from, test_to)
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
    local table_index_predict = {}
    local tb_nil = {}
    for step = 1, input_to - input_from + 1 do
        table.insert(table_index_predict, step, tb_nil)
        for i = 1, self.batch_size do
            for j = 1, self.max_dim do
                if table_targets[step + input_from - 1][i][j] ~= 0
                and table_targets[step + input_from - 1][i][j] ~= 68 then
                    table.insert(table_index_predict[step], table_targets[step + input_from - 1][i][j])
                end
            end
        end
    end

    return table_index_predict
end

function Testing:get_table_index_pred_(table_targets, input_from, input_to)
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

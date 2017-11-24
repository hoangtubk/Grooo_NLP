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
function Testing:test(model, table_inputs, table_targets, number_input)
    local sum_class_predict = 0
    local sum_class_exactly = 0
    local precision = 0
    for step = 0, number_input do
        local step = 1
        local output = model:forward(table_inputs[step])
        local new_output = output:clone()
        ---Tinh so luong nhan can du doan
        for i = 1, self.batch_size do
            for j = 1, self.max_dim do
                for k = 1, self.nout do
                    if new_output[i][j][k] ~= 0 then
                        sum_class_predict = sum_class_predict + 1
                    end
                end
            end
        end
        ---so nhan da du doan dung
        local value, index = torch.topk(output, 1, true)
        for i = 1, self.batch_size do
            for j = 1, 97 do
                if table_targets[step][i][j] == index[i][j][1] and table_targets[step][i][j] ~= 0 then
                    sum_class_exactly = sum_class_exactly + 1
                end
            end
        end
    end
    ---tinh do chinh xac
    precision = sum_class_exactly/sum_class_predict

    return precision
end

---
--- Created by tuhoangbk.
--- DateTime: 19/11/2017 20:26
---

require('rnn')
require('nn')
class = require('pl.class')

Model = class()

function Model:_init()

end

function Model:build_brnn()
    local dict_size = 28 ---26 chữ cái, dấu cách và dấu '<' (>)
    local hidden_size = 19
    local batch_size = 10 -- số lượng câu trong mỗi input
    local nout = 68 ---68 nhãn

    local lt = nn.LookupTableMaskZero(dict_size, hidden_size)
    local brnn =(nn.SeqBRNN(hidden_size, hidden_size))
    local linear = nn.Sequencer(nn.MaskZero(nn.Linear(hidden_size, nout),1))
    local logsoftmax = nn.Sequencer(nn.MaskZero(nn.LogSoftMax(),1))
    ---build model
    local rnn = nn.Sequential()
    rnn:add(lt)
    rnn:add(brnn)
    rnn:add(linear)
    rnn:add(logsoftmax)
    --rnn = nn.Sequencer(rnn)
    return rnn
end

function Model:get_criterion_func()
    return self.criterion
end

--model = Model()
--x = model:build_brnn()
--local inp = torch.LongTensor(10, 100):zero()
--for i = 1, 10 do
--    inp[i] = torch.LongTensor({20, 9, 13, 27, 3, 8, 15, 27, 20, 15, 27, 20, 8, 15, 14, 7, 27, 20, 9, 14, 27, 68, 21, 68, 27, 68, 21, 68, 27, 68, 21, 68, 27, 3, 21, 1, 27, 26, 9, 14, 7, 68, 13, 16, 68, 68, 3, 15, 13, 27, 22, 15, 9})
--end
--print(inp)
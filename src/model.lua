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
    local dict_size = 27 ---26 chữ cái và dấu cách
    local ndim = 512
    local batch_size = 10
    local seq_len = 10
    local nout = 68 ---68 nhãn

    local lt = nn.LookupTableMaskZero(dict_size, ndim)
    local brnn = nn.SeqBRNN(seq_len, batch_size, ndim)
    local tanh = nn.Tanh()
    local linear1 = nn.Linear(ndim, ndim)
    local linear2 = nn.Linear(ndim, nout)
    local rnn = nn.Sequential()
    local logsoftmax = nn.LogSoftMax()

    rnn:add(lt)
    --rnn:add(brnn)
    --rnn:add(tanh)
    --rnn:add(linear1)
    --rnn:add(tanh)
    --rnn:add(linear2)
    --rnn:add(logsoftmax)

    return rnn
end

--model = Model()
--x = model:build_brnn()
--local inp = torch.LongTensor(10, 100):zero()
--for i = 1, 10 do
--    inp[i] = torch.LongTensor({20, 9, 13, 27, 3, 8, 15, 27, 20, 15, 27, 20, 8, 15, 14, 7, 27, 20, 9, 14, 27, 68, 21, 68, 27, 68, 21, 68, 27, 68, 21, 68, 27, 3, 21, 1, 27, 26, 9, 14, 7, 68, 13, 16, 68, 68, 3, 15, 13, 27, 22, 15, 9})
--end
--print(inp)


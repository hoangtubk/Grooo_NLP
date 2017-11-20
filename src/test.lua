require('nn')
require('rnn')

batchSize = 8
rho = 5 -- sequence length
hiddenSize = 7
nIndex = 10
lr = 0.1

local fwd = nn.Recurrent(
   hiddenSize, nn.LookupTable(nIndex, hiddenSize), 
   nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(), 
   rho
)
pritn(fwd)

local bwd = fwd:clone()

local merge = nn.JoinTable(1, 1) 

local brnn = nn.BiSequencerLM(fwd, bwd, merge)

local rnn = nn.Sequential()
:add(brnn) 
:add(nn.Sequencer(nn.Linear(hiddenSize*2, nIndex))) -- times two due to JoinTable
:add(nn.Sequencer(nn.LogSoftMax()))

print(rnn)

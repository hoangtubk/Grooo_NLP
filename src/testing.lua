---
--- Created by tuhoangbk.
--- DateTime: 19/11/2017 20:26
---

class = require('pl.class')
Testing = class()

function Testing:_init()

end

function Testing:test(model, table_inputs, table_targets)
    --for step = 0, #table_inputs do
    local step = 1
    local output = model:forward(table_inputs[step])

    local value, index = torch.topk(output, 1, true)
    print(index)
    print(table_targets[step])
    local count = 0
    for i = 1, 10 do
        for j = 1, 97 do
            if table_targets[step][i][j] == index[i][j][1] then
                count = count + 1
            end
        end
    end
    print(count)
    assert(false)

    return
    --end
end

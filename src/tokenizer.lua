---
--- Created by tuhoangbk.
--- DateTime: 12/11/2017 12:30
---
local utf8 = require 'lua-utf8'
utils = require('pl.utils')
class = require('pl.class')

Tokenizer = class()

function Tokenizer:_init()
    self._tokens = {}
end

---@param str string
---@return table list string after split
function Tokenizer:split_word(inputstr, sep)
    if sep == nil then
        sep = "%s"
    end
    local t={} ; i=1
    for str in utf8.gmatch(inputstr, "([^"..sep.."]+)") do
        str = utf8:lower(str):gsub("%p", ""):gsub("”", ""):gsub("“", "")
        if tonumber(str) ~= nil then
            str = '<number>'
        end
        t[i] = str
        i = i + 1
    end
    self._tokens = t
    return t
end

function Tokenizer:split_word_only(inputstr, sep)
    if sep == nil then
        sep = "%s"
    end
    local t={} ; i=1
    for str in utf8.gmatch(inputstr, "([^"..sep.."]+)") do
        t[i] = str
        i = i + 1
    end

    return t
end
function Tokenizer:get_list_token()

    return self._tokens
end

--tknz = Tokenizer()
--xx = tknz:split_word('hoang anh tu', ' ')
--z = tknz:get_list_token()
--print(z)
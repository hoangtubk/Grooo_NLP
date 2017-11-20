---
--- Created by tuhoangbk.
--- DateTime: 10/11/2017 13:16
---

utils = require('pl.utils')
class = require('pl.class')

---@class
Utils = class()

---@param
function Utils:_init()

end

---@param file_name string pathfile
---@return string content of file
function Utils:read_file(file_name)

    return utils.readfile(file_name)
end

---return the contents of a file as a list of lines
---@param filename string pathfile
---@return table all lines
function Utils:read_lines(filename)

    return utils.readlines(filename)
end

---@param file_name string
---@param str string
---@return boolean true or nil if success
---@return string error message
function Utils:write_file(file_name, str)

    return utils.writefile(file_name, str, false)
end

---@param path string path of dir
---@return table list file name in path
function Utils:find_file(path)
    local i, t, popen = 0, {}, io.popen
    local pfile = popen('ls -a "'.. path ..'"')
    for filename in pfile:lines() do
        i = i + 1
        t[i] = filename
    end
    pfile:close()
    table.remove(t, 1)
    table.remove(t, 1)
    return t
end

--local test = Utils()

--a = test:read_file('xxx.txt')
--b = test:write_file('xxx.txt', 'hoang lung')
--m = test:find_file('/home/tuhoangbk/20171/Project_NLP/src')
--print(m)

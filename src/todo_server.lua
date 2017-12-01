#!/usr/bin/env lua

--[[

A simple todo-list server example.

This example requires the restserver-xavante rock to run.

A fun example session:

curl localhost:8080/todo
curl -v -H "Content-Type: application/json" -X POST -d '{ "task": "Clean bedroom" }' http://localhost:8080/todo
curl -v localhost:8080/todo
curl -v -H "Content-Type: application/json" -X POST -d '{ "task": "Groceries" }' http://localhost:8080/todo
curl -v localhost:8080/todo
curl -v localhost:8080/todo/2/status
curl -v localhost:8080/todo/2/done
curl -v localhost:8080/todo/2/status
curl -v localhost:8080/todo/9/status
curl -v -H "Content-Type: application/json" -X DELETE http://localhost:8080/todo/2
curl -v localhost:8080/todo

]]
require('rnn')
require('nn')
include('testing.lua')

local testing = Testing()
---Server
local restserver = require("restserver")
local server = restserver:new():port(8080)
local mlp_trained = torch.load('seqbnn.t7')
server:add_resource("tuha", {
   {
      method = "GET",
      path = "{id:.*}",
      produces = "application/json",
      handler = function(_, content)
         print('Input: ' .. content['content'])
         --local output = testing:test_string(mlp_trained, content['content'])
         --print('Output: ' .. output )
         local mess = 'Messange'
         return restserver.response():status(200):entity(output)
      end,
   },
})

-- This loads the restserver.xavante plugin
server:enable("restserver.xavante"):start()


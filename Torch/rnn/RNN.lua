--------------------------------------------------------------------------------
-- Simple RNN block
--
-- Written by: Abhishek Chaurasia
--------------------------------------------------------------------------------

local RNN = {}

--[[
                    +-----------+
                    |           |
                    |   +----+  |
                    V   |    |--+
              +--->(+)->| h1 |
              |x1       |    |------+
              |         +----+      |
              |                     |
              |                     |
              |                     |
              |     +-----------+   |
              |     |           |   |
      +-----+ |     |   +----+  |   |     +---+
      |   1 | |     V   |    |--+   +---->|   |
      | x:2 +-+--->(+)->| h2 |   +------->| y |
      |   3 | |x2       |    |---+  +---->|   |
      +-----+ |         +----+      |     +---+
              |                     |
              |                     |
              |                     |
              |     +-----------+   |
              |     |           |   |
              |     |   +----+  |   |
              |     V   |    |--+   |
              +--->(+)->| h3 |      |
               x3       |    |------+
                        +----+
--]]

-- n   : # of inputs
-- d   : # of neurons in hidden layer
-- nHL : # of hidden layers
-- K   : # of output neurons

-- Returns a simple RNN model
function RNN.getPrototype(n, d, nHL, K)
   local inputs = {}
   table.insert(inputs, nn.Identity()())       -- input X
   for j = 1, nHL do
      table.insert(inputs, nn.Identity()())    -- previous states h[j]
   end

   local x, nIn
   local outputs = {}
   for j = 1, nHL do
      if j == 1 then
         x = inputs[j]:annotate{name = 'x[t]',
                       graphAttributes = {
                       style = 'filled',
                       fillcolor = 'moccasin'}}
         nIn = n
      else
         x = outputs[j-1]
         nIn = d
      end

      local hPrev = inputs[j+1]:annotate{name = 'h^('..j..')[t-1]',
                                graphAttributes = {
                                style = 'filled',
                                fillcolor = 'lightpink'}}

      -- Concat input with previous state
      local nextH = ({x, hPrev} - nn.JoinTable(1) - nn.Linear(nIn + d, d) - nn.Tanh())
                    :annotate{name = 'h^('..j..')[t]',
                     graphAttributes = {
                     style = 'filled',
                     fillcolor = 'skyblue'}}

      table.insert(outputs, nextH)
   end

   local logsoft = (outputs[#outputs] - nn.Linear(d, K) - nn.LogSoftMax())
                   :annotate{name = 'y\'[t]',
                    graphAttributes = {
                    style = 'filled',
                    fillcolor = 'seagreen1'}}
   table.insert(outputs, logsoft)

   -- Output is table with {h, prediction}
   return nn.gModule(inputs, outputs)
end

return RNN

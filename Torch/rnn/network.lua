--------------------------------------------------------------------------------
-- A simple building block for RNN/LSTMs
--
-- Written by: Abhishek Chaurasia
--------------------------------------------------------------------------------

local network = {}
nngraph.setDebug(true)
-- n   : # of inputs
-- d   : # of neurons in hidden layer
-- nHL : # of hidden layers
-- K   : # of output neurons

-- Links all the prototypes, given the # of sequences
function network.getModel(n, d, nHL, K, T, mode)
   local prototype
   if mode == 'RNN' then
      local RNN = require 'RNN'
      prototype = RNN.getPrototype(n, d, nHL, K)
   elseif mode == 'GRU' then
      local GRU = require 'GRU'
      prototype = GRU.getPrototype(n, d, nHL, K)
   elseif mode == 'FW' then
      local FW = require 'FW'
      prototype = FW.getPrototype(n, d, nHL, K)
   else
      print("Invalid model type. Available options: (RNN/GRU)")
   end

   local clones = {}
   for i = 1, T do
      clones[i] = prototype:clone('weight', 'bias', 'gradWeight', 'gradBias')
   end

   local inputSequence = nn.Identity()()        -- Input sequence
   local H0 = {}                                -- Initial states of hidden layers
   local H = {}                                 -- Intermediate states
   local outputs = {}

   -- Linking initial states to intermediate states
   for l = 1, nHL do
      table.insert(H0, nn.Identity()())
      H0[#H0]:annotate{
         name = 'h^('..l..')[0]', graphAttributes = {
            style = 'filled', fillcolor = 'lightpink'
         }
      }
      table.insert(H, H0[#H0])
      if mode == 'FW' then
         table.insert(H0, nn.Identity()())
         H0[#H0]:annotate{
            name = 'A^('..l..')[0]', graphAttributes = {
               style = 'filled', fillcolor = 'plum'
            }
         }
         table.insert(H, H0[#H0])
      end
   end

   local splitInput = inputSequence - nn.SplitTable(1)

   for i = 1, T do
      local x = (splitInput - nn.SelectTable(i))
                :annotate{name = 'x['..i..']',
                 graphAttributes = {
                 style = 'filled',
                 fillcolor = 'moccasin'}}

      local tempStates = ({x, table.unpack(H)} - clones[i])
                         :annotate{name = mode .. '['..i..']',
                          graphAttributes = {
                          style = 'filled',
                          fillcolor = 'skyblue'}}

      local predIdx = nHL + 1
      if mode == 'FW' then predIdx = 2 * nHL + 1 end
      outputs[i] = (tempStates - nn.SelectTable(predIdx))  -- Prediction
                   :annotate{name = 'y\'['..i..']',
                    graphAttributes = {
                    style = 'filled',
                    fillcolor = 'seagreen1'}}

      if i < T then
         local j = 0
         for l = 1, nHL do                         -- State values passed to next sequence
            j = j + 1
            H[j] = (tempStates - nn.SelectTable(j)):annotate{
               name = 'h^('..l..')['..i..']', graphAttributes = {
                  style = 'filled', fillcolor = 'lightpink'}}
            if mode == 'FW' then
               j = j + 1
               H[j] = (tempStates - nn.SelectTable(j)):annotate{
                  name = 'A^('..l..')['..i..']', graphAttributes = {
                     style = 'filled', fillcolor = 'plum'}}
            end
         end
      else
         local j = 0
         for l = 1, nHL do                         -- State values passed to next sequence
            j = j + 1
            outputs[T + j] = (tempStates - nn.SelectTable(j)):annotate{
               name = 'h^('..l..')['..i..']', graphAttributes = {
                  style = 'filled', fillcolor = 'lightpink'}}
            if mode == 'FW' then
               j = j + 1
               outputs[T + j] = (tempStates - nn.SelectTable(j)):annotate{
                  name = 'A^('..l..')['..i..']', graphAttributes = {
                     style = 'filled', fillcolor = 'plum'}}
            end
         end
      end
   end

   -- Output is table of {Predictions, Hidden states of last sequence}
   local g = nn.gModule({inputSequence, table.unpack(H0)}, outputs)

   return g, clones[1]
end

return network

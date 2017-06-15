require 'TicTacToeUtil'

function TicTacToeRandomAgent(numActions)
  local agent = {}
  local util = TicTacToeUtil()
  
  function agent.chooseAction(state)
    local action
    while ( true ) do
      action = math.random(1, numActions)            
      if ( util.isActionable(state, action) )  then                
        break
      end      
    end
    return action
  end

  return agent
end
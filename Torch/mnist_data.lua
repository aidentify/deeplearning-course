--[[
MNIST data loader

@author socurites@aidentify.io
]]--

require 'paths'


function loadDataSet(path, file_name)
    local dataFile = paths.concat(path, file_name)
    local dataSet = torch.load(dataFile, 'ascii')
    dataSet.data = dataSet.data:double()      -- X
    dataSet.labels = dataSet.labels:double()  -- target Y

    ---- normalize training data
    local mean = dataSet.data[{ {}, {1}, {}, {}  }]:mean()
    dataSet.data[{ {}, {1}, {}, {}  }]:add(-mean)
    local stdv = dataSet.data[{ {}, {1}, {}, {}  }]:std()
    dataSet.data[{ {}, {1}, {}, {}  }]:div(stdv)

    return {dataSet.labels, dataSet.data}
end
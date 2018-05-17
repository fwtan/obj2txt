require 'nn'
require 'nngraph'
require 'misc.net_utils'

local Attention = {}

function Align(input_size, rnn_size)
   local inputs = {}
   -- first input sequence embedding matrix
   table.insert(inputs, nn.Identity()())
   -- second input hidden state vector expanded
   table.insert(inputs, nn.Identity()())
   local i2h = nn.Linear(input_size, rnn_size)(inputs[1])
   local h2h = nn.Linear(rnn_size, rnn_size)(inputs[2])
   local input_sum = nn.CAddTable()({i2h, h2h})
   local input_sum_transform = nn.Tanh()(input_sum)
   local score = nn.Linear(rnn_size, 1)(input_sum_transform)
   local output = {score}
   return nn.gModule(inputs, output)
end

function Attention.attend(seq_length, embedding_size, rnn_size)
   local inputs = {}
   table.insert(inputs, nn.Identity()())
   table.insert(inputs, nn.Identity()())
   local seq_embeddings = inputs[1] -- batch_size x seq_length x embedding_size
   local hidden_states = inputs[2]  -- batch_size x rnn_size
   local seq_embeddings_unravel = nn.View(-1, embedding_size)(seq_embeddings) -- (batch_size*seq_length) x embedding_size
   local hidden_states_expand = nn.FeatExpander(seq_length)(hidden_states) -- (batch_size*seq_length) x embedding_size
   local scores = Align(embedding_size, rnn_size)({seq_embeddings_unravel, hidden_states_expand}) -- (batch_size*seq_length) x 1
   local scores_ravel = nn.View(-1, seq_length)(scores)   -- batch_size x seq_length
   local probs = nn.SoftMax()(scores_ravel)            -- batch_size x seq_length
   local probs_column = nn.View(-1, seq_length, 1)(probs) -- batch_size x seq_length x 1
   local context = nn.MM(true, false)({seq_embeddings, probs_column}) -- batch_size x embedding_size x 1
   context = nn.Squeeze()(context) -- batch_size x embedding_size
   return nn.gModule(inputs, {context})
end



function Attention.test_attention()
   local atten = attend(seq_length, embedding_size, rnn_size)
   local batch_size = 10
   local seq_length = 3
   local embedding_size = 4
   local rnn_size = 5
   local seq_embeddings = torch.randn(batch_size, seq_length, embedding_size)
   local hidden_states = torch.randn(batch_size, rnn_size)
   local context = atten:forward({seq_embeddings, hidden_states})
   local dinputs = atten:backward({seq_embeddings, hidden_states}, context)
   local dseq_embeddings = dinputs[1]
   local dhidden_states = dinputs[2]
   print(dseq_embeddings:size())
   print(dhidden_states:size())
end


function Attention.attention_run()
   batch_size = 10
   seq_length = 3
   embedding_size = 4
   rnn_size = 5
   seq_embeddings = torch.randn(batch_size, seq_length, embedding_size)
   hidden_states = torch.randn(batch_size, rnn_size)
   seq_embeddings_unravel = nn.Unravel():forward(seq_embeddings) -- (batch_size*seq_length) x embedding_size
   hidden_states_expand = nn.FeatExpander(seq_length):forward(hidden_states) -- (batch_size*seq_length) x embedding_size
   scores = Align.align(embedding_size, rnn_size):forward({seq_embeddings_unravel, hidden_states_expand}) -- (batch_size x seq_length) x 1
   scores = nn.Squeeze():forward(scores)
   scores_ravel = nn.Ravel(seq_length):forward(scores)   -- batch_size x seq_length
   probs = nn.SoftMax():forward(scores_ravel)            -- batch_size x seq_length
   probs_column = nn.Unsqueeze(3):forward(probs) -- batch_size x seq_length x 1
   context = nn.MM(true, false):forward({seq_embeddings, probs_column}) -- batch_size x embedding_size x 1
   context = nn.Squeeze():forward(context) -- batch_size x embedding_size
end

-- function Align.test()
--    local rnn_size = 5
--    local embedding_size = 4
--    local seq_length
--    local seq_embedding = torch.randn(seq_length, embedding_size)
--    local hidden_state = torch.randn(seq_length, rnn_size)
--    Align.align(embedding_size, rnn_size):forward({seq_embedding, hidden_state})
-- end

function draw_graph()
   inputs = {}
   table.insert(inputs, nn.Identity()())
   table.insert(inputs, nn.Identity()())
   seq_embeddings = inputs[1]:annotate{name = 'Sequence Embeddings'}
   hidden_states = inputs[2]:annotate{name = 'Hidden States'}
   seq_embeddings_unravel = nn.Unravel()(seq_embeddings):annotate{name = 'Sequence Embeddings Unravel'} -- (batch_size*seq_length) x embedding_size
   hidden_states_expand = nn.FeatExpander(seq_length)(hidden_states):annotate{name = 'Hidden States Expand'} -- (batch_size*seq_length) x embedding_size
   scores = Align.align(embedding_size, rnn_size)({seq_embeddings_unravel, hidden_states_expand}):annotate{
      name = 'Alignment Model', description = 'Compute Relevent Scores',
      graphAttributes = {color = 'blue', fontcolor = 'green'}
                                                                                                          } -- (batch_size x seq_length) x 1
   scores = nn.Squeeze()(scores):annotate{name = 'Relevent Scores'}
   scores_ravel = nn.Ravel(seq_length)(scores):annotate{name = 'Relevent Scores'}   -- batch_size x seq_length
   probs = nn.SoftMax()(scores_ravel):annotate{name = 'Relevent Probabilities'}            -- batch_size x seq_length
   probs_column = nn.Unsqueeze(3)(probs):annotate{name = 'Relevent Probabilities'} -- batch_size x seq_length x 1
   context = nn.MM(true, false)({seq_embeddings, probs_column}):annotate{name = 'Context Vectors'} -- batch_size x embedding_size x 1
   context = nn.Squeeze()(context):annotate{name = 'Context Vectors'} -- batch_size x embedding_size
   g = nn.gModule(inputs, {context})

   batch_size = 10
   seq_length = 3
   embedding_size = 4
   rnn_size = 5

   seq_embeddings = torch.randn(batch_size, seq_length, embedding_size)
   hidden_states = torch.randn(batch_size, rnn_size)

   ddata = g:forward({seq_embeddings, hidden_states})
   g:backward({seq_embeddings, hidden_states}, ddata)

   graph.dot(g.fg, 'Forward Graph', '/tmp/fg')
   graph.dot(g.bg, 'Backward Graph', '/tmp/bg')
end

return Attention

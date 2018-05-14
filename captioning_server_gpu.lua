
require 'torch'
require 'nn'
require 'nngraph'
-- exotic things
--require 'loadcaffe'
-- local imports
local utils = require 'misc.utils'
--require 'misc.DataLoader'
require 'misc.LanguageModel'
local net_utils = require 'misc.net_utils'
require 'misc.optim_updates'
require 'misc.LayoutEncoder'
require 'misc.LayoutEncoderLocation'
require 'misc.LayoutEncoderAttention'
require 'misc.LanguageModelAttention'
require 'misc.LayoutEncoderLocationAttention'
require 'paths'

local c = require 'trepl.colorize'
local tablex = require 'pl.tablex'
local cjson = require 'cjson'
require 'cunn'
require 'cudnn'

local async = require 'async'
require('pl.text').format_operator()

-- COCO_label_dict = {["person"]=1, ["bicycle"]=2, ["car"]=3, ["motorcycle"]=4, ["airplane"]=5, ["bus"]=6, ["train"]=7, ["truck"]=8, ["boat"]=9, ["traffic light"]=10, ["fire hydrant"]=11, ["stop sign"]=13, ["parking meter"]=14, ["bench"]=15, ["bird"]=16, ["cat"]=17, ["dog"]=18, ["horse"]=19, ["sheep"]=20, ["cow"]=21, ["elephant"]=22, ["bear"]=23, ["zebra"]=24, ["giraffe"]=25, ["backpack"]=27, ["umbrella"]=28, ["handbag"]=31, ["tie"]=32, ["suitcase"]=33, ["frisbee"]=34, ["skis"]=35, ["snowboard"]=36, ["sports ball"]=37, ["kite"]=38, ["baseball bat"]=39, ["baseball glove"]=40, ["skateboard"]=41, ["surfboard"]=42, ["tennis racket"]=43, ["bottle"]=44, ["wine glass"]=46, ["cup"]=47, ["fork"]=48, ["knife"]=49, ["spoon"]=50, ["bowl"]=51, ["banana"]=52, ["apple"]=53, ["sandwich"]=54, ["orange"]=55, ["broccoli"]=56, ["carrot"]=57, ["hot dog"]=58, ["pizza"]=59, ["donut"]=60, ["cake"]=61, ["chair"]=62, ["couch"]=63, ["potted plant"]=64, ["bed"]=65, ["dining table"]=67, ["toilet"]=70, ["tv"]=72, ["laptop"]=73, ["mouse"]=74, ["remote"]=75, ["keyboard"]=76, ["cell phone"]=77, ["microwave"]=78, ["oven"]=79, ["toaster"]=80, ["sink"]=81, ["refrigerator"]=82, ["book"]=84, ["clock"]=85, ["vase"]=86, ["scissors"]=87, ["teddy bear"]=88, ["hair drier"]=89, ["toothbrush"]=90}

-- COCO_label_dict = {["person"]=1, ["bicycle"]=2, ["car"]=3, ["motorcycle"]=4, ["airplane"]=5, ["bus"]=6, ["train"]=7, ["truck"]=8, ["boat"]=9, ["traffic light"]=10, ["fire hydrant"]=11, ["stop sign"]=13, ["parking meter"]=14, ["bench"]=15, ["bird"]=16, ["cat"]=17, ["dog"]=18, ["horse"]=19, ["sheep"]=20, ["cow"]=21, ["elephant"]=22, ["bear"]=23, ["zebra"]=24, ["giraffe"]=25, ["backpack"]=27, ["umbrella"]=28, ["handbag"]=31, ["tie"]=32, ["suitcase"]=33, ["frisbee"]=34, ["skis"]=35, ["snowboard"]=36, ["sports ball"]=37, ["kite"]=38, ["baseball bat"]=39, ["baseball glove"]=40, ["skateboard"]=41, ["surfboard"]=42, ["tennis racket"]=43, ["bottle"]=44, ["wine glass"]=46, ["cup"]=47, ["fork"]=48, ["knife"]=49, ["spoon"]=50, ["bowl"]=51, ["banana"]=52, ["apple"]=53, ["sandwich"]=54, ["orange"]=55, ["broccoli"]=56, ["carrot"]=57, ["hot dog"]=58, ["pizza"]=59, ["donut"]=60, ["cake"]=61, ["chair"]=62, ["couch"]=63, ["potted plant"]=64, ["bed"]=65, ["dining table"]=67, ["toilet"]=70, ["tv"]=72, ["laptop"]=73, ["mouse"]=74, ["remote"]=75, ["keyboard"]=76, ["cell phone"]=77, ["microwave"]=78, ["oven"]=79, ["toaster"]=80, ["sink"]=81, ["refrigerator"]=82, ["book"]=84, ["clock"]=85, ["vase"]=86, ["scissors"]=87, ["teddy bear"]=88, ["hair drier"]=89, ["toothbrush"]=90}

COCO_label_dict = {["person"]=1, ["bicycle"]=2, ["car"]=3, ["motorbike"]=4, ["aeroplane"]=5, ["bus"]=6, ["train"]=7, ["truck"]=8, ["boat"]=9, ["traffic light"]=10, ["fire hydrant"]=11, ["stop sign"]=12, ["parking meter"]=13, ["bench"]=14, ["bird"]=15, ["cat"]=16, ["dog"]=17, ["horse"]=18, ["sheep"]=19, ["cow"]=20, ["elephant"]=21, ["bear"]=22, ["zebra"]=23, ["giraffe"]=24, ["backpack"]=25, ["umbrella"]=26, ["handbag"]=27, ["tie"]=28, ["suitcase"]=29, ["frisbee"]=30, ["skis"]=31, ["snowboard"]=32, ["sports ball"]=33, ["kite"]=34, ["baseball bat"]=35, ["baseball glove"]=36, ["skateboard"]=37, ["surfboard"]=38, ["tennis racket"]=39, ["bottle"]=40, ["wine glass"]=41, ["cup"]=42, ["fork"]=43, ["knife"]=44, ["spoon"]=45, ["bowl"]=46, ["banana"]=47, ["apple"]=48, ["sandwich"]=49, ["orange"]=50, ["broccoli"]=51, ["carrot"]=52, ["hot dog"]=53, ["pizza"]=54, ["donut"]=55, ["cake"]=56, ["chair"]=57, ["sofa"]=58, ["pottedplant"]=59, ["bed"]=60, ["diningtable"]=61, ["toilet"]=62, ["tvmonitor"]=63, ["laptop"]=64, ["mouse"]=65, ["remote"]=66, ["keyboard"]=67, ["cell phone"]=68, ["microwave"]=69, ["oven"]=70, ["toaster"]=71, ["sink"]=72, ["refrigerator"]=73, ["book"]=74, ["clock"]=75, ["vase"]=76, ["scissors"]=77, ["teddy bear"]=78, ["hair drier"]=79, ["toothbrush"]=80}
opt = {}
opt.start_from = 'model_idobjname_location.t7'
opt.port = 8008

opt.seed = 123
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')

-- local loader = DataLoader{h5_file = "/data/xy4cm/cocotalk_withcats_withpos_yolo_coco_full.h5", json_file = "/data/xy4cm/cocotalk_coco_full_convert.json"}
-- local vocab = loader:getVocab()
vocab = torch.load('vocab.t7')

print('initializing weights from ' .. opt.start_from)
local loaded_checkpoint = torch.load(opt.start_from)
protos = loaded_checkpoint.protos
net_utils.unsanitize_gradients(protos.cnn)
local lm_modules = protos.lm:getModulesList()
for k,v in pairs(lm_modules) do net_utils.unsanitize_gradients(v) end
local layout_encoder_modules = protos.layout_encoder:getModulesList()
for k,v in pairs(layout_encoder_modules) do net_utils.unsanitize_gradients(v) end

--for k,v in pairs(protos) do v=v:float() end

protos.cnn:evaluate()
protos.lm:evaluate()
protos.layout_encoder:evaluate()

-- async.http.listen('http://0.0.0.0:' .. opt.port, function(req,res)
--    print(req.body)
--    local status, annotations = pcall(cjson.decode, req.body)
--    if not status then return end
--    -- print(#annotations)
--    -- batch size is 1
--    -- local bbox_coords = torch.FloatTensor(1,#annotations,4):fill(0)
--    local bbox_coords = torch.FloatTensor(1,96,4):fill(0)
--    local full_category = torch.LongTensor(1,96):fill(0)
--    for k, v in pairs(annotations) do
--       bbox_coords[{1,k,1}] = v['left']
--       bbox_coords[{1,k,2}] = v['top']
--       bbox_coords[{1,k,3}] = v['width']
--       bbox_coords[{1,k,4}] = v['height']
--       full_category[{1,k}] = COCO_label_dict[v['label']]
--    end
--    bbox_coords:div(604.0)

--    local bbox_coords_T = bbox_coords:transpose(1,2):cuda()
--    local data_full_category_T = full_category:transpose(1,2):cuda()
--    local layout_feats = protos.layout_encoder:forward({ data_full_category_T, bbox_coords_T })
--    local sample_opts = { sample_max = 1, beam_size = 2, temperature = 1.0 }
--    local seq = protos.lm:sample(layout_feats, sample_opts)
--    local sents = net_utils.decode_sequence(vocab, seq)
--    print(sents[1])
--    res(sents[1], {['Content-Type']='text/html', ['Access-Control-Allow-Origin']='*'})
--    return
      

--    -- if (req.body.path == nil and req.body.source == nil) then 
--    -- 	res("LoL", {['Content-Type']='text/html', ['Access-Control-Allow-Origin']='*'})
--    --      return
--    -- end

--    -- local input = nil
--    -- if (req.body.path) then
--    --    local filename = string.match(req.body.path.data, "[a-zA-Z0-9]+.[a-zA-Z]+$")
--    --    input = torch.Tensor(1, 1, 32, 128)
--    --    status, code = pcall(image.load,filename, 1, nil)
--    --    if not status then
--    -- 	res("LoL", {['Content-Type']='text/html', ['Access-Control-Allow-Origin']='*'})
--    --      return
--    --    end
--    --     input[1] = code
--    -- else
--    --    input = gm.Image():fromString(req.body['source'].data):toTensor('float','I','DHW')
--    --    input = input:float()
--    --  -- im = torch.FloatTensor(48, math.floor(scale*img:size(2)))
--    --  -- image.scale(im, img)
--    --    if input:size(2) ~= 48 then
--    --      new_width = 48 * input:size(3) / input:size(2)
--    --    	input = image.scale(input, new_width, 48)
--    --    end
--    --    --image.save('input.jpg', input)
--    --    local result = predict_seq(input[1])
--    --    --result = '{"boxes":[{"x":1,"y":1,"width":32, "height":32}]}'
--    --    print(result)
--    --    res(result, {['Content-Type']='text/html', ['Access-Control-Allow-Origin']='*'})
--    --    --image.save('kkk.png', input)
--    -- end

--    -- -- reference
--    -- -- https://github.com/benglard/waffle/blob/master/waffle/request.lua#L34
--    -- -- https://github.com/clementfarabet/graphicsmagick on image:load(tensor,colorSpace,dimensions)
--    -- local prediction = predict(input)
--    -- print(prediction)
--    -- res(prediction, {['Content-Type']='text/html', ['Access-Control-Allow-Origin']='*'})

--    -- async.fs.writeFile(req.body['source'].filename, req.body['source'].data, function()
--    --                       async.process.spawn('open', {req.body['source'].filename}, function()
--    --                                                               end))

--    -- local resp
--    -- if req.url.path == '/test' then
--    --    resp  = [[
--    --    <p>You requested route /test</p>
--    --    ]]
--    -- else
--    --    -- Produce a random story:
--    --    resp = [[
--    --    <h1>From my server</h1>
--    --    <p>It's working!<p>
--    --    <p>Randomly generated number: ${number}</p>
--    --    <p>A variable in the global scope: ${ret}</p>
--    --    ]] % {
--    --       number = math.random(),
--    --       ret = ret
--    --    }
--    -- end

--    -- -- res(resp, {['Content-Type']='text/html'})
   
-- end)

-- print('server listening to port ' .. opt.port)

-- async.go()


require 'torch'
require 'nn'
require 'nngraph'
local utils = require 'misc.utils'
require 'misc.LanguageModel'
require 'misc.LanguageEncoder'
require 'misc.LanguageEncoderWithSpatial'
require 'misc.LanguageEncoderWithAttention'
require 'misc.LanguageModelWithAttention'
require 'misc.LanguageEncoderWithSpatialWithAttention'
local net_utils = require 'misc.net_utils'
require 'misc.optim_updates'
require 'paths'
require 'cutorch'
require 'cunn'
require 'cudnn'

require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
require 'xlua'
require 'optim'
local c = require 'trepl.colorize'
local tablex = require 'pl.tablex'
local cjson = require 'cjson'

require('pl.text').format_operator()

-- COCO_label_dict = {
--     ["person"]=1, ["bicycle"]=2, ["car"]=3, ["motorcycle"]=4, ["airplane"]=5, 
--     ["bus"]=6, ["train"]=7, ["truck"]=8, ["boat"]=9, ["traffic light"]=10, 
--     ["fire hydrant"]=11, ["stop sign"]=13, ["parking meter"]=14, ["bench"]=15, 
--     ["bird"]=16, ["cat"]=17, ["dog"]=18, ["horse"]=19, ["sheep"]=20, 
--     ["cow"]=21, ["elephant"]=22, ["bear"]=23, ["zebra"]=24, ["giraffe"]=25, 
--     ["backpack"]=27, ["umbrella"]=28, ["handbag"]=31, ["tie"]=28, ["suitcase"]=29, 
--     ["frisbee"]=30, ["skis"]=31, ["snowboard"]=32, ["sports ball"]=33, ["kite"]=34, 
--     ["baseball bat"]=35, ["baseball glove"]=36, ["skateboard"]=37, ["surfboard"]=38, 
--     ["tennis racket"]=39, ["bottle"]=40, ["wine glass"]=41, ["cup"]=42, ["fork"]=43, 
--     ["knife"]=44, ["spoon"]=45, ["bowl"]=46, ["banana"]=47, ["apple"]=48, ["sandwich"]=49, 
--     ["orange"]=50, ["broccoli"]=51, ["carrot"]=52, ["hot dog"]=53, ["pizza"]=54, 
--     ["donut"]=55, ["cake"]=56, ["chair"]=57, ["couch"]=58, ["potted plant"]=59, ["bed"]=60, 
--     ["dining table"]=61, ["toilet"]=62, ["tv"]=63, ["laptop"]=64, ["mouse"]=65, 
--     ["remote"]=66, ["keyboard"]=67, ["cell phone"]=68, ["microwave"]=69, ["oven"]=70, 
--     ["toaster"]=71, ["sink"]=72, ["refrigerator"]=73, ["book"]=74, ["clock"]=75, 
--     ["vase"]=76, ["scissors"]=77, ["teddy bear"]=78, ["hair drier"]=79, ["toothbrush"]=80}

coco_label2name = {
    1: 'person', 2: 'bicycle', 
    3: 'car', 4: 'motorcycle',
    5: 'airplane', 6: 'bus', 
    7: 'train', 8: 'truck', 
    9: 'boat', 10: 'traffic light', 
    11: 'fire hydrant', 13: 'stop sign', 
    14: 'parking meter', 15: 'bench', 
    16: 'bird', 17: 'cat', 
    18: 'dog', 19: 'horse', 
    20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 
    24: 'zebra', 25: 'giraffe', 27: 'backpack', 
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 
    38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 
    44: 'bottle', 46: 'wine glass', 47: 'cup', 
    48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 
    52: 'banana', 53: 'apple', 54: 'sandwich', 
    55: 'orange', 56: 'broccoli', 57: 'carrot', 
    58: 'hot dog', 59: 'pizza', 60: 'donut', 
    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 
    65: 'bed', 67: 'dining table', 70: 'toilet', 72: 
    'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 
    76: 'keyboard', 77: 'cell phone', 78: 'microwave', 
    79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 
    88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
    
    
COCO_label_dict = {v:k for k, v in coco_label2name.items()}
opt = {
    start_from = 'model_idfull_category_spatial_rnn_encoder_validate_1024.t7',
    input_dir = 'gt_scene_jsons',
    output_path = 'gt_scene_gen_sents.json',
}

vocab = torch.load('vocab.t7')

print('initializing weights from ' .. opt.start_from)
local loaded_checkpoint = torch.load(opt.start_from)
protos = loaded_checkpoint.protos
--net_utils.unsanitize_gradients(protos.cnn)
net_utils.unsanitize_gradients(protos.words_encoder)
local lm_modules = protos.lm:getModulesList()
for k,v in pairs(lm_modules) do net_utils.unsanitize_gradients(v) end
local le_modules = protos.le:getModulesList()
for k,v in pairs(le_modules) do net_utils.unsanitize_gradients(v) end
protos.crit = nn.LanguageModelCriterion() -- not in checkpoints, create manually
protos.keyword_expander = nn.FeatExpander(opt.seq_per_img) -- not in checkpoints, create manually
protos.expander = nn.FeatExpander(opt.seq_per_img) -- not in checkpoints, create manually
opt.start_iter = loaded_checkpoint.iter-1

-- protos.cnn:evaluate()
protos.words_encoder:evaluate()
protos.lm:evaluate()
protos.le:evaluate()


-- Create empty table to store file paths:
files = {}
-- Go over all files in directory. We use an iterator, paths.files().
for file in paths.files(opt.input_dir) do
   -- We only load files that match the extension
   if file:find('json' .. '$') then
      -- and insert the ones we care about in our table
      table.insert(files, paths.concat(opt.input_dir, file))
   end
end

-- Check files
if #files == 0 then
    error('given directory doesnt contain any files of type json')
end

out_dict = {}
for i, file in ipairs(files) do
    fp = io.open(file, "r")
    io.input(fp)
    js = io.read("*all")
    io.close(fp)
    entry = cjson.decode(js)
    local bbox_coords = torch.FloatTensor(1,96,4):fill(0)
    local full_category = torch.LongTensor(1,96):fill(0)
    for k, v in pairs(entry['seq']) do
        bbox_coords[{1,k,1}] = v['left']
        bbox_coords[{1,k,2}] = v['top']
        bbox_coords[{1,k,3}] = v['width']
        bbox_coords[{1,k,4}] = v['height']
        full_category[{1,k}] = COCO_label_dict[v['label']]
    end
    bbox_coords:div(604.0)

    local bbox_coords_T = bbox_coords:transpose(1,2):cuda()
    local data_full_category_T = full_category:transpose(1,2):cuda()
    local le_category_encoding = protos.le:forward({data_full_category_T, bbox_coords_T})
    local sample_opts = { sample_max = 1, beam_size = 1, temperature = 1.0 }
    local seq = protos.lm:sample(le_category_encoding, sample_opts)
    local sents = net_utils.decode_sequence(vocab, seq)


    out_entry = {}
    out_entry['image_id'] = entry['image_id']
    out_entry['caption'] = sents[1]
    out_dict[i] = out_entry

    -- if i > 2 then
    --     break
    -- end
    print(i)
end

json_text = cjson.encode(out_dict)
-- print(json_text)

out_file = io.open(opt.output_path, "w")
io.output(out_file)
io.write(json_text)
io.close(out_file)


-- async.http.listen('http://0.0.0.0:9001/', function(req,res)
--    print(req.body)
--    annotations = cjson.decode(req.body)
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
--    bbox_coords:div(604)

--    local bbox_coords_T = bbox_coords:transpose(1,2):cuda()
--    local data_full_category_T = full_category:transpose(1,2):cuda()
--    local le_category_encoding = protos.le:forward({data_full_category_T, bbox_coords_T})
--    local sample_opts = { sample_max = 1, beam_size = 1, temperature = 1.0 }
--    local seq = protos.lm:sample(le_category_encoding, sample_opts)
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

-- print('server listening to port 9001')

-- async.go()

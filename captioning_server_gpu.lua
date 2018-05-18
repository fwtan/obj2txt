
require 'torch'
require 'nn'
require 'nngraph'
local utils = require 'misc.utils'
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

require('pl.text').format_operator()

COCO_label_dict = {
    ["person"]=1, ["bicycle"]=2, ["car"]=3, ["motorcycle"]=4, ["airplane"]=5, 
    ["bus"]=6, ["train"]=7, ["truck"]=8, ["boat"]=9, ["traffic light"]=10, 
    ["fire hydrant"]=11, ["stop sign"]=13, ["parking meter"]=14, ["bench"]=15, 
    ["bird"]=16, ["cat"]=17, ["dog"]=18, ["horse"]=19, ["sheep"]=20, 
    ["cow"]=21, ["elephant"]=22, ["bear"]=23, ["zebra"]=24, ["giraffe"]=25, 
    ["backpack"]=27, ["umbrella"]=28, ["handbag"]=31, ["tie"]=32, ["suitcase"]=33, 
    ["frisbee"]=34, ["skis"]=35, ["snowboard"]=36, ["sports ball"]=37, ["kite"]=38, 
    ["baseball bat"]=39, ["baseball glove"]=40, ["skateboard"]=41, ["surfboard"]=42, 
    ["tennis racket"]=43, ["bottle"]=44, ["wine glass"]=46, ["cup"]=47, ["fork"]=48, 
    ["knife"]=49, ["spoon"]=50, ["bowl"]=51, ["banana"]=52, ["apple"]=53, ["sandwich"]=54, 
    ["orange"]=55, ["broccoli"]=56, ["carrot"]=57, ["hot dog"]=58, ["pizza"]=59, 
    ["donut"]=60, ["cake"]=61, ["chair"]=62, ["couch"]=63, ["potted plant"]=64, ["bed"]=65, 
    ["dining table"]=67, ["toilet"]=70, ["tv"]=72, ["laptop"]=73, ["mouse"]=74, 
    ["remote"]=75, ["keyboard"]=76, ["cell phone"]=77, ["microwave"]=78, ["oven"]=79, 
    ["toaster"]=80, ["sink"]=81, ["refrigerator"]=82, ["book"]=84, ["clock"]=85, 
    ["vase"]=86, ["scissors"]=87, ["teddy bear"]=88, ["hair drier"]=89, ["toothbrush"]=90}
    
opt = {
    start_from = 'ckt_v1.t7',
    input_dir = 'gt_scene_jsons',
    output_path = 'gt_scene_gen_sents.json',
}


opt.seed = 123
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')


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
    local layout_feats = protos.layout_encoder:forward({ data_full_category_T, bbox_coords_T })
    local sample_opts = { sample_max = 1, beam_size = 2, temperature = 1.0 }
    local seq = protos.lm:sample(layout_feats, sample_opts)
    local sents = net_utils.decode_sequence(vocab, seq)
    -- print(sents[1])

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




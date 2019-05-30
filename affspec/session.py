import sys
import torch
import os

import pipeline



weight_name = "weights/model_esp_2.0_expression_action_units_valence_arousal_27.pt"
#weight_name = "weights/model_mob_2.0_expression_action_units_valence_arousal_19.pt"
#weight_name = "weights/model_res_2.0_expression_action_units_valence_arousal_18.pt"

if len(sys.argv) == 1:
    print( "PLEASE Provid weight file!" )
    #quit()
else:
    weight_name = sys.argv[1]

print("weight name:", weight_name)

session = pipeline.Process( model_loc = weight_name )

expressions = [ 'neutral', 'joy', 'sad',  'surprise', 'fear',  'disgust',  'anger', 'contempt']
AUs = [ "1", "2", "4", "5", "6", "9", "12", "17", "20", "25", "26", "43" ] # the actual AUs

outdir = os.path.basename( weight_name )[:-3] + "/"
if not os.path.exists( outdir ):
    os.mkdir( outdir )
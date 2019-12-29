expressions = ['neutral', 'joy', 'sad',  'surprise', 'fear',  'disgust',  
               'anger', 'contempt']

# action unit ids
AUs = ["1", "2", "4", "5", "6", "9", "12", "17", "20", "25", "26", "43"]




AU_id_and_names = ["AU1 Inner Brow Raiser", 
             "AU2 Outer Brow Raiser ", 
             "AU4 Brow Lowerer", 
             "AU5 Upper Lid Raiser",
             "AU6 Cheek Raiser", 
             "AU9 Nose Wrinkler", 
             "AU12 Lip Corner Puller", 
             "AU17 Chin Raiser	", 
             "AU20 Lip stretcher", 
             "AU25 Lips Part", 
             "AU26 Jaw Drop", 
             "AU43 Eyes Closed" ] # the actual AUs

AU_names = ["Inner Brow Raiser", 
             "Outer Brow Raiser ", 
             "Brow Lowerer", 
             "Upper Lid Raiser",
             "Cheek Raiser", 
             "Nose Wrinkler", 
             "Lip Corner Puller", 
             "Chin Raiser", 
             "Lip stretcher", 
             "Lips Part", 
             "Jaw Drop", 
             "Eyes Closed" ] # the actual AUs

def au_array_2_description(arr):
    """
    Convert the input Action Units array to a readable string
    
    Args:
        arr (list): a list of action units identified by the model
        
    Output:
        rst (str): a string that describes the activated actions
    """

    rst = ""
    action_count = 0
    for i in range(len(AU_names)):
        if arr[i] == 0: 
            continue # if the val is zero, skip
        rst += "%s, " % AU_names[i]
        action_count += 1
        
    return rst.strip()
import json
from easydict import EasyDict as edict

default_config = {
    "gpu"               : None,                         #gpu device used (set to None to take the first available device)
    "data_dir"          : "data",                       #data dir containing train.txt test.txt
    "batch_size"        : 30,                           #batch size
    #"dataTypeRatio"     : "1, 4, 1, 2, 1, 3, 3",       #'1' means using the initial set size without boostig, '0' means data type is unused
    "dataTypeRatio"     : "1, 4, 1, 0, 0, 0, 0",        #'1' means using the initial set size without boostig, '0' means data type is unused
    "weight_decay"      : 0.000005,                     #decay on weights
    "bias_decay"        : 0.0000005,                    #decay on biases
    "snapdir"           : "./snaps",                    #path to save snapshots
    "epochs"            : 12,                           #number of epoches
    "logdir"            : "logs11",                      #tensorboard log dir
    "summary_interval"  : 10000,                        #num of interval to dump summary
    "val_interval"      : 100000,                       #num of interval to dump summary
    "lr_values"         : "0.00005;0.000005;0.0000005", #lr step values
    "lr_boundaries"     : "10;11",                      #epoches to jump between lr values
    "momentum"          : 0.9,                          #momentum
    "continue_training" : False,                         #resume training from latest checkpoint
    #"continue_training" : False,                         #resume training from latest checkpoint
    "checkpoint_file"   : './snaps/base9_newArc_fixPad_fixWeights_include_existence_blackList_classWeights_snowClass_onlyMain_cont_cont_cont/final.ckpt', #resume from specific ckpt file
    #"checkpoint_file"   : None,                         #resume from specific ckpt file
    "num_workers"       : 5,                            #data feed pipeline workers
    "max_snapshots_keep": 10,                           #max snaps to keep 0 means all
    "output_dir"        : "infer_out"                   #detections images output path

}

config =  edict(default_config)

def printCfg():
    print "CFG: {"
    for (k,v) in config.__dict__.iteritems():
        print "\t\t\"{}\" : {},".format(k , v if type(v) != str else "\"{}\"".format(v))
    print "}"


